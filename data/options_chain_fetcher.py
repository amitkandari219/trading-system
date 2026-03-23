"""
Options Chain Fetcher — fetch live Nifty options chain via Kite Connect.

Fetches full options chain (all strikes for nearest expiry), stores raw data
in options_chain_daily, computes summary metrics (max pain, PCR, OI walls,
IV skew, straddle premium, OI concentration) in options_daily_summary.

Designed to run intraday (after 9:30 AM, before 3:00 PM) via cron.

Usage:
    venv/bin/python3 -m data.options_chain_fetcher            # fetch today
    venv/bin/python3 -m data.options_chain_fetcher --dry-run   # print without DB write
"""

import argparse
import logging
import math
import re
import time
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import psycopg2
from psycopg2.extras import execute_values

from config.settings import DATABASE_DSN, NIFTY_LOT_SIZE
from data.kite_auth import get_kite

logger = logging.getLogger(__name__)

# Nifty index token (for spot price)
NIFTY_INDEX_TOKEN = 256265

# Strike range: fetch strikes within ±STRIKE_RANGE_PCT of spot
STRIKE_RANGE_PCT = 0.10

# Kite API rate limit: max 10 requests/second
_KITE_RATE_SLEEP = 0.12  # seconds between API calls


# ================================================================
# FETCH INSTRUMENTS + QUOTES
# ================================================================

def _get_nifty_spot(kite) -> Optional[float]:
    """Fetch current Nifty spot price."""
    try:
        quote = kite.ltp(f"NSE:NIFTY 50")
        return quote["NSE:NIFTY 50"]["last_price"]
    except Exception as e:
        logger.warning(f"Failed to fetch Nifty spot via ltp: {e}")
    # Fallback: use ohlc
    try:
        quote = kite.ohlc(f"NSE:NIFTY 50")
        return quote["NSE:NIFTY 50"]["last_price"]
    except Exception as e:
        logger.error(f"Failed to fetch Nifty spot: {e}")
        return None


def _get_nifty_option_instruments(kite, spot: float) -> List[Dict]:
    """
    Fetch all Nifty option instruments from NFO exchange.
    Filter to:
      - NIFTY options only (not BANKNIFTY, stock options)
      - Strikes within ±STRIKE_RANGE_PCT of spot
      - Nearest weekly + nearest monthly expiry
    """
    instruments = kite.instruments("NFO")

    # Filter to NIFTY options
    nifty_opts = [
        i for i in instruments
        if i["name"] == "NIFTY"
        and i["instrument_type"] in ("CE", "PE")
        and i["strike"] > 0
    ]

    if not nifty_opts:
        logger.error("No NIFTY option instruments found")
        return []

    # Filter strikes within range
    lo_strike = spot * (1 - STRIKE_RANGE_PCT)
    hi_strike = spot * (1 + STRIKE_RANGE_PCT)
    nifty_opts = [i for i in nifty_opts if lo_strike <= i["strike"] <= hi_strike]

    # Find nearest expiry (weekly) and next expiry
    expiries = sorted(set(i["expiry"] for i in nifty_opts))
    today = date.today()
    future_expiries = [e for e in expiries if e >= today]

    if not future_expiries:
        logger.error("No future expiries found")
        return []

    # Take nearest 2 expiries (current week + next)
    selected_expiries = set(future_expiries[:2])
    nifty_opts = [i for i in nifty_opts if i["expiry"] in selected_expiries]

    logger.info(
        f"Found {len(nifty_opts)} NIFTY options | "
        f"Expiries: {sorted(selected_expiries)} | "
        f"Strike range: {lo_strike:.0f}-{hi_strike:.0f}"
    )
    return nifty_opts


def _fetch_quotes_batched(kite, instruments: List[Dict]) -> Dict:
    """
    Fetch quotes for instruments in batches (Kite allows ~500 per call).
    Returns dict: instrument_token -> quote_data.
    """
    BATCH_SIZE = 400
    all_quotes = {}

    # Build token list
    tokens = [i["instrument_token"] for i in instruments]

    for i in range(0, len(tokens), BATCH_SIZE):
        batch = tokens[i:i + BATCH_SIZE]
        # Kite quote() accepts list of "exchange:tradingsymbol" or instrument tokens
        # Using instrument tokens directly
        try:
            quotes = kite.quote([str(t) for t in batch])
            all_quotes.update(quotes)
        except Exception as e:
            logger.warning(f"Quote batch {i//BATCH_SIZE} failed: {e}")
        time.sleep(_KITE_RATE_SLEEP)

    return all_quotes


def fetch_options_chain(kite, spot: float, trade_date: date) -> List[Dict]:
    """
    Fetch full Nifty options chain from Kite.
    Returns list of row dicts ready for DB insert.
    """
    instruments = _get_nifty_option_instruments(kite, spot)
    if not instruments:
        return []

    # Build lookup: instrument_token -> instrument info
    inst_lookup = {str(i["instrument_token"]): i for i in instruments}

    # Fetch quotes
    quotes = _fetch_quotes_batched(kite, instruments)

    rows = []
    for token_str, quote in quotes.items():
        inst = inst_lookup.get(token_str)
        if inst is None:
            continue

        oi = quote.get("oi", 0) or 0
        oi_day_change = quote.get("oi_day_change", 0) or 0
        volume = quote.get("volume", 0) or 0
        ltp = quote.get("last_price", 0) or 0

        # Depth data for bid/ask
        depth = quote.get("depth", {})
        buy_depth = depth.get("buy", [{}])
        sell_depth = depth.get("sell", [{}])
        bid_price = buy_depth[0].get("price", 0) if buy_depth else 0
        ask_price = sell_depth[0].get("price", 0) if sell_depth else 0

        # IV: Kite doesn't provide IV directly; we'll compute from LTP later
        # For now store None — will be populated from options_loader's BS model
        # or from the OHLC-based IV in the summary computation
        rows.append({
            "date": trade_date,
            "expiry_date": inst["expiry"],
            "strike": float(inst["strike"]),
            "option_type": inst["instrument_type"],  # CE or PE
            "open_interest": oi,
            "change_in_oi": oi_day_change,
            "volume": volume,
            "iv": None,  # computed below
            "ltp": ltp,
            "bid_price": bid_price or None,
            "ask_price": ask_price or None,
        })

    # Compute IV for each option using Black-Scholes
    _compute_ivs(rows, spot)

    logger.info(f"Fetched {len(rows)} option quotes for {trade_date}")
    return rows


def _compute_ivs(rows: List[Dict], spot: float):
    """Compute implied volatility for each option row in-place."""
    from data.options_loader import implied_volatility
    from config.settings import RISK_FREE_RATE

    for row in rows:
        ltp = row.get("ltp", 0)
        if not ltp or ltp <= 0:
            continue
        strike = row["strike"]
        expiry = row["expiry_date"]
        today = row["date"]
        dte = (expiry - today).days
        if dte <= 0:
            continue
        T = dte / 365.0
        iv = implied_volatility(ltp, spot, strike, T, RISK_FREE_RATE, row["option_type"])
        if iv is not None:
            row["iv"] = round(iv, 4)


# ================================================================
# SUMMARY COMPUTATION
# ================================================================

def compute_max_pain(rows: List[Dict]) -> Optional[float]:
    """
    Max pain = strike where total loss for option writers is minimized.
    At each strike K, compute:
      writer_loss = sum over all calls with strike <= K: (K - call_strike) * call_OI
                  + sum over all puts with strike >= K: (put_strike - K) * put_OI
    Return strike with minimum writer_loss.
    """
    calls = [(r["strike"], r["open_interest"]) for r in rows
             if r["option_type"] == "CE" and r["open_interest"] and r["open_interest"] > 0]
    puts = [(r["strike"], r["open_interest"]) for r in rows
            if r["option_type"] == "PE" and r["open_interest"] and r["open_interest"] > 0]

    if not calls and not puts:
        return None

    all_strikes = sorted(set(r["strike"] for r in rows))
    if not all_strikes:
        return None

    min_pain = float("inf")
    max_pain_strike = all_strikes[0]

    for K in all_strikes:
        pain = 0.0
        # Call buyers exercise when spot > strike → writer loses (K - call_strike) * OI
        # At expiry spot = K: calls with strike < K are ITM
        for c_strike, c_oi in calls:
            if K > c_strike:
                pain += (K - c_strike) * c_oi
        # Put buyers exercise when spot < strike → writer loses (put_strike - K) * OI
        for p_strike, p_oi in puts:
            if K < p_strike:
                pain += (p_strike - K) * p_oi

        if pain < min_pain:
            min_pain = pain
            max_pain_strike = K

    return max_pain_strike


def compute_summary(rows: List[Dict], spot: float, trade_date: date) -> Dict:
    """
    Compute options_daily_summary from raw chain data.
    """
    if not rows:
        return {}

    # Split by type
    calls = [r for r in rows if r["option_type"] == "CE"]
    puts = [r for r in rows if r["option_type"] == "PE"]

    # Total OI
    total_call_oi = sum(r["open_interest"] for r in calls if r["open_interest"])
    total_put_oi = sum(r["open_interest"] for r in puts if r["open_interest"])

    # PCR
    pcr_oi = (total_put_oi / total_call_oi) if total_call_oi > 0 else None

    # Put wall (strike with max put OI)
    put_oi_max_strike = None
    if puts:
        max_put = max(puts, key=lambda r: r["open_interest"] or 0)
        if max_put["open_interest"] and max_put["open_interest"] > 0:
            put_oi_max_strike = max_put["strike"]

    # Call wall (strike with max call OI)
    call_oi_max_strike = None
    if calls:
        max_call = max(calls, key=lambda r: r["open_interest"] or 0)
        if max_call["open_interest"] and max_call["open_interest"] > 0:
            call_oi_max_strike = max_call["strike"]

    # Max pain
    max_pain_strike = compute_max_pain(rows)

    # ATM strike: nearest to spot
    all_strikes = sorted(set(r["strike"] for r in rows))
    atm_strike = min(all_strikes, key=lambda s: abs(s - spot)) if all_strikes else None

    # ATM IVs
    atm_call_iv = None
    atm_put_iv = None
    atm_call_ltp = None
    atm_put_ltp = None

    if atm_strike is not None:
        for r in calls:
            if r["strike"] == atm_strike and r.get("iv"):
                atm_call_iv = r["iv"]
                atm_call_ltp = r.get("ltp", 0)
                break
        for r in puts:
            if r["strike"] == atm_strike and r.get("iv"):
                atm_put_iv = r["iv"]
                atm_put_ltp = r.get("ltp", 0)
                break

    # IV skew: put IV - call IV (at ATM)
    iv_skew = None
    if atm_put_iv is not None and atm_call_iv is not None:
        iv_skew = round(atm_put_iv - atm_call_iv, 4)

    # ATM straddle premium
    atm_straddle_premium = None
    if atm_call_ltp and atm_put_ltp:
        atm_straddle_premium = round(atm_call_ltp + atm_put_ltp, 2)

    # OI concentration: ratio of top-3 strikes' OI to total OI
    oi_concentration_ratio = None
    oi_concentration_center = None
    total_oi = total_call_oi + total_put_oi
    if total_oi > 0:
        # Aggregate OI per strike (call + put)
        strike_oi = {}
        for r in rows:
            s = r["strike"]
            strike_oi[s] = strike_oi.get(s, 0) + (r["open_interest"] or 0)

        top3 = sorted(strike_oi.items(), key=lambda x: x[1], reverse=True)[:3]
        top3_oi = sum(oi for _, oi in top3)
        oi_concentration_ratio = round(top3_oi / total_oi, 4) if total_oi > 0 else None

        # OI-weighted center strike
        weighted_sum = sum(s * oi for s, oi in strike_oi.items())
        oi_concentration_center = round(weighted_sum / total_oi, 2) if total_oi > 0 else None

    return {
        "date": trade_date,
        "max_pain_strike": max_pain_strike,
        "pcr_oi": round(pcr_oi, 4) if pcr_oi is not None else None,
        "put_oi_max_strike": put_oi_max_strike,
        "call_oi_max_strike": call_oi_max_strike,
        "total_put_oi": total_put_oi,
        "total_call_oi": total_call_oi,
        "atm_put_iv": round(atm_put_iv, 4) if atm_put_iv is not None else None,
        "atm_call_iv": round(atm_call_iv, 4) if atm_call_iv is not None else None,
        "iv_skew": iv_skew,
        "atm_straddle_premium": atm_straddle_premium,
        "oi_concentration_ratio": oi_concentration_ratio,
        "oi_concentration_center": oi_concentration_center,
    }


# ================================================================
# DATABASE
# ================================================================

def save_chain(conn, rows: List[Dict]):
    """Insert raw options chain rows into options_chain_daily."""
    if not rows:
        return 0
    cur = conn.cursor()
    values = [
        (r["date"], r["expiry_date"], r["strike"], r["option_type"],
         r["open_interest"], r["change_in_oi"], r["volume"],
         r["iv"], r["ltp"], r["bid_price"], r["ask_price"])
        for r in rows
    ]
    execute_values(cur, """
        INSERT INTO options_chain_daily
            (date, expiry_date, strike, option_type,
             open_interest, change_in_oi, volume, iv, ltp, bid_price, ask_price)
        VALUES %s
        ON CONFLICT (date, expiry_date, strike, option_type) DO UPDATE SET
            open_interest = EXCLUDED.open_interest,
            change_in_oi = EXCLUDED.change_in_oi,
            volume = EXCLUDED.volume,
            iv = EXCLUDED.iv,
            ltp = EXCLUDED.ltp,
            bid_price = EXCLUDED.bid_price,
            ask_price = EXCLUDED.ask_price
    """, values, page_size=500)
    conn.commit()
    return len(values)


def save_summary(conn, summary: Dict):
    """Insert/update options_daily_summary."""
    if not summary or not summary.get("date"):
        return
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO options_daily_summary
            (date, max_pain_strike, pcr_oi, put_oi_max_strike, call_oi_max_strike,
             total_put_oi, total_call_oi, atm_put_iv, atm_call_iv, iv_skew,
             atm_straddle_premium, oi_concentration_ratio, oi_concentration_center)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (date) DO UPDATE SET
            max_pain_strike = EXCLUDED.max_pain_strike,
            pcr_oi = EXCLUDED.pcr_oi,
            put_oi_max_strike = EXCLUDED.put_oi_max_strike,
            call_oi_max_strike = EXCLUDED.call_oi_max_strike,
            total_put_oi = EXCLUDED.total_put_oi,
            total_call_oi = EXCLUDED.total_call_oi,
            atm_put_iv = EXCLUDED.atm_put_iv,
            atm_call_iv = EXCLUDED.atm_call_iv,
            iv_skew = EXCLUDED.iv_skew,
            atm_straddle_premium = EXCLUDED.atm_straddle_premium,
            oi_concentration_ratio = EXCLUDED.oi_concentration_ratio,
            oi_concentration_center = EXCLUDED.oi_concentration_center
    """, (
        summary["date"],
        summary.get("max_pain_strike"),
        summary.get("pcr_oi"),
        summary.get("put_oi_max_strike"),
        summary.get("call_oi_max_strike"),
        summary.get("total_put_oi"),
        summary.get("total_call_oi"),
        summary.get("atm_put_iv"),
        summary.get("atm_call_iv"),
        summary.get("iv_skew"),
        summary.get("atm_straddle_premium"),
        summary.get("oi_concentration_ratio"),
        summary.get("oi_concentration_center"),
    ))
    conn.commit()


# ================================================================
# MAIN PIPELINE
# ================================================================

def run(dry_run: bool = False):
    """
    Main entry point: fetch options chain, compute summary, store in DB.
    """
    trade_date = date.today()
    print(f"Options chain fetch for {trade_date}...")

    # Authenticate with Kite
    kite = get_kite()
    if not kite:
        print("ERROR: Not authenticated. Run: venv/bin/python3 -m data.kite_auth --login")
        return False

    # Get spot price
    spot = _get_nifty_spot(kite)
    if not spot:
        print("ERROR: Could not fetch Nifty spot price")
        return False
    print(f"  Nifty spot: {spot:.2f}")

    # Fetch options chain
    rows = fetch_options_chain(kite, spot, trade_date)
    if not rows:
        print("ERROR: No options data fetched")
        return False

    # Filter to nearest expiry for summary (use all for chain storage)
    nearest_expiry = min(r["expiry_date"] for r in rows)
    near_rows = [r for r in rows if r["expiry_date"] == nearest_expiry]

    # Compute summary
    summary = compute_summary(near_rows, spot, trade_date)

    # Print summary
    print(f"  Options fetched: {len(rows)} (nearest expiry: {nearest_expiry})")
    if summary:
        print(f"  Max pain: {summary.get('max_pain_strike')}")
        print(f"  PCR (OI): {summary.get('pcr_oi')}")
        print(f"  Put wall: {summary.get('put_oi_max_strike')}")
        print(f"  Call wall: {summary.get('call_oi_max_strike')}")
        print(f"  ATM IV (CE/PE): {summary.get('atm_call_iv')} / {summary.get('atm_put_iv')}")
        print(f"  IV skew: {summary.get('iv_skew')}")
        print(f"  ATM straddle: {summary.get('atm_straddle_premium')}")
        print(f"  OI concentration: {summary.get('oi_concentration_ratio')}")

    if dry_run:
        print("  DRY RUN — not writing to DB")
        return True

    # Store to DB
    conn = psycopg2.connect(DATABASE_DSN)
    try:
        n_chain = save_chain(conn, rows)
        print(f"  Saved {n_chain} rows to options_chain_daily")

        save_summary(conn, summary)
        print(f"  Saved summary to options_daily_summary")
    except Exception as e:
        print(f"  DB ERROR: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nifty Options Chain Fetcher (Kite Connect)")
    parser.add_argument("--dry-run", action="store_true", help="Print data without DB write")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    success = run(dry_run=args.dry_run)
    print("Done" if success else "FAILED")
