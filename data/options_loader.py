"""
NSE Nifty Options data loader.

Downloads F&O bhavcopy from NSE archives, parses Nifty options chain,
calculates IV and Greeks using Black-Scholes, stores in PostgreSQL.

Data source: NSE F&O bhavcopy (free daily download)
Calculates: IV, delta, gamma, theta, vega, moneyness, DTE

Usage:
    python -m data.options_loader --init --start 2021-01-01
    python -m data.options_loader --update
    python -m data.options_loader --status
    python -m data.options_loader --show-chain    # today's options chain
"""

import argparse
import io
import logging
import math
import time
import zipfile
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import requests
from scipy.stats import norm

from config.settings import DATABASE_DSN, RISK_FREE_RATE

logger = logging.getLogger(__name__)

NSE_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0',
    'Accept': '*/*',
}

NIFTY_LOT_SIZE = 25  # post-2024


# ================================================================
# BLACK-SCHOLES + GREEKS (pure scipy, no external libs)
# ================================================================

def bs_price(S, K, T, r, sigma, option_type='CE'):
    """Black-Scholes option price."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if option_type == 'CE':
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def implied_volatility(market_price, S, K, T, r, option_type='CE', tol=1e-5, max_iter=100):
    """Calculate IV using Newton-Raphson method."""
    if market_price <= 0 or T <= 0 or S <= 0 or K <= 0:
        return None

    # Intrinsic value check
    if option_type == 'CE':
        intrinsic = max(S - K, 0)
    else:
        intrinsic = max(K - S, 0)

    if market_price < intrinsic * 0.99:
        return None

    # Better initial guess based on price/spot ratio
    sigma = max(0.10, min(2.0, market_price / S * 10))

    for _ in range(max_iter):
        price = bs_price(S, K, T, r, sigma, option_type)
        v = bs_vega(S, K, T, r, sigma) * 100  # bs_vega returns per 1%, need raw

        if v < 1e-10:
            # Fallback: bisection
            sigma = _iv_bisection(market_price, S, K, T, r, option_type)
            break

        diff = price - market_price
        sigma -= diff / v

        if sigma <= 0.001:
            sigma = 0.001
        if sigma > 10.0:
            sigma = 5.0
        if abs(diff) < tol:
            break

    return sigma if 0.01 < sigma < 5.0 else None


def _iv_bisection(market_price, S, K, T, r, option_type, tol=1e-4, max_iter=100):
    """Bisection fallback for IV when Newton fails."""
    lo, hi = 0.01, 5.0
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        p = bs_price(S, K, T, r, mid, option_type)
        if abs(p - market_price) < tol:
            return mid
        if p < market_price:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def bs_delta(S, K, T, r, sigma, option_type='CE'):
    """Black-Scholes delta."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    if option_type == 'CE':
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1


def bs_gamma(S, K, T, r, sigma):
    """Black-Scholes gamma (same for calls and puts)."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return norm.pdf(d1) / (S * sigma * math.sqrt(T))


def bs_theta(S, K, T, r, sigma, option_type='CE'):
    """Black-Scholes theta (per day)."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    term1 = -(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T))
    if option_type == 'CE':
        term2 = -r * K * math.exp(-r * T) * norm.cdf(d2)
    else:
        term2 = r * K * math.exp(-r * T) * norm.cdf(-d2)

    return (term1 + term2) / 365  # per calendar day


def bs_vega(S, K, T, r, sigma):
    """Black-Scholes vega (per 1% IV change)."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return S * norm.pdf(d1) * math.sqrt(T) / 100  # per 1% move


def compute_greeks(row, spot, r=RISK_FREE_RATE):
    """Compute all Greeks for a single option row."""
    S = spot
    K = row['strike']
    T = row['days_to_expiry'] / 365.0
    price = row['close']
    opt_type = row['option_type']

    if T <= 0 or price <= 0:
        return {'implied_volatility': None, 'delta': None,
                'gamma': None, 'theta': None, 'vega': None}

    iv = implied_volatility(price, S, K, T, r, opt_type)
    if iv is None:
        return {'implied_volatility': None, 'delta': None,
                'gamma': None, 'theta': None, 'vega': None}

    return {
        'implied_volatility': round(iv, 4),
        'delta': round(bs_delta(S, K, T, r, iv, opt_type), 4),
        'gamma': round(bs_gamma(S, K, T, r, iv), 6),
        'theta': round(bs_theta(S, K, T, r, iv, opt_type), 2),
        'vega': round(bs_vega(S, K, T, r, iv), 2),
    }


# ================================================================
# NSE DATA DOWNLOAD
# ================================================================

class OptionsLoader:

    # Old format (pre-2024)
    OLD_URL = (
        "https://archives.nseindia.com/content/historical/DERIVATIVES/"
        "{year}/{month}/fo{date_str}bhav.csv.zip"
    )
    # New format (2024+)
    NEW_URL = (
        "https://archives.nseindia.com/content/fo/"
        "BhavCopy_NSE_FO_0_0_0_{date_yyyymmdd}_F_0000.csv.zip"
    )

    def __init__(self, db_conn=None):
        self.conn = db_conn
        self.session = requests.Session()
        self.session.headers.update(NSE_HEADERS)

    def download_bhavcopy(self, dt: date) -> Optional[pd.DataFrame]:
        """Download and parse NSE F&O bhavcopy for a date."""
        # Try new format first, then old
        date_yyyymmdd = dt.strftime('%Y%m%d')
        date_str = dt.strftime('%d%b%Y').upper()
        month_str = dt.strftime('%b').upper()

        urls = [
            self.NEW_URL.format(date_yyyymmdd=date_yyyymmdd),
            self.OLD_URL.format(year=dt.year, month=month_str, date_str=date_str),
        ]

        for url in urls:
            try:
                resp = self.session.get(url, timeout=30)
                if resp.status_code != 200:
                    continue

                try:
                    zf = zipfile.ZipFile(io.BytesIO(resp.content))
                    csv_names = [n for n in zf.namelist() if n.endswith('.csv')]
                    if not csv_names:
                        continue
                    df = pd.read_csv(zf.open(csv_names[0]))
                except zipfile.BadZipFile:
                    # Maybe it's not zipped
                    df = pd.read_csv(io.StringIO(resp.text))

                df.columns = [c.strip().upper() for c in df.columns]
                return df

            except Exception as e:
                logger.debug(f"URL failed for {dt}: {e}")
                continue

        return None

    def parse_nifty_options(self, df_raw: pd.DataFrame, trade_date: date,
                            spot_price: float = None) -> pd.DataFrame:
        """
        Parse bhavcopy DataFrame, extract Nifty options, compute Greeks.
        """
        df = df_raw.copy()

        # Find column names (NSE changes format periodically)
        sym_col = self._find_col(df, ['TCKRSYMB', 'SYMBOL', 'TckrSymb'])
        type_col = self._find_col(df, ['OPTNTP', 'OPTION_TYP', 'OPTIONTYPE', 'OptnTp'])
        strike_col = self._find_col(df, ['STRKPRIC', 'STRIKE_PR', 'STRIKE', 'StrkPric'])
        expiry_col = self._find_col(df, ['XPRYDT', 'EXPIRY_DT', 'EXPIRY', 'XpryDt'])
        close_col = self._find_col(df, ['CLSPRIC', 'CLOSE', 'ClsPric'])
        oi_col = self._find_col(df, ['OPNINTRST', 'OPEN_INT', 'OI', 'OpnIntrst'])
        vol_col = self._find_col(df, ['TTLTRADGVOL', 'CONTRACTS', 'VOLUME', 'TtlTradgVol'])
        open_col = self._find_col(df, ['OPNPRIC', 'OPEN', 'OpnPric'])
        high_col = self._find_col(df, ['HGHPRIC', 'HIGH', 'HghPric'])
        low_col = self._find_col(df, ['LWPRIC', 'LOW', 'LwPric'])
        settle_col = self._find_col(df, ['STTLMPRIC', 'SETTLE_PR', 'SttlPric'])
        underlying_col = self._find_col(df, ['UNDRLYGPRIC'])

        if not all([sym_col, type_col, strike_col, expiry_col, close_col]):
            logger.warning(f"Missing columns for {trade_date}")
            return pd.DataFrame()

        # Filter: NIFTY index options only (exclude BANKNIFTY, stock options)
        df[sym_col] = df[sym_col].astype(str).str.strip()
        df[type_col] = df[type_col].astype(str).str.strip().str.upper()

        # New format uses FININSTRMNM like "NIFTY26MAR23000CE"
        inst_col = self._find_col(df, ['FININSTRMNM'])
        if inst_col:
            # Filter by instrument name starting with NIFTY (not BANKNIFTY)
            nifty = df[
                (df[inst_col].astype(str).str.startswith('NIFTY')) &
                (~df[inst_col].astype(str).str.startswith('NIFTYIT')) &
                (~df[inst_col].astype(str).str.startswith('BANKNIFTY')) &
                (df[type_col].isin(['CE', 'PE']))
            ].copy()
        else:
            nifty = df[
                (df[sym_col].isin(['NIFTY', 'NIFTY 50'])) &
                (df[type_col].isin(['CE', 'PE']))
            ].copy()

        # Use underlying price from bhavcopy if available
        if underlying_col and not nifty.empty:
            spot_from_bhav = pd.to_numeric(nifty[underlying_col], errors='coerce').dropna()
            if len(spot_from_bhav) > 0 and spot_price is None:
                spot_price = float(spot_from_bhav.iloc[0])

        if nifty.empty:
            return pd.DataFrame()

        # Parse expiry
        nifty['expiry'] = pd.to_datetime(nifty[expiry_col], errors='coerce').dt.date
        nifty['strike'] = pd.to_numeric(nifty[strike_col], errors='coerce')
        nifty['close'] = pd.to_numeric(nifty[close_col], errors='coerce')
        nifty['option_type'] = nifty[type_col]

        if oi_col:
            nifty['oi'] = pd.to_numeric(nifty[oi_col], errors='coerce').fillna(0).astype(int)
        else:
            nifty['oi'] = 0

        if vol_col:
            nifty['volume'] = pd.to_numeric(nifty[vol_col], errors='coerce').fillna(0).astype(int)
        else:
            nifty['volume'] = 0

        for src, dst in [(open_col, 'open'), (high_col, 'high'),
                          (low_col, 'low'), (settle_col, 'settle_price')]:
            if src:
                nifty[dst] = pd.to_numeric(nifty[src], errors='coerce')
            else:
                nifty[dst] = None

        # Compute DTE and moneyness
        nifty['date'] = trade_date
        nifty['days_to_expiry'] = nifty['expiry'].apply(
            lambda x: (x - trade_date).days if x else 0
        )
        nifty = nifty[nifty['days_to_expiry'] >= 0]

        # Get spot price
        if spot_price is None:
            spot_price = self._get_spot(trade_date)
        if spot_price is None or spot_price <= 0:
            logger.warning(f"No spot price for {trade_date}")
            return pd.DataFrame()

        nifty['moneyness'] = nifty['strike'] / spot_price

        # Filter: strikes within ±10% of spot, valid prices
        nifty = nifty[
            (nifty['moneyness'] > 0.9) & (nifty['moneyness'] < 1.1) &
            (nifty['close'] > 0) & (nifty['strike'] > 0)
        ]

        # Compute Greeks for each option
        greeks_list = []
        for _, row in nifty.iterrows():
            g = compute_greeks(row, spot_price)
            greeks_list.append(g)

        greeks_df = pd.DataFrame(greeks_list)
        for col in greeks_df.columns:
            nifty[col] = greeks_df[col].values

        # Select output columns
        result = nifty[['date', 'expiry', 'strike', 'option_type',
                         'open', 'high', 'low', 'close', 'settle_price',
                         'volume', 'oi', 'implied_volatility',
                         'delta', 'gamma', 'theta', 'vega',
                         'days_to_expiry', 'moneyness']].copy()

        return result.dropna(subset=['strike', 'close'])

    def compute_daily_metrics(self, options_df: pd.DataFrame,
                               spot: float, trade_date: date) -> dict:
        """Compute aggregate options metrics for the day."""
        if options_df.empty:
            return {}

        # ATM IV (nearest strike to spot)
        near_expiry = options_df[options_df['days_to_expiry'] > 0]['expiry'].min()
        near = options_df[options_df['expiry'] == near_expiry]

        atm_strike = near.iloc[(near['strike'] - spot).abs().argsort()[:2]]['strike'].values
        atm = near[near['strike'].isin(atm_strike)]
        atm_iv = atm['implied_volatility'].dropna().mean()

        # Put-call skew: OTM put IV - OTM call IV
        otm_puts = near[(near['option_type'] == 'PE') & (near['moneyness'] < 0.97)]
        otm_calls = near[(near['option_type'] == 'CE') & (near['moneyness'] > 1.03)]
        put_iv = otm_puts['implied_volatility'].dropna().mean()
        call_iv = otm_calls['implied_volatility'].dropna().mean()
        skew = (put_iv - call_iv) if (put_iv and call_iv) else None

        # Term structure
        far_expiry = options_df[options_df['expiry'] > near_expiry]['expiry'].min() if near_expiry else None
        near_iv = near['implied_volatility'].dropna().mean()
        far_iv = None
        if far_expiry is not None:
            far = options_df[options_df['expiry'] == far_expiry]
            far_iv = far['implied_volatility'].dropna().mean()

        term_struct = (near_iv - far_iv) if (near_iv and far_iv) else None

        return {
            'date': trade_date,
            'spot_close': spot,
            'atm_iv': round(atm_iv, 4) if atm_iv else None,
            'put_call_skew': round(skew, 4) if skew else None,
            'near_month_iv': round(near_iv, 4) if near_iv else None,
            'far_month_iv': round(far_iv, 4) if far_iv else None,
            'term_structure': round(term_struct, 4) if term_struct else None,
        }

    def load_historical(self, start_date: date, end_date: date = None):
        """Download options data for date range."""
        end_date = end_date or date.today()
        current = start_date
        loaded = 0
        errors = 0

        print(f"Loading options data: {start_date} to {end_date}")

        while current <= end_date:
            if current.weekday() >= 5:
                current += timedelta(days=1)
                continue

            try:
                df_raw = self.download_bhavcopy(current)
                if df_raw is not None:
                    spot = self._get_spot(current)
                    if spot:
                        options = self.parse_nifty_options(df_raw, current, spot)
                        if not options.empty:
                            self._save_options(options)
                            metrics = self.compute_daily_metrics(options, spot, current)
                            if metrics:
                                self._save_metrics(metrics)
                            loaded += 1
                            if loaded % 50 == 0:
                                print(f"  Loaded {loaded} days (at {current})...", flush=True)
                else:
                    errors += 1
            except Exception as e:
                errors += 1
                if errors < 5:
                    logger.warning(f"Error on {current}: {e}")

            current += timedelta(days=1)
            time.sleep(1.0)

        print(f"  Done: {loaded} days loaded, {errors} skipped")

    def _get_spot(self, dt: date) -> Optional[float]:
        """Get Nifty spot close from nifty_daily table."""
        if self.conn is None:
            return None
        cur = self.conn.cursor()
        cur.execute("SELECT close FROM nifty_daily WHERE date = %s", (dt,))
        row = cur.fetchone()
        return float(row[0]) if row else None

    def _find_col(self, df, candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    def _save_options(self, df):
        if self.conn is None:
            return
        from psycopg2.extras import execute_values
        cur = self.conn.cursor()
        rows = [
            (r['date'], r['expiry'], r['strike'], r['option_type'],
             r.get('open'), r.get('high'), r.get('low'), r['close'],
             r.get('settle_price'), r.get('volume', 0), r.get('oi', 0),
             r.get('implied_volatility'), r.get('delta'), r.get('gamma'),
             r.get('theta'), r.get('vega'), r['days_to_expiry'], r['moneyness'])
            for _, r in df.iterrows()
        ]
        execute_values(cur, """
            INSERT INTO nifty_options (date, expiry, strike, option_type,
                open, high, low, close, settle_price, volume, oi,
                implied_volatility, delta, gamma, theta, vega,
                days_to_expiry, moneyness)
            VALUES %s ON CONFLICT (date, expiry, strike, option_type) DO NOTHING
        """, rows, page_size=500)
        self.conn.commit()

    def _save_metrics(self, metrics):
        if self.conn is None or not metrics:
            return
        cur = self.conn.cursor()
        cur.execute("""
            INSERT INTO options_daily_metrics (date, spot_close, atm_iv,
                put_call_skew, near_month_iv, far_month_iv, term_structure)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (date) DO UPDATE SET
                atm_iv = EXCLUDED.atm_iv,
                put_call_skew = EXCLUDED.put_call_skew
        """, (metrics['date'], metrics.get('spot_close'),
              metrics.get('atm_iv'), metrics.get('put_call_skew'),
              metrics.get('near_month_iv'), metrics.get('far_month_iv'),
              metrics.get('term_structure')))
        self.conn.commit()

    def show_chain(self, dt: date = None):
        """Show today's options chain with Greeks."""
        dt = dt or date.today()
        # Try last 5 trading days
        for delta in range(5):
            check_date = dt - timedelta(days=delta)
            df_raw = self.download_bhavcopy(check_date)
            if df_raw is not None:
                spot = self._get_spot(check_date)
                if spot is None:
                    # Estimate from recent data
                    if self.conn:
                        cur = self.conn.cursor()
                        cur.execute("SELECT close FROM nifty_daily ORDER BY date DESC LIMIT 1")
                        row = cur.fetchone()
                        spot = float(row[0]) if row else 24000
                    else:
                        spot = 24000

                options = self.parse_nifty_options(df_raw, check_date, spot)
                if not options.empty:
                    # Filter to near expiry, ±5% strikes
                    near_exp = options['expiry'].min()
                    chain = options[
                        (options['expiry'] == near_exp) &
                        (options['moneyness'] > 0.95) &
                        (options['moneyness'] < 1.05)
                    ].sort_values(['strike', 'option_type'])

                    print(f"\nNifty Options Chain — {check_date} | Spot: {spot:.0f} | Expiry: {near_exp}")
                    print(f"{'Strike':>8s} {'Type':>4s} {'Close':>8s} {'IV':>7s} {'Delta':>7s} "
                          f"{'Gamma':>8s} {'Theta':>7s} {'Vega':>6s} {'OI':>10s} {'DTE':>4s}")
                    print("-" * 80)

                    for _, r in chain.iterrows():
                        iv_str = f"{r['implied_volatility']*100:.1f}%" if pd.notna(r['implied_volatility']) else "  --"
                        delta_str = f"{r['delta']:.3f}" if pd.notna(r['delta']) else "  --"
                        gamma_str = f"{r['gamma']:.5f}" if pd.notna(r['gamma']) else "   --"
                        theta_str = f"{r['theta']:.1f}" if pd.notna(r['theta']) else "  --"
                        vega_str = f"{r['vega']:.1f}" if pd.notna(r['vega']) else " --"

                        print(f"{r['strike']:>8.0f} {r['option_type']:>4s} {r['close']:>8.1f} "
                              f"{iv_str:>7s} {delta_str:>7s} {gamma_str:>8s} {theta_str:>7s} "
                              f"{vega_str:>6s} {r['oi']:>10,.0f} {r['days_to_expiry']:>4d}")

                    metrics = self.compute_daily_metrics(options, spot, check_date)
                    print(f"\nATM IV: {metrics.get('atm_iv', 'N/A')}")
                    print(f"Put-Call Skew: {metrics.get('put_call_skew', 'N/A')}")
                    print(f"Term Structure: {metrics.get('term_structure', 'N/A')}")
                    return

        print("No bhavcopy data found for recent dates")


if __name__ == '__main__':
    import psycopg2

    parser = argparse.ArgumentParser(description='Nifty Options Loader')
    parser.add_argument('--init', action='store_true')
    parser.add_argument('--update', action='store_true')
    parser.add_argument('--status', action='store_true')
    parser.add_argument('--show-chain', action='store_true')
    parser.add_argument('--start', type=str, default='2021-01-01')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    conn = psycopg2.connect(DATABASE_DSN)
    loader = OptionsLoader(db_conn=conn)

    if args.init:
        start = date.fromisoformat(args.start)
        loader.load_historical(start)
    elif args.update:
        yesterday = date.today() - timedelta(days=1)
        loader.load_historical(yesterday)
    elif args.show_chain:
        loader.show_chain()
    elif args.status:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*), MIN(date), MAX(date) FROM nifty_options")
        count, min_d, max_d = cur.fetchone()
        print(f"nifty_options: {count:,} rows, {min_d} to {max_d}")
        cur.execute("SELECT COUNT(*) FROM options_daily_metrics")
        mc = cur.fetchone()[0]
        print(f"options_daily_metrics: {mc} rows")
    else:
        parser.print_help()

    conn.close()
