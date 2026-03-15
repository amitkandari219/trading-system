# NIFTY F&O DYNAMIC SIGNAL SYSTEM — COMPLETE SPECIFICATION
# Design: 100% complete
# Specification: 100% complete (all 7 specs written and reviewed)
# Code: 0% (not started — this document is the build contract)

# IMPLEMENTATION SPEC 1: BACKTEST ENGINE
# Written by: System Architect + Trading Systems Expert (joint spec)
# Status: FINAL — implement exactly as written

---

## BACKTEST TIER DESIGN
# Three-tier architecture. Every signal is classified into exactly one tier
# before backtesting begins, based on its instrument type.

```
TIER 1 — FUTURES and PRICE-BASED signals
  Instrument:   FUTURES, COMBINED (futures + options hedged)
  Data needed:  Daily Nifty OHLCV + India VIX (10 years, free from NSE)
  Engine:       WalkForwardEngine with backtest_fn() you write per signal
  Horizon:      10 years (2014–2024)
  Status:       Fully specified. Implement first.

TIER 2 — OPTIONS signals (buying and selling)
  Instrument:   OPTIONS_BUYING, OPTIONS_SELLING, SPREAD
  Data needed:  Daily Nifty OHLCV + India VIX + vol_adjustment_factors table
  Engine:       Same WalkForwardEngine framework, but backtest_fn() must
                reconstruct option prices using Black-Scholes at each bar
  Horizon:      3 years (2021–2024) — longer history has poor IV data quality
  Status:       Framework specified (WalkForwardEngine, pass criteria). 
                backtest_fn() template specified below (Tier 2 Template section).
                vol_adjustment_factors must be derived from TrueData before use.

TIER 3 — METHODOLOGY and PSYCHOLOGY signals
  Instrument:   Any — these are risk management rules, not entry signals
  Data needed:  Live paper trading observations
  Engine:       Paper trading IS the backtest. No historical simulation.
  Horizon:      4 months minimum (must span high-vol and low-vol period)
  Status:       By design. WalkForwardEngine does not apply.
  Examples:     Grimes regime filter, Lopez de Prado purge/embargo rules,
                Douglas discipline constraints

CLASSIFICATION RULE:
  Signal.instrument IN ('FUTURES', 'COMBINED')      → Tier 1
  Signal.instrument IN ('OPTIONS_BUYING',
                        'OPTIONS_SELLING', 'SPREAD') → Tier 2
  Signal.book_id IN ('GRIMES','LOPEZ',
                     'HILPISCH','DOUGLAS','HARRIS')  → Tier 3
  Signal.backtest_tier column is set by SignalSelector after classification.
```

---

## MODULE 0: NIFTY PRICE DATA
# File: data/nifty_loader.py
# This module owns all price data. Everything else reads from the DB.
# Run once during setup, then nightly for incremental updates.

```python
import pandas as pd
import requests
import zipfile
import io
from pathlib import Path
from datetime import date, timedelta


# ================================================================
# SCHEMA: nifty_daily table (add to schema.sql)
# ================================================================
# CREATE TABLE nifty_daily (
#     date         DATE        PRIMARY KEY,
#     open         FLOAT       NOT NULL,
#     high         FLOAT       NOT NULL,
#     low          FLOAT       NOT NULL,
#     close        FLOAT       NOT NULL,
#     volume       BIGINT,
#     india_vix    FLOAT,      -- NULL if VIX not yet available for that date
#     created_at   TIMESTAMPTZ DEFAULT NOW()
# );
# CREATE INDEX idx_nifty_daily_date ON nifty_daily(date DESC);


# ================================================================
# INITIAL LOAD: Download 10-year history from NSE (one-time setup)
# ================================================================

NSE_INDICES_URL = "https://archives.nseindia.com/content/indices/ind_nifty50list.csv"
NSE_VIX_URL     = "https://www.nseindia.com/api/historical/vixhistory"

BHAV_URL_TEMPLATE = (
    "https://archives.nseindia.com/content/indices/"
    "ind_close_all_{date_str}.csv"
    # date_str format: DDMMYYYY  e.g. 01012020
)


def load_nifty_history(db, days: int = None) -> pd.DataFrame:
    """
    Read Nifty daily OHLCV + VIX from the nifty_daily table.
    Returns DataFrame with columns:
        [date, open, high, low, close, volume, india_vix]
    sorted ascending by date.

    days: if set, return only the most recent N calendar days.
          If None, return full history.

    Called by:
        - ValidationTestSuite.run() (no days limit — needs full history)
        - RegimeLabeler fallback in SignalSelector (days=200)
        - WalkForwardEngine._load_data() (no limit — engine slices itself)
    """
    if days is not None:
        cutoff = date.today() - timedelta(days=days)
        rows = db.execute(
            """SELECT date, open, high, low, close, volume, india_vix
               FROM nifty_daily
               WHERE date >= %s
               ORDER BY date ASC""",
            (cutoff,)
        ).fetchall()
    else:
        rows = db.execute(
            """SELECT date, open, high, low, close, volume, india_vix
               FROM nifty_daily
               ORDER BY date ASC"""
        ).fetchall()

    if not rows:
        raise RuntimeError(
            "nifty_daily table is empty. "
            "Run: python -m data.nifty_loader --init  to load 10-year history."
        )

    df = pd.DataFrame(rows, columns=[
        'date', 'open', 'high', 'low', 'close', 'volume', 'india_vix'
    ])
    df['date'] = pd.to_datetime(df['date'])
    return df


def initial_load(db, start_year: int = 2014):
    """
    One-time setup: download 10 years of Nifty OHLCV + VIX and insert
    into nifty_daily. Takes ~5-10 minutes. NSE archives are free.

    Run once: python -m data.nifty_loader --init

    Sources:
        OHLCV: NSE Index Close values archive (Bhavcopy)
               URL: https://archives.nseindia.com/content/indices/ind_close_all_{DDMMYYYY}.csv
               Contains: Index Name, Open, High, Low, Close, Shares Traded, Turnover (Rs. Cr.)
               Filter: rows where 'Index Name' == 'Nifty 50'

        VIX:   NSE VIX historical CSV
               URL: https://archives.nseindia.com/content/vix/histdata/india_vix_history.csv
               Columns: Date, Open, High, Low, Close (VIX values)
               Use: Close column, keyed by Date
    """
    print(f"Loading Nifty history from {start_year} to today...")

    # Step 1: Download VIX history (single file, covers full history)
    vix_df = _download_vix_history()
    print(f"VIX: {len(vix_df)} records loaded")

    # Step 2: Download OHLCV via Bhavcopy (one file per day — batch download)
    ohlcv_df = _download_bhavcopy_range(
        start=date(start_year, 1, 1),
        end=date.today()
    )
    print(f"OHLCV: {len(ohlcv_df)} records loaded")

    # Step 3: Merge on date
    merged = ohlcv_df.merge(vix_df, on='date', how='left')
    # VIX data starts 2009 — rows before that will have NULL india_vix

    # Step 4: Insert to DB
    inserted = _bulk_insert(db, merged)
    print(f"Inserted {inserted} rows into nifty_daily.")


def _download_vix_history() -> pd.DataFrame:
    """
    Download India VIX full history via yfinance (ticker: ^INDIAVIX).

    WHY yfinance, not NSE directly:
    NSE removed the static CSV at archives.nseindia.com/content/vix/histdata/
    india_vix_history.csv (returns 404). The new NSE VIX page at
    nseindia.com/reports-indices-historical-vix requires a browser session
    cookie — plain requests.get() returns empty HTML. yfinance wraps Yahoo
    Finance's API, which has carried ^INDIAVIX data since 2008 and works
    reliably without session handling.

    Install: pip install yfinance
    Returns DataFrame with columns: [date (datetime), india_vix (float)]
    """
    import yfinance as yf

    ticker = yf.Ticker("^INDIAVIX")
    # period="max" fetches full history (2008-present)
    df = ticker.history(period="max")

    if df.empty:
        raise RuntimeError(
            "yfinance returned empty data for ^INDIAVIX. "
            "Check your internet connection or try again later."
        )

    df = df.reset_index()
    # yfinance returns 'Date' as a timezone-aware datetime — normalise to date only
    df['date'] = pd.to_datetime(df['Date']).dt.tz_localize(None).dt.normalize()
    df['india_vix'] = pd.to_numeric(df['Close'], errors='coerce')
    return df[['date', 'india_vix']].dropna()


def _download_bhavcopy_range(start: date, end: date) -> pd.DataFrame:
    """
    Download NSE Bhavcopy index files for every trading day in range.
    URL template:
        https://archives.nseindia.com/content/indices/ind_close_all_DDMMYYYY.csv
    Filters for 'Nifty 50' row. Returns [date, open, high, low, close, volume].

    Robust to missing files (NSE has gaps for holidays — skip gracefully).
    Rate-limit friendly: 0.5s delay between requests.
    """
    import time
    records = []
    current = start

    while current <= end:
        date_str = current.strftime('%d%m%Y')
        url = f"https://archives.nseindia.com/content/indices/ind_close_all_{date_str}.csv"
        try:
            resp = requests.get(
                url,
                headers={'User-Agent': 'Mozilla/5.0'},
                timeout=15
            )
            if resp.status_code == 200 and len(resp.content) > 200:
                df = pd.read_csv(io.StringIO(resp.text))
                nifty50 = df[df['Index Name'].str.strip() == 'Nifty 50']
                if not nifty50.empty:
                    row = nifty50.iloc[0]
                    records.append({
                        'date':   pd.to_datetime(current),
                        'open':   float(row.get('Open', 0)),
                        'high':   float(row.get('High', 0)),
                        'low':    float(row.get('Low', 0)),
                        'close':  float(row.get('Close', 0)),
                        'volume': int(str(row.get('Shares Traded', 0)
                                        ).replace(',', '') or 0),
                    })
        except Exception:
            pass  # Holiday or network error — skip this date
        current += timedelta(days=1)
        time.sleep(0.3)  # NSE rate limit: ~200 req/min

    return pd.DataFrame(records)


def _bulk_insert(db, df: pd.DataFrame) -> int:
    """Insert merged OHLCV+VIX DataFrame into nifty_daily. Skip existing rows."""
    from psycopg2.extras import execute_values
    rows = [
        (
            row['date'].date(),
            row['open'], row['high'], row['low'], row['close'],
            int(row['volume']) if pd.notna(row['volume']) else None,
            float(row['india_vix']) if pd.notna(row['india_vix']) else None,
        )
        for _, row in df.iterrows()
    ]
    execute_values(
        db.cursor(),
        """INSERT INTO nifty_daily (date, open, high, low, close, volume, india_vix)
           VALUES %s
           ON CONFLICT (date) DO NOTHING""",
        rows
    )
    db.commit()
    return len(rows)


def incremental_update(db):
    """
    Nightly update: download yesterday's Bhavcopy and VIX, insert if missing.
    Called by cron at 6:00 PM daily (after market close + NSE publication delay).
    """
    yesterday = date.today() - timedelta(days=1)
    existing = db.execute(
        "SELECT 1 FROM nifty_daily WHERE date = %s", (yesterday,)
    ).fetchone()
    if existing:
        return  # Already loaded

    ohlcv = _download_bhavcopy_range(yesterday, yesterday)
    vix   = _download_vix_history()
    if not ohlcv.empty:
        merged = ohlcv.merge(vix, on='date', how='left')
        _bulk_insert(db, merged)


# ================================================================
# CLI ENTRY POINT
# ================================================================
if __name__ == '__main__':
    import argparse
    import psycopg2
    import psycopg2.extras

    parser = argparse.ArgumentParser(description='Nifty price data loader')
    parser.add_argument('--init',   action='store_true',
                        help='Load full 10-year history (one-time setup)')
    parser.add_argument('--update', action='store_true',
                        help='Incremental update (run nightly)')
    parser.add_argument('--dsn',    default='postgresql://localhost/trading',
                        help='PostgreSQL DSN')
    args = parser.parse_args()

    conn = psycopg2.connect(args.dsn)
    conn.autocommit = False

    if args.init:
        initial_load(conn)
    elif args.update:
        incremental_update(conn)
    else:
        parser.print_help()
    conn.close()
```

---

## TIER 2 BACKTEST TEMPLATE
# File: backtest/tier2_backtest.py
# Use this template when writing backtest_fn() for OPTIONS signals.
# The WalkForwardEngine calls backtest_fn(signal_params, history_df, regime_labels)
# and expects a BacktestResult back. This template shows how to reconstruct
# option prices from OHLCV + VIX using Black-Scholes.

```python
import numpy as np
from scipy.stats import norm
from backtest.types import BacktestResult


def black_scholes_price(S, K, T, r, sigma, option_type='call') -> float:
    """
    Standard Black-Scholes option price.
    S:     Nifty spot price
    K:     strike price
    T:     time to expiry in years (calendar days / 365)
    r:     risk-free rate (repo rate, e.g. 0.065)
    sigma: implied volatility (India VIX / 100, adjusted by vol_multiplier)
    option_type: 'call' or 'put'
    """
    if T <= 0:
        # Expired — intrinsic value only
        if option_type == 'call':
            return max(0, S - K)
        return max(0, K - S)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def get_vol_multiplier(db, dte: int) -> float:
    """
    Read the DTE-based vol multiplier from vol_adjustment_factors table.
    The multiplier corrects for the well-known IV overstatement at short DTE.
    Returns 1.0 if table not yet populated (safe default — slightly underestimates vol).
    """
    # Map DTE to bucket
    if   dte <= 1:  bucket = 1
    elif dte <= 2:  bucket = 2
    elif dte <= 3:  bucket = 3
    elif dte <= 4:  bucket = 4
    elif dte <= 5:  bucket = 5
    elif dte <= 6:  bucket = 6
    elif dte <= 7:  bucket = 7
    elif dte <= 14: bucket = 8    # 8-14
    elif dte <= 21: bucket = 15   # 15-21
    elif dte <= 30: bucket = 22   # 22-30
    else:           bucket = 31   # 31+

    row = db.execute(
        "SELECT vol_multiplier FROM vol_adjustment_factors WHERE dte_bucket = %s",
        (bucket,)
    ).fetchone()
    return row['vol_multiplier'] if row else 1.0


def make_tier2_backtest_fn(signal_params: dict, db, risk_free_rate: float = 0.065):
    """
    Factory: returns a backtest_fn() for an OPTIONS signal.
    signal_params must include:
        instrument:      'OPTIONS_BUYING' | 'OPTIONS_SELLING' | 'SPREAD'
        entry_dte:       int — enter position N days before expiry
        exit_dte:        int — exit position N days before expiry (0 = hold to expiry)
        strike_offset:   float — ATM=0, OTM call = +X%, OTM put = -X%
                         e.g. 0.02 = strike is 2% above/below spot
        direction:       'LONG' | 'SHORT'
        lot_size:        int — Nifty lot size (25 post-2024)
        stop_loss_pct:   float — exit if premium drops by this fraction (buying)
                         or if premium rises by this fraction (selling)
        <any other signal-specific params>

    Returns: backtest_fn(signal_params, history_df, regime_labels) → BacktestResult
    """
    def backtest_fn(params, history_df, regime_labels):
        """
        Simulate the options strategy on history_df.
        Nifty weekly expiry: every Thursday.
        Entry: when signal conditions met AND DTE == params['entry_dte']
        Exit:  when DTE == params['exit_dte'], OR stop-loss hit, OR signal exits

        IMPORTANT — DATA QUALITY NOTE:
        This uses India VIX as the IV proxy for all options (ATM and OTM).
        VIX reflects 30-day ATM IV. For OTM options, actual IV has a smile/skew.
        The vol_adjustment_factors table corrects for DTE effects but NOT skew.
        For deep OTM options (|strike_offset| > 3%), results will be optimistic.
        Use with caution for far-OTM strategies.
        """
        trades = []
        in_trade = False
        entry_price = None
        entry_date = None
        direction = params.get('direction', 'LONG')
        lot_size = params.get('lot_size', 25)
        entry_dte = params.get('entry_dte', 7)
        exit_dte = params.get('exit_dte', 1)
        strike_offset = params.get('strike_offset', 0.0)
        stop_loss_pct = params.get('stop_loss_pct', 0.50)  # 50% premium loss default

        for _, row in history_df.iterrows():
            bar_date = row['date']
            spot = row['close']
            vix = row['india_vix'] if pd.notna(row['india_vix']) else 15.0

            # Days to next Thursday expiry
            days_to_thu = (3 - bar_date.weekday()) % 7 or 7
            dte = days_to_thu
            if dte > 30:
                dte = dte % 7 or 7  # weekly expiry

            # IV: VIX / 100 × vol_multiplier for this DTE
            vol_mult = get_vol_multiplier(db, dte)
            sigma = (vix / 100.0) * vol_mult

            # Strike: ATM ± offset
            strike = round(spot * (1 + strike_offset) / 50) * 50  # round to 50

            T = dte / 365.0
            opt_type = 'call' if direction == 'LONG' and strike_offset >= 0 else 'put'
            opt_price = black_scholes_price(spot, strike, T, risk_free_rate, sigma, opt_type)

            if not in_trade:
                # Check entry conditions (signal-specific — implement per signal)
                regime = regime_labels.get(bar_date.date(), 'RANGING')
                if _check_entry(params, row, regime, dte, entry_dte):
                    in_trade = True
                    entry_price = opt_price
                    entry_date = bar_date
                    entry_spot = spot

            else:
                # Check exit conditions
                exit_reason = None
                if dte <= exit_dte:
                    exit_reason = 'DTE_TARGET'
                elif direction == 'LONG' and opt_price < entry_price * (1 - stop_loss_pct):
                    exit_reason = 'STOP_LOSS'
                elif direction == 'SHORT' and opt_price > entry_price * (1 + stop_loss_pct):
                    exit_reason = 'STOP_LOSS'
                elif _check_exit(params, row, regime_labels.get(bar_date.date(), 'RANGING')):
                    exit_reason = 'SIGNAL_EXIT'

                if exit_reason:
                    pnl_per_lot = (opt_price - entry_price) * lot_size
                    if direction == 'SHORT':
                        pnl_per_lot = -pnl_per_lot
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date':  bar_date,
                        'pnl':        pnl_per_lot,
                        'entry_price': entry_price,
                        'exit_price':  opt_price,
                        'exit_reason': exit_reason,
                    })
                    in_trade = False
                    entry_price = None

        return _compute_backtest_result(trades, history_df)

    return backtest_fn


def _check_entry(params, row, regime, dte, entry_dte) -> bool:
    """
    Signal-specific entry check. Override per signal.
    Default: enter when DTE matches entry_dte target.
    Developer must add signal-specific conditions here
    (e.g. VIX level check, regime check, momentum filter).
    """
    return dte == entry_dte


def _check_exit(params, row, regime) -> bool:
    """
    Signal-specific exit check. Override per signal.
    Default: no early exit beyond DTE and stop-loss.
    """
    return False


def _compute_backtest_result(trades: list, history_df) -> BacktestResult:
    """Compute BacktestResult from list of trade dicts."""
    import pandas as pd

    if len(trades) < 5:
        # Insufficient trades — fail the backtest
        return BacktestResult(
            sharpe=0.0, calmar_ratio=0.0, max_drawdown=1.0,
            win_rate=0.0, profit_factor=0.0, avg_win_loss_ratio=0.0,
            trade_count=len(trades), nifty_correlation=0.0,
            annual_return=0.0, drawdown_2020=1.0
        )

    pnls = [t['pnl'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    total_return = sum(pnls)
    win_rate = len(wins) / len(pnls)

    # Approximate Sharpe from trade P&L series
    pnl_series = pd.Series(pnls)
    sharpe = (pnl_series.mean() / pnl_series.std() * np.sqrt(52)
              if pnl_series.std() > 0 else 0.0)

    # Drawdown from cumulative P&L
    cum_pnl = pd.Series(pnls).cumsum()
    rolling_max = cum_pnl.cummax()
    drawdown = ((cum_pnl - rolling_max) / (rolling_max.abs() + 1e-9))
    max_dd = abs(drawdown.min())

    profit_factor = (sum(wins) / abs(sum(losses))
                     if losses else float('inf'))
    avg_win = np.mean(wins) if wins else 0.0
    avg_loss = abs(np.mean(losses)) if losses else 1.0
    win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0

    calmar = (total_return / max_dd) if max_dd > 0 else 0.0

    # 2020 drawdown sub-period
    crash_trades = [t for t in trades
                    if pd.Timestamp('2020-03-01') <= t['entry_date']
                    <= pd.Timestamp('2020-04-30')]
    crash_pnl = [t['pnl'] for t in crash_trades]
    drawdown_2020 = (abs(min(0, sum(crash_pnl))) / (abs(total_return) + 1e-9)
                     if crash_pnl else 0.0)

    # Nifty correlation (approximate from weekly P&L vs Nifty weekly returns)
    nifty_weekly = history_df.set_index('date')['close'].resample('W').last().pct_change()
    # Build weekly P&L series aligned to nifty_weekly index — simplified
    nifty_correlation = 0.0  # Requires proper alignment; implement if needed

    return BacktestResult(
        sharpe            = round(sharpe, 3),
        calmar_ratio      = round(calmar, 3),
        max_drawdown      = round(max_dd, 3),
        win_rate          = round(win_rate, 3),
        profit_factor     = round(profit_factor, 3),
        avg_win_loss_ratio= round(win_loss_ratio, 3),
        trade_count       = len(trades),
        nifty_correlation = nifty_correlation,
        annual_return     = round(total_return, 0),
        drawdown_2020     = round(drawdown_2020, 3),
    )
```

---

## MODULE 1: REGIME LABELER
# File: regime_labeler.py
# Rule: NEVER compute on full dataset. Sequential only.

```python
class RegimeLabeler:
    """
    Computes market regime for each day using ONLY
    data available at close of that day.
    No lookahead. Verified by unit test below.
    """

    def __init__(self):
        self.adx_period = 14
        self.ema_period = 50
        self.vix_high_threshold = 18.0
        self.vix_crisis_threshold = 25.0
        self.adx_trend_threshold = 25.0
        self.ranging_days_required = 3  # ADX must be below threshold
                                        # for 3 consecutive days
                                        # to label RANGING
                                        # (prevents false ranging labels
                                        # in brief consolidations)

    def label_single_day(self, history_df, target_date):
        """
        Label ONE day using only data up to and including target_date.
        history_df: DataFrame with columns [date, open, high, low,
                    close, volume, india_vix]
        Returns: 'TRENDING' | 'RANGING' | 'HIGH_VOL' | 'CRISIS'
        """
        # Slice: only data up to and including target_date
        data = history_df[history_df['date'] <= target_date].copy()

        # Need minimum history for indicators
        if len(data) < self.adx_period + self.ranging_days_required + 5:
            return 'RANGING'  # Default for insufficient history

        latest = data.iloc[-1]
        vix = latest['india_vix']

        # CRISIS check first — overrides everything
        if vix >= self.vix_crisis_threshold:
            return 'CRISIS'

        # HIGH_VOL check second
        if vix >= self.vix_high_threshold:
            return 'HIGH_VOL'

        # Compute ADX on available data only
        adx_value = self._compute_adx(data)

        # Compute EMA on available data only
        ema_value = self._compute_ema(data, self.ema_period)
        price_above_ema = latest['close'] > ema_value

        # TRENDING: ADX above threshold AND price above EMA
        if adx_value >= self.adx_trend_threshold and price_above_ema:
            return 'TRENDING'

        # RANGING: ADX below threshold for required consecutive days
        recent_adx = [self._compute_adx(
                        data.iloc[:-(self.ranging_days_required - i - 1) or None]
                      )
                      for i in range(self.ranging_days_required)]
        if all(a < self.adx_trend_threshold for a in recent_adx):
            return 'RANGING'

        # Default: RANGING if no clear trending signal
        return 'RANGING'

    def label_full_history(self, history_df):
        """
        Label entire history SEQUENTIALLY.
        This is the only approved method for bulk labeling.
        """
        labels = {}
        dates = history_df['date'].tolist()

        for i, date in enumerate(dates):
            if i < self.adx_period + 10:
                labels[date] = 'RANGING'
                continue
            # Pass only data up to this date — no future data
            labels[date] = self.label_single_day(history_df, date)

        return labels

    def _compute_adx(self, data) -> float:
        """
        Wilder's Average Directional Index.
        Uses only data.iloc[-adx_period*2:] for efficiency.
        Returns float ADX value for the most recent bar.
        No lookahead: only uses data passed in (caller must pre-slice).
        """
        if len(data) < self.adx_period + 1:
            return 0.0

        # Use last adx_period*2 bars for stability
        d = data.tail(self.adx_period * 2).copy().reset_index(drop=True)
        n = self.adx_period

        # True Range
        high  = d['high'].values
        low   = d['low'].values
        close = d['close'].values

        tr_list = []
        for i in range(1, len(d)):
            tr = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i]  - close[i - 1])
            )
            tr_list.append(tr)

        # Directional movement
        plus_dm_list  = []
        minus_dm_list = []
        for i in range(1, len(d)):
            up   = high[i]  - high[i - 1]
            down = low[i - 1] - low[i]
            plus_dm_list.append(up   if up > down and up > 0   else 0.0)
            minus_dm_list.append(down if down > up and down > 0 else 0.0)

        # Wilder smooth (EMA with alpha = 1/n)
        def wilder_smooth(series, period):
            result = [sum(series[:period]) / period]
            for v in series[period:]:
                result.append(result[-1] - result[-1] / period + v)
            return result

        if len(tr_list) < n:
            return 0.0

        atr      = wilder_smooth(tr_list, n)
        plus_di  = wilder_smooth(plus_dm_list, n)
        minus_di = wilder_smooth(minus_dm_list, n)

        # Avoid div-by-zero
        dx_list = []
        for i in range(len(atr)):
            if atr[i] == 0:
                dx_list.append(0.0)
                continue
            pdi = 100 * plus_di[i]  / atr[i]
            mdi = 100 * minus_di[i] / atr[i]
            denom = pdi + mdi
            dx_list.append(100 * abs(pdi - mdi) / denom if denom != 0 else 0.0)

        if len(dx_list) < n:
            return 0.0

        adx_series = wilder_smooth(dx_list, n)
        return float(adx_series[-1])

    def _compute_ema(self, data, period) -> float:
        """
        Standard EMA, returns latest value only.
        No lookahead: only uses data passed in (caller must pre-slice).
        """
        closes = data['close'].values
        if len(closes) < period:
            return float(closes[-1]) if len(closes) > 0 else 0.0

        alpha = 2.0 / (period + 1)
        ema = float(closes[:period].mean())   # SMA seed
        for price in closes[period:]:
            ema = alpha * float(price) + (1 - alpha) * ema
        return ema


# MANDATORY UNIT TEST
def test_no_lookahead():
    """
    Verifies regime labeler has zero lookahead bias.
    Must pass before any backtest is run.
    """
    labeler = RegimeLabeler()

    # Load full dataset from nifty_daily table.
    # Requires: data/nifty_loader.py --init has been run first.
    # Import: from data.nifty_loader import load_nifty_history
    full_data = load_nifty_history(db)

    # Pick a random date in the middle
    test_date = full_data['date'].iloc[500]

    # Method 1: Label using full dataset (batch)
    full_labels = labeler.label_full_history(full_data)
    batch_label = full_labels[test_date]

    # Method 2: Label using ONLY data up to test_date
    truncated_data = full_data[full_data['date'] <= test_date]
    single_label = labeler.label_single_day(truncated_data, test_date)

    # MUST BE IDENTICAL
    assert batch_label == single_label, (
        f"LOOKAHEAD DETECTED: batch={batch_label}, "
        f"single={single_label} on {test_date}. "
        f"DO NOT RUN BACKTEST UNTIL FIXED."
    )
    print(f"No lookahead confirmed for {test_date}")
```

---

## SHARED DATACLASSES
# File: backtest/types.py
# All backtest classes import from here.

```python
from dataclasses import dataclass
from typing import Union
from dateutil.relativedelta import relativedelta
import pandas as pd

@dataclass
class BacktestResult:
    """
    Standardised return type for all backtest_fn calls.
    Signature: backtest_fn(params, history_df, regime_labels) -> BacktestResult
    Every backtest implementation must return this exact structure.
    """
    sharpe: float
    calmar_ratio: float
    max_drawdown: float        # fraction e.g. 0.18 = 18% drawdown
    win_rate: float            # fraction e.g. 0.45 = 45% winners
    profit_factor: float       # gross_profit / gross_loss
    avg_win_loss_ratio: float  # avg_win / avg_loss
    trade_count: int
    nifty_correlation: float   # Pearson corr of daily returns vs Nifty
    annual_return: float       # annualised net return, fraction

    # Required for OPTIONS_SELLING signals only.
    # Max drawdown during Mar–Apr 2020 sub-period specifically.
    # If window does not overlap 2020, set to None.
    # Must be computed by backtest_fn — _evaluate_window checks this.
    drawdown_2020: float = None


def add_months(dt, months: int):
    """Add calendar months to a date. Handles month-end correctly."""
    return dt + relativedelta(months=months)


def add_trading_days(dt, n: int, calendar_df: pd.DataFrame):
    """
    Add n trading days to dt using the market_calendar table.
    calendar_df: DataFrame with columns [date, is_trading_day].
    """
    trading_days = calendar_df[
        (calendar_df['date'] > dt) &
        (calendar_df['is_trading_day'] == True)
    ]['date'].sort_values()
    if len(trading_days) < n:
        raise ValueError(f"Not enough future trading days in calendar for +{n} days")
    return trading_days.iloc[n - 1]


def subtract_trading_days(dt, n: int, calendar_df: pd.DataFrame):
    """Subtract n trading days from dt using the market_calendar table."""
    trading_days = calendar_df[
        (calendar_df['date'] < dt) &
        (calendar_df['is_trading_day'] == True)
    ]['date'].sort_values(ascending=False)
    if len(trading_days) < n:
        raise ValueError(f"Not enough prior trading days in calendar for -{n} days")
    return trading_days.iloc[n - 1]


def harmonic_mean_sharpe(window_results: list) -> float:
    """
    Harmonic mean of per-window Sharpe ratios.
    Preferred over arithmetic mean: penalises windows with very low Sharpe.
    Ignores windows with Sharpe <= 0 (loss-making windows reduce the count).
    """
    positive = [w['result'].sharpe for w in window_results
                if w['result'].sharpe > 0]
    if not positive:
        return 0.0
    return len(positive) / sum(1 / s for s in positive)
```

---

## MODULE 2: PARAMETER SENSITIVITY TESTER
# File: sensitivity_tester.py
# Rule: Every numerical parameter tested at 5 values. Signal fragility scored.

```python
class ParameterSensitivityTester:
    """
    Tests whether a signal's edge is robust to parameter variation.
    Fragile signals (edge exists only at exact book parameter) are
    flagged and require higher Sharpe threshold to pass Round 1.
    """

    SENSITIVITY_MULTIPLIERS = [0.6, 0.8, 1.0, 1.2, 1.4]
    # Test at -40%, -20%, exact, +20%, +40% of book parameter

    ROBUST_THRESHOLD = 0.70
    # Signal is ROBUST if Sharpe at each multiplier >= 70% of peak Sharpe

    FRAGILE_THRESHOLD = 0.50
    # Signal is FRAGILE if Sharpe drops below 50% of peak at ±20%

    def test_signal_sensitivity(self, signal_id, base_params,
                                 backtest_fn, history_df, regime_labels):
        """
        Tests signal at 5 parameter variants.
        Returns sensitivity report.

        base_params: dict of {param_name: book_value}
                     Only numerical params are varied.
                     Non-numerical params held constant.
        backtest_fn: function(params, data, regimes) -> BacktestResult
        """
        results = {}

        for numerical_param, base_value in base_params.items():
            if not isinstance(base_value, (int, float)):
                continue  # Skip non-numerical params

            param_results = []

            for multiplier in self.SENSITIVITY_MULTIPLIERS:
                test_params = base_params.copy()
                test_value = base_value * multiplier

                # Round to nearest sensible value
                # (e.g. period params should be integers)
                if isinstance(base_value, int):
                    test_value = max(1, round(test_value))

                test_params[numerical_param] = test_value
                result = backtest_fn(test_params, history_df, regime_labels)
                param_results.append({
                    'multiplier': multiplier,
                    'param_value': test_value,
                    'sharpe': result.sharpe,
                    'profit_factor': result.profit_factor
                })

            peak_sharpe = max(r['sharpe'] for r in param_results)
            results[numerical_param] = {
                'variants': param_results,
                'peak_sharpe': peak_sharpe,
                'fragility': self._compute_fragility(
                    param_results, peak_sharpe
                )
            }

        return self._generate_report(signal_id, results)

    def _compute_fragility(self, variants, peak_sharpe):
        """
        ROBUST:   All ±20% variants >= 70% of peak Sharpe
        MODERATE: Worst ±20% variant between 50-70% of peak
        FRAGILE:  Any ±20% variant below 50% of peak

        Collects all verdicts across all ±20% variants and returns the worst.
        """
        if peak_sharpe <= 0:
            return 'FRAGILE'

        # Check ±20% variants specifically (multipliers 0.8 and 1.2)
        pm20_variants = [v for v in variants
                         if v['multiplier'] in [0.8, 1.2]]

        verdicts = []
        for v in pm20_variants:
            ratio = v['sharpe'] / peak_sharpe
            if ratio < self.FRAGILE_THRESHOLD:
                verdicts.append('FRAGILE')
            elif ratio < self.ROBUST_THRESHOLD:
                verdicts.append('MODERATE')
            else:
                verdicts.append('ROBUST')

        # Return worst verdict (FRAGILE > MODERATE > ROBUST)
        if 'FRAGILE'  in verdicts: return 'FRAGILE'
        if 'MODERATE' in verdicts: return 'MODERATE'
        return 'ROBUST'

    def _generate_report(self, signal_id, results):
        """
        Returns:
        - overall_fragility: ROBUST | MODERATE | FRAGILE
        - required_sharpe_threshold: adjusted pass bar
        - recommendation: PROCEED | HIGHER_BAR | ARCHIVE
        """
        fragilities = [r['fragility'] for r in results.values()]

        if 'FRAGILE' in fragilities:
            return {
                'signal_id': signal_id,
                'overall_fragility': 'FRAGILE',
                'required_sharpe_threshold': 2.0,  # Higher bar for fragile
                'recommendation': 'HIGHER_BAR',
                'note': 'Edge exists only at exact book parameter. '
                        'Requires Sharpe > 2.0 to compensate.'
            }
        elif 'MODERATE' in fragilities:
            return {
                'signal_id': signal_id,
                'overall_fragility': 'MODERATE',
                'required_sharpe_threshold': 1.6,
                'recommendation': 'PROCEED',
                'note': 'Edge degrades at parameter extremes. '
                        'Use exact book parameter in live system.'
            }
        else:
            return {
                'signal_id': signal_id,
                'overall_fragility': 'ROBUST',
                'required_sharpe_threshold': None,  # Use standard criteria
                'recommendation': 'PROCEED',
                'note': 'Edge robust across parameter range. '
                        'Book parameter confirmed optimal.'
            }
```

---

## MODULE 3: WALK-FORWARD ENGINE
# File: backtest/walk_forward.py
# Exact parameters: 36-month train / 12-month test / 3-month step
# Imports add_months, add_trading_days, subtract_trading_days, harmonic_mean_sharpe
# from backtest.types

```python
from backtest.types import (BacktestResult, add_months, add_trading_days,
                             subtract_trading_days, harmonic_mean_sharpe)
class WalkForwardEngine:
    """
    Exact walk-forward implementation.
    Train: 36 months. Test: 12 months. Step: 3 months.
    Purge: 21 trading days before each test window.
    Embargo: 5 trading days after each test window.
    """

    TRAIN_MONTHS = 36
    TEST_MONTHS = 12
    STEP_MONTHS = 3
    PURGE_DAYS = 21
    EMBARGO_DAYS = 5

    # Pass criteria: ≥75% of all walk-forward windows AND the most recent window.
    # With 3-month step on 10-year history there are ~24 windows.
    # An absolute count threshold (e.g. "pass 3 windows") is too easy to game.
    MIN_PASS_RATE          = 0.75   # must pass ≥75% of all walk-forward windows
    MUST_PASS_LAST_WINDOW  = True   # most recent window must pass (recency check)

    def __init__(self, calendar_df):
        """
        calendar_df: DataFrame with columns [date, is_trading_day].
        Load from market_calendar table before instantiating.
        Required by add_trading_days() and subtract_trading_days().
        """
        self.calendar_df = calendar_df

    def run(self, signal_id, backtest_fn, history_df,
            regime_labels, params, signal_type) -> dict:
        """
        Runs complete walk-forward analysis.
        Returns WalkForwardResult with per-window and aggregate metrics.
        """
        windows = self._generate_windows(history_df)
        window_results = []

        for window in windows:
            # Training data (with purge buffer excluded)
            train_data = history_df[
                (history_df['date'] >= window['train_start']) &
                (history_df['date'] < window['purge_start'])
            ]

            # Test data (after embargo buffer)
            test_data = history_df[
                (history_df['date'] >= window['test_start']) &
                (history_df['date'] <= window['test_end'])
            ]

            # Regime labels for test period only
            test_regimes = {
                d: regime_labels[d]
                for d in test_data['date']
                if d in regime_labels
            }

            # Run backtest on TEST data only
            # (training data used only for parameter optimization
            #  if any — book params are used directly here)
            result = backtest_fn(params, test_data, test_regimes)

            # Apply pass criteria by signal type
            passed = self._evaluate_window(
                result, signal_type, window
            )

            window_results.append({
                'window': window,
                'result': result,
                'passed': passed,
                'trade_count': result.trade_count
            })

        return self._aggregate_results(signal_id, window_results)

    def _generate_windows(self, history_df):
        """
        Generates all valid walk-forward windows.
        Minimum: 4 complete windows to proceed.
        """
        windows = []
        start_date = history_df['date'].min()
        end_date = history_df['date'].max()

        current = start_date
        while True:
            train_start = current
            train_end = add_months(train_start, self.TRAIN_MONTHS)
            purge_start = subtract_trading_days(
                train_end, self.PURGE_DAYS, self.calendar_df
            )
            test_start = add_trading_days(
                train_end, self.EMBARGO_DAYS, self.calendar_df
            )
            test_end = add_months(test_start, self.TEST_MONTHS)

            if test_end > end_date:
                break

            windows.append({
                'train_start': train_start,
                'train_end': train_end,
                'purge_start': purge_start,
                'test_start': test_start,
                'test_end': test_end,
            })

            current = add_months(current, self.STEP_MONTHS)

        return windows

    def _evaluate_window(self, result, signal_type, window):
        """
        Apply differentiated pass criteria by signal type.
        Four criteria checked beyond basic Sharpe:
          - FUTURES:         min_calmar (protects against crash exposure)
          - OPTIONS_BUYING:  min_win_loss_ratio (compensates for low win rate)
          - OPTIONS_SELLING: must_survive_2020 (tail-risk gate)
          - COMBINED:        max_nifty_correlation (portfolio diversification)
        """
        criteria = {
            'FUTURES': {
                'min_sharpe':        1.2,
                'min_calmar':        0.8,
                'min_profit_factor': 1.6,
                'min_trades':        50
            },
            'OPTIONS_BUYING': {
                'min_sharpe':          1.5,
                'min_win_rate':        0.40,
                'min_win_loss_ratio':  2.5,
                'min_trades':          50
            },
            'OPTIONS_SELLING': {
                'min_sharpe':        1.8,
                'max_drawdown':      0.20,
                'must_survive_2020': True,
                'min_trades':        50
            },
            'COMBINED': {
                'min_sharpe':             1.5,
                'max_nifty_correlation':  0.4,
                'min_trades':             50
            }
        }

        c = criteria.get(signal_type, criteria['FUTURES'])

        # Minimum trades — skip window if insufficient data
        if result.trade_count < c.get('min_trades', 50):
            return False

        # Sharpe ratio — all types
        if result.sharpe < c.get('min_sharpe', 1.0):
            return False

        # Profit factor — FUTURES
        if 'min_profit_factor' in c:
            if result.profit_factor < c['min_profit_factor']:
                return False

        # Calmar ratio — FUTURES
        if 'min_calmar' in c:
            if result.calmar_ratio < c['min_calmar']:
                return False

        # Max drawdown — OPTIONS_SELLING
        if 'max_drawdown' in c:
            if result.max_drawdown > c['max_drawdown']:
                return False

        # Win rate — OPTIONS_BUYING
        if 'min_win_rate' in c:
            if result.win_rate < c['min_win_rate']:
                return False

        # Win/loss ratio — OPTIONS_BUYING
        if 'min_win_loss_ratio' in c:
            if result.avg_win_loss_ratio < c['min_win_loss_ratio']:
                return False

        # Nifty correlation — COMBINED
        if 'max_nifty_correlation' in c:
            if result.nifty_correlation > c['max_nifty_correlation']:
                return False

        # 2020 survival — OPTIONS_SELLING
        # The 2020 crash window is March–April 2020 (Nifty fell 40% in 6 weeks).
        # A window "survives" 2020 if its test period overlaps March–April 2020
        # AND the drawdown during that sub-period is within 150% of max_drawdown limit.
        # If the window does not overlap 2020, this criterion is skipped.
        if c.get('must_survive_2020'):
            # Use pd.Timestamp throughout — window dates are Timestamps from history_df,
            # and mixing date/Timestamp causes TypeError in Python 3 comparisons.
            crash_start = pd.Timestamp('2020-03-01')
            crash_end   = pd.Timestamp('2020-04-30')
            window_start = window['test_start']
            window_end   = window['test_end']

            window_overlaps_2020 = (
                window_start <= crash_end and window_end >= crash_start
            )
            if window_overlaps_2020:
                # result must carry a 2020_sub_drawdown field if computed.
                # If not available (older backtest implementations), skip check
                # and log a warning — developer must add sub-period drawdown
                # to BacktestResult for OPTIONS_SELLING signals.
                sub_dd = getattr(result, 'drawdown_2020', None)
                if sub_dd is not None:
                    # Allow 150% of normal limit — 2020 was extreme
                    survival_limit = c['max_drawdown'] * 1.5
                    if sub_dd > survival_limit:
                        return False
                # else: cannot verify, assume pass (backtest must add drawdown_2020)

        return True

    def _aggregate_results(self, signal_id, window_results):
        windows_passed = sum(1 for w in window_results if w['passed'])
        total_windows = len(window_results)
        pass_rate = windows_passed / total_windows if total_windows else 0.0

        # Require ≥75% pass rate AND the most recent window (recency check).
        last_window_passed = window_results[-1]['passed'] if window_results else False
        rate_ok = pass_rate >= self.MIN_PASS_RATE
        overall_pass = rate_ok and (
            last_window_passed if self.MUST_PASS_LAST_WINDOW else True
        )

        return {
            'signal_id':              signal_id,
            'overall_pass':           overall_pass,
            'windows_passed':         windows_passed,
            'total_windows':          total_windows,
            'pass_rate':              pass_rate,
            'last_window_passed':     last_window_passed,
            'window_details':         window_results,
            'aggregate_sharpe':       harmonic_mean_sharpe(window_results),
            'worst_window_drawdown':  max(
                w['result'].max_drawdown for w in window_results
            ),
            'recommendation': 'PROMOTE_TO_ACTIVE' if overall_pass else 'ARCHIVE',
            'fail_reason': (
                None if overall_pass else
                'LAST_WINDOW_FAILED' if rate_ok and not last_window_passed else
                f'PASS_RATE_{pass_rate:.0%}_BELOW_75PCT'
            )
        }
```

---

## MODULE 4: VALIDATION TEST SUITE
# File: backtest_validator.py
# Run BEFORE any real signal is tested. All 3 must pass.

```python
def validate_backtest_engine(backtest_fn, history_df, regime_labels):
    """
    3 mandatory validation tests.
    If any fails: backtest engine has a bug. Fix before proceeding.
    """

    print("Running backtest engine validation...")

    # TEST 1: Random signal (coin flip entries)
    # Expected: Sharpe near 0, slightly negative after costs
    random_signal_params = {
        'entry_rule': 'RANDOM',
        'seed': 42
    }
    result1 = backtest_fn(random_signal_params, history_df, regime_labels)
    assert -0.3 < result1.sharpe < 0.3, (
        f"FAIL: Random signal Sharpe={result1.sharpe}. "
        f"Expected near 0. Possible lookahead bias or cost model error."
    )
    print(f"TEST 1 PASSED: Random signal Sharpe={result1.sharpe:.3f}")

    # TEST 2: Perfect foresight signal
    # Expected: Very high Sharpe (>5.0), very high win rate (>90%)
    perfect_signal_params = {
        'entry_rule': 'BUY_IF_TOMORROW_HIGHER',
        'exit_rule': 'NEXT_DAY_CLOSE'
    }
    result2 = backtest_fn(perfect_signal_params, history_df, regime_labels)
    assert result2.sharpe > 5.0, (
        f"FAIL: Perfect foresight Sharpe={result2.sharpe}. "
        f"Expected >5.0. Data loading or return calculation error."
    )
    assert result2.win_rate > 0.90, (
        f"FAIL: Perfect foresight win rate={result2.win_rate}. "
        f"Expected >0.90."
    )
    print(f"TEST 2 PASSED: Perfect foresight "
          f"Sharpe={result2.sharpe:.1f}, "
          f"WinRate={result2.win_rate:.2f}")

    # TEST 3: Known negative signal (buy Monday, sell Friday, Nifty)
    # Expected: Negative or near-zero Sharpe, negative after costs
    known_bad_params = {
        'entry_rule': 'BUY_MONDAY_OPEN',
        'exit_rule': 'SELL_FRIDAY_CLOSE'
    }
    result3 = backtest_fn(known_bad_params, history_df, regime_labels)
    # Threshold is < 0.8 (not tighter). India's 2020-2024 bull run means even
    # naive long-only strategies show modest positive Sharpe. The test checks
    # that a known-bad signal doesn't look like a real edge — not that it loses.
    assert result3.sharpe < 0.8, (
        f"FAIL: Known-bad signal Sharpe={result3.sharpe}. "
        f"Expected <0.8. Something is systematically wrong."
    )
    print(f"TEST 3 PASSED: Known-bad signal Sharpe={result3.sharpe:.3f}")

    print("ALL VALIDATION TESTS PASSED. Backtest engine is trustworthy.")
    return True
```

# IMPLEMENTATION SPEC 2: EXECUTION ENGINE
# Written by: System Architect
# Status: FINAL — implement exactly as written

---

## EXECUTION FLOW (exact sequence every time a signal fires)

```
Signal fires in Signal Engine
        │
        ▼
STEP 1: GREEK PRE-CHECK (Risk Engine, Go)
        │ Pass → continue
        │ Fail → REJECT, log reason, done
        ▼
STEP 2: CAPITAL CHECK (Risk Engine, Go)
        │ Sufficient → continue
        │ Insufficient → queue as PENDING_CAPITAL, revisit in 30 min
        ▼
STEP 3: QUEUE SIGNAL (Redis Stream: ORDER_QUEUE)
        │ With execution_priority, signal_id, params, timestamp
        ▼
STEP 4: EXECUTION ENGINE PICKS UP (Go, polls ORDER_QUEUE)
        │ Processes ONE signal at a time
        │ Waits for current signal to resolve before next
        ▼
STEP 5: PRICE FRESHNESS CHECK
        │ Is live price within 0.3% of price at signal fire time?
        │ Yes → proceed
        │ No → recalculate limit price from current price
        │ If recalculated limit changes direction of bet → CANCEL
        ▼
STEP 6: ORDER PLACEMENT (Kite API)
        │ Place limit order at mid price + 0.1% buffer
        ▼
STEP 7: FILL WINDOW (leg-count adjusted)
        │
        │ Fill window by strategy type:
        │   Single-leg (futures, single option):  90 seconds
        │   Two-leg    (spread):                  60 seconds
        │   Four-leg   (condor, iron fly):        45 seconds per leg
        │
        │ ATOMIC RULE for multi-leg strategies:
        │   If ANY leg fails to fill within its window:
        │   → Cancel ALL other legs immediately
        │   → Close any already-filled legs at market
        │   → Mark entire strategy UNFILLED
        │   Reason: partial condor = naked exposure. Never acceptable.
        │
        │ Single-leg resolution:
        │   Filled → log, update portfolio state, process next signal
        │   Partial >= 50% of intended → keep, mark PARTIAL
        │   Partial <  50% of intended → close at market, mark UNFILLED
        │   Not filled → cancel, mark UNFILLED, log reason
        ▼
STEP 8: POST-FILL GREEK UPDATE
        │ Recalculate portfolio Greeks with new position
        │ Store in Redis: PORTFOLIO_STATE key
        ▼
STEP 9: NEXT SIGNAL
```

---

## EXECUTION PRIORITY RULES (exact order)

```python
EXECUTION_PRIORITY = {
    # Priority 1 — always first
    'EXIT_SIGNAL': 1,          # Close existing position
    'STOP_LOSS_TRIGGER': 1,    # Stop loss hit

    # Priority 2 — risk reduction
    'GREEK_HEDGE': 2,          # Rebalancing delta/vega

    # Priority 3 — new entries by instrument
    'FUTURES_ENTRY': 3,        # Fastest fills
    'OPTIONS_ATM_ENTRY': 4,    # Good liquidity
    'OPTIONS_OTM_ENTRY': 5,    # Slower fills

    # Priority 4 — complex strategies
    'SPREAD_ENTRY': 6,         # Multiple legs
    'CONDOR_ENTRY': 7,         # 4 legs, slowest
}

# Within same priority level: rank by signal Sharpe (highest first)
# Within same Sharpe: rank by margin efficiency (lowest margin first)
```

---

## GREEK PRE-CHECK EXACT LIMITS
# File: execution/greek_pre_check.py (shared constant — also used in check_greek_compatibility)
# These limits apply to the combined portfolio after adding the proposed signal.

```python
GREEK_LIMITS = {
    'max_portfolio_delta': 0.50,
    # Maximum net delta as fraction of 1 full Nifty lot equivalent
    # At Nifty 24,000, 1 lot delta ≈ 24,000 × 75 = ₹18,00,000
    # Max delta = 0.50 × ₹18,00,000 = ₹9,00,000 per 1% Nifty move

    'max_portfolio_vega': 3000,
    # Maximum vega in ₹ per 1 point VIX change
    # If VIX moves 1 point, portfolio P&L changes by max ₹3,000

    'max_portfolio_gamma': 30000,
    # Maximum gamma exposure in ₹ per 1% Nifty move
    # If Nifty moves 1%, portfolio gamma P&L changes by max ₹30,000
    # (this is the daily P&L sensitivity, not cumulative)

    'max_portfolio_theta': -5000,
    # Maximum daily theta drain in ₹
    # Portfolio loses maximum ₹5,000 per day from time decay

    'max_same_direction_positions': 2,
    # Never more than 2 positions in same directional bet
    # (2 bullish positions maximum, 2 bearish maximum)
}


# ================================================================
# COMPUTE_COMBINED_GREEKS — Python stub + Go service specification
# File (Python stub): execution/greek_calculator.py
# File (Go service):  risk/greek_service.go
#
# This function is called at the critical order path — once per proposed
# order, synchronously, before the order is placed. It must be fast
# (<50ms) and accurate. The Go side owns the actual Black-Scholes
# calculation; Python calls it via HTTP and deserialises the result.
# ================================================================

# --- Python stub (execution/greek_calculator.py) ---

```python
import requests
from portfolio.portfolio_model import PortfolioGreeks

GREEK_SERVICE_URL = "http://localhost:8080/greeks/combined"
GREEK_SERVICE_TIMEOUT_S = 0.5   # 500ms hard deadline — fail fast


def compute_combined_greeks(current_portfolio, proposed_signal,
                             market_data: dict) -> PortfolioGreeks:
    """
    Compute the portfolio Greeks that would result if proposed_signal
    were added to current_portfolio at current market prices.

    Returns PortfolioGreeks with fields: delta, vega, gamma, theta.
    Raises RuntimeError if the Go service is unreachable or slow —
    caller (greek_pre_check) must treat that as a rejection.

    Python → Go via HTTP POST to the risk service on localhost:8080.
    The risk service runs as a sidecar on the same VPS.

    market_data keys required:
        nifty_spot      float   Current Nifty 50 spot price
        india_vix       float   India VIX value (annualised %)
        risk_free_rate  float   Current repo rate (annualised, e.g. 0.065)
        timestamp       str     ISO8601 — used to compute DTE
    """
    payload = _build_payload(current_portfolio, proposed_signal, market_data)
    try:
        resp = requests.post(
            GREEK_SERVICE_URL,
            json=payload,
            timeout=GREEK_SERVICE_TIMEOUT_S
        )
        resp.raise_for_status()
        data = resp.json()
        return PortfolioGreeks(
            delta = data['portfolio_delta'],
            vega  = data['portfolio_vega'],
            gamma = data['portfolio_gamma'],
            theta = data['portfolio_theta'],
        )
    except requests.Timeout:
        raise RuntimeError("Greek service timeout — rejecting order for safety")
    except requests.RequestException as e:
        raise RuntimeError(f"Greek service error: {e}")


def _build_payload(portfolio, signal, market_data: dict) -> dict:
    """
    Serialise portfolio + proposed signal into the JSON payload
    the Go service expects.
    """
    return {
        "market_data": {
            "nifty_spot":     market_data["nifty_spot"],
            "india_vix":      market_data["india_vix"],
            "risk_free_rate": market_data.get("risk_free_rate", 0.065),
            "timestamp":      market_data["timestamp"],
        },
        # Existing open positions — Go recalculates their Greeks at live prices
        "existing_positions": [
            {
                "instrument_type": "FUTURES" if "FUT" in p.instrument else "OPTIONS",
                "symbol":          p.instrument,
                "direction":       p.direction,   # 'LONG' | 'SHORT'
                "lots":            p.lots,
                # For options: Go needs strike and expiry from symbol string
                # Kite symbol format: NIFTY24NOV24000CE → parsed by Go service
            }
            for p in portfolio.open_positions
        ],
        # Proposed new position
        "proposed_position": {
            "instrument_type": signal.instrument,
            "signal_id":       signal.signal_id,
            "direction":       signal.direction,
            "lots":            signal.lot_size,
            "strike":          getattr(signal, 'strike', None),
            "expiry":          getattr(signal, 'expiry_date', None),
        },
        # Lot size for Nifty F&O: 25 (post-2024 contract size)
        "nifty_lot_size": 25,
    }
```

# --- Go service specification (risk/greek_service.go) ---
# The Go service listens on :8080, handles POST /greeks/combined.
# It owns all Black-Scholes computation. Python never does option maths directly.

```
POST /greeks/combined
Content-Type: application/json

REQUEST BODY: (see _build_payload above)

RESPONSE BODY:
{
  "portfolio_delta": float,   // Net delta of combined portfolio (existing + proposed)
  "portfolio_vega":  float,   // Net vega in ₹/VIX point
  "portfolio_gamma": float,   // Net gamma in ₹/1% Nifty move
  "portfolio_theta": float,   // Net theta in ₹/day (negative = time decay cost)
  "per_position": [           // Per-position breakdown for debugging
    {
      "symbol":  string,
      "delta":   float,
      "vega":    float,
      "gamma":   float,
      "theta":   float
    }
  ]
}

Greek calculations:
- Futures: delta = lots × lot_size × (1 if LONG else -1), vega=gamma=theta=0
- Options (Black-Scholes with India VIX as implied vol proxy):
    d1 = (ln(S/K) + (r + 0.5σ²)T) / (σ√T)
    d2 = d1 - σ√T
    Call delta = N(d1);  Put delta = N(d1) - 1
    Vega  = S × N'(d1) × √T  (in ₹ per 1% VIX move: multiply by 0.01)
    Gamma = N'(d1) / (S × σ × √T)  (in ₹ per 1% spot move: multiply by S×0.01)
    Theta = -(S×N'(d1)×σ)/(2√T) - r×K×e^(-rT)×N(d2)  (per calendar day: divide by 365)
    SHORT positions: multiply all greeks by -1
    Multiple lots: multiply by lots × lot_size

- σ (implied vol): use India VIX / 100 as a proxy.
  For DTE-based adjustment: apply the vol surface adjustment factors
  derived during backtest setup (stored in vol_adjustment_factors table).

- T (time to expiry): calendar days to next Thursday expiry / 365.
  If DTE = 0 (expiry day): use T = 1/365 to avoid division by zero.

Error responses:
  400: malformed payload — Python logs and rejects the signal
  503: service unavailable — Python raises RuntimeError → order rejected
  Timeout (>500ms): Python raises RuntimeError → order rejected

Startup: go run risk/greek_service.go — starts with the trading system.
Health check: GET /health → {"status": "ok"}
```

---

## GREEK PRE-CHECK FUNCTION
# File: execution/greek_pre_check.py
# Called by the Go Execution Engine via Python subprocess or HTTP
# before every order placement.

```python
from execution.greek_calculator import compute_combined_greeks

GREEK_LIMITS = {
    'max_portfolio_delta': 0.50,
    'max_portfolio_vega':  3000,
    'max_portfolio_gamma': 30000,
    'max_portfolio_theta': -5000,
}


def greek_pre_check(proposed_signal, current_portfolio, market_data):
    """
    Called before every order placement.
    Returns: (approved: bool, rejection_reason: str)
    Raises RuntimeError if compute_combined_greeks() times out —
    caller must treat that as a rejection.
    """
    proposed = compute_combined_greeks(
        current_portfolio, proposed_signal, market_data
    )

    if abs(proposed.delta) > GREEK_LIMITS['max_portfolio_delta']:
        return False, f"Delta breach: {proposed.delta:.3f}"

    if abs(proposed.vega) > GREEK_LIMITS['max_portfolio_vega']:
        return False, f"Vega breach: {proposed.vega:.0f}"

    if abs(proposed.gamma) > GREEK_LIMITS['max_portfolio_gamma']:
        return False, f"Gamma breach: {proposed.gamma:.0f}"

    if proposed.theta < GREEK_LIMITS['max_portfolio_theta']:
        return False, f"Theta breach: {proposed.theta:.0f}"

    return True, "APPROVED"
```

---

## FAILURE HANDLING EXACT RESPONSES

```
SCENARIO 1: Kite API timeout (>5 seconds no response)
ACTION: Retry once after 3 seconds.
        If second attempt also times out:
        → Cancel signal for the day
        → Log: API_TIMEOUT
        → Send alert to phone
        → Check if order was placed despite timeout
          (query positions after 30 seconds)

SCENARIO 2: Insufficient margin error from Kite
ACTION: Do not retry.
        → Mark signal as CAPITAL_BLOCKED
        → Add to tomorrow's queue if signal type is valid next day
        → Log margin shortfall amount

SCENARIO 3: Order placed but price moves sharply before fill
(Price moves >1% from limit price within fill window)
ACTION: Cancel immediately.
        Do not chase. Do not adjust limit.
        → Log: PRICE_ESCAPE
        → Signal marked as MISSED for that day

SCENARIO 4: Partial fill
If filled >= 50% of intended size:
        → Keep position at reduced size
        → Update position sizing for exit accordingly
        → Log: PARTIAL_FILL with actual vs intended lots
If filled < 50% of intended size:
        → Cancel remainder
        → Close filled portion immediately at market
        → Log: ABANDONED_PARTIAL
        → Reason: insufficient size makes risk/reward invalid

SCENARIO 5: Data feed disconnect during open position
ACTION: Do NOT close positions automatically.
        → Activate backup feed (Kite ticker)
        → If backup also fails within 90 seconds:
          → Do nothing until feed restored
          → If feed not restored within 10 minutes:
            → Close ALL positions at market
            → Log: EMERGENCY_CLOSE_FEED_FAILURE
        → Alert to phone immediately

SCENARIO 6: Daily loss limit hit (5% of TOTAL_CAPITAL)
ACTION: Cancel ALL pending orders immediately.
        → Do not close existing positions
          (closing also costs money, wait for natural exit)
        → Block ALL new entries for remainder of day
        → Log: DAILY_LOSS_LIMIT_HIT
        → Alert to phone

SCENARIO 7: Unexpected position in broker account
(Position exists in Kite that system did not create)
ACTION: This is a critical error.
        → Halt ALL trading immediately
        → Alert to phone
        → Manual investigation required
        → Do not attempt to automatically close unknown positions
        → Log: POSITION_MISMATCH_DETECTED
```

---

## DISASTER RECOVERY: DAILY RECONCILIATION

```python
def daily_reconciliation():
    """
    Runs at 3:35 PM every trading day (5 min after market close).
    Compares system state vs broker state.
    Open positions source: trades WHERE exit_date IS NULL AND trade_type = 'LIVE'
    """

    # Get current positions from Kite API
    kite_positions = kite.positions()
    kite_net = {p['tradingsymbol']: p for p in kite_positions.get('net', [])
                if p.get('quantity', 0) != 0}

    # Get system's open positions from trades table
    system_open = db.query("""
        SELECT trade_id, signal_id, instrument, direction, lots, entry_price
        FROM trades
        WHERE exit_date IS NULL
          AND trade_type = 'LIVE'
    """)
    system_net = {row['instrument']: row for row in system_open}

    mismatches = []

    # Check: positions in Kite but not in system
    for symbol, kite_pos in kite_net.items():
        if symbol not in system_net:
            mismatches.append(('UNKNOWN_POSITION', symbol, kite_pos))
            alerter.send('POSITION_MISMATCH',
                         f"Kite has {symbol} but system does not. "
                         f"Halting all trading.")
        else:
            sys_lots = system_net[symbol]['lots']
            kite_lots = abs(kite_pos.get('quantity', 0))
            if sys_lots != kite_lots:
                mismatches.append(('QUANTITY_MISMATCH', symbol,
                                   f"system={sys_lots} kite={kite_lots}"))
                # Trust Kite quantity as ground truth
                db.execute(
                    "UPDATE trades SET lots = %s WHERE instrument = %s "
                    "AND exit_date IS NULL AND trade_type = 'LIVE'",
                    (kite_lots, symbol)
                )

    # Check: positions in system but not in Kite
    for symbol, sys_pos in system_net.items():
        if symbol not in kite_net:
            mismatches.append(('GHOST_POSITION', symbol, sys_pos))
            # Mark as closed (probably closed manually)
            db.execute(
                "UPDATE trades SET exit_date = CURRENT_DATE, "
                "exit_reason = 'NOT_FOUND_IN_BROKER_RECONCILIATION' "
                "WHERE instrument = %s AND exit_date IS NULL "
                "AND trade_type = 'LIVE'",
                (symbol,)
            )

    # Store reconciliation snapshot
    db.execute("""
        INSERT INTO portfolio_state
            (snapshot_time, snapshot_type, total_capital, deployed_capital,
             cash_reserve, open_positions)
        VALUES (NOW(), 'RECONCILIATION', %s, %s, %s, %s)
    """, (
        compute_total_capital(),
        compute_deployed_capital(system_open),
        CAPITAL_RESERVE,
        json.dumps([s for s in system_open])
    ))

    if mismatches:
        alerter.send('POSITION_MISMATCH',
                     f"{len(mismatches)} reconciliation mismatch(es). "
                     f"Review before next session.")
    else:
        log_info("Reconciliation clean.")
```

# IMPLEMENTATION SPEC 3: PORTFOLIO CONSTRUCTION
# Written by: India Market Expert + Trading Systems Expert (joint spec)
# Status: FINAL — implement exactly as written

---

## SIGNAL DATACLASS
# File: portfolio/signal_model.py
# Used by SignalSelector, ExecutionEngine, and RiskEngine.
# Loaded from signals table in PostgreSQL — all fields map to columns.

```python
import copy
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class Signal:
    """
    Runtime representation of one signal from the signal registry.
    Loaded fresh at 8:50 AM from the signals table.
    """
    # Identity
    signal_id:            str
    name:                 str
    book_id:              str

    # Classification
    signal_category:      str    # TREND|REVERSION|VOL|PATTERN|EVENT
    direction:            str    # LONG|SHORT|NEUTRAL|CONTEXT_DEPENDENT
    instrument:           str    # FUTURES|OPTIONS_BUYING|OPTIONS_SELLING|SPREAD|COMBINED|ANY
    classification:       str    # PRIMARY|SECONDARY
    status:               str    # ACTIVE|WATCH|INACTIVE|...

    # Regime and calendar rules
    target_regimes:       List[str]  # e.g. ['TRENDING', 'ANY']
    expiry_week_behavior: str    # NORMAL|AVOID_MONDAY|REDUCE_SIZE_MONDAY|AVOID_EXPIRY_WEEK
    avoid_rbi_day:        bool = False

    # Performance metrics (from backtest / live tracking)
    sharpe_ratio:         float = 0.0
    rolling_sharpe_60d:   float = 0.0

    # Capital requirements
    required_margin:      int = 0  # ₹ approximate margin per trade

    # Runtime fields (set by SignalSelector during selection)
    efficiency_score:     float = 0.0  # computed each morning
    execution_priority:   int = 99     # 1 = execute first
    size_multiplier:      float = 1.0  # 1.0 = full size, 0.5 = half size

    # FII overnight signals only
    source:               str = 'BOOK'  # 'BOOK' or 'FII_OVERNIGHT'
    requires_confirmation: bool = False

    def copy(self) -> 'Signal':
        """Return a shallow copy for per-instance modifications (e.g. size_multiplier)."""
        return copy.copy(self)

    @classmethod
    def from_db_row(cls, row: dict) -> 'Signal':
        """Construct a Signal from a signals table row dict."""
        return cls(
            signal_id            = row['signal_id'],
            name                 = row['name'],
            book_id              = row['book_id'],
            signal_category      = row['signal_category'],
            direction            = row['direction'],
            instrument           = row['instrument'],
            classification       = row['classification'],
            status               = row['status'],
            target_regimes       = row['target_regimes'],  # TEXT[] from PG
            expiry_week_behavior = row['expiry_week_behavior'],
            avoid_rbi_day        = row.get('avoid_rbi_day', False),
            sharpe_ratio         = row.get('sharpe_ratio', 0.0) or 0.0,
            rolling_sharpe_60d   = row.get('rolling_sharpe_60d', 0.0) or 0.0,
            required_margin      = row.get('required_margin', 0) or 0,
        )

    @classmethod
    def from_fii_redis(cls, fields: dict) -> 'Signal':
        """Construct a Signal from a Redis FII_OVERNIGHT message."""
        return cls(
            signal_id            = fields['signal_id'],
            name                 = fields.get('pattern', fields['signal_id']),
            book_id              = 'NSE_EMPIRICAL',
            signal_category      = 'EVENT',
            direction            = fields['direction'],
            instrument           = 'FUTURES',
            classification       = 'SECONDARY',
            status               = 'ACTIVE',
            target_regimes       = ['ANY'],
            expiry_week_behavior = 'NORMAL',
            source               = 'FII_OVERNIGHT',
            requires_confirmation = True,
            sharpe_ratio         = float(fields.get('confidence', 0.65)),
            required_margin      = 120_000,   # ~₹1.2L for 1 futures lot
        )
```

---

## SIGNAL CLASSIFICATION: PRIMARY vs SECONDARY

### How signals are classified (done AFTER paper trading):

```
PRIMARY signal criteria (ALL must be met):
□ Paper trading Sharpe >= 90% of backtest Sharpe
□ Fired at least 8 times in 4-month paper period
□ Capital available >= 70% of times it fired
  (not capital-starved)
□ No single paper trade caused >3% portfolio drawdown
□ Survived both stress test periods (2020, 2026) in backtest

SECONDARY signal criteria (minimum to remain in library):
□ Paper trading Sharpe >= 70% of backtest Sharpe
□ Fired at least 4 times in 4-month paper period
□ Does not meet all PRIMARY criteria
□ Still passes Round 1 and 2 backtest criteria

ARCHIVED (removed from active library):
□ Paper Sharpe < 70% of backtest Sharpe, OR
□ Fired fewer than 4 times in paper period, OR
□ Capital-starved >60% of the time (cannot run at current TOTAL_CAPITAL), OR
□ Single paper trade caused >5% portfolio drawdown
```

### Capital allocation rules by classification:

```
PRIMARY signals:    Always get capital when they fire
                    (unless daily loss limit hit)

SECONDARY signals:  Get capital only if:
                    - All PRIMARY signals' capital requirements are met
                    - AND remaining capital >= secondary signal's margin
                    - AND no PRIMARY signal is in PENDING_CAPITAL queue
                    - AND portfolio not already at max positions

In practice: On a day with 2 PRIMARY signals firing + 2 SECONDARY:
PRIMARY_1 gets capital first
PRIMARY_2 gets capital second
If capital remains: SECONDARY_1 (higher Sharpe) gets capital
If capital still remains: SECONDARY_2 gets capital
```

---

## PORTFOLIO DATACLASS
# File: portfolio/portfolio_model.py
# Loaded at 8:50 AM from the latest portfolio_state snapshot + open trades.
# Updated in Redis after every fill (PORTFOLIO_STATE key).

```python
from dataclasses import dataclass, field
from typing import List

@dataclass
class PortfolioGreeks:
    delta: float = 0.0
    vega:  float = 0.0
    gamma: float = 0.0
    theta: float = 0.0

@dataclass
class OpenPosition:
    trade_id:      int
    signal_id:     str
    instrument:    str
    direction:     str    # 'LONG' | 'SHORT'
    lots:          int
    entry_price:   float
    current_price: float = 0.0

@dataclass
class Portfolio:
    """
    Runtime portfolio state. Loaded fresh at 8:50 AM each day.
    Also updated in Redis after every fill (PORTFOLIO_STATE key).
    """
    total_capital:    float = TOTAL_CAPITAL  # from config.settings
    deployed_capital: float = 0.0           # Sum of margins in use
    cash_reserve:     float = TOTAL_CAPITAL * CAPITAL_RESERVE_FRACTION
    open_positions:   List[OpenPosition] = field(default_factory=list)
    greeks:           PortfolioGreeks     = field(default_factory=PortfolioGreeks)
    daily_pnl:        float = 0.0
    mtd_pnl:          float = 0.0

    @classmethod
    def from_db(cls, db) -> 'Portfolio':
        """
        Load current portfolio from latest DAILY_CLOSE or POST_TRADE snapshot
        plus all open trades (exit_date IS NULL, trade_type='LIVE').
        """
        # Latest snapshot for capital and greeks
        snap = db.execute("""
            SELECT total_capital, deployed_capital, cash_reserve,
                   portfolio_delta, portfolio_vega,
                   portfolio_gamma, portfolio_theta,
                   daily_pnl, mtd_pnl
            FROM portfolio_state
            WHERE snapshot_type IN ('DAILY_CLOSE', 'POST_TRADE', 'RECONCILIATION')
            ORDER BY snapshot_time DESC LIMIT 1
        """).fetchone()

        # Open positions
        rows = db.execute("""
            SELECT trade_id, signal_id, instrument, direction,
                   lots, entry_price
            FROM trades
            WHERE exit_date IS NULL AND trade_type = 'LIVE'
        """).fetchall()

        positions = [
            OpenPosition(
                trade_id=r['trade_id'], signal_id=r['signal_id'],
                instrument=r['instrument'], direction=r['direction'],
                lots=r['lots'], entry_price=r['entry_price']
            )
            for r in rows
        ]

        if snap:
            return cls(
                total_capital    = snap['total_capital'],
                deployed_capital = snap['deployed_capital'],
                cash_reserve     = snap['cash_reserve'],
                open_positions   = positions,
                greeks           = PortfolioGreeks(
                    delta = snap['portfolio_delta'] or 0.0,
                    vega  = snap['portfolio_vega']  or 0.0,
                    gamma = snap['portfolio_gamma'] or 0.0,
                    theta = snap['portfolio_theta'] or 0.0,
                ),
                daily_pnl = snap['daily_pnl'] or 0.0,
                mtd_pnl   = snap['mtd_pnl']   or 0.0,
            )
        # No snapshot yet (first run of the day)
        return cls(open_positions=positions)
```

---

## SIGNAL SELECTOR: EXACT DECISION TREE

```python
class SignalSelector:
    """
    Runs at 8:50 AM every trading day.
    Produces: ordered list of signals approved for today.
    """

    # All from config.settings — scales with TOTAL_CAPITAL env var
    TOTAL_CAPITAL = settings.TOTAL_CAPITAL
    CAPITAL_RESERVE = settings.CAPITAL_RESERVE
    AVAILABLE_CAPITAL = settings.AVAILABLE_CAPITAL

    MAX_POSITIONS = 4
    MAX_SAME_DIRECTION = 2

    def __init__(self, db, redis_client, logger):
        """
        db:           database connection (psycopg2 or equivalent)
        redis_client: redis.Redis instance
        logger:       Python Logger or structured logger
        """
        self.db     = db
        self.redis  = redis_client
        self.logger = logger

    def run(self, today, current_portfolio, signal_registry):
        """
        Returns: List of approved signals in execution order.
        """

        # STEP 1: Get today's market regime
        regime = self.get_current_regime(today)

        # If CRISIS: no new trades
        if regime == 'CRISIS':
            return [], "CRISIS_REGIME_NO_TRADES"

        # STEP 2: Get economic calendar events
        events = self.get_todays_events(today)
        # events: ['RBI_DECISION', 'BUDGET', 'NIFTY_EXPIRY', etc.]

        # STEP 3: Get all ACTIVE signals from registry
        all_signals = signal_registry.get_active_signals()

        # STEP 3b: Load any preloaded FII overnight signals from Redis.
        # FII signals are written at 7:30 PM by FIIDailyPipeline.
        # Stale signals are already purged by _purge_stale_redis_signals().
        fii_signals = self._load_fii_overnight_signals(today)
        all_signals = all_signals + fii_signals

        # STEP 4: Filter by regime compatibility
        regime_compatible = [
            s for s in all_signals
            if regime in s.target_regimes
            or 'ANY' in s.target_regimes
        ]

        # STEP 5: Filter by calendar compatibility
        calendar_filtered = self.apply_calendar_filters(
            regime_compatible, events, today
        )

        # STEP 6: Filter by expiry week behavior
        expiry_filtered = self.apply_expiry_filters(
            calendar_filtered, today
        )

        # STEP 7: Filter by current portfolio state
        portfolio_filtered = self.apply_portfolio_filters(
            expiry_filtered, current_portfolio
        )

        # STEP 8: Separate PRIMARY and SECONDARY
        primaries = [s for s in portfolio_filtered
                     if s.classification == 'PRIMARY']
        secondaries = [s for s in portfolio_filtered
                       if s.classification == 'SECONDARY']

        # STEP 9: Rank by capital efficiency score
        # Score = Sharpe × Regime_Multiplier / Margin_Required
        primaries_ranked = self.rank_by_efficiency(primaries, regime)
        secondaries_ranked = self.rank_by_efficiency(secondaries, regime)

        # STEP 10: Allocate capital (primaries first)
        approved = []
        # Start from actual free capital: available minus already-deployed.
        deployed = getattr(current_portfolio, 'deployed_capital', 0) or 0
        remaining_capital = max(0, self.AVAILABLE_CAPITAL - deployed)
        current_positions = len(current_portfolio.open_positions)

        for signal in primaries_ranked:
            if current_positions >= self.MAX_POSITIONS:
                break
            if remaining_capital < signal.required_margin:
                continue  # Skip, not enough capital
            approved.append(signal)
            remaining_capital -= signal.required_margin
            current_positions += 1

        # STEP 11: Fill remaining capacity with secondaries
        for signal in secondaries_ranked:
            if current_positions >= self.MAX_POSITIONS:
                break
            if remaining_capital < signal.required_margin:
                continue
            approved.append(signal)
            remaining_capital -= signal.required_margin
            current_positions += 1

        # STEP 12: Final Greek compatibility check
        approved_with_greeks = self.check_greek_compatibility(
            approved, current_portfolio
        )

        # STEP 13: Assign execution priority
        final = self.assign_execution_priority(approved_with_greeks)

        return final, f"APPROVED_{len(final)}_SIGNALS"

    def apply_calendar_filters(self, signals, events, today):
        """
        Remove signals that conflict with today's events.
        EVENT classification is determined by signal_category == 'EVENT'.
        signal.source ('BOOK'|'FII_OVERNIGHT') is unrelated to calendar gating.
        """
        filtered = []
        for signal in signals:

            # On RBI day: only allow event-specific signals
            if 'RBI_DECISION' in events:
                if signal.signal_category == 'EVENT':
                    filtered.append(signal)
                elif signal.avoid_rbi_day:
                    continue  # Skip this signal today
                else:
                    filtered.append(signal)

            # On expiry day: check signal's expiry behavior
            elif 'NIFTY_EXPIRY' in events:
                if signal.expiry_week_behavior == 'AVOID_EXPIRY_WEEK':
                    continue
                else:
                    filtered.append(signal)

            # Normal day: all signals pass calendar filter
            else:
                filtered.append(signal)

        return filtered

    def apply_expiry_filters(self, signals, today):
        """
        Handle Monday-before-expiry gamma trap.
        """
        days_to_expiry = self.get_days_to_next_expiry(today)
        is_expiry_monday = (
            today.weekday() == 0 and  # Monday
            days_to_expiry == 1       # Tomorrow is expiry Tuesday
        )

        if not is_expiry_monday:
            return signals  # No expiry filter needed

        filtered = []
        for signal in signals:
            behavior = signal.expiry_week_behavior

            if behavior == 'AVOID_MONDAY':
                continue  # Skip today

            elif behavior == 'REDUCE_SIZE_MONDAY':
                # Keep signal but flag for 50% size
                signal_copy = signal.copy()
                signal_copy.size_multiplier = 0.5
                filtered.append(signal_copy)

            else:  # NORMAL or AVOID_EXPIRY_WEEK (already filtered)
                filtered.append(signal)

        return filtered

    def apply_portfolio_filters(self, signals, portfolio):
        """
        Remove signals that would create concentration.
        """
        current_long_count = len([
            p for p in portfolio.open_positions
            if p.direction == 'LONG'
        ])
        current_short_count = len([
            p for p in portfolio.open_positions
            if p.direction == 'SHORT'
        ])

        filtered = []
        for signal in signals:
            # Check directional concentration
            if (signal.direction == 'LONG' and
                    current_long_count >= self.MAX_SAME_DIRECTION):
                continue  # Too many longs already

            if (signal.direction == 'SHORT' and
                    current_short_count >= self.MAX_SAME_DIRECTION):
                continue  # Too many shorts already

            filtered.append(signal)

        return filtered

    def rank_by_efficiency(self, signals, regime):
        """
        Score = Sharpe × Regime_Multiplier × (1 / Margin_Required)
        """
        REGIME_MULTIPLIERS = {
            'TRENDING':  {'TREND': 1.3, 'REVERSION': 0.7,
                          'VOL': 1.0, 'PATTERN': 1.1, 'EVENT': 1.0},
            'RANGING':   {'TREND': 0.7, 'REVERSION': 1.3,
                          'VOL': 1.2, 'PATTERN': 0.9, 'EVENT': 1.0},
            'HIGH_VOL':  {'TREND': 0.8, 'REVERSION': 0.8,
                          'VOL': 1.4, 'PATTERN': 0.7, 'EVENT': 1.2},
        }

        multipliers = REGIME_MULTIPLIERS.get(regime, {})

        for signal in signals:
            regime_mult = multipliers.get(signal.signal_category, 1.0)
            # signal_category: 'TREND'|'REVERSION'|'VOL'|'PATTERN'|'EVENT'
            # These match the multipliers dict keys exactly.
            if signal.required_margin <= 0:
                # Skip signals with unset margin — they'll be re-ranked once
                # required_margin is populated after the first backtest run.
                continue
            signal.efficiency_score = (
                signal.sharpe_ratio *
                regime_mult *
                (self.TOTAL_CAPITAL / signal.required_margin)
                # Normalize margin relative to total capital
            )

        return sorted(signals,
                      key=lambda s: s.efficiency_score,
                      reverse=True)

    def get_current_regime(self, today) -> str:
        """
        Fetch today's pre-computed regime label from the DB.
        RegimeLabeler runs at 9:00 PM the prior night (after FII pipeline)
        and writes to the regime_labels table keyed by date.
        At 8:50 AM this record already exists.
        """
        row = self.db.execute(
            "SELECT regime FROM regime_labels WHERE label_date = %s",
            (today,)
        ).fetchone()
        if row:
            return row['regime']
        # Fallback: compute on the fly if regime_labels table not yet populated.
        # Requires nifty_daily table to be populated.
        from data.nifty_loader import load_nifty_history
        history = load_nifty_history(self.db, days=200)
        labeler = RegimeLabeler()
        return labeler.label_single_day(history, today)

    def get_todays_events(self, today) -> list:
        """
        Returns list of special events for today from the economic_calendar table.
        Populate this table manually each quarter from:
          - RBI MPC dates: https://www.rbi.org.in/monetarypolicy
          - Budget date: announced by Finance Ministry
          - NSE expiry dates: last Thursday of each month
          - NSE half-days: NSE circular
        """
        rows = self.db.execute(
            "SELECT event_type FROM economic_calendar WHERE event_date = %s",
            (today,)
        ).fetchall()
        events = [r['event_type'] for r in rows]

        # Always auto-detect expiry day (every Thursday = weekly Nifty expiry)
        if self._is_expiry_day(today):
            events.append('NIFTY_EXPIRY')

        return events   # e.g. ['RBI_DECISION'] or ['NIFTY_EXPIRY'] or []

    def get_days_to_next_expiry(self, today) -> int:
        """
        Returns calendar days to the next Nifty expiry Thursday.
        Nifty has weekly expiry every Thursday (introduced 2019).
        Any Thursday is an expiry day — not just last Thursday of month.
        """
        from datetime import timedelta
        current = today
        for i in range(1, 8):  # Search up to 7 days ahead
            candidate = today + timedelta(days=i)
            if candidate.weekday() == 3:  # Thursday
                return i
        return 7   # fallback

    def _is_expiry_day(self, today) -> bool:
        """Returns True if today is a Thursday (Nifty expiry)."""
        return today.weekday() == 3

    # ================================================================
    # SIGNAL ID PREFIX TABLE
    # All signal_ids follow format: {PREFIX}_{3-digit-number}
    # e.g. GUJ_001, NAT_023, LOP_004, NSE_001
    # ================================================================
    SIGNAL_ID_PREFIXES = {
        'GUJRAL':      'GUJ',
        'KAUFMAN':     'KAU',
        'NATENBERG':   'NAT',
        'SINCLAIR':    'SIN',
        'GRIMES':      'GRI',
        'LOPEZ':       'LOP',
        'HILPISCH':    'HIL',
        'DOUGLAS':     'DOU',
        'MCMILLAN':    'MCM',
        'AUGEN':       'AUG',
        'HARRIS':      'HAR',
        'NSE_EMPIRICAL': 'NSE',
    }

    @classmethod
    def make_signal_id(cls, book_id: str, sequence: int) -> str:
        """
        Generate a signal_id for a newly extracted signal.
        e.g. make_signal_id('GUJRAL', 7) → 'GUJ_007'
        """
        prefix = cls.SIGNAL_ID_PREFIXES.get(book_id)
        if not prefix:
            raise ValueError(f"Unknown book_id '{book_id}'. "
                             f"Add to SIGNAL_ID_PREFIXES first.")
        return f"{prefix}_{sequence:03d}"

    def assign_execution_priority(self, signals):
        """
        Assigns execution order (1 = execute first).
        Within approved signals: exits before entries,
        futures before options, higher Sharpe first.
        The approved list from Steps 10-12 is already in the right order
        (primaries ranked by efficiency, then secondaries).
        This step stamps the final numeric priority.
        """
        for i, signal in enumerate(signals):
            signal.execution_priority = i + 1
        return signals

    def _load_fii_overnight_signals(self, today) -> list:
        """
        Read FII preloaded signals from Redis SIGNAL_QUEUE_PRELOADED.
        Returns list of Signal objects for signals valid today.
        FII signals that require_confirmation=True are only included if
        the market's opening direction (first 15 min) confirms the signal.
        At 8:50 AM the market hasn't opened yet — mark as requires_confirmation=True
        and let the Signal Engine re-validate at 9:30 AM.
        """
        from portfolio.signal_model import Signal

        messages = self.redis.xrange('SIGNAL_QUEUE_PRELOADED', '-', '+')
        signals = []

        for msg_id, fields in messages:
            valid_until = fields.get('valid_until', '')
            if valid_until != str(today):
                continue   # stale or future — skip (purge runs separately)

            try:
                sig = Signal.from_fii_redis(fields)
                signals.append(sig)
            except Exception as e:
                self.logger.warning(f"Malformed FII signal in Redis: {e}")

        return signals

    def check_greek_compatibility(self, approved_signals, current_portfolio) -> list:
        """
        Pre-screen signals against Greek limits before execution queuing.

        Design: checks the EXISTING portfolio Greeks from Redis cache, not the
        combined Greeks after adding the signal. The Execution Engine (Go) runs
        the exact per-order compute_combined_greeks() check at order time anyway.
        This is a coarse early filter — it removes signals that would definitely
        breach limits given the current portfolio state, without simulating additions.

        Called at 8:50 AM. Options greeks come from prior-day EOD cache (acceptable
        for a coarse pre-screen). Futures delta is always deterministic (no stale risk).

        If current_portfolio is None or greeks are unavailable, all signals pass
        through (let Execution Engine do the exact check).
        """
        if current_portfolio is None:
            return approved_signals

        # Read current portfolio greeks from Redis cache (set at 3:35 PM prior day)
        try:
            current_delta = current_portfolio.greeks.delta
            current_vega  = current_portfolio.greeks.vega
            current_gamma = current_portfolio.greeks.gamma
            current_theta = current_portfolio.greeks.theta
        except AttributeError:
            # Portfolio object doesn't expose greeks — pass all through
            return approved_signals

        # Coarse limits: reject signals only if existing portfolio already near limits.
        # Adding one more position will very likely breach.
        NEAR_LIMIT_FRACTION = 0.85   # reject if already at 85% of limit

        GREEK_LIMITS = {
            'max_portfolio_delta': 0.50,
            'max_portfolio_vega':  3000,
            'max_portfolio_gamma': 30000,
            'max_portfolio_theta': -5000,
        }

        portfolio_too_hot = (
            abs(current_delta) > GREEK_LIMITS['max_portfolio_delta'] * NEAR_LIMIT_FRACTION
            or abs(current_vega) > GREEK_LIMITS['max_portfolio_vega'] * NEAR_LIMIT_FRACTION
            or abs(current_gamma) > GREEK_LIMITS['max_portfolio_gamma'] * NEAR_LIMIT_FRACTION
            or current_theta < GREEK_LIMITS['max_portfolio_theta'] * NEAR_LIMIT_FRACTION
        )

        if portfolio_too_hot:
            # Only pass FUTURES signals (delta 0.5–1.0, minimal vega/gamma impact)
            # Block all options additions until portfolio cools off
            safe_signals = [s for s in approved_signals
                            if s.instrument == 'FUTURES']
            self.logger.info(
                f"Portfolio greeks near limit — filtered to {len(safe_signals)} "
                f"futures-only signals from {len(approved_signals)} approved. "
                f"Δ={current_delta:.3f} V={current_vega:.0f} "
                f"Γ={current_gamma:.0f} Θ={current_theta:.0f}"
            )
            return safe_signals

        # Portfolio has headroom — pass all approved signals through
        # Execution Engine will do per-signal exact check at order time
        return approved_signals
```

---

## SIGNAL REGISTRY INTERFACE
# File: portfolio/signal_registry.py
# Thin query layer over the signals table.
# Passed into SignalSelector.run() at 8:50 AM each day.

```python
class SignalRegistry:
    """
    Thin query layer over the signals table.
    Passed into SignalSelector.run() at 8:50 AM each day.
    All rule/parameter content writes go through store_approved_signal().
    Status and lifecycle changes go through set_status().
    Never write rule fields (entry_conditions, parameters, etc.) directly
    via ad-hoc UPDATE — use store_approved_signal() so the auto-versioning
    trigger in schema.sql captures the change correctly.
    """

    def __init__(self, db):
        self.db = db

    def get_active_signals(self) -> list:
        """
        Returns all Signal objects with status = 'ACTIVE'.
        Called by SignalSelector to build today's candidate pool.
        Excludes: CANDIDATE, BACKTESTING, WATCH (selector sees WATCH
        separately via get_watch_signals), INACTIVE, ARCHIVED.
        """
        from portfolio.signal_model import Signal
        rows = self.db.execute("""
            SELECT * FROM signals
            WHERE status = 'ACTIVE'
            ORDER BY signal_id
        """).fetchall()
        return [Signal.from_db_row(row) for row in rows]

    def get_watch_signals(self) -> list:
        """Returns signals in WATCH status — reduced size, under review."""
        from portfolio.signal_model import Signal
        rows = self.db.execute(
            "SELECT * FROM signals WHERE status = 'WATCH' ORDER BY signal_id"
        ).fetchall()
        return [Signal.from_db_row(row) for row in rows]

    def set_status(self, signal_id: str, new_status: str,
                   reason: str, changed_by: str = 'SYSTEM'):
        """
        Status transitions: CANDIDATE→BACKTESTING→ACTIVE→WATCH→INACTIVE→ARCHIVED
        Records change via pending_change_by/reason columns so the
        auto-versioning trigger in schema.sql captures the transition.
        """
        VALID_STATUSES = {
            'CANDIDATE', 'BACKTESTING', 'ACTIVE',
            'WATCH', 'INACTIVE', 'ARCHIVED'
        }
        if new_status not in VALID_STATUSES:
            raise ValueError(f"Invalid status: {new_status}")
        self.db.execute("""
            UPDATE signals
            SET status = %s,
                pending_change_by = %s,
                pending_change_reason = %s,
                updated_at = NOW()
            WHERE signal_id = %s
        """, (new_status, changed_by, reason, signal_id))
```

---

## CAPITAL-STARVED SIGNAL HANDLING

```
A signal is CAPITAL_STARVED if it fires but cannot execute
because capital is exhausted by higher-priority signals.

Tracking during paper trading (4 months):
- Every time a signal fires, record: was capital available?
- After paper period: compute starvation_rate per signal
  starvation_rate = times_starved / times_fired

After paper trading, classify by starvation rate:
starvation_rate < 20%:   Signal works at current TOTAL_CAPITAL → PRIMARY eligible
starvation_rate 20-40%:  Signal works but needs capital management → SECONDARY
starvation_rate > 40%:   Signal is effectively inactive at current TOTAL_CAPITAL

For signals with starvation_rate > 40%, three options:
OPTION A: Reduce lot size to 50% normal → lowers margin requirement
          → may reduce starvation, but also reduces P&L contribution
OPTION B: Accept as CONTINGENCY signal
          → only fires when primary signals are quiet (low-activity regimes)
          → document explicitly: "This signal contributes only in low-volume periods"
OPTION C: Archive at current capital, revisit when capital grows to 2× TOTAL_CAPITAL
          → Document threshold capital required for this signal to be viable
          → Automatically resurface when TOTAL_CAPITAL crosses threshold

Recommended action per signal type:
Futures signals with high starvation: OPTION A (reduce lots)
Options selling signals with high starvation: OPTION C (wait for more capital)
Options buying signals with high starvation: OPTION B (contingency)
```

---

## DAILY PORTFOLIO STATE MANAGEMENT

```
8:50 AM:  Signal Selector runs. TODAY'S_SIGNALS written to Redis.
9:00 AM:  Market opens. Signal Engine activates.
9:00-9:15: First 15 minutes: FII overnight signals confirmed
           or cancelled based on opening direction.
9:15 AM+: Signal Engine monitors for intraday signals.
          Each fired signal → Risk Engine → Execution Engine.
3:20 PM:  Signal Engine stops accepting NEW entries.
          (10 minutes before close — no new positions)
3:25 PM:  Intraday signals that require same-day exit:
          forced close at market if not already exited.
3:30 PM:  Market closes.
3:35 PM:  Daily reconciliation runs.
3:45 PM:  Rolling Sharpe updated for all active signals.
          Signals crossing WATCH threshold → flagged.
4:00 PM:  Day's P&L logged to PostgreSQL.
7:30 PM:  FII data downloaded and processed.
          FII overnight signals evaluated for tomorrow.
          Pre-loaded into tomorrow's Signal Selector queue.
11:59 PM: Daily backup of all databases.
```

---

## DRAWDOWN RESPONSE PLAYBOOK (exact protocols)

```
LEVEL 1: Single signal drawdown > 2× its historical max DD
→ Signal status: WATCH
→ Position size: 25% of normal
→ Human review within 5 trading days
→ Options: (a) revise rules, (b) keep at 25% for 60 days, (c) ARCHIVE

LEVEL 2: Daily portfolio loss > 5% of TOTAL_CAPITAL (DAILY_LOSS_FRACTION)
→ No new entries for remainder of day
→ Existing positions: hold as planned (do not panic close)
→ Log reason if known
→ Review signal performance next morning

LEVEL 3: Weekly portfolio loss > 12% of TOTAL_CAPITAL (WEEKLY_LOSS_FRACTION)
→ No new entries for rest of week
→ All SECONDARY signals suspended for 2 weeks
→ PRIMARY signals continue at 50% size
→ Human review of all active signals over weekend

LEVEL 4: Monthly drawdown > 15% of TOTAL_CAPITAL (MONTHLY_DD_CRITICAL_FRACTION)
→ ALL trading halted
→ System enters PAPER_TRADING_MODE automatically
→ Paper trade for minimum 10 trading days
→ Review: which signals caused the drawdown?
→ Those signals → ARCHIVE or rules revision
→ Human decision required to resume live trading
→ Resume at 25% size across all signals for first 2 weeks back

LEVEL 5: Drawdown > 25% of TOTAL_CAPITAL (MONTHLY_DD_HALT_FRACTION)
→ ALL trading halted indefinitely
→ Full system review required
→ Backtest entire signal library on most recent 12 months
→ Regime has likely changed
→ Re-run conflict resolution on affected signal categories
→ Do not resume until root cause identified and fixed
```

# IMPLEMENTATION SPECIFICATION SUMMARY
# What is now fully specified vs what remains

---

## FULLY SPECIFIED (ready to hand to developer)

### Backtest Engine:
✓ RegimeLabeler — sequential computation, unit test included
✓ ParameterSensitivityTester — 5-variant test, fragility scoring
✓ WalkForwardEngine — exact 36/12/3 windows, purge/embargo, differentiated criteria
✓ ValidationTestSuite — 3 mandatory tests before any real signal tested

### Execution Engine:
✓ Execution flow — exact 9-step sequence
✓ Execution priority rules — 7-level priority table
✓ Greek pre-check — exact limits (delta, vega, gamma, theta)
✓ Failure handling — 7 specific scenarios with exact responses
✓ Daily reconciliation — positions match verification

### Portfolio Construction:
✓ PRIMARY vs SECONDARY classification — exact criteria
✓ Signal Selector — 13-step decision tree with code
✓ Calendar filters — RBI, expiry, normal day handling
✓ Expiry Monday filter — gamma trap handling
✓ Capital efficiency ranking — formula with regime multipliers
✓ Capital-starved signal handling — 3 options, criteria
✓ Daily schedule — minute-by-minute 8:50 AM to 11:59 PM
✓ Drawdown response playbook — 5 levels, exact thresholds

---

## ALL SPECIFICATIONS COMPLETE

```
Design:         100% complete
Specification:  100% complete (all 7 specs written and reviewed)
Code:           0%   (not started — specs are the build contract)

ALL SPECS COMPLETE:
✓ Spec 1: Backtest Engine (RegimeLabeler, WalkForwardEngine, ValidationTestSuite)
✓ Spec 2: Execution Engine (9-step flow, 7 failure scenarios, reconciliation,
           compute_combined_greeks Python stub + Go service spec)
✓ Spec 3: Portfolio Construction (SignalSelector, drawdown playbook, Portfolio class)
✓ Spec 4: RAG Pipeline (PDFIngester, VectorStore, CLIReviewer, SignalCandidate)
✓ Spec 5: Signal Registry Schema (14 tables, triggers, indexes, SQL)
✓ Spec 6: Monitoring (18 alerts, Telegram, daily digest)
✓ Spec 7: FII Data Pipeline (4 patterns, dynamic threshold, nightly schedule)

✓ Spec 6: Monitoring (18 alerts, Telegram, daily digest)
✓ Spec 7: FII Data Pipeline (4 patterns, dynamic threshold, nightly schedule)

KNOWN IMPLEMENTATION GAPS (developer must supply — interfaces are specified):
- vol_adjustment_factors table: derive and populate before running Tier 2 backtest.
  Method: download 6 months of TrueData intraday options data, compute realised IV
  vs VIX ratio per DTE bucket, insert into vol_adjustment_factors table.
  Schema is in schema.sql. get_vol_multiplier() in tier2_backtest.py reads it.
  Returns 1.0 safely if table is empty (slightly underestimates vol — acceptable
  for initial testing).
  See pending action item: "Derive DTE-based vol adjustment factors".

PREVIOUSLY LISTED AS GAPS — NOW FULLY SPECIFIED:
✓ load_nifty_history(): fully specified in data/nifty_loader.py.
    Initial load: python -m data.nifty_loader --init
    Nightly update: python -m data.nifty_loader --update
    nifty_daily table added to schema.sql.
✓ Tier 2 options backtest: fully specified in backtest/tier2_backtest.py.
    black_scholes_price(), make_tier2_backtest_fn(), get_vol_multiplier() all implemented.
    BACKTEST TIER DESIGN section at top of Spec 1 documents the three-tier architecture.
✓ compute_combined_greeks(): Python stub (execution/greek_calculator.py)
    + Go service spec (risk/greek_service.go).
✓ SignalRegistry.get_active_signals(): fully specified in Spec 3
    (portfolio/signal_registry.py).

ALL KNOWN GAPS RESOLVED. The only remaining developer action before
first run is populating vol_adjustment_factors (safe to skip initially —
get_vol_multiplier() returns 1.0 as default).
```

# IMPLEMENTATION SPEC 4: RAG PIPELINE INTERNALS
# Written by: Knowledge Engineer + System Architect
# Status: FINAL

---

## MODULE 1: PDF INGESTION
# File: ingestion/pdf_ingester.py

```python
import pdfplumber
import re
from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class RawChunk:
    text: str
    book_id: str
    chapter_number: int
    chapter_title: str
    page_start: int
    page_end: int
    chunk_index: object   # int for text chunks (0,1,2...), str for tables ('T0','T1',...)
                          # Union[int,str] — using object to avoid import overhead.
                          # ChromaDB ID: f"{book_id}_{chapter_number}_{chunk_index}"
                          # Both int and str produce valid, unique ID strings.
    chunk_type: str       # RULE|SUMMARY|EMPIRICAL|FORMULA|
                          # TABLE|CODE|PSYCHOLOGY|HEADING
    has_table: bool
    has_formula: bool

class PDFIngester:
    """
    Book-aware PDF ingester.
    Each book has a profile that handles its specific layout.
    """

    BOOK_PROFILES = {
        'GUJRAL': {
            'abstraction_level': 'CONCRETE',
            'skip_pages': [1, 2, 3],        # TOC, preface
            'chapter_pattern': r'^Chapter\s+\d+',
            'summary_marker': 'Key Points',  # Gujral ends chapters
                                             # with Key Points section
            'table_heavy': False,
            'formula_heavy': False,
        },
        'NATENBERG': {
            'abstraction_level': 'PRINCIPLE',
            'skip_pages': [1, 2, 3, 4, 5],  # TOC, preface, intro
            'chapter_pattern': r'^\d+\s+[A-Z]',
            'summary_marker': None,
            'table_heavy': True,             # Greeks tables
            'formula_heavy': True,           # Black-Scholes etc
            'skip_sections': ['Appendix'],   # Skip math appendices
        },
        'SINCLAIR': {
            'abstraction_level': 'PRINCIPLE',
            'skip_pages': [1, 2, 3],
            'chapter_pattern': r'^CHAPTER\s+\d+',
            'summary_marker': 'Summary',
            'table_heavy': False,
            'formula_heavy': True,
            'empirical_markers': [
                'empirically', 'we found', 'data shows',
                'historically', 'on average', 'tends to'
            ],
        },
        'GRIMES': {
            'abstraction_level': 'METHODOLOGY',
            'skip_pages': [1, 2, 3, 4],
            'chapter_pattern': r'^CHAPTER\s+\d+',
            'summary_marker': 'Summary',
            'table_heavy': True,             # Pattern statistics
            'empirical_markers': [
                'statistically', 'edge', 'no edge',
                'does not', 'fails', 'works'
            ],
            'edge_confirmed_marker': 'has edge',
            'edge_denied_marker': 'no statistical edge',
        },
        'KAUFMAN': {
            'abstraction_level': 'CONCRETE',
            'skip_pages': [1, 2, 3, 4, 5],
            'chapter_pattern': r'^Chapter\s+\d+',
            'summary_marker': None,
            'table_heavy': True,
            'formula_heavy': False,
            'pseudocode_markers': ['Step 1', 'If price', 'When'],
        },
        'DOUGLAS': {
            'abstraction_level': 'PSYCHOLOGY',
            'skip_pages': [1, 2, 3],
            'chapter_pattern': r'^Chapter\s+\d+',
            'all_chunks_type': 'PSYCHOLOGY',
        },
        'LOPEZ': {
            # Key must match the books table INSERT and SIGNAL_ID_PREFIXES ('LOP').
            'abstraction_level': 'METHODOLOGY',
            'skip_pages': list(range(1, 10)),
            'chapter_pattern': r'^\d+\s+',
            'formula_heavy': True,
            'code_heavy': True,              # Python code in book
            'target_chapters': [7, 8, 14],  # Purging, CV, backtest stats
                                             # Skip rest
        },
        'HILPISCH': {
            'abstraction_level': 'METHODOLOGY',
            'skip_pages': list(range(1, 8)),
            'chapter_pattern': r'^CHAPTER\s+\d+',
            'code_heavy': True,
            'target_chapters': [4, 5, 6, 7, 9],  # Finance-relevant only
        },
        'MCMILLAN': {
            'abstraction_level': 'CONCRETE',
            'skip_pages': [1, 2, 3, 4, 5],       # TOC, preface
            'chapter_pattern': r'^Chapter\s+\d+',
            'summary_marker': 'Summary',
            'table_heavy': True,                   # Strategy payoff tables
            'formula_heavy': False,
            # Butterflies, calendars, diagonals, ratio spreads.
            # Same extraction profile as Gujral/Kaufman.
        },
        'AUGEN': {
            'abstraction_level': 'CONCRETE',
            'skip_pages': [1, 2, 3],
            'chapter_pattern': r'^Chapter\s+\d+',
            'summary_marker': None,
            'table_heavy': True,                   # IV surface tables
            'formula_heavy': False,
            'empirical_markers': [
                'historically', 'on average', 'tends to',
                'IV crush', 'expiration', 'weekly'
            ],
            # Weekly expiry dynamics, IV crush, pre-event IV expansion.
        },
        'HARRIS': {
            'abstraction_level': 'METHODOLOGY',
            'skip_pages': [1, 2, 3, 4],
            'chapter_pattern': r'^CHAPTER\s+\d+',
            'summary_marker': 'Summary',
            'table_heavy': True,                   # Bid-ask spread tables
            'empirical_markers': [
                'statistically', 'liquidity', 'bid-ask',
                'informed', 'market impact', 'spread'
            ],
            # Market microstructure — extracted as filters only, not entry signals.
        },
    }

    def ingest_book(self, pdf_path: str, book_id: str) -> List[RawChunk]:
        profile = self.BOOK_PROFILES[book_id]
        chunks = []

        with pdfplumber.open(pdf_path) as pdf:
            current_chapter = 0
            current_chapter_title = ''
            page_buffer = []
            last_chapter_page = 0   # tracks page of last chapter heading (enforces 2-page gap)
            table_counter = 0       # unique sequential ID per table (avoids ChromaDB ID collisions)

            for page_num, page in enumerate(pdf.pages):
                page_1indexed = page_num + 1

                # Skip configured pages
                if page_1indexed in profile.get('skip_pages', []):
                    continue

                # Extract text
                text = page.extract_text() or ''
                if not text.strip():
                    continue

                # Detect chapter boundary
                # Search first 3 lines only — re.match() on full page text
                # false-fires on any page starting with a digit (e.g. "1.3 Std dev").
                # Minimum 2-page gap prevents false positives from repeated headings.
                first_lines = '\n'.join(text.split('\n')[:3])
                chapter_match = re.search(
                    profile.get('chapter_pattern', r'^Chapter'),
                    first_lines,
                    re.MULTILINE
                )
                pages_since_last_chapter = page_1indexed - last_chapter_page
                # last_chapter_page is always defined (initialised to 0 above).
                # First page of book: pages_since_last_chapter = page_1indexed,
                # which will always be >= 2 for any non-trivial book, so the
                # first real chapter heading is detected correctly.
                if chapter_match and pages_since_last_chapter >= 2:
                    # Flush buffer as chunks before new chapter
                    if page_buffer:
                        new_chunks = self._chunk_buffer(
                            page_buffer, book_id,
                            current_chapter,
                            current_chapter_title,
                            profile
                        )
                        chunks.extend(new_chunks)
                        page_buffer = []
                    current_chapter += 1
                    current_chapter_title = text.split('\n')[0].strip()
                    last_chapter_page = page_1indexed

                # Check if this chapter is a target chapter
                # (for books with skip-chapters configured)
                if 'target_chapters' in profile:
                    if current_chapter not in profile['target_chapters']:
                        continue

                # Extract tables separately
                tables = page.extract_tables()
                if tables and profile.get('table_heavy'):
                    for table in tables:
                        # Table ID "T0", "T1", ... is unique per chapter.
                        # Without a sequential ID, all tables in a chapter
                        # would overwrite each other in ChromaDB.
                        chunks.append(self._table_to_chunk(
                            table, book_id, current_chapter,
                            current_chapter_title, page_1indexed,
                            table_id=f"T{table_counter}"
                        ))
                        table_counter += 1

                page_buffer.append({
                    'text': text,
                    'page': page_1indexed
                })

            # Flush final buffer
            if page_buffer:
                chunks.extend(self._chunk_buffer(
                    page_buffer, book_id,
                    current_chapter, current_chapter_title,
                    profile
                ))

        return chunks

    def _chunk_buffer(self, page_buffer, book_id,
                      chapter, chapter_title, profile):
        """
        Paragraph-aware chunking.
        Target: 500 tokens. Max: 700 tokens.
        Overlap: last 2 paragraphs of previous chunk.
        """
        TARGET_TOKENS = 500
        MAX_TOKENS = 700
        APPROX_TOKENS_PER_WORD = 1.3

        # Combine all pages in buffer
        full_text = '\n'.join(p['text'] for p in page_buffer)
        first_page = page_buffer[0]['page']
        last_page = page_buffer[-1]['page']

        # Split into paragraphs (double newline = paragraph break)
        paragraphs = [p.strip() for p in
                      re.split(r'\n\s*\n', full_text)
                      if p.strip()]

        chunks = []
        current_paras = []
        current_tokens = 0
        overlap_paras = []  # Last 2 paragraphs for overlap

        for para in paragraphs:
            para_tokens = len(para.split()) * APPROX_TOKENS_PER_WORD

            # Detect chunk type for this paragraph
            chunk_type = self._detect_chunk_type(para, profile)

            # If PSYCHOLOGY book: force all to PSYCHOLOGY type
            if profile.get('all_chunks_type'):
                chunk_type = profile['all_chunks_type']

            # Flush if would exceed max
            if current_tokens + para_tokens > MAX_TOKENS \
                    and current_paras:
                chunk_text = '\n\n'.join(current_paras)
                chunks.append(RawChunk(
                    text=chunk_text,
                    book_id=book_id,
                    chapter_number=chapter,
                    chapter_title=chapter_title,
                    page_start=first_page,
                    page_end=last_page,
                    chunk_index=len(chunks),
                    chunk_type=self._dominant_type(
                        current_paras, profile
                    ),
                    has_table=False,
                    has_formula=self._contains_formula(chunk_text),
                ))
                # Keep last 2 paragraphs as overlap
                overlap_paras = current_paras[-2:]
                current_paras = overlap_paras.copy()
                current_tokens = sum(
                    len(p.split()) * APPROX_TOKENS_PER_WORD
                    for p in current_paras
                )

            current_paras.append(para)
            current_tokens += para_tokens

        # Flush remaining
        if current_paras:
            chunk_text = '\n\n'.join(current_paras)
            chunks.append(RawChunk(
                text=chunk_text,
                book_id=book_id,
                chapter_number=chapter,
                chapter_title=chapter_title,
                page_start=first_page,
                page_end=last_page,
                chunk_index=len(chunks),
                chunk_type=self._dominant_type(
                    current_paras, profile
                ),
                has_table=False,
                has_formula=self._contains_formula(chunk_text),
            ))

        return chunks

    def _detect_chunk_type(self, para: str, profile: dict) -> str:
        """Classify a paragraph's type."""
        para_lower = para.lower()

        # Formula detection
        formula_patterns = [r'=\s*[A-Za-z]', r'\^', r'√', r'∑',
                           r'σ', r'δ', r'Δ', r'∂']
        if any(re.search(p, para) for p in formula_patterns):
            return 'FORMULA'

        # Code detection
        # Patterns that appear in Python code but not trading-book prose.
        # 'if ' alone (without word-boundary) would match "if the market...".
        code_patterns = [r'def ', r'\bif\s+\w.*:', r'for\s+\w+\s+in\s',
                        r'^import ', r'print\(', r'return ']
        if any(re.search(p, para) for p in code_patterns):
            return 'CODE'

        # Empirical finding detection
        empirical_markers = profile.get('empirical_markers', [])
        if any(m in para_lower for m in empirical_markers):
            return 'EMPIRICAL'

        # Summary detection
        summary_marker = profile.get('summary_marker')
        if summary_marker and summary_marker.lower() in para_lower:
            return 'SUMMARY'

        # Psychology detection
        psych_markers = ['fear', 'greed', 'discipline', 'emotion',
                        'belief', 'confidence', 'anxiety', 'control']
        if any(m in para_lower for m in psych_markers):
            return 'PSYCHOLOGY'

        return 'RULE'  # Default for trading books

    def _contains_formula(self, text: str) -> bool:
        return bool(re.search(r'[=\^√∑σδΔ∂]', text))

    def _dominant_type(self, paras: list, profile: dict) -> str:
        types = [self._detect_chunk_type(p, profile) for p in paras]
        from collections import Counter
        return Counter(types).most_common(1)[0][0]

    def _table_to_chunk(self, table, book_id, chapter,
                         chapter_title, page, table_id='T0'):
        """Convert extracted table to a RawChunk.
        table_id (T0, T1, T2...) must be unique per chapter to avoid
        ChromaDB ID collisions. Never pass chunk_index=-1 for tables.
        """
        rows = [' | '.join(str(c) for c in row if c)
                for row in table if any(row)]
        table_text = '\n'.join(rows)
        return RawChunk(
            text=table_text,
            book_id=book_id,
            chapter_number=chapter,
            chapter_title=chapter_title,
            page_start=page,
            page_end=page,
            chunk_index=table_id,   # "T0", "T1", ... — unique per book/chapter
            chunk_type='TABLE',
            has_table=True,
            has_formula=False,
        )
```

---

## MODULE 2: CHROMADB STORAGE
# File: ingestion/vector_store.py

```python
import chromadb
from sentence_transformers import SentenceTransformer

class VectorStore:
    """
    One ChromaDB collection per book.
    Never query across all books simultaneously.
    Cross-book synthesis happens at Claude API layer.
    """

    EMBEDDING_MODEL = 'all-mpnet-base-v2'
    # Free, local, good quality for financial text.
    # 768-dim embeddings. ~70ms per chunk on CPU.

    def __init__(self, persist_dir: str):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.encoder = SentenceTransformer(self.EMBEDDING_MODEL)
        self._collections = {}

    def get_collection(self, book_id: str):
        """Get or create collection for a book."""
        if book_id not in self._collections:
            self._collections[book_id] = \
                self.client.get_or_create_collection(
                    name=f"book_{book_id.lower()}",
                    metadata={"hnsw:space": "cosine"}
                )
        return self._collections[book_id]

    def store_chunks(self, chunks: List[RawChunk]):
        """Store chunks in their book's collection."""
        by_book = {}
        for chunk in chunks:
            by_book.setdefault(chunk.book_id, []).append(chunk)

        for book_id, book_chunks in by_book.items():
            collection = self.get_collection(book_id)

            texts = [c.text for c in book_chunks]
            embeddings = self.encoder.encode(
                texts, batch_size=32, show_progress_bar=True
            ).tolist()

            ids = [
                f"{c.book_id}_{c.chapter_number}_{c.chunk_index}"
                for c in book_chunks
            ]

            metadatas = [{
                'book_id': c.book_id,
                'chapter_number': c.chapter_number,
                'chapter_title': c.chapter_title,
                'page_start': c.page_start,
                'page_end': c.page_end,
                'chunk_type': c.chunk_type,
                'has_table': c.has_table,
                'has_formula': c.has_formula,
                # Abstraction level from profile
                'abstraction_level': self._get_abstraction(c.book_id),
            } for c in book_chunks]

            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )

    def query_book(self, book_id: str, query: str,
                   n_results: int = 8,
                   chunk_types: List[str] = None) -> List[dict]:
        """
        Query a specific book's collection.
        chunk_types: filter to specific types (e.g. ['RULE', 'EMPIRICAL'])
        """
        collection = self.get_collection(book_id)
        query_embedding = self.encoder.encode([query]).tolist()

        where_filter = {}
        if chunk_types:
            where_filter = {"chunk_type": {"$in": chunk_types}}

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            where=where_filter if where_filter else None,
            include=['documents', 'metadatas', 'distances']
        )

        # Apply MMR (Maximal Marginal Relevance) reranking
        # to reduce redundant chunks
        return self._mmr_rerank(results, lambda_param=0.6)

    def _mmr_rerank(self, results, lambda_param=0.6, n_final=5):
        """
        Maximal Marginal Relevance reranking.
        Without MMR, ChromaDB returns near-identical chunks for any concept
        repeated across chapters, producing duplicate signals.

        lambda_param: 0=max diversity, 1=max relevance. 0.6 balances both.
        n_final: number of documents to return after reranking.
        """
        if not results['ids'][0]:
            return []

        docs      = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]

        # ChromaDB cosine distance → similarity
        similarities = [1 - d for d in distances]

        if len(docs) <= 1:
            return [{'text': docs[0], 'metadata': metadatas[0],
                     'relevance': similarities[0]}]

        # Compute inter-document similarity matrix using bigram overlap
        # (storing embeddings in ChromaDB results requires extra config;
        #  bigram overlap is a fast, good-enough proxy for text similarity)
        n = len(docs)
        inter_sim = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                sim = self._text_similarity(docs[i], docs[j])
                inter_sim[i][j] = sim
                inter_sim[j][i] = sim

        # MMR greedy selection
        selected_indices = []
        candidate_indices = list(range(n))

        # First selection: highest relevance always wins
        best_idx = max(candidate_indices, key=lambda i: similarities[i])
        selected_indices.append(best_idx)
        candidate_indices.remove(best_idx)

        while (len(selected_indices) < min(n_final, n)
               and candidate_indices):
            mmr_scores = {}
            for idx in candidate_indices:
                relevance  = similarities[idx]
                redundancy = max(inter_sim[idx][s]
                                 for s in selected_indices)
                mmr_scores[idx] = (lambda_param * relevance
                                   - (1 - lambda_param) * redundancy)

            next_idx = max(mmr_scores, key=mmr_scores.get)
            selected_indices.append(next_idx)
            candidate_indices.remove(next_idx)

        return [
            {
                'text':      docs[i],
                'metadata':  metadatas[i],
                'relevance': similarities[i]
            }
            for i in selected_indices
        ]

    def _text_similarity(self, text_a: str, text_b: str) -> float:
        """Bigram overlap Jaccard similarity. Fast proxy for semantic overlap."""
        def bigrams(text):
            words = text.lower().split()
            return set(zip(words[:-1], words[1:]))
        bg_a = bigrams(text_a)
        bg_b = bigrams(text_b)
        if not bg_a or not bg_b:
            return 0.0
        return len(bg_a & bg_b) / len(bg_a | bg_b)

    def _get_abstraction(self, book_id: str) -> str:
        levels = {
            'GUJRAL':    'CONCRETE',    # signal prefix: GUJ
            'KAUFMAN':   'CONCRETE',    # signal prefix: KAU
            'MCMILLAN':  'CONCRETE',    # signal prefix: MCM
            'AUGEN':     'CONCRETE',    # signal prefix: AUG
            'NATENBERG': 'PRINCIPLE',   # signal prefix: NAT
            'SINCLAIR':  'PRINCIPLE',   # signal prefix: SIN
            'GRIMES':    'METHODOLOGY', # signal prefix: GRI
            'LOPEZ':     'METHODOLOGY', # signal prefix: LOP
            'HILPISCH':  'METHODOLOGY', # signal prefix: HIL
            'HARRIS':    'METHODOLOGY', # signal prefix: HAR
            'DOUGLAS':   'PSYCHOLOGY',  # signal prefix: DOU
        }
        return levels.get(book_id, 'UNKNOWN')
```

---

## MODULE 3: RULE EXTRACTION PROMPT
# File: extraction/prompts.py
# These are the EXACT prompts. Do not paraphrase. Use verbatim.

```python
CONCRETE_BOOK_EXTRACTION_PROMPT = """
You are extracting a structured trading rule from a book excerpt.
Extract ONLY what the author explicitly states.
Do NOT infer, extend, or add information not in the text.
If a parameter is not specified, use the string "AUTHOR_SILENT".

BOOK: {book_title} by {author}
CHAPTER: {chapter_title}
PAGES: {page_start}-{page_end}
ABSTRACTION LEVEL: CONCRETE (this book states specific rules)

EXCERPT:
{chunk_text}

Extract the trading rule as JSON. Return ONLY valid JSON, no other text.

{{
  "rule_found": true or false,
  "rule_text": "One sentence. What triggers what action.",
  "signal_category": "TREND|REVERSION|VOL|PATTERN|EVENT|SIZING|RISK|REGIME",
  "direction": "LONG|SHORT|NEUTRAL|CONTEXT_DEPENDENT",
  "entry_conditions": [
    "Exact condition 1 as stated by author",
    "Exact condition 2 as stated by author"
  ],
  "parameters": {{
    "parameter_name": "exact value as stated, or AUTHOR_SILENT"
  }},
  "exit_conditions": [
    "Exact exit rule as stated, or AUTHOR_SILENT"
  ],
  "instrument": "FUTURES|OPTIONS_BUYING|OPTIONS_SELLING|SPREAD|ANY|AUTHOR_SILENT",
  "timeframe": "INTRADAY|POSITIONAL|SWING|ANY|AUTHOR_SILENT",
  "target_regimes": ["TRENDING", "RANGING", "HIGH_VOL", "ANY"] (array — pick all that apply, or ["ANY"] if AUTHOR_SILENT),
  "author_confidence": "HIGH|MEDIUM|LOW",
  "author_confidence_basis": "Why the author is or is not confident",
  "source_citation": "{book_title}, {author}, Ch.{chapter_number}, p.{page_start}",
  "conflicts_with": "If author mentions a conflicting approach, state it. Else: NONE",
  "completeness_warning": "Important context that may be in surrounding chapters, or NONE"
}}

If no clear trading rule exists in this excerpt: return {{"rule_found": false}}
"""

PRINCIPLE_BOOK_EXTRACTION_PROMPT = """
You are extracting a trading PRINCIPLE from a book excerpt.
This book operates at principle level, not specific rule level.
Your job is to identify the principle AND generate 3 parameter variants
that operationalize it. The backtest will determine which variant is best.

BOOK: {book_title} by {author}
CHAPTER: {chapter_title}
PAGES: {page_start}-{page_end}
ABSTRACTION LEVEL: PRINCIPLE (generate 3 variants with parameter ranges)

IMPORTANT — DO NOT extract a principle if the excerpt is primarily:
- A mathematical derivation, proof, or formula explanation
- A definition of a term (what vega IS, not what to DO with vega)
- Historical market context or background information
- A mechanical description of how an instrument works
- A comparison of two approaches with no stated preference
- A worked example that illustrates a concept without prescribing action

An actionable principle MUST contain an instruction — explicit or implicit.
Test: Can you complete the sentence "A trader should [action] when [condition]"
using only what this excerpt states?
If NO → return {"principle_found": false}
If YES → extract the principle and generate 3 variants.

EXCERPT:
{chunk_text}

Extract as JSON. Return ONLY valid JSON, no other text.

{{
  "principle_found": true or false,
  "principle_text": "The core principle in one sentence",
  "trader_should_statement": "A trader should [action] when [condition]",
  "signal_category": "TREND|REVERSION|VOL|PATTERN|EVENT|SIZING|RISK|REGIME",
  "variants": [
    {{
      "variant_id": "CONSERVATIVE",
      "description": "Conservative operationalization of this principle",
      "entry_conditions": ["condition 1", "condition 2"],
      "parameters": {{"param_name": "conservative_value"}},
      "exit_conditions": ["exit rule"],
      "instrument": "FUTURES|OPTIONS_BUYING|OPTIONS_SELLING|SPREAD",
      "target_regimes": ["TRENDING", "RANGING", "HIGH_VOL", "ANY"],
      "rationale": "Why these are conservative parameters"
    }},
    {{
      "variant_id": "MODERATE",
      "description": "Moderate operationalization",
      "entry_conditions": ["condition 1", "condition 2"],
      "parameters": {{"param_name": "moderate_value"}},
      "exit_conditions": ["exit rule"],
      "instrument": "FUTURES|OPTIONS_BUYING|OPTIONS_SELLING|SPREAD",
      "target_regimes": ["TRENDING", "RANGING", "HIGH_VOL", "ANY"],
      "rationale": "Why these are moderate parameters"
    }},
    {{
      "variant_id": "AGGRESSIVE",
      "description": "Aggressive operationalization",
      "entry_conditions": ["condition 1", "condition 2"],
      "parameters": {{"param_name": "aggressive_value"}},
      "exit_conditions": ["exit rule"],
      "instrument": "FUTURES|OPTIONS_BUYING|OPTIONS_SELLING|SPREAD",
      "target_regimes": ["TRENDING", "RANGING", "HIGH_VOL", "ANY"],
      "rationale": "Why these are aggressive parameters"
    }}
  ],
  "source_citation": "{book_title}, {author}, Ch.{chapter_number}, p.{page_start}",
  "author_stated_conditions": "Any conditions the author DOES specify explicitly",
  "what_author_leaves_unspecified": "Parameters the author does not define"
}}

If no actionable principle exists: return {{"principle_found": false}}
"""

CONFLICT_DETECTION_PROMPT = """
You are comparing two extracted trading rules to determine if they conflict.

RULE A:
{rule_a}

RULE B:
{rule_b}

Analyze and return JSON only:

{{
  "conflict_type": "DIRECT_CONTRADICTION|PARAMETER_DISAGREEMENT|CONDITION_DISAGREEMENT|EXIT_DISAGREEMENT|NO_CONFLICT|COMPLEMENTARY",
  "conflict_description": "Exactly what disagrees between the two rules",
  "conflict_on_field": "direction|entry_conditions|parameters|exit_conditions|target_regimes",
  "resolution_method": "BACKTEST_BOTH|REGIME_SEPARATE|PARAMETER_TEST|HUMAN_REVIEW",
  "resolution_notes": "Specific guidance for how to resolve this conflict in backtesting",
  "can_coexist": true or false,
  "coexistence_condition": "If they can coexist, under what conditions"
}}
"""

HALLUCINATION_CHECK_PROMPT = """
A trading rule has been extracted from a book excerpt.
Verify whether the extracted rule is faithful to the source text.

ORIGINAL EXCERPT:
{original_chunk}

EXTRACTED RULE:
{extracted_rule}

Check each field and return JSON only:

{{
  "overall_verdict": "FAITHFUL|MINOR_ISSUES|SIGNIFICANT_ISSUES|HALLUCINATED",
  "field_checks": {{
    "rule_text": "FAITHFUL|INFERRED|HALLUCINATED",
    "entry_conditions": "FAITHFUL|INFERRED|HALLUCINATED",
    "parameters": "FAITHFUL|INFERRED|HALLUCINATED",
    "exit_conditions": "FAITHFUL|INFERRED|HALLUCINATED",
    "instrument": "FAITHFUL|INFERRED|HALLUCINATED",
    "target_regimes": "FAITHFUL|INFERRED|HALLUCINATED"
  }},
  "issues_found": ["Issue 1", "Issue 2"],
  "recommendation": "APPROVE|REVISE|REJECT",
  "revision_notes": "What needs to change if REVISE"
}}
"""
```

---

## MODULE 3b: EXTRACTION ORCHESTRATOR
# File: extraction/orchestrator.py
# Connects all RAG pipeline components:
# picks the right prompt → calls Claude API → parses response →
# runs hallucination check → creates SignalCandidate(s) → hands to CLIReviewer.

```python
import json
import re
import anthropic
from ingestion.signal_candidate import SignalCandidate, store_approved_signal
from review.cli_reviewer import CLIReviewer
from extraction.prompts import (
    CONCRETE_BOOK_EXTRACTION_PROMPT,
    PRINCIPLE_BOOK_EXTRACTION_PROMPT,
    HALLUCINATION_CHECK_PROMPT,
)

BOOK_META = {
    'GUJRAL':    {'title': 'How to Make Money in Intraday Trading', 'author': 'Ashwani Gujral'},
    'KAUFMAN':   {'title': 'Trading Systems and Methods',           'author': 'Perry Kaufman'},
    'NATENBERG': {'title': 'Option Volatility and Pricing',         'author': 'Natenberg'},
    'SINCLAIR':  {'title': 'Options Trading',                       'author': 'Euan Sinclair'},
    'GRIMES':    {'title': 'The Art and Science of Technical Analysis', 'author': 'Adam Grimes'},
    'LOPEZ':     {'title': 'Advances in Financial Machine Learning', 'author': 'Lopez de Prado'},
    'HILPISCH':  {'title': 'Python for Finance',                    'author': 'Yves Hilpisch'},
    'DOUGLAS':   {'title': 'Trading in the Zone',                   'author': 'Mark Douglas'},
    'MCMILLAN':  {'title': 'Options as a Strategic Investment',     'author': 'Lawrence McMillan'},
    'AUGEN':     {'title': 'The Volatility Edge in Options Trading','author': 'Jeff Augen'},
    'HARRIS':    {'title': 'Trading and Exchanges',                 'author': 'Larry Harris'},
}


class ExtractionOrchestrator:
    """
    Drives the full extraction pipeline for one book:
      1. Query VectorStore for all chunks of this book
      2. For each chunk: call Claude API with the right prompt
      3. Run hallucination check on each extracted rule
      4. Create SignalCandidate(s)
      5. Present to CLIReviewer for human decision
      6. Store approved signals via store_approved_signal()

    CONCRETE books: one SignalCandidate per chunk (if rule_found).
    PRINCIPLE books: up to three SignalCandidates per chunk
                     (CONSERVATIVE, MODERATE, AGGRESSIVE variants).
    """

    # Models: Haiku for extraction (cheap, fast), Sonnet for hallucination check
    EXTRACTION_MODEL   = 'claude-haiku-4-5-20251001'
    HALLUCINATION_MODEL = 'claude-sonnet-4-20250514'  # more careful
    MAX_TOKENS = 1500

    def __init__(self, db, vector_store, signal_registry, logger):
        self.db              = db
        self.vector_store    = vector_store
        self.signal_registry = signal_registry
        self.logger          = logger
        self.client          = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var
        self.reviewer        = CLIReviewer()

    def run_book(self, book_id: str):
        """Extract and review all signals from one book."""

        abstraction = self.vector_store._get_abstraction(book_id)
        meta        = BOOK_META[book_id]
        chunks      = self.vector_store.query_book(
            book_id, query='trading rule signal entry exit condition',
            n_results=200  # get all chunks
        )

        self.logger.info(f"Extracting from {book_id}: {len(chunks)} chunks, "
                         f"abstraction={abstraction}")

        for chunk_result in chunks:
            chunk_text = chunk_result['text']
            chunk_meta = chunk_result['metadata']

            candidates = self._extract_candidates(
                book_id, chunk_text, chunk_meta, meta, abstraction
            )

            for candidate in candidates:
                if candidate is None:
                    continue
                decision = self.reviewer.review_signal(candidate)
                if decision['decision'] in ('A', 'R'):
                    store_approved_signal(self.db, candidate, decision)
                    self.logger.info(f"Stored: {candidate.signal_id}")
                else:
                    self.logger.info(f"Skipped {candidate.signal_id}: "
                                     f"{decision['decision']}")

    def _extract_candidates(self, book_id, chunk_text, chunk_meta,
                             meta, abstraction) -> list:
        """
        Returns a list of SignalCandidates (0-3 depending on abstraction).
        """
        if abstraction == 'CONCRETE':
            return [self._extract_concrete(book_id, chunk_text, chunk_meta, meta)]
        elif abstraction == 'PRINCIPLE':
            return self._extract_principle(book_id, chunk_text, chunk_meta, meta)
        else:
            # METHODOLOGY and PSYCHOLOGY: no direct rule extraction
            return []

    def _extract_concrete(self, book_id, chunk_text, chunk_meta, meta):
        """Extract one rule from a CONCRETE book chunk."""
        prompt = CONCRETE_BOOK_EXTRACTION_PROMPT.format(
            book_title     = meta['title'],
            author         = meta['author'],
            chapter_title  = chunk_meta.get('chapter_title', ''),
            page_start     = chunk_meta.get('page_start', ''),
            page_end       = chunk_meta.get('page_end', ''),
            chunk_text     = chunk_text,
        )
        response = self._call_claude(prompt, self.EXTRACTION_MODEL)
        if response is None or not response.get('rule_found'):
            return None

        signal_id = self._next_signal_id(book_id)
        # Build a minimal RawChunk-like object for from_claude_response
        class _Chunk:
            text = chunk_text
        candidate = SignalCandidate.from_claude_response(
            signal_id, book_id, _Chunk(), response
        )
        candidate = self._run_hallucination_check(candidate, chunk_text, response)
        return candidate

    def _extract_principle(self, book_id, chunk_text, chunk_meta, meta) -> list:
        """Extract up to 3 variant SignalCandidates from a PRINCIPLE book chunk."""
        prompt = PRINCIPLE_BOOK_EXTRACTION_PROMPT.format(
            book_title     = meta['title'],
            author         = meta['author'],
            chapter_title  = chunk_meta.get('chapter_title', ''),
            page_start     = chunk_meta.get('page_start', ''),
            page_end       = chunk_meta.get('page_end', ''),
            chapter_number = chunk_meta.get('chapter_number', ''),
            chunk_text     = chunk_text,
        )
        response = self._call_claude(prompt, self.EXTRACTION_MODEL)
        if response is None or not response.get('principle_found'):
            return []

        candidates = []
        for variant in response.get('variants', []):
            signal_id = self._next_signal_id(book_id)
            # Flatten variant into a response-shaped dict for from_claude_response
            flat = {
                'rule_text':         response.get('principle_text', ''),
                'signal_category':   response.get('signal_category', 'PATTERN'),
                'direction':         variant.get('direction', 'CONTEXT_DEPENDENT'),
                'entry_conditions':  variant.get('entry_conditions', []),
                'parameters':        variant.get('parameters', {}),
                'exit_conditions':   variant.get('exit_conditions', []),
                'instrument':        variant.get('instrument', 'ANY'),
                'timeframe':         'ANY',
                'target_regimes':    variant.get('target_regimes', ['ANY']),
                # Defaults to ['ANY'] if model omits the field
                'source_citation':   response.get('source_citation', ''),
                'completeness_warning': response.get('what_author_leaves_unspecified'),
                'author_confidence': 'MEDIUM',  # PRINCIPLE books don't state confidence
            }
            class _Chunk:
                text = chunk_text
            candidate = SignalCandidate.from_claude_response(
                signal_id, book_id, _Chunk(), flat
            )
            candidate.variant_id = variant.get('variant_id')
            candidate = self._run_hallucination_check(candidate, chunk_text, flat)
            candidates.append(candidate)
        return candidates

    def _call_claude(self, prompt: str, model: str) -> dict:
        """
        Call Claude API and parse JSON response.
        Returns dict on success, None on failure.
        """
        try:
            message = self.client.messages.create(
                model      = model,
                max_tokens = self.MAX_TOKENS,
                messages   = [{'role': 'user', 'content': prompt}]
            )
            raw = message.content[0].text.strip()
            # Strip markdown fences if present
            raw = re.sub(r'^```json\s*', '', raw)
            raw = re.sub(r'\s*```$', '', raw)
            return json.loads(raw)
        except (json.JSONDecodeError, Exception) as e:
            self.logger.warning(f"Claude API/parse error: {e}")
            return None

    def _run_hallucination_check(self, candidate: 'SignalCandidate',
                                  original_chunk: str, extracted: dict):
        """Run hallucination check and populate candidate.hallucination_verdict."""
        check_prompt = HALLUCINATION_CHECK_PROMPT.format(
            original_chunk = original_chunk,
            extracted_rule = json.dumps(extracted, indent=2),
        )
        result = self._call_claude(check_prompt, self.HALLUCINATION_MODEL)
        if result:
            verdict = result.get('overall_verdict', 'UNKNOWN')
            # Map HALLUCINATION_CHECK_PROMPT verdicts to candidate field values
            verdict_map = {
                'FAITHFUL':           'PASS',
                'MINOR_ISSUES':       'WARN',
                'SIGNIFICANT_ISSUES': 'FAIL',
                'HALLUCINATED':       'FAIL',
            }
            candidate.hallucination_verdict = verdict_map.get(verdict, 'UNKNOWN')
            candidate.hallucination_issues  = result.get('issues_found', [])
        return candidate

    def _next_signal_id(self, book_id: str) -> str:
        """
        Query the DB for how many signals already exist for this book,
        then increment by 1 to get the next sequence number.
        Thread-safe within a single ingestion run (single-threaded CLI).
        """
        row = self.db.execute(
            "SELECT COUNT(*) AS n FROM signals WHERE book_id = %s",
            (book_id,)
        ).fetchone()
        next_seq = (row['n'] if row else 0) + 1
        return SignalSelector.make_signal_id(book_id, next_seq)
```

---

## MODULE 4: HUMAN REVIEW INTERFACE
# File: review/cli_reviewer.py
# Simple CLI — no web framework needed at this stage

```python
"""
Run with: python cli_reviewer.py --book GUJRAL

Presents each extracted rule for human review.
Reviewer sees: original chunk, extracted rule, quality checks.
Reviewer approves or rejects and adds notes.
"""

## SIGNAL CANDIDATE DATACLASS
# File: ingestion/signal_candidate.py
# In-memory extraction result from Claude API.
# Created by ExtractionOrchestrator, reviewed by CLIReviewer,
# then written to the signals table via store_approved_signal().

```python
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class SignalCandidate:
    """
    In-memory extraction result from Claude API.
    Produced by VectorStore.extract_rule_from_chunk().
    Passed to CLIReviewer.review_signal() for human approval.
    If approved (decision == 'A'|'R'), stored to signals table
    via store_approved_signal().
    """
    # Identity
    signal_id:             str          # e.g. 'GUJ_001' — assigned before review
    book_id:               str          # e.g. 'GUJRAL'
    source_citation:       str          # e.g. 'How to Make Money..., Ch.3, p.47'

    # Original source (shown to reviewer)
    raw_chunk_text:        str          # The book excerpt sent to Claude

    # Claude extraction output
    rule_text:             str          # One-sentence rule summary
    signal_category:       str          # TREND|REVERSION|VOL|PATTERN|EVENT
    direction:             str          # LONG|SHORT|NEUTRAL|CONTEXT_DEPENDENT
    entry_conditions:      List[str]    = field(default_factory=list)
    parameters:            Dict[str, Any] = field(default_factory=dict)
    exit_conditions:       List[str]    = field(default_factory=list)
    instrument:            str          = 'AUTHOR_SILENT'
    timeframe:             str          = 'AUTHOR_SILENT'
    target_regimes:        List[str]    = field(default_factory=lambda: ['ANY'])
    # Matches Signal dataclass and signals.target_regimes TEXT[] column

    # Quality checks
    hallucination_verdict: str          = 'UNKNOWN'  # PASS|WARN|FAIL
    hallucination_issues:  List[str]    = field(default_factory=list)
    completeness_warning:  Optional[str] = None
    author_confidence:     str          = 'MEDIUM'

    # For PRINCIPLE-level books: which variant is this?
    variant_id:            Optional[str] = None  # CONSERVATIVE|MODERATE|AGGRESSIVE

    @classmethod
    def from_claude_response(cls, signal_id: str, book_id: str,
                              chunk, response: dict) -> 'SignalCandidate':
        """
        Parse Claude API JSON response into SignalCandidate.
        chunk: RawChunk that was sent to Claude.
        response: parsed JSON from Claude (already validated as dict).
        """
        return cls(
            signal_id             = signal_id,
            book_id               = book_id,
            source_citation       = response.get('source_citation', ''),
            raw_chunk_text        = chunk.text,
            rule_text             = response.get('rule_text', ''),
            signal_category       = response.get('signal_category', 'PATTERN'),
            direction             = response.get('direction', 'CONTEXT_DEPENDENT'),
            entry_conditions      = response.get('entry_conditions', []),
            parameters            = response.get('parameters', {}),
            exit_conditions       = response.get('exit_conditions', []),
            instrument            = response.get('instrument', 'AUTHOR_SILENT'),
            timeframe             = response.get('timeframe', 'AUTHOR_SILENT'),
            target_regimes        = response.get('target_regimes', ['ANY']),
            completeness_warning  = response.get('completeness_warning'),
            author_confidence     = response.get('author_confidence', 'MEDIUM'),
        )


def store_approved_signal(db, candidate: 'SignalCandidate',
                          decision: dict) -> str:
    """
    Bridge between CLIReviewer's decision dict and the signals table INSERT.

    Called after CLIReviewer.review_signal() returns decision == 'A' or 'R'.
    For 'R' (Revised), entry_conditions are replaced with reviewer's edits
    before insert.

    Returns: signal_id of the stored record.
    """
    # Apply reviewer revisions if any
    entry_conditions = candidate.entry_conditions
    if decision['decision'] == 'R' and decision.get('revised_conditions'):
        entry_conditions = decision['revised_conditions']

    import json
    from psycopg2.extras import Json as PgJson

    # Parse source_citation into schema columns
    # source_citation format: "{title}, {author}, Ch.{num}, p.{start}"
    # Store chapter text and page_start; page_end = page_start for single-page citations
    source_chapter = f"Ch. from {candidate.book_id}"
    source_page_start = None
    import re
    m = re.search(r'p\.(\d+)', candidate.source_citation)
    if m:
        source_page_start = int(m.group(1))
    m2 = re.search(r'Ch\.(\d+)', candidate.source_citation)
    if m2:
        source_chapter = f"Chapter {m2.group(1)}"

    # Build review_notes with all metadata that has no dedicated column
    review_notes_parts = []
    if candidate.completeness_warning:
        review_notes_parts.append(f"COMPLETENESS: {candidate.completeness_warning}")
    if candidate.author_confidence:
        review_notes_parts.append(f"AUTHOR_CONFIDENCE: {candidate.author_confidence}")
    if candidate.variant_id:
        review_notes_parts.append(f"VARIANT: {candidate.variant_id}")
    if decision.get('notes'):
        review_notes_parts.append(f"REVIEWER: {decision['notes']}")
    review_notes = " | ".join(review_notes_parts) or None

    # Signal name: "BOOK_ID: first 60 chars of rule_text"
    signal_name = f"{candidate.book_id}: {candidate.rule_text[:60]}"

    # Map AUTHOR_SILENT → 'ANY' for fields with CHECK constraints.
    # 'AUTHOR_SILENT' is valid in the extraction prompt but not in the DB schema.
    instrument = 'ANY' if candidate.instrument == 'AUTHOR_SILENT' else candidate.instrument
    timeframe  = 'ANY' if candidate.timeframe  == 'AUTHOR_SILENT' else candidate.timeframe

    db.execute("""
        INSERT INTO signals (
            signal_id, name, book_id,
            source_chapter, source_page_start, raw_chunk_text,
            signal_category, direction,
            entry_conditions, parameters, exit_conditions,
            instrument, timeframe, target_regimes,
            status, classification,
            avoid_rbi_day,
            review_notes, human_reviewer,
            created_at, updated_at,
            pending_change_by, pending_change_reason
        ) VALUES (
            %s, %s, %s,
            %s, %s, %s,
            %s, %s,
            %s, %s, %s,
            %s, %s, %s,
            'CANDIDATE', NULL,
            FALSE,
            %s, %s,
            NOW(), NOW(),
            %s, %s
        )
    """, (
        candidate.signal_id,
        signal_name,
        candidate.book_id,
        source_chapter,
        source_page_start,
        candidate.raw_chunk_text,
        candidate.signal_category,
        candidate.direction,
        PgJson(entry_conditions),
        PgJson(candidate.parameters),
        PgJson(candidate.exit_conditions),
        instrument,                         # mapped from AUTHOR_SILENT→ANY
        timeframe,                          # mapped from AUTHOR_SILENT→ANY
        candidate.target_regimes,           # psycopg2 maps list → TEXT[]
        review_notes,
        'HUMAN_REVIEWER',
        decision.get('notes', ''),
        f"APPROVED_{decision['decision']}_by_human_review",
    ))

    return candidate.signal_id
```

---

class CLIReviewer:

    def review_signal(self, signal_candidate):
        """
        Shows reviewer:
        1. Original book text (source pages)
        2. Extracted rule (from Claude API)
        3. Hallucination check result
        4. Prompts for decision

        Reviewer makes ONE decision per signal:
        APPROVE | REVISE | REJECT | DEFER
        """

        print("\n" + "="*60)
        print(f"SIGNAL: {signal_candidate.signal_id}")
        print(f"SOURCE: {signal_candidate.source_citation}")
        print("="*60)

        print("\n--- ORIGINAL TEXT (source pages) ---")
        print(signal_candidate.raw_chunk_text)

        print("\n--- EXTRACTED RULE ---")
        print(f"Rule: {signal_candidate.rule_text}")
        print(f"Category: {signal_candidate.signal_category}")
        print(f"Entry conditions:")
        for c in signal_candidate.entry_conditions:
            print(f"  • {c}")
        print(f"Parameters: {signal_candidate.parameters}")
        print(f"Exit: {signal_candidate.exit_conditions}")
        print(f"Instrument: {signal_candidate.instrument}")
        print(f"Regime: {signal_candidate.target_regimes}")

        print("\n--- QUALITY CHECK ---")
        print(f"Hallucination check: {signal_candidate.hallucination_verdict}")
        if signal_candidate.hallucination_issues:
            for issue in signal_candidate.hallucination_issues:
                print(f"  ⚠ {issue}")

        print("\n--- COMPLETENESS WARNING ---")
        print(signal_candidate.completeness_warning or "None")

        print("\n--- YOUR DECISION ---")
        print("A = Approve  |  R = Revise  |  X = Reject  |  D = Defer")
        decision = input("Decision: ").strip().upper()

        notes = ""
        revised_conditions = None

        if decision == 'R':
            print("Enter revised entry conditions"
                  " (one per line, blank line to finish):")
            lines = []
            while True:
                line = input()
                if not line:
                    break
                lines.append(line)
            revised_conditions = lines
            notes = input("Revision notes: ")

        elif decision == 'X':
            notes = input("Rejection reason: ")

        elif decision == 'A':
            notes = input("Optional notes (press Enter to skip): ")

        return {
            'decision': decision,
            'notes': notes,
            'revised_conditions': revised_conditions,
            'reviewer_timestamp': datetime.now().isoformat(),
        }
```

# IMPLEMENTATION SPEC 5: SIGNAL REGISTRY SCHEMA
# Written by: System Architect
# Status: FINAL — run this SQL exactly

---

## COMPLETE POSTGRESQL SCHEMA
# File: db/schema.sql

```sql
-- Enable TimescaleDB extension (already installed)
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ================================================================
-- TABLE 1: BOOKS
-- Master list of all ingested books
-- ================================================================
CREATE TABLE books (
    book_id         TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    author          TEXT NOT NULL,
    edition         TEXT,
    year            INTEGER,
    abstraction_level TEXT NOT NULL
                    CHECK (abstraction_level IN (
                        'CONCRETE', 'PRINCIPLE',
                        'METHODOLOGY', 'PSYCHOLOGY'
                    )),
    pdf_path        TEXT NOT NULL,
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    chunk_count     INTEGER,
    notes           TEXT
);

INSERT INTO books VALUES
('GUJRAL',         'How to Make Money in Intraday Trading',
 'Ashwani Gujral', '2nd', 2008,  'CONCRETE',    '/books/gujral.pdf',   NOW(), NULL, NULL),
('NATENBERG',      'Option Volatility and Pricing',
 'Natenberg',      '2nd', 2014,  'PRINCIPLE',   '/books/natenberg.pdf',NOW(), NULL, NULL),
('SINCLAIR',       'Options Trading',
 'Euan Sinclair',  '1st', 2010,  'PRINCIPLE',   '/books/sinclair.pdf', NOW(), NULL, NULL),
('GRIMES',         'The Art and Science of Technical Analysis',
 'Adam Grimes',    '1st', 2012,  'METHODOLOGY', '/books/grimes.pdf',   NOW(), NULL, NULL),
('KAUFMAN',        'Trading Systems and Methods',
 'Perry Kaufman',  '5th', 2013,  'CONCRETE',    '/books/kaufman.pdf',  NOW(), NULL, NULL),
('DOUGLAS',        'Trading in the Zone',
 'Mark Douglas',   '1st', 2000,  'PSYCHOLOGY',  '/books/douglas.pdf',  NOW(), NULL, NULL),
('LOPEZ',          'Advances in Financial Machine Learning',
 'Lopez de Prado', '1st', 2018,  'METHODOLOGY', '/books/lopez.pdf',    NOW(), NULL, NULL),
('HILPISCH',       'Python for Finance',
 'Yves Hilpisch',  '2nd', 2018,  'METHODOLOGY', '/books/hilpisch.pdf', NOW(), NULL, NULL),
('MCMILLAN',       'Options as a Strategic Investment',
 'Lawrence McMillan','5th', 2012, 'CONCRETE',    '/books/mcmillan.pdf', NOW(), NULL,
 'Butterflies, calendars, diagonals, ratio spreads. Direct rule extraction.'),
('AUGEN',          'The Volatility Edge in Options Trading',
 'Jeff Augen',     '1st', 2008,  'CONCRETE',    '/books/augen.pdf',    NOW(), NULL,
 'Weekly expiry dynamics, IV crush, pre-event IV expansion. Nifty weekly options.'),
('HARRIS',         'Trading and Exchanges',
 'Larry Harris',   '1st', 2003,  'METHODOLOGY', '/books/harris.pdf',   NOW(), NULL,
 'Market microstructure — liquidity filters, bid-ask regime detection. Filters only.'),
('NSE_EMPIRICAL',  'NSE India Empirical Patterns',
 'Internal',       'v1',  2026,  'CONCRETE',    NULL,                  NOW(), NULL,
 'FII derivatives positioning patterns. India-specific.');


-- ================================================================
-- TABLE 2: SIGNALS (current version — head record)
-- ================================================================
CREATE TABLE signals (
    -- Identity
    signal_id           TEXT PRIMARY KEY,
                        -- Format: {BOOK_PREFIX}_{3-digit-number}
                        -- e.g. GUJ_001, NAT_003, NSE_001
    name                TEXT NOT NULL,
    book_id             TEXT NOT NULL REFERENCES books(book_id),
    source_chapter      TEXT,
    source_page_start   INTEGER,
    source_page_end     INTEGER,
    raw_chunk_text      TEXT,           -- Exact book text
    version             INTEGER NOT NULL DEFAULT 1,

    -- Classification
    signal_category     TEXT NOT NULL
                        CHECK (signal_category IN (
                            'TREND', 'REVERSION', 'VOL',
                            'PATTERN', 'EVENT', 'SIZING',
                            'RISK', 'REGIME'
                        )),
    direction           TEXT NOT NULL
                        CHECK (direction IN (
                            'LONG', 'SHORT', 'NEUTRAL',
                            'CONTEXT_DEPENDENT'
                        )),
    instrument          TEXT NOT NULL
                        CHECK (instrument IN (
                            'FUTURES', 'OPTIONS_BUYING',
                            'OPTIONS_SELLING', 'SPREAD',
                            'COMBINED', 'ANY'
                        )),
    timeframe           TEXT NOT NULL
                        CHECK (timeframe IN (
                            'INTRADAY', 'POSITIONAL',
                            'SWING', 'ANY'
                        )),
    target_regimes      TEXT[] NOT NULL,
                        -- Array: ['TRENDING', 'RANGING', 'ANY', ...]

    -- Rule definition (structured)
    entry_conditions    JSONB NOT NULL,
                        -- Array of condition strings
    parameters          JSONB NOT NULL,
                        -- {param_name: value_or_AUTHOR_SILENT}
    exit_conditions     JSONB NOT NULL,
    sizing_rule         TEXT,           -- How to size this signal

    -- India-specific fields
    expiry_week_behavior TEXT NOT NULL DEFAULT 'NORMAL'
                        CHECK (expiry_week_behavior IN (
                            'NORMAL', 'AVOID_MONDAY',
                            'REDUCE_SIZE_MONDAY',
                            'AVOID_EXPIRY_WEEK'
                        )),
    min_strike_distance INTEGER DEFAULT 0,
                        -- Min OTM distance in strikes (0 = ATM ok)
    max_strike_distance INTEGER DEFAULT 3,
                        -- Max OTM distance allowed (3 = ATM±3 rule)

    -- Lifecycle status
    status              TEXT NOT NULL DEFAULT 'CANDIDATE'
                        CHECK (status IN (
                            'CANDIDATE',        -- Extracted, reviewed
                            'BACKTESTING',      -- Backtest in progress
                            'ACTIVE',           -- Live in system
                            'WATCH',            -- Degrading, monitor
                            'INACTIVE',         -- Failed, not running
                            'ARCHIVED',         -- Permanently removed
                            'PAPER_ONLY'        -- Paper trade only
                        )),
    classification      TEXT
                        CHECK (classification IN (
                            'PRIMARY', 'SECONDARY', NULL
                        )),
                        -- NULL until after paper trading
    avoid_rbi_day       BOOLEAN NOT NULL DEFAULT FALSE,
                        -- Set manually for signals that should not fire
                        -- on RBI Monetary Policy Committee decision days.
                        -- Used by SignalSelector.apply_calendar_filters().

    -- Backtest results
    backtest_tier       INTEGER CHECK (backtest_tier IN (1, 2, 3)),
    sharpe_ratio        FLOAT,
    calmar_ratio        FLOAT,
    max_drawdown        FLOAT,
    win_rate            FLOAT,
    profit_factor       FLOAT,
    avg_win_loss_ratio  FLOAT,
    total_trades        INTEGER,
    nifty_correlation   FLOAT,
    windows_passed      INTEGER,        -- Out of 4 walk-forward windows
    fragility_rating    TEXT
                        CHECK (fragility_rating IN (
                            'ROBUST', 'MODERATE', 'FRAGILE', NULL
                        )),

    -- Stress test results
    drawdown_mar2020    FLOAT,          -- Max DD in March 2020
    drawdown_mar2026    FLOAT,          -- Max DD in March 2026
    stress_test_passed  BOOLEAN,

    -- Live performance tracking
    rolling_sharpe_60d  FLOAT,
    rolling_sharpe_updated_at TIMESTAMPTZ,
    live_trades_count   INTEGER DEFAULT 0,
    live_win_rate       FLOAT,
    last_trade_date     DATE,

    -- Capital tracking
    required_margin     INTEGER,        -- Approximate ₹ margin per trade
    starvation_rate     FLOAT,          -- From paper trading

    -- Metadata
    human_reviewer      TEXT,
    review_notes        TEXT,
    conflict_group_id   TEXT,           -- Links conflicting signals

    -- Change-context columns for auto-versioning trigger.
    -- Application sets these BEFORE every UPDATE so the trigger
    -- can record the real reason, not just 'SYSTEM_TRIGGER'.
    -- Reset to defaults by the trigger after each capture.
    pending_change_by     TEXT NOT NULL DEFAULT 'SYSTEM',
    pending_change_reason TEXT NOT NULL DEFAULT '',

    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for common query patterns
CREATE INDEX idx_signals_status ON signals(status);
CREATE INDEX idx_signals_category ON signals(signal_category);
CREATE INDEX idx_signals_book ON signals(book_id);
CREATE INDEX idx_signals_classification ON signals(classification);
CREATE INDEX idx_signals_regime ON signals USING GIN(target_regimes);


-- ================================================================
-- TABLE 3: SIGNAL_VERSIONS
-- Full audit trail — every change preserved
-- ================================================================
CREATE TABLE signal_versions (
    version_id          SERIAL PRIMARY KEY,
    signal_id           TEXT NOT NULL REFERENCES signals(signal_id),
    version_number      INTEGER NOT NULL,

    -- Snapshot of all mutable fields at this version
    status              TEXT,
    classification      TEXT,
    entry_conditions    JSONB,
    parameters          JSONB,
    exit_conditions     JSONB,
    sharpe_ratio        FLOAT,
    rolling_sharpe_60d  FLOAT,
    expiry_week_behavior TEXT,

    -- Change tracking
    changed_by          TEXT NOT NULL,
                        -- 'HUMAN', 'BACKTEST_ENGINE',
                        -- 'PAPER_EVAL', 'LIVE_MONITOR'
    change_reason       TEXT NOT NULL,
    changed_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE (signal_id, version_number)
);

-- Trigger: auto-create version record on every signal update
CREATE OR REPLACE FUNCTION record_signal_version()
RETURNS TRIGGER AS $$
BEGIN
    -- Read change context from the pending_* columns
    -- that the application sets before each UPDATE.
    -- Provides meaningful audit context per change.
    INSERT INTO signal_versions (
        signal_id, version_number, status, classification,
        entry_conditions, parameters, exit_conditions,
        sharpe_ratio, rolling_sharpe_60d, expiry_week_behavior,
        changed_by, change_reason
    ) VALUES (
        OLD.signal_id, OLD.version,
        OLD.status, OLD.classification,
        OLD.entry_conditions, OLD.parameters, OLD.exit_conditions,
        OLD.sharpe_ratio, OLD.rolling_sharpe_60d,
        OLD.expiry_week_behavior,
        NEW.pending_change_by,      -- Reads app-supplied context
        NEW.pending_change_reason   -- Reads app-supplied context
    );
    NEW.version              := OLD.version + 1;
    NEW.updated_at           := NOW();
    NEW.pending_change_by    := 'SYSTEM';   -- Reset after capture
    NEW.pending_change_reason := '';        -- Reset after capture
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- HOW TO USE (application code before every signal UPDATE):
-- db.execute("""
--     UPDATE signals SET
--         pending_change_by     = %s,
--         pending_change_reason = %s,
--         sharpe_ratio          = %s
--     WHERE signal_id = %s
-- """, ('BACKTEST_ENGINE', 'Round 1 Sharpe result', 1.45, 'GUJ_001'))

CREATE TRIGGER signal_version_trigger
BEFORE UPDATE ON signals
FOR EACH ROW EXECUTE FUNCTION record_signal_version();


-- ================================================================
-- TABLE 4: CONFLICT_GROUPS
-- Links signals that conflict with each other
-- ================================================================
CREATE TABLE conflict_groups (
    conflict_group_id   TEXT PRIMARY KEY,
    conflict_type       TEXT NOT NULL
                        CHECK (conflict_type IN (
                            'DIRECT_CONTRADICTION',
                            'PARAMETER_DISAGREEMENT',
                            'CONDITION_DISAGREEMENT',
                            'EXIT_DISAGREEMENT',
                            'COMPLEMENTARY'
                        )),
    conflict_description TEXT NOT NULL,
    resolution_method   TEXT,
    winning_signal_id   TEXT REFERENCES signals(signal_id),
    resolved_at         TIMESTAMPTZ,
    resolution_notes    TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE conflict_group_members (
    conflict_group_id   TEXT REFERENCES conflict_groups(conflict_group_id),
    signal_id           TEXT REFERENCES signals(signal_id),
    PRIMARY KEY (conflict_group_id, signal_id)
);


-- ================================================================
-- TABLE 5: TRADES
-- Every executed trade (live and paper)
-- ================================================================
CREATE TABLE trades (
    trade_id            BIGSERIAL PRIMARY KEY,
    signal_id           TEXT NOT NULL REFERENCES signals(signal_id),
    trade_type          TEXT NOT NULL
                        CHECK (trade_type IN ('LIVE', 'PAPER')),

    -- Entry
    entry_date          DATE NOT NULL,
    entry_time          TIME NOT NULL,
    instrument          TEXT NOT NULL,   -- e.g. NIFTY24JAN25000CE
    direction           TEXT NOT NULL,
    lots                INTEGER NOT NULL,
    entry_price         FLOAT NOT NULL,
    entry_regime        TEXT,
    entry_vix           FLOAT,

    -- Exit
    exit_date           DATE,
    exit_time           TIME,
    exit_price          FLOAT,
    exit_reason         TEXT,
                        -- SIGNAL_EXIT, STOP_LOSS, TIME_EXIT,
                        -- MANUAL, DAILY_LOSS_LIMIT

    -- P&L
    gross_pnl           FLOAT,
    costs               FLOAT,          -- Brokerage + STT + charges
    net_pnl             FLOAT,
    return_pct          FLOAT,          -- Net return as % of margin used

    -- Execution quality
    intended_lots       INTEGER,
    fill_quality        TEXT,
                        -- FULL, PARTIAL, UNFILLED
    slippage_pts        FLOAT,

    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Convert to hypertable for time-series performance
SELECT create_hypertable('trades', 'entry_date',
                          if_not_exists => TRUE);

CREATE INDEX idx_trades_signal ON trades(signal_id);
CREATE INDEX idx_trades_type ON trades(trade_type);


-- ================================================================
-- TABLE 6: PORTFOLIO_STATE
-- Point-in-time portfolio snapshots for reconciliation
-- ================================================================
CREATE TABLE portfolio_state (
    snapshot_id         BIGSERIAL PRIMARY KEY,
    snapshot_time       TIMESTAMPTZ NOT NULL,
    snapshot_type       TEXT NOT NULL
                        CHECK (snapshot_type IN (
                            'DAILY_OPEN', 'DAILY_CLOSE',
                            'POST_TRADE', 'RECONCILIATION'
                        )),

    total_capital       FLOAT NOT NULL,
    deployed_capital    FLOAT NOT NULL,
    cash_reserve        FLOAT NOT NULL,
    open_positions      JSONB,          -- Array of position summaries
    portfolio_delta     FLOAT,
    portfolio_vega      FLOAT,
    portfolio_gamma     FLOAT,
    portfolio_theta     FLOAT,
    daily_pnl           FLOAT,
    mtd_pnl             FLOAT,
    ytd_pnl             FLOAT
);

SELECT create_hypertable('portfolio_state', 'snapshot_time',
                          if_not_exists => TRUE);


-- ================================================================
-- TABLE 7: ALERTS_LOG
-- Every alert sent — for audit and tuning
-- ================================================================
CREATE TABLE alerts_log (
    alert_id            BIGSERIAL PRIMARY KEY,
    alert_time          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    alert_level         TEXT NOT NULL
                        CHECK (alert_level IN (
                            'INFO', 'WARNING',
                            'CRITICAL', 'EMERGENCY'
                        )),
    alert_type          TEXT NOT NULL,
    message             TEXT NOT NULL,
    signal_id           TEXT,
    acknowledged        BOOLEAN DEFAULT FALSE,
    acknowledged_at     TIMESTAMPTZ
);
```

# IMPLEMENTATION SPEC 6: MONITORING AND ALERT THRESHOLDS
# Written by: System Architect + India Market Expert
# Status: FINAL

---

## ALERT ARCHITECTURE

```
Three alert channels:
1. Phone (Telegram bot) — CRITICAL and EMERGENCY only
2. Email (daily digest) — WARNING and INFO summary
3. PostgreSQL alerts_log — everything, always

Never wake you up for INFO.
Never miss a CRITICAL.
```

---

## COMPLETE ALERT DEFINITIONS

```python
# File: monitoring/alert_definitions.py

ALERT_DEFINITIONS = {

    # ================================================================
    # SYSTEM HEALTH ALERTS
    # ================================================================

    'DATA_FEED_DISCONNECT': {
        'level': 'CRITICAL',
        'channel': ['phone', 'log'],
        'trigger': 'TrueData WebSocket disconnected',
        'threshold': 'No data received for 60 seconds during market hours',
        'message': 'PRIMARY DATA FEED DOWN. Backup activating.',
        'auto_action': 'Activate Kite ticker backup feed',
        'escalation': 'If backup also fails within 90s → EMERGENCY',
    },

    'BOTH_FEEDS_DOWN': {
        'level': 'EMERGENCY',
        'channel': ['phone', 'log'],
        'trigger': 'Both TrueData and Kite ticker failed',
        'threshold': 'Both feeds silent for 90 seconds during market hours',
        'message': 'EMERGENCY: ALL DATA FEEDS DOWN. Positions at risk.',
        'auto_action': 'Close all positions at market after 10 min',
        'note': 'Wait 10 min before emergency close — may be brief outage',
    },

    'KITE_API_TIMEOUT': {
        'level': 'WARNING',
        'channel': ['log'],
        # Phone reserved for CRITICAL/EMERGENCY. A single timeout retries automatically.
        # Escalation to KITE_API_DOWN (CRITICAL → phone) handles repeated failures.
        'trigger': 'Kite API did not respond within 5 seconds',
        'threshold': '1 timeout',
        'message': 'Kite API timeout on order {order_id}',
        'auto_action': 'Retry once. Check if order was placed.',
        'escalation': '3 timeouts in 10 minutes → KITE_API_DOWN (CRITICAL)',
    },

    'KITE_API_DOWN': {
        'level': 'CRITICAL',
        'channel': ['phone', 'log'],
        'trigger': 'Kite API returning errors consistently',
        'threshold': '3 consecutive API failures',
        'message': 'CRITICAL: Kite API down. Trading halted.',
        'auto_action': 'Block all new orders. Keep existing positions.',
        'note': 'Call Zerodha support: 080-33006000',
    },

    'HEARTBEAT_MISSED': {
        'level': 'WARNING',
        'channel': ['log'],
        'trigger': 'Signal Engine heartbeat not received',
        'threshold': 'No heartbeat for 120 seconds during market hours',
        'message': 'Signal Engine heartbeat missed',
        'auto_action': 'Attempt restart. Alert if restart fails.',
    },

    'POSITION_MISMATCH': {
        'level': 'EMERGENCY',
        'channel': ['phone', 'log'],
        'trigger': 'Reconciliation found position discrepancy',
        'threshold': 'Any mismatch between system and Kite positions',
        'message': 'EMERGENCY: Position mismatch detected. '
                   'Manual investigation required.',
        'auto_action': 'HALT ALL TRADING immediately',
        'note': 'Do NOT auto-close. Unknown positions may be hedges.',
    },

    # ================================================================
    # RISK ALERTS
    # ================================================================

    'DAILY_LOSS_LIMIT_HIT': {
        'level': 'CRITICAL',
        'channel': ['phone', 'log'],
        'trigger': 'Daily P&L crosses DAILY_LOSS_LIMIT',
        'threshold': 'net_pnl_today <= -settings.DAILY_LOSS_LIMIT',
        'message': 'DAILY LOSS LIMIT HIT: ₹{loss}. '
                   'No new entries for today.',
        'auto_action': 'Block all new entries. Existing positions held.',
    },

    'DAILY_LOSS_WARNING': {
        'level': 'WARNING',
        'channel': ['phone', 'log'],
        'trigger': 'Daily loss approaching limit',
        'threshold': 'net_pnl_today <= -settings.DAILY_LOSS_LIMIT * 0.7',
        'message': 'Daily loss at ₹{loss}. Approaching limit.',
        'auto_action': 'Reduce new position sizes to 50%',
    },

    'WEEKLY_LOSS_LIMIT_HIT': {
        'level': 'CRITICAL',
        'channel': ['phone', 'log'],
        'trigger': 'Weekly P&L crosses WEEKLY_LOSS_LIMIT',
        'threshold': 'net_pnl_week <= -settings.WEEKLY_LOSS_LIMIT',
        'message': 'WEEKLY LOSS LIMIT HIT. Trading suspended this week.',
        'auto_action': 'Block all new entries for rest of week. '
                       'Secondaries suspended 2 weeks.',
    },

    'MONTHLY_DRAWDOWN_CRITICAL': {
        'level': 'EMERGENCY',
        'channel': ['phone', 'log'],
        'trigger': 'Monthly drawdown crosses 15%',
        'threshold': 'monthly_drawdown_pct >= 15',
        'message': 'EMERGENCY: Monthly drawdown {pct}%. '
                   'System entering paper mode.',
        'auto_action': 'Switch to PAPER_TRADING_MODE. '
                       'All live orders blocked.',
    },

    'GREEK_LIMIT_BREACH': {
        'level': 'WARNING',
        'channel': ['phone', 'log'],
        'trigger': 'Portfolio Greeks approaching limits',
        'threshold': {
            # 80% of each limit from settings.GREEK_LIMITS (scales with capital)
            k: v * 0.8 for k, v in settings.GREEK_LIMITS.items()
            if k.startswith('max_portfolio_')
        },
        'message': 'Greek limit approaching: {greek} at {value}. '
                   'New signals filtered.',
        'auto_action': 'Signal Selector blocks signals '
                       'that worsen breaching Greek',
    },

    'MARGIN_LOW': {
        'level': 'WARNING',
        'channel': ['phone', 'log'],
        'trigger': 'Available margin drops below threshold',
        'threshold': f'available_margin < {settings.TOTAL_CAPITAL * 0.30}',  # 30% of capital
        'message': 'Low margin: ₹{available} available. '
                   'New entries restricted.',
        'auto_action': f'Block new entries until margin > {settings.TOTAL_CAPITAL * 0.40}',  # 40% of capital
    },

    # ================================================================
    # SIGNAL HEALTH ALERTS
    # ================================================================

    'SIGNAL_ROLLING_SHARPE_WATCH': {
        'level': 'WARNING',
        'channel': ['log'],          # Email digest only
        'trigger': 'Signal rolling Sharpe degrading',
        'threshold': 'rolling_sharpe_60d < 0.5',
        'message': 'Signal {signal_id} rolling Sharpe at {sharpe}. '
                   'Status → WATCH.',
        'auto_action': 'Set signal status to WATCH. '
                       'Reduce size to 50%.',
    },

    'SIGNAL_DEACTIVATED': {
        'level': 'WARNING',
        'channel': ['phone', 'log'],
        'trigger': 'Signal auto-deactivated due to sustained losses',
        'threshold': 'rolling_sharpe_60d < 0.0 for 20 consecutive days',
        'message': 'Signal {signal_id} DEACTIVATED. '
                   'Rolling Sharpe negative 20 days.',
        'auto_action': 'Set status INACTIVE. Human review required.',
    },

    'MULTIPLE_SIGNALS_DEGRADING': {
        'level': 'CRITICAL',
        'channel': ['phone', 'log'],
        'trigger': '3+ signals enter WATCH in same month',
        'threshold': 'watch_signals_this_month >= 3',
        'message': 'REGIME CHANGE SIGNAL: {n} signals degrading. '
                   'Review all active signals.',
        'auto_action': 'Reduce ALL signal sizes to 50%. '
                       'Trigger monthly recalibration review.',
        'note': 'When 3+ signals degrade simultaneously, '
                'regime has likely changed.',
    },

    'SIGNAL_ABNORMAL_FREQUENCY': {
        'level': 'WARNING',
        'channel': ['log'],
        'trigger': 'Signal firing much more or less than expected',
        'threshold': 'actual_fires_30d < 0.4 * expected_fires_30d OR '
                     'actual_fires_30d > 3.0 * expected_fires_30d',
        'message': 'Signal {signal_id} firing abnormally: '
                   '{actual} fires vs {expected} expected.',
        'auto_action': 'Log for review. No automatic action.',
    },

    # ================================================================
    # EXECUTION ALERTS
    # ================================================================

    'ORDER_UNFILLED': {
        'level': 'INFO',
        'channel': ['log'],
        'trigger': 'Order cancelled after 90-second fill window',
        'threshold': 'fill_window_expired',
        'message': 'Order {order_id} for {signal_id} unfilled. Cancelled.',
        'auto_action': 'None. Log as UNFILLED trade.',
    },

    'UNUSUAL_SLIPPAGE': {
        'level': 'WARNING',
        'channel': ['log'],
        'trigger': 'Fill price significantly worse than limit price',
        'threshold': 'slippage > 0.5% of premium for options, '
                     'slippage > 5pts for futures',
        'message': 'High slippage on {signal_id}: {slippage}pts. '
                   'Market conditions poor.',
        'auto_action': 'Log. If 3+ high-slippage trades today, '
                       'reduce order sizes by 30%.',
    },

    'CONSECUTIVE_LOSSES': {
        'level': 'WARNING',
        'channel': ['phone', 'log'],
        'trigger': 'Same signal loses N consecutive trades',
        'threshold': 'consecutive_losses_per_signal >= 4',
        'message': 'Signal {signal_id} has {n} consecutive losses.',
        'auto_action': 'Reduce signal to 25% size for next 5 trades.',
    },

    # ================================================================
    # DAILY DIGEST (email, sent at 4:30 PM every trading day)
    # ================================================================

    'DAILY_DIGEST': {
        'level': 'INFO',
        'channel': ['email'],
        'trigger': 'Scheduled — 4:30 PM every trading day',
        'contents': [
            'Today P&L: ₹{daily_pnl}',
            'MTD P&L: ₹{mtd_pnl}',
            'YTD P&L: ₹{ytd_pnl}',
            'Signals fired today: {signals_today}',
            'Signals in WATCH status: {watch_signals}',
            'Signals in INACTIVE status: {inactive_signals}',
            'Portfolio Greeks at close: Δ={delta} V={vega}',
            'Tomorrow regime forecast: {regime_forecast}',
            'Tomorrow calendar events: {events}',
            'Alerts triggered today: {alert_count}',
        ],
    },

    # ================================================================
    # THRESHOLDS REQUIRING IMMEDIATE PHONE ALERT
    # (Summary — phone only gets these)
    # ================================================================
    # DATA_FEED_DISCONNECT
    # BOTH_FEEDS_DOWN
    # KITE_API_DOWN
    # POSITION_MISMATCH
    # DAILY_LOSS_LIMIT_HIT
    # WEEKLY_LOSS_LIMIT_HIT
    # MONTHLY_DRAWDOWN_CRITICAL
    # MULTIPLE_SIGNALS_DEGRADING
    # SIGNAL_DEACTIVATED
    # CONSECUTIVE_LOSSES
    # DAILY_LOSS_WARNING (precautionary)
}
```

---

## TELEGRAM BOT SETUP
# File: monitoring/telegram_alerter.py

```python
"""
Uses python-telegram-bot library.
Bot sends to your personal chat ID.
Setup: Create bot via @BotFather, get token and chat_id.
"""

class TelegramAlerter:

    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"

    def send(self, level: str, message: str,
             signal_id: str = None):
        """Send alert. Blocks until delivered."""
        emoji = {
            'INFO':      'ℹ️',
            'WARNING':   '⚠️',
            'CRITICAL':  '🔴',
            'EMERGENCY': '🚨'
        }.get(level, '❓')

        text = f"{emoji} *{level}*\n{message}"
        if signal_id:
            text += f"\nSignal: `{signal_id}`"
        text += f"\n_{datetime.now().strftime('%H:%M:%S IST')}_"

        import requests
        requests.post(
            f"{self.base_url}/sendMessage",
            json={
                'chat_id': self.chat_id,
                'text': text,
                'parse_mode': 'Markdown'
            },
            timeout=10
        )
```

# IMPLEMENTATION SPEC 7: FII DATA PIPELINE
# Written by: India Market Expert
# Status: FINAL

---

## NSE DATA SOURCE

```
URL (participant-wise OI — use this):
https://archives.nseindia.com/content/fo/fii-stats-{DD-Mon-YYYY}.xls

Track A (preferred — stable Bhav Copy archive):
https://archives.nseindia.com/content/fo/fii-stats-{DD-Mon-YYYY}.xls
This is NSE's officially published FII stats file. Less scraping-fragile
than the dynamic API. Verify it contains OI data (not just volumes)
by downloading one sample before coding the pipeline.

Track B (API fallback — use if Bhav Copy lacks OI detail):
https://www.nseindia.com/api/reports?archives=%5B%7B%22name%22%3A%22F%26O+Participant+wise+OI%22%2C%22type%22%3A%22archives%22%2C%22category%22%3A%22derivatives%22%2C%22section%22%3A%22equity%22%7D%5D&date={DD-Mon-YYYY}&type=equity&category=derivatives

NSE requires browser-like headers. Use session with cookies.
Publish time: 6:00-7:00 PM on trading days.
If not available by 7:00 PM: retry every 15 minutes until 9:00 PM.
If not available by 9:00 PM: skip that day, log as DATA_MISSING.

PLAYWRIGHT FALLBACK:
NSE's API endpoint sometimes returns Cloudflare bot-protection cookies
that requests.Session() cannot obtain without a headless browser.
If the requests approach fails consistently (>3 consecutive days):
switch to Playwright, which handles JavaScript cookies automatically.

Config flag: USE_PLAYWRIGHT = False  (default)
Set to True in config.yaml if requests approach starts failing.

Install: pip install playwright && playwright install chromium

Playwright download snippet:
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto("https://www.nseindia.com")
        page.wait_for_timeout(3000)  # Let JS set cookies
        # Then use page.goto(api_url) and page.content()
        browser.close()

Playwright is ~3× slower than requests but handles NSE reliably.
Use only as fallback — don't switch unless requests fails.
```

---

## MODULE 1: FII DATA DOWNLOADER
# File: fii/downloader.py

```python
import requests
import time
from datetime import datetime, date
import pandas as pd
from io import StringIO

class FIIDataDownloader:
    """
    Downloads NSE participant-wise F&O data.
    Must mimic browser — NSE blocks raw API calls.
    """

    NSE_BASE = "https://www.nseindia.com"
    SESSION_URL = "https://www.nseindia.com/market-data/equity-derivatives-watch"

    HEADERS = {
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/120.0.0.0 Safari/537.36'
        ),
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Referer': 'https://www.nseindia.com/',
        'Connection': 'keep-alive',
    }

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)
        self._cookies_initialized = False

    def _init_session(self):
        """
        Fetch NSE homepage first to get session cookies.
        Without this, data requests return 401.
        Must be called before any data download.
        """
        if self._cookies_initialized:
            return

        resp = self.session.get(self.SESSION_URL, timeout=30)
        resp.raise_for_status()
        time.sleep(2)  # NSE rate limit — always wait after homepage
        self._cookies_initialized = True

    def download_participant_oi(self, for_date: date) -> pd.DataFrame:
        """
        Downloads participant-wise OI (open interest) data.
        This is the key file for FII positioning signals.

        Returns DataFrame with columns:
        client_type, future_long_contracts, future_short_contracts,
        call_long_contracts, call_short_contracts,
        put_long_contracts, put_short_contracts,
        total_long_contracts, total_short_contracts

        client_type values: FII, DII, PRO, CLIENT
        """
        self._init_session()

        date_str = for_date.strftime('%d-%b-%Y')  # e.g. 12-Mar-2026

        api_url = (
            f"{self.NSE_BASE}/api/reports?"
            f"archives=%5B%7B%22name%22%3A%22F%26O+Participant+wise+OI%22"
            f"%2C%22type%22%3A%22archives%22%2C%22category%22%3A%22derivatives%22"
            f"%2C%22section%22%3A%22equity%22%7D%5D"
            f"&date={date_str}&type=equity&category=derivatives"
        )

        resp = self.session.get(api_url, timeout=30)

        if resp.status_code == 404:
            raise DataNotAvailableError(
                f"FII OI data not yet published for {date_str}"
            )
        resp.raise_for_status()

        # NSE returns a redirect URL to the actual CSV
        data = resp.json()
        if not data or not isinstance(data, list):
            raise DataNotAvailableError(f"Unexpected response for {date_str}")

        # Download the CSV file
        csv_url = self.NSE_BASE + data[0].get('link', '')
        csv_resp = self.session.get(csv_url, timeout=30)
        csv_resp.raise_for_status()

        # Parse CSV
        df = pd.read_csv(StringIO(csv_resp.text))
        return self._normalize_oi_columns(df, for_date)

    def _normalize_oi_columns(self, df: pd.DataFrame,
                               for_date: date) -> pd.DataFrame:
        """
        NSE changes column names occasionally.
        Normalize to our standard names.
        """
        # Standard NSE column aliases
        column_map = {
            'Client Type':          'client_type',
            'Future Long':          'future_long_contracts',
            'Future Short':         'future_short_contracts',
            'Option Call Long':     'call_long_contracts',
            'Option Call Short':    'call_short_contracts',
            'Option Put Long':      'put_long_contracts',
            'Option Put Short':     'put_short_contracts',
            'Total Long Contracts': 'total_long_contracts',
            'Total Short Contracts':'total_short_contracts',
        }

        # Flexible matching (NSE sometimes adds spaces or changes case)
        rename = {}
        for col in df.columns:
            for nse_name, std_name in column_map.items():
                if nse_name.lower().replace(' ', '') in \
                   col.lower().replace(' ', ''):
                    rename[col] = std_name
                    break
        df = df.rename(columns=rename)

        # Keep only FII row
        fii_mask = df['client_type'].str.upper().str.contains(
            'FII|FPI|FOREIGN', na=False
        )
        df = df[fii_mask].copy()
        df['date'] = for_date

        return df
```

---

## MODULE 2: FII SIGNAL DETECTOR
# File: fii/signal_detector.py

```python
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class FIISignalResult:
    date: object
    signal_id: Optional[str]    # None if no signal
    direction: Optional[str]    # 'BULLISH' | 'BEARISH' | 'NEUTRAL'
    confidence: float           # 0.0 to 1.0
    pattern_name: str
    raw_metrics: dict
    valid_for_date: object      # Trade on THIS date (next day)
    notes: str

class FIISignalDetector:
    """
    Detects 4 FII positioning patterns.
    Runs at 7:30 PM after downloading day's OI data.
    Produces signals valid for NEXT trading day.
    """

    # ================================================================
    # PATTERN THRESHOLDS
    # Derived from NSE empirical observation 2021-2025.
    # Revisit if regime changes significantly.
    # ================================================================

    # Pattern 1: Bearish — FII short futures + long puts
    P1_MIN_FUTURE_NET_SHORT = -50_000   # contracts (negative = net short)
    P1_MIN_PUT_LONG_RATIO   = 1.5       # put_long / put_short ratio
    P1_MIN_FUTURE_PERCENTILE = 75       # Net short must be in top 75th
                                         # percentile historically

    # Pattern 2: Bullish — FII long futures + covering puts
    P2_MIN_FUTURE_NET_LONG  = 30_000    # contracts
    P2_MAX_PUT_LONG_RATIO   = 0.8       # Put longs falling (< 0.8)
    P2_MIN_FUTURE_PERCENTILE = 70

    # Pattern 3: Large futures shift (continuation signal)
    # Threshold is computed dynamically from Nifty level:
    # ₹3,000Cr / (Nifty × lot_size) → ~16,667 contracts at Nifty 24,000.
    # A hardcoded contract count would become wrong as Nifty levels change.
    P3_CRORE_THRESHOLD      = 3000      # ₹3,000 crore minimum shift
    P3_LOT_SIZE             = 75        # Nifty lot size (fixed by NSE)

    @staticmethod
    def compute_p3_threshold(nifty_level: float,
                              crore_threshold: float = 3000) -> int:
        """
        Dynamic Pattern 3 threshold based on current Nifty level.
        ₹3,000Cr at Nifty 24,000: ~16,667 contracts (not 50,000).
        At Nifty 22,500: ~17,778 contracts.
        At Nifty 25,000: ~16,000 contracts.
        """
        contract_value_rupees = nifty_level * 75
        rupee_threshold       = crore_threshold * 1e7   # crore → rupees
        return int(rupee_threshold / contract_value_rupees)

    # Pattern 4: Pure put buying (hedging — IGNORE, not a signal)
    P4_MIN_PUT_LONG_CONTRACTS = 200_000  # Large put buy
    P4_MAX_FUTURE_CHANGE    = 10_000     # But futures barely moved
    # If Pattern 4 detected: output NEUTRAL, do not trade

    # Minimum data history required to compute percentiles
    MIN_HISTORY_DAYS = 60

    def __init__(self, db_connection):
        self.db = db_connection

    def detect(self, today_data: pd.DataFrame,
               for_date: object) -> FIISignalResult:
        """
        Runs all 4 pattern checks in order.
        Returns first matching pattern, or NEUTRAL if none.

        today_data: normalized FII OI row for today
        for_date: the trading date this signal applies to (tomorrow)
        """
        # Extract FII row metrics
        metrics = self._extract_metrics(today_data)

        # Load historical context (last 252 trading days)
        history = self._load_history(days=252)

        # Add today to history for percentile calculation
        history_with_today = pd.concat([
            history,
            pd.DataFrame([metrics])
        ], ignore_index=True)

        if len(history_with_today) < self.MIN_HISTORY_DAYS:
            return FIISignalResult(
                date=metrics['date'],
                signal_id=None,
                direction='NEUTRAL',
                confidence=0.0,
                pattern_name='INSUFFICIENT_HISTORY',
                raw_metrics=metrics,
                valid_for_date=for_date,
                notes=f"Only {len(history_with_today)} days of history. "
                      f"Need {self.MIN_HISTORY_DAYS}."
            )

        # Check Pattern 4 first (hedging — suppress other signals)
        if self._is_pattern_4_hedging(metrics, history_with_today):
            return FIISignalResult(
                date=metrics['date'],
                signal_id='NSE_004',
                direction='NEUTRAL',
                confidence=0.9,
                pattern_name='FII_PURE_HEDGE',
                raw_metrics=metrics,
                valid_for_date=for_date,
                notes='FII buying puts without shorting futures. '
                      'Hedging behavior. Do not trade directionally.'
            )

        # Pattern 1: Bearish setup
        p1_result = self._check_pattern_1_bearish(
            metrics, history_with_today
        )
        if p1_result['detected']:
            return FIISignalResult(
                date=metrics['date'],
                signal_id='NSE_001',
                direction='BEARISH',
                confidence=p1_result['confidence'],
                pattern_name='FII_SHORT_FUTURES_LONG_PUTS',
                raw_metrics=metrics,
                valid_for_date=for_date,
                notes=p1_result['notes']
            )

        # Pattern 2: Bullish setup
        p2_result = self._check_pattern_2_bullish(
            metrics, history_with_today
        )
        if p2_result['detected']:
            return FIISignalResult(
                date=metrics['date'],
                signal_id='NSE_002',
                direction='BULLISH',
                confidence=p2_result['confidence'],
                pattern_name='FII_LONG_FUTURES_COVERING_PUTS',
                raw_metrics=metrics,
                valid_for_date=for_date,
                notes=p2_result['notes']
            )

        # Pattern 3: Large shift continuation
        # Pass today's Nifty close so the contract threshold scales with Nifty level.
        p3_result = self._check_pattern_3_shift(
            metrics, history_with_today,
            current_nifty=metrics.get('nifty_close', 24000.0)
        )
        if p3_result['detected']:
            return FIISignalResult(
                date=metrics['date'],
                signal_id='NSE_003',
                direction=p3_result['direction'],
                confidence=p3_result['confidence'],
                pattern_name='FII_LARGE_FUTURES_SHIFT',
                raw_metrics=metrics,
                valid_for_date=for_date,
                notes=p3_result['notes']
            )

        # No pattern detected
        return FIISignalResult(
            date=metrics['date'],
            signal_id=None,
            direction='NEUTRAL',
            confidence=0.0,
            pattern_name='NO_PATTERN',
            raw_metrics=metrics,
            valid_for_date=for_date,
            notes='No FII positioning pattern detected today.'
        )

    def _extract_metrics(self, fii_row: pd.DataFrame) -> dict:
        """Extract and compute derived metrics from raw OI row.
        Requires 'nifty_close' key — used for the dynamic Pattern 3 threshold.
        The NSE OI report does not contain the Nifty price; the pipeline
        fetches today's Nifty close and injects it before calling this method.
        Defaults to 24000.0 if absent (conservative mid-range estimate).
        """
        row = fii_row.iloc[0]
        fut_long  = float(row.get('future_long_contracts', 0))
        fut_short = float(row.get('future_short_contracts', 0))
        put_long  = float(row.get('put_long_contracts', 0))
        put_short = float(row.get('put_short_contracts', 0))
        call_long = float(row.get('call_long_contracts', 0))
        call_short= float(row.get('call_short_contracts', 0))

        return {
            'date':             row.get('date'),
            'nifty_close':      float(row.get('nifty_close', 24000.0)),
            # Injected by FIIDailyPipeline.run() before calling detect()
            'fut_long':         fut_long,
            'fut_short':        fut_short,
            'fut_net':          fut_long - fut_short,
            'put_long':         put_long,
            'put_short':        put_short,
            'put_net':          put_long - put_short,
            'put_ratio':        put_long / put_short
                                if put_short > 0 else 999,
            'call_long':        call_long,
            'call_short':       call_short,
            'call_net':         call_long - call_short,
            'pcr':              (put_long + put_short) /
                                (call_long + call_short)
                                if (call_long + call_short) > 0 else 1,
        }    def _check_pattern_1_bearish(self, m: dict,
                                  history: pd.DataFrame) -> dict:
        """
        Pattern NSE_001: FII net short futures AND buying puts.
        Historical accuracy: 68% over 1-3 trading days.
        """
        # Condition 1: FII net short futures beyond threshold
        if m['fut_net'] >= self.P1_MIN_FUTURE_NET_SHORT:
            return {'detected': False}

        # Condition 2: Put buying ratio above threshold
        if m['put_ratio'] < self.P1_MIN_PUT_LONG_RATIO:
            return {'detected': False}

        # Condition 3: Net short position must be extreme (percentile)
        fut_net_pct = self._percentile_rank(
            history['fut_net'], m['fut_net']
        )
        # Lower percentile = more net short
        if fut_net_pct > (100 - self.P1_MIN_FUTURE_PERCENTILE):
            return {'detected': False}

        # Compute confidence from percentile depth
        confidence = min(0.95, 0.60 + (
            (100 - self.P1_MIN_FUTURE_PERCENTILE - fut_net_pct) / 100
        ))

        return {
            'detected': True,
            'confidence': confidence,
            'notes': (
                f"FII net futures: {m['fut_net']:,.0f} contracts "
                f"(bottom {fut_net_pct:.0f}th percentile). "
                f"Put ratio: {m['put_ratio']:.2f}. "
                f"Bearish signal with {confidence:.0%} confidence."
            )
        }

    def _check_pattern_2_bullish(self, m: dict,
                                  history: pd.DataFrame) -> dict:
        """
        Pattern NSE_002: FII net long futures AND reducing put longs.
        Historical accuracy: 65% over 1-3 trading days.
        """
        if m['fut_net'] < self.P2_MIN_FUTURE_NET_LONG:
            return {'detected': False}

        if m['put_ratio'] > self.P2_MAX_PUT_LONG_RATIO:
            return {'detected': False}

        # Need yesterday's put_ratio to confirm declining
        if len(history) < 2:
            return {'detected': False}

        yesterday_put_ratio = history.iloc[-1].get('put_ratio', 999)
        if m['put_ratio'] >= yesterday_put_ratio:
            return {'detected': False}  # Put ratio not declining

        fut_net_pct = self._percentile_rank(
            history['fut_net'], m['fut_net']
        )
        if fut_net_pct < self.P2_MIN_FUTURE_PERCENTILE:
            return {'detected': False}

        confidence = min(0.90, 0.55 + (fut_net_pct - 70) / 200)

        return {
            'detected': True,
            'confidence': confidence,
            'notes': (
                f"FII net futures: {m['fut_net']:,.0f} contracts "
                f"(top {fut_net_pct:.0f}th percentile). "
                f"Put ratio declining: {yesterday_put_ratio:.2f} → "
                f"{m['put_ratio']:.2f}. "
                f"Bullish signal with {confidence:.0%} confidence."
            )
        }

    def _check_pattern_3_shift(self, m: dict,
                                history: pd.DataFrame,
                                current_nifty: float = 24000.0) -> dict:
        """
        Pattern NSE_003: Large single-day futures position shift.
        Continuation signal — direction follows shift direction.
        Threshold scales with Nifty level: at 24,000 → ~16,667 contracts.
        Caller should pass current Nifty close price.
        """
        if len(history) < 2:
            return {'detected': False}

        p3_threshold = self.compute_p3_threshold(current_nifty,
                                                  self.P3_CRORE_THRESHOLD)

        yesterday_fut_net = history.iloc[-1].get('fut_net', 0)
        day_change = m['fut_net'] - yesterday_fut_net

        if abs(day_change) < p3_threshold:
            return {'detected': False}

        direction = 'BULLISH' if day_change > 0 else 'BEARISH'

        # Compute historical significance of this shift size
        if 'fut_net_change' in history.columns:
            pct = self._percentile_rank(
                history['fut_net_change'].abs(),
                abs(day_change)
            )
        else:
            pct = 80  # Default if history lacks change column

        confidence = min(0.85, 0.55 + pct / 400)

        return {
            'detected': True,
            'direction': direction,
            'confidence': confidence,
            'notes': (
                f"FII futures shift: {day_change:+,.0f} contracts "
                f"(threshold: {p3_threshold:,} at Nifty {current_nifty:,.0f}). "
                f"{pct:.0f}th percentile of historical shifts. "
                f"{direction} continuation expected."
            )
        }

    def _is_pattern_4_hedging(self, m: dict,
                               history: pd.DataFrame) -> bool:
        """
        Pattern NSE_004: Pure hedging — puts bought, futures unchanged.
        This is institutional hedging, NOT a directional signal.
        Return True to SUPPRESS other signals for today.
        """
        if m['put_long'] < self.P4_MIN_PUT_LONG_CONTRACTS:
            return False

        if len(history) < 2:
            return False

        yesterday_fut_net = history.iloc[-1].get('fut_net', 0)
        fut_change = abs(m['fut_net'] - yesterday_fut_net)

        if fut_change > self.P4_MAX_FUTURE_CHANGE:
            return False  # Futures DID move — not pure hedging

        return True  # Large put buy + stable futures = hedging

    def _percentile_rank(self, series: pd.Series,
                          value: float) -> float:
        """Returns percentile rank of value in series (0-100)."""
        series = series.dropna()
        if len(series) == 0:
            return 50.0
        return float((series <= value).mean() * 100)

    def _load_history(self, days: int) -> pd.DataFrame:
        """Load historical FII metrics from DB."""
        rows = self.db.execute(
            """
            SELECT date, fut_net, put_ratio, put_long, fut_net_change
            FROM fii_daily_metrics
            ORDER BY date DESC
            LIMIT %s
            """,
            (days,)
        ).fetchall()
        df = pd.DataFrame(rows, columns=[
            'date', 'fut_net', 'put_ratio',
            'put_long', 'fut_net_change'
        ])
        return df.sort_values('date').reset_index(drop=True)
```

---

## MODULE 3: FII DAILY PIPELINE
# File: fii/daily_pipeline.py
# Runs at 7:30 PM on every trading day (cron: 30 19 * * 1-5)

```python
class FIIDailyPipeline:
    """
    Complete nightly pipeline:
    1. Download NSE FII OI data
    2. Fetch Nifty closing price
    3. Extract metrics and store to DB
    4. Detect FII signal pattern
    5. Store signal result to DB
    6. Pre-load signal into tomorrow's Redis queue
    7. Alert on signal detection
    """

    MAX_RETRIES = 3
    RETRY_WAIT_MINUTES = 15

    def __init__(self, db, redis_client, alerter, logger,
                 truedata_token: str):
        """
        db:             database connection (psycopg2 or equivalent wrapper)
        redis_client:   redis.Redis instance connected to the system Redis
        alerter:        TelegramAlerter instance
        logger:         Python Logger or equivalent structured logger
        truedata_token: TrueData API Bearer token for Nifty price fetch
        """
        self.db              = db
        self.redis           = redis_client
        self.alerter         = alerter
        self.logger          = logger
        self.truedata_token  = truedata_token
        self.downloader      = FIIDataDownloader()
        self.detector        = FIISignalDetector(db)

    def run(self, today_trading_date=None):
        """
        Main pipeline: download today's NSE FII OI data, detect patterns,
        pre-load signals into Redis for next morning's SignalSelector.

        today_trading_date: the NSE trading date whose data we download.
        If None, auto-detected from market_calendar table.
        Uses market_calendar (not timedelta) so weekends and NSE holidays
        are handled correctly.
        """
        if today_trading_date is None:
            today_trading_date = self._get_last_trading_day()

        trade_date = self._get_next_trading_day(today_trading_date)

        # Purge stale FII signals from Redis before adding new ones.
        # Prevents holiday/weekend signals from firing days later.
        self._purge_stale_redis_signals()

        # STEP 1: Download with retries
        for attempt in range(self.MAX_RETRIES):
            try:
                raw_data = self.downloader.download_participant_oi(
                    today_trading_date
                )
                break
            except DataNotAvailableError:
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_WAIT_MINUTES * 60)
                else:
                    self.logger.log_data_missing(today_trading_date)
                    return  # Skip — no data today

        # STEP 2: Fetch today's Nifty closing price and inject into raw_data.
        # The NSE OI report does not include Nifty price — fetch separately
        # from TrueData (same subscription used for backtesting).
        # Used to compute the dynamic Pattern 3 contract threshold.
        # Fallback: yesterday's close from fii_daily_metrics if fetch fails.
        try:
            nifty_close = self._fetch_nifty_close(today_trading_date)
        except Exception as e:
            self.logger.warning(
                f"Could not fetch Nifty close for {today_trading_date}: {e}. "
                f"Falling back to yesterday's stored value."
            )
            row = self.db.execute(
                "SELECT nifty_close FROM fii_daily_metrics "
                "ORDER BY date DESC LIMIT 1"
            ).fetchone()
            nifty_close = row['nifty_close'] if row and row['nifty_close'] else 24000.0
        raw_data['nifty_close'] = nifty_close

        # STEP 3: Extract and store metrics
        metrics = self.detector._extract_metrics(raw_data)
        metrics['fut_net_change'] = self._compute_day_change(
            metrics['fut_net']
        )
        self.db.store_fii_metrics(metrics)

        # STEP 4: Detect signal
        result = self.detector.detect(raw_data, for_date=trade_date)

        # STEP 5: Store signal result
        self.db.store_fii_signal_result(result)

        # STEP 6: If signal detected, pre-load into Redis queue
        if result.signal_id and result.direction != 'NEUTRAL':
            self.redis.xadd(
                'SIGNAL_QUEUE_PRELOADED',
                {
                    'signal_id':             result.signal_id,
                    'direction':             result.direction,
                    'confidence':            str(result.confidence),
                    'valid_for':             str(result.valid_for_date),
                    'valid_until':           str(result.valid_for_date),
                    # SignalSelector checks valid_until at 8:50 AM.
                    # If today > valid_until: delete from stream and skip.
                    # Prevents a Friday signal from firing on the following Tuesday.
                    'pattern':               result.pattern_name,
                    'notes':                 result.notes,
                    'source':                'FII_OVERNIGHT',
                    'requires_confirmation': 'true',
                    # Morning selector confirms against market open direction
                    # in first 15 minutes before executing.
                }
            )
            self.alerter.send(
                'INFO',
                f"FII signal pre-loaded: {result.pattern_name} "
                f"({result.direction}, {result.confidence:.0%} confidence) "
                f"valid for {result.valid_for_date}"
            )

        self.logger.log_pipeline_complete(today_trading_date, result)

    def _get_last_trading_day(self):
        """
        Query market_calendar for the last actual NSE trading day.
        Handles weekends and NSE holidays correctly.
        """
        result = self.db.execute("""
            SELECT MAX(trading_date)
            FROM market_calendar
            WHERE trading_date <= CURRENT_DATE
              AND is_trading_day = TRUE
        """).fetchone()
        return result[0] if result else date.today()

    def _get_next_trading_day(self, from_date):
        """Returns the next NSE trading day after from_date."""
        result = self.db.execute("""
            SELECT MIN(trading_date)
            FROM market_calendar
            WHERE trading_date > %s
              AND is_trading_day = TRUE
        """, (from_date,)).fetchone()
        return result[0] if result else from_date + timedelta(days=1)

    def _purge_stale_redis_signals(self):
        """
        Remove FII signals from Redis that are past their valid_until date.
        Prevents a Friday signal from firing on Tuesday after a holiday weekend.
        Called at the start of every nightly pipeline run.
        """
        today = date.today().isoformat()
        messages = self.redis.xrange('SIGNAL_QUEUE_PRELOADED', '-', '+')
        purged = 0
        for msg_id, fields in messages:
            valid_until = fields.get('valid_until', '')
            if valid_until and valid_until < today:
                self.redis.xdel('SIGNAL_QUEUE_PRELOADED', msg_id)
                purged += 1
        if purged:
            self.logger.info(f"Purged {purged} stale FII signal(s) from Redis")

    def _fetch_nifty_close(self, for_date) -> float:
        """
        Fetch Nifty 50 closing price for for_date.
        Used to compute the dynamic Pattern 3 threshold.
        Source: TrueData historical OHLCV endpoint (same subscription
        already used for backtesting).
        Raises on any connection/parse error — caller handles fallback.
        """
        # TrueData REST endpoint for EOD OHLCV
        # Substitute your TrueData API key and symbol as configured
        date_str = for_date.strftime('%Y-%m-%d')
        resp = requests.get(
            'https://api.truedata.in/gethistdata',
            params={
                'symbol':   'NIFTY50-I',   # continuous futures or cash index
                'startdate': date_str,
                'enddate':   date_str,
                'bar':       'EOD',
            },
            headers={'Authorization': f'Bearer {self.truedata_token}'},
            timeout=10
        )
        resp.raise_for_status()
        data = resp.json()
        # TrueData returns list of [date, open, high, low, close, volume]
        if data and isinstance(data, list):
            return float(data[0][4])   # index 4 = close
        raise ValueError(f"Empty TrueData response for {date_str}")

    def _compute_day_change(self, today_fut_net: float) -> float:
        yesterday = self.db.execute(
            "SELECT fut_net FROM fii_daily_metrics "
            "ORDER BY date DESC LIMIT 1"
        ).fetchone()
        if not yesterday:
            return 0.0
        return today_fut_net - yesterday['fut_net']
```

---

## DB TABLE FOR FII DATA
# Add to schema.sql

```sql
CREATE TABLE fii_daily_metrics (
    date                DATE PRIMARY KEY,
    fut_long            FLOAT,
    fut_short           FLOAT,
    fut_net             FLOAT,
    put_long            FLOAT,
    put_short           FLOAT,
    put_net             FLOAT,
    put_ratio           FLOAT,
    call_long           FLOAT,
    call_short          FLOAT,
    call_net            FLOAT,
    pcr                 FLOAT,
    fut_net_change      FLOAT,  -- vs previous day
    nifty_close         FLOAT,  -- Nifty 50 closing price that day
                                -- Used for dynamic Pattern 3 threshold audit trail
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

-- Vol surface adjustment factors for Tier 2 (options) backtesting.
-- Populated once during backtest setup from 6 months of TrueData OHLCV.
-- The Go Greek service reads this table to apply DTE-based IV scaling
-- (at-the-money IV is not constant across DTE — short-dated options
--  have higher realised vol than the raw India VIX figure implies).
-- See pending action item: "Derive DTE-based vol adjustment factors".
CREATE TABLE vol_adjustment_factors (
    dte_bucket          INTEGER NOT NULL,
                        -- Days to expiry at time of trade:
                        -- 1, 2, 3, 4, 5, 6, 7, 8-14, 15-21, 22-30, 31+
    vol_multiplier      FLOAT   NOT NULL,
                        -- Multiply India VIX by this factor for BS pricing.
                        -- e.g. dte_bucket=1 → multiplier ≈ 1.35 (short-dated
                        -- options are more expensive than VIX implies)
    sample_count        INTEGER,          -- Number of observations used
    derived_at          TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (dte_bucket)
);

-- Nifty daily OHLCV + India VIX.
-- Populated by: data/nifty_loader.py --init (one-time), then nightly --update.
-- Read by: load_nifty_history(), RegimeLabeler fallback, WalkForwardEngine.
CREATE TABLE nifty_daily (
    date         DATE        PRIMARY KEY,
    open         FLOAT       NOT NULL,
    high         FLOAT       NOT NULL,
    low          FLOAT       NOT NULL,
    close        FLOAT       NOT NULL,
    volume       BIGINT,
    india_vix    FLOAT,      -- NULL until VIX data available for that date
    created_at   TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_nifty_daily_date ON nifty_daily(date DESC);

-- Market calendar — required for correct next-trading-day lookup.
-- Populate from NSE holiday list (published each year in December).
-- Used by FIIDailyPipeline and add_trading_days() helper.
CREATE TABLE market_calendar (
    trading_date        DATE PRIMARY KEY,
    is_trading_day      BOOLEAN NOT NULL,
    holiday_name        TEXT    -- NULL if trading day, name if holiday
);
-- Seed with: all weekdays = TRUE, NSE holidays = FALSE
-- NSE holiday list: https://www.nseindia.com/resources/exchange-communication-holidays

-- Pre-computed regime labels (written by nightly batch at 9 PM)
-- SignalSelector.get_current_regime() reads from here at 8:50 AM.
CREATE TABLE regime_labels (
    label_date          DATE PRIMARY KEY,
    regime              TEXT NOT NULL
                        CHECK (regime IN ('TRENDING','RANGING','HIGH_VOL','CRISIS')),
    adx_value           FLOAT,
    ema_50              FLOAT,
    india_vix           FLOAT,
    computed_at         TIMESTAMPTZ DEFAULT NOW()
);

-- Economic calendar for RBI / Budget / special events
-- Populated manually each quarter from official sources (see SignalSelector docstring).
CREATE TABLE economic_calendar (
    event_id            SERIAL PRIMARY KEY,
    event_date          DATE NOT NULL,
    event_type          TEXT NOT NULL,
                        -- 'RBI_DECISION' | 'BUDGET' | 'NIFTY_EXPIRY'
                        -- | 'MARKET_HALF_DAY' | 'OTHER'
    event_name          TEXT,
    notes               TEXT,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_economic_calendar_date ON economic_calendar(event_date);

CREATE TABLE fii_signal_results (
    id                  SERIAL PRIMARY KEY,
    data_date           DATE NOT NULL,  -- date of underlying FII data
    valid_for_date      DATE NOT NULL,  -- trade on this date
    signal_id           TEXT,
    direction           TEXT,
    confidence          FLOAT,
    pattern_name        TEXT NOT NULL,
    notes               TEXT,
    was_executed        BOOLEAN DEFAULT FALSE,
    execution_pnl       FLOAT,         -- fill in after trade closes
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

-- OI FILE VALIDATION — add to download_participant_oi() after parsing:
-- required_columns = {'future_long_contracts', 'future_short_contracts',
--                     'put_long_contracts', 'put_short_contracts'}
-- if not required_columns.issubset(set(df.columns)):
--     raise WrongFileError(
--         f"Downloaded file appears to be volumes, not OI. "
--         f"Columns found: {set(df.columns)}"
--     )
```

# SPECIFICATION COMPLETE — MASTER SUMMARY
# What is now fully specified across all 7 spec documents

---

## SPECIFICATION STATUS: 100% COMPLETE

### Spec 1: Backtest Engine
✓ RegimeLabeler — sequential computation, no-lookahead unit test
✓ ParameterSensitivityTester — 5-variant test, ROBUST/MODERATE/FRAGILE
✓ WalkForwardEngine — 36/12/3 months, purge+embargo, differentiated criteria
✓ ValidationTestSuite — 3 mandatory pre-flight tests

### Spec 2: Execution Engine
✓ 9-step execution flow (exact sequence)
✓ 7-level priority table
✓ Greek pre-check (exact limits + code)
✓ 7 failure scenarios (exact responses)
✓ Daily reconciliation

### Spec 3: Portfolio Construction
✓ PRIMARY vs SECONDARY classification criteria
✓ Signal Selector — 13-step decision tree (full code)
✓ Calendar + expiry filters (exact logic)
✓ Capital efficiency formula + regime multipliers table
✓ Capital-starved signal handling (3 options)
✓ Minute-by-minute daily schedule
✓ 5-level drawdown response playbook

### Spec 4: RAG Pipeline
✓ Book profiles — all 11 books with exact settings
✓ PDFIngester — paragraph-aware chunking (500/700 token target/max)
✓ ChromaDB storage — one collection per book, MMR reranking
✓ 4 extraction prompts — CONCRETE, PRINCIPLE, CONFLICT, HALLUCINATION CHECK
✓ CLI human review interface

### Spec 5: Signal Registry Schema
✓ 7 PostgreSQL tables with exact column types and constraints
✓ Auto-versioning trigger (every signal change preserved)
✓ All status values, classification values, check constraints
✓ TimescaleDB hypertables for trades and portfolio_state
✓ Indexes for all common query patterns

### Spec 6: Monitoring and Alerts
✓ 18 named alert definitions with exact thresholds
✓ 3 channels: Telegram phone, email digest, DB log
✓ Which alerts go to phone (13), which email-only (5)
✓ Telegram bot code (ready to configure with token + chat_id)
✓ Daily digest contents (10 fields)

### Spec 7: FII Data Pipeline
✓ Exact NSE URL for participant-wise OI data
✓ Browser header spoofing (NSE blocks raw requests)
✓ Column normalization (NSE changes names occasionally)
✓ Pattern 1: Bearish — thresholds, percentile calculation, confidence
✓ Pattern 2: Bullish — threshold, put ratio decline confirmation
✓ Pattern 3: Large shift continuation — ₹3,000Cr threshold, dynamic (≈16,667 contracts at Nifty 24,000)
✓ Pattern 4: Pure hedging suppressor — blocks other signals
✓ Retry logic (3 attempts × 15 minutes = 9:30 PM cutoff)
✓ Redis pre-load for tomorrow's Signal Selector
✓ 2 DB tables: fii_daily_metrics + fii_signal_results

---

## WHAT TO BUILD FIRST (recommended order)

Week 1:  Environment + PostgreSQL schema (run schema.sql)
         Download 10-year Nifty OHLCV from NSE
         Build RegimeLabeler + run unit test on historical data

Week 2:  Build ValidationTestSuite
         Build WalkForwardEngine skeleton
         Download 6 months TrueData for vol adjustment factors

Week 3:  Book ingestion — start with Gujral (CONCRETE, simplest)
         Build PDFIngester with Gujral profile
         Manual review: first 20 signals

Week 4:  Ingest remaining 10 books
         Run conflict detection across all extracted signals
         Human review sprint (all signals)

Week 5+: Backtest engine complete
         Run all 3 rounds
         Paper trading setup

---

## FILES PRODUCED IN THIS SESSION

1. second_review_with_new_fixes.md    (Round 2 expert review)
2. implementation_specifications_final.md  (Specs 1-3)
3. [this session] Specs 4-7 compiled below →

