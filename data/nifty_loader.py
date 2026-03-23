"""
Nifty 50 OHLCV + India VIX data loader.
Owns all price data. Everything else reads from the DB.

Usage:
    python -m data.nifty_loader --init          # One-time: load 10-year history
    python -m data.nifty_loader --update        # Nightly: incremental update
    python -m data.nifty_loader --status        # Check what's loaded
"""

import pandas as pd
import requests
import io
import time
from pathlib import Path
from datetime import date, timedelta


# ================================================================
# NSE URL templates
# ================================================================
BHAV_URL_TEMPLATE = (
    "https://archives.nseindia.com/content/indices/"
    "ind_close_all_{date_str}.csv"
)

VIX_HISTORY_URL = (
    "https://archives.nseindia.com/content/vix/"
    "histdata/india_vix_history.csv"
)

NSE_HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/120.0.0.0 Safari/537.36'
    ),
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
}


def load_nifty_history(db, days=None):
    """
    Read Nifty daily OHLCV + VIX from the nifty_daily table.
    Returns DataFrame with columns:
        [date, open, high, low, close, volume, india_vix]
    sorted ascending by date.

    days: if set, return only the most recent N calendar days.
          If None, return full history.
    """
    cur = db.cursor()
    if days is not None:
        cutoff = date.today() - timedelta(days=days)
        cur.execute(
            """SELECT date, open, high, low, close, volume, india_vix
               FROM nifty_daily
               WHERE date >= %s
               ORDER BY date ASC""",
            (cutoff,)
        )
    else:
        cur.execute(
            """SELECT date, open, high, low, close, volume, india_vix
               FROM nifty_daily
               ORDER BY date ASC"""
        )

    rows = cur.fetchall()
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


def initial_load(db, start_year=2014):
    """
    One-time setup: download 10 years of Nifty OHLCV + VIX and insert
    into nifty_daily. Takes ~5-10 minutes depending on network.
    """
    print(f"Loading Nifty history from {start_year} to today...")

    # Step 1: Download VIX history
    print("Downloading India VIX history...")
    vix_df = _download_vix_history()
    print(f"  VIX: {len(vix_df)} records loaded")

    # Step 2: Download OHLCV via Bhavcopy
    print("Downloading Nifty OHLCV from NSE Bhavcopy archives...")
    print("  This will take several minutes (one request per trading day)...")
    ohlcv_df = _download_bhavcopy_range(
        start=date(start_year, 1, 1),
        end=date.today()
    )
    print(f"  OHLCV: {len(ohlcv_df)} records loaded")

    if ohlcv_df.empty:
        print("ERROR: No OHLCV data downloaded. Check network and NSE access.")
        return

    # Step 3: Merge on date
    merged = ohlcv_df.merge(vix_df, on='date', how='left')

    # Step 4: Insert to DB
    inserted = _bulk_insert(db, merged)
    print(f"Inserted {inserted} rows into nifty_daily.")

    # Step 5: Populate market_calendar from the dates we have
    _populate_market_calendar(db, start_year)
    print("Market calendar populated.")


def _download_vix_history():
    """
    Download India VIX full history via yfinance (ticker: ^INDIAVIX).

    WHY yfinance, not NSE directly:
    NSE removed the static CSV at archives.nseindia.com/content/vix/histdata/
    india_vix_history.csv (returns 404). The new NSE VIX page at
    nseindia.com/reports-indices-historical-vix requires a browser session
    cookie — plain requests.get() returns empty HTML. yfinance wraps Yahoo
    Finance's API, which has carried ^INDIAVIX data since 2008 and works
    reliably without session handling.

    Returns DataFrame with columns: [date (datetime), india_vix (float)]
    """
    try:
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
        # yfinance returns 'Date' as a timezone-aware datetime — normalise
        df['date'] = pd.to_datetime(df['Date']).dt.tz_localize(None).dt.normalize()
        df['india_vix'] = pd.to_numeric(df['Close'], errors='coerce')
        result = df[['date', 'india_vix']].dropna()
        print(f"  VIX: {len(result)} records loaded via yfinance (^INDIAVIX)")
        return result
    except Exception as e:
        print(f"  WARNING: Could not download VIX history: {e}")
        print("  VIX values will be NULL in nifty_daily (acceptable for initial setup)")
        return pd.DataFrame(columns=['date', 'india_vix'])


def _download_bhavcopy_range(start, end):
    """
    Download NSE Bhavcopy index files for every trading day in range.
    Filters for 'Nifty 50' row.
    Returns DataFrame: [date, open, high, low, close, volume].
    """
    records = []
    current = start
    total_days = (end - start).days
    downloaded = 0
    errors = 0

    while current <= end:
        # Skip weekends
        if current.weekday() >= 5:
            current += timedelta(days=1)
            continue

        date_str = current.strftime('%d%m%Y')
        url = (
            f"https://archives.nseindia.com/content/indices/"
            f"ind_close_all_{date_str}.csv"
        )

        try:
            resp = requests.get(url, headers=NSE_HEADERS, timeout=15)
            if resp.status_code == 200 and len(resp.content) > 200:
                df = pd.read_csv(io.StringIO(resp.text))
                # Find Nifty 50 row — column name varies
                idx_col = [c for c in df.columns if 'index' in c.lower() and 'name' in c.lower()]
                if idx_col:
                    nifty50 = df[df[idx_col[0]].str.strip() == 'Nifty 50']
                else:
                    nifty50 = df[df.iloc[:, 0].str.strip() == 'Nifty 50']

                if not nifty50.empty:
                    row = nifty50.iloc[0]
                    records.append({
                        'date': pd.to_datetime(current),
                        'open': _safe_float(row, 'Open'),
                        'high': _safe_float(row, 'High'),
                        'low': _safe_float(row, 'Low'),
                        'close': _safe_float(row, 'Closing'),
                        'volume': _safe_int(row, 'Volume'),
                    })
                    downloaded += 1

            # Progress indicator every 100 days
            if downloaded > 0 and downloaded % 100 == 0:
                print(f"  Downloaded {downloaded} trading days...")

        except Exception:
            errors += 1

        current += timedelta(days=1)
        time.sleep(0.3)  # NSE rate limit

    if errors > 0:
        print(f"  Skipped {errors} days (holidays/errors)")

    return pd.DataFrame(records)


def _safe_float(row, col_hint):
    """Extract float from row, searching for column by hint."""
    for col in row.index:
        if col_hint.lower() in col.lower():
            val = str(row[col]).replace(',', '').strip()
            try:
                return float(val)
            except (ValueError, TypeError):
                return 0.0
    return 0.0


def _safe_int(row, col_hint):
    """Extract int from row, searching for column by hint."""
    for col in row.index:
        if col_hint.lower().replace(' ', '') in col.lower().replace(' ', ''):
            val = str(row[col]).replace(',', '').strip()
            try:
                return int(float(val))
            except (ValueError, TypeError):
                return 0
    return 0


def _bulk_insert(db, df):
    """Insert merged OHLCV+VIX DataFrame into nifty_daily.

    On conflict, update OHLCV and backfill india_vix if currently NULL.
    This ensures VIX data that arrives later (e.g. yfinance lag) gets filled in.
    """
    from psycopg2.extras import execute_values

    rows = []
    for _, row in df.iterrows():
        rows.append((
            row['date'].date() if hasattr(row['date'], 'date') else row['date'],
            row['open'], row['high'], row['low'], row['close'],
            int(row['volume']) if pd.notna(row.get('volume')) else None,
            float(row['india_vix']) if pd.notna(row.get('india_vix')) else None,
        ))

    if not rows:
        return 0

    cur = db.cursor()
    execute_values(
        cur,
        """INSERT INTO nifty_daily (date, open, high, low, close, volume, india_vix)
           VALUES %s
           ON CONFLICT (date) DO UPDATE SET
               open = COALESCE(EXCLUDED.open, nifty_daily.open),
               high = COALESCE(EXCLUDED.high, nifty_daily.high),
               low = COALESCE(EXCLUDED.low, nifty_daily.low),
               close = COALESCE(EXCLUDED.close, nifty_daily.close),
               volume = COALESCE(EXCLUDED.volume, nifty_daily.volume),
               india_vix = COALESCE(EXCLUDED.india_vix, nifty_daily.india_vix)""",
        rows
    )
    db.commit()
    return len(rows)


def _populate_market_calendar(db, start_year):
    """
    Populate market_calendar table.
    All weekdays = trading days, weekends = non-trading.
    NSE holidays must be manually marked later.
    """
    from psycopg2.extras import execute_values

    cur = db.cursor()
    start = date(start_year, 1, 1)
    end = date.today() + timedelta(days=365)  # one year ahead
    current = start
    rows = []

    while current <= end:
        is_trading = current.weekday() < 5  # Mon-Fri
        rows.append((current, is_trading, None))
        current += timedelta(days=1)

    execute_values(
        cur,
        """INSERT INTO market_calendar (trading_date, is_trading_day, holiday_name)
           VALUES %s
           ON CONFLICT (trading_date) DO NOTHING""",
        rows
    )
    db.commit()


def _fetch_vix_from_kite(from_date, to_date):
    """
    Fetch India VIX daily close via Kite historical data API.

    Uses instrument token 264969 (India VIX on NSE).
    Returns DataFrame with columns: [date (datetime), india_vix (float)].
    """
    try:
        from data.kite_auth import get_kite
        kite = get_kite()
        if not kite:
            print("  VIX via Kite: not authenticated, skipping")
            return pd.DataFrame(columns=['date', 'india_vix'])

        VIX_TOKEN = 264969
        from datetime import datetime
        from_dt = datetime.combine(from_date, datetime.min.time())
        to_dt = datetime.combine(to_date, datetime.min.time()) + timedelta(hours=23)

        data = kite.historical_data(
            instrument_token=VIX_TOKEN,
            from_date=from_dt,
            to_date=to_dt,
            interval='day',
        )

        if not data:
            print("  VIX via Kite: no data returned")
            return pd.DataFrame(columns=['date', 'india_vix'])

        records = []
        for candle in data:
            dt = candle['date']
            if hasattr(dt, 'tzinfo') and dt.tzinfo:
                dt = dt.replace(tzinfo=None)
            records.append({
                'date': pd.to_datetime(dt).normalize(),
                'india_vix': float(candle['close']),
            })

        df = pd.DataFrame(records)
        print(f"  VIX via Kite: {len(df)} records fetched")
        return df

    except Exception as e:
        print(f"  VIX via Kite failed: {e}")
        return pd.DataFrame(columns=['date', 'india_vix'])


def _fetch_vix_best_effort(from_date, to_date):
    """
    Fetch India VIX data, trying Kite first, then yfinance as fallback.

    Returns DataFrame with columns: [date (datetime), india_vix (float)].
    """
    # Try Kite first (more reliable for recent dates)
    vix_df = _fetch_vix_from_kite(from_date, to_date)
    if not vix_df.empty:
        return vix_df

    # Fall back to yfinance
    print("  Trying yfinance for VIX...")
    vix_df = _download_vix_history()
    if not vix_df.empty:
        # Filter to requested date range
        vix_df = vix_df[
            (vix_df['date'] >= pd.to_datetime(from_date)) &
            (vix_df['date'] <= pd.to_datetime(to_date))
        ]
    return vix_df


def incremental_update(db):
    """
    Nightly update: download yesterday's Bhavcopy and VIX, insert if missing.
    Also backfills VIX for any recent rows that have NULL india_vix.
    Called by cron at 6:00 PM daily.
    """
    yesterday = date.today() - timedelta(days=1)

    # Skip weekends
    if yesterday.weekday() >= 5:
        print(f"Skipping {yesterday} (weekend)")
        # Still try to backfill VIX for recent NULL rows
        backfill_vix(db, days=10)
        return

    cur = db.cursor()
    cur.execute("SELECT 1 FROM nifty_daily WHERE date = %s", (yesterday,))
    if cur.fetchone():
        print(f"OHLCV for {yesterday} already loaded.")
    else:
        ohlcv = _download_bhavcopy_range(yesterday, yesterday)
        vix = _fetch_vix_best_effort(yesterday, yesterday)
        if not ohlcv.empty:
            merged = ohlcv.merge(vix, on='date', how='left')
            inserted = _bulk_insert(db, merged)
            print(f"Inserted {inserted} row(s) for {yesterday}")
        else:
            print(f"No data available for {yesterday} (likely holiday)")

    # Always try to backfill VIX for recent rows with NULL india_vix
    backfill_vix(db, days=10)


def backfill_vix(db, days=10):
    """
    Backfill india_vix for recent nifty_daily rows where it is NULL.

    Tries Kite historical API first, then yfinance as fallback.
    Updates existing rows in-place (does not insert new rows).

    Parameters
    ----------
    db : psycopg2 connection
    days : int
        Look back this many calendar days for NULL VIX rows.
    """
    cur = db.cursor()
    cur.execute(
        """SELECT date FROM nifty_daily
           WHERE india_vix IS NULL AND date >= %s
           ORDER BY date""",
        (date.today() - timedelta(days=days),)
    )
    null_dates = [r[0] for r in cur.fetchall()]

    if not null_dates:
        print("VIX backfill: no NULL rows found — nothing to do.")
        return

    print(f"VIX backfill: {len(null_dates)} rows with NULL india_vix: "
          f"{null_dates[0]} to {null_dates[-1]}")

    from_date = null_dates[0] - timedelta(days=1)  # one extra day for safety
    to_date = null_dates[-1] + timedelta(days=1)

    vix_df = _fetch_vix_best_effort(from_date, to_date)

    if vix_df.empty:
        print("VIX backfill: no VIX data available from any source.")
        return

    # Normalize dates for matching
    vix_df['date_norm'] = pd.to_datetime(vix_df['date']).dt.date
    vix_map = dict(zip(vix_df['date_norm'], vix_df['india_vix']))

    updated = 0
    for d in null_dates:
        vix_val = vix_map.get(d)
        if vix_val is not None and not pd.isna(vix_val) and float(vix_val) > 0:
            cur.execute(
                "UPDATE nifty_daily SET india_vix = %s WHERE date = %s",
                (float(vix_val), d)
            )
            updated += 1
            print(f"  Updated {d}: india_vix = {float(vix_val):.2f}")

    db.commit()
    print(f"VIX backfill: updated {updated}/{len(null_dates)} rows.")


def check_status(db):
    """Print summary of loaded data."""
    cur = db.cursor()

    cur.execute("SELECT COUNT(*), MIN(date), MAX(date) FROM nifty_daily")
    count, min_date, max_date = cur.fetchone()
    print(f"nifty_daily: {count} rows, {min_date} to {max_date}")

    cur.execute("SELECT COUNT(*) FROM nifty_daily WHERE india_vix IS NOT NULL")
    vix_count = cur.fetchone()[0]
    print(f"  With VIX data: {vix_count} rows")

    cur.execute("SELECT COUNT(*) FROM market_calendar")
    cal_count = cur.fetchone()[0]
    print(f"market_calendar: {cal_count} rows")

    cur.execute("SELECT COUNT(*) FROM regime_labels")
    regime_count = cur.fetchone()[0]
    print(f"regime_labels: {regime_count} rows")


# ================================================================
# CLI ENTRY POINT
# ================================================================
if __name__ == '__main__':
    import argparse
    import psycopg2
    from config.settings import DATABASE_DSN

    parser = argparse.ArgumentParser(description='Nifty price data loader')
    parser.add_argument('--init', action='store_true',
                        help='Load full 10-year history (one-time setup)')
    parser.add_argument('--update', action='store_true',
                        help='Incremental update (run nightly)')
    parser.add_argument('--status', action='store_true',
                        help='Check loaded data status')
    parser.add_argument('--backfill-vix', type=int, nargs='?', const=10,
                        metavar='DAYS',
                        help='Backfill NULL india_vix for last N days (default 10)')
    parser.add_argument('--dsn', default=DATABASE_DSN,
                        help='PostgreSQL DSN')
    args = parser.parse_args()

    conn = psycopg2.connect(args.dsn)
    conn.autocommit = False

    if args.init:
        initial_load(conn)
    elif args.update:
        incremental_update(conn)
    elif args.backfill_vix is not None:
        backfill_vix(conn, days=args.backfill_vix)
    elif args.status:
        check_status(conn)
    else:
        parser.print_help()

    conn.close()
