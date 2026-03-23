"""
Cross-asset data loader for overlay signals.

Downloads USDINR, crude, gold, US indices, Asian indices, US yields
via yfinance. Computes derived features (returns, z-scores, momentum).

Usage:
    python -m data.cross_asset_loader init      # backfill 5 years
    python -m data.cross_asset_loader update    # today's data
    python -m data.cross_asset_loader status    # data quality check
"""

import argparse
import logging
import time

import numpy as np
import pandas as pd
import psycopg2
import yfinance as yf
from psycopg2.extras import execute_values

from config.settings import DATABASE_DSN

logger = logging.getLogger(__name__)

INSTRUMENTS = {
    'USDINR':       'USDINR=X',
    'CRUDE_WTI':    'CL=F',
    'CRUDE_BRENT':  'BZ=F',
    'GOLD':         'GC=F',
    'SP500':        '^GSPC',
    'VIX_US':       '^VIX',
    'NIKKEI':       '^N225',
    'HANGSENG':     '^HSI',
    'US10Y':        '^TNX',
}


def download_all(period='5y'):
    """Download all instruments, compute features, return dict of DataFrames."""
    all_data = {}

    for name, ticker in INSTRUMENTS.items():
        for attempt in range(3):
            try:
                df = yf.download(ticker, period=period, progress=False)
                if df.columns.nlevels > 1:
                    df = df.droplevel('Ticker', axis=1)
                if not df.empty:
                    df = df.reset_index()
                    df.columns = [c.lower() for c in df.columns]
                    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None).dt.date
                    df = df[['date', 'close']].dropna()
                    df = compute_features(df)
                    df['instrument'] = name
                    all_data[name] = df
                    break
            except Exception as e:
                if attempt == 2:
                    logger.warning(f"{name}: failed after 3 attempts: {e}")
            time.sleep(0.5)

    return all_data


def compute_features(df):
    """Compute derived features for a single instrument."""
    df = df.copy().sort_values('date').reset_index(drop=True)
    close = df['close']

    df['daily_return'] = close.pct_change(1)
    df['weekly_return'] = close.pct_change(5)
    df['monthly_return'] = close.pct_change(20)

    sma_20 = close.rolling(20).mean()
    std_20 = close.rolling(20).std()
    df['zscore_20'] = (close - sma_20) / std_20.replace(0, np.nan)

    sma_60 = close.rolling(60).mean()
    std_60 = close.rolling(60).std()
    df['zscore_60'] = (close - sma_60) / std_60.replace(0, np.nan)

    df['momentum_5d'] = np.sign(close.pct_change(5))

    return df


def save_to_db(conn, all_data):
    """Save all instruments to cross_asset_daily table."""
    cur = conn.cursor()
    total = 0

    for name, df in all_data.items():
        rows = [
            (r['date'], name, r['close'], r.get('daily_return'),
             r.get('weekly_return'), r.get('monthly_return'),
             r.get('zscore_20'), r.get('zscore_60'), r.get('momentum_5d'))
            for _, r in df.iterrows()
            if pd.notna(r['close'])
        ]
        execute_values(cur, """
            INSERT INTO cross_asset_daily (date, instrument, close,
                daily_return, weekly_return, monthly_return,
                zscore_20, zscore_60, momentum_5d)
            VALUES %s ON CONFLICT (date, instrument) DO UPDATE SET
                close = EXCLUDED.close, daily_return = EXCLUDED.daily_return
        """, rows, page_size=500)
        total += len(rows)

    conn.commit()
    return total


def load_for_date(conn, trade_date):
    """Load cross-asset data for a specific date. Returns dict."""
    cur = conn.cursor()
    cur.execute("""
        SELECT instrument, close, daily_return, weekly_return,
               monthly_return, zscore_20, zscore_60, momentum_5d
        FROM cross_asset_daily WHERE date = %s
    """, (trade_date,))

    data = {}
    for row in cur.fetchall():
        data[row[0]] = {
            'close': row[1], 'daily_return': row[2], 'weekly_return': row[3],
            'monthly_return': row[4], 'zscore_20': row[5], 'zscore_60': row[6],
            'momentum_5d': row[7],
        }
    return data


def load_previous_day(conn, trade_date):
    """Load previous trading day's data (for US overnight signal)."""
    cur = conn.cursor()
    cur.execute("""
        SELECT instrument, close, daily_return, weekly_return
        FROM cross_asset_daily
        WHERE date < %s
        ORDER BY date DESC LIMIT 20
    """, (trade_date,))

    # Group by instrument, take most recent
    data = {}
    for row in cur.fetchall():
        if row[0] not in data:
            data[row[0]] = {
                'close': row[1], 'daily_return': row[2], 'weekly_return': row[3],
            }
    return data


def check_status(conn):
    """Print data quality report."""
    cur = conn.cursor()
    cur.execute("""
        SELECT instrument, COUNT(*), MIN(date), MAX(date)
        FROM cross_asset_daily
        GROUP BY instrument ORDER BY instrument
    """)
    rows = cur.fetchall()
    if not rows:
        print("cross_asset_daily: EMPTY")
        return

    print("Cross-Asset Data Quality:")
    print(f"{'Instrument':<14s} {'Days':>6s} {'From':>12s} {'To':>12s}")
    print("-" * 50)
    for inst, count, mn, mx in rows:
        print(f"{inst:<14s} {count:>6d} {str(mn):>12s} {str(mx):>12s}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cross-asset data loader')
    parser.add_argument('command', choices=['init', 'update', 'status'])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    conn = psycopg2.connect(DATABASE_DSN)

    if args.command == 'init':
        print("Downloading 5 years of cross-asset data...")
        data = download_all(period='5y')
        total = save_to_db(conn, data)
        print(f"Saved {total:,} rows across {len(data)} instruments")
        check_status(conn)

    elif args.command == 'update':
        data = download_all(period='5d')
        total = save_to_db(conn, data)
        print(f"Updated {total} rows")

    elif args.command == 'status':
        check_status(conn)

    conn.close()
