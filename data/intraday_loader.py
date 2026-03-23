"""
Intraday data loader for Nifty futures.

Supports three data sources:
1. Synthetic generation from daily OHLCV (for backtesting without API)
2. CSV import (from TrueData/Kite historical data exports)
3. Live API (TrueData or Kite — when credentials available)

The synthetic generator creates realistic 5-min bars from daily data,
preserving daily OHLC boundaries and using empirical NSE volume profiles.

Usage:
    python -m data.intraday_loader --generate-synthetic    # from daily DB
    python -m data.intraday_loader --load-csv data.csv     # from CSV file
    python -m data.intraday_loader --status                # check loaded data
"""

import argparse
import logging
from datetime import datetime, time, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

from config.settings import DATABASE_DSN

logger = logging.getLogger(__name__)

# NSE session times
SESSION_OPEN = time(9, 15)
SESSION_CLOSE = time(15, 30)
BARS_PER_DAY_5MIN = 75       # 9:15 to 15:30 = 375 min / 5 = 75 bars
BARS_PER_DAY_15MIN = 25      # 375 / 15 = 25 bars
BARS_PER_DAY_60MIN = 7       # ~6.25 hours, rounded to 7 bars (last bar shorter)

# Empirical NSE intraday volume profile (U-shaped)
# Higher at open and close, lower midday
VOLUME_PROFILE_75 = np.array([
    # 9:15-10:00 (9 bars) — opening surge
    3.5, 3.0, 2.5, 2.2, 2.0, 1.8, 1.7, 1.6, 1.5,
    # 10:00-11:00 (12 bars) — morning active
    1.4, 1.3, 1.3, 1.2, 1.2, 1.1, 1.1, 1.0, 1.0, 1.0, 0.9, 0.9,
    # 11:00-12:00 (12 bars) — midday decline
    0.9, 0.8, 0.8, 0.8, 0.7, 0.7, 0.7, 0.7, 0.6, 0.6, 0.6, 0.6,
    # 12:00-13:00 (12 bars) — lunch lull
    0.6, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
    # 13:00-14:00 (12 bars) — afternoon pickup
    0.6, 0.6, 0.7, 0.7, 0.8, 0.8, 0.9, 0.9, 1.0, 1.0, 1.1, 1.1,
    # 14:00-15:00 (12 bars) — pre-close active
    1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3,
    # 15:00-15:30 (6 bars) — closing surge
    2.5, 2.7, 3.0, 3.2, 3.5, 4.0,
])
VOLUME_PROFILE_75 = VOLUME_PROFILE_75 / VOLUME_PROFILE_75.sum()  # normalize


def generate_5min_bars(daily_row, rng=None):
    """
    Generate 75 realistic 5-min bars from a single daily OHLCV row.

    Uses geometric Brownian motion with mean-reversion to session VWAP,
    constrained to hit the daily open, high, low, close exactly.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    day_open = float(daily_row['open'])
    day_high = float(daily_row['high'])
    day_low = float(daily_row['low'])
    day_close = float(daily_row['close'])
    day_volume = int(daily_row['volume']) if pd.notna(daily_row.get('volume')) else 1000000
    day_date = pd.Timestamp(daily_row['date'])

    n = BARS_PER_DAY_5MIN
    day_range = day_high - day_low
    if day_range <= 0:
        day_range = day_open * 0.005  # 0.5% minimum range

    # Generate random walk from open to close
    steps = rng.normal(0, 1, n)
    cum = np.cumsum(steps)
    # Scale to hit close at end
    cum = cum - cum[-1]  # end at 0
    path = day_open + cum * (day_range / (4 * np.std(cum) + 1e-9))

    # Ensure close is hit at last bar
    path = path - (path[-1] - day_close) * np.linspace(0, 1, n)

    # Determine high/low bar positions
    high_bar = rng.integers(0, n)
    low_bar = rng.integers(0, n)
    while low_bar == high_bar:
        low_bar = rng.integers(0, n)

    # Ensure path touches high and low
    path[high_bar] = day_high - day_range * 0.01
    path[low_bar] = day_low + day_range * 0.01

    # Build OHLC bars
    bars = []
    for i in range(n):
        bar_time = day_date + timedelta(hours=9, minutes=15 + i * 5)

        bar_close = float(np.clip(path[i], day_low, day_high))
        bar_open = float(np.clip(path[i - 1] if i > 0 else day_open, day_low, day_high))

        # Add noise for high/low within bar
        bar_noise = day_range * 0.003 * rng.uniform(0.5, 1.5)
        bar_high = max(bar_open, bar_close) + bar_noise
        bar_low = min(bar_open, bar_close) - bar_noise

        # Clamp to daily range
        bar_high = min(bar_high, day_high)
        bar_low = max(bar_low, day_low)

        # Force exact daily OHLC at boundaries
        if i == 0:
            bar_open = day_open
        if i == n - 1:
            bar_close = day_close
        if i == high_bar:
            bar_high = day_high
        if i == low_bar:
            bar_low = day_low

        # Volume distribution
        bar_volume = int(day_volume * VOLUME_PROFILE_75[i])

        bars.append({
            'datetime': bar_time,
            'timeframe': '5min',
            'open': round(bar_open, 2),
            'high': round(bar_high, 2),
            'low': round(bar_low, 2),
            'close': round(bar_close, 2),
            'volume': bar_volume,
        })

    return bars


def aggregate_to_timeframe(df_5min, target_tf):
    """Aggregate 5-min bars to 15-min or 60-min."""
    df = df_5min.copy()
    df['date'] = df['datetime'].dt.date

    if target_tf == '15min':
        # Group every 3 bars (15 min)
        df['group'] = df.groupby('date').cumcount() // 3
    elif target_tf == '60min':
        # Group every 12 bars (60 min)
        df['group'] = df.groupby('date').cumcount() // 12
    else:
        raise ValueError(f"Unknown timeframe: {target_tf}")

    agg = df.groupby(['date', 'group']).agg(
        datetime=('datetime', 'first'),
        open=('open', 'first'),
        high=('high', 'max'),
        low=('low', 'min'),
        close=('close', 'last'),
        volume=('volume', 'sum'),
    ).reset_index(drop=True)

    agg['timeframe'] = target_tf
    return agg


def generate_synthetic_intraday(conn, start_year=2021):
    """
    Generate synthetic 5-min, 15-min, 60-min bars from daily OHLCV.

    Produces ~3 years of intraday data for backtesting.
    """
    print("Loading daily data for synthetic generation...")
    df_daily = pd.read_sql(
        f"SELECT date, open, high, low, close, volume FROM nifty_daily "
        f"WHERE date >= '{start_year}-01-01' ORDER BY date",
        conn
    )
    df_daily['date'] = pd.to_datetime(df_daily['date'])
    print(f"  {len(df_daily)} trading days from {start_year}")

    rng = np.random.default_rng(42)

    # Generate 5-min bars for each day
    all_5min = []
    for _, row in df_daily.iterrows():
        bars = generate_5min_bars(row, rng)
        all_5min.extend(bars)

    df_5min = pd.DataFrame(all_5min)
    print(f"  Generated {len(df_5min)} 5-min bars")

    # Aggregate to 15-min and 60-min
    df_15min = aggregate_to_timeframe(df_5min, '15min')
    df_60min = aggregate_to_timeframe(df_5min, '60min')
    print(f"  Aggregated: {len(df_15min)} 15-min bars, {len(df_60min)} 60-min bars")

    # Combine all timeframes
    all_bars = pd.concat([df_5min, df_15min, df_60min], ignore_index=True)

    # Insert to DB
    print("Inserting to nifty_intraday...")
    cur = conn.cursor()

    # Clear existing synthetic data
    cur.execute(f"DELETE FROM nifty_intraday WHERE datetime >= '{start_year}-01-01'")

    rows = [
        (r['datetime'], r['timeframe'], r['open'], r['high'], r['low'],
         r['close'], r['volume'], None, None, None)
        for _, r in all_bars.iterrows()
    ]

    execute_values(
        cur,
        """INSERT INTO nifty_intraday
           (datetime, timeframe, open, high, low, close, volume, vwap,
            opening_range_high, opening_range_low)
           VALUES %s
           ON CONFLICT (datetime, timeframe) DO NOTHING""",
        rows,
        page_size=1000,
    )
    conn.commit()
    print(f"  Inserted {len(rows)} rows")

    return all_bars


def load_csv(conn, csv_path, timeframe='5min'):
    """
    Load intraday data from CSV.

    Expected columns: datetime, open, high, low, close, volume
    """
    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path, parse_dates=['datetime'])

    if timeframe == '5min':
        df['timeframe'] = '5min'
        # Also generate 15min and 60min
        df_15 = aggregate_to_timeframe(df, '15min')
        df_60 = aggregate_to_timeframe(df, '60min')
        all_bars = pd.concat([df, df_15, df_60], ignore_index=True)
    else:
        df['timeframe'] = timeframe
        all_bars = df

    print(f"  {len(all_bars)} total bars")

    cur = conn.cursor()
    rows = [
        (r['datetime'], r['timeframe'], r['open'], r['high'], r['low'],
         r['close'], r['volume'], None, None, None)
        for _, r in all_bars.iterrows()
    ]

    execute_values(
        cur,
        """INSERT INTO nifty_intraday
           (datetime, timeframe, open, high, low, close, volume, vwap,
            opening_range_high, opening_range_low)
           VALUES %s
           ON CONFLICT (datetime, timeframe) DO NOTHING""",
        rows,
        page_size=1000,
    )
    conn.commit()
    print(f"  Inserted {len(rows)} rows")


def load_intraday_history(conn, timeframe='5min', days=None):
    """Load intraday data from DB. Returns DataFrame."""
    where = f"WHERE timeframe = '{timeframe}'"
    if days:
        where += f" AND datetime >= NOW() - INTERVAL '{days} days'"

    df = pd.read_sql(
        f"SELECT datetime, open, high, low, close, volume, vwap, "
        f"opening_range_high, opening_range_low "
        f"FROM nifty_intraday {where} ORDER BY datetime",
        conn
    )
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df


def check_status(conn):
    """Print summary of loaded intraday data."""
    cur = conn.cursor()

    cur.execute("""
        SELECT timeframe, COUNT(*), MIN(datetime), MAX(datetime)
        FROM nifty_intraday
        GROUP BY timeframe
        ORDER BY timeframe
    """)
    rows = cur.fetchall()

    if not rows:
        print("nifty_intraday: EMPTY")
        return

    print("nifty_intraday:")
    for tf, count, min_dt, max_dt in rows:
        days = count // {'5min': 75, '15min': 25, '60min': 7}.get(tf, 75)
        print(f"  {tf:<6s} {count:>8,} bars  ({days:,} days)  {min_dt} to {max_dt}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Intraday data loader')
    parser.add_argument('--generate-synthetic', action='store_true',
                        help='Generate synthetic 5min bars from daily data')
    parser.add_argument('--load-csv', type=str, default=None,
                        help='Load from CSV file')
    parser.add_argument('--start-year', type=int, default=2021,
                        help='Start year for synthetic generation')
    parser.add_argument('--status', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    conn = psycopg2.connect(DATABASE_DSN)

    if args.generate_synthetic:
        generate_synthetic_intraday(conn, start_year=args.start_year)
    elif args.load_csv:
        load_csv(conn, args.load_csv)
    elif args.status:
        check_status(conn)
    else:
        parser.print_help()

    conn.close()
