"""
Fetch historical 5-min OHLCV data from Kite Connect API.

Kite limits:
  - 5-min candles: max 60 days per request
  - Rate limit: 3 requests/second
  - Historical data available for ~2 years

Usage:
    # Fetch last 1 year of Nifty 5-min data:
    python -m data.kite_historical

    # Fetch specific date range:
    python -m data.kite_historical --from 2025-01-01 --to 2026-03-21

    # Fetch BankNifty too:
    python -m data.kite_historical --banknifty

    # Check what's loaded:
    python -m data.kite_historical --status
"""

import argparse
import logging
import time
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

from data.kite_auth import get_kite
from config.settings import DATABASE_DSN

logger = logging.getLogger(__name__)

# Kite instrument tokens (NSE indices)
INSTRUMENTS = {
    'NIFTY': {
        'token': 256265,          # NIFTY 50 index
        'exchange': 'NSE',
        'table': 'nifty_intraday',
    },
    'BANKNIFTY': {
        'token': 260105,          # BANK NIFTY index
        'exchange': 'NSE',
        'table': 'banknifty_intraday',
    },
}

# Kite API limits
MAX_DAYS_PER_REQUEST = 55  # 5-min candles: max 60 days, use 55 for safety
RATE_LIMIT_SLEEP = 0.4     # seconds between requests (3 req/sec limit)


def fetch_candles(kite, instrument_token, from_date, to_date, interval='5minute'):
    """Fetch candles from Kite historical data API."""
    # Kite expects datetime, not date
    if isinstance(from_date, date) and not isinstance(from_date, datetime):
        from_date = datetime.combine(from_date, datetime.min.time())
    if isinstance(to_date, date) and not isinstance(to_date, datetime):
        to_date = datetime.combine(to_date, datetime.max.time().replace(microsecond=0))

    data = kite.historical_data(
        instrument_token=instrument_token,
        from_date=from_date,
        to_date=to_date,
        interval=interval,
    )
    return data


def fetch_and_store(kite, instrument_name, from_date, to_date, conn):
    """Fetch historical data in chunks and store to DB."""
    config = INSTRUMENTS[instrument_name]
    table = config['table']
    token = config['token']

    print(f"\nFetching {instrument_name} 5-min data: {from_date} to {to_date}")

    # Split into chunks of MAX_DAYS_PER_REQUEST
    current = from_date
    total_bars = 0
    chunk_num = 0

    while current < to_date:
        chunk_end = min(current + timedelta(days=MAX_DAYS_PER_REQUEST), to_date)
        chunk_num += 1

        try:
            data = fetch_candles(kite, token, current, chunk_end, '5minute')

            if data:
                rows = []
                for candle in data:
                    dt = candle['date']
                    # Kite returns datetime with timezone — strip it for DB
                    if isinstance(dt, str):
                        dt = datetime.strptime(dt, '%Y-%m-%d %H:%M:%S')
                    elif hasattr(dt, 'tzinfo') and dt.tzinfo is not None:
                        dt = dt.replace(tzinfo=None)

                    rows.append((
                        dt,
                        '5min',
                        float(candle['open']),
                        float(candle['high']),
                        float(candle['low']),
                        float(candle['close']),
                        int(candle['volume']),
                        None, None, None,  # vwap, opening_range_high, opening_range_low
                    ))

                if rows:
                    cur = conn.cursor()
                    execute_values(
                        cur,
                        f"""INSERT INTO {table}
                            (datetime, timeframe, open, high, low, close, volume,
                             vwap, opening_range_high, opening_range_low)
                            VALUES %s
                            ON CONFLICT (datetime, timeframe) DO UPDATE SET
                                open = EXCLUDED.open, high = EXCLUDED.high,
                                low = EXCLUDED.low, close = EXCLUDED.close,
                                volume = EXCLUDED.volume""",
                        rows,
                        page_size=1000,
                    )
                    conn.commit()
                    total_bars += len(rows)

                print(f"  Chunk {chunk_num}: {current} to {chunk_end} "
                      f"— {len(data)} bars (total: {total_bars:,})", flush=True)
            else:
                print(f"  Chunk {chunk_num}: {current} to {chunk_end} — no data")

        except Exception as e:
            print(f"  Chunk {chunk_num}: ERROR — {e}")
            conn.rollback()

        current = chunk_end + timedelta(days=1)
        time.sleep(RATE_LIMIT_SLEEP)  # Rate limit

    print(f"  Done: {total_bars:,} total bars fetched for {instrument_name}")
    return total_bars


def ensure_table(conn, table_name):
    """Create intraday table if it doesn't exist (for BankNifty)."""
    cur = conn.cursor()
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            datetime    TIMESTAMP NOT NULL,
            timeframe   VARCHAR(5) NOT NULL,
            open        FLOAT NOT NULL,
            high        FLOAT NOT NULL,
            low         FLOAT NOT NULL,
            close       FLOAT NOT NULL,
            volume      BIGINT,
            vwap        FLOAT,
            opening_range_high FLOAT,
            opening_range_low  FLOAT,
            created_at  TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (datetime, timeframe)
        );
        CREATE INDEX IF NOT EXISTS idx_{table_name}_tf
            ON {table_name}(timeframe, datetime DESC);
    """)
    conn.commit()


def show_status(conn):
    """Show loaded data status."""
    for name, config in INSTRUMENTS.items():
        table = config['table']
        try:
            cur = conn.cursor()
            cur.execute(f"""
                SELECT timeframe, COUNT(*), MIN(datetime), MAX(datetime)
                FROM {table}
                GROUP BY timeframe ORDER BY timeframe
            """)
            rows = cur.fetchall()
            if rows:
                print(f"\n{name} ({table}):")
                for tf, count, min_dt, max_dt in rows:
                    days = count // {'5min': 75, '15min': 25, '60min': 7}.get(tf, 75)
                    source = 'synthetic' if min_dt and min_dt.hour == 9 and min_dt.minute == 15 else 'real/mixed'
                    print(f"  {tf:<6} {count:>8,} bars ({days:,} days)  "
                          f"{min_dt} to {max_dt}")
            else:
                print(f"\n{name} ({table}): EMPTY")
        except Exception as e:
            print(f"\n{name} ({table}): table not found ({e})")
            conn.rollback()


def main():
    parser = argparse.ArgumentParser(description='Fetch historical 5-min data from Kite')
    parser.add_argument('--from', dest='from_date', type=str, default=None,
                        help='Start date (YYYY-MM-DD). Default: 1 year ago')
    parser.add_argument('--to', dest='to_date', type=str, default=None,
                        help='End date (YYYY-MM-DD). Default: today')
    parser.add_argument('--banknifty', action='store_true',
                        help='Also fetch BankNifty')
    parser.add_argument('--status', action='store_true',
                        help='Show loaded data status')
    args = parser.parse_args()

    conn = psycopg2.connect(DATABASE_DSN)

    if args.status:
        show_status(conn)
        conn.close()
        return

    # Authenticate
    kite = get_kite()
    if not kite:
        print("ERROR: Not authenticated. Run first:")
        print("  venv/bin/python3 -m data.kite_auth --login")
        conn.close()
        return

    # Verify token
    try:
        profile = kite.profile()
        print(f"Authenticated as: {profile['user_name']} ({profile['user_id']})")
    except Exception as e:
        print(f"ERROR: Token invalid — {e}")
        print("Run: venv/bin/python3 -m data.kite_auth --login")
        conn.close()
        return

    # Date range
    to_date = datetime.strptime(args.to_date, '%Y-%m-%d').date() if args.to_date else date.today()
    from_date = datetime.strptime(args.from_date, '%Y-%m-%d').date() if args.from_date else to_date - timedelta(days=365)

    # Fetch Nifty
    ensure_table(conn, 'nifty_intraday')
    fetch_and_store(kite, 'NIFTY', from_date, to_date, conn)

    # Fetch BankNifty if requested
    if args.banknifty:
        ensure_table(conn, 'banknifty_intraday')
        fetch_and_store(kite, 'BANKNIFTY', from_date, to_date, conn)

    # Show final status
    print("\n" + "=" * 60)
    show_status(conn)

    # Reminder about synthetic data
    print("\nNote: Existing synthetic data (if any) has been overwritten")
    print("with real Kite data where dates overlap.")

    conn.close()


if __name__ == '__main__':
    main()
