"""
Daily Kite data refresh — fetch yesterday's + today's 5-min bars.

Run after market close (3:30 PM) or before next day's open.
Designed to be called from cron or the daily pipeline.

Usage:
    venv/bin/python3 -m data.kite_daily_refresh          # fetch last 2 days
    venv/bin/python3 -m data.kite_daily_refresh --days 5  # fetch last 5 days
"""

import argparse
import logging
from datetime import date, datetime, timedelta

import psycopg2
from psycopg2.extras import execute_values

from config.settings import DATABASE_DSN
from data.kite_auth import get_kite

logger = logging.getLogger(__name__)

NIFTY_TOKEN = 256265
BANKNIFTY_TOKEN = 260105


def refresh(days=2):
    """Fetch recent 5-min bars for Nifty and BankNifty."""
    kite = get_kite()
    if not kite:
        print("ERROR: Not authenticated. Run: venv/bin/python3 -m data.kite_auth --login")
        return False

    conn = psycopg2.connect(DATABASE_DSN)
    to_dt = datetime.now()
    from_dt = datetime.combine(date.today() - timedelta(days=days), datetime.min.time())

    for name, token, table in [
        ('NIFTY', NIFTY_TOKEN, 'nifty_intraday'),
        ('BANKNIFTY', BANKNIFTY_TOKEN, 'banknifty_intraday'),
    ]:
        try:
            data = kite.historical_data(
                instrument_token=token,
                from_date=from_dt,
                to_date=to_dt,
                interval='5minute',
            )
            if not data:
                print(f"  {name}: no data")
                continue

            rows = []
            for candle in data:
                dt = candle['date']
                if hasattr(dt, 'tzinfo') and dt.tzinfo:
                    dt = dt.replace(tzinfo=None)
                rows.append((dt, '5min', candle['open'], candle['high'],
                             candle['low'], candle['close'], candle['volume'],
                             None, None, None))

            cur = conn.cursor()
            execute_values(cur,
                f"""INSERT INTO {table}
                    (datetime, timeframe, open, high, low, close, volume,
                     vwap, opening_range_high, opening_range_low)
                    VALUES %s
                    ON CONFLICT (datetime, timeframe) DO UPDATE SET
                        open=EXCLUDED.open, high=EXCLUDED.high,
                        low=EXCLUDED.low, close=EXCLUDED.close,
                        volume=EXCLUDED.volume""",
                rows, page_size=1000)
            conn.commit()

            dates = set(r[0].date() for r in rows)
            print(f"  {name}: {len(rows)} bars ({len(dates)} days)")

        except Exception as e:
            print(f"  {name}: ERROR — {e}")
            conn.rollback()

    conn.close()
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--days', type=int, default=2)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    print(f"Kite daily refresh (last {args.days} days)...")
    refresh(args.days)
    print("Done")
