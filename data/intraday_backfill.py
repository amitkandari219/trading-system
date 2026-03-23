"""
Intraday 5-minute bar backfill from Kite Connect API to PostgreSQL.

Downloads historical 5-min OHLCV data for NIFTY and BANKNIFTY,
stores in `intraday_bars` table with upsert semantics.

Usage:
    python -m data.intraday_backfill --init            # Create table
    python -m data.intraday_backfill --backfill        # Full 2-year backfill
    python -m data.intraday_backfill --backfill --start 2025-01-01
    python -m data.intraday_backfill --status          # Quality report
    python -m data.intraday_backfill --volume-profile  # Volume by time slot
"""

import argparse
import logging
import time
from collections import defaultdict
from datetime import datetime, date, timedelta

import psycopg2
import psycopg2.extras

from config.settings import DATABASE_DSN
from data.kite_auth import get_kite

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INSTRUMENTS = {
    "NIFTY":     256265,
    "BANKNIFTY": 260105,
}

INTERVAL = "5minute"
CHUNK_DAYS = 60          # Kite historical API limit per request
RATE_LIMIT_DELAY = 0.35  # seconds between requests (max 3 req/s)

MARKET_OPEN = "09:15"
MARKET_CLOSE = "15:30"
EXPECTED_BARS_PER_DAY = 75  # 6h15m / 5min = 75

# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------

CREATE_TABLE_SQL = """\
CREATE TABLE IF NOT EXISTS intraday_bars (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    instrument VARCHAR(16) NOT NULL,
    open FLOAT,
    high FLOAT,
    low FLOAT,
    close FLOAT,
    volume BIGINT,
    oi BIGINT DEFAULT 0,
    UNIQUE(timestamp, instrument)
);
CREATE INDEX IF NOT EXISTS idx_intraday_bars_inst_ts
    ON intraday_bars(instrument, timestamp);
"""

INSERT_SQL = """\
INSERT INTO intraday_bars (timestamp, instrument, open, high, low, close, volume, oi)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (timestamp, instrument) DO NOTHING;
"""


# ---------------------------------------------------------------------------
# IntradayBackfill
# ---------------------------------------------------------------------------

class IntradayBackfill:
    """Downloads 5-min historical data from Kite and stores in PostgreSQL."""

    def __init__(self, kite=None, db_conn=None):
        self.kite = kite or get_kite()
        if self.kite is None:
            raise RuntimeError(
                "No authenticated Kite session. Run: python -m data.kite_auth --login"
            )

        if db_conn is not None:
            self.conn = db_conn
            self._owns_conn = False
        else:
            self.conn = psycopg2.connect(DATABASE_DSN)
            self._owns_conn = True

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self):
        if self._owns_conn and self.conn and not self.conn.closed:
            self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    @staticmethod
    def create_table(dsn: str = None):
        """Run the migration to create intraday_bars table."""
        conn = psycopg2.connect(dsn or DATABASE_DSN)
        try:
            with conn.cursor() as cur:
                cur.execute(CREATE_TABLE_SQL)
            conn.commit()
            logger.info("Table intraday_bars created (or already exists).")
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Backfill
    # ------------------------------------------------------------------

    def backfill(self, instrument: str, start_date: date, end_date: date):
        """
        Download 5-min data for *instrument* in 60-day chunks and upsert
        into the database.  Respects Kite rate limits.
        """
        token = INSTRUMENTS[instrument]
        total_inserted = 0
        chunk_start = start_date

        while chunk_start <= end_date:
            chunk_end = min(chunk_start + timedelta(days=CHUNK_DAYS - 1), end_date)

            logger.info(
                "Fetching %s  %s -> %s ...",
                instrument,
                chunk_start.isoformat(),
                chunk_end.isoformat(),
            )

            try:
                records = self.kite.historical_data(
                    instrument_token=token,
                    from_date=datetime.combine(chunk_start, datetime.min.time()),
                    to_date=datetime.combine(chunk_end, datetime.max.time().replace(microsecond=0)),
                    interval=INTERVAL,
                    oi=True,
                )
            except Exception as e:
                logger.error(
                    "Kite API error for %s [%s - %s]: %s",
                    instrument, chunk_start, chunk_end, e,
                )
                # Back off and retry once
                time.sleep(2)
                try:
                    records = self.kite.historical_data(
                        instrument_token=token,
                        from_date=datetime.combine(chunk_start, datetime.min.time()),
                        to_date=datetime.combine(chunk_end, datetime.max.time().replace(microsecond=0)),
                        interval=INTERVAL,
                        oi=True,
                    )
                except Exception as e2:
                    logger.error("Retry failed for %s: %s  — skipping chunk.", instrument, e2)
                    chunk_start = chunk_end + timedelta(days=1)
                    time.sleep(RATE_LIMIT_DELAY)
                    continue

            rows = [
                (
                    rec["date"],
                    instrument,
                    rec["open"],
                    rec["high"],
                    rec["low"],
                    rec["close"],
                    rec["volume"],
                    rec.get("oi", 0) or 0,
                )
                for rec in records
            ]

            if rows:
                with self.conn.cursor() as cur:
                    psycopg2.extras.execute_batch(cur, INSERT_SQL, rows, page_size=500)
                self.conn.commit()
                total_inserted += len(rows)
                logger.info(
                    "  -> %d bars written (%d total so far).", len(rows), total_inserted
                )

            chunk_start = chunk_end + timedelta(days=1)
            time.sleep(RATE_LIMIT_DELAY)

        logger.info(
            "Backfill complete for %s: %d bars from %s to %s.",
            instrument, total_inserted, start_date, end_date,
        )
        return total_inserted

    def backfill_all(self, start_date: date = None):
        """Backfill both NIFTY and BANKNIFTY. Default start = 2 years ago."""
        end = date.today()
        start = start_date or (end - timedelta(days=730))

        summary = {}
        for instrument in INSTRUMENTS:
            count = self.backfill(instrument, start, end)
            summary[instrument] = count
        return summary

    # ------------------------------------------------------------------
    # Quality check
    # ------------------------------------------------------------------

    def quality_check(self, instrument: str) -> dict:
        """
        Return quality stats:
          - total_bars, total_days
          - min/max/avg bars_per_day
          - gap_days (trading days with zero bars)
          - coverage_pct (avg_bars_per_day / 75 * 100)
          - date_range (first_ts, last_ts)
        """
        with self.conn.cursor() as cur:
            # Total bars
            cur.execute(
                "SELECT COUNT(*) FROM intraday_bars WHERE instrument = %s",
                (instrument,),
            )
            total_bars = cur.fetchone()[0]

            if total_bars == 0:
                return {
                    "instrument": instrument,
                    "total_bars": 0,
                    "total_days": 0,
                    "avg_bars_per_day": 0,
                    "min_bars_per_day": 0,
                    "max_bars_per_day": 0,
                    "gap_days": 0,
                    "coverage_pct": 0.0,
                    "first_ts": None,
                    "last_ts": None,
                }

            # Bars per day distribution
            cur.execute(
                """
                SELECT timestamp::date AS dt, COUNT(*) AS cnt
                FROM intraday_bars
                WHERE instrument = %s
                GROUP BY dt
                ORDER BY dt
                """,
                (instrument,),
            )
            day_rows = cur.fetchall()
            days = [r[0] for r in day_rows]
            counts = [r[1] for r in day_rows]

            total_days = len(days)
            avg_bars = sum(counts) / total_days
            min_bars = min(counts)
            max_bars = max(counts)

            # Gap detection: count business days between first and last that
            # have zero bars (excluding weekends).
            first_day, last_day = days[0], days[-1]
            present_set = set(days)
            gap_days = 0
            d = first_day
            while d <= last_day:
                if d.weekday() < 5 and d not in present_set:
                    gap_days += 1
                d += timedelta(days=1)

            # Date range
            cur.execute(
                "SELECT MIN(timestamp), MAX(timestamp) FROM intraday_bars WHERE instrument = %s",
                (instrument,),
            )
            first_ts, last_ts = cur.fetchone()

            coverage_pct = round(avg_bars / EXPECTED_BARS_PER_DAY * 100, 2)

            return {
                "instrument": instrument,
                "total_bars": total_bars,
                "total_days": total_days,
                "avg_bars_per_day": round(avg_bars, 1),
                "min_bars_per_day": min_bars,
                "max_bars_per_day": max_bars,
                "gap_days": gap_days,
                "coverage_pct": coverage_pct,
                "first_ts": first_ts.isoformat() if first_ts else None,
                "last_ts": last_ts.isoformat() if last_ts else None,
            }

    # ------------------------------------------------------------------
    # Volume profile
    # ------------------------------------------------------------------

    def volume_profile(self, instrument: str) -> dict:
        """
        Average volume by 5-min time slot across all days.

        Returns:
            {
                "instrument": ...,
                "slots": {
                    "09:15": avg_vol,
                    "09:20": avg_vol,
                    ...
                },
                "high_volume_slots": [...top 5...],
                "low_volume_slots": [...bottom 5...],
            }
        """
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    TO_CHAR(timestamp, 'HH24:MI') AS slot,
                    AVG(volume)::BIGINT            AS avg_vol,
                    COUNT(*)                        AS sample_size
                FROM intraday_bars
                WHERE instrument = %s
                GROUP BY slot
                ORDER BY slot
                """,
                (instrument,),
            )
            rows = cur.fetchall()

        if not rows:
            return {
                "instrument": instrument,
                "slots": {},
                "high_volume_slots": [],
                "low_volume_slots": [],
            }

        slots = {r[0]: int(r[1]) for r in rows}
        sorted_by_vol = sorted(slots.items(), key=lambda x: x[1], reverse=True)

        return {
            "instrument": instrument,
            "slots": slots,
            "high_volume_slots": sorted_by_vol[:5],
            "low_volume_slots": sorted_by_vol[-5:],
        }


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _print_quality(report: dict):
    """Pretty-print a quality report."""
    print(f"\n{'='*50}")
    print(f"  Quality Report: {report['instrument']}")
    print(f"{'='*50}")
    print(f"  Total bars      : {report['total_bars']:,}")
    print(f"  Trading days    : {report['total_days']:,}")
    print(f"  Bars/day (avg)  : {report['avg_bars_per_day']}")
    print(f"  Bars/day (min)  : {report['min_bars_per_day']}")
    print(f"  Bars/day (max)  : {report['max_bars_per_day']}")
    print(f"  Gap days        : {report['gap_days']}")
    print(f"  Coverage        : {report['coverage_pct']}%")
    print(f"  First timestamp : {report['first_ts']}")
    print(f"  Last timestamp  : {report['last_ts']}")


def _print_volume_profile(profile: dict):
    """Pretty-print a volume profile."""
    print(f"\n{'='*50}")
    print(f"  Volume Profile: {profile['instrument']}")
    print(f"{'='*50}")

    if not profile["slots"]:
        print("  No data available.")
        return

    max_vol = max(profile["slots"].values()) or 1
    for slot, vol in sorted(profile["slots"].items()):
        bar_len = int(vol / max_vol * 40)
        bar = "#" * bar_len
        print(f"  {slot}  {vol:>12,}  {bar}")

    print(f"\n  High-volume windows:")
    for slot, vol in profile["high_volume_slots"]:
        print(f"    {slot}  {vol:>12,}")

    print(f"\n  Low-volume windows:")
    for slot, vol in profile["low_volume_slots"]:
        print(f"    {slot}  {vol:>12,}")


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Intraday 5-min bar backfill")
    parser.add_argument("--init", action="store_true", help="Create intraday_bars table")
    parser.add_argument("--backfill", action="store_true", help="Download historical data")
    parser.add_argument("--start", type=str, default=None, help="Backfill start date (YYYY-MM-DD)")
    parser.add_argument("--status", action="store_true", help="Print quality report")
    parser.add_argument("--volume-profile", action="store_true", help="Print volume by time slot")
    args = parser.parse_args()

    if not any([args.init, args.backfill, args.status, args.volume_profile]):
        parser.print_help()
        return

    if args.init:
        IntradayBackfill.create_table()
        print("Table intraday_bars ready.")

    if args.backfill:
        start = None
        if args.start:
            start = date.fromisoformat(args.start)
        with IntradayBackfill() as bf:
            summary = bf.backfill_all(start_date=start)
            print("\nBackfill summary:")
            for inst, cnt in summary.items():
                print(f"  {inst}: {cnt:,} bars")

    if args.status:
        with IntradayBackfill() as bf:
            for instrument in INSTRUMENTS:
                report = bf.quality_check(instrument)
                _print_quality(report)

    if args.volume_profile:
        with IntradayBackfill() as bf:
            for instrument in INSTRUMENTS:
                profile = bf.volume_profile(instrument)
                _print_volume_profile(profile)


if __name__ == "__main__":
    main()
