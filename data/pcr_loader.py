"""
PCR (Put/Call Ratio) data loader from NSE F&O Bhavcopy.

NSE publishes daily options bhavcopy at:
https://archives.nseindia.com/content/historical/
DERIVATIVES/{YYYY}/{MMM}/fo{DD}{MMM}{YYYY}bhav.csv.zip

PCR = sum(Put OI) / sum(Call OI) for NIFTY index options.

Usage:
    python -m data.pcr_loader --init                    # Load 5-year history
    python -m data.pcr_loader --update                  # Update today
    python -m data.pcr_loader --status                  # Check loaded data
    python -m data.pcr_loader --start 2020-01-01        # Load from specific date
"""

import io
import logging
import time
import zipfile
from datetime import date, timedelta
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

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


class PCRLoader:
    """Downloads and computes Nifty PCR from NSE F&O bhavcopy archives."""

    BASE_URL = (
        "https://archives.nseindia.com/content/"
        "historical/DERIVATIVES/{year}/{month}/"
        "fo{date_str}bhav.csv.zip"
    )

    def __init__(self, db_conn=None):
        self.conn = db_conn
        self.session = requests.Session()
        self.session.headers.update(NSE_HEADERS)

    def download_bhavcopy(self, dt: date) -> Optional[dict]:
        """
        Download NSE F&O bhavcopy for given date.
        Filter for NIFTY index options, compute PCR.

        Returns dict: {date, pcr_oi, total_pe_oi, total_ce_oi} or None.
        """
        date_str = dt.strftime('%d%b%Y').upper()
        month_str = dt.strftime('%b').upper()
        url = self.BASE_URL.format(
            year=dt.year,
            month=month_str,
            date_str=date_str,
        )

        try:
            resp = self.session.get(url, timeout=30)
            if resp.status_code != 200:
                logger.debug(f"HTTP {resp.status_code} for {dt} (likely holiday)")
                return None

            # Extract CSV from zip
            zf = zipfile.ZipFile(io.BytesIO(resp.content))
            csv_names = [n for n in zf.namelist() if n.endswith('.csv')]
            if not csv_names:
                logger.warning(f"No CSV in zip for {dt}")
                return None

            df = pd.read_csv(zf.open(csv_names[0]))

            # Normalize column names (NSE uses varying case/spacing)
            df.columns = [c.strip().upper() for c in df.columns]

            # Find correct column names
            symbol_col = self._find_column(df, ['SYMBOL'])
            opttype_col = self._find_column(df, ['OPTION_TYP', 'OPTIONTYPE', 'OPTION_TYPE'])
            oi_col = self._find_column(df, ['OPEN_INT', 'OI', 'OPENINT', 'OPEN INT'])

            if not all([symbol_col, opttype_col, oi_col]):
                logger.warning(f"Missing columns for {dt}: sym={symbol_col} opt={opttype_col} oi={oi_col}")
                return None

            # Filter: NIFTY index options only (exclude BANKNIFTY, stock options)
            df[symbol_col] = df[symbol_col].str.strip()
            df[opttype_col] = df[opttype_col].str.strip().str.upper()
            nifty_opts = df[df[symbol_col] == 'NIFTY']

            # Separate puts and calls
            puts = nifty_opts[nifty_opts[opttype_col] == 'PE']
            calls = nifty_opts[nifty_opts[opttype_col] == 'CE']

            total_pe_oi = int(puts[oi_col].sum())
            total_ce_oi = int(calls[oi_col].sum())

            if total_ce_oi == 0:
                logger.warning(f"Zero call OI for {dt}")
                return None

            pcr_oi = total_pe_oi / total_ce_oi

            return {
                'date': dt,
                'pcr_oi': round(pcr_oi, 4),
                'total_pe_oi': total_pe_oi,
                'total_ce_oi': total_ce_oi,
            }

        except zipfile.BadZipFile:
            logger.debug(f"Bad zip for {dt} (likely holiday)")
            return None
        except Exception as e:
            logger.warning(f"Error downloading bhavcopy for {dt}: {e}")
            return None

    def _find_column(self, df: pd.DataFrame, candidates: list) -> Optional[str]:
        """Find first matching column name from candidates."""
        for c in candidates:
            if c in df.columns:
                return c
        return None

    def load_historical(self, start_date: date, end_date: date = None) -> pd.DataFrame:
        """
        Download PCR for every trading day in range.
        Skips weekends. Sleeps 1s between downloads.

        Returns DataFrame with columns: [date, pcr_oi, total_pe_oi, total_ce_oi]
        """
        end_date = end_date or date.today()
        records = []
        current = start_date
        downloaded = 0
        skipped = 0

        total_days = (end_date - start_date).days
        print(f"PCR Loader: downloading {start_date} to {end_date} ({total_days} calendar days)")

        while current <= end_date:
            # Skip weekends
            if current.weekday() >= 5:
                current += timedelta(days=1)
                continue

            result = self.download_bhavcopy(current)
            if result:
                records.append(result)
                downloaded += 1
                if downloaded % 50 == 0:
                    print(f"  Downloaded {downloaded} days (at {current})...")
            else:
                skipped += 1

            current += timedelta(days=1)
            time.sleep(1.0)  # NSE rate limit

        print(f"  Done: {downloaded} trading days, {skipped} skipped (holidays/errors)")

        df = pd.DataFrame(records)
        if not df.empty:
            df = self.compute_derived_columns(df)
        return df

    def compute_derived_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived columns to PCR data:

        pcr_5d_avg:  rolling 5-day mean of pcr_oi
        pcr_20d_avg: rolling 20-day mean
        pcr_20d_std: rolling 20-day std dev
        pcr_zscore:  (pcr_oi - pcr_20d_avg) / pcr_20d_std

        pcr_zscore interpretation:
          > +2.0: extreme fear (rare, strong buy signal)
          > +1.5: elevated fear (moderate buy signal)
          -1.5 to +1.5: neutral (no signal)
          < -1.5: elevated greed (moderate sell signal)
          < -2.0: extreme greed (rare, strong sell signal)
        """
        df = df.sort_values('date').reset_index(drop=True)

        df['pcr_5d_avg'] = df['pcr_oi'].rolling(5, min_periods=3).mean()
        df['pcr_20d_avg'] = df['pcr_oi'].rolling(20, min_periods=10).mean()
        df['pcr_20d_std'] = df['pcr_oi'].rolling(20, min_periods=10).std()

        df['pcr_zscore'] = df.apply(
            lambda r: (r['pcr_oi'] - r['pcr_20d_avg']) / r['pcr_20d_std']
            if pd.notna(r['pcr_20d_std']) and r['pcr_20d_std'] > 0
            else 0.0,
            axis=1,
        )

        return df

    def save_to_db(self, df: pd.DataFrame):
        """Save PCR data to pcr_daily table and update nifty_daily."""
        if self.conn is None:
            raise RuntimeError("No DB connection")

        cur = self.conn.cursor()

        # Create pcr_daily table if not exists
        cur.execute("""
            CREATE TABLE IF NOT EXISTS pcr_daily (
                date DATE PRIMARY KEY,
                pcr_oi FLOAT,
                total_pe_oi BIGINT,
                total_ce_oi BIGINT,
                pcr_5d_avg FLOAT,
                pcr_20d_avg FLOAT,
                pcr_20d_std FLOAT,
                pcr_zscore FLOAT
            )
        """)

        # Upsert PCR data
        for _, row in df.iterrows():
            cur.execute("""
                INSERT INTO pcr_daily (date, pcr_oi, total_pe_oi, total_ce_oi,
                                       pcr_5d_avg, pcr_20d_avg, pcr_20d_std, pcr_zscore)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (date) DO UPDATE SET
                    pcr_oi = EXCLUDED.pcr_oi,
                    total_pe_oi = EXCLUDED.total_pe_oi,
                    total_ce_oi = EXCLUDED.total_ce_oi,
                    pcr_5d_avg = EXCLUDED.pcr_5d_avg,
                    pcr_20d_avg = EXCLUDED.pcr_20d_avg,
                    pcr_20d_std = EXCLUDED.pcr_20d_std,
                    pcr_zscore = EXCLUDED.pcr_zscore
            """, (
                row['date'],
                row.get('pcr_oi'),
                row.get('total_pe_oi'),
                row.get('total_ce_oi'),
                row.get('pcr_5d_avg'),
                row.get('pcr_20d_avg'),
                row.get('pcr_20d_std'),
                row.get('pcr_zscore'),
            ))

        # Add PCR columns to nifty_daily if not present
        for col in ['pcr_oi', 'pcr_5d_avg', 'pcr_20d_avg', 'pcr_zscore']:
            cur.execute(f"""
                DO $$ BEGIN
                    ALTER TABLE nifty_daily ADD COLUMN {col} FLOAT;
                EXCEPTION WHEN duplicate_column THEN NULL;
                END $$;
            """)

        # Update nifty_daily with PCR values
        cur.execute("""
            UPDATE nifty_daily n
            SET pcr_oi = p.pcr_oi,
                pcr_5d_avg = p.pcr_5d_avg,
                pcr_20d_avg = p.pcr_20d_avg,
                pcr_zscore = p.pcr_zscore
            FROM pcr_daily p
            WHERE n.date = p.date
        """)

        self.conn.commit()
        logger.info(f"Saved {len(df)} PCR records to DB and updated nifty_daily")

    def update_today(self):
        """Download today's (or latest trading day's) PCR and save to DB."""
        today = date.today()
        # If weekend, get Friday
        if today.weekday() == 5:
            today = today - timedelta(days=1)
        elif today.weekday() == 6:
            today = today - timedelta(days=2)

        result = self.download_bhavcopy(today)
        if result is None:
            print(f"No bhavcopy available for {today} (market may still be open or holiday)")
            return

        # Load recent 30 days for z-score computation
        start = today - timedelta(days=45)
        df = self.load_historical(start, today)

        if not df.empty and self.conn:
            self.save_to_db(df)
            pcr = result['pcr_oi']
            print(f"PCR for {today}: {pcr:.4f} "
                  f"(PE OI: {result['total_pe_oi']:,} / CE OI: {result['total_ce_oi']:,})")


def parse_bhavcopy_df(df: pd.DataFrame) -> Optional[dict]:
    """
    Parse a raw bhavcopy DataFrame into PCR result.
    Used for testing without network access.
    """
    df.columns = [c.strip().upper() for c in df.columns]

    nifty = df[df['SYMBOL'].str.strip() == 'NIFTY']
    puts = nifty[nifty['OPTION_TYP'].str.strip().str.upper() == 'PE']
    calls = nifty[nifty['OPTION_TYP'].str.strip().str.upper() == 'CE']

    total_pe_oi = int(puts['OPEN_INT'].sum())
    total_ce_oi = int(calls['OPEN_INT'].sum())

    if total_ce_oi == 0:
        return None

    return {
        'pcr_oi': round(total_pe_oi / total_ce_oi, 4),
        'total_pe_oi': total_pe_oi,
        'total_ce_oi': total_ce_oi,
    }


# ================================================================
# CLI ENTRY POINT
# ================================================================
if __name__ == '__main__':
    import argparse
    import psycopg2
    from config.settings import DATABASE_DSN

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    parser = argparse.ArgumentParser(description='PCR data loader from NSE bhavcopy')
    parser.add_argument('command', nargs='?', default='status',
                        choices=['init', 'update_today', 'status'],
                        help='Command to run')
    parser.add_argument('--start', type=str, default=None,
                        help='Start date YYYY-MM-DD (default: 5 years ago)')
    parser.add_argument('--end', type=str, default=None,
                        help='End date YYYY-MM-DD (default: today)')
    parser.add_argument('--no-db', action='store_true',
                        help='Skip DB operations, print to stdout')
    args = parser.parse_args()

    conn = None if args.no_db else psycopg2.connect(DATABASE_DSN)
    loader = PCRLoader(db_conn=conn)

    if args.command == 'init':
        start = date.fromisoformat(args.start) if args.start else date.today() - timedelta(days=5*365)
        end = date.fromisoformat(args.end) if args.end else date.today()
        df = loader.load_historical(start, end)
        print(f"\nLoaded {len(df)} trading days")
        if not df.empty:
            print(f"PCR range: {df['pcr_oi'].min():.4f} to {df['pcr_oi'].max():.4f}")
            print(f"PCR mean:  {df['pcr_oi'].mean():.4f}")
            print(f"PCR std:   {df['pcr_oi'].std():.4f}")
            if conn:
                loader.save_to_db(df)
                print("Saved to DB")

    elif args.command == 'update_today':
        loader.update_today()

    elif args.command == 'status':
        if conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*), MIN(date), MAX(date) FROM pcr_daily")
            count, min_d, max_d = cur.fetchone()
            print(f"pcr_daily: {count} rows, {min_d} to {max_d}")

            cur.execute("SELECT COUNT(*) FROM nifty_daily WHERE pcr_oi IS NOT NULL")
            n_with_pcr = cur.fetchone()[0]
            print(f"nifty_daily with PCR: {n_with_pcr} rows")
        else:
            print("No DB connection (--no-db mode)")

    if conn:
        conn.close()
