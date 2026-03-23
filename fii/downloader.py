"""
FII data downloader — downloads NSE participant-wise F&O OI data.

Downloads from NSE archives, parses participant-wise open interest CSV,
computes derived metrics (net_futures, net_calls, net_puts, PCR),
and stores to the fii_daily_metrics DB table.

Must mimic browser session — NSE blocks raw API calls.

Rate limit: 1 req/sec, max 60 requests per session.
"""

import logging
import time
from datetime import date, timedelta
from io import StringIO
from typing import Dict, List, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)


class DataNotAvailableError(Exception):
    """Raised when FII OI data is not yet published for the requested date."""
    pass


class WrongFileError(Exception):
    """Raised when downloaded file appears to contain volumes instead of OI."""
    pass


class FIIDataDownloader:
    """
    Downloads NSE participant-wise F&O data, parses and stores to DB.

    NSE requires browser-like session cookies — we hit the homepage first,
    then use the session to fetch data. Rate limited to 1 req/sec.

    Two download paths:
    1. Archive CSV (reliable, works for historical):
       https://archives.nseindia.com/content/nsccl/fao_participant_oi_{ddMMMyyyy}.csv
    2. NSE API (for current day, returns redirect to CSV):
       /api/reports?archives=[...]&date={dd-Mon-YYYY}

    Columns stored to fii_daily_metrics:
        date, fut_long, fut_short, fut_net, put_long, put_short, put_net,
        put_ratio, call_long, call_short, call_net, pcr, fut_net_change,
        nifty_close
    """

    NSE_BASE = "https://www.nseindia.com"
    SESSION_URL = "https://www.nseindia.com/market-data/equity-derivatives-watch"
    ARCHIVE_URL_TEMPLATE = (
        "https://archives.nseindia.com/content/nsccl/"
        "fao_participant_oi_{date_str}.csv"
    )

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

    # Rate limiting
    MIN_REQUEST_INTERVAL = 1.0  # seconds between requests
    MAX_REQUESTS_PER_SESSION = 60
    MAX_RETRIES = 3
    RETRY_DELAY = 5  # seconds

    # Standard column mapping for NSE CSV
    # NSE uses "Future Index Long", "Option Index Call Long", etc.
    COLUMN_MAP = {
        'Client Type':             'client_type',
        'Future Index Long':       'future_long_contracts',
        'Future Index Short':      'future_short_contracts',
        'Future Stock Long':       'future_stock_long',
        'Future Stock Short':      'future_stock_short',
        'Future Long':             'future_long_contracts',
        'Future Short':            'future_short_contracts',
        'Option Index Call Long':  'call_long_contracts',
        'Option Index Call Short': 'call_short_contracts',
        'Option Index Put Long':   'put_long_contracts',
        'Option Index Put Short':  'put_short_contracts',
        'Option Call Long':        'call_long_contracts',
        'Option Call Short':       'call_short_contracts',
        'Option Put Long':         'put_long_contracts',
        'Option Put Short':        'put_short_contracts',
        'Total Long Contracts':    'total_long_contracts',
        'Total Short Contracts':   'total_short_contracts',
    }

    REQUIRED_COLUMNS = {
        'future_long_contracts', 'future_short_contracts',
        'put_long_contracts', 'put_short_contracts',
    }

    def __init__(self, db_conn=None):
        self.db = db_conn
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)
        self._cookies_initialized = False
        self._last_request_time = 0.0
        self._request_count = 0

    # ================================================================
    # SESSION MANAGEMENT
    # ================================================================

    def _init_session(self):
        """
        Fetch NSE homepage first to get session cookies.
        Without this, data requests return 401/403.
        Resets after MAX_REQUESTS_PER_SESSION to avoid blocks.
        """
        if self._cookies_initialized and self._request_count < self.MAX_REQUESTS_PER_SESSION:
            return

        if self._request_count >= self.MAX_REQUESTS_PER_SESSION:
            logger.info("Session limit reached — refreshing NSE cookies")
            self.session = requests.Session()
            self.session.headers.update(self.HEADERS)
            self._request_count = 0

        self._rate_limit()
        try:
            resp = self.session.get(self.SESSION_URL, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.warning(f"NSE session init failed: {e}")
            raise DataNotAvailableError(
                f"Cannot establish NSE session: {e}"
            )
        time.sleep(2)  # NSE rate limit after homepage
        self._cookies_initialized = True

    def _rate_limit(self):
        """Enforce minimum interval between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.MIN_REQUEST_INTERVAL:
            time.sleep(self.MIN_REQUEST_INTERVAL - elapsed)
        self._last_request_time = time.time()
        self._request_count += 1

    # ================================================================
    # PUBLIC: download_date
    # ================================================================

    def download_date(self, as_of_date: date) -> Optional[pd.DataFrame]:
        """
        Fetch NSE participant OI CSV for a single date, parse, compute
        derived metrics, and optionally store to DB.

        Returns DataFrame with one row (FII) containing:
            date, future_long_contracts, future_short_contracts,
            call_long_contracts, call_short_contracts,
            put_long_contracts, put_short_contracts,
            [derived]: fut_net, call_net, put_net, pcr

        Returns None if data not available (holiday/weekend).
        Raises DataNotAvailableError after MAX_RETRIES failures.
        """
        self._init_session()

        raw_df = None
        last_error = None

        for attempt in range(self.MAX_RETRIES):
            try:
                # Try archive URL first (more reliable)
                raw_df = self._download_archive_csv(as_of_date)
                break
            except DataNotAvailableError:
                # Try API fallback
                try:
                    raw_df = self._download_api_csv(as_of_date)
                    break
                except (DataNotAvailableError, Exception) as e:
                    last_error = e
                    if attempt < self.MAX_RETRIES - 1:
                        logger.info(
                            f"Retry {attempt + 1}/{self.MAX_RETRIES} for "
                            f"{as_of_date}: {e}"
                        )
                        time.sleep(self.RETRY_DELAY)
            except requests.HTTPError as e:
                last_error = e
                if e.response is not None and e.response.status_code == 403:
                    logger.warning(
                        f"NSE 403 on attempt {attempt + 1} — "
                        f"refreshing session"
                    )
                    self._cookies_initialized = False
                    self._init_session()
                elif attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_DELAY)
                else:
                    raise
            except Exception as e:
                last_error = e
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_DELAY)

        if raw_df is None:
            if last_error:
                raise DataNotAvailableError(
                    f"FII OI data not available for {as_of_date} "
                    f"after {self.MAX_RETRIES} attempts: {last_error}"
                )
            return None

        # Normalize columns and filter to FII row
        normalized = self._normalize_oi_columns(raw_df, as_of_date)
        if normalized.empty:
            raise DataNotAvailableError(
                f"No FII/FPI row found in OI data for {as_of_date}"
            )

        # Validate required columns
        if not self.REQUIRED_COLUMNS.issubset(set(normalized.columns)):
            raise WrongFileError(
                f"Downloaded file appears to be volumes, not OI. "
                f"Columns found: {set(normalized.columns)}"
            )

        # Compute derived metrics
        enriched = self._compute_derived(normalized)

        # Store to DB if connection available
        if self.db is not None:
            self._store_to_db(enriched)

        logger.info(
            f"FII data downloaded for {as_of_date}: "
            f"fut_net={enriched.iloc[0].get('fut_net', 0):,.0f} "
            f"pcr={enriched.iloc[0].get('pcr', 0):.3f}"
        )

        return enriched

    # ================================================================
    # PUBLIC: download_range
    # ================================================================

    def download_range(self, start_date: date,
                       end_date: date) -> List[date]:
        """
        Download FII OI data for a date range with rate limiting.

        Returns list of dates successfully downloaded.
        Skips weekends and holidays (404s) silently.
        """
        downloaded = []
        current = start_date

        while current <= end_date:
            # Skip weekends
            if current.weekday() >= 5:
                current += timedelta(days=1)
                continue

            try:
                result = self.download_date(current)
                if result is not None:
                    downloaded.append(current)
            except DataNotAvailableError:
                logger.debug(f"No data for {current} (holiday?)")
            except Exception as e:
                logger.warning(f"Failed to download {current}: {e}")

            # Rate limit: 1 second between dates
            time.sleep(self.MIN_REQUEST_INTERVAL)
            current += timedelta(days=1)

        logger.info(
            f"Downloaded FII data for {len(downloaded)} days "
            f"({start_date} to {end_date})"
        )
        return downloaded

    # ================================================================
    # PUBLIC: initial_load
    # ================================================================

    def initial_load(self, months: int = 6) -> List[date]:
        """
        Bootstrap historical FII data. Downloads last N months of
        trading day data from NSE archives.

        Returns list of dates successfully downloaded.
        """
        end = date.today()
        start = end - timedelta(days=months * 30)
        logger.info(
            f"Initial FII load: {start} to {end} ({months} months)"
        )
        return self.download_range(start, end)

    # ================================================================
    # PUBLIC: get_latest
    # ================================================================

    def get_latest(self) -> Optional[Dict]:
        """
        Return the most recent FII data from DB.

        Returns dict with: date, fut_net, put_ratio, pcr, fut_net_change,
        nifty_close, or None if no data.
        """
        if self.db is None:
            return None

        try:
            cur = self.db.cursor()
            cur.execute("""
                SELECT date, fut_long, fut_short, fut_net,
                       put_long, put_short, put_net, put_ratio,
                       call_long, call_short, call_net, pcr,
                       fut_net_change, nifty_close
                FROM fii_daily_metrics
                ORDER BY date DESC LIMIT 1
            """)
            row = cur.fetchone()
            if not row:
                return None

            columns = [
                'date', 'fut_long', 'fut_short', 'fut_net',
                'put_long', 'put_short', 'put_net', 'put_ratio',
                'call_long', 'call_short', 'call_net', 'pcr',
                'fut_net_change', 'nifty_close',
            ]
            return dict(zip(columns, row))
        except Exception as e:
            logger.error(f"get_latest failed: {e}")
            return None

    # ================================================================
    # PRIVATE: Download methods
    # ================================================================

    def _download_archive_csv(self, for_date: date) -> pd.DataFrame:
        """
        Download from NSE archive URL:
        https://archives.nseindia.com/content/nsccl/fao_participant_oi_{ddmmyyyy}.csv
        """
        # NSE archive uses ddmmyyyy format (numeric month): 20032026
        date_str = for_date.strftime('%d%m%Y')
        url = self.ARCHIVE_URL_TEMPLATE.format(date_str=date_str)

        self._rate_limit()
        resp = self.session.get(url, timeout=30)

        if resp.status_code == 404:
            raise DataNotAvailableError(
                f"Archive CSV not found for {for_date}"
            )
        resp.raise_for_status()

        # Validate CSV content
        text = resp.text.strip()
        if not text or '<html' in text.lower():
            raise DataNotAvailableError(
                f"Archive returned HTML instead of CSV for {for_date}"
            )

        # NSE CSV has a title line first, then header row
        # e.g. line 0: "Participant wise Open Interest..."
        #      line 1: Client Type,Future Index Long,...
        lines = text.split('\n')
        if len(lines) > 2 and 'participant' in lines[0].lower():
            text = '\n'.join(lines[1:])

        return pd.read_csv(StringIO(text))

    def _download_api_csv(self, for_date: date) -> pd.DataFrame:
        """
        Download via NSE API (reports endpoint). Falls back when
        archive URL is not yet available for recent dates.
        """
        date_str = for_date.strftime('%d-%b-%Y')  # e.g. 12-Mar-2026

        api_url = (
            f"{self.NSE_BASE}/api/reports?"
            f"archives=%5B%7B%22name%22%3A%22F%26O+Participant+wise+OI%22"
            f"%2C%22type%22%3A%22archives%22%2C%22category%22%3A%22derivatives%22"
            f"%2C%22section%22%3A%22equity%22%7D%5D"
            f"&date={date_str}&type=equity&category=derivatives"
        )

        self._rate_limit()
        resp = self.session.get(api_url, timeout=30)

        if resp.status_code == 404:
            raise DataNotAvailableError(
                f"API: FII OI not published for {date_str}"
            )
        resp.raise_for_status()

        data = resp.json()
        if not data or not isinstance(data, list):
            raise DataNotAvailableError(
                f"API: unexpected response for {date_str}"
            )

        # Download the actual CSV
        csv_url = self.NSE_BASE + data[0].get('link', '')
        self._rate_limit()
        csv_resp = self.session.get(csv_url, timeout=30)
        csv_resp.raise_for_status()

        return pd.read_csv(StringIO(csv_resp.text))

    # ================================================================
    # PRIVATE: Normalization and enrichment
    # ================================================================

    def _normalize_oi_columns(self, df: pd.DataFrame,
                               for_date: date) -> pd.DataFrame:
        """
        NSE changes column names occasionally.
        Normalize to our standard names using flexible matching.
        """
        # Strip whitespace from column names first
        df.columns = [c.strip() for c in df.columns]

        # Dynamic column detection — match by cleaned names
        rename = {}
        for col in df.columns:
            col_clean = col.lower().strip().replace('_', ' ')
            for nse_name, std_name in self.COLUMN_MAP.items():
                nse_clean = nse_name.lower().strip()
                if col_clean == nse_clean or nse_clean in col_clean:
                    rename[col] = std_name
                    break
        df = df.rename(columns=rename)

        # Ensure client_type column exists
        if 'client_type' not in df.columns:
            # Try first column as client_type
            if len(df.columns) > 0:
                df = df.rename(columns={df.columns[0]: 'client_type'})
            else:
                return pd.DataFrame()

        # Keep only FII/FPI row
        fii_mask = df['client_type'].astype(str).str.upper().str.contains(
            'FII|FPI|FOREIGN', na=False
        )
        df = df[fii_mask].copy()

        if df.empty:
            return df

        # Add date
        df['date'] = for_date

        # Convert numeric columns
        for col in self.REQUIRED_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace(',', ''),
                    errors='coerce'
                ).fillna(0).astype(float)

        # Also convert optional columns
        for col in ['call_long_contracts', 'call_short_contracts',
                     'total_long_contracts', 'total_short_contracts']:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace(',', ''),
                    errors='coerce'
                ).fillna(0).astype(float)

        return df

    def _compute_derived(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute derived metrics from raw OI data.

        Adds: fut_net, call_net, put_net, pcr (put-call ratio).
        """
        row = df.iloc[0]

        fut_long = float(row.get('future_long_contracts', 0))
        fut_short = float(row.get('future_short_contracts', 0))
        put_long = float(row.get('put_long_contracts', 0))
        put_short = float(row.get('put_short_contracts', 0))
        call_long = float(row.get('call_long_contracts', 0))
        call_short = float(row.get('call_short_contracts', 0))

        df = df.copy()
        df['fut_net'] = fut_long - fut_short
        df['call_net'] = call_long - call_short
        df['put_net'] = put_long - put_short
        df['pcr'] = (
            (put_long + put_short) / (call_long + call_short)
            if (call_long + call_short) > 0 else 1.0
        )
        df['put_ratio'] = (
            put_long / put_short if put_short > 0 else 999.0
        )

        return df

    # ================================================================
    # PRIVATE: DB storage
    # ================================================================

    def _store_to_db(self, df: pd.DataFrame):
        """Store enriched FII data to fii_daily_metrics table."""
        if self.db is None or df.empty:
            return

        row = df.iloc[0]
        try:
            cur = self.db.cursor()
            # Table columns: trade_date, fii_long_contracts, fii_short_contracts,
            # fii_long_oi, fii_short_oi, dii_long_contracts, dii_short_contracts,
            # net_long_ratio, source
            fut_long = float(row.get('future_long_contracts', 0))
            fut_short = float(row.get('future_short_contracts', 0))
            net_ratio = fut_long / fut_short if fut_short > 0 else 0.0
            cur.execute("""
                INSERT INTO fii_daily_metrics
                    (trade_date, fii_long_contracts, fii_short_contracts,
                     fii_long_oi, fii_short_oi,
                     dii_long_contracts, dii_short_contracts,
                     source)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (trade_date) DO UPDATE SET
                    fii_long_contracts = EXCLUDED.fii_long_contracts,
                    fii_short_contracts = EXCLUDED.fii_short_contracts,
                    fii_long_oi = EXCLUDED.fii_long_oi,
                    fii_short_oi = EXCLUDED.fii_short_oi,
                    dii_long_contracts = EXCLUDED.dii_long_contracts,
                    dii_short_contracts = EXCLUDED.dii_short_contracts,
                    source = EXCLUDED.source
            """, (
                row.get('date'),
                fut_long,
                fut_short,
                float(row.get('put_long_contracts', 0)),
                float(row.get('put_short_contracts', 0)),
                float(row.get('call_long_contracts', 0)),
                float(row.get('call_short_contracts', 0)),
                'NSE_ARCHIVE',
            ))
            self.db.commit()
        except Exception as e:
            logger.error(f"FII DB store failed for {row.get('date')}: {e}")
            try:
                self.db.rollback()
            except Exception:
                pass

    # ================================================================
    # ALSO: Keep DII data for divergence signal
    # ================================================================

    def download_dii_date(self, as_of_date: date) -> Optional[pd.DataFrame]:
        """
        Download DII positioning for the same date.
        Used for FII-DII divergence signal (Pattern 5).
        Returns DII row DataFrame or None.
        """
        self._init_session()

        try:
            date_str = as_of_date.strftime('%d%b%Y')
            url = self.ARCHIVE_URL_TEMPLATE.format(date_str=date_str)
            self._rate_limit()
            resp = self.session.get(url, timeout=30)

            if resp.status_code != 200:
                return None

            text = resp.text.strip()
            if not text or '<html' in text.lower():
                return None

            df = pd.read_csv(StringIO(text))

            # Normalize columns
            rename = {}
            for col in df.columns:
                col_clean = col.lower().replace(' ', '').replace('_', '')
                for nse_name, std_name in self.COLUMN_MAP.items():
                    nse_clean = nse_name.lower().replace(' ', '')
                    if nse_clean in col_clean:
                        rename[col] = std_name
                        break
            df = df.rename(columns=rename)

            if 'client_type' not in df.columns:
                if len(df.columns) > 0:
                    df = df.rename(columns={df.columns[0]: 'client_type'})
                else:
                    return None

            # DII row
            dii_mask = df['client_type'].astype(str).str.upper().str.contains(
                'DII|DOMESTIC', na=False
            )
            dii_df = df[dii_mask].copy()

            if dii_df.empty:
                return None

            dii_df['date'] = as_of_date
            for col in self.REQUIRED_COLUMNS:
                if col in dii_df.columns:
                    dii_df[col] = pd.to_numeric(
                        dii_df[col].astype(str).str.replace(',', ''),
                        errors='coerce'
                    ).fillna(0).astype(float)

            return dii_df

        except Exception as e:
            logger.debug(f"DII download failed for {as_of_date}: {e}")
            return None
