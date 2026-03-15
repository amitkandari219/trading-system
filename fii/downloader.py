"""
FII data downloader — downloads NSE participant-wise F&O OI data.
Must mimic browser session — NSE blocks raw API calls.
"""

import requests
import time
from datetime import date
import pandas as pd
from io import StringIO


class DataNotAvailableError(Exception):
    """Raised when FII OI data is not yet published for the requested date."""
    pass


class WrongFileError(Exception):
    """Raised when downloaded file appears to contain volumes instead of OI."""
    pass


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

        Returns DataFrame with columns:
        client_type, future_long_contracts, future_short_contracts,
        call_long_contracts, call_short_contracts,
        put_long_contracts, put_short_contracts,
        total_long_contracts, total_short_contracts
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

        # Validate OI file (not volumes)
        required_columns = {'future_long_contracts', 'future_short_contracts',
                            'put_long_contracts', 'put_short_contracts'}
        normalized = self._normalize_oi_columns(df, for_date)
        if not required_columns.issubset(set(normalized.columns)):
            raise WrongFileError(
                f"Downloaded file appears to be volumes, not OI. "
                f"Columns found: {set(normalized.columns)}"
            )

        return normalized

    def _normalize_oi_columns(self, df: pd.DataFrame,
                               for_date: date) -> pd.DataFrame:
        """
        NSE changes column names occasionally.
        Normalize to our standard names.
        """
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
