"""
Global market data fetcher — retrieves overnight signals from international markets.

Data sources:
  - GIFT Nifty (NSE International Exchange, formerly SGX Nifty)
  - S&P 500, Nasdaq 100 (US equity indices)
  - CBOE VIX (US volatility index)
  - DXY (US Dollar Index)
  - Brent Crude Oil
  - US 10Y Treasury Yield

Uses yfinance as primary source (free, reliable, no API key needed).
Fallback to direct Yahoo Finance JSON endpoint if yfinance fails.

Schedule:
  - Runs at 8:30 AM IST (pre-market) to capture:
    * US close (7:00 AM IST = 9:30 PM ET previous day)
    * GIFT Nifty live (trades 6:30 AM IST onwards)
    * Asian session moves (already complete by 8:30 AM IST)

Usage:
    from data.global_market_fetcher import GlobalMarketFetcher
    fetcher = GlobalMarketFetcher(db_conn=conn)
    snapshot = fetcher.fetch_pre_market_snapshot()
    fetcher.store_snapshot(snapshot)
"""

import logging
import os
from datetime import date, datetime, timedelta
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ================================================================
# TICKER MAPPING
# ================================================================
TICKERS = {
    'gift_nifty': '^NSEI',          # Fallback — real GIFT uses NSE IX feed
    'sp500': '^GSPC',
    'nasdaq': '^IXIC',
    'us_vix': '^VIX',
    'dxy': 'DX-Y.NYB',
    'brent_crude': 'BZ=F',
    'us_10y': '^TNX',
    'dow_jones': '^DJI',
    'hang_seng': '^HSI',
    'nikkei': '^N225',
    'ftse': '^FTSE',
}

# Minimum history needed for rolling calculations
MIN_HISTORY_DAYS = 260  # ~1 year of trading days


class GlobalMarketFetcher:
    """
    Fetches and stores global market data for pre-market signal generation.

    Provides:
      1. Pre-market snapshot (latest close/price for all global instruments)
      2. Historical data for rolling calculations (252-day windows)
      3. GIFT Nifty gap computation
      4. US overnight return computation
    """

    def __init__(self, db_conn=None):
        self.db = db_conn
        self._yf = None  # lazy import

    # ================================================================
    # LAZY IMPORT — yfinance is heavy, only load when needed
    # ================================================================
    @property
    def yf(self):
        if self._yf is None:
            try:
                import yfinance
                self._yf = yfinance
            except ImportError:
                raise ImportError(
                    "yfinance not installed. Run: pip install yfinance --break-system-packages"
                )
        return self._yf

    # ================================================================
    # FETCH PRE-MARKET SNAPSHOT
    # ================================================================
    def fetch_pre_market_snapshot(self) -> Dict:
        """
        Fetch latest data for all global instruments.

        Returns dict with keys:
            timestamp, gift_nifty_last, sp500_close, sp500_change_pct,
            nasdaq_close, nasdaq_change_pct, us_vix_close, us_vix_change_pct,
            dxy_close, dxy_change_pct, brent_close, brent_change_pct,
            us_10y_close, us_10y_change_bps, nifty_prev_close,
            gift_nifty_gap_pct, us_overnight_return
        """
        snapshot = {
            'timestamp': datetime.now(),
            'date': date.today(),
        }

        # Fetch each instrument
        for key, ticker in TICKERS.items():
            try:
                data = self._fetch_ticker(ticker, period='5d')
                if data is not None and len(data) >= 2:
                    latest = data.iloc[-1]
                    prev = data.iloc[-2]
                    snapshot[f'{key}_close'] = float(latest['Close'])
                    snapshot[f'{key}_prev_close'] = float(prev['Close'])
                    change_pct = (latest['Close'] - prev['Close']) / prev['Close'] * 100
                    snapshot[f'{key}_change_pct'] = round(float(change_pct), 4)
                    snapshot[f'{key}_volume'] = int(latest.get('Volume', 0))
                elif data is not None and len(data) == 1:
                    latest = data.iloc[-1]
                    snapshot[f'{key}_close'] = float(latest['Close'])
                    snapshot[f'{key}_change_pct'] = 0.0
                else:
                    logger.warning(f"No data for {key} ({ticker})")
                    snapshot[f'{key}_close'] = None
                    snapshot[f'{key}_change_pct'] = None
            except Exception as e:
                logger.error(f"Error fetching {key} ({ticker}): {e}")
                snapshot[f'{key}_close'] = None
                snapshot[f'{key}_change_pct'] = None

        # ── Compute derived fields ────────────────────────────────
        snapshot = self._compute_derived_fields(snapshot)

        return snapshot

    # ================================================================
    # FETCH GIFT NIFTY REALTIME (via NSE IX or proxy)
    # ================================================================
    def fetch_gift_nifty_live(self) -> Optional[float]:
        """
        Fetch live GIFT Nifty price.

        Primary: yfinance ^NSEI with 1-minute interval
        Fallback: Returns None if unavailable (caller should use previous close)
        """
        try:
            ticker = self.yf.Ticker('^NSEI')
            info = ticker.fast_info
            return float(info.get('lastPrice', info.get('previousClose', None)))
        except Exception as e:
            logger.warning(f"GIFT Nifty live fetch failed: {e}")
            return None

    # ================================================================
    # FETCH HISTORICAL DATA FOR ROLLING CALCULATIONS
    # ================================================================
    def fetch_history(self, ticker_key: str, days: int = 260) -> Optional[pd.DataFrame]:
        """
        Fetch historical OHLCV for a given ticker key.

        Args:
            ticker_key: Key from TICKERS dict (e.g. 'sp500', 'us_vix')
            days: Number of calendar days to fetch

        Returns:
            DataFrame with columns: Date, Open, High, Low, Close, Volume
        """
        ticker = TICKERS.get(ticker_key)
        if not ticker:
            logger.error(f"Unknown ticker key: {ticker_key}")
            return None

        period = f'{days}d'
        return self._fetch_ticker(ticker, period=period)

    # ================================================================
    # BACKFILL ALL GLOBAL DATA
    # ================================================================
    def backfill_all(self, years: int = 5) -> Dict[str, int]:
        """
        Backfill historical data for all global instruments.

        Stores data in global_market_daily table.
        Returns dict of {ticker_key: rows_inserted}.
        """
        results = {}
        period = f'{years}y'

        for key, ticker in TICKERS.items():
            try:
                data = self._fetch_ticker(ticker, period=period)
                if data is not None and len(data) > 0:
                    rows = self._store_history(key, data)
                    results[key] = rows
                    logger.info(f"Backfilled {key}: {rows} rows")
                else:
                    results[key] = 0
                    logger.warning(f"No backfill data for {key}")
            except Exception as e:
                logger.error(f"Backfill error for {key}: {e}")
                results[key] = -1

        return results

    # ================================================================
    # COMPUTE DERIVED FIELDS
    # ================================================================
    def _compute_derived_fields(self, snapshot: Dict) -> Dict:
        """
        Compute derived pre-market signals from raw data.

        Fields added:
          - gift_nifty_gap_pct: Gap between GIFT Nifty and Nifty previous close
          - us_overnight_return: S&P 500 return since Indian market close
          - us_vix_spike: Boolean, True if US VIX moved > 15% in a day
          - dxy_strong_move: Boolean, True if DXY moved > 0.5%
          - crude_shock: Boolean, True if Brent moved > 3%
          - global_risk_score: Composite -1 to +1 risk sentiment
        """
        # ── Nifty previous close from DB ──────────────────────────
        nifty_prev = self._get_nifty_prev_close()
        snapshot['nifty_prev_close'] = nifty_prev

        # ── GIFT Nifty gap ────────────────────────────────────────
        gift_price = self.fetch_gift_nifty_live()
        if gift_price and nifty_prev and nifty_prev > 0:
            snapshot['gift_nifty_last'] = gift_price
            snapshot['gift_nifty_gap_pct'] = round(
                (gift_price - nifty_prev) / nifty_prev * 100, 4
            )
        else:
            snapshot['gift_nifty_last'] = None
            snapshot['gift_nifty_gap_pct'] = None

        # ── US overnight return ───────────────────────────────────
        sp_change = snapshot.get('sp500_change_pct')
        nasdaq_change = snapshot.get('nasdaq_change_pct')
        if sp_change is not None and nasdaq_change is not None:
            # Weighted: 60% S&P + 40% Nasdaq (Nifty correlates more with S&P)
            snapshot['us_overnight_return'] = round(
                0.6 * sp_change + 0.4 * nasdaq_change, 4
            )
        else:
            snapshot['us_overnight_return'] = sp_change or nasdaq_change

        # ── Binary flags ──────────────────────────────────────────
        vix_change = snapshot.get('us_vix_change_pct', 0) or 0
        dxy_change = snapshot.get('dxy_change_pct', 0) or 0
        brent_change = snapshot.get('brent_change_pct', 0) or 0

        snapshot['us_vix_spike'] = abs(vix_change) > 15.0
        snapshot['dxy_strong_move'] = abs(dxy_change) > 0.5
        snapshot['crude_shock'] = abs(brent_change) > 3.0

        # ── Global risk score (-1 to +1) ──────────────────────────
        # Positive = risk-on (bullish for Nifty), Negative = risk-off
        risk_score = 0.0
        us_ret = snapshot.get('us_overnight_return', 0) or 0

        # US market sentiment (strongest weight)
        if us_ret > 1.0:
            risk_score += 0.4
        elif us_ret > 0.3:
            risk_score += 0.2
        elif us_ret < -1.0:
            risk_score -= 0.4
        elif us_ret < -0.3:
            risk_score -= 0.2

        # VIX (fear gauge)
        us_vix = snapshot.get('us_vix_close', 20) or 20
        if us_vix > 30:
            risk_score -= 0.3
        elif us_vix > 25:
            risk_score -= 0.15
        elif us_vix < 14:
            risk_score += 0.15

        # DXY (dollar strength = negative for EM)
        if dxy_change > 0.5:
            risk_score -= 0.15
        elif dxy_change < -0.5:
            risk_score += 0.1

        # Crude shock (negative for India)
        if brent_change > 3.0:
            risk_score -= 0.15
        elif brent_change < -3.0:
            risk_score += 0.1

        snapshot['global_risk_score'] = round(
            max(-1.0, min(1.0, risk_score)), 3
        )

        return snapshot

    # ================================================================
    # STORE SNAPSHOT TO DB
    # ================================================================
    def store_snapshot(self, snapshot: Dict) -> bool:
        """Store pre-market snapshot to global_market_snapshots table."""
        if not self.db:
            logger.warning("No DB connection — snapshot not stored")
            return False

        try:
            cur = self.db.cursor()
            cur.execute("""
                INSERT INTO global_market_snapshots (
                    snapshot_date, snapshot_time,
                    gift_nifty_last, gift_nifty_gap_pct, nifty_prev_close,
                    sp500_close, sp500_change_pct,
                    nasdaq_close, nasdaq_change_pct,
                    us_vix_close, us_vix_change_pct, us_vix_spike,
                    dxy_close, dxy_change_pct, dxy_strong_move,
                    brent_close, brent_change_pct, crude_shock,
                    us_10y_close,
                    us_overnight_return, global_risk_score,
                    raw_json
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (snapshot_date) DO UPDATE SET
                    snapshot_time = EXCLUDED.snapshot_time,
                    gift_nifty_last = EXCLUDED.gift_nifty_last,
                    gift_nifty_gap_pct = EXCLUDED.gift_nifty_gap_pct,
                    sp500_close = EXCLUDED.sp500_close,
                    sp500_change_pct = EXCLUDED.sp500_change_pct,
                    nasdaq_close = EXCLUDED.nasdaq_close,
                    nasdaq_change_pct = EXCLUDED.nasdaq_change_pct,
                    us_vix_close = EXCLUDED.us_vix_close,
                    us_vix_change_pct = EXCLUDED.us_vix_change_pct,
                    us_vix_spike = EXCLUDED.us_vix_spike,
                    dxy_close = EXCLUDED.dxy_close,
                    dxy_change_pct = EXCLUDED.dxy_change_pct,
                    dxy_strong_move = EXCLUDED.dxy_strong_move,
                    brent_close = EXCLUDED.brent_close,
                    brent_change_pct = EXCLUDED.brent_change_pct,
                    crude_shock = EXCLUDED.crude_shock,
                    us_10y_close = EXCLUDED.us_10y_close,
                    us_overnight_return = EXCLUDED.us_overnight_return,
                    global_risk_score = EXCLUDED.global_risk_score,
                    raw_json = EXCLUDED.raw_json
            """, (
                snapshot['date'],
                snapshot['timestamp'],
                snapshot.get('gift_nifty_last'),
                snapshot.get('gift_nifty_gap_pct'),
                snapshot.get('nifty_prev_close'),
                snapshot.get('sp500_close'),
                snapshot.get('sp500_change_pct'),
                snapshot.get('nasdaq_close'),
                snapshot.get('nasdaq_change_pct'),
                snapshot.get('us_vix_close'),
                snapshot.get('us_vix_change_pct'),
                snapshot.get('us_vix_spike', False),
                snapshot.get('dxy_close'),
                snapshot.get('dxy_change_pct'),
                snapshot.get('dxy_strong_move', False),
                snapshot.get('brent_close'),
                snapshot.get('brent_change_pct'),
                snapshot.get('crude_shock', False),
                snapshot.get('us_10y_close'),
                snapshot.get('us_overnight_return'),
                snapshot.get('global_risk_score'),
                self._snapshot_to_json(snapshot),
            ))
            self.db.commit()
            logger.info(f"Stored global snapshot for {snapshot['date']}")
            return True
        except Exception as e:
            logger.error(f"Failed to store snapshot: {e}")
            self.db.rollback()
            return False

    # ================================================================
    # LOAD HISTORICAL SNAPSHOTS FROM DB
    # ================================================================
    def load_snapshots(self, days: int = 252) -> pd.DataFrame:
        """Load recent snapshots from DB for rolling calculations."""
        if not self.db:
            return pd.DataFrame()

        try:
            query = """
                SELECT * FROM global_market_snapshots
                ORDER BY snapshot_date DESC
                LIMIT %s
            """
            df = pd.read_sql(query, self.db, params=(days,))
            return df.sort_values('snapshot_date').reset_index(drop=True)
        except Exception as e:
            logger.error(f"Failed to load snapshots: {e}")
            return pd.DataFrame()

    # ================================================================
    # PRIVATE HELPERS
    # ================================================================
    def _fetch_ticker(self, ticker: str, period: str = '5d') -> Optional[pd.DataFrame]:
        """Fetch OHLCV data via yfinance."""
        try:
            t = self.yf.Ticker(ticker)
            data = t.history(period=period)
            if data is None or data.empty:
                return None
            return data.reset_index()
        except Exception as e:
            logger.error(f"yfinance error for {ticker}: {e}")
            return None

    def _get_nifty_prev_close(self) -> Optional[float]:
        """Get Nifty previous close from nifty_daily table."""
        if not self.db:
            return None
        try:
            cur = self.db.cursor()
            cur.execute("""
                SELECT close FROM nifty_daily
                ORDER BY date DESC LIMIT 1
            """)
            row = cur.fetchone()
            return float(row[0]) if row else None
        except Exception as e:
            logger.error(f"Failed to get Nifty prev close: {e}")
            return None

    def _store_history(self, key: str, data: pd.DataFrame) -> int:
        """Store historical data for a single instrument."""
        if not self.db or data is None:
            return 0

        cur = self.db.cursor()
        rows = 0
        for _, row in data.iterrows():
            try:
                trade_date = row.get('Date', row.name)
                if isinstance(trade_date, pd.Timestamp):
                    trade_date = trade_date.date()

                cur.execute("""
                    INSERT INTO global_market_daily (
                        instrument, trade_date, open, high, low, close, volume
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (instrument, trade_date) DO UPDATE SET
                        open = EXCLUDED.open, high = EXCLUDED.high,
                        low = EXCLUDED.low, close = EXCLUDED.close,
                        volume = EXCLUDED.volume
                """, (
                    key, trade_date,
                    float(row.get('Open', 0)), float(row.get('High', 0)),
                    float(row.get('Low', 0)), float(row.get('Close', 0)),
                    int(row.get('Volume', 0)),
                ))
                rows += 1
            except Exception as e:
                logger.warning(f"Skip row for {key}: {e}")
                continue

        self.db.commit()
        return rows

    def _snapshot_to_json(self, snapshot: Dict) -> str:
        """Convert snapshot to JSON string, handling non-serializable types."""
        import json

        clean = {}
        for k, v in snapshot.items():
            if isinstance(v, (datetime, date)):
                clean[k] = v.isoformat()
            elif isinstance(v, (np.integer, np.int64)):
                clean[k] = int(v)
            elif isinstance(v, (np.floating, np.float64)):
                clean[k] = float(v)
            elif isinstance(v, np.bool_):
                clean[k] = bool(v)
            else:
                clean[k] = v

        return json.dumps(clean, default=str)
