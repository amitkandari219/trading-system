"""
Unified data provider — loads market data and computes indicators.

Wraps data.nifty_loader and backtest.indicators into a single interface
that the orchestrator and signals consume.

Usage:
    from core.data_provider import DataProvider
    dp = DataProvider(db_connection=conn)
    history = dp.load_history(days=252)
    ctx = dp.market_context()
"""

import logging
from datetime import date, timedelta
from typing import Dict, Optional

import numpy as np
import pandas as pd

from config import settings

logger = logging.getLogger(__name__)


class DataProvider:
    """Loads Nifty daily data from DB and computes indicators."""

    def __init__(self, db_connection=None, cache_ttl_seconds: int = 300):
        self._db = db_connection
        self._cache: Optional[pd.DataFrame] = None
        self._cache_ts: Optional[float] = None
        self._cache_ttl = cache_ttl_seconds
        self._market_ctx: Optional[Dict] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_history(self, days: Optional[int] = 252) -> pd.DataFrame:
        """
        Load Nifty daily OHLCV + VIX from the database.

        Returns DataFrame with columns:
            [date, open, high, low, close, volume, india_vix]
        plus computed indicator columns when add_indicators=True.
        """
        import time
        now = time.time()
        if self._cache is not None and (now - self._cache_ts) < self._cache_ttl:
            return self._cache.copy()

        from data.nifty_loader import load_nifty_history
        df = load_nifty_history(self._db, days=days)

        # Add technical indicators
        try:
            from backtest.indicators import add_all_indicators
            df = add_all_indicators(df)
        except Exception as e:
            logger.warning("Could not compute indicators: %s", e)

        self._cache = df
        self._cache_ts = now
        return df.copy()

    def market_context(self, as_of: Optional[date] = None) -> Dict:
        """
        Build a market context dict for the given date (default: latest).

        Keys:
            spot_price, india_vix, day_of_week, prev_close,
            sma_50, rsi_14, adx_14, atr_14, volume
        """
        df = self.load_history()
        if df.empty:
            return self._empty_context()

        if as_of is not None:
            if 'date' in df.columns:
                mask = df['date'].dt.date <= as_of if hasattr(df['date'].iloc[0], 'date') else df['date'] <= pd.Timestamp(as_of)
                df = df[mask]
        if df.empty:
            return self._empty_context()

        row = df.iloc[-1]
        dt = row.get('date', date.today())
        if hasattr(dt, 'date'):
            dt = dt.date()

        ctx = {
            'spot_price': float(row.get('close', 0)),
            'india_vix': float(row.get('india_vix', 15)),
            'day_of_week': dt.weekday(),
            'prev_close': float(df.iloc[-2]['close']) if len(df) > 1 else float(row.get('close', 0)),
            'sma_50': float(row.get('sma_50', row.get('close', 0))),
            'rsi_14': float(row.get('rsi_14', 50)),
            'adx_14': float(row.get('adx_14', 20)),
            'atr_14': float(row.get('atr_14', 0)),
            'volume': float(row.get('volume', 0)),
            'date': dt,
        }
        self._market_ctx = ctx
        return ctx

    def invalidate_cache(self):
        """Force a fresh DB read on next call."""
        self._cache = None
        self._cache_ts = None

    def handles_missing_data(self) -> bool:
        """Return True if the provider returns a valid (possibly empty) result for missing dates."""
        return True

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_context() -> Dict:
        return {
            'spot_price': 0.0,
            'india_vix': 15.0,
            'day_of_week': 0,
            'prev_close': 0.0,
            'sma_50': 0.0,
            'rsi_14': 50.0,
            'adx_14': 20.0,
            'atr_14': 0.0,
            'volume': 0.0,
            'date': date.today(),
        }
