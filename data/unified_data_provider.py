"""
Unified data provider — single entry point for ALL market data access.

Consolidates the scattered SQL queries in signal_compute.py, enhanced_daily_run.py,
and nifty_loader.py into one cached provider. Every signal and overlay reads from
this class instead of hitting the DB directly.

Usage:
    from data.unified_data_provider import UnifiedDataProvider

    provider = UnifiedDataProvider(db_conn=conn)
    df = provider.get_nifty_daily(lookback=500)
    ctx = provider.get_market_context()
    vix = provider.get_india_vix()
"""

import logging
from datetime import date, datetime, timedelta
from typing import Dict, Optional

import numpy as np
import pandas as pd

from backtest.indicators import add_all_indicators
from config.unified_config import (
    ADX_TREND_THRESHOLD,
    DATABASE_DSN,
    NIFTY_LOT_SIZE,
    VIX_EXTREME,
    VIX_HIGH,
    VIX_LOW,
    VIX_NORMAL,
    WEEKLY_EXPIRY_DAY,
)

logger = logging.getLogger(__name__)


def _safe_float(val, default=0.0):
    """Extract float from a pandas value, handling NaN/None."""
    if val is None:
        return default
    try:
        f = float(val)
        return default if np.isnan(f) else f
    except (TypeError, ValueError):
        return default


class UnifiedDataProvider:
    """
    Central data provider for the trading system.

    Loads market data from the nifty_daily table, computes all indicators
    via backtest.indicators.add_all_indicators, and caches results per
    session so repeated calls don't re-query the database.

    Parameters
    ----------
    db_conn : psycopg2 connection
        Active database connection.
    kite : KiteConnect instance, optional
        If provided, can be used for real-time data augmentation.
        Currently unused — reserved for live trading integration.
    """

    def __init__(self, db_conn, kite=None):
        self.conn = db_conn
        self.kite = kite

        # Session cache — cleared only by calling invalidate_cache()
        self._nifty_daily_cache: Dict[int, pd.DataFrame] = {}
        self._market_context_cache: Dict[str, dict] = {}
        self._vix_cache: Optional[float] = None
        self._open_positions_cache: Dict[str, list] = {}

    # ================================================================
    # NIFTY DAILY OHLCV + INDICATORS
    # ================================================================

    def get_nifty_daily(self, lookback: int = 500,
                        as_of: date = None) -> pd.DataFrame:
        """
        Load Nifty daily OHLCV from nifty_daily table and add all indicators.

        Parameters
        ----------
        lookback : int
            Number of most recent trading days to load. Default 500
            (roughly 2 years — enough for SMA_200 + warmup).
        as_of : date, optional
            Load data up to this date. Defaults to today.

        Returns
        -------
        pd.DataFrame
            Sorted ascending by date with all indicator columns added.
            Columns include: date, open, high, low, close, volume,
            india_vix, sma_*, ema_*, rsi_*, macd, atr_*, adx_14,
            stoch_k, bb_*, pivot, r1/s1/r2/s2, etc.

        Raises
        ------
        RuntimeError
            If the nifty_daily table is empty or has < 200 rows.
        """
        as_of = as_of or date.today()
        cache_key = (lookback, as_of)

        if cache_key in self._nifty_daily_cache:
            return self._nifty_daily_cache[cache_key]

        df = self._query_nifty_daily(lookback, as_of)

        if df is None or len(df) < 200:
            raise RuntimeError(
                f"Insufficient market data: got {len(df) if df is not None else 0} rows, "
                f"need >= 200. Run: python -m data.nifty_loader --init"
            )

        # Compute all standard indicators
        df = add_all_indicators(df)

        self._nifty_daily_cache[cache_key] = df
        logger.debug(
            f"Loaded {len(df)} nifty_daily rows up to {as_of} "
            f"with {len(df.columns)} indicator columns"
        )
        return df

    def _query_nifty_daily(self, lookback: int,
                           as_of: date) -> Optional[pd.DataFrame]:
        """Raw SQL query for nifty_daily — no caching, no indicators."""
        try:
            df = pd.read_sql(
                "SELECT date, open, high, low, close, volume, india_vix "
                "FROM nifty_daily WHERE date <= %s "
                "ORDER BY date DESC LIMIT %s",
                self.conn,
                params=(as_of, lookback),
            )
            if df.empty:
                return None
            df = df.sort_values('date').reset_index(drop=True)
            df['date'] = pd.to_datetime(df['date'])
            return df
        except Exception as e:
            logger.error(f"Failed to query nifty_daily: {e}")
            return None

    # ================================================================
    # VIX
    # ================================================================

    def get_india_vix(self, as_of: date = None) -> float:
        """
        Get the latest India VIX value.

        Reads from the most recent row in nifty_daily (india_vix column).
        Cached after first call.

        Returns
        -------
        float
            India VIX value, or 0.0 if unavailable.
        """
        if self._vix_cache is not None:
            return self._vix_cache

        as_of = as_of or date.today()
        try:
            cur = self.conn.cursor()
            cur.execute(
                "SELECT india_vix FROM nifty_daily "
                "WHERE india_vix IS NOT NULL AND date <= %s "
                "ORDER BY date DESC LIMIT 1",
                (as_of,),
            )
            row = cur.fetchone()
            self._vix_cache = float(row[0]) if row and row[0] else 0.0
        except Exception as e:
            logger.error(f"Failed to fetch india_vix: {e}")
            self._vix_cache = 0.0

        return self._vix_cache

    def get_vix_regime(self, vix: float = None) -> str:
        """
        Classify VIX into regime bucket.

        Returns one of: 'LOW', 'NORMAL', 'HIGH', 'EXTREME'.
        """
        if vix is None:
            vix = self.get_india_vix()

        if vix >= VIX_EXTREME:
            return 'EXTREME'
        elif vix >= VIX_HIGH:
            return 'HIGH'
        elif vix >= VIX_NORMAL:
            return 'NORMAL'
        else:
            return 'LOW'

    # ================================================================
    # MARKET CONTEXT — complete dict for any signal
    # ================================================================

    def get_market_context(self, as_of: date = None,
                           lookback: int = 500) -> dict:
        """
        Build a complete market context dict with ALL fields any signal needs.

        This is the single gateway between raw data and signal computation.
        Every signal receives this dict — no signal should query the DB directly.

        Parameters
        ----------
        as_of : date, optional
            Compute context as of this date. Defaults to today.
        lookback : int
            Days of history to load for indicator computation.

        Returns
        -------
        dict with keys:
            # Price data
            'date', 'close', 'open', 'high', 'low', 'volume',
            'prev_close', 'prev_open', 'prev_high', 'prev_low', 'prev_volume',

            # Moving averages
            'sma_5' .. 'sma_200', 'ema_5' .. 'ema_200',

            # Oscillators
            'rsi_7', 'rsi_14', 'rsi_21', 'stoch_k', 'stoch_d',
            'stoch_k_5', 'stoch_d_5', 'macd', 'macd_signal', 'macd_hist',
            'connors_rsi',

            # Volatility
            'atr_7', 'atr_14', 'atr_20', 'india_vix', 'vix_regime',
            'hvol_6', 'hvol_20', 'hvol_100', 'bb_upper', 'bb_lower',
            'bb_pct_b', 'bb_bandwidth',

            # Trend
            'adx_14', 'is_trending', 'trend_direction',

            # Pivots
            'pivot', 'r1', 's1', 'r2', 's2', 'r3', 's3',

            # Calendar
            'day_of_week', 'is_monday', 'is_friday',
            'is_expiry_week', 'days_to_weekly_expiry',
            'is_month_end', 'month',

            # Regime / structure
            'hurst_100', 'vol_ratio_20', 'price_pos_20',

            # Bar properties
            'body', 'body_pct', 'upper_wick', 'lower_wick', 'range',
            'returns', 'log_returns',

            # Full DataFrame reference (for signals needing history)
            'df': pd.DataFrame,
            'today_row': pd.Series,
            'yesterday_row': pd.Series,
        """
        as_of = as_of or date.today()
        cache_key = f"{as_of}_{lookback}"

        if cache_key in self._market_context_cache:
            return self._market_context_cache[cache_key]

        df = self.get_nifty_daily(lookback=lookback, as_of=as_of)
        today = df.iloc[-1]
        yesterday = df.iloc[-2]

        as_of_dt = pd.Timestamp(as_of)

        # Calendar computations
        dow = as_of_dt.dayofweek  # 0=Mon, 6=Sun
        days_to_expiry = (WEEKLY_EXPIRY_DAY - dow) % 7
        if days_to_expiry == 0:
            days_to_expiry = 0  # Today is expiry
        is_expiry_week = days_to_expiry <= 4  # Always true for weekly, but meaningful for monthly

        # Month-end detection: is this the last trading day of the month?
        next_trading_day = as_of_dt + pd.tseries.offsets.BDay(1)
        is_month_end = next_trading_day.month != as_of_dt.month

        # VIX
        vix_val = _safe_float(today.get('india_vix'))
        vix_regime = self.get_vix_regime(vix_val)

        # Trend direction from SMA crossovers
        sma_50 = _safe_float(today.get('sma_50'))
        sma_200 = _safe_float(today.get('sma_200'))
        close_val = _safe_float(today.get('close'))

        if sma_50 > 0 and sma_200 > 0:
            if close_val > sma_50 > sma_200:
                trend_direction = 'BULL'
            elif close_val < sma_50 < sma_200:
                trend_direction = 'BEAR'
            else:
                trend_direction = 'NEUTRAL'
        else:
            trend_direction = 'NEUTRAL'

        adx_val = _safe_float(today.get('adx_14'))
        is_trending = adx_val >= ADX_TREND_THRESHOLD

        ctx = {
            # Metadata
            'date': as_of,
            'as_of_str': str(as_of),

            # Price data
            'close': close_val,
            'open': _safe_float(today.get('open')),
            'high': _safe_float(today.get('high')),
            'low': _safe_float(today.get('low')),
            'volume': _safe_float(today.get('volume')),
            'prev_close': _safe_float(today.get('prev_close')),
            'prev_open': _safe_float(today.get('prev_open')),
            'prev_high': _safe_float(today.get('prev_high')),
            'prev_low': _safe_float(today.get('prev_low')),
            'prev_volume': _safe_float(today.get('prev_volume')),

            # Moving averages
            'sma_5': _safe_float(today.get('sma_5')),
            'sma_10': _safe_float(today.get('sma_10')),
            'sma_20': _safe_float(today.get('sma_20')),
            'sma_40': _safe_float(today.get('sma_40')),
            'sma_50': sma_50,
            'sma_80': _safe_float(today.get('sma_80')),
            'sma_100': _safe_float(today.get('sma_100')),
            'sma_200': sma_200,
            'ema_5': _safe_float(today.get('ema_5')),
            'ema_10': _safe_float(today.get('ema_10')),
            'ema_20': _safe_float(today.get('ema_20')),
            'ema_50': _safe_float(today.get('ema_50')),
            'ema_100': _safe_float(today.get('ema_100')),
            'ema_200': _safe_float(today.get('ema_200')),

            # RSI
            'rsi_7': _safe_float(today.get('rsi_7')),
            'rsi_14': _safe_float(today.get('rsi_14')),
            'rsi_21': _safe_float(today.get('rsi_21')),
            'rsi_2': _safe_float(today.get('rsi_2')),
            'rsi_3': _safe_float(today.get('rsi_3')),
            'connors_rsi': _safe_float(today.get('connors_rsi')),

            # Stochastic
            'stoch_k': _safe_float(today.get('stoch_k')),
            'stoch_d': _safe_float(today.get('stoch_d')),
            'stoch_k_5': _safe_float(today.get('stoch_k_5')),
            'stoch_d_5': _safe_float(today.get('stoch_d_5')),

            # MACD
            'macd': _safe_float(today.get('macd')),
            'macd_signal': _safe_float(today.get('macd_signal')),
            'macd_hist': _safe_float(today.get('macd_hist')),

            # Volatility
            'atr_7': _safe_float(today.get('atr_7')),
            'atr_14': _safe_float(today.get('atr_14')),
            'atr_20': _safe_float(today.get('atr_20')),
            'india_vix': vix_val,
            'vix_regime': vix_regime,
            'hvol_6': _safe_float(today.get('hvol_6')),
            'hvol_20': _safe_float(today.get('hvol_20')),
            'hvol_100': _safe_float(today.get('hvol_100')),
            'bb_upper': _safe_float(today.get('bb_upper')),
            'bb_middle': _safe_float(today.get('bb_middle')),
            'bb_lower': _safe_float(today.get('bb_lower')),
            'bb_pct_b': _safe_float(today.get('bb_pct_b')),
            'bb_bandwidth': _safe_float(today.get('bb_bandwidth')),

            # Trend
            'adx_14': adx_val,
            'is_trending': is_trending,
            'trend_direction': trend_direction,

            # Pivots
            'pivot': _safe_float(today.get('pivot')),
            'r1': _safe_float(today.get('r1')),
            's1': _safe_float(today.get('s1')),
            'r2': _safe_float(today.get('r2')),
            's2': _safe_float(today.get('s2')),
            'r3': _safe_float(today.get('r3')),
            's3': _safe_float(today.get('s3')),

            # Calendar
            'day_of_week': dow,
            'is_monday': dow == 0,
            'is_friday': dow == 4,
            'is_expiry_week': is_expiry_week,
            'days_to_weekly_expiry': days_to_expiry,
            'is_month_end': is_month_end,
            'month': as_of_dt.month,

            # Donchian
            'dc_upper': _safe_float(today.get('dc_upper')),
            'dc_lower': _safe_float(today.get('dc_lower')),
            'dc_middle': _safe_float(today.get('dc_middle')),

            # Regime / structure
            'hurst_100': _safe_float(today.get('hurst_100')),
            'vol_ratio_20': _safe_float(today.get('vol_ratio_20')),
            'price_pos_20': _safe_float(today.get('price_pos_20')),
            'zscore_20': _safe_float(today.get('zscore_20')),
            'zscore_50': _safe_float(today.get('zscore_50')),

            # Bar properties
            'body': _safe_float(today.get('body')),
            'body_pct': _safe_float(today.get('body_pct')),
            'upper_wick': _safe_float(today.get('upper_wick')),
            'lower_wick': _safe_float(today.get('lower_wick')),
            'range': _safe_float(today.get('range')),
            'returns': _safe_float(today.get('returns')),
            'log_returns': _safe_float(today.get('log_returns')),

            # Chandelier stops
            'chandelier_long': _safe_float(today.get('chandelier_long')),
            'chandelier_short': _safe_float(today.get('chandelier_short')),

            # Full DataFrame + row references (for signals needing history)
            'df': df,
            'today_row': today,
            'yesterday_row': yesterday,
        }

        self._market_context_cache[cache_key] = ctx
        logger.debug(f"Built market context for {as_of}: {len(ctx)} fields")
        return ctx

    # ================================================================
    # OPEN POSITIONS
    # ================================================================

    def get_open_positions(self, trade_type: str = 'PAPER') -> list:
        """
        Load open positions (no exit_date) from trades table.

        Parameters
        ----------
        trade_type : str
            One of 'PAPER', 'SHADOW', 'OVERLAY'.

        Returns
        -------
        list of dict
            Each dict has: trade_id, signal_id, direction, entry_price, entry_date.
        """
        if trade_type in self._open_positions_cache:
            return self._open_positions_cache[trade_type]

        try:
            cur = self.conn.cursor()
            cur.execute(
                """
                SELECT trade_id, signal_id, direction, entry_price, entry_date
                FROM trades
                WHERE trade_type = %s AND exit_date IS NULL
                ORDER BY entry_date
                """,
                (trade_type,),
            )
            rows = cur.fetchall()
            positions = [
                {
                    'trade_id': r[0],
                    'signal_id': r[1],
                    'direction': r[2],
                    'entry_price': float(r[3]) if r[3] else 0.0,
                    'entry_date': r[4],
                }
                for r in rows
            ]
        except Exception as e:
            logger.error(f"Failed to load open {trade_type} positions: {e}")
            positions = []

        self._open_positions_cache[trade_type] = positions
        return positions

    # ================================================================
    # CACHE MANAGEMENT
    # ================================================================

    def invalidate_cache(self):
        """Clear all cached data. Call between trading sessions."""
        self._nifty_daily_cache.clear()
        self._market_context_cache.clear()
        self._vix_cache = None
        self._open_positions_cache.clear()
        logger.debug("UnifiedDataProvider cache invalidated")

    def invalidate_positions_cache(self):
        """Clear only the positions cache (after a trade is recorded)."""
        self._open_positions_cache.clear()
