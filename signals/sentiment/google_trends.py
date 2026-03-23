"""
Google Trends Fear Gauge Signal.

Tracks search interest for panic-related terms ("nifty crash",
"stock market crash india", "market correction").  Spikes in search
volume historically mark retail panic — a contrarian BUY signal that
identifies bottoms within 1-2 weeks.

Signal logic:
    fear_index = google_trends_fear column (0-100 scale)
    rolling_mean = 30-day SMA of fear_index

    fear_index > 2.0 * rolling_mean  -> LONG  (retail panic = contrarian buy)
    fear_index < 0.5 * rolling_mean  -> SHORT (complacency = caution)
    otherwise                        -> no signal

    Strength is proportional to how far fear_index deviates from its mean.

Column required: google_trends_fear (float, 0-100)
"""

import logging
import math
from datetime import date
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ================================================================
# CONSTANTS
# ================================================================

SIGNAL_ID = 'GOOGLE_TRENDS_FEAR'

# Rolling window for baseline
ROLLING_WINDOW = 30

# Thresholds (multiples of rolling mean)
PANIC_THRESHOLD = 2.0       # fear > 2x mean -> panic -> contrarian LONG
COMPLACENCY_THRESHOLD = 0.5 # fear < 0.5x mean -> complacency -> caution SHORT

# Strength scaling
MAX_STRENGTH = 0.95
MIN_STRENGTH = 0.10

# Size modifiers
PANIC_SIZE_BOOST = 1.2
COMPLACENCY_SIZE_REDUCE = 0.8
BASE_SIZE = 1.0

COLUMN = 'google_trends_fear'


# ================================================================
# HELPERS
# ================================================================

def _safe_float(val: Any, default: float = float('nan')) -> float:
    """Safely cast to float."""
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


# ================================================================
# SIGNAL CLASS
# ================================================================

class GoogleTrendsFearSignal:
    """
    Contrarian signal based on Google Trends search interest for
    panic-related terms.  Retail fear spikes mark bottoms; extreme
    complacency warns of tops.
    """

    SIGNAL_ID = SIGNAL_ID

    def __init__(self) -> None:
        logger.info('GoogleTrendsFearSignal initialised')

    def evaluate(self, df: pd.DataFrame, trade_date: date) -> Optional[Dict]:
        """
        Evaluate Google Trends fear gauge.

        Parameters
        ----------
        df         : DataFrame with at least `google_trends_fear` column
                     and a DatetimeIndex or 'date' column.
        trade_date : The date to evaluate.

        Returns
        -------
        dict with signal_id, direction, strength, price, reason, metadata
        or None if no signal / missing data.
        """
        try:
            return self._evaluate_inner(df, trade_date)
        except Exception as e:
            logger.error('GoogleTrendsFearSignal.evaluate error: %s', e, exc_info=True)
            return None

    def _evaluate_inner(self, df: pd.DataFrame, trade_date: date) -> Optional[Dict]:
        # ── Validate column exists ───────────────────────────────
        if df is None or df.empty:
            logger.debug('Empty dataframe')
            return None

        if COLUMN not in df.columns:
            logger.debug('Column %s not found in dataframe', COLUMN)
            return None

        # ── Get data up to trade_date ────────────────────────────
        df = df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
            else:
                logger.debug('No date index or date column')
                return None

        df = df.sort_index()
        mask = df.index.date <= trade_date
        df_hist = df.loc[mask]

        if len(df_hist) < ROLLING_WINDOW + 1:
            logger.debug('Insufficient history: %d rows (need %d)',
                         len(df_hist), ROLLING_WINDOW + 1)
            return None

        # ── Compute rolling mean and current value ───────────────
        fear_series = df_hist[COLUMN].astype(float)
        rolling_mean = fear_series.rolling(window=ROLLING_WINDOW).mean()

        current_fear = _safe_float(fear_series.iloc[-1])
        current_mean = _safe_float(rolling_mean.iloc[-1])

        if math.isnan(current_fear) or math.isnan(current_mean) or current_mean <= 0:
            logger.debug('Invalid fear=%.2f or mean=%.2f', current_fear, current_mean)
            return None

        ratio = current_fear / current_mean

        # ── Get price for output ─────────────────────────────────
        price = float('nan')
        for col in ('close', 'Close', 'price', 'nifty_close'):
            if col in df_hist.columns:
                price = _safe_float(df_hist[col].iloc[-1])
                break

        # ── Signal logic ─────────────────────────────────────────
        direction = None
        strength = 0.0
        size_modifier = BASE_SIZE

        if ratio >= PANIC_THRESHOLD:
            # Retail panic -> contrarian LONG
            direction = 'LONG'
            # Strength scales with how extreme the spike is (2x = base, 5x = max)
            strength = min(MAX_STRENGTH, 0.4 + (ratio - PANIC_THRESHOLD) * 0.15)
            strength = max(MIN_STRENGTH, strength)
            size_modifier = PANIC_SIZE_BOOST

        elif ratio <= COMPLACENCY_THRESHOLD:
            # Complacency -> caution SHORT
            direction = 'SHORT'
            # Strength scales inversely (0.5x = base, 0.1x = max)
            strength = min(MAX_STRENGTH, 0.3 + (COMPLACENCY_THRESHOLD - ratio) * 0.8)
            strength = max(MIN_STRENGTH, strength)
            size_modifier = COMPLACENCY_SIZE_REDUCE

        else:
            # No signal in neutral zone
            return None

        reason_parts = [
            SIGNAL_ID,
            f"fear_index={current_fear:.1f}",
            f"30d_mean={current_mean:.1f}",
            f"ratio={ratio:.2f}x",
            f"{'PANIC' if direction == 'LONG' else 'COMPLACENCY'}",
        ]

        logger.info('%s signal: %s on %s ratio=%.2f strength=%.3f',
                     self.SIGNAL_ID, direction, trade_date, ratio, strength)

        return {
            'signal_id': self.SIGNAL_ID,
            'direction': direction,
            'strength': round(strength, 4),
            'price': round(price, 2) if not math.isnan(price) else None,
            'reason': ' | '.join(reason_parts),
            'metadata': {
                'fear_index': round(current_fear, 2),
                'rolling_mean_30d': round(current_mean, 2),
                'ratio_to_mean': round(ratio, 4),
                'size_modifier': size_modifier,
                'trade_date': trade_date.isoformat(),
            },
        }

    def reset(self) -> None:
        """Reset internal state."""
        pass

    def __repr__(self) -> str:
        return f"GoogleTrendsFearSignal(signal_id='{self.SIGNAL_ID}')"
