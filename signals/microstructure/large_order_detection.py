"""
Block / Iceberg Order Detection Signal.

Detects unusually large orders or sustained institutional buying/selling
pressure.  Large orders (> 500 lots) are tracked as a percentage of
total volume.  When institutional participation dominates, this signal
fires directionally.

Signal logic:
    large_buy_ratio  = large_buy_volume / total_volume
    large_sell_ratio = large_sell_volume / total_volume

    Filter: total volume > 1.5x 20-day average (confirming participation)

    large_buy_ratio  > 30%  -> LONG  (institutional buying)
    large_sell_ratio > 30%  -> SHORT (institutional selling)
    otherwise               -> no signal

Columns required: large_buy_volume, large_sell_volume, volume (or total_volume)
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

SIGNAL_ID = 'LARGE_ORDER_DETECTION'

# Volume filter
VOLUME_LOOKBACK = 20
VOLUME_THRESHOLD_MULTIPLIER = 1.5

# Large order ratio threshold
LARGE_ORDER_RATIO_THRESHOLD = 0.30   # 30% of total volume

# Strength scaling
MAX_STRENGTH = 0.90
MIN_STRENGTH = 0.20
BASE_STRENGTH = 0.45

# Size modifier
INSTITUTIONAL_SIZE_BOOST = 1.15
BASE_SIZE = 1.0

BUY_COL = 'large_buy_volume'
SELL_COL = 'large_sell_volume'


# ================================================================
# HELPERS
# ================================================================

def _safe_float(val: Any, default: float = float('nan')) -> float:
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _get_volume_col(df: pd.DataFrame) -> Optional[str]:
    """Find the volume column name."""
    for col in ('volume', 'Volume', 'total_volume', 'traded_volume'):
        if col in df.columns:
            return col
    return None


# ================================================================
# SIGNAL CLASS
# ================================================================

class LargeOrderDetectionSignal:
    """
    Detects institutional buying/selling pressure from large order
    flow data (orders > 500 lots).
    """

    SIGNAL_ID = SIGNAL_ID

    def __init__(self) -> None:
        logger.info('LargeOrderDetectionSignal initialised')

    def evaluate(self, df: pd.DataFrame, trade_date: date) -> Optional[Dict]:
        """
        Evaluate large order detection signal.

        Parameters
        ----------
        df         : DataFrame with large_buy_volume, large_sell_volume,
                     and volume columns.
        trade_date : The date to evaluate.

        Returns
        -------
        dict with signal_id, direction, strength, price, reason, metadata
        or None.
        """
        try:
            return self._evaluate_inner(df, trade_date)
        except Exception as e:
            logger.error('LargeOrderDetectionSignal.evaluate error: %s', e, exc_info=True)
            return None

    def _evaluate_inner(self, df: pd.DataFrame, trade_date: date) -> Optional[Dict]:
        if df is None or df.empty:
            return None

        # ── Validate required columns ────────────────────────────
        if BUY_COL not in df.columns or SELL_COL not in df.columns:
            logger.debug('Missing columns: %s or %s', BUY_COL, SELL_COL)
            return None

        vol_col = _get_volume_col(df)
        if vol_col is None:
            logger.debug('No volume column found')
            return None

        # ── Align to trade_date ──────────────────────────────────
        df = df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
            else:
                return None

        df = df.sort_index()
        mask = df.index.date <= trade_date
        df_hist = df.loc[mask]

        if len(df_hist) < VOLUME_LOOKBACK + 1:
            logger.debug('Insufficient history: %d rows', len(df_hist))
            return None

        # ── Volume filter ────────────────────────────────────────
        vol_series = df_hist[vol_col].astype(float)
        vol_20d_avg = vol_series.rolling(window=VOLUME_LOOKBACK).mean().iloc[-1]
        current_vol = _safe_float(vol_series.iloc[-1])

        if math.isnan(current_vol) or math.isnan(vol_20d_avg) or vol_20d_avg <= 0:
            return None

        if current_vol < vol_20d_avg * VOLUME_THRESHOLD_MULTIPLIER:
            logger.debug('Volume %.0f < %.1fx 20d avg %.0f — insufficient participation',
                         current_vol, VOLUME_THRESHOLD_MULTIPLIER, vol_20d_avg)
            return None

        # ── Compute large order ratios ───────────────────────────
        large_buy = _safe_float(df_hist[BUY_COL].iloc[-1])
        large_sell = _safe_float(df_hist[SELL_COL].iloc[-1])

        if math.isnan(large_buy) or math.isnan(large_sell):
            return None

        if current_vol <= 0:
            return None

        buy_ratio = large_buy / current_vol
        sell_ratio = large_sell / current_vol

        # ── Get price ────────────────────────────────────────────
        price = float('nan')
        for col in ('close', 'Close', 'price', 'nifty_close'):
            if col in df_hist.columns:
                price = _safe_float(df_hist[col].iloc[-1])
                break

        # ── Signal logic ─────────────────────────────────────────
        direction = None
        strength = 0.0
        size_modifier = BASE_SIZE
        label = ''

        if buy_ratio > LARGE_ORDER_RATIO_THRESHOLD:
            direction = 'LONG'
            # Strength: 30% ratio -> base, 60%+ -> max
            strength = min(MAX_STRENGTH,
                           BASE_STRENGTH + (buy_ratio - LARGE_ORDER_RATIO_THRESHOLD) * 1.5)
            size_modifier = INSTITUTIONAL_SIZE_BOOST
            label = 'INSTITUTIONAL_BUYING'

        elif sell_ratio > LARGE_ORDER_RATIO_THRESHOLD:
            direction = 'SHORT'
            strength = min(MAX_STRENGTH,
                           BASE_STRENGTH + (sell_ratio - LARGE_ORDER_RATIO_THRESHOLD) * 1.5)
            size_modifier = INSTITUTIONAL_SIZE_BOOST
            label = 'INSTITUTIONAL_SELLING'

        else:
            return None

        strength = max(MIN_STRENGTH, strength)

        vol_ratio = current_vol / vol_20d_avg

        reason_parts = [
            SIGNAL_ID,
            f"buy_ratio={buy_ratio:.1%}",
            f"sell_ratio={sell_ratio:.1%}",
            f"vol_ratio={vol_ratio:.2f}x",
            f"flow={label}",
        ]

        logger.info('%s signal: %s on %s buy_ratio=%.1f%% sell_ratio=%.1f%%',
                     self.SIGNAL_ID, direction, trade_date, buy_ratio * 100, sell_ratio * 100)

        return {
            'signal_id': self.SIGNAL_ID,
            'direction': direction,
            'strength': round(strength, 4),
            'price': round(price, 2) if not math.isnan(price) else None,
            'reason': ' | '.join(reason_parts),
            'metadata': {
                'large_buy_volume': round(large_buy, 0),
                'large_sell_volume': round(large_sell, 0),
                'total_volume': round(current_vol, 0),
                'buy_ratio': round(buy_ratio, 4),
                'sell_ratio': round(sell_ratio, 4),
                'volume_vs_20d_avg': round(vol_ratio, 4),
                'flow_label': label,
                'size_modifier': size_modifier,
                'trade_date': trade_date.isoformat(),
            },
        }

    def reset(self) -> None:
        pass

    def __repr__(self) -> str:
        return f"LargeOrderDetectionSignal(signal_id='{self.SIGNAL_ID}')"
