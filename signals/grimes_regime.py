"""
Grimes 4-phase market regime overlay.

Maps Grimes' market structure phases to signal-specific size multipliers:
  1. TRENDING_UP: ADX>25 + price>SMA50 + higher highs → boost trend signals
  2. TRENDING_DOWN: ADX>25 + price<SMA50 + lower lows → boost short signals
  3. RANGING: ADX<20 + price near SMA50 → boost MR signals, reduce trend
  4. VOLATILE: ATR expanding + VIX elevated → reduce all sizes

Different from the existing VIX regime (which only uses VIX levels).
Grimes uses ADX + price structure for more granular classification.
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Grimes regime → signal type → size multiplier
GRIMES_REGIME_MULTIPLIERS = {
    'TRENDING_UP': {
        'TREND_FOLLOW': 1.3,   # DRY_13 (breakout), GRIMES combos
        'MOMENTUM': 1.2,       # DRY_12 (price/vol divergence)
        'MEAN_REVERSION': 0.5, # DRY_8 (pivot), CHAN_AT (BB)
        'PIVOT': 1.0,          # DRY_16, DRY_20
        'COMBO': 1.2,          # Validated combos
    },
    'TRENDING_DOWN': {
        'TREND_FOLLOW': 1.3,   # Short entries boosted
        'MOMENTUM': 1.2,
        'MEAN_REVERSION': 0.5, # Don't fade strong downtrends
        'PIVOT': 0.8,
        'COMBO': 1.2,
    },
    'RANGING': {
        'TREND_FOLLOW': 0.5,   # Trend signals whipsaw in ranges
        'MOMENTUM': 0.7,
        'MEAN_REVERSION': 1.3, # MR thrives in ranges
        'PIVOT': 1.2,          # Pivot signals work well in ranges
        'COMBO': 1.0,
    },
    'VOLATILE': {
        'TREND_FOLLOW': 0.7,
        'MOMENTUM': 0.5,
        'MEAN_REVERSION': 0.3, # MR gets destroyed in volatile markets
        'PIVOT': 0.7,
        'COMBO': 0.7,
    },
}

# Map each signal to its type
SIGNAL_TYPE_MAP = {
    'KAUFMAN_DRY_20': 'PIVOT',
    'KAUFMAN_DRY_16': 'PIVOT',
    'KAUFMAN_DRY_12': 'MOMENTUM',
    'GUJRAL_DRY_8': 'MEAN_REVERSION',
    'GUJRAL_DRY_13': 'TREND_FOLLOW',
    'CANDLESTICK_DRY_0': 'MOMENTUM',
    'CHAN_AT_DRY_4': 'MEAN_REVERSION',
    'KAUFMAN_DRY_7': 'PIVOT',
    'GUJRAL_DRY_9': 'MEAN_REVERSION',
    'COMBO_GRIMES32_DRY12_SEQ5': 'COMBO',
    'COMBO_GRIMES60_DRY8_AND': 'COMBO',
    'COMBO_GRIMES50_DRY16_SEQ3': 'COMBO',
    'COMBO_GRIMES50_DRY16_SEQ5': 'COMBO',
}


def classify_grimes_regime(row) -> str:
    """
    Classify current market phase using Grimes' criteria.

    Args:
        row: pandas Series with adx_14, close, sma_50, atr_14, prev_high, prev_low,
             india_vix, hvol_6, hvol_100
    """
    adx = float(row.get('adx_14', 20)) if pd.notna(row.get('adx_14')) else 20
    close = float(row.get('close', 0))
    sma_50 = float(row.get('sma_50', close)) if pd.notna(row.get('sma_50')) else close
    vix = float(row.get('india_vix', 15)) if pd.notna(row.get('india_vix')) else 15
    hvol_6 = float(row.get('hvol_6', 0.15)) if pd.notna(row.get('hvol_6')) else 0.15
    hvol_100 = float(row.get('hvol_100', 0.15)) if pd.notna(row.get('hvol_100')) else 0.15

    # Volatile: VIX > 20 OR short-term vol expanding rapidly
    if vix > 20 or (hvol_100 > 0 and hvol_6 / hvol_100 > 1.5):
        return 'VOLATILE'

    # Trending: ADX > 25 with clear direction
    if adx > 25:
        if close > sma_50:
            return 'TRENDING_UP'
        else:
            return 'TRENDING_DOWN'

    # Ranging: ADX < 20, price near SMA50
    if adx < 20:
        return 'RANGING'

    # Transitional (ADX 20-25): use price position
    if close > sma_50 * 1.02:
        return 'TRENDING_UP'
    elif close < sma_50 * 0.98:
        return 'TRENDING_DOWN'
    return 'RANGING'


def get_grimes_multiplier(signal_id: str, row) -> float:
    """
    Get Grimes regime-based size multiplier for a signal.

    Args:
        signal_id: signal being sized
        row: current bar data

    Returns:
        float multiplier (0.3 - 1.3)
    """
    regime = classify_grimes_regime(row)
    signal_type = SIGNAL_TYPE_MAP.get(signal_id, 'PIVOT')
    multipliers = GRIMES_REGIME_MULTIPLIERS.get(regime, {})
    mult = multipliers.get(signal_type, 1.0)

    logger.debug(f"Grimes regime={regime}, signal={signal_id} type={signal_type} → {mult:.1f}x")
    return mult
