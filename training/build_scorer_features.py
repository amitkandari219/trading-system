"""
Feature extraction for signal intensity scorer.

Computes 10 dimensions at signal fire time (no lookahead).
Each dimension normalized 0-1.
"""

import numpy as np
import pandas as pd


def compute_scorer_features(signal_id, direction, today, yesterday, df_history,
                             active_signals=None, regime='RANGING'):
    """
    Compute 10-dimension feature vector at signal fire time.

    Args:
        signal_id: which signal fired
        direction: 'LONG' or 'SHORT'
        today: current bar (Series with indicators)
        yesterday: previous bar
        df_history: last 60 days of OHLCV for rolling calcs
        active_signals: list of other signals currently firing
        regime: current regime string

    Returns:
        dict of 10 features, each 0-1 normalized
    """
    features = {}

    # 1. Signal extremity — how far past threshold
    # Use RSI distance from 50 as proxy (higher = more extreme)
    rsi = float(today.get('rsi_14', 50)) if pd.notna(today.get('rsi_14')) else 50
    features['dim1_extremity'] = min(1.0, abs(rsi - 50) / 30)

    # 2. Regime alignment — does signal match regime?
    regime_match = {
        'KAUFMAN_DRY_20': {'TRENDING': 1.0, 'RANGING': 0.3, 'HIGH_VOL': 0.0},
        'KAUFMAN_DRY_16': {'TRENDING': 1.0, 'RANGING': 0.3, 'HIGH_VOL': 0.0},
        'KAUFMAN_DRY_12': {'TRENDING': 1.0, 'RANGING': 0.3, 'HIGH_VOL': 0.0},
        'GUJRAL_DRY_8':   {'TRENDING': 1.0, 'RANGING': 0.6, 'HIGH_VOL': 0.0},
        'GUJRAL_DRY_13':  {'TRENDING': 1.0, 'RANGING': 0.5, 'HIGH_VOL': 0.0},
        'CHAN_AT_DRY_4':   {'TRENDING': 0.3, 'RANGING': 1.0, 'HIGH_VOL': 0.2},
    }
    sig_regime = regime_match.get(signal_id, {})
    features['dim2_regime'] = sig_regime.get(regime, 0.5)

    # 3. Timeframe confirmation — weekly trend aligned with daily signal
    close = float(today['close'])
    sma_50 = float(today.get('sma_50', close)) if pd.notna(today.get('sma_50')) else close
    sma_200 = float(today.get('sma_200', close)) if pd.notna(today.get('sma_200')) else close
    if direction == 'LONG':
        features['dim3_timeframe'] = 1.0 if close > sma_50 > sma_200 else (0.5 if close > sma_50 else 0.2)
    else:
        features['dim3_timeframe'] = 1.0 if close < sma_50 < sma_200 else (0.5 if close < sma_50 else 0.2)

    # 4. Volume confirmation
    vol_ratio = float(today.get('vol_ratio_20', 1.0)) if pd.notna(today.get('vol_ratio_20')) else 1.0
    features['dim4_volume'] = min(1.0, vol_ratio / 2.0)

    # 5. Cross-signal agreement
    if active_signals:
        same_dir = sum(1 for s in active_signals if s.get('direction') == direction)
        features['dim5_cross_signal'] = min(1.0, same_dir / 3)
    else:
        features['dim5_cross_signal'] = 0.0

    # 6. Gujral pivot level — proximity to key pivot
    pivot = float(today.get('pivot', close)) if pd.notna(today.get('pivot')) else close
    r1 = float(today.get('r1', close * 1.01)) if pd.notna(today.get('r1')) else close * 1.01
    s1 = float(today.get('s1', close * 0.99)) if pd.notna(today.get('s1')) else close * 0.99
    pivot_dist = min(abs(close - pivot), abs(close - r1), abs(close - s1))
    features['dim6_pivot'] = max(0, 1.0 - pivot_dist / (close * 0.01))

    # 7. Bulkowski pattern quality — bb_width as proxy for breakout quality
    bb_width = float(today.get('bb_bandwidth', 0.04)) if pd.notna(today.get('bb_bandwidth')) else 0.04
    features['dim7_pattern'] = min(1.0, bb_width / 0.08)

    # 8. VIX level — low VIX = more reliable
    vix = float(today.get('india_vix', 15)) if pd.notna(today.get('india_vix')) else 15
    features['dim8_vix'] = max(0, 1.0 - (vix - 10) / 25)

    # 9. FII flow — PCR as proxy (high PCR = fear = contrarian buy)
    pcr = float(today.get('pcr_oi', 1.0)) if pd.notna(today.get('pcr_oi')) else 1.0
    if direction == 'LONG':
        features['dim9_fii'] = min(1.0, pcr / 1.5)  # high PCR = bullish for longs
    else:
        features['dim9_fii'] = min(1.0, (2.0 - pcr) / 1.5)

    # 10. Historical performance — rolling win rate (from last 20 trades)
    # Placeholder — needs trade history to compute
    features['dim10_history'] = 0.5  # default neutral

    return features


FEATURE_COLS = [
    'dim1_extremity', 'dim2_regime', 'dim3_timeframe', 'dim4_volume',
    'dim5_cross_signal', 'dim6_pivot', 'dim7_pattern', 'dim8_vix',
    'dim9_fii', 'dim10_history',
]


def features_to_array(features_dict):
    """Convert features dict to numpy array in consistent order."""
    return np.array([features_dict[c] for c in FEATURE_COLS])


def compute_fixed_score(features_dict):
    """Original fixed-weight scorer. Returns 0-170 points."""
    weights = {
        'dim1_extremity': 40,
        'dim2_regime': 25,
        'dim3_timeframe': 15,
        'dim4_volume': 10,
        'dim5_cross_signal': 20,
        'dim6_pivot': 10,
        'dim7_pattern': 15,
        'dim8_vix': 10,
        'dim9_fii': 10,
        'dim10_history': 15,
    }
    return sum(features_dict[k] * weights[k] for k in FEATURE_COLS)


def score_to_grade(score):
    """Convert 0-170 score to grade + size multiplier."""
    if score >= 85:  return 'S+', 2.0
    if score >= 70:  return 'A', 1.5
    if score >= 55:  return 'B', 1.0
    if score >= 40:  return 'C', 0.6
    if score >= 25:  return 'D', 0.3
    return 'F', 0.0
