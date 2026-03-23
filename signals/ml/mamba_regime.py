"""
Mamba/SSM Regime Detector Signal.

State-space model for market regime detection: TRENDING, MEAN_REVERTING,
VOLATILE, or MIXED.  When no trained model is available, falls back to
simple ADX + VIX threshold rules.

Features (trained mode):
    - Returns (1d, 5d, 20d)
    - Volume ratio (current vs 20-day average)
    - India VIX
    - ADX (14-period)
    - Market breadth (advance/decline ratio)

Fallback (no model):
    - ADX > 25 and VIX < 20  -> TRENDING
    - ADX < 20 and VIX < 15  -> MEAN_REVERTING
    - VIX > 25               -> VOLATILE
    - else                   -> MIXED

Model path: models/mamba_regime.pkl

Usage:
    from signals.ml.mamba_regime import MambaRegimeDetector

    regime = MambaRegimeDetector()
    result = regime.evaluate(df, date)
"""

import logging
import os
import pickle
from datetime import date
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ================================================================
# CONSTANTS
# ================================================================

SIGNAL_ID = 'MAMBA_REGIME'

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models')

# Regime labels
REGIME_TRENDING = 'TRENDING'
REGIME_MEAN_REVERTING = 'MEAN_REVERTING'
REGIME_VOLATILE = 'VOLATILE'
REGIME_MIXED = 'MIXED'

# Fallback thresholds
ADX_TRENDING_THRESHOLD = 25.0
ADX_MEAN_REVERT_THRESHOLD = 20.0
VIX_TRENDING_MAX = 20.0
VIX_MEAN_REVERT_MAX = 15.0
VIX_VOLATILE_MIN = 25.0

# Signal type preferences by regime
REGIME_SIGNAL_PREFERENCE = {
    REGIME_TRENDING: 'MOMENTUM',
    REGIME_MEAN_REVERTING: 'REVERSION',
    REGIME_VOLATILE: 'VOLATILITY',
    REGIME_MIXED: 'BALANCED',
}

# Direction mapping: trending -> follow trend, mean_reverting -> fade, etc.
REGIME_DIRECTION_BIAS = {
    REGIME_TRENDING: None,       # Direction depends on trend direction
    REGIME_MEAN_REVERTING: None, # Direction depends on deviation
    REGIME_VOLATILE: 'NEUTRAL',  # Reduce exposure
    REGIME_MIXED: 'NEUTRAL',
}


# ================================================================
# HELPERS
# ================================================================

def _safe_float(val: Any, default: float = float('nan')) -> float:
    """Safely cast to float."""
    if val is None:
        return default
    try:
        v = float(val)
        return v
    except (TypeError, ValueError):
        return default


def _compute_adx(df: Any, period: int = 14) -> float:
    """Compute ADX from dataframe. Returns NaN if insufficient data."""
    try:
        if not hasattr(df, 'iloc') or len(df) < period + 1:
            return float('nan')

        high = df['high'].astype(float).values if 'high' in df.columns else df['High'].astype(float).values
        low = df['low'].astype(float).values if 'low' in df.columns else df['Low'].astype(float).values
        close = df['close'].astype(float).values if 'close' in df.columns else df['Close'].astype(float).values

        n = len(close)
        if n < period + 1:
            return float('nan')

        # True Range
        tr = np.zeros(n)
        plus_dm = np.zeros(n)
        minus_dm = np.zeros(n)

        for i in range(1, n):
            h_l = high[i] - low[i]
            h_pc = abs(high[i] - close[i - 1])
            l_pc = abs(low[i] - close[i - 1])
            tr[i] = max(h_l, h_pc, l_pc)

            up_move = high[i] - high[i - 1]
            down_move = low[i - 1] - low[i]

            plus_dm[i] = up_move if (up_move > down_move and up_move > 0) else 0.0
            minus_dm[i] = down_move if (down_move > up_move and down_move > 0) else 0.0

        # Smoothed averages (Wilder's method)
        atr = np.zeros(n)
        plus_di_arr = np.zeros(n)
        minus_di_arr = np.zeros(n)

        atr[period] = np.mean(tr[1:period + 1])
        s_plus = np.mean(plus_dm[1:period + 1])
        s_minus = np.mean(minus_dm[1:period + 1])

        for i in range(period + 1, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
            s_plus = (s_plus * (period - 1) + plus_dm[i]) / period
            s_minus = (s_minus * (period - 1) + minus_dm[i]) / period

            if atr[i] > 0:
                plus_di_arr[i] = 100 * s_plus / atr[i]
                minus_di_arr[i] = 100 * s_minus / atr[i]

        # DX and ADX
        dx = np.zeros(n)
        for i in range(period + 1, n):
            denom = plus_di_arr[i] + minus_di_arr[i]
            if denom > 0:
                dx[i] = 100 * abs(plus_di_arr[i] - minus_di_arr[i]) / denom

        # ADX = smoothed DX over last period values
        valid_dx = dx[period + 1:]
        if len(valid_dx) < period:
            return float(np.mean(valid_dx)) if len(valid_dx) > 0 else float('nan')

        adx = np.mean(valid_dx[-period:])
        return float(adx)
    except Exception as e:
        logger.debug('ADX computation failed: %s', e)
        return float('nan')


# ================================================================
# SIGNAL CLASS
# ================================================================

class MambaRegimeDetector:
    """
    Mamba/SSM-based regime detector.  Falls back to ADX + VIX rules
    when no trained model is available.
    """

    SIGNAL_ID = SIGNAL_ID
    MODEL_PATH = os.path.join(MODEL_DIR, 'mamba_regime.pkl')

    def __init__(self) -> None:
        self._model = self._load_model()
        mode = 'ML' if self._model is not None else 'FALLBACK'
        logger.info('MambaRegimeDetector initialised (mode=%s)', mode)

    # ----------------------------------------------------------
    # Model loading
    # ----------------------------------------------------------
    def _load_model(self) -> Any:
        """Try to load a trained Mamba/SSM model from disk."""
        try:
            if os.path.exists(self.MODEL_PATH):
                with open(self.MODEL_PATH, 'rb') as f:
                    model = pickle.load(f)
                logger.info('Loaded Mamba regime model from %s', self.MODEL_PATH)
                return model
        except ImportError:
            logger.warning('torch/mamba not installed — using fallback')
        except Exception as e:
            logger.warning('Failed to load Mamba regime model: %s', e)
        return None

    # ----------------------------------------------------------
    # ML prediction
    # ----------------------------------------------------------
    def _predict_ml(self, features: np.ndarray) -> Dict:
        """Predict regime using the trained model."""
        try:
            pred = self._model.predict(features.reshape(1, -1))
            if hasattr(self._model, 'predict_proba'):
                probs = self._model.predict_proba(features.reshape(1, -1))[0]
                confidence = float(np.max(probs))
            else:
                confidence = 0.7

            regime_map = {
                0: REGIME_TRENDING,
                1: REGIME_MEAN_REVERTING,
                2: REGIME_VOLATILE,
                3: REGIME_MIXED,
            }
            regime = regime_map.get(int(pred[0]), REGIME_MIXED)
            return {
                'regime': regime,
                'confidence': confidence,
                'mode': 'ML',
            }
        except Exception as e:
            logger.warning('ML regime prediction failed: %s', e)
            return {}

    # ----------------------------------------------------------
    # Fallback prediction
    # ----------------------------------------------------------
    def _predict_fallback(self, df: Any, dt: date) -> Optional[Dict]:
        """Rule-based regime detection using ADX + VIX thresholds."""
        adx = float('nan')
        vix = float('nan')
        momentum_20d = 0.0

        try:
            if hasattr(df, 'loc') and hasattr(df, 'columns'):
                # Get data up to the date
                if hasattr(df.index, 'date'):
                    mask = df.index.date <= dt
                else:
                    mask = df.index <= str(dt)
                sub = df.loc[mask]

                if sub.empty:
                    return None

                # VIX
                for col in ['india_vix', 'vix', 'India_VIX', 'VIX']:
                    if col in sub.columns:
                        vix = _safe_float(sub[col].iloc[-1])
                        break

                # ADX
                for col in ['adx', 'ADX', 'adx_14']:
                    if col in sub.columns:
                        adx = _safe_float(sub[col].iloc[-1])
                        break

                # Compute ADX if not in columns
                if np.isnan(adx):
                    adx = _compute_adx(sub)

                # 20-day momentum for trend direction
                close_col = 'close' if 'close' in sub.columns else 'Close' if 'Close' in sub.columns else None
                if close_col and len(sub) >= 20:
                    cur = float(sub[close_col].iloc[-1])
                    prev = float(sub[close_col].iloc[-20])
                    if prev > 0:
                        momentum_20d = ((cur / prev) - 1.0) * 100
        except Exception as e:
            logger.debug('Fallback data extraction failed: %s', e)
            return None

        # Default VIX if still NaN
        if np.isnan(vix):
            vix = 18.0  # Assume moderate VIX
        if np.isnan(adx):
            adx = 20.0  # Assume moderate ADX

        # Rule-based regime
        if adx > ADX_TRENDING_THRESHOLD and vix < VIX_TRENDING_MAX:
            regime = REGIME_TRENDING
            confidence = min(0.85, 0.55 + (adx - 25) * 0.01 + (20 - vix) * 0.01)
        elif adx < ADX_MEAN_REVERT_THRESHOLD and vix < VIX_MEAN_REVERT_MAX:
            regime = REGIME_MEAN_REVERTING
            confidence = min(0.80, 0.55 + (20 - adx) * 0.01 + (15 - vix) * 0.01)
        elif vix > VIX_VOLATILE_MIN:
            regime = REGIME_VOLATILE
            confidence = min(0.85, 0.55 + (vix - 25) * 0.01)
        else:
            regime = REGIME_MIXED
            confidence = 0.45

        confidence = max(0.10, min(0.90, confidence))

        # Determine direction bias
        if regime == REGIME_TRENDING:
            direction = 'LONG' if momentum_20d > 0 else 'SHORT'
        elif regime == REGIME_MEAN_REVERTING:
            # Fade recent momentum
            direction = 'SHORT' if momentum_20d > 2 else 'LONG' if momentum_20d < -2 else 'NEUTRAL'
        else:
            direction = 'NEUTRAL'

        return {
            'regime': regime,
            'confidence': round(confidence, 3),
            'direction': direction,
            'adx': round(adx, 2),
            'vix': round(vix, 2),
            'momentum_20d': round(momentum_20d, 3),
            'mode': 'FALLBACK',
        }

    # ----------------------------------------------------------
    # Feature construction (for ML mode)
    # ----------------------------------------------------------
    def _build_features(self, df: Any, dt: date) -> Optional[np.ndarray]:
        """Build feature vector for ML model."""
        try:
            if not hasattr(df, 'loc'):
                return None

            if hasattr(df.index, 'date'):
                mask = df.index.date <= dt
            else:
                mask = df.index <= str(dt)
            sub = df.loc[mask]

            if sub.empty or len(sub) < 21:
                return None

            close_col = 'close' if 'close' in sub.columns else 'Close'
            closes = sub[close_col].astype(float).values

            ret_1d = (closes[-1] / closes[-2] - 1) if len(closes) >= 2 else 0
            ret_5d = (closes[-1] / closes[-5] - 1) if len(closes) >= 5 else 0
            ret_20d = (closes[-1] / closes[-20] - 1) if len(closes) >= 20 else 0

            # Volume ratio
            vol_col = 'volume' if 'volume' in sub.columns else 'Volume' if 'Volume' in sub.columns else None
            vol_ratio = 1.0
            if vol_col and len(sub) >= 20:
                vol = sub[vol_col].astype(float).values
                avg_vol = np.mean(vol[-20:])
                vol_ratio = vol[-1] / avg_vol if avg_vol > 0 else 1.0

            # VIX
            vix = 18.0
            for col in ['india_vix', 'vix', 'VIX']:
                if col in sub.columns:
                    vix = _safe_float(sub[col].iloc[-1], 18.0)
                    break

            # ADX
            adx = _compute_adx(sub)
            if np.isnan(adx):
                adx = 20.0

            # Breadth
            breadth = 1.0
            for col in ['breadth', 'advance_decline_ratio', 'ad_ratio']:
                if col in sub.columns:
                    breadth = _safe_float(sub[col].iloc[-1], 1.0)
                    break

            features = np.array([
                ret_1d, ret_5d, ret_20d,
                vol_ratio, vix, adx, breadth,
            ], dtype=np.float64)
            return features
        except Exception as e:
            logger.debug('Feature build failed: %s', e)
            return None

    # ----------------------------------------------------------
    # Main evaluate
    # ----------------------------------------------------------
    def evaluate(self, df: Any, dt: date) -> Optional[Dict]:
        """
        Evaluate the regime detection signal.

        Parameters
        ----------
        df : DataFrame with OHLCV + VIX data.
        dt : Trade date.

        Returns
        -------
        dict with signal_id, direction, strength, price, reason, metadata.
        None if no signal.
        """
        try:
            return self._evaluate_inner(df, dt)
        except Exception as e:
            logger.error('MambaRegimeDetector.evaluate error: %s', e, exc_info=True)
            return None

    def _evaluate_inner(self, df: Any, dt: date) -> Optional[Dict]:
        result = None

        # Try ML mode first
        if self._model is not None:
            features = self._build_features(df, dt)
            if features is not None:
                result = self._predict_ml(features)
                if not result:
                    result = None

        # Fallback
        if result is None:
            result = self._predict_fallback(df, dt)

        if result is None:
            return None

        regime = result['regime']
        confidence = result.get('confidence', 0.5)
        direction = result.get('direction', 'NEUTRAL')
        signal_pref = REGIME_SIGNAL_PREFERENCE.get(regime, 'BALANCED')

        # Extract current price
        price = 0.0
        try:
            if hasattr(df, 'loc'):
                close_col = 'close' if 'close' in df.columns else 'Close' if 'Close' in df.columns else None
                if close_col:
                    if hasattr(df.index, 'date'):
                        row = df.loc[df.index.date == dt]
                    else:
                        row = df.loc[df.index == str(dt)]
                    if not row.empty:
                        price = round(float(row[close_col].iloc[-1]), 2)
        except Exception:
            pass

        reason_parts = [
            'MAMBA_REGIME',
            f"Regime={regime}",
            f"Mode={result.get('mode', 'UNKNOWN')}",
            f"ADX={result.get('adx', 'N/A')}",
            f"VIX={result.get('vix', 'N/A')}",
            f"Preference={signal_pref}",
        ]

        return {
            'signal_id': self.SIGNAL_ID,
            'direction': direction,
            'strength': round(confidence, 3),
            'price': price,
            'reason': ' | '.join(reason_parts),
            'metadata': {
                'regime': regime,
                'mode': result.get('mode', 'UNKNOWN'),
                'confidence': confidence,
                'signal_type_preference': signal_pref,
                'adx': result.get('adx'),
                'vix': result.get('vix'),
                'momentum_20d': result.get('momentum_20d'),
            },
        }

    def reset(self) -> None:
        """Reset internal state."""
        pass

    def __repr__(self) -> str:
        mode = 'ML' if self._model is not None else 'FALLBACK'
        return f"MambaRegimeDetector(signal_id='{self.SIGNAL_ID}', mode='{mode}')"
