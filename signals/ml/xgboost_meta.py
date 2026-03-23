"""
XGBoost Meta-Learner Signal.

Combines outputs of all other signals into a meta-prediction using XGBoost.
When no trained model is available, falls back to a majority-vote heuristic
weighted by each signal's historical accuracy.

Features (trained mode):
    - Per-signal: fired (binary), direction (+1/-1/0), strength (0-1)
    - Regime label (one-hot)
    - India VIX
    - 5-day and 20-day momentum

Fallback (no model):
    - Majority vote of fired signals
    - Weighted by configurable historical accuracy per signal
    - Direction = sign of weighted vote sum
    - Confidence = |weighted sum| / total weight of fired signals

Model path: models/xgboost_meta.pkl

Usage:
    from signals.ml.xgboost_meta import XGBoostMetaLearner

    meta = XGBoostMetaLearner()
    result = meta.evaluate(df, date)
"""

import logging
import os
import pickle
from datetime import date
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ================================================================
# CONSTANTS
# ================================================================

SIGNAL_ID = 'XGBOOST_META'

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models')

# Default historical accuracy weights per signal family (fallback mode)
DEFAULT_SIGNAL_WEIGHTS: Dict[str, float] = {
    'GIFT_CONVERGENCE': 0.62,
    'FII_FUTURES_OI': 0.58,
    'PCR_SIGNAL': 0.55,
    'DELIVERY_SIGNAL': 0.57,
    'IV_RANK': 0.54,
    'GAMMA_EXPOSURE': 0.56,
    'VOL_TERM_STRUCTURE': 0.53,
    'MAX_PAIN': 0.55,
    'ROLLOVER_SIGNAL': 0.56,
    'BOND_YIELD': 0.52,
    'SEASONALITY': 0.51,
    'ORDER_FLOW': 0.58,
    'GLOBAL_COMPOSITE': 0.54,
    'REGIME': 0.60,
}
DEFAULT_WEIGHT = 0.50  # For unknown signals


# ================================================================
# HELPERS
# ================================================================

def _safe_float(val: Any, default: float = 0.0) -> float:
    """Safely cast to float."""
    if val is None:
        return default
    try:
        v = float(val)
        return default if np.isnan(v) else v
    except (TypeError, ValueError):
        return default


def _direction_to_numeric(direction: Optional[str]) -> float:
    """Convert direction string to numeric."""
    if direction is None:
        return 0.0
    d = str(direction).upper().strip()
    if d == 'LONG':
        return 1.0
    elif d == 'SHORT':
        return -1.0
    return 0.0


# ================================================================
# SIGNAL CLASS
# ================================================================

class XGBoostMetaLearner:
    """
    XGBoost Meta-Learner that combines all signal outputs into a
    single meta-prediction.  Falls back to weighted majority vote
    when no trained model is available.
    """

    SIGNAL_ID = SIGNAL_ID
    MODEL_PATH = os.path.join(MODEL_DIR, 'xgboost_meta.pkl')

    def __init__(self, signal_weights: Optional[Dict[str, float]] = None) -> None:
        self._signal_weights = signal_weights or DEFAULT_SIGNAL_WEIGHTS
        self._model = self._load_model()
        mode = 'ML' if self._model is not None else 'FALLBACK'
        logger.info('XGBoostMetaLearner initialised (mode=%s)', mode)

    # ----------------------------------------------------------
    # Model loading
    # ----------------------------------------------------------
    def _load_model(self) -> Any:
        """Try to load a trained XGBoost model from disk."""
        try:
            if os.path.exists(self.MODEL_PATH):
                with open(self.MODEL_PATH, 'rb') as f:
                    model = pickle.load(f)
                logger.info('Loaded XGBoost meta model from %s', self.MODEL_PATH)
                return model
        except ImportError:
            logger.warning('xgboost/sklearn not installed — using fallback')
        except Exception as e:
            logger.warning('Failed to load XGBoost meta model: %s', e)
        return None

    # ----------------------------------------------------------
    # ML prediction
    # ----------------------------------------------------------
    def _predict_ml(self, features: np.ndarray) -> Dict:
        """Predict using the trained XGBoost model."""
        try:
            prob = self._model.predict_proba(features.reshape(1, -1))[0]
            # Assume classes: 0=SHORT, 1=NEUTRAL, 2=LONG
            if len(prob) == 3:
                pred_class = int(np.argmax(prob))
                confidence = float(prob[pred_class])
                direction_map = {0: 'SHORT', 1: 'NEUTRAL', 2: 'LONG'}
                direction = direction_map[pred_class]
            elif len(prob) == 2:
                # Binary: 0=SHORT, 1=LONG
                pred_class = int(np.argmax(prob))
                confidence = float(prob[pred_class])
                direction = 'LONG' if pred_class == 1 else 'SHORT'
            else:
                pred = float(self._model.predict(features.reshape(1, -1))[0])
                direction = 'LONG' if pred > 0 else 'SHORT'
                confidence = min(abs(pred), 1.0)

            size_modifier = 0.5 + confidence * 0.5  # Scale 0.5-1.0
            return {
                'direction': direction,
                'confidence': confidence,
                'size_modifier': round(size_modifier, 2),
                'mode': 'ML',
            }
        except Exception as e:
            logger.warning('ML prediction failed: %s — falling back', e)
            return {}

    # ----------------------------------------------------------
    # Fallback prediction
    # ----------------------------------------------------------
    def _predict_fallback(self, df: Any, dt: date) -> Optional[Dict]:
        """
        Weighted majority vote of fired signals.

        Expects df to have columns like:
            signal_<name>_fired (bool/int)
            signal_<name>_direction (str)
            signal_<name>_strength (float)
        Or a 'signals' column containing a list of signal result dicts.
        """
        signals = self._extract_signals(df, dt)
        if not signals:
            logger.debug('No fired signals for meta-learner on %s', dt)
            return None

        weighted_sum = 0.0
        total_weight = 0.0
        fired_names: List[str] = []

        for sig in signals:
            sig_id = sig.get('signal_id', 'UNKNOWN')
            direction = _direction_to_numeric(sig.get('direction'))
            strength = _safe_float(sig.get('strength', sig.get('confidence', 0.5)))

            if direction == 0.0:
                continue

            weight = self._signal_weights.get(sig_id, DEFAULT_WEIGHT)
            weighted_sum += direction * strength * weight
            total_weight += weight
            fired_names.append(sig_id)

        if total_weight == 0:
            return None

        normalised = weighted_sum / total_weight
        direction = 'LONG' if normalised > 0 else 'SHORT'
        confidence = min(abs(normalised), 1.0)

        # Need meaningful confidence
        if confidence < 0.05:
            return None

        size_modifier = 0.5 + confidence * 0.5
        return {
            'direction': direction,
            'confidence': round(confidence, 3),
            'size_modifier': round(size_modifier, 2),
            'fired_signals': fired_names,
            'weighted_score': round(normalised, 4),
            'mode': 'FALLBACK',
        }

    # ----------------------------------------------------------
    # Signal extraction helper
    # ----------------------------------------------------------
    def _extract_signals(self, df: Any, dt: date) -> List[Dict]:
        """Extract signal results from the dataframe row for a given date."""
        signals: List[Dict] = []
        try:
            if hasattr(df, 'loc'):
                row = df.loc[df.index == str(dt)] if hasattr(df.index, 'date') is False else df.loc[df.index.date == dt]
                if hasattr(row, 'empty') and row.empty:
                    # Try string match
                    row = df.loc[df.index == str(dt)]
                if hasattr(row, 'empty') and row.empty:
                    return signals

                row = row.iloc[0] if len(row) > 1 else row.squeeze()

                # Check for 'signals' column (list of dicts)
                if 'signals' in row.index if hasattr(row, 'index') else False:
                    raw = row['signals']
                    if isinstance(raw, list):
                        return raw

                # Check for individual signal columns
                for col in (row.index if hasattr(row, 'index') else []):
                    col_str = str(col)
                    if col_str.endswith('_fired') and row[col]:
                        sig_name = col_str.replace('signal_', '').replace('_fired', '').upper()
                        dir_col = col_str.replace('_fired', '_direction')
                        str_col = col_str.replace('_fired', '_strength')
                        signals.append({
                            'signal_id': sig_name,
                            'direction': str(row.get(dir_col, 'NEUTRAL')) if hasattr(row, 'get') else 'NEUTRAL',
                            'strength': _safe_float(row.get(str_col, 0.5) if hasattr(row, 'get') else 0.5),
                        })
            elif isinstance(df, list):
                # df is already a list of signal dicts
                signals = df
            elif isinstance(df, dict):
                # Single signal dict or dict of signals
                if 'signal_id' in df:
                    signals = [df]
                else:
                    signals = list(df.values()) if all(isinstance(v, dict) for v in df.values()) else []
        except Exception as e:
            logger.debug('Error extracting signals: %s', e)
        return signals

    # ----------------------------------------------------------
    # Feature construction (for ML mode)
    # ----------------------------------------------------------
    def _build_features(self, df: Any, dt: date) -> Optional[np.ndarray]:
        """Build feature vector for the ML model."""
        try:
            signals = self._extract_signals(df, dt)
            if not signals:
                return None

            # Signal features: fired, direction, strength per known signal
            known_ids = list(self._signal_weights.keys())
            sig_features = []
            for sid in known_ids:
                found = [s for s in signals if s.get('signal_id') == sid]
                if found:
                    s = found[0]
                    sig_features.extend([
                        1.0,
                        _direction_to_numeric(s.get('direction')),
                        _safe_float(s.get('strength', s.get('confidence', 0.5))),
                    ])
                else:
                    sig_features.extend([0.0, 0.0, 0.0])

            # Market features from df
            vix = 0.0
            momentum_5d = 0.0
            momentum_20d = 0.0

            if hasattr(df, 'loc'):
                try:
                    row = df.loc[df.index.date == dt] if hasattr(df.index, 'date') else df.loc[df.index == str(dt)]
                    if not row.empty:
                        r = row.iloc[-1]
                        vix = _safe_float(r.get('india_vix', r.get('vix', 0))) if hasattr(r, 'get') else 0.0
                        if 'close' in (r.index if hasattr(r, 'index') else []):
                            idx_pos = df.index.get_loc(r.name)
                            if isinstance(idx_pos, slice):
                                idx_pos = idx_pos.stop - 1
                            if idx_pos >= 5:
                                momentum_5d = (float(r['close']) / float(df.iloc[idx_pos - 5]['close']) - 1.0) * 100
                            if idx_pos >= 20:
                                momentum_20d = (float(r['close']) / float(df.iloc[idx_pos - 20]['close']) - 1.0) * 100
                except Exception:
                    pass

            market_features = [vix, momentum_5d, momentum_20d]
            return np.array(sig_features + market_features, dtype=np.float64)
        except Exception as e:
            logger.debug('Feature build failed: %s', e)
            return None

    # ----------------------------------------------------------
    # Main evaluate
    # ----------------------------------------------------------
    def evaluate(self, df: Any, dt: date) -> Optional[Dict]:
        """
        Evaluate the meta-learner signal.

        Parameters
        ----------
        df : DataFrame or list of signal dicts for the date.
        dt : Trade date.

        Returns
        -------
        dict with signal_id, direction, strength, price, reason, metadata.
        None if no signal.
        """
        try:
            return self._evaluate_inner(df, dt)
        except Exception as e:
            logger.error('XGBoostMetaLearner.evaluate error: %s', e, exc_info=True)
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

        direction = result['direction']
        confidence = result.get('confidence', 0.5)
        size_modifier = result.get('size_modifier', 1.0)

        # Extract current price
        price = self._get_price(df, dt)

        reason_parts = [
            'XGBOOST_META',
            f"Mode={result.get('mode', 'UNKNOWN')}",
            f"Direction={direction}",
            f"Confidence={confidence:.3f}",
            f"SizeModifier={size_modifier:.2f}",
        ]
        if 'fired_signals' in result:
            reason_parts.append(f"Fired={len(result['fired_signals'])}")

        return {
            'signal_id': self.SIGNAL_ID,
            'direction': direction,
            'strength': round(confidence, 3),
            'price': price,
            'reason': ' | '.join(reason_parts),
            'metadata': {
                'mode': result.get('mode', 'UNKNOWN'),
                'size_modifier': size_modifier,
                'fired_signals': result.get('fired_signals', []),
                'weighted_score': result.get('weighted_score'),
                'confidence': confidence,
            },
        }

    # ----------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------
    def _get_price(self, df: Any, dt: date) -> float:
        """Extract the latest close price for the date."""
        try:
            if hasattr(df, 'loc'):
                row = df.loc[df.index.date == dt] if hasattr(df.index, 'date') else df.loc[df.index == str(dt)]
                if not row.empty:
                    r = row.iloc[-1]
                    for col in ['close', 'Close', 'last_price', 'ltp']:
                        if col in (r.index if hasattr(r, 'index') else []):
                            return round(float(r[col]), 2)
        except Exception:
            pass
        return 0.0

    def reset(self) -> None:
        """Reset internal state."""
        pass

    def __repr__(self) -> str:
        mode = 'ML' if self._model is not None else 'FALLBACK'
        return f"XGBoostMetaLearner(signal_id='{self.SIGNAL_ID}', mode='{mode}')"
