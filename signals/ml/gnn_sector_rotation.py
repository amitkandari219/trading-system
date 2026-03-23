"""
Graph Neural Network Sector Rotation Signal.

Models sector correlations as a graph to detect rotation patterns
across NSE sectors.  When no trained model is available, falls back
to simple relative-strength rotation logic.

Sectors: IT, Bank, Pharma, Auto, Metal, FMCG, Energy, Realty

Fallback (no model):
    - Compare each sector's 20-day return vs Nifty 50's 20-day return
    - Sector outperforming Nifty by > 3%  -> money flowing in
    - If defensive (Pharma, FMCG) outperforming -> risk-off -> SHORT bias
    - If cyclical (Metal, Auto, Realty) outperforming -> risk-on -> LONG bias

Model path: models/gnn_rotation.pkl

Usage:
    from signals.ml.gnn_sector_rotation import GNNSectorRotation

    gnn = GNNSectorRotation()
    result = gnn.evaluate(df, date)
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

SIGNAL_ID = 'GNN_SECTOR_ROTATION'

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models')

# Sector definitions
SECTORS = ['IT', 'BANK', 'PHARMA', 'AUTO', 'METAL', 'FMCG', 'ENERGY', 'REALTY']

DEFENSIVE_SECTORS = {'PHARMA', 'FMCG'}
CYCLICAL_SECTORS = {'METAL', 'AUTO', 'REALTY'}
GROWTH_SECTORS = {'IT', 'ENERGY'}
FINANCIAL_SECTORS = {'BANK'}

# Column name patterns for sector data
SECTOR_COLUMN_PATTERNS = {
    'IT': ['nifty_it', 'it_index', 'sector_it', 'NIFTY_IT'],
    'BANK': ['nifty_bank', 'bank_nifty', 'banknifty', 'sector_bank', 'NIFTY_BANK'],
    'PHARMA': ['nifty_pharma', 'pharma_index', 'sector_pharma', 'NIFTY_PHARMA'],
    'AUTO': ['nifty_auto', 'auto_index', 'sector_auto', 'NIFTY_AUTO'],
    'METAL': ['nifty_metal', 'metal_index', 'sector_metal', 'NIFTY_METAL'],
    'FMCG': ['nifty_fmcg', 'fmcg_index', 'sector_fmcg', 'NIFTY_FMCG'],
    'ENERGY': ['nifty_energy', 'energy_index', 'sector_energy', 'NIFTY_ENERGY'],
    'REALTY': ['nifty_realty', 'realty_index', 'sector_realty', 'NIFTY_REALTY'],
}

NIFTY_COLUMN_PATTERNS = ['close', 'Close', 'nifty_50', 'nifty50', 'NIFTY_50']

# Thresholds
OUTPERFORMANCE_THRESHOLD = 0.03  # 3% outperformance
RS_LOOKBACK = 20                 # 20-day relative strength

# Confidence
BASE_CONFIDENCE = 0.50
SECTOR_CONVICTION_BOOST = 0.05   # Per strongly outperforming sector


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


def _find_column(df: Any, patterns: List[str]) -> Optional[str]:
    """Find the first matching column name from a list of patterns."""
    if not hasattr(df, 'columns'):
        return None
    for p in patterns:
        if p in df.columns:
            return p
        # Case-insensitive fallback
        for col in df.columns:
            if col.lower() == p.lower():
                return col
    return None


# ================================================================
# SIGNAL CLASS
# ================================================================

class GNNSectorRotation:
    """
    GNN-based sector rotation signal.  Falls back to relative-strength
    rotation analysis when no trained model is available.
    """

    SIGNAL_ID = SIGNAL_ID
    MODEL_PATH = os.path.join(MODEL_DIR, 'gnn_rotation.pkl')

    def __init__(self) -> None:
        self._model = self._load_model()
        mode = 'ML' if self._model is not None else 'FALLBACK'
        logger.info('GNNSectorRotation initialised (mode=%s)', mode)

    # ----------------------------------------------------------
    # Model loading
    # ----------------------------------------------------------
    def _load_model(self) -> Any:
        """Try to load a trained GNN model from disk."""
        try:
            if os.path.exists(self.MODEL_PATH):
                with open(self.MODEL_PATH, 'rb') as f:
                    model = pickle.load(f)
                logger.info('Loaded GNN rotation model from %s', self.MODEL_PATH)
                return model
        except ImportError:
            logger.warning('torch-geometric/torch not installed — using fallback')
        except Exception as e:
            logger.warning('Failed to load GNN rotation model: %s', e)
        return None

    # ----------------------------------------------------------
    # ML prediction
    # ----------------------------------------------------------
    def _predict_ml(self, features: np.ndarray) -> Dict:
        """Predict sector rotation using the trained GNN model."""
        try:
            pred = self._model.predict(features.reshape(1, -1))
            # Expect model to return: [rotation_score, regime_label, *sector_scores]
            if hasattr(pred, '__len__') and len(pred) >= 2:
                rotation_score = float(pred[0])
                regime_code = int(pred[1]) if len(pred) > 1 else 0
                regime_map = {0: 'RISK_ON', 1: 'RISK_OFF', 2: 'NEUTRAL'}
                regime = regime_map.get(regime_code, 'NEUTRAL')
            else:
                rotation_score = float(pred[0]) if hasattr(pred, '__len__') else float(pred)
                regime = 'RISK_ON' if rotation_score > 0 else 'RISK_OFF'

            return {
                'rotation_score': rotation_score,
                'regime': regime,
                'mode': 'ML',
            }
        except Exception as e:
            logger.warning('GNN ML prediction failed: %s', e)
            return {}

    # ----------------------------------------------------------
    # Fallback prediction
    # ----------------------------------------------------------
    def _predict_fallback(self, df: Any, dt: date) -> Optional[Dict]:
        """
        Simple relative-strength rotation analysis.

        Compares each sector's 20-day return against Nifty 50.
        Classifies market regime as RISK_ON or RISK_OFF based on
        which sector types are outperforming.
        """
        try:
            if not hasattr(df, 'loc') or not hasattr(df, 'columns'):
                return None

            # Get data up to date
            if hasattr(df.index, 'date'):
                mask = df.index.date <= dt
            else:
                mask = df.index <= str(dt)
            sub = df.loc[mask]

            if sub.empty or len(sub) < RS_LOOKBACK + 1:
                return None

            # Nifty 50 return
            nifty_col = _find_column(sub, NIFTY_COLUMN_PATTERNS)
            if nifty_col is None:
                return None

            nifty_values = sub[nifty_col].astype(float).values
            nifty_current = nifty_values[-1]
            nifty_prev = nifty_values[-(RS_LOOKBACK + 1)]
            nifty_return = (nifty_current / nifty_prev - 1.0) if nifty_prev > 0 else 0.0

            # Compute sector returns
            sector_returns: Dict[str, float] = {}
            sector_relative: Dict[str, float] = {}

            for sector in SECTORS:
                col = _find_column(sub, SECTOR_COLUMN_PATTERNS[sector])
                if col is None:
                    continue

                vals = sub[col].astype(float).values
                if len(vals) < RS_LOOKBACK + 1:
                    continue

                sec_current = vals[-1]
                sec_prev = vals[-(RS_LOOKBACK + 1)]
                if sec_prev <= 0:
                    continue

                sec_return = sec_current / sec_prev - 1.0
                sector_returns[sector] = sec_return
                sector_relative[sector] = sec_return - nifty_return

            if not sector_relative:
                # No sector data available — return a neutral signal
                return {
                    'regime': 'NEUTRAL',
                    'direction': 'NEUTRAL',
                    'leading_sectors': [],
                    'lagging_sectors': [],
                    'confidence': 0.30,
                    'rotation_signal': 0.0,
                    'nifty_return_20d': round(nifty_return, 6),
                    'mode': 'FALLBACK',
                }

            # Identify outperformers and underperformers
            leading = [s for s, r in sector_relative.items() if r > OUTPERFORMANCE_THRESHOLD]
            lagging = [s for s, r in sector_relative.items() if r < -OUTPERFORMANCE_THRESHOLD]

            # Classify regime
            defensive_leading = [s for s in leading if s in DEFENSIVE_SECTORS]
            cyclical_leading = [s for s in leading if s in CYCLICAL_SECTORS]

            defensive_score = len(defensive_leading)
            cyclical_score = len(cyclical_leading)

            if cyclical_score > defensive_score and cyclical_score >= 2:
                regime = 'RISK_ON'
                direction = 'LONG'
                rotation_signal = 1.0
            elif defensive_score > cyclical_score and defensive_score >= 1:
                regime = 'RISK_OFF'
                direction = 'SHORT'
                rotation_signal = -1.0
            elif cyclical_score > 0 and defensive_score == 0:
                regime = 'RISK_ON'
                direction = 'LONG'
                rotation_signal = 0.5
            elif defensive_score > 0 and cyclical_score == 0:
                regime = 'RISK_OFF'
                direction = 'SHORT'
                rotation_signal = -0.5
            else:
                regime = 'NEUTRAL'
                direction = 'NEUTRAL'
                rotation_signal = 0.0

            # Confidence
            total_leading = len(leading)
            confidence = BASE_CONFIDENCE + total_leading * SECTOR_CONVICTION_BOOST
            if direction == 'NEUTRAL':
                confidence = max(0.20, confidence * 0.6)
            confidence = max(0.10, min(0.85, confidence))

            return {
                'regime': regime,
                'direction': direction,
                'rotation_signal': rotation_signal,
                'leading_sectors': leading,
                'lagging_sectors': lagging,
                'sector_relative': {k: round(v, 6) for k, v in sector_relative.items()},
                'nifty_return_20d': round(nifty_return, 6),
                'confidence': round(confidence, 3),
                'mode': 'FALLBACK',
            }
        except Exception as e:
            logger.debug('GNN fallback prediction failed: %s', e)
            return None

    # ----------------------------------------------------------
    # Feature construction (for ML mode)
    # ----------------------------------------------------------
    def _build_features(self, df: Any, dt: date) -> Optional[np.ndarray]:
        """Build feature vector for the GNN model."""
        try:
            if not hasattr(df, 'loc'):
                return None

            if hasattr(df.index, 'date'):
                mask = df.index.date <= dt
            else:
                mask = df.index <= str(dt)
            sub = df.loc[mask]

            if sub.empty or len(sub) < RS_LOOKBACK + 1:
                return None

            features = []

            # Nifty return
            nifty_col = _find_column(sub, NIFTY_COLUMN_PATTERNS)
            nifty_return = 0.0
            if nifty_col:
                vals = sub[nifty_col].astype(float).values
                if len(vals) >= RS_LOOKBACK + 1 and vals[-(RS_LOOKBACK + 1)] > 0:
                    nifty_return = vals[-1] / vals[-(RS_LOOKBACK + 1)] - 1.0
            features.append(nifty_return)

            # Sector returns (relative to Nifty)
            for sector in SECTORS:
                col = _find_column(sub, SECTOR_COLUMN_PATTERNS[sector])
                if col:
                    vals = sub[col].astype(float).values
                    if len(vals) >= RS_LOOKBACK + 1 and vals[-(RS_LOOKBACK + 1)] > 0:
                        sec_ret = vals[-1] / vals[-(RS_LOOKBACK + 1)] - 1.0
                        features.append(sec_ret - nifty_return)
                    else:
                        features.append(0.0)
                else:
                    features.append(0.0)

            return np.array(features, dtype=np.float64)
        except Exception as e:
            logger.debug('GNN feature build failed: %s', e)
            return None

    # ----------------------------------------------------------
    # Main evaluate
    # ----------------------------------------------------------
    def evaluate(self, df: Any, dt: date) -> Optional[Dict]:
        """
        Evaluate the sector rotation signal.

        Parameters
        ----------
        df : DataFrame with Nifty and sector index data.
        dt : Trade date.

        Returns
        -------
        dict with signal_id, direction, strength, price, reason, metadata.
        None if no signal.
        """
        try:
            return self._evaluate_inner(df, dt)
        except Exception as e:
            logger.error('GNNSectorRotation.evaluate error: %s', e, exc_info=True)
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

        regime = result.get('regime', 'NEUTRAL')
        direction = result.get('direction', 'NEUTRAL')
        confidence = result.get('confidence', 0.5)
        leading = result.get('leading_sectors', [])
        lagging = result.get('lagging_sectors', [])
        rotation_signal = result.get('rotation_signal', 0.0)

        # Extract current price
        price = 0.0
        try:
            if hasattr(df, 'loc'):
                nifty_col = _find_column(df, NIFTY_COLUMN_PATTERNS)
                if nifty_col:
                    if hasattr(df.index, 'date'):
                        row = df.loc[df.index.date == dt]
                    else:
                        row = df.loc[df.index == str(dt)]
                    if not row.empty:
                        price = round(float(row[nifty_col].iloc[-1]), 2)
        except Exception:
            pass

        reason_parts = [
            'GNN_SECTOR_ROTATION',
            f"Mode={result.get('mode', 'UNKNOWN')}",
            f"Regime={regime}",
            f"Leading={','.join(leading) if leading else 'NONE'}",
            f"Lagging={','.join(lagging) if lagging else 'NONE'}",
            f"RotSignal={rotation_signal:+.1f}",
        ]

        return {
            'signal_id': self.SIGNAL_ID,
            'direction': direction,
            'strength': round(confidence, 3),
            'price': price,
            'reason': ' | '.join(reason_parts),
            'metadata': {
                'mode': result.get('mode', 'UNKNOWN'),
                'regime': regime,
                'rotation_signal': rotation_signal,
                'leading_sectors': leading,
                'lagging_sectors': lagging,
                'sector_relative': result.get('sector_relative', {}),
                'nifty_return_20d': result.get('nifty_return_20d'),
                'confidence': confidence,
            },
        }

    def reset(self) -> None:
        """Reset internal state."""
        pass

    def __repr__(self) -> str:
        mode = 'ML' if self._model is not None else 'FALLBACK'
        return f"GNNSectorRotation(signal_id='{self.SIGNAL_ID}', mode='{mode}')"
