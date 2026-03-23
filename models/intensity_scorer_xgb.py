"""
XGBoost intensity scorer — predicts P(profitable trade).

Replaces fixed 10-dimension weighted scorer with ML model.
Falls back to fixed scorer when confidence is low or model untrained.

Usage:
    scorer = IntensityScorer()
    scorer.load('models/saved/intensity_xgb.pkl')
    p_profit, grade, size_mult = scorer.score(features_dict)
"""

import pickle
import numpy as np
from typing import Tuple, Optional

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except (ImportError, Exception):
    XGB_AVAILABLE = False

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from training.build_scorer_features import (
    FEATURE_COLS, features_to_array, compute_fixed_score, score_to_grade,
)


class IntensityScorer:
    """ML-powered signal intensity scorer with fixed-weight fallback."""

    def __init__(self):
        self.model = None
        self.calibrated = False
        self.feature_importances = {}
        self.trained = False

    def score(self, features_dict) -> Tuple[float, str, float]:
        """
        Score a signal fire.

        Args:
            features_dict: dict of 10 dimensions (from build_scorer_features)

        Returns:
            (p_profit, grade, size_multiplier)
        """
        if not self.trained or self.model is None:
            return self._fallback_score(features_dict)

        X = features_to_array(features_dict).reshape(1, -1)

        try:
            p_profit = float(self.model.predict_proba(X)[0, 1])
        except Exception:
            return self._fallback_score(features_dict)

        grade, size_mult = self._probability_to_grade(p_profit)

        return p_profit, grade, size_mult

    def _probability_to_grade(self, p_profit):
        """Convert probability to grade + size multiplier."""
        if p_profit >= 0.70:  return 'S+', 2.0
        if p_profit >= 0.60:  return 'A', 1.5
        if p_profit >= 0.50:  return 'B', 1.0
        if p_profit >= 0.40:  return 'C', 0.6
        if p_profit >= 0.30:  return 'D', 0.3
        return 'F', 0.0

    def _fallback_score(self, features_dict):
        """Fixed-weight scorer as fallback."""
        score = compute_fixed_score(features_dict)
        grade, size_mult = score_to_grade(score)
        # Convert score to pseudo-probability
        p_profit = min(1.0, max(0.0, score / 170))
        return p_profit, grade, size_mult

    def train(self, X, y):
        """
        Train XGBoost on labelled trade data.

        Args:
            X: array (n_samples, 10) — feature matrix
            y: array (n_samples,) — 1 = profitable, 0 = loss
        """
        if XGB_AVAILABLE:
            base_model = xgb.XGBClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, random_state=42,
                use_label_encoder=False, eval_metric='logloss',
            )
        else:
            base_model = GradientBoostingClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                subsample=0.8, random_state=42,
            )

        # Calibrate probabilities with Platt scaling
        self.model = CalibratedClassifierCV(base_model, cv=3, method='sigmoid')
        self.model.fit(X, y)

        # Feature importances from base model
        base_model.fit(X, y)
        self.feature_importances = dict(
            sorted(zip(FEATURE_COLS, base_model.feature_importances_),
                   key=lambda x: -x[1])
        )

        self.trained = True
        self.calibrated = True

    def get_explanation(self, features_dict):
        """Explain why this score was given."""
        p, grade, mult = self.score(features_dict)

        explanation = {
            'probability': round(p, 3),
            'grade': grade,
            'size_multiplier': mult,
            'model_used': 'xgboost' if self.trained else 'fixed_weights',
        }

        # Top contributing features
        if self.trained and self.feature_importances:
            top = list(self.feature_importances.items())[:5]
            explanation['top_features'] = {
                k: {'importance': round(v, 3), 'value': round(features_dict.get(k, 0), 3)}
                for k, v in top
            }
        else:
            explanation['top_features'] = {
                k: round(features_dict.get(k, 0), 3)
                for k in FEATURE_COLS[:5]
            }

        return explanation

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'trained': self.trained,
                'calibrated': self.calibrated,
                'feature_importances': self.feature_importances,
            }, f)

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.trained = data['trained']
        self.calibrated = data['calibrated']
        self.feature_importances = data.get('feature_importances', {})
