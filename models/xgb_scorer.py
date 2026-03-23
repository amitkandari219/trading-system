"""
Enhanced XGBoost Intensity Scorer v2 with Chan statistical features.

Extends the original 10-dimension scorer with 10 Chan features (total 14 used):
  - Original: extremity, regime, timeframe, volume (4 key features)
  - Chan: zscore_20/40/60, hurst_100, half_life, vr_2/5/10/20, coint_spread_z

Target: next-5-day forward return sign (binary: 1=positive, 0=negative)
Validation: CPCV inner loop for hyperparameter selection
Feature importance: SHAP values

Compatible with existing IntensityScorer interface (score, train, save, load).

Usage:
    scorer = XGBScorerV2()
    scorer.train_full(df_nifty, df_banknifty)
    scorer.save('models/artifacts/xgb_scorer_v2.pkl')

    # At signal fire time:
    p_profit, grade, size_mult = scorer.score(features_dict)
"""

import logging
import os
import pickle
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from models.cpcv import CombinatorialPurgedCV, deflated_sharpe


# Feature columns: 4 original + 10 Chan
ORIGINAL_FEATURES = ['dim1_extremity', 'dim2_regime', 'dim3_timeframe', 'dim4_volume']

CHAN_FEATURES = [
    'chan_zscore_20', 'chan_zscore_40', 'chan_zscore_60',
    'chan_hurst_100',
    'chan_half_life',
    'chan_vr_2', 'chan_vr_5', 'chan_vr_10', 'chan_vr_20',
    'chan_coint_spread_z',
]

ALL_FEATURES_V2 = ORIGINAL_FEATURES + CHAN_FEATURES

# Hyperparameter grid
PARAM_GRID = {
    'max_depth': [3, 5, 7],
    'n_estimators': [100, 200, 500],
    'learning_rate': [0.01, 0.05, 0.1],
}


class XGBScorerV2:
    """Enhanced XGBoost scorer with Chan features and CPCV validation."""

    def __init__(self):
        self.model = None
        self.trained = False
        self.calibrated = False
        self.feature_importances = {}
        self.shap_values = None
        self.best_params = {}
        self.training_metrics = {}
        self.feature_cols = ALL_FEATURES_V2

    def score(self, features_dict: Dict) -> Tuple[float, str, float]:
        """
        Score a signal fire (compatible with IntensityScorer interface).

        Args:
            features_dict: dict containing at least the features in feature_cols

        Returns:
            (p_profit, grade, size_multiplier)
        """
        if not self.trained or self.model is None:
            return 0.5, 'B', 1.0  # neutral fallback

        X = np.array([[features_dict.get(c, 0.5) for c in self.feature_cols]])

        try:
            p_profit = float(self.model.predict_proba(X)[0, 1])
        except Exception as e:
            logger.warning(f"XGB predict failed: {e}")
            return 0.5, 'B', 1.0

        grade, size_mult = self._probability_to_grade(p_profit)
        return p_profit, grade, size_mult

    def train_full(self, df_nifty: pd.DataFrame,
                   df_banknifty: Optional[pd.DataFrame] = None,
                   forward_days: int = 5) -> Dict:
        """
        Full training pipeline:
        1. Build features (original + Chan)
        2. Create target (next N-day return sign)
        3. CPCV hyperparameter search
        4. Train final model
        5. Compute SHAP importance

        Args:
            df_nifty: Nifty daily DataFrame with OHLCV + indicators
            df_banknifty: BankNifty daily DataFrame (optional)
            forward_days: forward return horizon for target

        Returns:
            dict with training results and metrics
        """
        if not XGB_AVAILABLE:
            raise ImportError("xgboost not installed. Run: pip install xgboost")

        logger.info("Building feature matrix...")
        X, y, dates, feature_df = self._build_dataset(
            df_nifty, df_banknifty, forward_days
        )

        logger.info(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Target balance: {y.mean():.3f} (1-class ratio)")

        # CPCV hyperparameter search
        logger.info("Running CPCV hyperparameter search...")
        best_params, search_results = self._cpcv_param_search(X, y, dates)
        self.best_params = best_params
        logger.info(f"Best params: {best_params}")

        # Train final model on all data
        logger.info("Training final model...")
        self._train_final(X, y, best_params)

        # CPCV evaluation of final model
        logger.info("Running CPCV evaluation of final model...")
        cpcv = CombinatorialPurgedCV(n_groups=6, test_groups=2, embargo_days=5)
        cpcv_results = cpcv.backtest_paths(
            X, y, dates,
            model_factory=lambda: xgb.XGBClassifier(
                **best_params, random_state=42, use_label_encoder=False,
                eval_metric='logloss', verbosity=0
            )
        )

        # SHAP feature importance
        if SHAP_AVAILABLE:
            logger.info("Computing SHAP values...")
            self._compute_shap(X, feature_df)

        self.training_metrics = {
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'target_mean': float(y.mean()),
            'best_params': best_params,
            'cpcv_mean_auc': cpcv_results['mean_metrics'].get('auc', None),
            'cpcv_std_auc': cpcv_results['std_metrics'].get('auc', None),
            'cpcv_mean_accuracy': cpcv_results['mean_metrics'].get('accuracy', None),
            'cpcv_n_paths': cpcv_results['n_paths'],
            'search_results': search_results,
        }

        return self.training_metrics

    def _build_dataset(self, df_nifty: pd.DataFrame,
                       df_banknifty: Optional[pd.DataFrame],
                       forward_days: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
        """Build feature matrix and target from raw data."""
        from signals.chan_features import ChanFeatureExtractor
        from backtest.indicators import add_all_indicators

        # Add indicators if not already present
        df = df_nifty.copy()
        if 'sma_20' not in df.columns:
            df = add_all_indicators(df)

        # Compute Chan features
        chan_ext = ChanFeatureExtractor()
        chan_feats = chan_ext.compute_all(df, df_banknifty)

        # Build feature DataFrame
        feature_df = pd.DataFrame(index=df.index)

        # Original features (simplified — computed from indicators)
        close = df['close'].astype(float)
        rsi_14 = df.get('rsi_14', pd.Series(50.0, index=df.index))
        feature_df['dim1_extremity'] = (rsi_14 - 50).abs().clip(0, 30) / 30.0

        # Regime: use Hurst as proxy (trending vs ranging)
        hurst = df.get('hurst_100', pd.Series(0.5, index=df.index))
        feature_df['dim2_regime'] = hurst.fillna(0.5)

        # Timeframe: close vs SMA50 vs SMA200
        sma_50 = df.get('sma_50', close)
        sma_200 = df.get('sma_200', close)
        feature_df['dim3_timeframe'] = np.where(
            close > sma_50, np.where(sma_50 > sma_200, 1.0, 0.6), 0.2
        )

        # Volume
        vol_ratio = df.get('vol_ratio_20', pd.Series(1.0, index=df.index))
        feature_df['dim4_volume'] = (vol_ratio / 2.0).clip(0, 1)

        # Chan features
        for name, series in chan_feats.items():
            feature_df[name] = series

        # Target: next-N-day return sign
        fwd_return = close.shift(-forward_days) / close - 1.0
        target = (fwd_return > 0).astype(int)

        # Drop NaN rows
        valid_mask = feature_df.notna().all(axis=1) & target.notna()
        feature_df = feature_df[valid_mask]
        target = target[valid_mask]
        dates_arr = df.loc[valid_mask, 'date'].values if 'date' in df.columns else np.arange(valid_mask.sum())

        X = feature_df[self.feature_cols].values
        y = target.values

        return X, y, dates_arr, feature_df[self.feature_cols]

    def _cpcv_param_search(self, X: np.ndarray, y: np.ndarray,
                           dates: np.ndarray) -> Tuple[Dict, List]:
        """CPCV inner-loop hyperparameter search."""
        cpcv = CombinatorialPurgedCV(n_groups=6, test_groups=2, embargo_days=5)

        search_results = []
        best_auc = -1
        best_params = {}

        for md in PARAM_GRID['max_depth']:
            for ne in PARAM_GRID['n_estimators']:
                for lr in PARAM_GRID['learning_rate']:
                    params = {
                        'max_depth': md,
                        'n_estimators': ne,
                        'learning_rate': lr,
                        'subsample': 0.8,
                        'colsample_bytree': 0.8,
                    }

                    def make_model(p=params):
                        return xgb.XGBClassifier(
                            **p, random_state=42,
                            use_label_encoder=False,
                            eval_metric='logloss',
                            verbosity=0
                        )

                    results = cpcv.backtest_paths(X, y, dates, make_model)
                    mean_auc = results['mean_metrics'].get('auc', 0) or 0

                    entry = {
                        'params': params.copy(),
                        'mean_auc': mean_auc,
                        'std_auc': results['std_metrics'].get('auc', 0),
                        'mean_accuracy': results['mean_metrics'].get('accuracy', 0),
                        'n_paths': results['n_paths'],
                    }
                    search_results.append(entry)

                    if mean_auc > best_auc:
                        best_auc = mean_auc
                        best_params = params.copy()

                    logger.debug(
                        f"  md={md} ne={ne} lr={lr} → AUC={mean_auc:.4f}"
                    )

        logger.info(f"Best CPCV AUC: {best_auc:.4f} with {best_params}")
        return best_params, search_results

    def _train_final(self, X: np.ndarray, y: np.ndarray, params: Dict):
        """Train final model on full dataset."""
        from sklearn.calibration import CalibratedClassifierCV

        base = xgb.XGBClassifier(
            **params, random_state=42,
            use_label_encoder=False, eval_metric='logloss',
            verbosity=0
        )

        # Calibrate probabilities
        self.model = CalibratedClassifierCV(base, cv=3, method='sigmoid')
        self.model.fit(X, y)

        # Feature importances from uncalibrated model
        base.fit(X, y)
        self.feature_importances = dict(
            sorted(
                zip(self.feature_cols, base.feature_importances_),
                key=lambda x: -x[1]
            )
        )

        self.trained = True
        self.calibrated = True

    def _compute_shap(self, X: np.ndarray, feature_df: pd.DataFrame):
        """Compute SHAP values for feature importance analysis."""
        try:
            # Use base model for SHAP (not calibrated wrapper)
            base = xgb.XGBClassifier(
                **self.best_params, random_state=42,
                use_label_encoder=False, eval_metric='logloss',
                verbosity=0
            )
            base.fit(X, np.zeros(len(X)))  # just for structure
            # Re-fit on actual data
            base.fit(X, (feature_df.index % 2).astype(int) if len(feature_df) > 0 else np.zeros(len(X)))

            # Actually use the trained base model from calibrated wrapper
            # Extract from CalibratedClassifierCV
            if hasattr(self.model, 'calibrated_classifiers_'):
                base_model = self.model.calibrated_classifiers_[0].estimator
            else:
                base_model = base

            explainer = shap.TreeExplainer(base_model)
            # Sample if too large
            sample_size = min(500, len(X))
            X_sample = X[:sample_size]
            self.shap_values = explainer.shap_values(X_sample)
            logger.info("SHAP values computed successfully")
        except Exception as e:
            logger.warning(f"SHAP computation failed: {e}")
            self.shap_values = None

    def get_shap_importance(self) -> Dict[str, float]:
        """Return mean absolute SHAP values per feature."""
        if self.shap_values is None:
            return self.feature_importances

        try:
            if isinstance(self.shap_values, list):
                # Binary classification: use class 1
                vals = np.abs(self.shap_values[1] if len(self.shap_values) > 1
                              else self.shap_values[0])
            else:
                vals = np.abs(self.shap_values)

            mean_shap = np.mean(vals, axis=0)
            importance = dict(zip(self.feature_cols, mean_shap))
            return dict(sorted(importance.items(), key=lambda x: -x[1]))
        except Exception:
            return self.feature_importances

    @staticmethod
    def _probability_to_grade(p_profit: float) -> Tuple[str, float]:
        """Convert probability to grade + size multiplier."""
        if p_profit >= 0.70: return 'S+', 2.0
        if p_profit >= 0.60: return 'A', 1.5
        if p_profit >= 0.50: return 'B', 1.0
        if p_profit >= 0.40: return 'C', 0.6
        if p_profit >= 0.30: return 'D', 0.3
        return 'F', 0.0

    def save(self, path: str):
        """Save model and metadata."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'trained': self.trained,
                'calibrated': self.calibrated,
                'feature_importances': self.feature_importances,
                'best_params': self.best_params,
                'training_metrics': self.training_metrics,
                'feature_cols': self.feature_cols,
                'shap_importance': self.get_shap_importance(),
            }, f)
        logger.info(f"XGBScorerV2 saved to {path}")

    def load(self, path: str):
        """Load model and metadata."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.trained = data['trained']
        self.calibrated = data['calibrated']
        self.feature_importances = data.get('feature_importances', {})
        self.best_params = data.get('best_params', {})
        self.training_metrics = data.get('training_metrics', {})
        self.feature_cols = data.get('feature_cols', ALL_FEATURES_V2)
        logger.info(f"XGBScorerV2 loaded from {path}")
