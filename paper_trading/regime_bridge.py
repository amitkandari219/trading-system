"""
Regime classification bridge: ML ensemble with fixed-rule fallback.

This module provides intelligent regime classification by attempting to use
a trained ML ensemble (HMM + Random Forest) for regime prediction, with
automatic fallback to fixed ADX/VIX rules if the ensemble is unavailable
or confidence is too low.

Features:
- Loads pre-trained HMM and RF models from models/regime_hmm.pkl and regime_rf.pkl
- Returns standardized result dict with regime, confidence, and method used
- Logs all decisions with confidence levels
- Implements size_modifier based on confidence (1.0 high, 0.8 fallback)
"""

import logging
import os
import pickle
from typing import Dict, Optional

import pandas as pd
import numpy as np

from regime_labeler import RegimeLabeler

logger = logging.getLogger(__name__)

# Must match regime_features.py FEATURE_COLS used during training
ENSEMBLE_FEATURE_COLS = [
    'returns_1d', 'returns_5d', 'returns_20d', 'returns_60d',
    'hvol_20', 'hvol_60', 'hvol_ratio',
    'adx_14',
    'vix', 'vix_change_5d', 'vix_zscore',
    'volume_ratio',
    'bb_width',
    'rsi_14',
    'close_vs_sma_200', 'close_vs_sma_50',
    'macd_hist_norm',
    'atr_pct',
    'rolling_sharpe_20',
]

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')


class RegimeBridge:
    """ML ensemble regime classifier with fixed-rule fallback.

    Attempts to classify market regime using a trained ML ensemble (HMM + RF).
    If models are unavailable or confidence is below threshold, falls back to
    proven fixed-rule classification based on ADX and VIX.

    Attributes:
        confidence_threshold: Minimum confidence for ensemble prediction (default 0.6).
            Below this threshold, fixed rules are used.
        fixed_labeler: RegimeLabeler instance for fallback classification.
        ensemble: RegimeEnsemble instance if models load successfully, None otherwise.
    """

    def __init__(self, confidence_threshold: float = 0.6):
        """Initialize the regime bridge.

        Args:
            confidence_threshold: Confidence threshold (0-1) for ensemble predictions.
                Predictions below this trigger fallback to fixed rules.
        """
        self.confidence_threshold = confidence_threshold
        self.fixed_labeler = RegimeLabeler()
        self.ensemble = None
        self._load_ensemble()

    def _load_ensemble(self) -> None:
        """Attempt to load trained ensemble models.

        Looks for pre-trained HMM and RF models in models/saved/ directory.
        Models are saved via RegimeHMM.save() / RegimeRF.save() which pickle
        dicts — so we instantiate fresh objects and call .load() on them.
        """
        saved_dir = os.path.join(MODEL_DIR, 'saved')
        hmm_path = os.path.join(saved_dir, 'regime_hmm.pkl')
        rf_path = os.path.join(saved_dir, 'regime_rf.pkl')

        if not (os.path.exists(hmm_path) and os.path.exists(rf_path)):
            logger.info("Regime ensemble models not found in models/saved/ — using fixed rules only")
            return

        try:
            from models.regime_ensemble import RegimeEnsemble
            from models.regime_hmm import RegimeHMM
            from models.regime_rf import RegimeRF

            self.ensemble = RegimeEnsemble(confidence_threshold=self.confidence_threshold)

            # Load using each model's .load() method (handles dict unpacking)
            hmm = RegimeHMM()
            hmm.load(hmm_path)
            self.ensemble.hmm = hmm

            rf = RegimeRF()
            rf.load(rf_path)
            self.ensemble.rf = rf

            logger.info("Regime ensemble loaded successfully (HMM + RF) from models/saved/")
        except Exception as e:
            logger.warning(f"Failed to load regime ensemble: {e} — falling back to fixed rules")
            self.ensemble = None

    def classify(self, today_row: pd.Series, df: pd.DataFrame = None) -> Dict:
        """Classify regime for today using ML ensemble with fixed-rule fallback.

        Attempts to classify the market regime for a given day using the trained
        ensemble model. If ensemble is unavailable, confidence is too low, or
        classification fails, falls back to proven fixed-rule classification.

        Args:
            today_row: Series with indicator values for today. Must contain at least
                the columns in ENSEMBLE_FEATURE_COLS for ensemble classification.
            df: Full DataFrame (used for fixed-rule fallback if needed).

        Returns:
            Dict with keys:
                'regime': str
                    One of: 'TRENDING_UP', 'TRENDING_DOWN', 'RANGING', 'HIGH_VOL', 'CRISIS'
                'confidence': float (0-1)
                    Confidence score for the prediction. High for ensemble, 0.5 for fallback.
                'method': str
                    Either 'ensemble' or 'fixed_rules' indicating which method was used.
                'details': dict
                    Per-model breakdown for ensemble, or fallback reason for fixed rules.
                'size_modifier': float
                    Recommended position size multiplier (1.0 for high confidence,
                    0.85 medium, 0.8 for low confidence/fallback).
        """
        # Try ML ensemble first
        if self.ensemble is not None:
            try:
                features = self._extract_features(today_row)
                if features is not None:
                    label, confidence, details = self.ensemble.predict_single(
                        features, ENSEMBLE_FEATURE_COLS
                    )

                    if confidence >= self.confidence_threshold:
                        # High confidence — trust the ensemble
                        size_mod = 1.0 if confidence >= 0.75 else 0.85
                        logger.info(
                            f"Regime (ensemble): {label} conf={confidence:.2f} "
                            f"[HMM={details.get('hmm')}, RF={details.get('rf')}, "
                            f"agree={details.get('agree')}]"
                        )
                        return {
                            'regime': label,
                            'confidence': confidence,
                            'method': 'ensemble',
                            'details': details,
                            'size_modifier': size_mod,
                        }
                    else:
                        logger.info(
                            f"Regime ensemble low confidence ({confidence:.2f} < {self.confidence_threshold}) "
                            f"— falling back to fixed rules"
                        )
            except Exception as e:
                logger.warning(f"Regime ensemble prediction failed: {e} — falling back to fixed rules")

        # Fixed-rule fallback
        fixed_regime = self._fixed_rule_classify(today_row, df)
        logger.info(f"Regime (fixed rules): {fixed_regime}")
        return {
            'regime': fixed_regime,
            'confidence': 0.5,  # Fixed rules get moderate confidence
            'method': 'fixed_rules',
            'details': {'fallback_reason': 'ensemble_unavailable_or_low_confidence'},
            'size_modifier': 0.8,  # Slightly reduce sizing when using fallback
        }

    def _extract_features(self, row: pd.Series) -> Optional[Dict]:
        """Extract feature dict from today's indicator row.

        The ensemble expects derived features (returns, rolling stats) matching
        regime_features.py FEATURE_COLS. We map from the raw indicators available
        in signal_compute's output to the expected feature names.

        Args:
            row: Series containing indicator values for a single day.
                Expected keys from signal_compute: close, prev_close, sma_50,
                sma_200, adx_14, india_vix, volume, bb_bandwidth, rsi_14,
                atr_14, macd_hist, hvol_20, etc.

        Returns:
            Dict mapping feature names to values if enough features available,
            None if critical features are missing.
        """
        try:
            close = float(row.get('close', 0))
            prev_close = float(row.get('prev_close', close))
            if close == 0:
                return None

            features = {}

            # Returns (approximate from available data)
            features['returns_1d'] = (close - prev_close) / prev_close if prev_close else 0
            features['returns_5d'] = float(row.get('returns_5d', features['returns_1d']))
            features['returns_20d'] = float(row.get('returns_20d', features['returns_1d']))
            features['returns_60d'] = float(row.get('returns_60d', features['returns_1d']))

            # Volatility
            features['hvol_20'] = float(row.get('hvol_20', 0.15))
            features['hvol_60'] = float(row.get('hvol_60', features['hvol_20']))
            features['hvol_ratio'] = features['hvol_20'] / features['hvol_60'] if features['hvol_60'] else 1.0

            # ADX
            features['adx_14'] = float(row.get('adx_14', 20))

            # VIX features
            vix = float(row.get('india_vix', 15))
            features['vix'] = vix
            features['vix_change_5d'] = float(row.get('vix_change_5d', 0))
            features['vix_zscore'] = float(row.get('vix_zscore', 0))

            # Volume
            features['volume_ratio'] = float(row.get('vol_ratio_20', row.get('volume_ratio', 1.0)))

            # Bollinger
            features['bb_width'] = float(row.get('bb_bandwidth', row.get('bb_width', 0.05)))

            # RSI
            features['rsi_14'] = float(row.get('rsi_14', 50))

            # Price vs SMA
            sma_200 = float(row.get('sma_200', close))
            sma_50 = float(row.get('sma_50', close))
            features['close_vs_sma_200'] = (close / sma_200 - 1) if sma_200 else 0
            features['close_vs_sma_50'] = (close / sma_50 - 1) if sma_50 else 0

            # MACD normalized
            macd_hist = float(row.get('macd_hist', 0))
            features['macd_hist_norm'] = macd_hist / close if close else 0

            # ATR as %
            atr_val = float(row.get('atr_14', close * 0.015))
            features['atr_pct'] = atr_val / close if close else 0.015

            # Rolling Sharpe
            features['rolling_sharpe_20'] = float(row.get('rolling_sharpe_20', 0))

            # Validate no NaN
            for k, v in features.items():
                if pd.isna(v):
                    logger.debug(f"NaN in feature '{k}' — falling back")
                    return None

            return features

        except Exception as e:
            logger.debug(f"Feature extraction failed: {e}")
            return None

    def _fixed_rule_classify(self, today_row: pd.Series, df: pd.DataFrame = None) -> str:
        """Classify using fixed ADX/VIX rules.

        Implements simple but proven rules:
        - CRISIS: VIX >= 25
        - HIGH_VOL: VIX >= 18
        - TRENDING_UP: ADX > 25 AND close > SMA-50
        - TRENDING_DOWN: ADX > 25 AND close < SMA-50
        - RANGING: all others

        Args:
            today_row: Series containing today's indicators.
            df: Optional full DataFrame (not used currently).

        Returns:
            str regime label (CRISIS, HIGH_VOL, TRENDING_UP, TRENDING_DOWN, or RANGING).
        """
        vix = float(today_row.get('india_vix', 15)) if pd.notna(today_row.get('india_vix')) else 15.0
        adx = float(today_row.get('adx_14', 20)) if pd.notna(today_row.get('adx_14')) else 20.0
        close = float(today_row.get('close', 0))
        sma_50 = float(today_row.get('sma_50', close)) if pd.notna(today_row.get('sma_50')) else close

        if vix >= 25:
            return 'CRISIS'
        if vix >= 18:
            return 'HIGH_VOL'
        if adx > 25:
            if close > sma_50:
                return 'TRENDING_UP'
            else:
                return 'TRENDING_DOWN'
        return 'RANGING'
