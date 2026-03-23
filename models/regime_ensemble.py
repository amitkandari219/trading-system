"""
Ensemble regime classifier: HMM + RF weighted vote.

HMM captures regime persistence (temporal structure).
RF captures feature-regime relationships (cross-sectional).
Ensemble combines both for robust classification.
"""

import numpy as np
from models.regime_hmm import RegimeHMM, REGIME_NAMES
from models.regime_rf import RegimeRF


class RegimeEnsemble:
    """Weighted ensemble of HMM + RF regime classifiers."""

    def __init__(self, hmm_weight=0.6, rf_weight=0.4, confidence_threshold=0.6):
        self.hmm = RegimeHMM()
        self.rf = RegimeRF()
        self.hmm_weight = hmm_weight
        self.rf_weight = rf_weight
        self.confidence_threshold = confidence_threshold

    def fit(self, features_df, feature_cols):
        """Train both models."""
        hmm_states = self.hmm.fit(features_df, feature_cols)
        rf_labels = self.rf.fit(features_df, feature_cols)
        return hmm_states, rf_labels

    def predict(self, features_df, feature_cols):
        """
        Predict regime using weighted ensemble.

        Returns:
            labels: list of regime strings
            confidences: list of float (0-1)
            details: list of dicts with per-model breakdown
        """
        hmm_labels, hmm_probs, hmm_conf = self.hmm.predict(features_df, feature_cols)

        # Inject HMM state as RF feature (RF was trained with hmm_state column)
        features_df = features_df.copy()
        try:
            X_scaled = self.hmm.scaler.transform(features_df[feature_cols].values)
            features_df['hmm_state'] = self.hmm.model.predict(X_scaled)
        except Exception:
            features_df['hmm_state'] = 0

        rf_labels, rf_probs, rf_conf = self.rf.predict(features_df)

        labels = []
        confidences = []
        details = []

        for i in range(len(hmm_labels)):
            # Weighted vote
            votes = {}
            for regime in REGIME_NAMES:
                # HMM probability for this regime
                hmm_p = 0
                for state_idx, name in self.hmm.state_map.items():
                    if name == regime and state_idx < hmm_probs.shape[1]:
                        hmm_p = hmm_probs[i, state_idx]

                # RF probability for this regime
                rf_p = 0
                if regime in self.rf.model.classes_:
                    rf_idx = list(self.rf.model.classes_).index(regime)
                    if rf_idx < rf_probs.shape[1]:
                        rf_p = rf_probs[i, rf_idx]

                votes[regime] = self.hmm_weight * hmm_p + self.rf_weight * rf_p

            # Pick highest weighted vote
            best_regime = max(votes, key=votes.get)
            confidence = votes[best_regime]

            # If confidence below threshold, default to RANGING (safest)
            if confidence < self.confidence_threshold:
                best_regime = 'RANGING'
                confidence = self.confidence_threshold

            labels.append(best_regime)
            confidences.append(round(confidence, 3))
            details.append({
                'hmm': hmm_labels[i],
                'hmm_conf': round(hmm_conf[i], 3),
                'rf': rf_labels[i],
                'rf_conf': round(rf_conf[i], 3),
                'ensemble': best_regime,
                'ensemble_conf': round(confidence, 3),
                'agree': hmm_labels[i] == rf_labels[i],
            })

        return labels, confidences, details

    def predict_single(self, features_row, feature_cols):
        """Predict regime for a single day (for signal_compute.py)."""
        import pandas as pd
        df = pd.DataFrame([features_row])
        labels, confs, details = self.predict(df, feature_cols)
        return labels[0], confs[0], details[0]
