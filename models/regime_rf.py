"""
Random Forest regime classifier (supervised).

Uses fixed-rule regime labels as ground truth for training,
then provides feature importances and probability estimates.
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


REGIME_NAMES = ['TRENDING_UP', 'TRENDING_DOWN', 'RANGING', 'HIGH_VOL', 'CRISIS']


def label_regime_rules(row):
    """Fixed-rule regime labeling (ground truth for RF training)."""
    vix = row.get('vix', 15)
    adx = row.get('adx_14', 20)
    ret_20 = row.get('returns_20d', 0)
    close_vs_200 = row.get('close_vs_sma_200', 0)

    if vix >= 25:
        return 'CRISIS'
    elif vix >= 18:
        return 'HIGH_VOL'
    elif adx > 25 and ret_20 > 0.02 and close_vs_200 > 0:
        return 'TRENDING_UP'
    elif adx > 25 and ret_20 < -0.02 and close_vs_200 < 0:
        return 'TRENDING_DOWN'
    else:
        return 'RANGING'


class RegimeRF:
    """Random Forest regime classifier."""

    def __init__(self, n_estimators=200, max_depth=8, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            class_weight='balanced',
        )
        self.scaler = StandardScaler()
        self.feature_cols = None

    def fit(self, features_df, feature_cols, labels=None):
        """
        Train RF on features with regime labels.
        If labels not provided, generates from fixed rules.
        """
        self.feature_cols = feature_cols
        X = features_df[feature_cols].values
        X_scaled = self.scaler.fit_transform(X)

        if labels is None:
            labels = features_df.apply(label_regime_rules, axis=1).values

        self.model.fit(X_scaled, labels)
        return labels

    def predict(self, features_df):
        """Predict regime + probabilities."""
        X = features_df[self.feature_cols].values
        X_scaled = self.scaler.transform(X)

        labels = self.model.predict(X_scaled)
        probs = self.model.predict_proba(X_scaled)
        confidences = [float(probs[i].max()) for i in range(len(labels))]

        return list(labels), probs, confidences

    def feature_importances(self):
        """Return sorted feature importances."""
        if self.feature_cols is None:
            return {}
        imp = self.model.feature_importances_
        return dict(sorted(zip(self.feature_cols, imp), key=lambda x: -x[1]))

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'model': self.model, 'scaler': self.scaler,
                         'feature_cols': self.feature_cols}, f)

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_cols = data['feature_cols']
