"""
Hidden Markov Model for market regime detection.

5 states: TRENDING_UP, TRENDING_DOWN, RANGING, HIGH_VOL, CRISIS
Trained on standardized daily features. Output: regime label + probabilities.
"""

import numpy as np
import pandas as pd
import pickle
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


REGIME_NAMES = ['TRENDING_UP', 'TRENDING_DOWN', 'RANGING', 'HIGH_VOL', 'CRISIS']


class RegimeHMM:
    """Gaussian HMM for regime classification."""

    def __init__(self, n_states=5, n_iter=200, random_state=42):
        self.n_states = n_states
        self.n_iter = n_iter
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.state_map = {}  # HMM state index -> regime name

    def fit(self, features_df, feature_cols):
        """
        Train HMM on feature matrix.

        Args:
            features_df: DataFrame with feature columns
            feature_cols: list of column names to use
        """
        X = features_df[feature_cols].values
        X_scaled = self.scaler.fit_transform(X)

        self.model = GaussianHMM(
            n_components=self.n_states,
            covariance_type='full',
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        self.model.fit(X_scaled)

        # Decode states
        states = self.model.predict(X_scaled)

        # Map HMM states to regime names using feature characteristics
        self._map_states(features_df, feature_cols, states)

        return states

    def predict(self, features_df, feature_cols):
        """
        Predict regime for each row.

        Returns:
            labels: list of regime name strings
            probabilities: array of shape (n_samples, n_states)
        """
        X = features_df[feature_cols].values
        X_scaled = self.scaler.transform(X)

        states = self.model.predict(X_scaled)
        probs = self.model.predict_proba(X_scaled)

        labels = [self.state_map.get(s, 'RANGING') for s in states]
        confidences = [float(probs[i, states[i]]) for i in range(len(states))]

        return labels, probs, confidences

    def _map_states(self, features_df, feature_cols, states):
        """
        Map HMM state indices to regime names based on feature means.

        Logic:
        - Highest avg returns_20d + highest adx → TRENDING_UP
        - Lowest avg returns_20d + high adx → TRENDING_DOWN
        - Highest avg vix → CRISIS (if vix > 25)
        - High vix (18-25) → HIGH_VOL
        - Remaining → RANGING
        """
        state_stats = {}
        for s in range(self.n_states):
            mask = states == s
            if mask.sum() == 0:
                continue
            state_stats[s] = {
                'count': int(mask.sum()),
                'returns_20d': float(features_df.loc[mask, 'returns_20d'].mean()),
                'adx_14': float(features_df.loc[mask, 'adx_14'].mean()),
                'vix': float(features_df.loc[mask, 'vix'].mean()),
                'hvol_20': float(features_df.loc[mask, 'hvol_20'].mean()),
                'bb_width': float(features_df.loc[mask, 'bb_width'].mean()),
            }

        # Sort by vix descending to find CRISIS/HIGH_VOL first
        by_vix = sorted(state_stats.items(), key=lambda x: -x[1]['vix'])

        assigned = {}
        used = set()

        # CRISIS: highest VIX state (if mean VIX > 22)
        if by_vix[0][1]['vix'] > 22:
            assigned[by_vix[0][0]] = 'CRISIS'
            used.add(by_vix[0][0])

        # HIGH_VOL: second highest VIX (if mean VIX > 16)
        for s, stats in by_vix:
            if s not in used and stats['vix'] > 16:
                assigned[s] = 'HIGH_VOL'
                used.add(s)
                break

        # TRENDING_UP: highest returns_20d among remaining
        remaining = [(s, st) for s, st in state_stats.items() if s not in used]
        if remaining:
            best_up = max(remaining, key=lambda x: x[1]['returns_20d'])
            assigned[best_up[0]] = 'TRENDING_UP'
            used.add(best_up[0])

        # TRENDING_DOWN: lowest returns_20d among remaining
        remaining = [(s, st) for s, st in state_stats.items() if s not in used]
        if remaining:
            worst = min(remaining, key=lambda x: x[1]['returns_20d'])
            assigned[worst[0]] = 'TRENDING_DOWN'
            used.add(worst[0])

        # RANGING: whatever's left
        for s in state_stats:
            if s not in used:
                assigned[s] = 'RANGING'

        self.state_map = assigned
        self.state_stats = state_stats

    def get_state_summary(self):
        """Return summary of learned states."""
        summary = {}
        for s, name in self.state_map.items():
            stats = self.state_stats.get(s, {})
            summary[name] = {
                'hmm_state': s,
                'count': stats.get('count', 0),
                'avg_returns_20d': round(stats.get('returns_20d', 0) * 100, 2),
                'avg_adx': round(stats.get('adx_14', 0), 1),
                'avg_vix': round(stats.get('vix', 0), 1),
                'avg_hvol': round(stats.get('hvol_20', 0) * 100, 2),
            }
        return summary

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'model': self.model, 'scaler': self.scaler,
                         'state_map': self.state_map}, f)

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.scaler = data['scaler']
        self.state_map = data['state_map']
