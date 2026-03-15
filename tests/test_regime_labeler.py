"""
Unit tests for RegimeLabeler.
Run: python -m pytest tests/test_regime_labeler.py -v
"""

import pandas as pd
import numpy as np
import pytest
from regime_labeler import RegimeLabeler


def _make_history(n_days=200, base_price=20000, trend='up', vix=15.0):
    """Generate synthetic Nifty history for testing."""
    dates = pd.date_range('2023-01-01', periods=n_days, freq='B')
    np.random.seed(42)

    prices = [base_price]
    for i in range(1, n_days):
        if trend == 'up':
            change = np.random.normal(0.001, 0.01)
        elif trend == 'down':
            change = np.random.normal(-0.001, 0.01)
        else:  # ranging
            change = np.random.normal(0, 0.005)
        prices.append(prices[-1] * (1 + change))

    prices = np.array(prices)
    highs = prices * (1 + np.random.uniform(0.001, 0.015, n_days))
    lows = prices * (1 - np.random.uniform(0.001, 0.015, n_days))
    opens = prices * (1 + np.random.normal(0, 0.005, n_days))

    return pd.DataFrame({
        'date': dates[:n_days],
        'open': opens,
        'high': highs,
        'low': lows,
        'close': prices,
        'volume': np.random.randint(100000, 500000, n_days),
        'india_vix': vix + np.random.normal(0, 1, n_days),
    })


class TestRegimeLabeler:

    def test_crisis_regime(self):
        """VIX >= 25 should always return CRISIS."""
        labeler = RegimeLabeler()
        history = _make_history(100, vix=30.0)
        target = history['date'].iloc[-1]
        regime = labeler.label_single_day(history, target)
        assert regime == 'CRISIS'

    def test_high_vol_regime(self):
        """VIX >= 18 but < 25 should return HIGH_VOL."""
        labeler = RegimeLabeler()
        history = _make_history(100, vix=20.0)
        target = history['date'].iloc[-1]
        regime = labeler.label_single_day(history, target)
        assert regime == 'HIGH_VOL'

    def test_low_vix_not_crisis(self):
        """VIX < 18 should never return CRISIS or HIGH_VOL."""
        labeler = RegimeLabeler()
        history = _make_history(100, vix=12.0)
        target = history['date'].iloc[-1]
        regime = labeler.label_single_day(history, target)
        assert regime in ('TRENDING', 'RANGING')

    def test_insufficient_history_returns_ranging(self):
        """Less than minimum history should default to RANGING."""
        labeler = RegimeLabeler()
        history = _make_history(10, vix=12.0)
        target = history['date'].iloc[-1]
        regime = labeler.label_single_day(history, target)
        assert regime == 'RANGING'

    def test_no_lookahead(self):
        """
        Core test: labeling with full data vs truncated data
        must produce identical results.
        """
        labeler = RegimeLabeler()
        history = _make_history(300, vix=14.0)

        test_idx = 150
        test_date = history['date'].iloc[test_idx]

        # Method 1: Batch label with full data
        full_labels = labeler.label_full_history(history)
        batch_label = full_labels[test_date]

        # Method 2: Label with only data up to test_date
        truncated = history[history['date'] <= test_date].copy()
        single_label = labeler.label_single_day(truncated, test_date)

        assert batch_label == single_label, (
            f"LOOKAHEAD: batch={batch_label}, single={single_label}"
        )

    def test_no_lookahead_multiple_dates(self):
        """Test no-lookahead at multiple points in the dataset."""
        labeler = RegimeLabeler()
        history = _make_history(300, vix=14.0)
        full_labels = labeler.label_full_history(history)

        for idx in [50, 100, 150, 200, 250]:
            test_date = history['date'].iloc[idx]
            truncated = history[history['date'] <= test_date].copy()
            single = labeler.label_single_day(truncated, test_date)
            batch = full_labels[test_date]
            assert batch == single, (
                f"LOOKAHEAD at idx {idx}: batch={batch}, single={single}"
            )

    def test_label_full_history_returns_all_dates(self):
        """Every date in input should have a label."""
        labeler = RegimeLabeler()
        history = _make_history(100, vix=14.0)
        labels = labeler.label_full_history(history)
        assert len(labels) == len(history)

    def test_all_regimes_are_valid(self):
        """All labels must be one of the four valid regimes."""
        labeler = RegimeLabeler()
        history = _make_history(200, vix=14.0)
        labels = labeler.label_full_history(history)
        valid = {'TRENDING', 'RANGING', 'HIGH_VOL', 'CRISIS'}
        for dt, regime in labels.items():
            assert regime in valid, f"Invalid regime '{regime}' on {dt}"

    def test_adx_returns_float(self):
        """ADX computation should return a float."""
        labeler = RegimeLabeler()
        history = _make_history(100, vix=14.0)
        adx = labeler._compute_adx(history)
        assert isinstance(adx, float)
        assert 0 <= adx <= 100

    def test_ema_returns_float(self):
        """EMA computation should return a float."""
        labeler = RegimeLabeler()
        history = _make_history(100, vix=14.0)
        ema = labeler._compute_ema(history, 50)
        assert isinstance(ema, float)
        assert ema > 0
