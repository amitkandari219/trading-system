"""
CUSUM (Cumulative Sum) changepoint detector for signal decay.

Detects mean shift in P&L faster than rolling Sharpe.
Parameters k (sensitivity) and h (threshold) auto-calibrated
from historical P&L volatility.
"""

import numpy as np
from typing import Tuple


class CUSUMDetector:
    """Sequential CUSUM test on daily P&L."""

    def __init__(self, k=None, h=None):
        self.k = k    # sensitivity (slack), auto-set if None
        self.h = h    # threshold, auto-set if None
        self.S_pos = 0.0  # upper CUSUM
        self.S_neg = 0.0  # lower CUSUM
        self.changepoint_detected = False
        self.days_since_change = 0

    def calibrate(self, pnl_history):
        """
        Auto-calibrate k and h from historical P&L.
        k = 0.5 * target_shift (detect shift of 1 sigma)
        h = 4 * sigma (control false alarm rate)
        """
        if len(pnl_history) < 20:
            self.k = 0.0
            self.h = 1.0
            return

        sigma = np.std(pnl_history)
        mean = np.mean(pnl_history)

        self.k = 0.5 * sigma  # detect shift of 1 sigma
        self.h = 4.0 * sigma  # threshold

    def update(self, pnl_today, expected_mean=0.0) -> Tuple[bool, float]:
        """
        Update CUSUM with today's P&L.

        Args:
            pnl_today: today's P&L (points or ₹)
            expected_mean: expected daily P&L under null hypothesis

        Returns:
            (changepoint_detected, cusum_value)
        """
        residual = pnl_today - expected_mean

        self.S_pos = max(0, self.S_pos + residual - self.k)
        self.S_neg = min(0, self.S_neg + residual + self.k)

        if self.S_pos > self.h or self.S_neg < -self.h:
            self.changepoint_detected = True
            self.days_since_change = 0
            # Reset after detection
            self.S_pos = 0
            self.S_neg = 0
        else:
            self.changepoint_detected = False
            self.days_since_change += 1

        return self.changepoint_detected, max(abs(self.S_pos), abs(self.S_neg))

    def reset(self):
        self.S_pos = 0.0
        self.S_neg = 0.0
        self.changepoint_detected = False
        self.days_since_change = 0
