"""
Kelly-fraction position sizing using IntensityScorer predictions.

Combines XGBoost P(profit) predictions with Kelly criterion to
size positions proportional to edge. Uses quarter-Kelly for safety.

Kelly formula: f* = (bp - q) / b
  where b = avg_win/avg_loss, p = P(win), q = 1-p
  Quarter-Kelly: f = f* / 4

Grade-to-size mapping (from intensity_scorer_xgb.py):
  S+ (p >= 0.70): 2.0x
  A  (p >= 0.60): 1.5x
  B  (p >= 0.50): 1.0x
  C  (p >= 0.40): 0.6x
  D  (p >= 0.30): 0.3x
  F  (p <  0.30): 0.0x (skip trade)
"""
import logging
import os
import pickle
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class KellySizer:
    """Position sizing using Kelly criterion with ML intensity scoring."""

    def __init__(self, kelly_fraction: float = 0.25,
                 max_size_multiplier: float = 2.5,
                 min_size_multiplier: float = 0.5):
        """
        Args:
            kelly_fraction: Fraction of full Kelly to use (0.25 = quarter-Kelly)
            max_size_multiplier: Cap on position size multiplier
            min_size_multiplier: Floor on position size (below this, skip trade)
        """
        self.kelly_fraction = kelly_fraction
        self.max_size = max_size_multiplier
        self.min_size = min_size_multiplier
        self.scorer = None
        self._load_scorer()

        # Per-signal historical stats (updated from backtest results)
        # avg_expectancy_per_trade: mean P&L per trade in points (from backtest)
        # This is the ground truth — Kelly formula can disagree due to distribution skew
        self.signal_stats = {
            'KAUFMAN_DRY_20': {'avg_win_loss_ratio': 1.42, 'historical_win_rate': 0.473,
                               'avg_expectancy_pts': 1543 / 25},  # ₹1,543 / lot_size
            'KAUFMAN_DRY_16': {'avg_win_loss_ratio': 1.35, 'historical_win_rate': 0.486,
                               'avg_expectancy_pts': 1004 / 25},
            'KAUFMAN_DRY_12': {'avg_win_loss_ratio': 1.28, 'historical_win_rate': 0.391,
                               'avg_expectancy_pts': 811 / 25},
            'GUJRAL_DRY_8':   {'avg_win_loss_ratio': 1.30, 'historical_win_rate': 0.461,
                               'avg_expectancy_pts': 643 / 25},
            'GUJRAL_DRY_13':  {'avg_win_loss_ratio': 1.85, 'historical_win_rate': 0.550,
                               'avg_expectancy_pts': 3256 / 25},
        }

    def _load_scorer(self):
        """Load trained IntensityScorer if available."""
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'models', 'saved', 'intensity_xgb.pkl'
        )
        if os.path.exists(model_path):
            try:
                from models.intensity_scorer_xgb import IntensityScorer
                self.scorer = IntensityScorer()
                self.scorer.load(model_path)
                logger.info("IntensityScorer loaded for Kelly sizing")
            except Exception as e:
                logger.warning(f"Failed to load IntensityScorer: {e}")
                self.scorer = None
        else:
            logger.info("IntensityScorer model not found — using historical stats for Kelly sizing")

    def compute_size(self, signal_id: str, features: Dict,
                     regime: str = None, vix: float = None) -> Dict:
        """
        Compute position size multiplier using Kelly criterion.

        Args:
            signal_id: Which signal is firing
            features: Dict of 10 scoring dimensions (for IntensityScorer)
            regime: Current market regime
            vix: Current VIX level

        Returns:
            {
                'size_multiplier': float,
                'p_profit': float,
                'grade': str,
                'kelly_raw': float (full Kelly fraction),
                'kelly_adjusted': float (quarter-Kelly),
                'method': 'ml_scorer' | 'historical_stats',
                'skip_trade': bool,
            }
        """
        # Get P(profit) from ML scorer or historical stats
        if self.scorer and self.scorer.trained and features:
            p_profit, grade, ml_mult = self.scorer.score(features)
            method = 'ml_scorer'
        else:
            stats = self.signal_stats.get(signal_id, {})
            p_profit = stats.get('historical_win_rate', 0.45)
            grade = self._simple_grade(p_profit)
            ml_mult = 1.0
            method = 'historical_stats'

        # Get avg win/loss ratio for Kelly calculation
        stats = self.signal_stats.get(signal_id, {})
        b = stats.get('avg_win_loss_ratio', 1.3)  # Default from portfolio averages

        # Kelly formula: f* = (bp - q) / b
        q = 1.0 - p_profit
        kelly_raw = (b * p_profit - q) / b if b > 0 else 0.0

        # Quarter-Kelly for safety
        kelly_adjusted = kelly_raw * self.kelly_fraction

        # Convert to size multiplier (1.0 = baseline position)
        # Kelly fraction of 0.02 maps to 1.0x, scale linearly
        baseline_kelly = 0.02  # Expected Kelly for a decent signal
        size_mult = kelly_adjusted / baseline_kelly if baseline_kelly > 0 else 1.0

        # OVERRIDE: If Kelly formula gives <=0 but realized expectancy is positive,
        # the signal has edge that Kelly misses (skewed win distribution).
        # Use a floor of 0.5x instead of skipping.
        avg_exp = stats.get('avg_expectancy_pts', None)
        if size_mult <= 0 and avg_exp is not None and avg_exp > 0:
            size_mult = 0.5  # Reduced size but don't skip positive-expectancy signals
            logger.info(
                f"Kelly override for {signal_id}: Kelly<=0 but avg_exp={avg_exp:.1f}pts > 0, "
                f"using floor 0.5x"
            )

        # Cap the multiplier
        size_mult = max(0.0, min(self.max_size, size_mult))

        # VIX adjustment: reduce in high-vol
        if vix and vix > 25:
            vix_reduction = 1.0 - min(0.5, (vix - 25) / 20)
            size_mult *= vix_reduction

        # Skip if below minimum — but never skip positive-expectancy signals
        has_positive_edge = avg_exp is not None and avg_exp > 0
        skip = size_mult < self.min_size and not has_positive_edge
        if has_positive_edge and size_mult < self.min_size:
            size_mult = self.min_size  # Use minimum, don't skip

        if skip:
            logger.info(
                f"Kelly sizing SKIP {signal_id}: p={p_profit:.2f}, grade={grade}, "
                f"kelly={kelly_adjusted:.4f}, size={size_mult:.2f}x (< {self.min_size})"
            )
        else:
            logger.info(
                f"Kelly sizing {signal_id}: p={p_profit:.2f}, grade={grade}, "
                f"kelly_raw={kelly_raw:.4f}, kelly_adj={kelly_adjusted:.4f}, "
                f"size={size_mult:.2f}x"
            )

        return {
            'size_multiplier': round(size_mult, 3),
            'p_profit': round(p_profit, 3),
            'grade': grade,
            'kelly_raw': round(kelly_raw, 4),
            'kelly_adjusted': round(kelly_adjusted, 4),
            'method': method,
            'skip_trade': skip,
        }

    def update_signal_stats(self, signal_id: str, win_rate: float,
                            avg_win_loss: float):
        """Update historical stats for a signal (call after each closed trade batch)."""
        self.signal_stats[signal_id] = {
            'avg_win_loss_ratio': avg_win_loss,
            'historical_win_rate': win_rate,
        }
        logger.info(f"Updated stats for {signal_id}: WR={win_rate:.3f}, W/L={avg_win_loss:.3f}")

    @staticmethod
    def _simple_grade(p_profit: float) -> str:
        """Simple grade from probability."""
        if p_profit >= 0.70: return 'S+'
        if p_profit >= 0.60: return 'A'
        if p_profit >= 0.50: return 'B'
        if p_profit >= 0.40: return 'C'
        if p_profit >= 0.30: return 'D'
        return 'F'
