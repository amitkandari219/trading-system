"""
Cash-Futures Basis Z-Score Mean-Reversion Signal.

Edge:
    The Nifty futures basis (futures - spot) reflects the market's
    cost-of-carry plus a sentiment premium/discount.  Under normal
    conditions, basis trades in a narrow band set by interest rates.
    When fear spikes, futures trade at an unusual discount (negative
    basis); when greed dominates, futures trade at an unusual premium.

    These extremes revert as arbitrageurs and market makers step in.
    A z-score below -2 on the normalised basis historically precedes
    a spot rally (buy-the-fear), and above +2 precedes a pullback.

Signal logic:
    basis_pct    = (futures_close - close) / close
    z_score      = rolling_zscore(basis_pct, window=20)

    z_score < -2.0  -> LONG  (futures discount = fear = buy spot)
    z_score >  2.0  -> SHORT (futures premium = greed = sell spot)

Usage:
    from signals.mean_reversion.basis_zscore import BasisZScoreSignal

    sig = BasisZScoreSignal()
    result = sig.evaluate(df, date)
"""

import logging
import math
from datetime import date
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BasisZScoreSignal:
    """
    Cash-futures basis mispricing signal.

    Computes a rolling z-score of the normalised basis between
    Nifty spot and Nifty futures, and generates mean-reversion
    signals at extreme readings.
    """

    SIGNAL_ID = 'BASIS_ZSCORE'

    # ── Walk-forward parameters ──────────────────────────────────
    LOOKBACK_WINDOW = 20           # Rolling z-score window (trading days)
    Z_ENTRY_THRESHOLD = 2.0        # Enter when |z| > 2.0
    Z_EXIT_LOW = -0.5              # Exit zone lower bound
    Z_EXIT_HIGH = 0.5              # Exit zone upper bound
    MIN_PERIODS = 15               # Minimum obs for rolling stats

    # ── Column names ─────────────────────────────────────────────
    SPOT_CLOSE_COL = 'close'
    FUTURES_CLOSE_COL = 'futures_close'

    # ── Confidence ───────────────────────────────────────────────
    BASE_CONFIDENCE = 0.55
    EXTREME_Z_BOOST = 0.10         # Extra when |z| > 2.5
    MILD_Z_PENALTY = -0.03         # Slightly less near threshold

    def __init__(self) -> None:
        self._position_state: Optional[str] = None  # 'LONG', 'SHORT', or None
        logger.info('BasisZScoreSignal initialised')

    # ──────────────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────────────

    def evaluate(self, df: pd.DataFrame, date: date) -> Optional[Dict]:
        """
        Evaluate the cash-futures basis z-score signal.

        Parameters
        ----------
        df   : DataFrame with at least `LOOKBACK_WINDOW` rows ending on
               or before `date`.  Must contain 'close' (spot) and
               'futures_close' columns.
        date : The evaluation date.

        Returns
        -------
        dict with signal details, or None if no trade.
        """
        try:
            return self._evaluate_inner(df, date)
        except Exception as e:
            logger.error(
                'BasisZScoreSignal.evaluate error: %s', e, exc_info=True,
            )
            return None

    # ──────────────────────────────────────────────────────────────
    # INTERNALS
    # ──────────────────────────────────────────────────────────────

    def _evaluate_inner(self, df: pd.DataFrame, eval_date: date) -> Optional[Dict]:
        # ── Validate DataFrame ───────────────────────────────────
        if df is None or df.empty:
            logger.debug('Empty DataFrame — skip')
            return None

        for col in (self.SPOT_CLOSE_COL, self.FUTURES_CLOSE_COL):
            if col not in df.columns:
                logger.debug('Missing column %s — skip', col)
                return None

        # ── Slice up to eval_date ────────────────────────────────
        if hasattr(df.index, 'date'):
            mask = df.index.date <= eval_date if isinstance(eval_date, date) else df.index <= eval_date
            df_slice = df.loc[mask]
        else:
            df_slice = df

        if len(df_slice) < self.MIN_PERIODS:
            logger.debug(
                'Insufficient data: %d rows < %d min_periods',
                len(df_slice), self.MIN_PERIODS,
            )
            return None

        # ── Compute basis percentage and z-score ─────────────────
        spot = df_slice[self.SPOT_CLOSE_COL].astype(float)
        futures = df_slice[self.FUTURES_CLOSE_COL].astype(float)

        # Guard against zero / NaN spot
        if (spot <= 0).any():
            spot = spot.replace(0, np.nan)
        if spot.iloc[-self.LOOKBACK_WINDOW:].isna().sum() > 3:
            logger.debug('Too many invalid spot values — skip')
            return None

        basis_pct = (futures - spot) / spot

        rolling_mean = basis_pct.rolling(
            window=self.LOOKBACK_WINDOW, min_periods=self.MIN_PERIODS,
        ).mean()
        rolling_std = basis_pct.rolling(
            window=self.LOOKBACK_WINDOW, min_periods=self.MIN_PERIODS,
        ).std()

        current_basis_pct = float(basis_pct.iloc[-1])
        current_mean = float(rolling_mean.iloc[-1])
        current_std = float(rolling_std.iloc[-1])
        current_spot = float(spot.iloc[-1])
        current_futures = float(futures.iloc[-1])

        if any(math.isnan(v) for v in (current_basis_pct, current_mean, current_std)):
            logger.debug('NaN in rolling stats — skip')
            return None

        if current_std <= 0:
            logger.debug('Zero std — skip')
            return None

        z_score = (current_basis_pct - current_mean) / current_std

        # ── Check for EXIT condition ─────────────────────────────
        if self._position_state is not None:
            if self.Z_EXIT_LOW <= z_score <= self.Z_EXIT_HIGH:
                exit_direction = self._position_state
                self._position_state = None
                logger.info(
                    '%s EXIT: z=%.3f reverted (was %s)',
                    self.SIGNAL_ID, z_score, exit_direction,
                )
                return {
                    'signal_id': self.SIGNAL_ID,
                    'direction': 'EXIT',
                    'strength': 0.0,
                    'price': round(current_spot, 2),
                    'reason': (
                        f"{self.SIGNAL_ID} | EXIT {exit_direction} | "
                        f"basis_z={z_score:+.3f} reverted to neutral"
                    ),
                    'metadata': {
                        'z_score': round(z_score, 4),
                        'basis_pct': round(current_basis_pct * 100, 4),
                        'spot_close': round(current_spot, 2),
                        'futures_close': round(current_futures, 2),
                        'eval_date': str(eval_date),
                        'previous_position': exit_direction,
                    },
                }

        # ── Check for ENTRY condition ────────────────────────────
        direction = None
        if z_score < -self.Z_ENTRY_THRESHOLD:
            direction = 'LONG'    # Futures discount = fear = buy
        elif z_score > self.Z_ENTRY_THRESHOLD:
            direction = 'SHORT'   # Futures premium = greed = sell

        if direction is None:
            logger.debug(
                '%s: basis_z=%.3f — no signal', self.SIGNAL_ID, z_score,
            )
            return None

        # ── Confidence / strength ────────────────────────────────
        strength = self.BASE_CONFIDENCE
        abs_z = abs(z_score)
        if abs_z > 2.5:
            strength += self.EXTREME_Z_BOOST
        elif abs_z < 2.2:
            strength += self.MILD_Z_PENALTY
        strength = min(1.0, max(0.0, strength))

        self._position_state = direction

        basis_pts = current_futures - current_spot

        reason_parts = [
            self.SIGNAL_ID,
            f"basis_z={z_score:+.3f}",
            f"Basis={basis_pts:+.2f}pts ({current_basis_pct * 100:+.3f}%)",
            f"Spot={current_spot:.2f}",
            f"Futures={current_futures:.2f}",
            f"RollingMean={current_mean * 100:.3f}%",
        ]

        logger.info(
            '%s signal: %s basis_z=%.3f basis_pct=%.4f date=%s',
            self.SIGNAL_ID, direction, z_score, current_basis_pct, eval_date,
        )

        return {
            'signal_id': self.SIGNAL_ID,
            'direction': direction,
            'strength': round(strength, 4),
            'price': round(current_spot, 2),
            'reason': ' | '.join(reason_parts),
            'metadata': {
                'z_score': round(z_score, 4),
                'basis_pct': round(current_basis_pct * 100, 4),
                'basis_pts': round(basis_pts, 2),
                'rolling_mean_pct': round(current_mean * 100, 4),
                'rolling_std_pct': round(current_std * 100, 4),
                'spot_close': round(current_spot, 2),
                'futures_close': round(current_futures, 2),
                'lookback_window': self.LOOKBACK_WINDOW,
                'eval_date': str(eval_date),
            },
        }

    # ──────────────────────────────────────────────────────────────
    # UTILITY
    # ──────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset internal state for a fresh backtest run."""
        self._position_state = None

    def __repr__(self) -> str:
        return f"BasisZScoreSignal(signal_id='{self.SIGNAL_ID}')"
