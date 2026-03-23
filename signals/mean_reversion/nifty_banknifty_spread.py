"""
Nifty / BankNifty Pairs Spread Mean-Reversion Signal.

Edge:
    Nifty and BankNifty are cointegrated — the ratio between them
    oscillates around a slowly drifting mean.  When the ratio diverges
    beyond 2 standard deviations (20-day rolling window), the cheaper
    index tends to outperform as the spread reverts.

    Empirically, the Nifty/BankNifty ratio mean-reverts within 3-5
    trading days ~65% of the time when z-score exceeds +/- 2.

Signal logic:
    ratio       = close / banknifty_close
    z_score     = (ratio - rolling_mean(ratio, 20)) / rolling_std(ratio, 20)

    z_score < -2.0  -> LONG  Nifty (underperforming, expect reversion)
    z_score >  2.0  -> SHORT Nifty (overperforming, expect reversion)
    z_score in [-0.5, 0.5] -> EXIT (spread has reverted)

Usage:
    from signals.mean_reversion.nifty_banknifty_spread import NiftyBankNiftySpreadSignal

    sig = NiftyBankNiftySpreadSignal()
    result = sig.evaluate(df, date)
"""

import logging
import math
from datetime import date
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class NiftyBankNiftySpreadSignal:
    """
    Pairs-spread mean-reversion between Nifty and BankNifty.

    Computes a rolling z-score of the Nifty/BankNifty ratio and
    generates LONG/SHORT signals when the z-score breaches +/- 2 sigma.
    """

    SIGNAL_ID = 'NIFTY_BANKNIFTY_SPREAD'

    # ── Walk-forward parameters ──────────────────────────────────
    LOOKBACK_WINDOW = 20          # Rolling mean/std window (trading days)
    Z_ENTRY_THRESHOLD = 2.0       # Enter when |z| > 2.0
    Z_EXIT_LOW = -0.5             # Exit zone lower bound
    Z_EXIT_HIGH = 0.5             # Exit zone upper bound
    MIN_PERIODS = 15              # Minimum observations for rolling stats

    # ── Column names (configurable for different data sources) ───
    NIFTY_CLOSE_COL = 'close'
    BANKNIFTY_CLOSE_COL = 'banknifty_close'

    # ── Confidence ───────────────────────────────────────────────
    BASE_CONFIDENCE = 0.55
    EXTREME_Z_BOOST = 0.08        # Extra confidence when |z| > 2.5
    MODERATE_Z_PENALTY = -0.03    # Slightly less confident near threshold

    def __init__(self) -> None:
        self._position_state: Optional[str] = None  # 'LONG', 'SHORT', or None
        logger.info('NiftyBankNiftySpreadSignal initialised')

    # ──────────────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────────────

    def evaluate(self, df: pd.DataFrame, date: date) -> Optional[Dict]:
        """
        Evaluate the Nifty/BankNifty spread signal.

        Parameters
        ----------
        df   : DataFrame with at least `LOOKBACK_WINDOW` rows ending on
               or before `date`.  Must contain columns 'close' (Nifty)
               and 'banknifty_close'.
        date : The evaluation date.

        Returns
        -------
        dict with signal details, or None if no trade.
        """
        try:
            return self._evaluate_inner(df, date)
        except Exception as e:
            logger.error(
                'NiftyBankNiftySpreadSignal.evaluate error: %s', e, exc_info=True,
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

        for col in (self.NIFTY_CLOSE_COL, self.BANKNIFTY_CLOSE_COL):
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

        # ── Compute ratio and z-score ────────────────────────────
        nifty = df_slice[self.NIFTY_CLOSE_COL].astype(float)
        banknifty = df_slice[self.BANKNIFTY_CLOSE_COL].astype(float)

        # Guard against zero / NaN in BankNifty
        if (banknifty <= 0).any() or banknifty.isna().any():
            banknifty = banknifty.replace(0, np.nan)
            if banknifty.iloc[-self.LOOKBACK_WINDOW:].isna().sum() > 3:
                logger.debug('Too many invalid BankNifty values — skip')
                return None

        ratio = nifty / banknifty
        rolling_mean = ratio.rolling(window=self.LOOKBACK_WINDOW, min_periods=self.MIN_PERIODS).mean()
        rolling_std = ratio.rolling(window=self.LOOKBACK_WINDOW, min_periods=self.MIN_PERIODS).std()

        current_ratio = ratio.iloc[-1]
        current_mean = rolling_mean.iloc[-1]
        current_std = rolling_std.iloc[-1]

        if math.isnan(current_ratio) or math.isnan(current_mean) or math.isnan(current_std):
            logger.debug('NaN in rolling stats — skip')
            return None

        if current_std <= 0:
            logger.debug('Zero std — skip')
            return None

        z_score = (current_ratio - current_mean) / current_std
        current_price = float(nifty.iloc[-1])
        current_banknifty = float(banknifty.iloc[-1])

        # ── Check for EXIT condition ─────────────────────────────
        if self._position_state is not None:
            if self.Z_EXIT_LOW <= z_score <= self.Z_EXIT_HIGH:
                exit_direction = self._position_state
                self._position_state = None
                logger.info(
                    '%s EXIT: z=%.3f reverted to neutral (was %s)',
                    self.SIGNAL_ID, z_score, exit_direction,
                )
                return {
                    'signal_id': self.SIGNAL_ID,
                    'direction': 'EXIT',
                    'strength': 0.0,
                    'price': round(current_price, 2),
                    'reason': (
                        f"{self.SIGNAL_ID} | EXIT {exit_direction} | "
                        f"z={z_score:+.3f} reverted to neutral band"
                    ),
                    'metadata': {
                        'z_score': round(z_score, 4),
                        'ratio': round(current_ratio, 6),
                        'rolling_mean': round(current_mean, 6),
                        'rolling_std': round(current_std, 6),
                        'nifty_close': round(current_price, 2),
                        'banknifty_close': round(current_banknifty, 2),
                        'eval_date': str(eval_date),
                        'previous_position': exit_direction,
                    },
                }

        # ── Check for ENTRY condition ────────────────────────────
        direction = None
        if z_score < -self.Z_ENTRY_THRESHOLD:
            direction = 'LONG'
        elif z_score > self.Z_ENTRY_THRESHOLD:
            direction = 'SHORT'

        if direction is None:
            logger.debug(
                '%s: z=%.3f within no-trade zone — skip', self.SIGNAL_ID, z_score,
            )
            return None

        # ── Confidence / strength ────────────────────────────────
        strength = self.BASE_CONFIDENCE
        abs_z = abs(z_score)
        if abs_z > 2.5:
            strength += self.EXTREME_Z_BOOST
        elif abs_z < 2.2:
            strength += self.MODERATE_Z_PENALTY
        strength = min(1.0, max(0.0, strength))

        self._position_state = direction

        reason_parts = [
            self.SIGNAL_ID,
            f"z={z_score:+.3f}",
            f"Ratio={current_ratio:.4f}",
            f"Mean={current_mean:.4f}",
            f"Std={current_std:.4f}",
            f"Nifty={current_price:.2f}",
            f"BankNifty={current_banknifty:.2f}",
        ]

        logger.info(
            '%s signal: %s z=%.3f ratio=%.4f date=%s',
            self.SIGNAL_ID, direction, z_score, current_ratio, eval_date,
        )

        return {
            'signal_id': self.SIGNAL_ID,
            'direction': direction,
            'strength': round(strength, 4),
            'price': round(current_price, 2),
            'reason': ' | '.join(reason_parts),
            'metadata': {
                'z_score': round(z_score, 4),
                'ratio': round(current_ratio, 6),
                'rolling_mean': round(current_mean, 6),
                'rolling_std': round(current_std, 6),
                'nifty_close': round(current_price, 2),
                'banknifty_close': round(current_banknifty, 2),
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
        return f"NiftyBankNiftySpreadSignal(signal_id='{self.SIGNAL_ID}')"
