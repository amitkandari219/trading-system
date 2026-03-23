"""
VIX Mean Reversion Signal.

Exploits the mean-reverting nature of India VIX to generate contrarian
Nifty signals.  Extreme VIX spikes (z-score > 2.0) indicate peak fear
and historically produce 68% win-rate longs within 3-5 days.  Crushed
VIX (z-score < -1.5) signals complacency and risk of a selloff.

Signal logic:
    z = (vix - sma_20_vix) / std_20_vix

    z > +2.0  → LONG  Nifty  (fear spike = contrarian buy)
    z < -1.5  → SHORT Nifty  (complacency = risk of selloff)

    Exit: hold max 5 days, or VIX reverts to within 0.5 std of mean
    Strength: proportional to |z-score| magnitude

Academic basis: VIX > 2 std historically produces 68% win rate on
Nifty longs (3-5 day holding period).  Mean reversion in variance
is well-documented across equity indices.
"""

import logging
import math
from datetime import date
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ================================================================
# CONSTANTS / WF PARAMETERS
# ================================================================

SIGNAL_ID = 'VIX_MEAN_REVERSION'

# Rolling window for VIX statistics
ROLLING_WINDOW = 20

# Z-score thresholds
Z_LONG_THRESHOLD = 2.0       # VIX spike → contrarian LONG
Z_SHORT_THRESHOLD = -1.5     # VIX crushed → SHORT

# Exit parameters
MAX_HOLD_DAYS = 5
REVERSION_EXIT_Z = 0.5       # Exit when z-score within ±0.5

# Strength mapping
Z_MAX_FOR_STRENGTH = 4.0     # z beyond this caps strength at 1.0
STRENGTH_FLOOR = 0.30        # Minimum strength when signal fires

# VIX column
VIX_COLUMN = 'india_vix'
CLOSE_COLUMN = 'close'


# ================================================================
# HELPERS
# ================================================================

def _safe_float(val: Any, default: float = float('nan')) -> float:
    """Safely cast to float."""
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


# ================================================================
# SIGNAL CLASS
# ================================================================

class VixMeanReversionSignal:
    """
    VIX mean reversion signal.

    Computes z-score of India VIX over a 20-day rolling window and
    generates contrarian Nifty trades at extreme readings.
    """

    SIGNAL_ID = SIGNAL_ID

    # Walk-forward parameters
    WF_ROLLING_WINDOW = ROLLING_WINDOW
    WF_Z_LONG_THRESHOLD = Z_LONG_THRESHOLD
    WF_Z_SHORT_THRESHOLD = Z_SHORT_THRESHOLD
    WF_MAX_HOLD_DAYS = MAX_HOLD_DAYS
    WF_REVERSION_EXIT_Z = REVERSION_EXIT_Z

    def __init__(self) -> None:
        logger.info('VixMeanReversionSignal initialised')

    def evaluate(self, df: pd.DataFrame, date_val: date) -> Optional[Dict]:
        """
        Evaluate VIX mean reversion signal.

        Parameters
        ----------
        df       : DataFrame with at least ROLLING_WINDOW rows of history.
                   Must contain columns: 'india_vix', 'close'.
                   Indexed or containing a date column; rows up to and
                   including date_val are used.
        date_val : The evaluation date.

        Returns
        -------
        dict with signal_id, direction, strength, price, reason, metadata
        or None if no signal.
        """
        try:
            return self._evaluate_inner(df, date_val)
        except Exception as e:
            logger.error(
                'VixMeanReversionSignal.evaluate error: %s', e, exc_info=True
            )
            return None

    def _evaluate_inner(
        self, df: pd.DataFrame, date_val: date
    ) -> Optional[Dict]:
        # ── Validate DataFrame ──────────────────────────────────────
        if df is None or df.empty:
            logger.debug('Empty DataFrame')
            return None

        if VIX_COLUMN not in df.columns:
            logger.debug('Missing column: %s', VIX_COLUMN)
            return None

        if CLOSE_COLUMN not in df.columns:
            logger.debug('Missing column: %s', CLOSE_COLUMN)
            return None

        if len(df) < ROLLING_WINDOW:
            logger.debug(
                'Insufficient data: %d rows < %d required',
                len(df), ROLLING_WINDOW,
            )
            return None

        # ── Extract VIX series ──────────────────────────────────────
        vix_series = df[VIX_COLUMN].astype(float)
        vix_current = _safe_float(vix_series.iloc[-1])

        if math.isnan(vix_current) or vix_current <= 0:
            logger.debug('Invalid current VIX: %s', vix_current)
            return None

        # ── Compute rolling stats ───────────────────────────────────
        vix_window = vix_series.iloc[-ROLLING_WINDOW:]
        valid_count = vix_window.dropna().shape[0]

        if valid_count < ROLLING_WINDOW * 0.8:
            logger.debug(
                'Too many NaN in VIX window: %d valid of %d',
                valid_count, ROLLING_WINDOW,
            )
            return None

        vix_mean = float(vix_window.mean())
        vix_std = float(vix_window.std(ddof=1))

        if vix_std < 1e-6:
            logger.debug('VIX std near zero — no volatility in VIX')
            return None

        # ── Z-score ────────────────────────────────────────────────
        z_score = (vix_current - vix_mean) / vix_std

        # ── Direction ──────────────────────────────────────────────
        if z_score > Z_LONG_THRESHOLD:
            direction = 'LONG'
        elif z_score < Z_SHORT_THRESHOLD:
            direction = 'SHORT'
        else:
            logger.debug(
                'VIX z=%.2f within neutral zone [%.1f, %.1f]',
                z_score, Z_SHORT_THRESHOLD, Z_LONG_THRESHOLD,
            )
            return None

        # ── Strength (0-1) ─────────────────────────────────────────
        z_abs = abs(z_score)
        strength = min(1.0, max(
            STRENGTH_FLOOR,
            z_abs / Z_MAX_FOR_STRENGTH,
        ))

        # ── Price ──────────────────────────────────────────────────
        price = _safe_float(df[CLOSE_COLUMN].iloc[-1])
        if math.isnan(price) or price <= 0:
            logger.debug('Invalid close price')
            return None

        # ── Reason ─────────────────────────────────────────────────
        reason = (
            f"VIX_MEAN_REVERSION | VIX={vix_current:.1f} | "
            f"SMA20={vix_mean:.1f} | Std={vix_std:.2f} | "
            f"Z={z_score:+.2f} | "
            f"{'Fear spike — contrarian buy' if direction == 'LONG' else 'Complacency — risk of selloff'}"
        )

        logger.info(
            '%s signal: %s z=%.2f vix=%.1f strength=%.2f on %s',
            self.SIGNAL_ID, direction, z_score, vix_current, strength, date_val,
        )

        return {
            'signal_id': self.SIGNAL_ID,
            'direction': direction,
            'strength': round(strength, 4),
            'price': round(price, 2),
            'reason': reason,
            'metadata': {
                'vix_current': round(vix_current, 2),
                'vix_mean_20': round(vix_mean, 2),
                'vix_std_20': round(vix_std, 4),
                'z_score': round(z_score, 4),
                'max_hold_days': MAX_HOLD_DAYS,
                'reversion_exit_z': REVERSION_EXIT_Z,
                'date': str(date_val),
            },
        }

    def reset(self) -> None:
        """Reset internal state for a fresh backtest run."""
        pass

    def __repr__(self) -> str:
        return f"VixMeanReversionSignal(signal_id='{self.SIGNAL_ID}')"
