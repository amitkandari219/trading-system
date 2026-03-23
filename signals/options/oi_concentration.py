"""
OI Concentration Ratio Signal.

Measures how concentrated open interest is across the options chain.
When OI clusters heavily at a few strikes (>50%), option writers are
"pinning" the index — price tends to mean-revert toward those strikes.
When OI is dispersed (<25%), the market is trending and momentum
strategies are preferred.

Signal logic:
    oi_concentration = OI at top 3 strikes / total OI

    concentration > 50%:
        -> Pinning regime — mean reversion expected
        -> If price BELOW concentration zone center -> LONG
        -> If price ABOVE concentration zone center -> SHORT

    concentration < 25%:
        -> Dispersed / trending regime — no directional signal from OI
        -> Acts as a regime modifier (metadata reports regime)
        -> Returns signal with direction=None (regime info only)

    25% <= concentration <= 50%:
        -> Neutral — no signal

    Strength scales with concentration level and distance from
    concentration zone center.

Columns required in df:
    close, oi_concentration_ratio (0-1 scale)
    Optional: oi_concentration_center (weighted avg of top-3 strikes)

Walk-forward parameters exposed as class constants.

Usage:
    from signals.options.oi_concentration import OIConcentrationSignal
    sig = OIConcentrationSignal()
    result = sig.evaluate(df, date)
"""

import logging
import math
from datetime import date
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ================================================================
# CONSTANTS  (walk-forward tunable)
# ================================================================

SIGNAL_ID = 'OI_CONCENTRATION'

# Concentration thresholds (ratio 0-1)
HIGH_CONCENTRATION = 0.50         # > 50% -> pinning regime
LOW_CONCENTRATION = 0.25          # < 25% -> dispersed / trending
EXTREME_CONCENTRATION = 0.75      # Very high -> very strong pinning

# Distance from concentration center to trigger directional trade
MIN_DISTANCE_PCT = 0.3            # Minimum % away from center to trade
MAX_DISTANCE_PCT = 3.0            # Sanity cap

# Strength
BASE_STRENGTH = 0.40
CONCENTRATION_STRENGTH_SCALE = 0.20  # Bonus at extreme concentration
DISTANCE_STRENGTH_SCALE = 0.08       # Per 1% distance beyond min
REGIME_ONLY_STRENGTH = 0.20          # When reporting regime only (dispersed)
MAX_STRENGTH = 0.85
MIN_STRENGTH = 0.10


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

class OIConcentrationSignal:
    """
    OI concentration ratio signal and regime modifier.

    High OI concentration (>50%) signals a pinning regime with mean
    reversion toward the concentration zone.  Low concentration (<25%)
    signals a dispersed/trending regime where momentum signals are
    preferred.  Acts as both a directional signal and a regime modifier.
    """

    SIGNAL_ID = SIGNAL_ID

    # -- Walk-forward params --
    WF_HIGH_CONCENTRATION = HIGH_CONCENTRATION
    WF_LOW_CONCENTRATION = LOW_CONCENTRATION
    WF_EXTREME_CONCENTRATION = EXTREME_CONCENTRATION
    WF_MIN_DISTANCE_PCT = MIN_DISTANCE_PCT
    WF_MAX_DISTANCE_PCT = MAX_DISTANCE_PCT
    WF_BASE_STRENGTH = BASE_STRENGTH

    def __init__(self) -> None:
        self._last_fire_date: Optional[date] = None
        logger.info('OIConcentrationSignal initialised')

    # ----------------------------------------------------------
    # evaluate
    # ----------------------------------------------------------
    def evaluate(self, df: pd.DataFrame, trade_date: date) -> Optional[Dict]:
        """
        Evaluate OI concentration signal.

        Parameters
        ----------
        df         : DataFrame with columns [close, oi_concentration_ratio].
                     Optionally [oi_concentration_center] for the weighted
                     average strike of top-3 OI.  If center is absent,
                     signal reports regime only (no directional call).
        trade_date : The date to evaluate.

        Returns
        -------
        dict with keys: signal_id, direction, strength, price, reason,
        metadata — or None if in the neutral zone (25-50%).
        direction may be None for regime-only signals (dispersed market).
        """
        try:
            return self._evaluate_inner(df, trade_date)
        except Exception as e:
            logger.error('OIConcentrationSignal.evaluate error: %s', e, exc_info=True)
            return None

    def _evaluate_inner(self, df: pd.DataFrame, trade_date: date) -> Optional[Dict]:
        if self._last_fire_date == trade_date:
            return None

        if df is None or df.empty:
            logger.debug('Empty DataFrame')
            return None

        row = df.iloc[-1]
        price = _safe_float(row.get('close'))
        concentration = _safe_float(row.get('oi_concentration_ratio'))

        if math.isnan(price) or price <= 0:
            logger.debug('Invalid price')
            return None
        if math.isnan(concentration):
            logger.debug('Missing oi_concentration_ratio — options data unavailable')
            return None

        # Normalise: if passed as percentage (0-100), convert to 0-1
        if concentration > 1.0:
            concentration = concentration / 100.0

        if concentration < 0 or concentration > 1.0:
            logger.debug('Invalid concentration ratio: %.4f', concentration)
            return None

        # ── Determine regime ────────────────────────────────────
        concentration_center = _safe_float(row.get('oi_concentration_center'))

        if concentration < self.WF_LOW_CONCENTRATION:
            # Dispersed / trending regime — report as regime modifier only
            regime = 'DISPERSED'
            direction = None
            strength = REGIME_ONLY_STRENGTH

            reason_parts = [
                'OI_CONCENTRATION (REGIME: DISPERSED)',
                f'Price={price:.2f}',
                f'Concentration={concentration:.1%}',
                'Momentum strategies preferred',
            ]

            self._last_fire_date = trade_date

            logger.info(
                '%s regime: DISPERSED %s conc=%.1f%%',
                self.SIGNAL_ID, trade_date, concentration * 100,
            )

            return {
                'signal_id': self.SIGNAL_ID,
                'direction': direction,
                'strength': round(strength, 4),
                'price': round(price, 2),
                'reason': ' | '.join(reason_parts),
                'metadata': {
                    'oi_concentration_ratio': round(concentration, 4),
                    'regime': regime,
                    'concentration_center': round(concentration_center, 2)
                        if not math.isnan(concentration_center) else None,
                },
            }

        if concentration < self.WF_HIGH_CONCENTRATION:
            # Neutral zone — no signal
            logger.debug('Concentration %.1f%% in neutral zone', concentration * 100)
            return None

        # ── High concentration: pinning regime ──────────────────
        regime = 'PINNING'

        # Need concentration center for directional signal
        if math.isnan(concentration_center) or concentration_center <= 0:
            # No center available — report regime without direction
            direction = None
            strength = self.WF_BASE_STRENGTH

            reason_parts = [
                'OI_CONCENTRATION (REGIME: PINNING)',
                f'Price={price:.2f}',
                f'Concentration={concentration:.1%}',
                'Mean reversion expected but center unknown',
            ]

            self._last_fire_date = trade_date

            return {
                'signal_id': self.SIGNAL_ID,
                'direction': direction,
                'strength': round(strength, 4),
                'price': round(price, 2),
                'reason': ' | '.join(reason_parts),
                'metadata': {
                    'oi_concentration_ratio': round(concentration, 4),
                    'regime': regime,
                    'concentration_center': None,
                },
            }

        # ── Directional signal based on distance from center ────
        distance_pct = ((price - concentration_center) / concentration_center) * 100.0

        if abs(distance_pct) < self.WF_MIN_DISTANCE_PCT:
            logger.debug('Price within %.1f%% of center — too close', distance_pct)
            return None

        if abs(distance_pct) > self.WF_MAX_DISTANCE_PCT:
            logger.debug('Distance %.1f%% exceeds sanity cap', distance_pct)
            return None

        if distance_pct > 0:
            direction = 'SHORT'   # Above concentration zone -> mean revert down
        else:
            direction = 'LONG'    # Below concentration zone -> mean revert up

        # ── Strength ────────────────────────────────────────────
        strength = self.WF_BASE_STRENGTH

        # Concentration level bonus
        if concentration >= self.WF_EXTREME_CONCENTRATION:
            strength += CONCENTRATION_STRENGTH_SCALE

        # Distance bonus
        extra_dist = abs(distance_pct) - self.WF_MIN_DISTANCE_PCT
        strength += extra_dist * DISTANCE_STRENGTH_SCALE

        strength = min(MAX_STRENGTH, max(MIN_STRENGTH, strength))

        # ── Reason ──────────────────────────────────────────────
        reason_parts = [
            'OI_CONCENTRATION (PINNING)',
            f'Price={price:.2f}',
            f'Center={concentration_center:.2f}',
            f'Distance={distance_pct:+.2f}%',
            f'Concentration={concentration:.1%}',
            f'Strength={strength:.2f}',
        ]

        self._last_fire_date = trade_date

        logger.info(
            '%s signal: %s %s conc=%.1f%% dist=%.2f%% strength=%.3f',
            self.SIGNAL_ID, direction, trade_date, concentration * 100,
            distance_pct, strength,
        )

        return {
            'signal_id': self.SIGNAL_ID,
            'direction': direction,
            'strength': round(strength, 4),
            'price': round(price, 2),
            'reason': ' | '.join(reason_parts),
            'metadata': {
                'oi_concentration_ratio': round(concentration, 4),
                'regime': regime,
                'concentration_center': round(concentration_center, 2),
                'distance_from_center_pct': round(distance_pct, 4),
            },
        }

    # ----------------------------------------------------------
    def reset(self) -> None:
        """Reset internal state for a fresh backtest run."""
        self._last_fire_date = None

    def __repr__(self) -> str:
        return f"OIConcentrationSignal(signal_id='{self.SIGNAL_ID}')"
