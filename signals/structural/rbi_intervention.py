"""
RBI FX Intervention Overlay — detects INR defense patterns.

Monitors USD/INR spot moves, intraday range characteristics, and RBI
reserve changes to classify whether the RBI is actively intervening in
the FX market.  Heavy intervention or defense of psychological round
numbers (83, 84, 85, 86, 87) reduces position sizing as the market
environment becomes less predictable for directional trades.

Signal logic:
    usdinr_change = usdinr_spot - usdinr_prev
    usdinr_range  = usdinr_high - usdinr_low
    range_pct     = usdinr_range / usdinr_prev * 100

    Intervention categories:
        HEAVY_INTERVENTION : touches high, reverses > 0.2% from high
        LIKELY_INTERVENTION: range > 0.3% but close near open (±0.05%)
        DEFENSE_MODE       : near round number (83/84/85/86/87) within 0.15
        NO_INTERVENTION    : none of the above

    Reserve categories (optional reserve_change_usd_bn):
        HEAVY_DEPLOY    : drop > 3B
        MODERATE_DEPLOY : drop 1-3B
        LIGHT_DEPLOY    : drop < 1B (but negative)
        ACCUMULATION    : increase (positive)
        UNKNOWN         : data unavailable

    Size modifiers:
        HEAVY_DEPLOY + DEFENSE_MODE -> 0.80
        HEAVY_DEPLOY               -> 0.85
        MODERATE_DEPLOY             -> 0.90
        LIGHT_DEPLOY                -> 0.95
        NO intervention data        -> 1.00
        ACCUMULATION                -> 1.05

Academic basis: Central bank FX intervention dampens volatility in the
immediate term but creates mean-reversion setups at key levels.  RBI
has historically defended round-number levels on USDINR with aggressive
spot and NDF market operations.

Usage:
    from signals.structural.rbi_intervention import RBIInterventionSignal

    sig = RBIInterventionSignal()
    result = sig.evaluate({
        'usdinr_spot': 85.05,
        'usdinr_prev': 84.92,
        'usdinr_high': 85.12,
        'usdinr_low': 84.88,
        'reserve_change_usd_bn': -2.5,
    })
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ================================================================
# CONSTANTS
# ================================================================

SIGNAL_ID = 'RBI_INTERVENTION'

# Round-number levels the RBI typically defends
ROUND_NUMBERS = [83.0, 84.0, 85.0, 86.0, 87.0]
ROUND_NUMBER_TOLERANCE = 0.15

# Intervention detection thresholds
HEAVY_REVERSAL_PCT = 0.20      # Reversal from high > 0.2%
LIKELY_RANGE_PCT = 0.30        # Range > 0.3%
LIKELY_CLOSE_OPEN_PCT = 0.05   # Close near open within 0.05%

# Reserve change thresholds (USD billions, negative = deploy)
HEAVY_DEPLOY_THRESHOLD = -3.0
MODERATE_DEPLOY_THRESHOLD = -1.0

# Size modifiers
SIZE_MOD_HEAVY_DEPLOY_DEFENSE = 0.80
SIZE_MOD_HEAVY_DEPLOY = 0.85
SIZE_MOD_MODERATE_DEPLOY = 0.90
SIZE_MOD_LIGHT_DEPLOY = 0.95
SIZE_MOD_NONE = 1.00
SIZE_MOD_ACCUMULATION = 1.05

# Confidence by intervention category
CONFIDENCE_MAP = {
    'HEAVY_INTERVENTION': 0.80,
    'LIKELY_INTERVENTION': 0.65,
    'DEFENSE_MODE': 0.70,
    'NO_INTERVENTION': 0.50,
}


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


def _is_valid(val: float) -> bool:
    """Check if a float is valid (not NaN and finite)."""
    return not (math.isnan(val) or math.isinf(val))


def _at_round_number(usdinr: float) -> bool:
    """Check if USDINR is within tolerance of any defended round number."""
    if not _is_valid(usdinr):
        return False
    for level in ROUND_NUMBERS:
        if abs(usdinr - level) <= ROUND_NUMBER_TOLERANCE:
            return True
    return False


def _nearest_round_number(usdinr: float) -> Optional[float]:
    """Return the nearest defended round number, or None."""
    if not _is_valid(usdinr):
        return None
    nearest = min(ROUND_NUMBERS, key=lambda x: abs(usdinr - x))
    if abs(usdinr - nearest) <= ROUND_NUMBER_TOLERANCE:
        return nearest
    return None


def _classify_intervention(
    usdinr_spot: float,
    usdinr_prev: float,
    usdinr_high: float,
    usdinr_low: float,
) -> str:
    """
    Classify intervention type based on intraday USDINR price action.
    """
    if usdinr_prev <= 0:
        return 'NO_INTERVENTION'

    range_val = usdinr_high - usdinr_low
    range_pct = range_val / usdinr_prev * 100.0

    # Check heavy intervention: touched high but reversed significantly
    if usdinr_high > 0:
        reversal_from_high = (usdinr_high - usdinr_spot) / usdinr_prev * 100.0
        if reversal_from_high > HEAVY_REVERSAL_PCT:
            return 'HEAVY_INTERVENTION'

    # Check likely intervention: wide range but close near open
    close_vs_open_pct = abs(usdinr_spot - usdinr_prev) / usdinr_prev * 100.0
    if range_pct > LIKELY_RANGE_PCT and close_vs_open_pct <= LIKELY_CLOSE_OPEN_PCT:
        return 'LIKELY_INTERVENTION'

    # Check defense mode: near round number
    if _at_round_number(usdinr_spot):
        return 'DEFENSE_MODE'

    return 'NO_INTERVENTION'


def _classify_reserves(reserve_change_usd_bn: float) -> str:
    """Classify RBI reserve change category."""
    if not _is_valid(reserve_change_usd_bn):
        return 'UNKNOWN'

    if reserve_change_usd_bn <= HEAVY_DEPLOY_THRESHOLD:
        return 'HEAVY_DEPLOY'
    elif reserve_change_usd_bn <= MODERATE_DEPLOY_THRESHOLD:
        return 'MODERATE_DEPLOY'
    elif reserve_change_usd_bn < 0:
        return 'LIGHT_DEPLOY'
    elif reserve_change_usd_bn >= 0:
        return 'ACCUMULATION'
    return 'UNKNOWN'


def _compute_size_modifier(
    intervention_category: str,
    reserve_category: str,
) -> float:
    """Compute size modifier from intervention and reserve categories."""
    # Special combo: heavy deploy + defense mode
    if (reserve_category == 'HEAVY_DEPLOY'
            and intervention_category == 'DEFENSE_MODE'):
        return SIZE_MOD_HEAVY_DEPLOY_DEFENSE

    # Reserve-based modifiers
    if reserve_category == 'HEAVY_DEPLOY':
        return SIZE_MOD_HEAVY_DEPLOY
    elif reserve_category == 'MODERATE_DEPLOY':
        return SIZE_MOD_MODERATE_DEPLOY
    elif reserve_category == 'LIGHT_DEPLOY':
        return SIZE_MOD_LIGHT_DEPLOY
    elif reserve_category == 'ACCUMULATION':
        return SIZE_MOD_ACCUMULATION

    # No reserve data — use intervention category for sizing
    if intervention_category == 'HEAVY_INTERVENTION':
        return SIZE_MOD_HEAVY_DEPLOY
    elif intervention_category == 'LIKELY_INTERVENTION':
        return SIZE_MOD_MODERATE_DEPLOY
    elif intervention_category == 'DEFENSE_MODE':
        return SIZE_MOD_LIGHT_DEPLOY

    return SIZE_MOD_NONE


def _build_neutral_context(reason: str = 'Missing or invalid data') -> Dict:
    """Return a neutral context dict when data is unavailable."""
    return {
        'signal_id': SIGNAL_ID,
        'signal': 'NO_INTERVENTION',
        'intervention_category': 'NO_INTERVENTION',
        'reserve_category': 'UNKNOWN',
        'usdinr_spot': None,
        'usdinr_change_pct': 0.0,
        'at_round_number': False,
        'nearest_round_number': None,
        'bias': 'NEUTRAL',
        'size_modifier': 1.0,
        'confidence': 0.0,
        'monitoring_note': reason,
    }


# ================================================================
# SIGNAL CLASS
# ================================================================

class RBIInterventionSignal:
    """
    RBI FX Intervention Overlay.

    Detects patterns consistent with RBI intervention in the USDINR
    market and adjusts position sizing accordingly.  Active intervention
    or reserve deployment reduces sizing; accumulation increases it.
    """

    SIGNAL_ID = SIGNAL_ID

    def __init__(self) -> None:
        logger.info('RBIInterventionSignal initialised')

    # ----------------------------------------------------------
    # EVALUATE
    # ----------------------------------------------------------
    def evaluate(self, market_data: dict) -> dict:
        """
        Evaluate RBI intervention overlay.

        Parameters
        ----------
        market_data : dict with keys:
            usdinr_spot           : Current USDINR spot rate.
            usdinr_prev           : Previous session USDINR close.
            usdinr_high           : Intraday USDINR high.
            usdinr_low            : Intraday USDINR low.
            reserve_change_usd_bn : Weekly reserve change in USD bn (optional).

        Returns
        -------
        dict with intervention classification, size_modifier, and context.
        Always returns a dict (never None).
        """
        try:
            return self._evaluate_inner(market_data)
        except Exception as e:
            logger.error(
                'RBIInterventionSignal.evaluate error: %s', e, exc_info=True
            )
            return _build_neutral_context(f'Error: {e}')

    def _evaluate_inner(self, market_data: dict) -> dict:
        # ── Extract and validate inputs ─────────────────────────
        usdinr_spot = _safe_float(market_data.get('usdinr_spot'))
        usdinr_prev = _safe_float(market_data.get('usdinr_prev'))
        usdinr_high = _safe_float(market_data.get('usdinr_high'))
        usdinr_low = _safe_float(market_data.get('usdinr_low'))
        reserve_change = _safe_float(
            market_data.get('reserve_change_usd_bn'), default=float('nan')
        )

        # Validate required fields
        if not _is_valid(usdinr_spot) or not _is_valid(usdinr_prev):
            return _build_neutral_context('USDINR spot/prev data missing')

        if usdinr_prev <= 0:
            return _build_neutral_context('USDINR prev must be positive')

        if not _is_valid(usdinr_high) or not _is_valid(usdinr_low):
            return _build_neutral_context('USDINR high/low data missing')

        # ── Compute change ──────────────────────────────────────
        usdinr_change_pct = (
            (usdinr_spot - usdinr_prev) / usdinr_prev * 100.0
        )

        # ── Classify intervention ───────────────────────────────
        intervention_category = _classify_intervention(
            usdinr_spot, usdinr_prev, usdinr_high, usdinr_low
        )

        # ── Classify reserves ───────────────────────────────────
        reserve_category = _classify_reserves(reserve_change)

        # ── Round number detection ──────────────────────────────
        at_round = _at_round_number(usdinr_spot)
        nearest_round = _nearest_round_number(usdinr_spot)

        # ── Size modifier ───────────────────────────────────────
        size_modifier = _compute_size_modifier(
            intervention_category, reserve_category
        )

        # ── Bias ────────────────────────────────────────────────
        if intervention_category in ('HEAVY_INTERVENTION', 'LIKELY_INTERVENTION'):
            # RBI defending — INR likely to strengthen (USDINR down)
            bias = 'INR_BULLISH'
        elif reserve_category == 'ACCUMULATION':
            bias = 'INR_BULLISH'
        elif intervention_category == 'DEFENSE_MODE':
            bias = 'RANGE_BOUND'
        else:
            bias = 'NEUTRAL'

        # ── Confidence ──────────────────────────────────────────
        confidence = CONFIDENCE_MAP.get(intervention_category, 0.50)

        # Boost confidence if reserve data corroborates
        if reserve_category in ('HEAVY_DEPLOY', 'MODERATE_DEPLOY'):
            if intervention_category in (
                'HEAVY_INTERVENTION', 'LIKELY_INTERVENTION', 'DEFENSE_MODE'
            ):
                confidence = min(0.95, confidence + 0.10)

        # ── Monitoring note ─────────────────────────────────────
        range_val = usdinr_high - usdinr_low
        range_pct = range_val / usdinr_prev * 100.0

        if intervention_category == 'HEAVY_INTERVENTION':
            monitoring_note = (
                f'Heavy RBI intervention detected. '
                f'USDINR touched {usdinr_high:.4f}, reversed to '
                f'{usdinr_spot:.4f} (range {range_pct:.3f}%). '
                f'Reserve: {reserve_category}. Size={size_modifier:.2f}x.'
            )
        elif intervention_category == 'LIKELY_INTERVENTION':
            monitoring_note = (
                f'Likely RBI intervention — wide range '
                f'({range_pct:.3f}%) but close near open. '
                f'Reserve: {reserve_category}. Size={size_modifier:.2f}x.'
            )
        elif intervention_category == 'DEFENSE_MODE':
            monitoring_note = (
                f'USDINR at {usdinr_spot:.4f}, near round level '
                f'{nearest_round}. Defense mode active. '
                f'Reserve: {reserve_category}. Size={size_modifier:.2f}x.'
            )
        else:
            monitoring_note = (
                f'No RBI intervention signals. '
                f'USDINR {usdinr_spot:.4f} ({usdinr_change_pct:+.3f}%). '
                f'Reserve: {reserve_category}.'
            )

        logger.info(
            '%s: %s reserve=%s usdinr=%.4f round=%s size=%.2f',
            self.SIGNAL_ID, intervention_category, reserve_category,
            usdinr_spot, at_round, size_modifier,
        )

        return {
            'signal_id': self.SIGNAL_ID,
            'signal': intervention_category,
            'intervention_category': intervention_category,
            'reserve_category': reserve_category,
            'usdinr_spot': round(usdinr_spot, 4),
            'usdinr_change_pct': round(usdinr_change_pct, 4),
            'at_round_number': at_round,
            'nearest_round_number': nearest_round,
            'bias': bias,
            'size_modifier': size_modifier,
            'confidence': round(confidence, 3),
            'monitoring_note': monitoring_note,
        }

    # ----------------------------------------------------------
    # Utility
    # ----------------------------------------------------------
    def __repr__(self) -> str:
        return f"RBIInterventionSignal(signal_id='{self.SIGNAL_ID}')"
