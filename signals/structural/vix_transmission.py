"""
VIX Transmission Overlay — US VIX overnight spike to India VIX lag.

Tracks the transmission of US VIX movements to India VIX during the
overnight session.  When US VIX spikes significantly (e.g. due to FOMC,
geopolitical shock, or macro data), India VIX typically follows with a
dampened response (empirical correlation ~0.65).  This overlay adjusts
position sizing based on whether the transmission is complete or still
pending.

Signal logic:
    us_vix_change = (us_vix_current - us_vix_prev) / us_vix_prev * 100

    Spike categories:
        EXTREME_SPIKE  : change > +25%
        MAJOR_SPIKE    : change > +15%
        MODERATE_SPIKE : change > +10%
        ELEVATED       : change > +5%
        NORMAL         : change <= +5% and change >= -10%
        VIX_CRUSH      : change < -10%

    Expected India VIX response = us_vix_change * 0.65

    transmission_complete = True if India VIX moved >= 60% of expected

    Size modifiers:
        EXTREME_SPIKE  -> 0.70  (reduce size — fear regime)
        MAJOR_SPIKE    -> 0.85
        MODERATE_SPIKE -> 0.95
        NORMAL         -> 1.00
        VIX_CRUSH      -> 1.10  (increase size — vol compression)

    Bias:
        EXTREME/MAJOR/MODERATE/ELEVATED spike -> BEARISH
        NORMAL                                -> NEUTRAL
        VIX_CRUSH                             -> BULLISH

Academic basis: Cross-market volatility transmission — US VIX leads
India VIX with 0.65 beta and ~70% same-day transmission rate on
non-holiday sessions (empirical study of VIX-India VIX dynamics).

Usage:
    from signals.structural.vix_transmission import VIXTransmissionSignal

    sig = VIXTransmissionSignal()
    result = sig.evaluate({
        'us_vix_current': 22.5,
        'us_vix_prev': 17.0,
        'india_vix_current': 16.8,
        'india_vix_prev': 14.5,
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

SIGNAL_ID = 'VIX_TRANSMISSION'

# US VIX change thresholds (percentage)
EXTREME_SPIKE_PCT = 25.0
MAJOR_SPIKE_PCT = 15.0
MODERATE_SPIKE_PCT = 10.0
ELEVATED_PCT = 5.0
VIX_CRUSH_PCT = -10.0

# Cross-market correlation
US_INDIA_VIX_CORRELATION = 0.65

# Transmission completeness threshold
TRANSMISSION_COMPLETE_RATIO = 0.60

# Size modifiers per spike category
SIZE_MODIFIERS = {
    'EXTREME_SPIKE': 0.70,
    'MAJOR_SPIKE': 0.85,
    'MODERATE_SPIKE': 0.95,
    'ELEVATED': 0.95,
    'NORMAL': 1.00,
    'VIX_CRUSH': 1.10,
}

# Bias mapping
BIAS_MAP = {
    'EXTREME_SPIKE': 'BEARISH',
    'MAJOR_SPIKE': 'BEARISH',
    'MODERATE_SPIKE': 'BEARISH',
    'ELEVATED': 'BEARISH',
    'NORMAL': 'NEUTRAL',
    'VIX_CRUSH': 'BULLISH',
}

# Confidence by category
CONFIDENCE_MAP = {
    'EXTREME_SPIKE': 0.85,
    'MAJOR_SPIKE': 0.75,
    'MODERATE_SPIKE': 0.65,
    'ELEVATED': 0.55,
    'NORMAL': 0.50,
    'VIX_CRUSH': 0.70,
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


def _classify_us_vix_change(change_pct: float) -> str:
    """Classify US VIX percentage change into a spike category."""
    if change_pct > EXTREME_SPIKE_PCT:
        return 'EXTREME_SPIKE'
    elif change_pct > MAJOR_SPIKE_PCT:
        return 'MAJOR_SPIKE'
    elif change_pct > MODERATE_SPIKE_PCT:
        return 'MODERATE_SPIKE'
    elif change_pct > ELEVATED_PCT:
        return 'ELEVATED'
    elif change_pct < VIX_CRUSH_PCT:
        return 'VIX_CRUSH'
    else:
        return 'NORMAL'


def _check_transmission_complete(
    india_vix_current: float,
    india_vix_prev: float,
    expected_india_move_pct: float,
) -> bool:
    """
    Check if India VIX has completed the expected transmission.

    Returns True if India VIX has moved >= 60% of the expected move.
    """
    if india_vix_prev <= 0 or not _is_valid(india_vix_prev):
        return False
    if abs(expected_india_move_pct) < 0.01:
        # Negligible expected move — consider transmission complete
        return True

    actual_india_change_pct = (
        (india_vix_current - india_vix_prev) / india_vix_prev * 100.0
    )

    # For spikes, both should be positive; for crush, both negative
    if expected_india_move_pct > 0:
        return actual_india_change_pct >= (
            expected_india_move_pct * TRANSMISSION_COMPLETE_RATIO
        )
    else:
        return actual_india_change_pct <= (
            expected_india_move_pct * TRANSMISSION_COMPLETE_RATIO
        )


def _build_neutral_context(reason: str = 'Missing or invalid data') -> Dict:
    """Return a neutral context dict when data is unavailable."""
    return {
        'signal_id': SIGNAL_ID,
        'signal': 'NEUTRAL',
        'us_vix_change_pct': 0.0,
        'spike_category': 'NORMAL',
        'india_vix_current': None,
        'india_vix_expected_move_pct': 0.0,
        'transmission_complete': False,
        'bias': 'NEUTRAL',
        'size_modifier': 1.0,
        'confidence': 0.0,
        'monitoring_note': reason,
    }


# ================================================================
# SIGNAL CLASS
# ================================================================

class VIXTransmissionSignal:
    """
    VIX Transmission Overlay.

    Monitors US VIX overnight changes and their transmission to India
    VIX.  Produces a sizing overlay that reduces position size during
    volatility spikes and increases it during vol crush regimes.
    """

    SIGNAL_ID = SIGNAL_ID

    def __init__(self) -> None:
        logger.info('VIXTransmissionSignal initialised')

    # ----------------------------------------------------------
    # EVALUATE
    # ----------------------------------------------------------
    def evaluate(self, market_data: dict) -> dict:
        """
        Evaluate VIX transmission overlay.

        Parameters
        ----------
        market_data : dict with keys:
            us_vix_current    : Current US VIX level.
            us_vix_prev       : Previous-session US VIX close.
            india_vix_current : Current India VIX level.
            india_vix_prev    : Previous-session India VIX close.

        Returns
        -------
        dict with signal context including bias, size_modifier, and
        transmission status.  Always returns a dict (never None).
        """
        try:
            return self._evaluate_inner(market_data)
        except Exception as e:
            logger.error(
                'VIXTransmissionSignal.evaluate error: %s', e, exc_info=True
            )
            return _build_neutral_context(f'Error: {e}')

    def _evaluate_inner(self, market_data: dict) -> dict:
        # ── Extract and validate inputs ─────────────────────────
        us_vix_current = _safe_float(market_data.get('us_vix_current'))
        us_vix_prev = _safe_float(market_data.get('us_vix_prev'))
        india_vix_current = _safe_float(market_data.get('india_vix_current'))
        india_vix_prev = _safe_float(market_data.get('india_vix_prev'))

        # Check US VIX inputs (required)
        if not _is_valid(us_vix_current) or not _is_valid(us_vix_prev):
            return _build_neutral_context('US VIX data missing or invalid')

        if us_vix_prev <= 0:
            return _build_neutral_context('US VIX prev must be positive')

        # Check India VIX inputs (required)
        if not _is_valid(india_vix_current) or not _is_valid(india_vix_prev):
            return _build_neutral_context('India VIX data missing or invalid')

        if india_vix_prev <= 0:
            return _build_neutral_context('India VIX prev must be positive')

        # ── Compute US VIX change ───────────────────────────────
        us_vix_change_pct = (
            (us_vix_current - us_vix_prev) / us_vix_prev * 100.0
        )

        # ── Classify spike category ─────────────────────────────
        spike_category = _classify_us_vix_change(us_vix_change_pct)

        # ── Expected India VIX response ─────────────────────────
        expected_india_move_pct = us_vix_change_pct * US_INDIA_VIX_CORRELATION

        # ── Check transmission completeness ─────────────────────
        transmission_complete = _check_transmission_complete(
            india_vix_current, india_vix_prev, expected_india_move_pct
        )

        # ── Size modifier ───────────────────────────────────────
        size_modifier = SIZE_MODIFIERS.get(spike_category, 1.0)

        # ── Bias ────────────────────────────────────────────────
        bias = BIAS_MAP.get(spike_category, 'NEUTRAL')

        # ── Confidence ──────────────────────────────────────────
        confidence = CONFIDENCE_MAP.get(spike_category, 0.50)

        # Adjust confidence based on transmission status
        if transmission_complete:
            # Transmission done — signal is more reliable
            confidence = min(0.95, confidence + 0.05)

        # ── Build monitoring note ───────────────────────────────
        actual_india_change_pct = (
            (india_vix_current - india_vix_prev) / india_vix_prev * 100.0
        )

        if spike_category in ('EXTREME_SPIKE', 'MAJOR_SPIKE'):
            if transmission_complete:
                monitoring_note = (
                    f'US VIX {spike_category} ({us_vix_change_pct:+.1f}%) — '
                    f'India VIX transmission complete '
                    f'({actual_india_change_pct:+.1f}% vs '
                    f'{expected_india_move_pct:+.1f}% expected). '
                    f'Risk regime active, size reduced to {size_modifier:.2f}x.'
                )
            else:
                monitoring_note = (
                    f'US VIX {spike_category} ({us_vix_change_pct:+.1f}%) — '
                    f'India VIX transmission PENDING '
                    f'({actual_india_change_pct:+.1f}% vs '
                    f'{expected_india_move_pct:+.1f}% expected). '
                    f'Caution: further India VIX spike likely.'
                )
        elif spike_category == 'VIX_CRUSH':
            monitoring_note = (
                f'US VIX crush ({us_vix_change_pct:+.1f}%) — '
                f'Vol compression regime. '
                f'India VIX moved {actual_india_change_pct:+.1f}%. '
                f'Size increased to {size_modifier:.2f}x.'
            )
        elif spike_category == 'ELEVATED':
            monitoring_note = (
                f'US VIX elevated ({us_vix_change_pct:+.1f}%) — '
                f'mild caution, size at {size_modifier:.2f}x.'
            )
        else:
            monitoring_note = (
                f'US VIX normal ({us_vix_change_pct:+.1f}%). '
                f'No overlay adjustment.'
            )

        # ── Build signal name ───────────────────────────────────
        signal_name = spike_category

        logger.info(
            '%s: %s us_change=%.1f%% india_vix=%.1f bias=%s size=%.2f',
            self.SIGNAL_ID, spike_category, us_vix_change_pct,
            india_vix_current, bias, size_modifier,
        )

        return {
            'signal_id': self.SIGNAL_ID,
            'signal': signal_name,
            'us_vix_change_pct': round(us_vix_change_pct, 2),
            'spike_category': spike_category,
            'india_vix_current': round(india_vix_current, 2),
            'india_vix_expected_move_pct': round(expected_india_move_pct, 2),
            'transmission_complete': transmission_complete,
            'bias': bias,
            'size_modifier': size_modifier,
            'confidence': round(confidence, 3),
            'monitoring_note': monitoring_note,
        }

    # ----------------------------------------------------------
    # Utility
    # ----------------------------------------------------------
    def __repr__(self) -> str:
        return f"VIXTransmissionSignal(signal_id='{self.SIGNAL_ID}')"
