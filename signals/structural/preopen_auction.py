"""
Pre-Open Auction Overlay — institutional intent from 9:00-9:08 AM auction.

NSE runs a pre-open auction session from 09:00 to 09:08 AM IST where
institutional orders establish an equilibrium price.  The deviation of
this equilibrium from the previous close — especially relative to the
GIFT Nifty implied open — reveals domestic institutional intent that
may not be captured by the overnight GIFT price.

Signal logic:
    preopen_deviation_pct = (auction_equilibrium - prev_close) / prev_close * 100

    domestic_adjustment_pct = (auction_equilibrium - gift_implied_open) / prev_close * 100
        (only if gift_implied_open is available)

    Classification (based on domestic_adjustment_pct, or preopen_deviation_pct
    if GIFT data unavailable):
        STRONG_BUY  : adjustment > +0.15%
        MILD_BUY    : adjustment  +0.05% to +0.15%
        NEUTRAL     : adjustment  -0.05% to +0.05%
        MILD_SELL   : adjustment  -0.15% to -0.05%
        STRONG_SELL : adjustment < -0.15%

    Volume classification (auction_volume vs avg_auction_volume):
        HIGH   : volume > 1.5x average
        NORMAL : volume 0.7x to 1.5x average
        LOW    : volume < 0.7x average

    LOW volume overrides any classification to NEUTRAL (low conviction).

    Size modifiers:
        STRONG_BUY  + HIGH   -> 1.10
        STRONG_BUY  + NORMAL -> 1.05
        MILD_BUY    + any    -> 1.02
        NEUTRAL     + any    -> 1.00
        MILD_SELL   + any    -> 0.98
        STRONG_SELL + NORMAL -> 0.95
        STRONG_SELL + HIGH   -> 0.90
        any         + LOW    -> 1.00  (override)

Academic basis: Pre-open auction price discovery reflects institutional
order flow that diverges from GIFT Nifty pricing due to domestic fund
rebalancing, ETF creation/redemption, and DII mandate flows.

Usage:
    from signals.structural.preopen_auction import PreOpenAuctionSignal

    sig = PreOpenAuctionSignal()
    result = sig.evaluate({
        'auction_equilibrium': 22560,
        'prev_close': 22500,
        'gift_implied_open': 22530,
        'auction_volume': 1200000,
        'avg_auction_volume': 800000,
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

SIGNAL_ID = 'PREOPEN_AUCTION'

# Classification thresholds (percentage)
STRONG_BUY_PCT = 0.15
MILD_BUY_PCT = 0.05
MILD_SELL_PCT = -0.05
STRONG_SELL_PCT = -0.15

# Volume ratio thresholds
VOLUME_HIGH_RATIO = 1.5
VOLUME_LOW_RATIO = 0.7

# Size modifier table
SIZE_MOD_STRONG_BUY_HIGH = 1.10
SIZE_MOD_STRONG_BUY_NORMAL = 1.05
SIZE_MOD_MILD_BUY = 1.02
SIZE_MOD_NEUTRAL = 1.00
SIZE_MOD_MILD_SELL = 0.98
SIZE_MOD_STRONG_SELL_NORMAL = 0.95
SIZE_MOD_STRONG_SELL_HIGH = 0.90
SIZE_MOD_LOW_VOLUME_OVERRIDE = 1.00

# Confidence by classification
CONFIDENCE_MAP = {
    'STRONG_BUY': 0.75,
    'MILD_BUY': 0.60,
    'NEUTRAL': 0.50,
    'MILD_SELL': 0.60,
    'STRONG_SELL': 0.75,
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


def _classify_signal(adjustment_pct: float) -> str:
    """Classify the pre-open signal based on adjustment percentage."""
    if adjustment_pct > STRONG_BUY_PCT:
        return 'STRONG_BUY'
    elif adjustment_pct > MILD_BUY_PCT:
        return 'MILD_BUY'
    elif adjustment_pct >= MILD_SELL_PCT:
        return 'NEUTRAL'
    elif adjustment_pct >= STRONG_SELL_PCT:
        return 'MILD_SELL'
    else:
        return 'STRONG_SELL'


def _classify_volume(
    auction_volume: float,
    avg_auction_volume: float,
) -> str:
    """Classify auction volume relative to average."""
    if not _is_valid(auction_volume) or not _is_valid(avg_auction_volume):
        return 'NORMAL'  # Default when data unavailable
    if avg_auction_volume <= 0:
        return 'NORMAL'

    ratio = auction_volume / avg_auction_volume

    if ratio > VOLUME_HIGH_RATIO:
        return 'HIGH'
    elif ratio < VOLUME_LOW_RATIO:
        return 'LOW'
    else:
        return 'NORMAL'


def _compute_size_modifier(
    signal_class: str,
    volume_class: str,
) -> float:
    """Compute size modifier from signal classification and volume."""
    # LOW volume overrides everything to neutral
    if volume_class == 'LOW':
        return SIZE_MOD_LOW_VOLUME_OVERRIDE

    if signal_class == 'STRONG_BUY':
        if volume_class == 'HIGH':
            return SIZE_MOD_STRONG_BUY_HIGH
        return SIZE_MOD_STRONG_BUY_NORMAL

    elif signal_class == 'MILD_BUY':
        return SIZE_MOD_MILD_BUY

    elif signal_class == 'NEUTRAL':
        return SIZE_MOD_NEUTRAL

    elif signal_class == 'MILD_SELL':
        return SIZE_MOD_MILD_SELL

    elif signal_class == 'STRONG_SELL':
        if volume_class == 'HIGH':
            return SIZE_MOD_STRONG_SELL_HIGH
        return SIZE_MOD_STRONG_SELL_NORMAL

    return SIZE_MOD_NEUTRAL


def _signal_to_bias(signal_class: str) -> str:
    """Map signal classification to directional bias."""
    if signal_class in ('STRONG_BUY', 'MILD_BUY'):
        return 'BULLISH'
    elif signal_class in ('STRONG_SELL', 'MILD_SELL'):
        return 'BEARISH'
    return 'NEUTRAL'


def _build_neutral_context(reason: str = 'Missing or invalid data') -> Dict:
    """Return a neutral context dict when data is unavailable."""
    return {
        'signal_id': SIGNAL_ID,
        'signal': 'NEUTRAL',
        'preopen_deviation_pct': 0.0,
        'domestic_adjustment_pct': None,
        'signal_classification': 'NEUTRAL',
        'volume_classification': 'NORMAL',
        'bias': 'NEUTRAL',
        'size_modifier': 1.0,
        'confidence': 0.0,
        'monitoring_note': reason,
    }


# ================================================================
# SIGNAL CLASS
# ================================================================

class PreOpenAuctionSignal:
    """
    Pre-Open Auction Overlay.

    Analyses the NSE pre-open auction equilibrium price to detect
    domestic institutional intent.  Adjusts position sizing based on
    the strength of the signal and auction volume.
    """

    SIGNAL_ID = SIGNAL_ID

    def __init__(self) -> None:
        logger.info('PreOpenAuctionSignal initialised')

    # ----------------------------------------------------------
    # EVALUATE
    # ----------------------------------------------------------
    def evaluate(self, market_data: dict) -> dict:
        """
        Evaluate pre-open auction overlay.

        Parameters
        ----------
        market_data : dict with keys:
            auction_equilibrium  : Pre-open auction equilibrium price.
            prev_close           : Previous session close price.
            gift_implied_open    : GIFT Nifty implied open (optional).
            auction_volume       : Pre-open auction volume (optional).
            avg_auction_volume   : Average auction volume (optional).

        Returns
        -------
        dict with signal classification, size_modifier, and context.
        Always returns a dict (never None).
        """
        try:
            return self._evaluate_inner(market_data)
        except Exception as e:
            logger.error(
                'PreOpenAuctionSignal.evaluate error: %s', e, exc_info=True
            )
            return _build_neutral_context(f'Error: {e}')

    def _evaluate_inner(self, market_data: dict) -> dict:
        # ── Extract and validate required inputs ────────────────
        auction_eq = _safe_float(market_data.get('auction_equilibrium'))
        prev_close = _safe_float(market_data.get('prev_close'))

        if not _is_valid(auction_eq) or not _is_valid(prev_close):
            return _build_neutral_context(
                'Auction equilibrium or prev_close missing'
            )

        if prev_close <= 0:
            return _build_neutral_context('prev_close must be positive')

        # ── Extract optional inputs ─────────────────────────────
        gift_implied = _safe_float(market_data.get('gift_implied_open'))
        auction_volume = _safe_float(market_data.get('auction_volume'))
        avg_auction_volume = _safe_float(
            market_data.get('avg_auction_volume')
        )

        # ── Compute pre-open deviation ──────────────────────────
        preopen_deviation_pct = (
            (auction_eq - prev_close) / prev_close * 100.0
        )

        # ── Compute domestic adjustment ─────────────────────────
        domestic_adjustment_pct = None
        has_gift = _is_valid(gift_implied) and gift_implied > 0

        if has_gift:
            domestic_adjustment_pct = (
                (auction_eq - gift_implied) / prev_close * 100.0
            )

        # ── Classification ──────────────────────────────────────
        # Use domestic adjustment if GIFT data available, else use
        # raw pre-open deviation
        classification_pct = (
            domestic_adjustment_pct if has_gift else preopen_deviation_pct
        )

        signal_class = _classify_signal(classification_pct)

        # ── Volume classification ───────────────────────────────
        volume_class = _classify_volume(auction_volume, avg_auction_volume)

        # ── LOW volume override ─────────────────────────────────
        original_class = signal_class
        if volume_class == 'LOW':
            signal_class = 'NEUTRAL'

        # ── Size modifier ───────────────────────────────────────
        size_modifier = _compute_size_modifier(signal_class, volume_class)

        # ── Bias ────────────────────────────────────────────────
        bias = _signal_to_bias(signal_class)

        # ── Confidence ──────────────────────────────────────────
        confidence = CONFIDENCE_MAP.get(signal_class, 0.50)

        # Adjust confidence based on data availability
        if has_gift:
            confidence = min(0.95, confidence + 0.05)

        if volume_class == 'HIGH':
            confidence = min(0.95, confidence + 0.05)
        elif volume_class == 'LOW':
            confidence = max(0.10, confidence - 0.15)

        # ── Monitoring note ─────────────────────────────────────
        parts = [
            f'Auction={auction_eq:.2f}',
            f'PrevClose={prev_close:.2f}',
            f'Deviation={preopen_deviation_pct:+.3f}%',
        ]

        if has_gift:
            parts.append(f'GIFT={gift_implied:.2f}')
            parts.append(f'DomAdj={domestic_adjustment_pct:+.3f}%')

        parts.append(f'Vol={volume_class}')
        parts.append(f'Class={signal_class}')

        if volume_class == 'LOW' and original_class != 'NEUTRAL':
            parts.append(
                f'(overridden from {original_class} due to LOW volume)'
            )

        parts.append(f'Size={size_modifier:.2f}x')

        monitoring_note = ' | '.join(parts)

        logger.info(
            '%s: %s vol=%s dev=%.3f%% adj=%s size=%.2f',
            self.SIGNAL_ID, signal_class, volume_class,
            preopen_deviation_pct,
            f'{domestic_adjustment_pct:+.3f}%' if has_gift else 'N/A',
            size_modifier,
        )

        return {
            'signal_id': self.SIGNAL_ID,
            'signal': signal_class,
            'preopen_deviation_pct': round(preopen_deviation_pct, 4),
            'domestic_adjustment_pct': (
                round(domestic_adjustment_pct, 4)
                if domestic_adjustment_pct is not None else None
            ),
            'signal_classification': signal_class,
            'volume_classification': volume_class,
            'bias': bias,
            'size_modifier': size_modifier,
            'confidence': round(confidence, 3),
            'monitoring_note': monitoring_note,
        }

    # ----------------------------------------------------------
    # Utility
    # ----------------------------------------------------------
    def __repr__(self) -> str:
        return f"PreOpenAuctionSignal(signal_id='{self.SIGNAL_ID}')"
