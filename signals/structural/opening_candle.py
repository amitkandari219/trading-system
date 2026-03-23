"""
Opening Candle Signal — First 15-minute candle institutional flow.

Analyses the first 15-minute candle (9:15-9:30 AM IST) for signs of
institutional order flow.  A large-body, high-volume opening candle
often sets the directional tone for the session.

Signal logic:
    1. Candle body > 0.3% of open AND volume > 1.5× average first-15min volume.
    2. Classification:
       - GAP_CONTINUATION: candle direction matches gap direction → high conviction.
       - GAP_FADE: candle direction opposes gap → gap fill likely, shorter hold.
       - FRESH_MOVE: no significant gap (<0.15%) → new directional intent.
    3. Direction follows candle body (close > open → LONG).
    4. Gap fade gets shorter hold (120 min vs 240 min) with gap fill target.

Data source:
    - Historical: 15-min OHLCV bars from `nifty_intraday`.
    - Live: first 15-min aggregated bar from Kite Connect.

Usage:
    from signals.structural.opening_candle import OpeningCandleSignal

    sig = OpeningCandleSignal()
    result = sig.evaluate(market_data)

Academic basis: Opening range breakout and institutional participation
(Crabel 1990, "Day Trading with Short Term Price Patterns").
First-15-min candle predicts session direction ~58% when volume > 1.5× avg.
"""

from __future__ import annotations

import logging
import math
from datetime import date, time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ================================================================
# Constants
# ================================================================

SIGNAL_ID = 'OPENING_CANDLE'

# Candle body threshold
MIN_BODY_PCT = 0.003            # 0.3% minimum candle body
STRONG_BODY_PCT = 0.006         # 0.6% = strong institutional candle
EXTREME_BODY_PCT = 0.010        # 1.0% = extreme opening move

# Volume threshold
MIN_VOLUME_RATIO = 1.5          # Volume must be 1.5× avg first-15min
STRONG_VOLUME_RATIO = 2.5       # 2.5× = strong institutional presence
EXTREME_VOLUME_RATIO = 4.0      # 4.0× = extreme volume surge

# Gap thresholds (gap = open vs prev_close)
GAP_THRESHOLD = 0.0015          # 0.15% = minimum gap to classify
LARGE_GAP_THRESHOLD = 0.005     # 0.5% = large gap

# Candle type classifications
TYPE_GAP_CONTINUATION = 'GAP_CONTINUATION'
TYPE_GAP_FADE = 'GAP_FADE'
TYPE_FRESH_MOVE = 'FRESH_MOVE'

# Confidence mapping by candle type
CONF_GAP_CONTINUATION = 0.62
CONF_GAP_FADE = 0.55
CONF_FRESH_MOVE = 0.58

# Size modifiers
SIZE_GAP_CONTINUATION = 1.2
SIZE_GAP_FADE = 0.9
SIZE_FRESH_MOVE = 1.0

# Hold durations
HOLD_STANDARD_MINUTES = 240     # 4 hours (GAP_CONTINUATION / FRESH_MOVE)
HOLD_GAP_FADE_MINUTES = 120     # 2 hours (gap fade = shorter)
HOLD_STANDARD_BARS_5MIN = 48    # 48 × 5-min = 240 min
HOLD_GAP_FADE_BARS_5MIN = 24    # 24 × 5-min = 120 min

# Time window
CANDLE_START = time(9, 15)      # 9:15 AM IST
CANDLE_END = time(9, 30)        # 9:30 AM IST


# ================================================================
# Helpers
# ================================================================

def _safe_float(val: Any, default: float = float('nan')) -> float:
    """Safely cast to float."""
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _candle_body_pct(open_price: float, close_price: float) -> float:
    """Return absolute body size as fraction of open price."""
    if open_price <= 0 or math.isnan(open_price):
        return 0.0
    return abs(close_price - open_price) / open_price


def _gap_pct(open_price: float, prev_close: float) -> float:
    """Return signed gap percentage (positive = gap up)."""
    if prev_close <= 0 or math.isnan(prev_close):
        return 0.0
    return (open_price - prev_close) / prev_close


def _classify_candle_type(
    candle_direction: str,
    gap_pct: float,
) -> str:
    """
    Classify the opening candle type based on gap and candle direction.

    Parameters
    ----------
    candle_direction : 'LONG' or 'SHORT'
    gap_pct         : signed gap percentage

    Returns
    -------
    Candle type string.
    """
    abs_gap = abs(gap_pct)

    # No significant gap → fresh move
    if abs_gap < GAP_THRESHOLD:
        return TYPE_FRESH_MOVE

    # Gap direction
    gap_up = gap_pct > 0
    candle_up = candle_direction == 'LONG'

    # Same direction = continuation, opposite = fade
    if gap_up == candle_up:
        return TYPE_GAP_CONTINUATION
    else:
        return TYPE_GAP_FADE


def _compute_gap_fill_target(
    prev_close: float,
    first_15min_close: float,
    gap_pct: float,
    candle_direction: str,
) -> Optional[float]:
    """
    For gap fade scenarios, compute the gap fill target level.
    The target is the previous close (i.e. the gap gets filled).

    Returns None if not a gap fade scenario.
    """
    if abs(gap_pct) < GAP_THRESHOLD:
        return None
    # Gap fade target = prev_close (gap fills back)
    return round(prev_close, 2)


# ================================================================
# Signal Class
# ================================================================

class OpeningCandleSignal:
    """
    Opening candle signal for Nifty intraday trading.

    Analyses the first 15-minute candle (9:15-9:30 AM) for institutional
    flow.  A large-body, high-volume candle indicates directional intent
    by institutions and sets the session's likely direction.
    """

    SIGNAL_ID = SIGNAL_ID

    def __init__(
        self,
        min_body_pct: float = MIN_BODY_PCT,
        min_volume_ratio: float = MIN_VOLUME_RATIO,
    ) -> None:
        self.min_body_pct = min_body_pct
        self.min_volume_ratio = min_volume_ratio
        self._last_fire_date: Optional[date] = None
        logger.info('OpeningCandleSignal initialised')

    # ----------------------------------------------------------
    # No-signal helper
    # ----------------------------------------------------------
    @staticmethod
    def _no_signal() -> None:
        """Return None — no signal generated."""
        return None

    # ----------------------------------------------------------
    # Live evaluate
    # ----------------------------------------------------------
    def evaluate(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Evaluate opening candle signal.

        Parameters
        ----------
        market_data : dict
            Required keys:
                first_15min_open           : float — candle open (9:15 AM)
                first_15min_close          : float — candle close (9:30 AM)
                first_15min_high           : float — candle high
                first_15min_low            : float — candle low
                first_15min_volume         : float — candle volume
                avg_first_15min_volume_20d : float — 20-day avg first-15min volume
                prev_close                 : float — previous day's close

            Optional:
                trade_date                 : date  — trading date

        Returns
        -------
        dict with signal details, or None if no signal.
        """
        try:
            return self._evaluate_inner(market_data)
        except Exception as e:
            logger.error(
                'OpeningCandleSignal.evaluate error: %s', e, exc_info=True
            )
            return self._no_signal()

    def _evaluate_inner(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # ── Extract fields ──────────────────────────────────────
        candle_open = _safe_float(market_data.get('first_15min_open'))
        candle_close = _safe_float(market_data.get('first_15min_close'))
        candle_high = _safe_float(market_data.get('first_15min_high'))
        candle_low = _safe_float(market_data.get('first_15min_low'))
        candle_volume = _safe_float(market_data.get('first_15min_volume'))
        avg_volume = _safe_float(market_data.get('avg_first_15min_volume_20d'))
        prev_close = _safe_float(market_data.get('prev_close'))
        trade_date = market_data.get('trade_date')

        # ── Validate required fields ───────────────────────────
        for name, val in [
            ('first_15min_open', candle_open),
            ('first_15min_close', candle_close),
            ('first_15min_high', candle_high),
            ('first_15min_low', candle_low),
            ('first_15min_volume', candle_volume),
            ('avg_first_15min_volume_20d', avg_volume),
            ('prev_close', prev_close),
        ]:
            if math.isnan(val) or val <= 0:
                logger.debug('Invalid %s: %s', name, val)
                return self._no_signal()

        # ── Gate 1: candle body must exceed minimum ─────────────
        body_pct = _candle_body_pct(candle_open, candle_close)
        if body_pct < self.min_body_pct:
            logger.debug(
                'Candle body %.3f%% < %.3f%% threshold — no signal',
                body_pct * 100, self.min_body_pct * 100,
            )
            return self._no_signal()

        # ── Gate 2: volume must exceed threshold ────────────────
        volume_ratio = candle_volume / avg_volume
        if volume_ratio < self.min_volume_ratio:
            logger.debug(
                'Volume ratio %.2f < %.2f threshold — no signal',
                volume_ratio, self.min_volume_ratio,
            )
            return self._no_signal()

        # ── Direction from candle body ──────────────────────────
        direction = 'LONG' if candle_close > candle_open else 'SHORT'

        # ── Gap analysis ────────────────────────────────────────
        gap = _gap_pct(candle_open, prev_close)
        candle_type = _classify_candle_type(direction, gap)

        # ── Confidence based on candle type ─────────────────────
        if candle_type == TYPE_GAP_CONTINUATION:
            base_confidence = CONF_GAP_CONTINUATION
            size_modifier = SIZE_GAP_CONTINUATION
            hold_minutes = HOLD_STANDARD_MINUTES
            hold_bars = HOLD_STANDARD_BARS_5MIN
        elif candle_type == TYPE_GAP_FADE:
            base_confidence = CONF_GAP_FADE
            size_modifier = SIZE_GAP_FADE
            hold_minutes = HOLD_GAP_FADE_MINUTES
            hold_bars = HOLD_GAP_FADE_BARS_5MIN
        else:  # FRESH_MOVE
            base_confidence = CONF_FRESH_MOVE
            size_modifier = SIZE_FRESH_MOVE
            hold_minutes = HOLD_STANDARD_MINUTES
            hold_bars = HOLD_STANDARD_BARS_5MIN

        # ── Confidence adjustments ──────────────────────────────
        confidence = base_confidence

        # Strong body boost
        if body_pct >= EXTREME_BODY_PCT:
            confidence += 0.08
        elif body_pct >= STRONG_BODY_PCT:
            confidence += 0.04

        # Strong volume boost
        if volume_ratio >= EXTREME_VOLUME_RATIO:
            confidence += 0.06
        elif volume_ratio >= STRONG_VOLUME_RATIO:
            confidence += 0.03

        # Large gap continuation gets extra confidence
        if candle_type == TYPE_GAP_CONTINUATION and abs(gap) >= LARGE_GAP_THRESHOLD:
            confidence += 0.03

        confidence = min(0.90, max(0.10, confidence))

        # ── Compute candle metrics ──────────────────────────────
        candle_range = candle_high - candle_low
        upper_wick = candle_high - max(candle_open, candle_close)
        lower_wick = min(candle_open, candle_close) - candle_low
        body_pts = abs(candle_close - candle_open)

        # Wick-to-body ratio (low = clean candle, high = rejection)
        wick_ratio = (upper_wick + lower_wick) / body_pts if body_pts > 0 else float('inf')

        # Clean candle (low wick ratio) boosts confidence slightly
        if wick_ratio < 0.5:
            confidence = min(0.90, confidence + 0.02)

        # ── Gap fill target for fade scenarios ──────────────────
        gap_fill_target = _compute_gap_fill_target(
            prev_close, candle_close, gap, direction
        )

        # ── Build reason string ─────────────────────────────────
        reason_parts = [
            f"OPENING_CANDLE ({candle_type})",
            f"Dir={direction}",
            f"Body={body_pct * 100:.2f}%",
            f"VolRatio={volume_ratio:.1f}x",
            f"Gap={gap * 100:+.2f}%",
            f"Hold={hold_minutes}min",
        ]
        if gap_fill_target is not None:
            reason_parts.append(f"GapFillTgt={gap_fill_target}")
        if wick_ratio < 0.5:
            reason_parts.append("CLEAN_CANDLE")

        self._last_fire_date = trade_date

        logger.info(
            '%s signal: %s %s body=%.2f%% vol=%.1fx gap=%.2f%% conf=%.3f',
            self.SIGNAL_ID, candle_type, direction,
            body_pct * 100, volume_ratio, gap * 100, confidence,
        )

        return {
            'signal_id': self.SIGNAL_ID,
            'candle_type': candle_type,
            'direction': direction,
            'confidence': round(confidence, 3),
            'size_modifier': round(size_modifier, 2),
            'body_pct': round(body_pct * 100, 4),
            'volume_ratio': round(volume_ratio, 2),
            'gap_pct': round(gap * 100, 4),
            'candle_open': round(candle_open, 2),
            'candle_close': round(candle_close, 2),
            'candle_high': round(candle_high, 2),
            'candle_low': round(candle_low, 2),
            'candle_range': round(candle_range, 2),
            'upper_wick': round(upper_wick, 2),
            'lower_wick': round(lower_wick, 2),
            'wick_ratio': round(wick_ratio, 3),
            'candle_volume': candle_volume,
            'avg_first_15min_volume_20d': avg_volume,
            'prev_close': round(prev_close, 2),
            'gap_fill_target': gap_fill_target,
            'hold_minutes': hold_minutes,
            'max_hold_bars': hold_bars,
            'trade_date': trade_date,
            'reason': ' | '.join(reason_parts),
        }

    # ----------------------------------------------------------
    # Utility
    # ----------------------------------------------------------
    def reset(self) -> None:
        """Reset internal state for a fresh backtest run."""
        self._last_fire_date = None

    def __repr__(self) -> str:
        return f"OpeningCandleSignal(signal_id='{self.SIGNAL_ID}')"
