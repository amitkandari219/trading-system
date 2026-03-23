"""
Pre-Market Gap (Gift Nifty) Signal.

Uses the gap between today's open and yesterday's close to generate
gap-fill or gap-follow signals in the first bar of the session.

Signal logic:
    gap_pct = (today_open - prev_close) / prev_close × 100

    Gap Fill (mean-reversion):
      -1.5% < gap_pct < -0.5% → LONG  (expect 60%+ gap fill)
       0.5% < gap_pct <  1.5% → SHORT (expect 60%+ gap fill)

    Gap Follow (momentum):
      gap_pct <= -1.5% → SHORT (large gap = momentum, not fill)
      gap_pct >=  1.5% → LONG  (large gap = momentum, not fill)

    Risk:
      SL  = gap extends 50% further from open
      TGT = 60% gap fill (for fill trades) or 60% gap extension (follow)

    Filters:
      - Only fire on the first bar: 09:15 – 09:30 IST
      - Requires prev_day_close in context
      - Absolute gap must be > 0.5%

Academic basis: Bulkowski gap statistics — 70%+ of small gaps fill intraday.

Usage:
    from signals.intraday.gift_gap_signal import GiftGapSignal
    sig = GiftGapSignal()
    result = sig.evaluate(trade_date, current_time, bar_data, context)
"""

import logging
import math
from datetime import date, time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ================================================================
# CONSTANTS
# ================================================================
SIGNAL_WINDOW_START = time(9, 15)
SIGNAL_WINDOW_END = time(9, 30)

# Gap thresholds (percent)
GAP_MIN_PCT = 0.5                # Minimum gap to act on
GAP_MOMENTUM_PCT = 1.5           # Above this → follow gap, don't fade

# Risk / reward
SL_GAP_EXTENSION = 0.50          # SL if gap extends 50% more
TGT_GAP_FILL_PCT = 0.60          # Target: 60% gap fill
TGT_GAP_FOLLOW_PCT = 0.60        # Target: 60% further extension

# Confidence
CONF_GAP_FILL = 0.58             # Gap fill has ~70% hit-rate historically
CONF_GAP_FOLLOW = 0.50           # Gap follow is lower probability, higher payoff


# ================================================================
# HELPERS
# ================================================================
def _safe_float(val: Any, default: float = float('nan')) -> float:
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


# ================================================================
# SIGNAL CLASS
# ================================================================
class GiftGapSignal:
    """
    Pre-market gap signal based on open-vs-previous-close.

    Fires at most once per day, on the first 5-min bar.
    """

    SIGNAL_ID_FILL = 'GAP_FILL'
    SIGNAL_ID_FOLLOW = 'GAP_FOLLOW'

    def __init__(self) -> None:
        self._last_fire_date: Optional[date] = None
        logger.info('GiftGapSignal initialised')

    # ----------------------------------------------------------
    # Main evaluation
    # ----------------------------------------------------------
    def evaluate(
        self,
        trade_date: date,
        current_time: time,
        bar_data: Dict,
        context: Dict,
    ) -> Optional[Dict]:
        """
        Evaluate gap signal on the first bar of the day.

        Parameters
        ----------
        trade_date : date
        current_time : time
        bar_data : dict
            First bar: open, high, low, close, volume, timestamp.
        context : dict
            Must contain 'prev_day_close' (float).

        Returns
        -------
        dict or None
        """
        try:
            return self._evaluate_inner(trade_date, current_time, bar_data, context)
        except Exception as e:
            logger.error('GiftGapSignal.evaluate error: %s', e, exc_info=True)
            return None

    def _evaluate_inner(
        self,
        trade_date: date,
        current_time: time,
        bar_data: Dict,
        context: Dict,
    ) -> Optional[Dict]:
        # ── Time guard: only first bar ──────────────────────────
        if current_time < SIGNAL_WINDOW_START or current_time >= SIGNAL_WINDOW_END:
            return None

        # ── Max 1 fire per day ──────────────────────────────────
        if self._last_fire_date == trade_date:
            return None

        # ── Get previous close ──────────────────────────────────
        prev_close = _safe_float(context.get('prev_day_close'))
        if math.isnan(prev_close) or prev_close <= 0:
            return None

        # ── Get today's open ────────────────────────────────────
        today_open = _safe_float(bar_data.get('open'))
        if math.isnan(today_open) or today_open <= 0:
            return None

        # ── Compute gap ─────────────────────────────────────────
        gap_pts = today_open - prev_close
        gap_pct = (gap_pts / prev_close) * 100.0
        abs_gap_pct = abs(gap_pct)

        if abs_gap_pct < GAP_MIN_PCT:
            return None   # Gap too small to act on

        # ── Determine signal type ───────────────────────────────
        entry_price = _safe_float(bar_data.get('close'))
        if math.isnan(entry_price):
            entry_price = today_open

        if abs_gap_pct >= GAP_MOMENTUM_PCT:
            # ── Gap Follow (momentum) ───────────────────────────
            signal_id = self.SIGNAL_ID_FOLLOW
            confidence = CONF_GAP_FOLLOW

            if gap_pct >= GAP_MOMENTUM_PCT:
                direction = 'LONG'
                # SL: gap retraces 50% toward prev close
                stop_loss = today_open - abs(gap_pts) * SL_GAP_EXTENSION
                # TGT: extends 60% further
                target = today_open + abs(gap_pts) * TGT_GAP_FOLLOW_PCT
            else:
                direction = 'SHORT'
                stop_loss = today_open + abs(gap_pts) * SL_GAP_EXTENSION
                target = today_open - abs(gap_pts) * TGT_GAP_FOLLOW_PCT

            reason_type = 'GAP_FOLLOW (momentum)'

        else:
            # ── Gap Fill (mean-reversion) ───────────────────────
            signal_id = self.SIGNAL_ID_FILL
            confidence = CONF_GAP_FILL

            if gap_pct < -GAP_MIN_PCT:
                direction = 'LONG'   # Gap down → expect fill upward
                # SL: gap extends 50% more downward
                stop_loss = today_open - abs(gap_pts) * SL_GAP_EXTENSION
                # TGT: fill 60% of the gap
                target = today_open + abs(gap_pts) * TGT_GAP_FILL_PCT
            else:
                direction = 'SHORT'  # Gap up → expect fill downward
                stop_loss = today_open + abs(gap_pts) * SL_GAP_EXTENSION
                target = today_open - abs(gap_pts) * TGT_GAP_FILL_PCT

            reason_type = 'GAP_FILL (mean-reversion)'

        # ── Size modifier ───────────────────────────────────────
        # Larger gaps → slightly larger size for follow, smaller for fill
        if signal_id == self.SIGNAL_ID_FOLLOW:
            size_modifier = min(1.5, 1.0 + (abs_gap_pct - GAP_MOMENTUM_PCT) * 0.15)
        else:
            # Gap fill: moderate size, scale inversely with gap size
            size_modifier = max(0.7, 1.1 - (abs_gap_pct - GAP_MIN_PCT) * 0.3)
        size_modifier = min(1.5, max(0.5, size_modifier))

        # ── Confidence adjustments ──────────────────────────────
        # Gap fill: small gaps fill more reliably
        if signal_id == self.SIGNAL_ID_FILL and abs_gap_pct < 0.8:
            confidence += 0.08

        # Higher volume first bar adds conviction
        first_bar_vol = _safe_float(bar_data.get('volume'), 0)
        avg_vol = _safe_float(context.get('avg_bar_volume'), 0)
        if avg_vol > 0 and first_bar_vol > avg_vol * 1.5:
            confidence += 0.05

        confidence = min(0.90, max(0.10, confidence))

        # ── Build reason ────────────────────────────────────────
        risk = abs(entry_price - stop_loss)
        reward = abs(target - entry_price)
        rr = reward / risk if risk > 0 else 0.0

        reason_parts = [
            reason_type,
            f"Gap={gap_pct:+.2f}%",
            f"PrevClose={prev_close:.2f}",
            f"Open={today_open:.2f}",
            f"R:R={rr:.1f}",
        ]

        self._last_fire_date = trade_date

        return {
            'signal_id': signal_id,
            'direction': direction,
            'confidence': round(confidence, 3),
            'size_modifier': round(size_modifier, 2),
            'entry_price': round(entry_price, 2),
            'stop_loss': round(stop_loss, 2),
            'target': round(target, 2),
            'reason': ' | '.join(reason_parts),
        }
