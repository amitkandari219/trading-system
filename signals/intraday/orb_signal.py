"""
Opening Range Breakout (ORB) Signal.

Computes the 15-minute opening range from the first three 5-min bars of the
session (9:15-9:30 IST) and fires a breakout signal when price closes outside
the range.

Signal logic:
    OR_high = max(high of first 3 bars)
    OR_low  = min(low  of first 3 bars)
    OR_width_pct = (OR_high - OR_low) / midpoint * 100

    Filters:
      - OR width must be 0.3%-1.2% of price  (< 0.3% = noise, > 1.2% = risk)
      - Only fire between 09:30 and 14:00 IST
      - Maximum 1 ORB signal per calendar day

    Entry:
      close > OR_high → LONG   (breakout above range)
      close < OR_low  → SHORT  (breakout below range)

    Risk:
      SL  = opposite end of the opening range
      TGT = entry ± 1.5 × range width

Academic basis: Crabel (1990), Toby — Day Trading with Short-Term Price
Patterns and Opening Range Breakout.

Usage:
    from signals.intraday.orb_signal import ORBSignal
    sig = ORBSignal()
    result = sig.evaluate(trade_date, current_time, bar_data, context)
"""

import logging
import math
from datetime import date, datetime, time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# ================================================================
# CONSTANTS
# ================================================================
OR_BAR_COUNT = 3                 # First 3 five-minute bars → 15 min
OR_START = time(9, 15)
OR_END = time(9, 30)
SIGNAL_WINDOW_START = time(9, 30)
SIGNAL_WINDOW_END = time(14, 0)

# Opening range width filters (as pct of midpoint)
OR_MIN_WIDTH_PCT = 0.3
OR_MAX_WIDTH_PCT = 1.2

# Risk-reward
SL_MULTIPLIER = 1.0              # SL at opposite end of OR
TARGET_MULTIPLIER = 1.5          # Target = 1.5× range width

# Confidence mapping
BASE_CONFIDENCE = 0.55
VOLUME_BOOST = 0.10              # Extra confidence on high-volume breakout
RETEST_BOOST = 0.08              # Extra if price retested OR boundary first


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


def _bar_time(bar: Dict) -> Optional[time]:
    ts = bar.get('timestamp')
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts.time()
    if isinstance(ts, str):
        try:
            return datetime.fromisoformat(ts).time()
        except ValueError:
            return None
    return None


# ================================================================
# SIGNAL CLASS
# ================================================================
class ORBSignal:
    """
    Opening Range Breakout signal on 5-min bars.

    Tracks the high/low of the first 15 minutes and fires once per day
    when price breaks out of that range with adequate range width.
    """

    SIGNAL_ID = 'ORB_BREAKOUT'

    def __init__(self) -> None:
        # Track whether we already fired today (reset per date)
        self._last_fire_date: Optional[date] = None
        self._or_high: Optional[float] = None
        self._or_low: Optional[float] = None
        logger.info('ORBSignal initialised')

    # ----------------------------------------------------------
    # Opening range computation
    # ----------------------------------------------------------
    def _compute_opening_range(
        self, session_bars: Sequence[Dict]
    ) -> Optional[Dict[str, float]]:
        """
        Extract OR_high and OR_low from the first 3 five-minute bars.

        Returns dict with 'or_high', 'or_low', 'or_mid', 'or_width',
        'or_width_pct', 'or_volume' or None if insufficient data.
        """
        or_bars: List[Dict] = []
        for bar in session_bars:
            t = _bar_time(bar)
            if t is None:
                continue
            if OR_START <= t < OR_END:
                or_bars.append(bar)
            if len(or_bars) >= OR_BAR_COUNT:
                break

        if len(or_bars) < OR_BAR_COUNT:
            return None

        highs = [_safe_float(b.get('high')) for b in or_bars]
        lows = [_safe_float(b.get('low')) for b in or_bars]
        volumes = [_safe_float(b.get('volume'), 0) for b in or_bars]

        if any(math.isnan(h) for h in highs) or any(math.isnan(lo) for lo in lows):
            return None

        or_high = max(highs)
        or_low = min(lows)
        or_mid = (or_high + or_low) / 2.0
        or_width = or_high - or_low

        if or_mid <= 0:
            return None

        or_width_pct = (or_width / or_mid) * 100.0

        return {
            'or_high': or_high,
            'or_low': or_low,
            'or_mid': or_mid,
            'or_width': or_width,
            'or_width_pct': or_width_pct,
            'or_volume': sum(v for v in volumes if not math.isnan(v)),
        }

    # ----------------------------------------------------------
    # Volume confirmation
    # ----------------------------------------------------------
    @staticmethod
    def _breakout_volume_strong(
        bar: Dict, or_volume: float
    ) -> bool:
        """Check if breakout bar volume exceeds average OR bar volume."""
        bar_vol = _safe_float(bar.get('volume'), 0)
        avg_or_vol = or_volume / OR_BAR_COUNT if or_volume > 0 else 0
        return bar_vol > avg_or_vol * 1.2

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
        Evaluate ORB signal on the current 5-min bar.

        Parameters
        ----------
        trade_date : date
            Current trading day.
        current_time : time
            Timestamp of the current bar.
        bar_data : dict
            Current bar with keys: open, high, low, close, volume, timestamp.
        context : dict
            Must contain 'session_bars' (list of all bars so far today).

        Returns
        -------
        dict with signal_id, direction, confidence, size_modifier,
             entry_price, stop_loss, target, reason
        or None if no signal.
        """
        try:
            return self._evaluate_inner(trade_date, current_time, bar_data, context)
        except Exception as e:
            logger.error('ORBSignal.evaluate error: %s', e, exc_info=True)
            return None

    def _evaluate_inner(
        self,
        trade_date: date,
        current_time: time,
        bar_data: Dict,
        context: Dict,
    ) -> Optional[Dict]:
        # ── Time guard ──────────────────────────────────────────
        if current_time < SIGNAL_WINDOW_START or current_time >= SIGNAL_WINDOW_END:
            return None

        # ── Max 1 fire per day ──────────────────────────────────
        if self._last_fire_date == trade_date:
            return None

        # ── Build opening range ─────────────────────────────────
        session_bars = context.get('session_bars', [])
        if not session_bars:
            return None

        orng = self._compute_opening_range(session_bars)
        if orng is None:
            return None

        or_high = orng['or_high']
        or_low = orng['or_low']
        or_width = orng['or_width']
        or_width_pct = orng['or_width_pct']

        # ── Range width filter ──────────────────────────────────
        if or_width_pct < OR_MIN_WIDTH_PCT:
            return None   # Too narrow — likely noise
        if or_width_pct > OR_MAX_WIDTH_PCT:
            return None   # Too wide — excessive risk

        # ── Breakout detection ──────────────────────────────────
        close = _safe_float(bar_data.get('close'))
        if math.isnan(close):
            return None

        direction: Optional[str] = None
        if close > or_high:
            direction = 'LONG'
        elif close < or_low:
            direction = 'SHORT'
        else:
            return None   # Still inside the range

        # ── Entry / SL / Target ─────────────────────────────────
        entry_price = close
        if direction == 'LONG':
            stop_loss = or_low
            target = entry_price + (or_width * TARGET_MULTIPLIER)
        else:
            stop_loss = or_high
            target = entry_price - (or_width * TARGET_MULTIPLIER)

        # ── Confidence ──────────────────────────────────────────
        confidence = BASE_CONFIDENCE

        # Volume boost
        if self._breakout_volume_strong(bar_data, orng['or_volume']):
            confidence += VOLUME_BOOST

        # Width bonus: mid-range widths are most reliable
        if 0.5 <= or_width_pct <= 0.9:
            confidence += 0.05

        # Early breakout (9:30-10:00) is cleaner
        if current_time < time(10, 0):
            confidence += 0.05

        confidence = min(0.90, max(0.10, confidence))

        # ── Size modifier ───────────────────────────────────────
        # Tighter ranges → smaller size, wider → larger (within filter)
        size_modifier = 0.8 + (or_width_pct - OR_MIN_WIDTH_PCT) / (
            OR_MAX_WIDTH_PCT - OR_MIN_WIDTH_PCT
        ) * 0.7  # maps [0.3, 1.2] → [0.8, 1.5]
        size_modifier = min(1.5, max(0.5, size_modifier))

        # ── Build reason ────────────────────────────────────────
        reason_parts = [
            f"ORB {direction}",
            f"OR=[{or_low:.2f}, {or_high:.2f}]",
            f"Width={or_width_pct:.2f}%",
            f"Close={close:.2f}",
        ]
        if self._breakout_volume_strong(bar_data, orng['or_volume']):
            reason_parts.append('VolConfirm')

        # ── Record fire ─────────────────────────────────────────
        self._last_fire_date = trade_date

        return {
            'signal_id': self.SIGNAL_ID,
            'direction': direction,
            'confidence': round(confidence, 3),
            'size_modifier': round(size_modifier, 2),
            'entry_price': round(entry_price, 2),
            'stop_loss': round(stop_loss, 2),
            'target': round(target, 2),
            'reason': ' | '.join(reason_parts),
        }
