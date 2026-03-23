"""
Time-of-Day Sizing — intraday OVERLAY signal.

Applies well-known NSE session time-based sizing rules and optionally
blends in historical win-rate data per time bucket.

Time windows (IST) and base modifiers:
  09:15–09:30  →  0.70   Opening volatility, gap absorption — reduce size
  09:30–11:00  →  1.20   Trend establishment, primary entry window
  11:00–13:00  →  0.60   Lunch lull, liquidity drops — avoid new entries
  13:00–14:30  →  1.00   Afternoon awakening, normal sizing
  14:30–15:30  →  0.50   Closing auction zone — exit only, no new entries

Historical win-rate adjustment:
  If context provides win rates per bucket, the base modifier is blended
  with (win_rate / 0.5) to reward historically profitable windows.

This signal ALWAYS returns a size_modifier; it never returns None or
skips — every point in the session has a defined sizing rule.

Data expectations (via `context`):
  context['historical_win_rates'] : dict[str, float] (optional)
      Keys: 'OPENING', 'MORNING', 'LUNCH', 'AFTERNOON', 'CLOSING'
      Values: win rate 0.0–1.0

Safety:
  - Always returns a valid size_modifier (never None).
  - Modifiers clamped to [0.5, 1.5].

Usage:
    from signals.intraday.time_seasonality import TimeSeasonality
    ts = TimeSeasonality()
    result = ts.evaluate(trade_date, current_time, bar_data, context)
"""

import logging
from datetime import date, time
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════
# TIME WINDOWS (IST)
# ════════════════════════════════════════════════════════════════════

# Each window: (start_time, end_time, label, base_modifier, description)
TIME_WINDOWS = [
    (time(9, 15),  time(9, 30),  'OPENING',   0.70, 'Opening volatility — reduce size'),
    (time(9, 30),  time(11, 0),  'MORNING',   1.20, 'Trend establishment — primary entry'),
    (time(11, 0),  time(13, 0),  'LUNCH',     0.60, 'Lunch lull — avoid entries'),
    (time(13, 0),  time(14, 30), 'AFTERNOON', 1.00, 'Afternoon awakening — normal'),
    (time(14, 30), time(15, 30), 'CLOSING',   0.50, 'Closing zone — exit only'),
]

# Day-of-week adjustments (Monday=0 .. Friday=4)
DOW_ADJUSTMENTS = {
    0: 0.95,   # Monday — gap risk from weekend, slight reduction
    1: 1.00,   # Tuesday — normal
    2: 1.00,   # Wednesday — normal
    3: 1.05,   # Thursday — expiry day often has good setups
    4: 0.90,   # Friday — pre-weekend risk-off
}

# Win-rate blending weight: how much to trust historical win rates
# vs the base modifier. 0 = ignore win rates, 1 = fully weight them.
WIN_RATE_BLEND = 0.3

MIN_MODIFIER = 0.5
MAX_MODIFIER = 1.5


# ════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════

def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _time_in_range(t: time, start: time, end: time) -> bool:
    """Check if t is in [start, end). Handles same-day only."""
    return start <= t < end


# ════════════════════════════════════════════════════════════════════
# SIGNAL CLASS
# ════════════════════════════════════════════════════════════════════

class TimeSeasonality:
    """
    Intraday overlay that adjusts sizing based on time-of-day and
    optional historical win-rate data.
    """

    SIGNAL_ID = 'INTRADAY_TIME_SEASONALITY'

    def evaluate(
        self,
        trade_date: date,
        current_time: time,
        bar_data: Dict,
        context: Dict,
    ) -> Dict:
        """
        Evaluate time-of-day sizing.

        Always returns a valid result — never None.

        Returns:
            dict with signal_id, direction, confidence, size_modifier,
            reason, window_label, base_modifier, dow_modifier
        """
        # ── 1. Find current time window ─────────────────────────
        window_label, base_modifier, description = self._get_window(current_time)

        # ── 2. Day-of-week adjustment ───────────────────────────
        dow = trade_date.weekday()  # 0=Monday
        dow_modifier = DOW_ADJUSTMENTS.get(dow, 1.0)

        # ── 3. Historical win-rate blending ─────────────────────
        win_rate_modifier = 1.0
        historical = context.get('historical_win_rates', {})
        if isinstance(historical, dict) and window_label in historical:
            wr = historical[window_label]
            if isinstance(wr, (int, float)) and 0.0 <= wr <= 1.0:
                # win_rate of 0.5 → neutral, 0.7 → 1.4× boost, 0.3 → 0.6×
                wr_factor = wr / 0.5 if wr > 0 else 0.5
                wr_factor = _clamp(wr_factor, 0.6, 1.5)
                win_rate_modifier = 1.0 + WIN_RATE_BLEND * (wr_factor - 1.0)

        # ── 4. Composite modifier ───────────────────────────────
        raw = base_modifier * dow_modifier * win_rate_modifier
        size_modifier = _clamp(raw, MIN_MODIFIER, MAX_MODIFIER)

        # ── 5. Direction hint ───────────────────────────────────
        # Time-of-day doesn't dictate direction, only sizing.
        # But we flag CLOSING as exit-only guidance.
        direction = None
        if window_label == 'CLOSING':
            direction = 'EXIT_ONLY'

        # ── 6. Confidence ───────────────────────────────────────
        # High confidence because time rules are deterministic.
        confidence = 0.9 if window_label != 'AFTERNOON' else 0.6

        # ── 7. Reason ──────────────────────────────────────────
        dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        dow_name = dow_names[dow] if dow < len(dow_names) else '?'
        reason_parts = [
            description,
            f'{dow_name} adj {dow_modifier:.2f}',
        ]
        if win_rate_modifier != 1.0:
            reason_parts.append(f'Win-rate adj {win_rate_modifier:.2f}')
        reason = '; '.join(reason_parts)

        return {
            'signal_id': self.SIGNAL_ID,
            'direction': direction,
            'confidence': round(confidence, 3),
            'size_modifier': round(size_modifier, 3),
            'reason': reason,
            'window_label': window_label,
            'base_modifier': base_modifier,
            'dow_modifier': dow_modifier,
        }

    # ----------------------------------------------------------------
    # window lookup
    # ----------------------------------------------------------------

    @staticmethod
    def _get_window(current_time: time) -> Tuple[str, float, str]:
        """
        Find which session window the current time falls in.
        Returns (label, base_modifier, description).
        """
        for start, end, label, modifier, desc in TIME_WINDOWS:
            if _time_in_range(current_time, start, end):
                return label, modifier, desc

        # Pre-market or after-hours → minimal sizing
        if current_time < time(9, 15):
            return 'PRE_MARKET', 0.5, 'Pre-market — no entries'
        if current_time >= time(15, 30):
            return 'POST_MARKET', 0.5, 'Post-market — session closed'

        # Should not reach here, but safety fallback
        return 'UNKNOWN', 1.0, 'Unknown time window'

    # ----------------------------------------------------------------
    # utility: check if current time allows new entries
    # ----------------------------------------------------------------

    @staticmethod
    def is_entry_allowed(current_time: time) -> bool:
        """
        Convenience check: should the execution engine allow new entries?
        Returns False during CLOSING and lunch-lull windows.
        """
        if current_time >= time(14, 30):
            return False  # closing zone — exit only
        if time(11, 0) <= current_time < time(13, 0):
            return False  # lunch lull — avoid
        if current_time < time(9, 15):
            return False  # pre-market
        return True

    @staticmethod
    def is_exit_only(current_time: time) -> bool:
        """Returns True if in CLOSING window."""
        return current_time >= time(14, 30)
