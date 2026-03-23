"""
Expiry Day Scalper — intraday OVERLAY signal.

Active ONLY on weekly expiry Thursdays (and monthly expiry last Thursday).
Detects gamma acceleration, pin risk, and straddle decay to provide
directional guidance and sizing modifier.

Components:
  1. Gamma acceleration: ATR expansion in the last 2 hours signals
     increased gamma exposure → trend amplification or reversal.
  2. Pin risk: price gravitating toward a round 100-point Nifty strike.
     If within 0.2% of a round strike → fade away (mean-reversion).
  3. Straddle decay: theta collapse is imminent in the last 90 minutes.
     When VIX is calm, premium sellers benefit → size up.

Size modifier:
  Calm VIX + expiry day         → 1.3  (premium decay favours seller)
  Volatile VIX + expiry day     → 0.5  (gamma risk, reduce exposure)
  Pin detected + within 0.2%    → direction=FADE, modifier depends on VIX

Non-expiry days → returns neutral (size_modifier=1.0, direction=None).

Data expectations (via `context`):
  context['spot_price']         : float — current Nifty spot
  context['india_vix']          : float — India VIX
  context['daily_atr']          : float — daily ATR(14)
  context['bars_today']         : list[dict] — intraday bars
  context['is_weekly_expiry']   : bool (optional, auto-detected from weekday)
  context['is_monthly_expiry']  : bool (optional)
  context['atm_straddle_price'] : float (optional) — ATM straddle LTP
  context['morning_straddle']   : float (optional) — straddle at 9:30

Safety:
  - Returns neutral on non-expiry days.
  - Returns neutral when data is insufficient.
  - All modifiers clamped to [0.5, 1.5].

Usage:
    from signals.intraday.expiry_scalper import ExpiryScalper
    es = ExpiryScalper()
    result = es.evaluate(trade_date, current_time, bar_data, context)
"""

import logging
import math
from datetime import date, time, timedelta
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════════

NIFTY_PIN_INTERVAL = 100         # Round 100-point strikes for pin detection
PIN_PROXIMITY_PCT = 0.002        # 0.2% of spot → "pinned"
PIN_OUTER_PCT = 0.005            # 0.5% — outer zone, mild fade

# VIX thresholds
VIX_CALM = 14.0                  # Below this → calm, favour premium sellers
VIX_ELEVATED = 18.0              # Above this → elevated gamma risk
VIX_EXTREME = 24.0               # Extreme → hard reduce

# Gamma acceleration: compare last-2-hour ATR to session ATR
GAMMA_ACCEL_THRESHOLD = 1.5      # 1.5× session ATR → gamma expansion
GAMMA_EXTREME_THRESHOLD = 2.5    # 2.5× → extreme gamma

# Straddle decay window
DECAY_WINDOW_START = time(13, 30)  # Theta collapse accelerates after 1:30 PM
DECAY_WINDOW_HEAVY = time(14, 30)  # Last hour — maximum decay

# Session times for ATR split
SESSION_OPEN = time(9, 15)
LATE_SESSION_START = time(13, 30)

MIN_MODIFIER = 0.5
MAX_MODIFIER = 1.5
MIN_BARS_FOR_GAMMA = 6


# ════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════

def _safe_float(val: Any, default: float = 0.0) -> float:
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _nearest_round_strike(spot: float, interval: int = NIFTY_PIN_INTERVAL) -> int:
    return round(spot / interval) * interval


def _is_thursday(d: date) -> bool:
    return d.weekday() == 3


def _is_last_thursday_of_month(d: date) -> bool:
    """Check if date is the last Thursday of its month."""
    if d.weekday() != 3:
        return False
    # Next Thursday would be in next month
    next_thu = d + timedelta(days=7)
    return next_thu.month != d.month


def _bar_atr(bars: List[Dict]) -> float:
    """Compute average true range over a list of bars."""
    if not bars:
        return 0.0
    trs = []
    for i, b in enumerate(bars):
        h = _safe_float(b.get('high'))
        l = _safe_float(b.get('low'))
        tr = h - l
        if i > 0:
            prev_c = _safe_float(bars[i - 1].get('close'))
            tr = max(tr, abs(h - prev_c), abs(l - prev_c))
        trs.append(tr)
    return float(np.mean(trs)) if trs else 0.0


def _get_bar_time(bar: Dict) -> Optional[time]:
    """Extract time from bar timestamp."""
    from datetime import datetime as dt
    ts = bar.get('timestamp')
    if ts is None:
        return None
    if isinstance(ts, dt):
        return ts.time()
    if isinstance(ts, time):
        return ts
    if isinstance(ts, str):
        try:
            return dt.fromisoformat(ts).time()
        except ValueError:
            return None
    return None


# ════════════════════════════════════════════════════════════════════
# SIGNAL CLASS
# ════════════════════════════════════════════════════════════════════

class ExpiryScalper:
    """
    Expiry day overlay — gamma acceleration, pin risk, theta decay.
    Active only on Thursdays (weekly expiry).
    """

    SIGNAL_ID = 'INTRADAY_EXPIRY_SCALPER'

    def evaluate(
        self,
        trade_date: date,
        current_time: time,
        bar_data: Dict,
        context: Dict,
    ) -> Dict:
        """
        Evaluate expiry-day dynamics and return overlay result.

        Returns:
            dict with signal_id, direction, confidence, size_modifier,
            reason, is_expiry, is_monthly, nearest_strike, pin_distance_pct,
            gamma_ratio, straddle_decay_pct
        """
        # ── 0. Expiry check ─────────────────────────────────────
        is_weekly = context.get('is_weekly_expiry', _is_thursday(trade_date))
        is_monthly = context.get('is_monthly_expiry', _is_last_thursday_of_month(trade_date))

        if not is_weekly:
            return self._neutral_result(
                'Not an expiry day',
                is_expiry=False,
                is_monthly=False,
            )

        spot = _safe_float(context.get('spot_price'))
        vix = _safe_float(context.get('india_vix'))
        daily_atr = _safe_float(context.get('daily_atr'))
        bars_today: List[Dict] = context.get('bars_today', [])

        if spot <= 0:
            return self._neutral_result(
                'No spot price',
                is_expiry=True,
                is_monthly=is_monthly,
            )

        # ── 1. Pin risk detection ───────────────────────────────
        nearest_strike = _nearest_round_strike(spot)
        pin_distance = abs(spot - nearest_strike)
        pin_distance_pct = pin_distance / spot if spot > 0 else 1.0

        pin_active = pin_distance_pct <= PIN_PROXIMITY_PCT
        pin_outer = pin_distance_pct <= PIN_OUTER_PCT and not pin_active

        # Direction: fade AWAY from the pin strike
        pin_direction = None
        if pin_active:
            if spot > nearest_strike:
                pin_direction = 'SHORT'  # fade back to strike
            else:
                pin_direction = 'LONG'   # fade up to strike

        # ── 2. Gamma acceleration ───────────────────────────────
        gamma_ratio = self._compute_gamma_ratio(bars_today, current_time)

        # ── 3. VIX-based regime ─────────────────────────────────
        vix_modifier = self._vix_to_modifier(vix)

        # ── 4. Straddle decay ───────────────────────────────────
        straddle_decay_pct = self._compute_straddle_decay(context)
        decay_modifier = self._decay_to_modifier(
            straddle_decay_pct, current_time,
        )

        # ── 5. Gamma modifier ──────────────────────────────────
        gamma_modifier = self._gamma_to_modifier(gamma_ratio)

        # ── 6. Pin modifier ─────────────────────────────────────
        pin_modifier = 1.0
        if pin_active:
            pin_modifier = 1.1  # pinning is tradeable — slight boost for fade
        elif pin_outer:
            pin_modifier = 1.0

        # ── 7. Composite ────────────────────────────────────────
        raw = vix_modifier * gamma_modifier * decay_modifier * pin_modifier
        size_modifier = _clamp(raw, MIN_MODIFIER, MAX_MODIFIER)

        # ── 8. Direction ────────────────────────────────────────
        # Pin takes precedence for direction
        direction = pin_direction
        if gamma_ratio >= GAMMA_EXTREME_THRESHOLD and not pin_active:
            # Extreme gamma — no directional call, just reduce size
            direction = None

        # ── 9. Confidence ───────────────────────────────────────
        confidence = self._compute_confidence(
            pin_active, gamma_ratio, vix, straddle_decay_pct,
        )

        # ── 10. Reason ─────────────────────────────────────────
        reason_parts = []
        expiry_type = 'Monthly' if is_monthly else 'Weekly'
        reason_parts.append(f'{expiry_type} expiry')
        if pin_active:
            reason_parts.append(
                f'PIN at {nearest_strike} (dist {pin_distance_pct*100:.2f}%)'
            )
        if gamma_ratio > GAMMA_ACCEL_THRESHOLD:
            reason_parts.append(f'Gamma accel {gamma_ratio:.1f}x')
        if vix > 0:
            reason_parts.append(f'VIX {vix:.1f}')
        if straddle_decay_pct > 0:
            reason_parts.append(f'Straddle decay {straddle_decay_pct:.0%}')
        reason = '; '.join(reason_parts)

        return {
            'signal_id': self.SIGNAL_ID,
            'direction': direction,
            'confidence': round(confidence, 3),
            'size_modifier': round(size_modifier, 3),
            'reason': reason,
            'is_expiry': True,
            'is_monthly': is_monthly,
            'nearest_strike': nearest_strike,
            'pin_distance_pct': round(pin_distance_pct * 100, 3),
            'gamma_ratio': round(gamma_ratio, 3),
            'straddle_decay_pct': round(straddle_decay_pct, 3),
        }

    # ----------------------------------------------------------------
    # gamma acceleration
    # ----------------------------------------------------------------

    @staticmethod
    def _compute_gamma_ratio(
        bars: List[Dict],
        current_time: time,
    ) -> float:
        """
        Compare ATR of late-session bars (after 1:30 PM) to full-session ATR.
        Ratio > 1.5 = gamma acceleration.
        """
        if len(bars) < MIN_BARS_FOR_GAMMA:
            return 1.0

        full_atr = _bar_atr(bars)
        if full_atr <= 0:
            return 1.0

        # Split bars into late-session
        late_bars = []
        for b in bars:
            bt = _get_bar_time(b)
            if bt is not None and bt >= LATE_SESSION_START:
                late_bars.append(b)

        if len(late_bars) < 3:
            return 1.0

        late_atr = _bar_atr(late_bars)
        return late_atr / full_atr if full_atr > 0 else 1.0

    @staticmethod
    def _gamma_to_modifier(gamma_ratio: float) -> float:
        if gamma_ratio >= GAMMA_EXTREME_THRESHOLD:
            return 0.6   # extreme gamma → hard reduce
        if gamma_ratio >= GAMMA_ACCEL_THRESHOLD:
            return 0.8   # gamma expansion → reduce
        return 1.0

    # ----------------------------------------------------------------
    # VIX regime
    # ----------------------------------------------------------------

    @staticmethod
    def _vix_to_modifier(vix: float) -> float:
        if vix <= 0:
            return 1.0  # no VIX data
        if vix <= VIX_CALM:
            return 1.3   # calm VIX → premium decay favours sellers
        if vix <= VIX_ELEVATED:
            return 1.0   # normal
        if vix <= VIX_EXTREME:
            return 0.7   # elevated → reduce
        return 0.5        # extreme → hard reduce

    # ----------------------------------------------------------------
    # straddle decay
    # ----------------------------------------------------------------

    @staticmethod
    def _compute_straddle_decay(context: Dict) -> float:
        """
        Compute straddle decay % from morning vs current straddle price.
        Returns 0.0 if data is unavailable.
        """
        current = _safe_float(context.get('atm_straddle_price'))
        morning = _safe_float(context.get('morning_straddle'))
        if morning <= 0 or current <= 0:
            return 0.0
        decay = (morning - current) / morning
        return _clamp(decay, 0.0, 1.0)

    @staticmethod
    def _decay_to_modifier(decay_pct: float, current_time: time) -> float:
        """
        Straddle decay modifier — rewards premium sellers when decay is high
        and we are in the heavy decay window.
        """
        if decay_pct <= 0:
            return 1.0

        in_decay_window = current_time >= DECAY_WINDOW_START
        in_heavy_decay = current_time >= DECAY_WINDOW_HEAVY

        if in_heavy_decay and decay_pct > 0.4:
            return 1.15  # strong theta collapse — premium sellers thrive
        if in_decay_window and decay_pct > 0.3:
            return 1.10
        return 1.0

    # ----------------------------------------------------------------
    # confidence
    # ----------------------------------------------------------------

    @staticmethod
    def _compute_confidence(
        pin_active: bool,
        gamma_ratio: float,
        vix: float,
        straddle_decay: float,
    ) -> float:
        score = 0.0

        # Pin component (0–0.35)
        if pin_active:
            score += 0.35

        # VIX clarity (0–0.25) — strong signal at extremes
        if vix > 0:
            if vix <= VIX_CALM or vix >= VIX_EXTREME:
                score += 0.25
            elif vix >= VIX_ELEVATED:
                score += 0.15

        # Gamma (0–0.2)
        if gamma_ratio >= GAMMA_EXTREME_THRESHOLD:
            score += 0.2
        elif gamma_ratio >= GAMMA_ACCEL_THRESHOLD:
            score += 0.1

        # Straddle decay (0–0.2)
        if straddle_decay > 0.4:
            score += 0.2
        elif straddle_decay > 0.2:
            score += 0.1

        return _clamp(score, 0.0, 1.0)

    # ----------------------------------------------------------------
    # neutral fallback
    # ----------------------------------------------------------------

    def _neutral_result(
        self,
        reason: str = '',
        is_expiry: bool = False,
        is_monthly: bool = False,
    ) -> Dict:
        return {
            'signal_id': self.SIGNAL_ID,
            'direction': None,
            'confidence': 0.0,
            'size_modifier': 1.0,
            'reason': reason,
            'is_expiry': is_expiry,
            'is_monthly': is_monthly,
            'nearest_strike': None,
            'pin_distance_pct': 0.0,
            'gamma_ratio': 1.0,
            'straddle_decay_pct': 0.0,
        }
