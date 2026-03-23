"""
Thursday Pin Setup Signal — DUAL (scoring + overlay).

Detects support/resistance zones from next-week option writing on
Thursday/Friday/Wednesday around the weekly expiry cycle.  Post Sept 2025,
Nifty weekly options expire on TUESDAY, so the heaviest next-week OI
buildup occurs Thursday-Friday of the prior week and Wednesday of
expiry week.

Signal logic:
    1. Only fires on Thu (3), Fri (4), or Wed (2) — days when next-week
       OI positioning is most active.
    2. Find max NEW put OI buildup strike → support ("put floor").
    3. Find max NEW call OI buildup strike → resistance ("call ceiling").
    4. LONG if spot is near put floor (within 0.5%).
    5. SHORT if spot is near call ceiling (within 0.5%).
    6. size_modifier: 1.10 near support, 0.90 near resistance.
    7. Proxy: if detailed OI unavailable, use max_put_oi_strike / max_call_oi_strike.
    8. Concentration check: if floor/ceiling OI > 2× next-highest → stronger signal.

Data source:
    - NSE option chain with strike-wise OI and previous-day OI.
    - Proxy: max_put_oi_strike, max_call_oi_strike from NSE daily reports.

Usage:
    from signals.structural.thursday_pin_setup import ThursdayPinSetupSignal

    sig = ThursdayPinSetupSignal()
    result = sig.evaluate({
        'day_of_week': 3,  # Thursday
        'spot_price': 24200,
        'next_week_put_oi_by_strike': {24000: 500000, 24100: 300000},
        'next_week_call_oi_by_strike': {24500: 600000, 24400: 200000},
        'prev_next_week_put_oi_by_strike': {24000: 200000, 24100: 150000},
        'prev_next_week_call_oi_by_strike': {24500: 300000, 24400: 100000},
    })

Academic basis: Avellaneda & Lipkin (2003) — "A market-induced mechanism
for stock pinning", extended to weekly option expiry OI clustering.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ================================================================
# Constants
# ================================================================

SIGNAL_ID = 'THURSDAY_PIN_SETUP'

# Valid days of week: Wed=2, Thu=3, Fri=4
VALID_DAYS = {2, 3, 4}
DAY_NAMES = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri'}

# Proximity threshold
PROXIMITY_PCT = 0.005          # within 0.5% of floor/ceiling

# Size modifiers
SIZE_MOD_SUPPORT = 1.10        # near support → slightly bigger (bounce expected)
SIZE_MOD_RESISTANCE = 0.90     # near resistance → slightly smaller (fade expected)
SIZE_MOD_NEUTRAL = 1.00

# OI concentration bonus
CONCENTRATION_RATIO = 2.0     # max_oi > 2× next_highest → strong level

# Confidence
BASE_CONFIDENCE = 0.52
CONCENTRATION_BOOST = 0.06
THURSDAY_BOOST = 0.03          # Thursday has strongest next-week positioning
NEW_OI_BOOST = 0.04            # significant new OI buildup

# Risk management
MAX_HOLD_BARS = 60             # 60 × 5-min = 5 hours
STOP_LOSS_PCT = 0.006          # 0.6%
TARGET_PCT = 0.004             # 0.4%


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


def _safe_int(val: Any, default: int = 0) -> int:
    """Safely cast to int."""
    if val is None:
        return default
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _find_max_new_oi_strike(
    current_oi: Dict[int, int],
    prev_oi: Optional[Dict[int, int]],
) -> Optional[Tuple[int, int, int]]:
    """
    Find the strike with maximum NEW OI buildup.

    Returns (strike, new_oi, total_oi) or None.
    """
    if not current_oi:
        return None

    best_strike = None
    best_new_oi = 0
    best_total_oi = 0

    for strike, oi in current_oi.items():
        strike = _safe_int(strike)
        oi = _safe_int(oi)
        if oi <= 0:
            continue

        prev = 0
        if prev_oi and strike in prev_oi:
            prev = _safe_int(prev_oi[strike])

        new_oi = max(0, oi - prev)

        if new_oi > best_new_oi:
            best_new_oi = new_oi
            best_strike = strike
            best_total_oi = oi

    if best_strike is None:
        return None

    return best_strike, best_new_oi, best_total_oi


def _find_max_oi_strike(oi_by_strike: Dict[int, int]) -> Optional[Tuple[int, int]]:
    """Find strike with maximum total OI. Returns (strike, oi) or None."""
    if not oi_by_strike:
        return None

    best_strike = None
    best_oi = 0

    for strike, oi in oi_by_strike.items():
        strike = _safe_int(strike)
        oi = _safe_int(oi)
        if oi > best_oi:
            best_oi = oi
            best_strike = strike

    if best_strike is None:
        return None
    return best_strike, best_oi


def _check_concentration(oi_by_strike: Dict[int, int], max_strike: int) -> bool:
    """Check if max OI strike has > 2× the next highest OI."""
    if not oi_by_strike or len(oi_by_strike) < 2:
        return False

    sorted_ois = sorted(
        [_safe_int(v) for k, v in oi_by_strike.items()
         if _safe_int(k) != max_strike],
        reverse=True,
    )

    if not sorted_ois:
        return True  # only one strike → concentrated by default

    max_oi = _safe_int(oi_by_strike.get(max_strike, 0))
    second_oi = sorted_ois[0]

    return second_oi > 0 and max_oi >= second_oi * CONCENTRATION_RATIO


# ================================================================
# Signal Class
# ================================================================

class ThursdayPinSetupSignal:
    """
    Thursday pin setup — DUAL signal (scoring + overlay).

    Detects next-week support/resistance from option OI buildup on
    Thu/Fri/Wed.  Generates LONG near put floor and SHORT near call
    ceiling, with size_modifier overlay.
    """

    SIGNAL_ID = SIGNAL_ID

    def __init__(self) -> None:
        logger.info('ThursdayPinSetupSignal initialised')

    # ----------------------------------------------------------
    # Evaluate
    # ----------------------------------------------------------
    def evaluate(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Evaluate Thursday pin setup signal.

        Parameters
        ----------
        market_data : dict
            Required keys:
                day_of_week                      : int — 0=Mon ... 4=Fri
                spot_price                       : float — current Nifty spot
            Primary OI keys:
                next_week_put_oi_by_strike       : dict {strike: oi}
                next_week_call_oi_by_strike      : dict {strike: oi}
                prev_next_week_put_oi_by_strike  : dict {strike: oi} (optional)
                prev_next_week_call_oi_by_strike : dict {strike: oi} (optional)
            Proxy keys (fallback):
                max_put_oi_strike                : int
                max_call_oi_strike               : int

        Returns
        -------
        dict with signal details, or None if no signal.
        """
        try:
            return self._evaluate_inner(market_data)
        except Exception as e:
            logger.error(
                'ThursdayPinSetupSignal.evaluate error: %s', e, exc_info=True
            )
            return None

    def _evaluate_inner(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # ── Day filter ────────────────────────────────────────────
        day_of_week = _safe_int(market_data.get('day_of_week'), -1)
        if day_of_week not in VALID_DAYS:
            logger.debug(
                'THURSDAY_PIN_SETUP: day_of_week=%d not in valid days %s',
                day_of_week, VALID_DAYS,
            )
            return None

        # ── Spot price ────────────────────────────────────────────
        spot = _safe_float(market_data.get('spot_price'))
        if not (spot and spot > 0 and not math.isnan(spot)):
            logger.debug('THURSDAY_PIN_SETUP: invalid spot_price')
            return None

        # ── Find support (put floor) and resistance (call ceiling) ─
        put_oi = market_data.get('next_week_put_oi_by_strike')
        call_oi = market_data.get('next_week_call_oi_by_strike')
        prev_put_oi = market_data.get('prev_next_week_put_oi_by_strike')
        prev_call_oi = market_data.get('prev_next_week_call_oi_by_strike')

        proxy_mode = False
        put_floor = None
        call_ceiling = None
        put_floor_new_oi = 0
        call_ceiling_new_oi = 0
        put_floor_total_oi = 0
        call_ceiling_total_oi = 0

        if put_oi and call_oi:
            # Primary mode: use detailed OI data
            put_result = _find_max_new_oi_strike(put_oi, prev_put_oi)
            call_result = _find_max_new_oi_strike(call_oi, prev_call_oi)

            if put_result:
                put_floor, put_floor_new_oi, put_floor_total_oi = put_result
            if call_result:
                call_ceiling, call_ceiling_new_oi, call_ceiling_total_oi = call_result
        else:
            # Proxy mode
            proxy_mode = True
            put_floor = _safe_int(market_data.get('max_put_oi_strike'))
            call_ceiling = _safe_int(market_data.get('max_call_oi_strike'))
            if put_floor <= 0:
                put_floor = None
            if call_ceiling <= 0:
                call_ceiling = None

        if put_floor is None and call_ceiling is None:
            logger.debug('THURSDAY_PIN_SETUP: no floor or ceiling found')
            return None

        # ── Proximity checks ──────────────────────────────────────
        near_support = False
        near_resistance = False

        if put_floor is not None:
            dist_to_floor = abs(spot - put_floor) / spot
            near_support = dist_to_floor <= PROXIMITY_PCT

        if call_ceiling is not None:
            dist_to_ceiling = abs(spot - call_ceiling) / spot
            near_resistance = dist_to_ceiling <= PROXIMITY_PCT

        if not near_support and not near_resistance:
            logger.debug(
                'THURSDAY_PIN_SETUP: spot %.1f not near floor=%s or ceiling=%s',
                spot, put_floor, call_ceiling,
            )
            return None

        # ── Direction and size modifier ───────────────────────────
        if near_support and near_resistance:
            # Ambiguous — skip or take the one with more OI
            if put_floor_total_oi >= call_ceiling_total_oi:
                direction = 'LONG'
                size_modifier = SIZE_MOD_SUPPORT
            else:
                direction = 'SHORT'
                size_modifier = SIZE_MOD_RESISTANCE
        elif near_support:
            direction = 'LONG'
            size_modifier = SIZE_MOD_SUPPORT
        else:
            direction = 'SHORT'
            size_modifier = SIZE_MOD_RESISTANCE

        # ── Confidence ────────────────────────────────────────────
        confidence = BASE_CONFIDENCE

        # Thursday bonus
        if day_of_week == 3:
            confidence += THURSDAY_BOOST

        # Concentration check
        concentrated = False
        if not proxy_mode:
            if direction == 'LONG' and put_oi and put_floor:
                concentrated = _check_concentration(put_oi, put_floor)
            elif direction == 'SHORT' and call_oi and call_ceiling:
                concentrated = _check_concentration(call_oi, call_ceiling)

        if concentrated:
            confidence += CONCENTRATION_BOOST

        # New OI buildup bonus
        relevant_new_oi = put_floor_new_oi if direction == 'LONG' else call_ceiling_new_oi
        if relevant_new_oi > 0:
            confidence += NEW_OI_BOOST

        # Proxy penalty
        if proxy_mode:
            confidence -= 0.05

        confidence = min(0.85, max(0.10, confidence))

        # ── Build result ──────────────────────────────────────────
        reason_parts = [
            f"THURSDAY_PIN_SETUP",
            f"Dir={direction}",
            f"Day={DAY_NAMES.get(day_of_week, '?')}",
            f"Spot={spot:.1f}",
            f"PutFloor={put_floor}" if put_floor else "PutFloor=N/A",
            f"CallCeiling={call_ceiling}" if call_ceiling else "CallCeiling=N/A",
            f"SizeMod={size_modifier:.2f}",
        ]
        if proxy_mode:
            reason_parts.append('PROXY')
        if concentrated:
            reason_parts.append('CONCENTRATED')

        logger.info(
            '%s signal: %s spot=%.1f floor=%s ceil=%s size=%.2f conf=%.3f',
            self.SIGNAL_ID, direction, spot, put_floor, call_ceiling,
            size_modifier, confidence,
        )

        return {
            'signal_id': self.SIGNAL_ID,
            'direction': direction,
            'confidence': round(confidence, 3),
            'size_modifier': size_modifier,
            'spot_price': round(spot, 2),
            'put_floor': put_floor,
            'call_ceiling': call_ceiling,
            'put_floor_new_oi': put_floor_new_oi,
            'call_ceiling_new_oi': call_ceiling_new_oi,
            'put_floor_total_oi': put_floor_total_oi,
            'call_ceiling_total_oi': call_ceiling_total_oi,
            'near_support': near_support,
            'near_resistance': near_resistance,
            'concentrated': concentrated,
            'day_of_week': day_of_week,
            'proxy_mode': proxy_mode,
            'stop_loss_pct': STOP_LOSS_PCT,
            'target_pct': TARGET_PCT,
            'max_hold_bars': MAX_HOLD_BARS,
            'reason': ' | '.join(reason_parts),
        }

    # ----------------------------------------------------------
    # Utility
    # ----------------------------------------------------------
    def __repr__(self) -> str:
        return f"ThursdayPinSetupSignal(signal_id='{self.SIGNAL_ID}')"
