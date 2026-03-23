"""
Put Wall / Call Wall OI Shift Signal.

Tracks the migration of open-interest "walls" — the strikes where the
largest put OI and call OI accumulate.  When these walls shift
directionally, it reveals the consensus of option writers (primarily
institutional) about expected support and resistance.

Signal logic:
    put_wall_shift  = today's put_oi_max_strike  - yesterday's put_oi_max_strike
    call_wall_shift = today's call_oi_max_strike - yesterday's call_oi_max_strike

    put_wall shifted UP > 100 pts    ->  LONG   (support rising, bullish)
    call_wall shifted DOWN > 100 pts ->  SHORT  (resistance falling, bearish)

    Both conditions met simultaneously: use the larger magnitude shift.
    Neither condition met: NO TRADE.

    Strength scales with magnitude of the shift (100-500 pts range).

Columns required in df:
    close, put_oi_max_strike, call_oi_max_strike

Walk-forward parameters exposed as class constants.

Usage:
    from signals.options.oi_wall_shift import OIWallShiftSignal
    sig = OIWallShiftSignal()
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

SIGNAL_ID = 'OI_WALL_SHIFT'

# Shift thresholds (Nifty points)
MIN_SHIFT_PTS = 100              # Minimum wall shift to trigger signal
MAX_SHIFT_PTS = 1000             # Sanity cap
LARGE_SHIFT_PTS = 300            # Shift beyond this = high conviction

# Strength
BASE_STRENGTH = 0.45
SHIFT_STRENGTH_PER_100 = 0.10    # Extra strength per 100 pts beyond threshold
BOTH_WALLS_BOOST = 0.10          # Both walls moving same direction
MAX_STRENGTH = 0.90
MIN_STRENGTH = 0.10

# Lookback for yesterday's wall
LOOKBACK_ROWS = 2                # Need at least 2 rows (today + yesterday)


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

class OIWallShiftSignal:
    """
    Put wall / call wall OI shift signal.

    Tracks day-over-day migration of the maximum-OI put and call strikes.
    A rising put wall signals institutional support moving higher (bullish);
    a falling call wall signals resistance moving lower (bearish).
    """

    SIGNAL_ID = SIGNAL_ID

    # -- Walk-forward params --
    WF_MIN_SHIFT_PTS = MIN_SHIFT_PTS
    WF_MAX_SHIFT_PTS = MAX_SHIFT_PTS
    WF_LARGE_SHIFT_PTS = LARGE_SHIFT_PTS
    WF_BASE_STRENGTH = BASE_STRENGTH
    WF_SHIFT_STRENGTH_PER_100 = SHIFT_STRENGTH_PER_100
    WF_BOTH_WALLS_BOOST = BOTH_WALLS_BOOST

    def __init__(self) -> None:
        self._last_fire_date: Optional[date] = None
        logger.info('OIWallShiftSignal initialised')

    # ----------------------------------------------------------
    # evaluate
    # ----------------------------------------------------------
    def evaluate(self, df: pd.DataFrame, trade_date: date) -> Optional[Dict]:
        """
        Evaluate OI wall shift signal.

        Parameters
        ----------
        df         : DataFrame with columns [close, put_oi_max_strike,
                     call_oi_max_strike].  Must have at least 2 rows.
        trade_date : The date to evaluate.

        Returns
        -------
        dict with keys: signal_id, direction, strength, price, reason,
        metadata — or None if no trade.
        """
        try:
            return self._evaluate_inner(df, trade_date)
        except Exception as e:
            logger.error('OIWallShiftSignal.evaluate error: %s', e, exc_info=True)
            return None

    def _evaluate_inner(self, df: pd.DataFrame, trade_date: date) -> Optional[Dict]:
        if self._last_fire_date == trade_date:
            return None

        if df is None or len(df) < LOOKBACK_ROWS:
            logger.debug('Insufficient data (%s rows)', 0 if df is None else len(df))
            return None

        # ── Extract today and yesterday ─────────────────────────
        today = df.iloc[-1]
        yesterday = df.iloc[-2]

        price = _safe_float(today.get('close'))
        if math.isnan(price) or price <= 0:
            logger.debug('Invalid price')
            return None

        put_wall_today = _safe_float(today.get('put_oi_max_strike'))
        put_wall_yest = _safe_float(yesterday.get('put_oi_max_strike'))
        call_wall_today = _safe_float(today.get('call_oi_max_strike'))
        call_wall_yest = _safe_float(yesterday.get('call_oi_max_strike'))

        # ── Validate OI wall data ──────────────────────────────
        has_put_walls = not (math.isnan(put_wall_today) or math.isnan(put_wall_yest))
        has_call_walls = not (math.isnan(call_wall_today) or math.isnan(call_wall_yest))

        if not has_put_walls and not has_call_walls:
            logger.debug('Missing OI wall data — options data unavailable')
            return None

        # ── Compute shifts ─────────────────────────────────────
        put_shift = (put_wall_today - put_wall_yest) if has_put_walls else 0.0
        call_shift = (call_wall_today - call_wall_yest) if has_call_walls else 0.0

        # Determine signal
        put_bullish = has_put_walls and put_shift > self.WF_MIN_SHIFT_PTS
        call_bearish = has_call_walls and call_shift < -self.WF_MIN_SHIFT_PTS

        # Sanity cap
        if has_put_walls and abs(put_shift) > self.WF_MAX_SHIFT_PTS:
            put_bullish = False
        if has_call_walls and abs(call_shift) > self.WF_MAX_SHIFT_PTS:
            call_bearish = False

        if not put_bullish and not call_bearish:
            logger.debug('No significant wall shift (put=%.0f, call=%.0f)', put_shift, call_shift)
            return None

        # ── Resolve direction when both fire ────────────────────
        if put_bullish and call_bearish:
            # Conflicting — use larger magnitude
            if abs(put_shift) >= abs(call_shift):
                direction = 'LONG'
                primary_shift = put_shift
            else:
                direction = 'SHORT'
                primary_shift = call_shift
        elif put_bullish:
            direction = 'LONG'
            primary_shift = put_shift
        else:
            direction = 'SHORT'
            primary_shift = call_shift

        # ── Strength ────────────────────────────────────────────
        strength = self.WF_BASE_STRENGTH
        extra_shift = abs(primary_shift) - self.WF_MIN_SHIFT_PTS
        strength += (extra_shift / 100.0) * self.WF_SHIFT_STRENGTH_PER_100

        # Both walls moving in same bullish/bearish direction
        put_up = has_put_walls and put_shift > 0
        call_up = has_call_walls and call_shift > 0
        if direction == 'LONG' and put_up and call_up:
            strength += self.WF_BOTH_WALLS_BOOST
        elif direction == 'SHORT' and (has_put_walls and put_shift < 0) and (has_call_walls and call_shift < 0):
            strength += self.WF_BOTH_WALLS_BOOST

        strength = min(MAX_STRENGTH, max(MIN_STRENGTH, strength))

        # ── Reason ──────────────────────────────────────────────
        reason_parts = [
            'OI_WALL_SHIFT',
            f'Price={price:.2f}',
        ]
        if has_put_walls:
            reason_parts.append(f'PutWall={put_wall_today:.0f}(shift={put_shift:+.0f})')
        if has_call_walls:
            reason_parts.append(f'CallWall={call_wall_today:.0f}(shift={call_shift:+.0f})')
        reason_parts.append(f'Strength={strength:.2f}')

        self._last_fire_date = trade_date

        logger.info(
            '%s signal: %s %s put_shift=%.0f call_shift=%.0f strength=%.3f',
            self.SIGNAL_ID, direction, trade_date, put_shift, call_shift, strength,
        )

        return {
            'signal_id': self.SIGNAL_ID,
            'direction': direction,
            'strength': round(strength, 4),
            'price': round(price, 2),
            'reason': ' | '.join(reason_parts),
            'metadata': {
                'put_wall_today': round(put_wall_today, 2) if has_put_walls else None,
                'put_wall_yesterday': round(put_wall_yest, 2) if has_put_walls else None,
                'put_shift': round(put_shift, 2) if has_put_walls else None,
                'call_wall_today': round(call_wall_today, 2) if has_call_walls else None,
                'call_wall_yesterday': round(call_wall_yest, 2) if has_call_walls else None,
                'call_shift': round(call_shift, 2) if has_call_walls else None,
            },
        }

    # ----------------------------------------------------------
    def reset(self) -> None:
        """Reset internal state for a fresh backtest run."""
        self._last_fire_date = None

    def __repr__(self) -> str:
        return f"OIWallShiftSignal(signal_id='{self.SIGNAL_ID}')"
