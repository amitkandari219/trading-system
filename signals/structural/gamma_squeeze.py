"""
Gamma Squeeze Signal — Expiry-week gamma-driven acceleration.

Detects gamma squeeze setups on Nifty weekly expiry week (Monday/Tuesday
for Tuesday expiry).  When dealers are short options near ATM on expiry day,
their delta-hedging amplifies directional moves — a gamma squeeze.

Signal logic:
    1. Only fires when days_to_weekly_expiry <= 1 (Monday or Tuesday).
    2. Requires a meaningful morning move: > 0.5 % by 10:30 AM.
    3. Checks ATM OI concentration: ATM OI > 10 % of total OI = high gamma.
    4. Checks acceleration: next 45-min move in same direction as first 30-min.
    5. Gamma score from move size, OI concentration, acceleration, expiry-day bonus.

    Score >= 7 → STRONG, 5-6 → MODERATE, 3-4 → WEAK, below → no signal.

Data source:
    - Historical: 5-min OHLC bars + option chain OI snapshots.
    - Live: real-time spot price + ATM OI from Kite option chain.

Usage:
    from signals.structural.gamma_squeeze import GammaSqueezeSignal

    sig = GammaSqueezeSignal()
    result = sig.evaluate(market_data)

Academic basis: Dealer gamma exposure and hedging feedback loops
(Barbon & Buraschi 2021, "Gamma Fragility").
"""

from __future__ import annotations

import logging
import math
from datetime import date
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ================================================================
# Constants
# ================================================================

SIGNAL_ID = 'GAMMA_SQUEEZE'

# Expiry proximity
MAX_DAYS_TO_EXPIRY = 1          # Only fire Mon (1 day) or Tue (0 days)

# Morning move thresholds
MIN_MORNING_MOVE_PCT = 0.005    # 0.5% minimum move by 10:30 AM
MORNING_MOVE_LARGE = 0.010      # 1.0% = large move
MORNING_MOVE_EXTREME = 0.015    # 1.5% = extreme move

# ATM OI concentration thresholds
ATM_OI_HIGH = 0.10              # 10% of total OI at ATM = high gamma
ATM_OI_EXTREME = 0.20           # 20% = extreme concentration

# Acceleration: next 45-min move must be same direction
MIN_ACCELERATION_PCT = 0.001    # At least 0.1% continuation

# Score thresholds
SCORE_STRONG = 7
SCORE_MODERATE = 5
SCORE_WEAK = 3

# Confidence mapping
CONF_STRONG = 0.62
CONF_MODERATE = 0.52
CONF_WEAK = 0.42

# Size modifiers
SIZE_STRONG = 1.2
SIZE_MODERATE = 1.0
SIZE_WEAK = 0.7

# Expiry day bonus
EXPIRY_DAY_BONUS = 2            # Extra score points on expiry day itself

# Hold parameters
MAX_HOLD_MINUTES = 120          # 2 hours max hold
MAX_HOLD_BARS_5MIN = 24         # 24 × 5-min = 120 min


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


def _score_morning_move(move_pct: float) -> int:
    """Score 0-3 based on absolute morning move magnitude."""
    abs_move = abs(move_pct)
    if abs_move >= MORNING_MOVE_EXTREME:
        return 3
    elif abs_move >= MORNING_MOVE_LARGE:
        return 2
    elif abs_move >= MIN_MORNING_MOVE_PCT:
        return 1
    return 0


def _score_oi_concentration(atm_oi_pct: float) -> int:
    """Score 0-2 based on ATM OI as % of total OI."""
    if atm_oi_pct >= ATM_OI_EXTREME:
        return 2
    elif atm_oi_pct >= ATM_OI_HIGH:
        return 1
    return 0


def _score_acceleration(first_30min_move: float, next_45min_move: float) -> int:
    """
    Score 0-2 based on whether the next 45-min move continues
    in the same direction as the first 30-min move.
    """
    if abs(next_45min_move) < MIN_ACCELERATION_PCT:
        return 0

    # Same direction check
    same_direction = (first_30min_move > 0 and next_45min_move > 0) or \
                     (first_30min_move < 0 and next_45min_move < 0)

    if not same_direction:
        return 0

    # Strong acceleration: next 45min move >= first 30min move magnitude
    if abs(next_45min_move) >= abs(first_30min_move):
        return 2
    return 1


# ================================================================
# Signal Class
# ================================================================

class GammaSqueezeSignal:
    """
    Gamma squeeze signal for Nifty weekly expiry.

    Detects dealer gamma exposure creating amplified moves on
    expiry day/day-before-expiry.  Requires a morning directional
    move plus high ATM OI concentration plus acceleration.
    """

    SIGNAL_ID = SIGNAL_ID

    def __init__(
        self,
        max_days_to_expiry: int = MAX_DAYS_TO_EXPIRY,
        min_morning_move_pct: float = MIN_MORNING_MOVE_PCT,
        score_strong: int = SCORE_STRONG,
        score_moderate: int = SCORE_MODERATE,
        score_weak: int = SCORE_WEAK,
    ) -> None:
        self.max_days_to_expiry = max_days_to_expiry
        self.min_morning_move_pct = min_morning_move_pct
        self.score_strong = score_strong
        self.score_moderate = score_moderate
        self.score_weak = score_weak
        self._last_fire_date: Optional[date] = None
        logger.info('GammaSqueezeSignal initialised')

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
        Evaluate gamma squeeze signal.

        Parameters
        ----------
        market_data : dict
            Required keys:
                day_of_week           : int   — 0=Mon ... 4=Fri
                days_to_weekly_expiry : int   — 0 on expiry day, 1 on day before
                day_open              : float — Nifty open at 9:15 AM
                current_price         : float — Nifty price at ~10:30 AM
                atm_oi_pct_of_total   : float — ATM OI / total OI (0-1 scale)
                first_30min_move_pct  : float — signed % move 9:15-9:45 AM
                next_45min_move_pct   : float — signed % move 9:45-10:30 AM

        Returns
        -------
        dict with signal details, or None if no signal.
        """
        try:
            return self._evaluate_inner(market_data)
        except Exception as e:
            logger.error(
                'GammaSqueezeSignal.evaluate error: %s', e, exc_info=True
            )
            return self._no_signal()

    def _evaluate_inner(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # ── Extract fields ──────────────────────────────────────
        day_of_week = _safe_int(market_data.get('day_of_week'), -1)
        days_to_expiry = _safe_int(market_data.get('days_to_weekly_expiry'), 99)
        day_open = _safe_float(market_data.get('day_open'))
        current_price = _safe_float(market_data.get('current_price'))
        atm_oi_pct = _safe_float(market_data.get('atm_oi_pct_of_total'), 0.0)
        first_30min_move = _safe_float(market_data.get('first_30min_move_pct'), 0.0)
        next_45min_move = _safe_float(market_data.get('next_45min_move_pct'), 0.0)
        trade_date = market_data.get('trade_date')

        # ── Gate 1: expiry proximity ────────────────────────────
        if days_to_expiry > self.max_days_to_expiry:
            logger.debug(
                'days_to_weekly_expiry=%d > %d — not expiry week window',
                days_to_expiry, self.max_days_to_expiry,
            )
            return self._no_signal()

        # ── Validate price data ─────────────────────────────────
        if math.isnan(day_open) or day_open <= 0:
            logger.debug('Invalid day_open: %s', day_open)
            return self._no_signal()

        if math.isnan(current_price) or current_price <= 0:
            logger.debug('Invalid current_price: %s', current_price)
            return self._no_signal()

        # ── Gate 2: morning move must exceed minimum ────────────
        morning_move_pct = (current_price - day_open) / day_open
        if abs(morning_move_pct) < self.min_morning_move_pct:
            logger.debug(
                'Morning move %.3f%% < %.3f%% threshold — no squeeze',
                abs(morning_move_pct) * 100, self.min_morning_move_pct * 100,
            )
            return self._no_signal()

        # ── Compute component scores ────────────────────────────
        move_score = _score_morning_move(morning_move_pct)
        oi_score = _score_oi_concentration(atm_oi_pct)
        accel_score = _score_acceleration(first_30min_move, next_45min_move)

        # Expiry day bonus (days_to_expiry == 0 means expiry day itself)
        expiry_bonus = EXPIRY_DAY_BONUS if days_to_expiry == 0 else 0

        # Composite score (max possible = 3 + 2 + 2 + 2 = 9)
        composite_score = move_score + oi_score + accel_score + expiry_bonus

        # ── Check thresholds ────────────────────────────────────
        if composite_score >= self.score_strong:
            strength = 'STRONG'
            confidence = CONF_STRONG
            size_modifier = SIZE_STRONG
        elif composite_score >= self.score_moderate:
            strength = 'MODERATE'
            confidence = CONF_MODERATE
            size_modifier = SIZE_MODERATE
        elif composite_score >= self.score_weak:
            strength = 'WEAK'
            confidence = CONF_WEAK
            size_modifier = SIZE_WEAK
        else:
            logger.debug(
                'Gamma score %d below threshold %d — no signal',
                composite_score, self.score_weak,
            )
            return self._no_signal()

        # ── Direction follows the morning move ──────────────────
        direction = 'LONG' if morning_move_pct > 0 else 'SHORT'

        # ── Build reason string ─────────────────────────────────
        reason_parts = [
            f"GAMMA_SQUEEZE ({strength})",
            f"Dir={direction}",
            f"Score={composite_score}",
            f"MorningMove={morning_move_pct * 100:+.2f}%",
            f"ATM_OI={atm_oi_pct * 100:.1f}%",
            f"Accel={next_45min_move * 100:+.2f}%",
            f"DTE={days_to_expiry}",
        ]
        if expiry_bonus > 0:
            reason_parts.append("EXPIRY_DAY_BONUS")

        self._last_fire_date = trade_date

        logger.info(
            '%s signal: %s %s score=%d move=%.2f%% atm_oi=%.1f%% dte=%d',
            self.SIGNAL_ID, strength, direction, composite_score,
            morning_move_pct * 100, atm_oi_pct * 100, days_to_expiry,
        )

        return {
            'signal_id': self.SIGNAL_ID,
            'direction': direction,
            'strength': strength,
            'confidence': round(confidence, 3),
            'size_modifier': round(size_modifier, 2),
            'composite_score': composite_score,
            'move_score': move_score,
            'oi_score': oi_score,
            'accel_score': accel_score,
            'expiry_bonus': expiry_bonus,
            'morning_move_pct': round(morning_move_pct * 100, 4),
            'atm_oi_pct_of_total': round(atm_oi_pct * 100, 2),
            'first_30min_move_pct': round(first_30min_move * 100, 4),
            'next_45min_move_pct': round(next_45min_move * 100, 4),
            'day_open': round(day_open, 2),
            'current_price': round(current_price, 2),
            'days_to_weekly_expiry': days_to_expiry,
            'day_of_week': day_of_week,
            'max_hold_bars': MAX_HOLD_BARS_5MIN,
            'max_hold_minutes': MAX_HOLD_MINUTES,
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
        return f"GammaSqueezeSignal(signal_id='{self.SIGNAL_ID}')"
