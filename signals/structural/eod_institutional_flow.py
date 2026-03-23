"""
EOD Institutional Flow Signal.

Detects heavy institutional buying or selling in the last 60 minutes of
the trading session (2:30-3:30 PM IST).  Large institutions (mutual funds,
FIIs) often execute bulk orders in the closing auction / last hour to
minimise market impact, creating a detectable volume + directional footprint.

Signal logic:
    1. Volume surge: last_hour_volume / avg_last_hour_volume_20d → score 0-3.
    2. Last hour dominance: last_hour_volume / morning_volume → score 0-2.
    3. Directional return: abs(last_hour_close - last_hour_open) / last_hour_open → score 0-2.
    4. Delivery %: if available and > 50 %, +1 to score (institutional = delivery).
    5. Friday multiplier: Friday signals get 1.3× size modifier and 2-day hold.

    Score >= 8 → STRONG, 6-7 → MODERATE, 4-5 → WEAK, below → no signal.
    Direction follows last-hour return direction (close > open → LONG).

Data source:
    - Historical: daily bars with last-hour volume split from `nifty_intraday`.
    - Live: real-time 5-min bars aggregated for last hour.

Usage:
    from signals.structural.eod_institutional_flow import EODInstitutionalFlowSignal

    sig = EODInstitutionalFlowSignal()
    result = sig.evaluate(market_data)

Academic basis: Institutional order flow clustering at close (Admati & Pfleiderer 1988).
End-of-day volume surges predict next-day continuation 55-60% of the time.
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

SIGNAL_ID = 'EOD_INSTITUTIONAL_FLOW'

# Volume surge thresholds (last hour vs 20-day avg last hour)
VOLUME_SURGE_EXTREME = 3.0      # 3× average → max score
VOLUME_SURGE_HIGH = 2.0         # 2× average
VOLUME_SURGE_MODERATE = 1.5     # 1.5× average

# Last hour dominance (last hour / morning volume)
DOMINANCE_HIGH = 0.50           # Last hour is 50%+ of morning volume
DOMINANCE_MODERATE = 0.30       # 30%+ of morning

# Directional return thresholds (last hour)
RETURN_STRONG = 0.005           # 0.5% move in last hour
RETURN_MODERATE = 0.003         # 0.3% move

# Delivery thresholds
DELIVERY_INSTITUTIONAL = 0.50   # 50% delivery = institutional

# Score thresholds
SCORE_STRONG = 8
SCORE_MODERATE = 6
SCORE_WEAK = 4

# Friday multiplier
FRIDAY_SIZE_MULTIPLIER = 1.3
FRIDAY_HOLD_DAYS = 2
WEEKDAY_HOLD_DAYS = 1

# Confidence mapping
CONF_STRONG = 0.65
CONF_MODERATE = 0.55
CONF_WEAK = 0.45

# Size modifiers
SIZE_STRONG = 1.2
SIZE_MODERATE = 1.0
SIZE_WEAK = 0.8

# Time window
LAST_HOUR_START = time(14, 30)  # 2:30 PM IST
LAST_HOUR_END = time(15, 30)    # 3:30 PM IST


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


def _score_volume_surge(last_hour_vol: float, avg_vol: float) -> int:
    """Score 0-3 based on volume surge ratio."""
    if avg_vol <= 0 or math.isnan(avg_vol):
        return 0
    ratio = last_hour_vol / avg_vol
    if ratio >= VOLUME_SURGE_EXTREME:
        return 3
    elif ratio >= VOLUME_SURGE_HIGH:
        return 2
    elif ratio >= VOLUME_SURGE_MODERATE:
        return 1
    return 0


def _score_dominance(last_hour_vol: float, morning_vol: float) -> int:
    """Score 0-2 based on last hour volume as fraction of morning volume."""
    if morning_vol <= 0 or math.isnan(morning_vol):
        return 0
    ratio = last_hour_vol / morning_vol
    if ratio >= DOMINANCE_HIGH:
        return 2
    elif ratio >= DOMINANCE_MODERATE:
        return 1
    return 0


def _score_directional_return(last_hour_open: float, last_hour_close: float) -> int:
    """Score 0-2 based on directional move in the last hour."""
    if last_hour_open <= 0 or math.isnan(last_hour_open):
        return 0
    ret = abs(last_hour_close - last_hour_open) / last_hour_open
    if ret >= RETURN_STRONG:
        return 2
    elif ret >= RETURN_MODERATE:
        return 1
    return 0


def _score_delivery(delivery_pct: Optional[float]) -> int:
    """Score 0-1 based on delivery percentage."""
    if delivery_pct is None:
        return 0
    dpct = _safe_float(delivery_pct, 0.0)
    if dpct >= DELIVERY_INSTITUTIONAL:
        return 1
    return 0


# ================================================================
# Signal Class
# ================================================================

class EODInstitutionalFlowSignal:
    """
    Detects heavy institutional buying/selling in the last 60 minutes
    of the trading session.  Scores multiple factors and fires when
    composite score exceeds threshold.
    """

    SIGNAL_ID = SIGNAL_ID

    def __init__(
        self,
        score_strong: int = SCORE_STRONG,
        score_moderate: int = SCORE_MODERATE,
        score_weak: int = SCORE_WEAK,
        friday_multiplier: float = FRIDAY_SIZE_MULTIPLIER,
    ) -> None:
        self.score_strong = score_strong
        self.score_moderate = score_moderate
        self.score_weak = score_weak
        self.friday_multiplier = friday_multiplier
        self._last_fire_date: Optional[date] = None
        logger.info('EODInstitutionalFlowSignal initialised')

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
        Evaluate EOD institutional flow signal.

        Parameters
        ----------
        market_data : dict
            Required keys:
                last_hour_volume        : float — volume traded 2:30-3:30 PM
                avg_last_hour_volume_20d: float — 20-day average last-hour volume
                morning_volume          : float — volume traded 9:15 AM - 2:30 PM
                last_hour_close         : float — Nifty close at 3:30 PM
                last_hour_open          : float — Nifty price at 2:30 PM
                day_of_week             : int   — 0=Mon ... 4=Fri
            Optional keys:
                delivery_pct            : float — delivery percentage (0-100 scale or 0-1)
                trade_date              : date  — trading date

        Returns
        -------
        dict with signal details, or None if no signal.
        """
        try:
            return self._evaluate_inner(market_data)
        except Exception as e:
            logger.error(
                'EODInstitutionalFlowSignal.evaluate error: %s', e, exc_info=True
            )
            return self._no_signal()

    def _evaluate_inner(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # ── Extract required fields ─────────────────────────────
        last_hour_volume = _safe_float(market_data.get('last_hour_volume'))
        avg_last_hour_volume = _safe_float(market_data.get('avg_last_hour_volume_20d'))
        morning_volume = _safe_float(market_data.get('morning_volume'))
        last_hour_close = _safe_float(market_data.get('last_hour_close'))
        last_hour_open = _safe_float(market_data.get('last_hour_open'))
        day_of_week = _safe_int(market_data.get('day_of_week'), -1)
        delivery_pct = market_data.get('delivery_pct')
        trade_date = market_data.get('trade_date')

        # ── Validate required fields ───────────────────────────
        if math.isnan(last_hour_volume) or last_hour_volume <= 0:
            logger.debug('Invalid last_hour_volume: %s', last_hour_volume)
            return self._no_signal()

        if math.isnan(avg_last_hour_volume) or avg_last_hour_volume <= 0:
            logger.debug('Invalid avg_last_hour_volume_20d: %s', avg_last_hour_volume)
            return self._no_signal()

        if math.isnan(last_hour_close) or last_hour_close <= 0:
            logger.debug('Invalid last_hour_close: %s', last_hour_close)
            return self._no_signal()

        if math.isnan(last_hour_open) or last_hour_open <= 0:
            logger.debug('Invalid last_hour_open: %s', last_hour_open)
            return self._no_signal()

        if day_of_week < 0 or day_of_week > 4:
            logger.debug('Invalid day_of_week: %s', day_of_week)
            return self._no_signal()

        # ── Normalise delivery_pct to 0-1 scale ────────────────
        if delivery_pct is not None:
            dpct = _safe_float(delivery_pct, 0.0)
            # If passed as percentage (e.g. 55.0), convert to fraction
            if dpct > 1.0:
                dpct = dpct / 100.0
            delivery_pct = dpct

        # ── Compute component scores ───────────────────────────
        vol_surge_score = _score_volume_surge(last_hour_volume, avg_last_hour_volume)
        dominance_score = _score_dominance(last_hour_volume, morning_volume)
        return_score = _score_directional_return(last_hour_open, last_hour_close)
        delivery_score = _score_delivery(delivery_pct)

        # Base composite score (max possible = 3 + 2 + 2 + 1 = 8)
        composite_score = vol_surge_score + dominance_score + return_score + delivery_score

        # ── Friday bonus: +1 to composite score ────────────────
        is_friday = (day_of_week == 4)
        if is_friday:
            composite_score += 1  # Weekend positioning adds conviction

        # ── Check thresholds ───────────────────────────────────
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
                'Score %d below threshold %d — no signal',
                composite_score, self.score_weak,
            )
            return self._no_signal()

        # ── Direction from last-hour return ─────────────────────
        last_hour_return = (last_hour_close - last_hour_open) / last_hour_open
        direction = 'LONG' if last_hour_return > 0 else 'SHORT'

        # ── Friday adjustments ──────────────────────────────────
        hold_days = WEEKDAY_HOLD_DAYS
        if is_friday:
            size_modifier = round(size_modifier * self.friday_multiplier, 2)
            hold_days = FRIDAY_HOLD_DAYS

        # ── Volume surge ratio for metadata ─────────────────────
        volume_surge_ratio = round(last_hour_volume / avg_last_hour_volume, 2)
        dominance_ratio = round(
            last_hour_volume / morning_volume, 2
        ) if morning_volume > 0 else 0.0

        # ── Build reason string ─────────────────────────────────
        reason_parts = [
            f"EOD_INSTITUTIONAL_FLOW ({strength})",
            f"Dir={direction}",
            f"Score={composite_score}",
            f"VolSurge={volume_surge_ratio}x",
            f"Dominance={dominance_ratio}",
            f"LastHrReturn={last_hour_return * 100:+.2f}%",
            f"Hold={hold_days}d",
        ]
        if delivery_pct is not None:
            reason_parts.append(f"Delivery={delivery_pct * 100:.1f}%")
        if is_friday:
            reason_parts.append("FRIDAY_BOOST")

        # ── Track fire date ─────────────────────────────────────
        self._last_fire_date = trade_date

        logger.info(
            '%s signal: %s %s score=%d vol_surge=%.2fx conf=%.3f',
            self.SIGNAL_ID, strength, direction, composite_score,
            volume_surge_ratio, confidence,
        )

        return {
            'signal_id': self.SIGNAL_ID,
            'direction': direction,
            'strength': strength,
            'confidence': round(confidence, 3),
            'size_modifier': round(size_modifier, 2),
            'composite_score': composite_score,
            'vol_surge_score': vol_surge_score,
            'dominance_score': dominance_score,
            'return_score': return_score,
            'delivery_score': delivery_score,
            'volume_surge_ratio': volume_surge_ratio,
            'dominance_ratio': dominance_ratio,
            'last_hour_return_pct': round(last_hour_return * 100, 4),
            'last_hour_volume': last_hour_volume,
            'avg_last_hour_volume_20d': avg_last_hour_volume,
            'morning_volume': morning_volume,
            'last_hour_open': round(last_hour_open, 2),
            'last_hour_close': round(last_hour_close, 2),
            'delivery_pct': round(delivery_pct * 100, 1) if delivery_pct is not None else None,
            'day_of_week': day_of_week,
            'is_friday': is_friday,
            'hold_days': hold_days,
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
        return f"EODInstitutionalFlowSignal(signal_id='{self.SIGNAL_ID}')"
