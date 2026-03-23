"""
F&O Expiry Week Tuesday Effect Signal.

Post September 2025, NSE weekly F&O expiry moved from Thursday to Tuesday.
This signal exploits the predictable theta decay and gamma pinning dynamics
around Tuesday expiry.

Signal logic:
    Monday (T-1, day before expiry):
        Theta decay accelerates — premium sellers active.
        If Nifty is near a major strike (max pain), expect pinning.
        Signal: LONG on Monday close (expecting gamma-pin convergence).

    Tuesday (T+0, expiry day):
        Gamma pinning near max pain → high volume, narrow range.
        Signal: exit LONG from Monday at Tuesday close.

    Identification:
        Expiry Tuesdays: every Tuesday from Oct 2025 onwards is a weekly
        expiry (unless holiday — then previous trading day).

    Filters:
        - Only fires if close is within 1.5% of nearest 50-point strike
        - india_vix < 25 (extreme vol breaks pinning)
        - Volume confirmation (optional)

Walk-forward parameters exposed as class constants.

Usage:
    from signals.seasonality.expiry_week_tuesday import ExpiryWeekTuesdaySignal

    sig = ExpiryWeekTuesdaySignal()
    result = sig.evaluate(df, date(2026, 3, 23))  # a Monday
"""

import logging
import math
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ================================================================
# CONSTANTS / WF PARAMETERS
# ================================================================

SIGNAL_ID = 'EXPIRY_WEEK_TUESDAY'

# Policy change date: weekly expiry moved to Tuesday
TUESDAY_EXPIRY_START = date(2025, 10, 1)

# Strike proximity
STRIKE_INTERVAL = 50             # Nifty strikes are at 50-pt intervals
MAX_DISTANCE_FROM_STRIKE_PCT = 1.5  # Within 1.5% of nearest strike

# Pin / convergence
PIN_DISTANCE_PTS = 75            # Must be within 75 pts of nearest strike
STRONG_PIN_PTS = 30              # Within 30 pts → strong pin expected

# Strength
BASE_STRENGTH_MONDAY = 0.60
BASE_STRENGTH_TUESDAY = 0.55
PIN_PROXIMITY_BOOST = 0.10       # Close to strike → higher conviction
MIN_STRENGTH = 0.10
MAX_STRENGTH = 0.90

# Filters
VIX_MAX = 25.0

# Hold
HOLD_DAYS = 1                    # Enter Monday, exit Tuesday


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


def _nearest_strike(price: float, interval: int = STRIKE_INTERVAL) -> float:
    """Round price to the nearest strike (multiple of interval)."""
    return round(price / interval) * interval


def _distance_to_nearest_strike(price: float, interval: int = STRIKE_INTERVAL) -> float:
    """Absolute distance from price to nearest strike in points."""
    strike = _nearest_strike(price, interval)
    return abs(price - strike)


def _is_expiry_tuesday(td: date) -> bool:
    """Check if a date is a Tuesday expiry (post Oct 2025)."""
    if td < TUESDAY_EXPIRY_START:
        return False
    return td.weekday() == 1  # 0=Mon, 1=Tue


def _get_next_expiry_tuesday(td: date) -> Optional[date]:
    """Get the next Tuesday expiry on or after td."""
    if td < TUESDAY_EXPIRY_START:
        return None
    current = td
    for _ in range(7):
        if current.weekday() == 1:
            return current
        current += timedelta(days=1)
    return None


def _is_monday_before_expiry(td: date) -> bool:
    """Check if td is the Monday before a Tuesday expiry."""
    if td < TUESDAY_EXPIRY_START:
        return False
    if td.weekday() != 0:  # Not Monday
        return False
    # Tuesday after this Monday would be an expiry
    next_tue = td + timedelta(days=1)
    return _is_expiry_tuesday(next_tue)


# ================================================================
# SIGNAL CLASS
# ================================================================

class ExpiryWeekTuesdaySignal:
    """
    F&O Expiry Week Tuesday signal for Nifty.

    LONG on Monday close (before expiry), exit on Tuesday close.
    Exploits gamma pinning near max pain on expiry day.
    """

    SIGNAL_ID = SIGNAL_ID
    TUESDAY_EXPIRY_START = TUESDAY_EXPIRY_START

    # WF parameters
    STRIKE_INTERVAL = STRIKE_INTERVAL
    PIN_DISTANCE_PTS = PIN_DISTANCE_PTS
    VIX_MAX = VIX_MAX
    HOLD_DAYS = HOLD_DAYS
    BASE_STRENGTH_MONDAY = BASE_STRENGTH_MONDAY

    def __init__(self) -> None:
        self._last_fire_date: Optional[date] = None
        logger.info('ExpiryWeekTuesdaySignal initialised')

    # ----------------------------------------------------------
    # PUBLIC evaluate
    # ----------------------------------------------------------
    def evaluate(self, df: pd.DataFrame, trade_date: date) -> Optional[Dict]:
        """
        Evaluate Expiry Week Tuesday signal.

        Parameters
        ----------
        df         : DataFrame with columns: date, open, high, low, close,
                     india_vix (optional), max_pain (optional).
        trade_date : The date to evaluate.

        Returns
        -------
        dict with keys: signal_id, direction, strength, price, reason, metadata
        or None if no signal.
        """
        try:
            return self._evaluate_inner(df, trade_date)
        except Exception as e:
            logger.error('ExpiryWeekTuesdaySignal.evaluate error: %s', e, exc_info=True)
            return None

    def _evaluate_inner(self, df: pd.DataFrame, trade_date: date) -> Optional[Dict]:
        if df is None or df.empty:
            return None
        if 'date' not in df.columns:
            return None
        if trade_date < self.TUESDAY_EXPIRY_START:
            logger.debug('Before Tuesday expiry regime — skip')
            return None

        td = trade_date

        # Get current row
        row = df[df['date'] == pd.Timestamp(td)]
        if row.empty:
            return None
        row = row.iloc[-1]

        close = _safe_float(row.get('close'))
        if math.isnan(close) or close <= 0:
            return None

        # ── Monday before expiry → ENTRY signal ──────────────────
        if _is_monday_before_expiry(td):
            return self._monday_signal(df, row, td, close)

        # ── Tuesday expiry → EXIT signal ─────────────────────────
        if _is_expiry_tuesday(td):
            return self._tuesday_signal(df, row, td, close)

        return None

    # ----------------------------------------------------------
    # Day signals
    # ----------------------------------------------------------
    def _monday_signal(
        self, df: pd.DataFrame, row: pd.Series, td: date, close: float,
    ) -> Optional[Dict]:
        """Monday before expiry: LONG entry if near strike."""
        # Filter: VIX
        vix = _safe_float(row.get('india_vix'))
        if not math.isnan(vix) and vix >= self.VIX_MAX:
            logger.debug('VIX %.1f >= %.1f — skip', vix, self.VIX_MAX)
            return None

        # Check proximity to nearest strike
        nearest = _nearest_strike(close, self.STRIKE_INTERVAL)
        dist_pts = _distance_to_nearest_strike(close, self.STRIKE_INTERVAL)
        dist_pct = (dist_pts / close) * 100.0 if close > 0 else float('inf')

        if dist_pct > MAX_DISTANCE_FROM_STRIKE_PCT:
            logger.debug('Distance %.2f%% from strike %d — too far', dist_pct, nearest)
            return None

        # Strength
        strength = self.BASE_STRENGTH_MONDAY
        if dist_pts <= STRONG_PIN_PTS:
            strength = min(MAX_STRENGTH, strength + PIN_PROXIMITY_BOOST)

        # Check max_pain column if available
        max_pain = _safe_float(row.get('max_pain'))
        max_pain_note = ''
        if not math.isnan(max_pain) and max_pain > 0:
            mp_dist = abs(close - max_pain)
            if mp_dist < PIN_DISTANCE_PTS:
                strength = min(MAX_STRENGTH, strength + 0.05)
                max_pain_note = f'MaxPain={max_pain:.0f} dist={mp_dist:.0f}pts'

        expiry_tue = td + timedelta(days=1)

        reason = (
            f"EXPIRY_MONDAY | LONG for Tue expiry {expiry_tue} | "
            f"Near strike {nearest} (dist={dist_pts:.0f}pts/{dist_pct:.2f}%) | "
            f"{max_pain_note}"
        )

        logger.info('%s MONDAY: %s LONG strike=%d dist=%.0f strength=%.3f',
                     self.SIGNAL_ID, td, nearest, dist_pts, strength)

        return {
            'signal_id': self.SIGNAL_ID,
            'direction': 'LONG',
            'strength': round(max(MIN_STRENGTH, strength), 3),
            'price': round(close, 2),
            'reason': reason.strip(),
            'metadata': {
                'phase': 'ENTRY_MONDAY',
                'expiry_date': expiry_tue.isoformat(),
                'nearest_strike': int(nearest),
                'distance_pts': round(dist_pts, 1),
                'distance_pct': round(dist_pct, 3),
                'hold_days': self.HOLD_DAYS,
            },
        }

    def _tuesday_signal(
        self, df: pd.DataFrame, row: pd.Series, td: date, close: float,
    ) -> Optional[Dict]:
        """Tuesday expiry: EXIT signal — close the Monday LONG."""
        nearest = _nearest_strike(close, self.STRIKE_INTERVAL)
        dist_pts = _distance_to_nearest_strike(close, self.STRIKE_INTERVAL)

        strength = BASE_STRENGTH_TUESDAY

        # Check if pinning occurred (close very near strike)
        pin_occurred = dist_pts <= STRONG_PIN_PTS
        if pin_occurred:
            strength = min(MAX_STRENGTH, strength + 0.05)

        reason = (
            f"EXPIRY_TUESDAY | Exit day | "
            f"{'Pin confirmed' if pin_occurred else 'No strong pin'} "
            f"near {nearest} (dist={dist_pts:.0f}pts)"
        )

        logger.info('%s TUESDAY: %s EXIT strength=%.3f pin=%s',
                     self.SIGNAL_ID, td, strength, pin_occurred)

        return {
            'signal_id': self.SIGNAL_ID,
            'direction': 'NEUTRAL',  # Exit signal
            'strength': round(max(MIN_STRENGTH, strength), 3),
            'price': round(close, 2),
            'reason': reason,
            'metadata': {
                'phase': 'EXIT_TUESDAY',
                'expiry_date': td.isoformat(),
                'nearest_strike': int(nearest),
                'pin_occurred': pin_occurred,
                'distance_pts': round(dist_pts, 1),
            },
        }

    # ----------------------------------------------------------
    # Utility
    # ----------------------------------------------------------
    def reset(self) -> None:
        """Reset internal state for a fresh backtest run."""
        self._last_fire_date = None

    def __repr__(self) -> str:
        return f"ExpiryWeekTuesdaySignal(signal_id='{self.SIGNAL_ID}')"
