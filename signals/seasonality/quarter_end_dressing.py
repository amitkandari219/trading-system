"""
Quarter-End Window Dressing Signal.

Fund managers engage in "window dressing" in the last 5 trading days of
each quarter-ending month (March, June, September, December).  They chase
returns by buying recent winners and selling losers, creating a net
positive bias in the Nifty.

Signal logic:
    Last 5 trading days of March, June, September, December:
        LONG signal — positive bias from institutional window dressing.

    Strength:
        Based on how far Nifty is from quarter-start level.
        Bigger gap (up) = more window dressing (managers want to lock in).
        Bigger gap (down) = some desperation buying but weaker.

    Exit:
        First trading day of the new quarter.

    Filters:
        - Only fires in quarter-ending months (3, 6, 9, 12)
        - Must have sufficient history to compute quarter return

Walk-forward parameters exposed as class constants.

Usage:
    from signals.seasonality.quarter_end_dressing import QuarterEndDressingSignal

    sig = QuarterEndDressingSignal()
    result = sig.evaluate(df, date(2026, 3, 26))
"""

import logging
import math
from datetime import date, timedelta
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ================================================================
# CONSTANTS / WF PARAMETERS
# ================================================================

SIGNAL_ID = 'QUARTER_END_DRESSING'

# Quarter-ending months
QUARTER_END_MONTHS = {3, 6, 9, 12}

# Window
LAST_N_DAYS = 5                  # Last 5 trading days of quarter

# Quarter start months (for computing quarter return)
QUARTER_START_MAP = {3: 1, 6: 4, 9: 7, 12: 10}

# Strength
BASE_STRENGTH = 0.55
POSITIVE_QTR_BOOST = 0.15        # Quarter return > 0 → more dressing
STRONG_QTR_BOOST = 0.08          # Quarter return > 5% → extra boost
NEGATIVE_QTR_PENALTY = -0.05     # Quarter return < 0 → weaker signal
MIN_STRENGTH = 0.10
MAX_STRENGTH = 0.90

# Return scaling
QTR_RETURN_SCALE_PCT = 10.0      # 10% quarter return → full boost


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


def _is_quarter_end_month(td: date) -> bool:
    """Check if the date is in a quarter-ending month."""
    return td.month in QUARTER_END_MONTHS


def _is_last_n_trading_days_of_month(df: pd.DataFrame, td: date, n: int) -> bool:
    """Check if td is within the last N trading days of its month."""
    if df.empty or 'date' not in df.columns:
        return False

    month_end = td.replace(day=28) + timedelta(days=4)
    month_end = month_end.replace(day=1) - timedelta(days=1)

    remaining = df[
        (df['date'] >= pd.Timestamp(td)) &
        (df['date'] <= pd.Timestamp(month_end))
    ]
    return len(remaining) <= n


def _get_quarter_start_close(df: pd.DataFrame, td: date) -> float:
    """Get the close on the first trading day of the current quarter."""
    qtr_start_month = QUARTER_START_MAP.get(td.month)
    if qtr_start_month is None:
        return float('nan')

    qtr_start = date(td.year, qtr_start_month, 1)
    qtr_data = df[df['date'] >= pd.Timestamp(qtr_start)].sort_values('date')

    if qtr_data.empty:
        return float('nan')

    first_close = _safe_float(qtr_data.iloc[0].get('close'))
    return first_close


def _is_first_day_of_new_quarter(td: date) -> bool:
    """Check if td is in the first month of a quarter (Jan, Apr, Jul, Oct)."""
    return td.month in {1, 4, 7, 10}


# ================================================================
# SIGNAL CLASS
# ================================================================

class QuarterEndDressingSignal:
    """
    Quarter-end window dressing signal for Nifty.

    LONG in the last 5 trading days of March, June, September, December.
    Strength scaled by quarter-to-date return (bigger gap = more dressing).
    Exit on first trading day of new quarter.
    """

    SIGNAL_ID = SIGNAL_ID

    # WF parameters
    LAST_N_DAYS = LAST_N_DAYS
    BASE_STRENGTH = BASE_STRENGTH
    QUARTER_END_MONTHS = QUARTER_END_MONTHS
    QTR_RETURN_SCALE_PCT = QTR_RETURN_SCALE_PCT

    def __init__(self) -> None:
        self._last_fire_date: Optional[date] = None
        logger.info('QuarterEndDressingSignal initialised')

    # ----------------------------------------------------------
    # PUBLIC evaluate
    # ----------------------------------------------------------
    def evaluate(self, df: pd.DataFrame, trade_date: date) -> Optional[Dict]:
        """
        Evaluate Quarter-End Window Dressing signal.

        Parameters
        ----------
        df         : DataFrame with columns: date, close.
        trade_date : The date to evaluate.

        Returns
        -------
        dict with keys: signal_id, direction, strength, price, reason, metadata
        or None if no signal.
        """
        try:
            return self._evaluate_inner(df, trade_date)
        except Exception as e:
            logger.error('QuarterEndDressingSignal.evaluate error: %s', e, exc_info=True)
            return None

    def _evaluate_inner(self, df: pd.DataFrame, trade_date: date) -> Optional[Dict]:
        if df is None or df.empty:
            return None
        if 'date' not in df.columns:
            return None

        td = trade_date

        # Only fires in quarter-ending months
        if not _is_quarter_end_month(td):
            # Check if it's the exit day (first trading day of new quarter)
            if _is_first_day_of_new_quarter(td):
                return self._exit_signal(df, td)
            return None

        # Check if we're in last N trading days
        if not _is_last_n_trading_days_of_month(df, td, self.LAST_N_DAYS):
            return None

        # Get current row
        row = df[df['date'] == pd.Timestamp(td)]
        if row.empty:
            return None
        row = row.iloc[-1]

        close = _safe_float(row.get('close'))
        if math.isnan(close) or close <= 0:
            return None

        # Get quarter-start close for return calculation
        qtr_start_close = _get_quarter_start_close(df, td)
        qtr_return_pct = float('nan')
        qtr_note = ''

        if not math.isnan(qtr_start_close) and qtr_start_close > 0:
            qtr_return_pct = ((close - qtr_start_close) / qtr_start_close) * 100.0
            qtr_note = f'QTR_return={qtr_return_pct:+.2f}%'

        # Compute strength
        strength = self.BASE_STRENGTH

        if not math.isnan(qtr_return_pct):
            if qtr_return_pct > 0:
                strength += POSITIVE_QTR_BOOST
                if qtr_return_pct > 5.0:
                    strength += STRONG_QTR_BOOST
                # Scale with magnitude
                flow_boost = min(0.10, (qtr_return_pct / self.QTR_RETURN_SCALE_PCT) * 0.10)
                strength += flow_boost
            elif qtr_return_pct < 0:
                strength += NEGATIVE_QTR_PENALTY
                # Even in down quarters, some window dressing occurs

        strength = max(MIN_STRENGTH, min(MAX_STRENGTH, strength))

        # Count days until quarter end
        month_end = td.replace(day=28) + timedelta(days=4)
        month_end = month_end.replace(day=1) - timedelta(days=1)
        remaining = df[
            (df['date'] > pd.Timestamp(td)) &
            (df['date'] <= pd.Timestamp(month_end))
        ].shape[0]

        # Determine quarter label
        quarter_label = f"Q{((td.month - 1) // 3) + 1} {td.year}"

        reason = (
            f"QUARTER_END_DRESSING | LONG | {quarter_label} | "
            f"{remaining} trading days left | {qtr_note}"
        )

        logger.info('%s: LONG %s qtr_ret=%.2f%% strength=%.3f',
                     self.SIGNAL_ID, td,
                     qtr_return_pct if not math.isnan(qtr_return_pct) else 0.0,
                     strength)

        return {
            'signal_id': self.SIGNAL_ID,
            'direction': 'LONG',
            'strength': round(strength, 3),
            'price': round(close, 2),
            'reason': reason.strip(),
            'metadata': {
                'quarter': quarter_label,
                'qtr_return_pct': round(qtr_return_pct, 4) if not math.isnan(qtr_return_pct) else None,
                'qtr_start_close': round(qtr_start_close, 2) if not math.isnan(qtr_start_close) else None,
                'remaining_trading_days': remaining,
                'window_days': self.LAST_N_DAYS,
            },
        }

    def _exit_signal(self, df: pd.DataFrame, td: date) -> Optional[Dict]:
        """First day of new quarter: EXIT signal."""
        row = df[df['date'] == pd.Timestamp(td)]
        if row.empty:
            return None
        row = row.iloc[-1]

        close = _safe_float(row.get('close'))
        if math.isnan(close) or close <= 0:
            return None

        # Verify previous month was a quarter-ending month
        prev_month_end = td.replace(day=1) - timedelta(days=1)
        if prev_month_end.month not in QUARTER_END_MONTHS:
            return None

        quarter_label = f"Q{((prev_month_end.month - 1) // 3) + 1} {prev_month_end.year}"

        reason = (
            f"QUARTER_END_DRESSING_EXIT | Close LONG from {quarter_label} | "
            f"New quarter started"
        )

        logger.info('%s EXIT: %s after %s', self.SIGNAL_ID, td, quarter_label)

        return {
            'signal_id': self.SIGNAL_ID,
            'direction': 'NEUTRAL',  # Exit signal
            'strength': round(0.50, 3),
            'price': round(close, 2),
            'reason': reason,
            'metadata': {
                'phase': 'EXIT',
                'previous_quarter': quarter_label,
                'exit_date': td.isoformat(),
            },
        }

    # ----------------------------------------------------------
    # Utility
    # ----------------------------------------------------------
    def reset(self) -> None:
        """Reset internal state for a fresh backtest run."""
        self._last_fire_date = None

    def __repr__(self) -> str:
        return f"QuarterEndDressingSignal(signal_id='{self.SIGNAL_ID}')"
