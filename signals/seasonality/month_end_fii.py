"""
Month-End FII Rebalancing Signal.

Foreign Institutional Investors (FII) conduct portfolio rebalancing in the
last 3 trading days of each month.  The direction of this rebalancing flow
is predictable from the cumulative FII net investment during the month.

Signal logic:
    Last 3 trading days of month:
        Cumulative monthly FII net > 0      → LONG  (buying pressure continues)
        Cumulative monthly FII net < -5000 Cr → SHORT (accelerated selling)
        In between                           → NO TRADE

    Strength:
        Scaled by magnitude of cumulative FII flow.
        Larger absolute flow → stronger signal.

    Required column:
        fii_net — daily FII net investment in Crores (from fii_daily table).

    Filters:
        - fii_net column must be present and have sufficient data
        - At least 10 trading days of FII data in the month

Walk-forward parameters exposed as class constants.

Usage:
    from signals.seasonality.month_end_fii import MonthEndFIISignal

    sig = MonthEndFIISignal()
    result = sig.evaluate(df, date(2026, 3, 28))
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

SIGNAL_ID = 'MONTH_END_FII'

# Window
LAST_N_DAYS = 3                  # Last 3 trading days of month

# FII thresholds (in Crores)
FII_NET_LONG_THRESHOLD = 0       # Cumulative monthly FII > 0 → LONG
FII_NET_SHORT_THRESHOLD = -5000  # Cumulative monthly FII < -5000 → SHORT

# Strength scaling
BASE_STRENGTH = 0.55
FII_FLOW_SCALE_CR = 10000       # FII flow of 10,000 Cr → full boost
MAX_FLOW_BOOST = 0.25
MIN_STRENGTH = 0.10
MAX_STRENGTH = 0.90

# Minimum data requirement
MIN_MONTH_TRADING_DAYS = 10      # Need at least 10 days of FII data


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


def _is_last_n_trading_days(df: pd.DataFrame, td: date, n: int = LAST_N_DAYS) -> bool:
    """Check if td is within the last N trading days of its month."""
    if df.empty or 'date' not in df.columns:
        return False

    month_end = td.replace(day=28) + timedelta(days=4)
    month_end = month_end.replace(day=1) - timedelta(days=1)  # Last calendar day

    # Get all trading days in this month at or after td
    remaining = df[
        (df['date'] >= pd.Timestamp(td)) &
        (df['date'] <= pd.Timestamp(month_end))
    ]
    return len(remaining) <= n


def _get_month_fii_cumulative(df: pd.DataFrame, td: date) -> float:
    """Compute cumulative FII net for the month up to td (inclusive)."""
    if 'fii_net' not in df.columns:
        return float('nan')

    first_of_month = td.replace(day=1)
    month_data = df[
        (df['date'] >= pd.Timestamp(first_of_month)) &
        (df['date'] <= pd.Timestamp(td))
    ]

    if len(month_data) < MIN_MONTH_TRADING_DAYS:
        return float('nan')

    fii_sum = month_data['fii_net'].sum()
    return float(fii_sum) if not pd.isna(fii_sum) else float('nan')


# ================================================================
# SIGNAL CLASS
# ================================================================

class MonthEndFIISignal:
    """
    Month-end FII rebalancing signal for Nifty.

    LONG if cumulative monthly FII > 0 in last 3 trading days.
    SHORT if cumulative monthly FII < -5000 Cr in last 3 trading days.
    """

    SIGNAL_ID = SIGNAL_ID

    # WF parameters
    LAST_N_DAYS = LAST_N_DAYS
    FII_NET_LONG_THRESHOLD = FII_NET_LONG_THRESHOLD
    FII_NET_SHORT_THRESHOLD = FII_NET_SHORT_THRESHOLD
    BASE_STRENGTH = BASE_STRENGTH
    FII_FLOW_SCALE_CR = FII_FLOW_SCALE_CR

    def __init__(self) -> None:
        self._last_fire_date: Optional[date] = None
        logger.info('MonthEndFIISignal initialised')

    # ----------------------------------------------------------
    # PUBLIC evaluate
    # ----------------------------------------------------------
    def evaluate(self, df: pd.DataFrame, trade_date: date) -> Optional[Dict]:
        """
        Evaluate Month-End FII Rebalancing signal.

        Parameters
        ----------
        df         : DataFrame with columns: date, close, fii_net.
        trade_date : The date to evaluate.

        Returns
        -------
        dict with keys: signal_id, direction, strength, price, reason, metadata
        or None if no signal.
        """
        try:
            return self._evaluate_inner(df, trade_date)
        except Exception as e:
            logger.error('MonthEndFIISignal.evaluate error: %s', e, exc_info=True)
            return None

    def _evaluate_inner(self, df: pd.DataFrame, trade_date: date) -> Optional[Dict]:
        if df is None or df.empty:
            return None
        if 'date' not in df.columns:
            return None
        if 'fii_net' not in df.columns:
            logger.debug('fii_net column missing')
            return None

        td = trade_date

        # Check if we're in last N trading days of month
        if not _is_last_n_trading_days(df, td, self.LAST_N_DAYS):
            return None

        # Get current row
        row = df[df['date'] == pd.Timestamp(td)]
        if row.empty:
            return None
        row = row.iloc[-1]

        close = _safe_float(row.get('close'))
        if math.isnan(close) or close <= 0:
            return None

        # Compute cumulative FII flow for the month
        fii_cum = _get_month_fii_cumulative(df, td)
        if math.isnan(fii_cum):
            logger.debug('Insufficient FII data for %s', td)
            return None

        # Determine direction
        if fii_cum > self.FII_NET_LONG_THRESHOLD:
            direction = 'LONG'
        elif fii_cum < self.FII_NET_SHORT_THRESHOLD:
            direction = 'SHORT'
        else:
            logger.debug('FII cum %.0f Cr in neutral zone — no signal', fii_cum)
            return None

        # Compute strength
        strength = self.BASE_STRENGTH
        flow_ratio = min(1.0, abs(fii_cum) / self.FII_FLOW_SCALE_CR)
        strength += flow_ratio * MAX_FLOW_BOOST
        strength = max(MIN_STRENGTH, min(MAX_STRENGTH, strength))

        # Count remaining trading days in month
        month_end_cal = td.replace(day=28) + timedelta(days=4)
        month_end_cal = month_end_cal.replace(day=1) - timedelta(days=1)
        remaining_days = df[
            (df['date'] > pd.Timestamp(td)) &
            (df['date'] <= pd.Timestamp(month_end_cal))
        ].shape[0]

        reason = (
            f"MONTH_END_FII | {direction} | "
            f"FII_cum={fii_cum:+,.0f} Cr | "
            f"{remaining_days} trading days left in month"
        )

        logger.info('%s: %s %s fii_cum=%.0f strength=%.3f',
                     self.SIGNAL_ID, direction, td, fii_cum, strength)

        return {
            'signal_id': self.SIGNAL_ID,
            'direction': direction,
            'strength': round(strength, 3),
            'price': round(close, 2),
            'reason': reason,
            'metadata': {
                'fii_cumulative_cr': round(fii_cum, 0),
                'month': td.strftime('%Y-%m'),
                'remaining_trading_days': remaining_days,
                'long_threshold': self.FII_NET_LONG_THRESHOLD,
                'short_threshold': self.FII_NET_SHORT_THRESHOLD,
            },
        }

    # ----------------------------------------------------------
    # Utility
    # ----------------------------------------------------------
    def reset(self) -> None:
        """Reset internal state for a fresh backtest run."""
        self._last_fire_date = None

    def __repr__(self) -> str:
        return f"MonthEndFIISignal(signal_id='{self.SIGNAL_ID}')"
