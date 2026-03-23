"""
Union Budget Day Volatility Signal.

Exploits the predictable IV expansion/contraction cycle around India's
Union Budget presentation (Feb 1, or next trading day if holiday).

Signal logic:
    Phase 1 — PRE-BUDGET (T-5 to T-1):
        IV expands as market prices in uncertainty.
        Signal: REDUCE_SIZE overlay — avoid new directional entries,
        or sell premium (straddle/strangle) to capture elevated IV.

    Phase 2 — BUDGET DAY (T+0):
        Direction follows first 30-min reaction (momentum).
        If Nifty rises > 0.5% in first 30 min → LONG.
        If Nifty falls > 0.5% in first 30 min → SHORT.
        Otherwise → NO TRADE (choppy).

    Phase 3 — POST-BUDGET (T+1):
        IV crush within hours — premium sellers profit.
        Signal: LONG (bullish drift post-event resolution).

    Filters:
        - india_vix column required for pre-budget IV detection
        - close, open columns required

Walk-forward parameters exposed as class constants.

Usage:
    from signals.seasonality.budget_day import BudgetDaySignal

    sig = BudgetDaySignal()
    result = sig.evaluate(df, date(2026, 2, 1))
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

SIGNAL_ID = 'BUDGET_DAY'

# Historical Union Budget dates (Feb 1 or adjusted for holidays)
BUDGET_DATES: List[date] = [
    date(2017, 2, 1),
    date(2018, 2, 1),
    date(2019, 2, 1),
    date(2020, 2, 1),
    date(2021, 2, 1),
    date(2022, 2, 1),
    date(2023, 2, 1),
    date(2024, 2, 1),
    date(2025, 2, 1),
    date(2026, 2, 2),   # Feb 1 is Sunday → moved to Monday Feb 2
]

# Pre-budget window
PRE_BUDGET_DAYS = 5              # Signal fires 5 days before budget
PRE_BUDGET_IV_EXPANSION = 1.15   # IV typically 15%+ above 20-day mean

# Day-of thresholds
FIRST_30MIN_THRESHOLD_PCT = 0.5  # 0.5% move in first 30 min → momentum
DAY_OF_BASE_STRENGTH = 0.70

# Post-budget
POST_BUDGET_BULLISH_BIAS = 0.60  # Historical bullish drift probability

# Strength scaling
PRE_BUDGET_STRENGTH = 0.55       # Overlay: reduce size
POST_BUDGET_STRENGTH = 0.60
MIN_STRENGTH = 0.10
MAX_STRENGTH = 0.90

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


def _get_budget_date_for_year(year: int) -> Optional[date]:
    """Return the budget date for a given year from the constant list."""
    for d in BUDGET_DATES:
        if d.year == year:
            return d
    # Default: Feb 1 of the year
    candidate = date(year, 2, 1)
    # Adjust if Saturday/Sunday
    if candidate.weekday() == 5:   # Saturday
        candidate = candidate + timedelta(days=2)
    elif candidate.weekday() == 6:  # Sunday
        candidate = candidate + timedelta(days=1)
    return candidate


def _trading_days_between(df: pd.DataFrame, start: date, end: date) -> int:
    """Count trading days in df between start and end (exclusive of end)."""
    if df.empty or 'date' not in df.columns:
        return -1
    mask = (df['date'] >= pd.Timestamp(start)) & (df['date'] < pd.Timestamp(end))
    return int(mask.sum())


# ================================================================
# SIGNAL CLASS
# ================================================================

class BudgetDaySignal:
    """
    Union Budget day volatility signal for Nifty.

    Fires up to 3 phases:
        - PRE_BUDGET (T-5 to T-1): reduce size / sell premium
        - BUDGET_DAY (T+0): momentum following first 30-min move
        - POST_BUDGET (T+1): IV crush → bullish drift
    """

    SIGNAL_ID = SIGNAL_ID
    BUDGET_DATES = BUDGET_DATES

    # WF parameters
    PRE_BUDGET_DAYS = PRE_BUDGET_DAYS
    FIRST_30MIN_THRESHOLD_PCT = FIRST_30MIN_THRESHOLD_PCT
    PRE_BUDGET_IV_EXPANSION = PRE_BUDGET_IV_EXPANSION
    DAY_OF_BASE_STRENGTH = DAY_OF_BASE_STRENGTH

    def __init__(self) -> None:
        self._last_fire_date: Optional[date] = None
        logger.info('BudgetDaySignal initialised')

    # ----------------------------------------------------------
    # PUBLIC evaluate
    # ----------------------------------------------------------
    def evaluate(self, df: pd.DataFrame, trade_date: date) -> Optional[Dict]:
        """
        Evaluate Budget Day signal.

        Parameters
        ----------
        df         : DataFrame with columns: date, open, high, low, close,
                     india_vix (optional). Must include history around budget.
        trade_date : The date to evaluate.

        Returns
        -------
        dict with keys: signal_id, direction, strength, price, reason, metadata
        or None if no signal.
        """
        try:
            return self._evaluate_inner(df, trade_date)
        except Exception as e:
            logger.error('BudgetDaySignal.evaluate error: %s', e, exc_info=True)
            return None

    def _evaluate_inner(self, df: pd.DataFrame, trade_date: date) -> Optional[Dict]:
        if df is None or df.empty:
            logger.debug('Empty DataFrame')
            return None

        # Normalise date column
        if 'date' not in df.columns:
            logger.debug('Missing date column')
            return None

        budget_date = _get_budget_date_for_year(trade_date.year)
        if budget_date is None:
            return None

        # Determine phase
        td = trade_date
        delta_days = (budget_date - td).days  # positive = before budget

        # Get current row
        row = df[df['date'] == pd.Timestamp(td)]
        if row.empty:
            logger.debug('No data for %s', td)
            return None
        row = row.iloc[-1]

        close = _safe_float(row.get('close'))
        open_price = _safe_float(row.get('open'))
        if math.isnan(close) or close <= 0:
            return None

        # ── Phase 1: PRE-BUDGET (5 days before to 1 day before) ──
        if 1 <= delta_days <= self.PRE_BUDGET_DAYS:
            return self._pre_budget_signal(df, row, td, close, delta_days, budget_date)

        # ── Phase 2: BUDGET DAY ──────────────────────────────────
        if td == budget_date:
            return self._budget_day_signal(row, td, open_price, close)

        # ── Phase 3: POST-BUDGET (1 day after) ───────────────────
        if delta_days == -1 or (budget_date - td).days == -1:
            return self._post_budget_signal(df, row, td, close, budget_date)

        # Check if it's 1 trading day after budget
        if -3 <= delta_days <= -1:
            # Approximate: check if budget was the previous trading day
            budget_row = df[df['date'] == pd.Timestamp(budget_date)]
            if not budget_row.empty:
                dates_after = df[df['date'] > pd.Timestamp(budget_date)].sort_values('date')
                if not dates_after.empty and dates_after.iloc[0]['date'] == pd.Timestamp(td):
                    return self._post_budget_signal(df, row, td, close, budget_date)

        return None

    # ----------------------------------------------------------
    # Phase signals
    # ----------------------------------------------------------
    def _pre_budget_signal(
        self, df: pd.DataFrame, row: pd.Series, td: date,
        close: float, days_before: int, budget_date: date,
    ) -> Optional[Dict]:
        """PRE-BUDGET: IV expansion detected → reduce size overlay."""
        strength = PRE_BUDGET_STRENGTH

        # Check IV expansion if india_vix available
        vix = _safe_float(row.get('india_vix'))
        iv_note = ''
        if not math.isnan(vix) and vix > 0:
            # Compute 20-day mean VIX
            hist = df[df['date'] < pd.Timestamp(td)].tail(20)
            if 'india_vix' in hist.columns and len(hist) >= 10:
                mean_vix = hist['india_vix'].mean()
                if not math.isnan(mean_vix) and mean_vix > 0:
                    iv_ratio = vix / mean_vix
                    if iv_ratio >= self.PRE_BUDGET_IV_EXPANSION:
                        strength = min(MAX_STRENGTH, strength + 0.10)
                        iv_note = f'IV_ratio={iv_ratio:.2f}'
                    else:
                        iv_note = f'IV_ratio={iv_ratio:.2f}(low)'

        # Closer to budget → stronger signal
        strength = min(MAX_STRENGTH, strength + (self.PRE_BUDGET_DAYS - days_before) * 0.03)

        reason = (
            f"PRE_BUDGET | {days_before}d before budget {budget_date} | "
            f"Reduce size / sell premium | {iv_note}"
        )

        logger.info('%s PRE_BUDGET: %s days_before=%d strength=%.3f',
                     self.SIGNAL_ID, td, days_before, strength)

        return {
            'signal_id': self.SIGNAL_ID,
            'direction': 'NEUTRAL',
            'strength': round(max(MIN_STRENGTH, strength), 3),
            'price': round(close, 2),
            'reason': reason.strip(),
            'metadata': {
                'phase': 'PRE_BUDGET',
                'days_before_budget': days_before,
                'budget_date': budget_date.isoformat(),
                'overlay': 'REDUCE_SIZE',
            },
        }

    def _budget_day_signal(
        self, row: pd.Series, td: date, open_price: float, close: float,
    ) -> Optional[Dict]:
        """BUDGET DAY: momentum signal based on first 30-min reaction."""
        if math.isnan(open_price) or open_price <= 0:
            return None

        # Use open→close as proxy for first-30-min direction in daily data
        move_pct = ((close - open_price) / open_price) * 100.0

        if abs(move_pct) < self.FIRST_30MIN_THRESHOLD_PCT:
            logger.debug('Budget day move %.2f%% below threshold', move_pct)
            return None

        direction = 'LONG' if move_pct > 0 else 'SHORT'

        # Strength scales with magnitude of move
        strength = self.DAY_OF_BASE_STRENGTH + min(0.20, abs(move_pct) * 0.05)
        strength = min(MAX_STRENGTH, max(MIN_STRENGTH, strength))

        reason = (
            f"BUDGET_DAY | Momentum {direction} | "
            f"Move={move_pct:+.2f}% | Open={open_price:.2f} Close={close:.2f}"
        )

        logger.info('%s BUDGET_DAY: %s %s move=%.2f%%',
                     self.SIGNAL_ID, direction, td, move_pct)

        return {
            'signal_id': self.SIGNAL_ID,
            'direction': direction,
            'strength': round(strength, 3),
            'price': round(close, 2),
            'reason': reason,
            'metadata': {
                'phase': 'BUDGET_DAY',
                'move_pct': round(move_pct, 4),
                'open': round(open_price, 2),
                'close': round(close, 2),
                'signal_type': 'MOMENTUM',
            },
        }

    def _post_budget_signal(
        self, df: pd.DataFrame, row: pd.Series, td: date,
        close: float, budget_date: date,
    ) -> Optional[Dict]:
        """POST-BUDGET: IV crush → bullish drift."""
        strength = POST_BUDGET_STRENGTH

        # Check if VIX dropped from budget day
        vix = _safe_float(row.get('india_vix'))
        budget_row = df[df['date'] == pd.Timestamp(budget_date)]
        vix_drop_note = ''
        if not budget_row.empty:
            budget_vix = _safe_float(budget_row.iloc[-1].get('india_vix'))
            if not math.isnan(vix) and not math.isnan(budget_vix) and budget_vix > 0:
                vix_drop = ((budget_vix - vix) / budget_vix) * 100.0
                if vix_drop > 5:
                    strength = min(MAX_STRENGTH, strength + 0.10)
                    vix_drop_note = f'VIX_crush={vix_drop:.1f}%'

        reason = (
            f"POST_BUDGET | IV crush after {budget_date} | "
            f"Bullish drift bias | {vix_drop_note}"
        )

        logger.info('%s POST_BUDGET: %s strength=%.3f',
                     self.SIGNAL_ID, td, strength)

        return {
            'signal_id': self.SIGNAL_ID,
            'direction': 'LONG',
            'strength': round(max(MIN_STRENGTH, strength), 3),
            'price': round(close, 2),
            'reason': reason.strip(),
            'metadata': {
                'phase': 'POST_BUDGET',
                'budget_date': budget_date.isoformat(),
                'signal_type': 'IV_CRUSH',
            },
        }

    # ----------------------------------------------------------
    # Utility
    # ----------------------------------------------------------
    def reset(self) -> None:
        """Reset internal state for a fresh backtest run."""
        self._last_fire_date = None

    def __repr__(self) -> str:
        return f"BudgetDaySignal(signal_id='{self.SIGNAL_ID}')"
