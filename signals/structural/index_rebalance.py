"""
MSCI / FTSE Index Rebalancing Flow Signal — SCORING signal.

Exploits the predictable flow around quarterly MSCI and FTSE index
rebalancing events.  When stocks are added to (or deleted from) these
global indices, passive funds must buy (or sell), creating price pressure
on Nifty constituents and hence on the Nifty index itself.

Signal logic:
    1. Identify upcoming MSCI/FTSE effective rebalance dates.
    2. In the T-10 to T-7 window before the effective date:
       - If HIGH_IMPACT additions expected (flow > 3x ADTV) → LONG Nifty
       - If net deletions dominate → SHORT Nifty
    3. Hold until T+1 (day after effective date).
    4. SL: 1.5%, TGT: 3%

Backtest approximation:
    Since actual MSCI addition/deletion lists are not available historically,
    we use the empirical observation that Nifty tends to be slightly bullish
    (+0.3% to +0.5%) in the 10 days before rebalance effective dates.

Data source:
    - Hardcoded 2026 rebalance calendar (MSCI + FTSE)
    - Rebalance event details (additions/deletions) passed at runtime

Usage:
    from signals.structural.index_rebalance import IndexRebalanceSignal

    sig = IndexRebalanceSignal()

    # Check upcoming
    upcoming = sig.get_upcoming_rebalances(as_of=date(2026, 2, 15))

    # Evaluate
    result = sig.evaluate(
        trade_date=date(2026, 2, 18),
        rebalance_events=[
            {'index': 'MSCI', 'action': 'ADD', 'stock': 'RELIANCE',
             'estimated_flow_adtv_multiple': 4.2},
        ],
    )

    # Backtest
    result = sig.backtest_evaluate(trade_date, daily_df)
"""

from __future__ import annotations

import logging
import math
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ================================================================
# CONSTANTS
# ================================================================

SIGNAL_ID = 'INDEX_REBALANCE'

# Pre-rebalance trading window
WINDOW_START_DAYS = 10      # T-10 before effective date
WINDOW_END_DAYS = 7         # T-7 before effective date (enter early, ride flow)

# Flow thresholds
HIGH_IMPACT_ADTV_MULTIPLE = 3.0   # Flow > 3x ADTV = high impact

# Risk management
SL_PCT = 1.5
TGT_PCT = 3.0
HOLD_UNTIL_T_PLUS = 1      # Hold until T+1 after effective date

# Confidence
CONF_HIGH_IMPACT_ADD = 0.62
CONF_MODERATE_ADD = 0.55
CONF_NET_DELETION = 0.54
CONF_BACKTEST_DEFAULT = 0.52    # Lower confidence for approximate backtest signals

# Size
BASE_SIZE_MODIFIER = 1.0
MAX_SIZE_MODIFIER = 1.25
MIN_SIZE_MODIFIER = 0.7

# ================================================================
# REBALANCE CALENDAR — 2026
# ================================================================

MSCI_DATES_2026 = {
    'announcement': [
        date(2026, 2, 13),
        date(2026, 5, 13),
        date(2026, 8, 12),
        date(2026, 11, 12),
    ],
    'effective': [
        date(2026, 2, 27),
        date(2026, 5, 29),
        date(2026, 8, 28),
        date(2026, 11, 30),
    ],
}

FTSE_DATES_2026 = {
    'announcement': [
        date(2026, 3, 6),
        date(2026, 6, 5),
        date(2026, 9, 4),
        date(2026, 12, 4),
    ],
    'effective': [
        date(2026, 3, 20),
        date(2026, 6, 19),
        date(2026, 9, 18),
        date(2026, 12, 18),
    ],
}

# Combined calendar for convenience
_ALL_REBALANCES: List[Dict] = []

for i, eff in enumerate(MSCI_DATES_2026['effective']):
    _ALL_REBALANCES.append({
        'index': 'MSCI',
        'quarter': f'Q{i + 1}',
        'announcement': MSCI_DATES_2026['announcement'][i],
        'effective': eff,
    })

for i, eff in enumerate(FTSE_DATES_2026['effective']):
    _ALL_REBALANCES.append({
        'index': 'FTSE',
        'quarter': f'Q{i + 1}',
        'announcement': FTSE_DATES_2026['announcement'][i],
        'effective': eff,
    })


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


def _calendar_days_between(d1: date, d2: date) -> int:
    """Signed calendar days from d1 to d2."""
    return (d2 - d1).days


def _trading_days_between(start: date, end: date) -> int:
    """Approximate trading days between two dates (Mon-Fri)."""
    if start > end:
        return 0
    count = 0
    d = start
    while d <= end:
        if d.weekday() < 5:
            count += 1
        d += timedelta(days=1)
    return count


# ================================================================
# MAIN SIGNAL CLASS
# ================================================================

class IndexRebalanceSignal:
    """
    MSCI / FTSE index rebalancing flow — SCORING signal.

    Fires directional trades in the pre-rebalance window (T-10 to T-7)
    based on expected index constituent changes.
    """

    SIGNAL_ID = SIGNAL_ID

    def __init__(
        self,
        extra_rebalances: Optional[List[Dict]] = None,
        holiday_dates: Optional[List[date]] = None,
    ):
        """
        Parameters
        ----------
        extra_rebalances : list of dict, optional
            Additional rebalance events beyond the hardcoded 2026 calendar.
            Each dict: {index, quarter, announcement, effective}.
        holiday_dates : list of date, optional
            Known market holidays for better window calculation.
        """
        self._rebalances = list(_ALL_REBALANCES)
        if extra_rebalances:
            self._rebalances.extend(extra_rebalances)
        self._holidays: set = set(holiday_dates) if holiday_dates else set()

    # ----------------------------------------------------------------
    # PUBLIC — live evaluation
    # ----------------------------------------------------------------

    def evaluate(
        self,
        trade_date: date,
        rebalance_events: Optional[List[Dict]] = None,
        nifty_price: Optional[float] = None,
    ) -> Optional[Dict]:
        """
        Evaluate index rebalance flow for a given date.

        Parameters
        ----------
        trade_date : date
            Current trading date.
        rebalance_events : list of dict, optional
            Expected additions/deletions for an upcoming rebalance.
            Each dict should have:
                action: 'ADD' or 'DELETE'
                stock: str (ticker)
                estimated_flow_adtv_multiple: float
                index: 'MSCI' or 'FTSE' (optional)
            If None, the signal checks the calendar but cannot determine
            direction without event details (backtest mode handles this).
        nifty_price : float, optional
            Current Nifty price for SL/TGT computation.

        Returns
        -------
        dict or None
        """
        # --- Find active rebalance window ---
        active_rebalance = self._find_active_rebalance(trade_date)
        if active_rebalance is None:
            logger.debug('[%s] %s not in any pre-rebalance window', SIGNAL_ID, trade_date)
            return None

        effective_date = active_rebalance['effective']
        rebalance_index = active_rebalance['index']
        days_to_effective = _calendar_days_between(trade_date, effective_date)

        # --- Classify direction from events ---
        if rebalance_events is None or len(rebalance_events) == 0:
            logger.info('[%s] No rebalance events provided for %s — cannot determine direction',
                        SIGNAL_ID, trade_date)
            return None

        direction, confidence, event_summary = self._classify_events(rebalance_events)
        if direction is None:
            logger.info('[%s] Rebalance events on %s are net neutral — no trade',
                        SIGNAL_ID, trade_date)
            return None

        # --- Size modifier ---
        size_mod = BASE_SIZE_MODIFIER
        high_impact_count = sum(
            1 for e in rebalance_events
            if _safe_float(e.get('estimated_flow_adtv_multiple', 0)) >= HIGH_IMPACT_ADTV_MULTIPLE
        )
        if high_impact_count >= 3:
            size_mod += 0.15
        elif high_impact_count >= 1:
            size_mod += 0.08
        size_mod = max(MIN_SIZE_MODIFIER, min(MAX_SIZE_MODIFIER, size_mod))

        # --- SL / TGT ---
        entry_price = _safe_float(nifty_price) if nifty_price is not None else None
        if entry_price is not None and math.isnan(entry_price):
            entry_price = None

        stop_loss = None
        target = None
        if entry_price is not None:
            if direction == 'LONG':
                stop_loss = round(entry_price * (1 - SL_PCT / 100), 2)
                target = round(entry_price * (1 + TGT_PCT / 100), 2)
            else:
                stop_loss = round(entry_price * (1 + SL_PCT / 100), 2)
                target = round(entry_price * (1 - TGT_PCT / 100), 2)

        # --- Hold period ---
        hold_days = days_to_effective + HOLD_UNTIL_T_PLUS

        return {
            'signal_id': SIGNAL_ID,
            'trade_date': trade_date,
            'direction': direction,
            'confidence': round(confidence, 4),
            'size_modifier': round(size_mod, 2),
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'target': target,
            'hold_days': hold_days,
            'sl_pct': SL_PCT,
            'tgt_pct': TGT_PCT,
            'rebalance_index': rebalance_index,
            'effective_date': effective_date,
            'days_to_effective': days_to_effective,
            'event_summary': event_summary,
            'high_impact_additions': high_impact_count,
        }

    # ----------------------------------------------------------------
    # PUBLIC — backtest evaluation
    # ----------------------------------------------------------------

    def backtest_evaluate(
        self,
        trade_date: date,
        daily_df: pd.DataFrame,
    ) -> Optional[Dict]:
        """
        Backtest evaluation using historical price data.

        Since actual MSCI/FTSE addition/deletion data is not available,
        we use the empirical observation that Nifty is slightly bullish
        in the pre-rebalance window.  This produces a LONG bias with
        lower confidence.

        Parameters
        ----------
        trade_date : date
        daily_df : pd.DataFrame
            Nifty daily data with columns: date, open, high, low, close.

        Returns
        -------
        dict or None
        """
        # --- Check if in a pre-rebalance window ---
        active_rebalance = self._find_active_rebalance(trade_date)
        if active_rebalance is None:
            return None

        effective_date = active_rebalance['effective']
        rebalance_index = active_rebalance['index']
        days_to_effective = _calendar_days_between(trade_date, effective_date)

        # --- Get Nifty price ---
        nifty_price = self._get_nifty_close(daily_df, trade_date)

        # --- Historical bias: pre-rebalance is slightly bullish ---
        # Check recent momentum to modulate the direction
        direction = 'LONG'  # Default historical bias
        confidence = CONF_BACKTEST_DEFAULT

        recent_return = self._recent_return(daily_df, trade_date, lookback=10)
        if recent_return is not None:
            if recent_return > 1.5:
                # Already rallied significantly — reduce confidence
                confidence -= 0.03
            elif recent_return < -2.0:
                # Sharp sell-off — rebalance flow may not overcome
                direction = 'SHORT'
                confidence = 0.50
            elif recent_return > 0:
                # Mild positive — aligns with rebalance flow
                confidence += 0.02

        confidence = max(0.45, min(0.70, confidence))
        size_mod = BASE_SIZE_MODIFIER  # Conservative for backtest

        # --- SL / TGT ---
        entry_price = nifty_price
        stop_loss = None
        target = None
        if entry_price is not None:
            if direction == 'LONG':
                stop_loss = round(entry_price * (1 - SL_PCT / 100), 2)
                target = round(entry_price * (1 + TGT_PCT / 100), 2)
            else:
                stop_loss = round(entry_price * (1 + SL_PCT / 100), 2)
                target = round(entry_price * (1 - TGT_PCT / 100), 2)

        hold_days = days_to_effective + HOLD_UNTIL_T_PLUS

        return {
            'signal_id': SIGNAL_ID,
            'trade_date': trade_date,
            'direction': direction,
            'confidence': round(confidence, 4),
            'size_modifier': round(size_mod, 2),
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'target': target,
            'hold_days': hold_days,
            'sl_pct': SL_PCT,
            'tgt_pct': TGT_PCT,
            'rebalance_index': rebalance_index,
            'effective_date': effective_date,
            'days_to_effective': days_to_effective,
            'event_summary': 'BACKTEST_APPROXIMATION — historical bullish bias pre-rebalance',
            'high_impact_additions': 0,
        }

    # ----------------------------------------------------------------
    # PUBLIC — upcoming rebalances
    # ----------------------------------------------------------------

    def get_upcoming_rebalances(self, as_of: date) -> List[Dict]:
        """
        Return rebalance events within 30 days of as_of.

        Parameters
        ----------
        as_of : date

        Returns
        -------
        list of dict
            Each dict has: index, quarter, announcement, effective,
            days_until_effective, in_pre_rebalance_window.
        """
        horizon = as_of + timedelta(days=30)
        results = []

        for rb in self._rebalances:
            eff = rb['effective']
            if as_of <= eff <= horizon:
                days_until = _calendar_days_between(as_of, eff)
                in_window = self._is_in_pre_rebalance_window(as_of, eff)
                results.append({
                    'index': rb['index'],
                    'quarter': rb.get('quarter', ''),
                    'announcement': rb.get('announcement'),
                    'effective': eff,
                    'days_until_effective': days_until,
                    'in_pre_rebalance_window': in_window,
                })

        # Sort by effective date
        results.sort(key=lambda x: x['effective'])
        return results

    # ----------------------------------------------------------------
    # PRE-REBALANCE WINDOW
    # ----------------------------------------------------------------

    def _is_in_pre_rebalance_window(
        self,
        trade_date: date,
        effective_date: date,
    ) -> bool:
        """
        Check if trade_date is in the T-10 to T-7 calendar-day window
        before the effective_date.

        The window is [effective - 10, effective - 7] inclusive.
        """
        days_before = _calendar_days_between(trade_date, effective_date)
        return WINDOW_END_DAYS <= days_before <= WINDOW_START_DAYS

    def _find_active_rebalance(self, trade_date: date) -> Optional[Dict]:
        """
        Find a rebalance event whose pre-rebalance window contains trade_date.

        Returns the rebalance dict or None.
        """
        for rb in self._rebalances:
            eff = rb['effective']
            if self._is_in_pre_rebalance_window(trade_date, eff):
                return rb
        return None

    # ----------------------------------------------------------------
    # EVENT CLASSIFICATION
    # ----------------------------------------------------------------

    @staticmethod
    def _classify_events(
        events: List[Dict],
    ) -> Tuple[Optional[str], float, str]:
        """
        Classify rebalance events into a direction.

        Parameters
        ----------
        events : list of dict
            Each: {action, stock, estimated_flow_adtv_multiple, ...}

        Returns
        -------
        (direction, confidence, summary_string)
        direction is 'LONG', 'SHORT', or None (neutral).
        """
        additions = []
        deletions = []
        high_impact_adds = 0
        high_impact_dels = 0

        for ev in events:
            action = str(ev.get('action', '')).upper().strip()
            stock = ev.get('stock', 'UNKNOWN')
            flow_mult = _safe_float(ev.get('estimated_flow_adtv_multiple', 0))
            if math.isnan(flow_mult):
                flow_mult = 0.0

            if action in ('ADD', 'ADDITION', 'INCLUDE'):
                additions.append(stock)
                if flow_mult >= HIGH_IMPACT_ADTV_MULTIPLE:
                    high_impact_adds += 1
            elif action in ('DELETE', 'DELETION', 'REMOVE', 'EXCLUDE'):
                deletions.append(stock)
                if flow_mult >= HIGH_IMPACT_ADTV_MULTIPLE:
                    high_impact_dels += 1

        n_adds = len(additions)
        n_dels = len(deletions)

        # Decision logic
        if high_impact_adds > 0 and high_impact_adds > high_impact_dels:
            # High-impact additions dominate → LONG
            direction = 'LONG'
            confidence = CONF_HIGH_IMPACT_ADD
            if high_impact_adds >= 3:
                confidence += 0.04
            summary = (f'{n_adds} additions ({high_impact_adds} high-impact), '
                       f'{n_dels} deletions ({high_impact_dels} high-impact)')

        elif n_adds > n_dels and n_adds > 0:
            # More additions but not necessarily high-impact
            direction = 'LONG'
            confidence = CONF_MODERATE_ADD
            summary = f'{n_adds} additions, {n_dels} deletions (moderate impact)'

        elif n_dels > n_adds and n_dels > 0:
            # Net deletions → SHORT
            direction = 'SHORT'
            confidence = CONF_NET_DELETION
            if high_impact_dels >= 2:
                confidence += 0.03
            summary = f'{n_dels} deletions ({high_impact_dels} high-impact), {n_adds} additions'

        elif n_adds == n_dels and n_adds > 0:
            # Balanced — check impact
            if high_impact_adds > high_impact_dels:
                direction = 'LONG'
                confidence = CONF_MODERATE_ADD - 0.02
                summary = f'Balanced count ({n_adds} each) but add-heavy impact'
            elif high_impact_dels > high_impact_adds:
                direction = 'SHORT'
                confidence = CONF_NET_DELETION - 0.02
                summary = f'Balanced count ({n_adds} each) but delete-heavy impact'
            else:
                return None, 0.0, 'Balanced additions/deletions — neutral'
        else:
            return None, 0.0, 'No additions or deletions'

        confidence = max(0.45, min(0.75, confidence))
        return direction, confidence, summary

    # ----------------------------------------------------------------
    # DATA HELPERS
    # ----------------------------------------------------------------

    @staticmethod
    def _get_nifty_close(daily_df: pd.DataFrame, trade_date: date) -> Optional[float]:
        """Get Nifty close price for trade_date."""
        if daily_df is None or daily_df.empty:
            return None

        df = daily_df.copy()
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date']).dt.date

        row = df[df['date'] == trade_date]
        if row.empty:
            row = df[df['date'] <= trade_date].tail(1)
        if row.empty:
            return None

        val = _safe_float(row.iloc[0]['close'])
        return val if not math.isnan(val) else None

    @staticmethod
    def _recent_return(
        daily_df: pd.DataFrame,
        trade_date: date,
        lookback: int = 10,
    ) -> Optional[float]:
        """
        Compute Nifty return (%) over the last `lookback` trading days.
        """
        if daily_df is None or daily_df.empty:
            return None

        df = daily_df.copy()
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date']).dt.date

        recent = df[df['date'] <= trade_date].tail(lookback)
        if len(recent) < max(3, lookback // 2):
            return None

        first_close = _safe_float(recent.iloc[0]['close'])
        last_close = _safe_float(recent.iloc[-1]['close'])

        if math.isnan(first_close) or math.isnan(last_close) or first_close <= 0:
            return None

        return round((last_close - first_close) / first_close * 100, 4)

    # ----------------------------------------------------------------
    # UTILITY
    # ----------------------------------------------------------------

    def __repr__(self) -> str:
        return f'IndexRebalanceSignal(signal_id={SIGNAL_ID!r})'
