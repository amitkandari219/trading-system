"""
NSE Structural Signals — calendar/event-based signals
directly observable in price + calendar data.

No books, no extraction, no DSL translation needed.
These are market structure patterns validated by SSRN papers.
"""

import calendar
import logging
from datetime import date, timedelta
from typing import List, Dict

import pandas as pd

logger = logging.getLogger(__name__)

# RBI MPC dates (update annually)
RBI_MPC_DATES = {
    2024: ['2024-02-08', '2024-04-05', '2024-06-07',
           '2024-08-08', '2024-10-09', '2024-12-06'],
    2025: ['2025-02-07', '2025-04-09', '2025-06-06',
           '2025-08-06', '2025-10-08', '2025-12-05'],
    2026: ['2026-02-06', '2026-04-09', '2026-06-05',
           '2026-08-06', '2026-10-08', '2026-12-04'],
}


class NSEStructuralSignals:

    def compute(self, today: date, df: pd.DataFrame = None) -> List[Dict]:
        """
        Compute all NSE structural signals for today.
        df: recent market data (optional, for FII/gap signals)
        """
        signals = []

        # Signal 1: Late Month Bias
        signals.extend(self._late_month_bias(today))

        # Signal 2: Expiry Week Effect
        signals.extend(self._expiry_week_effect(today))

        # Signal 3: March / Nov-Dec Seasonality
        signals.extend(self._monthly_seasonality(today))

        # Signal 4: RBI MPC Drift
        signals.extend(self._rbi_mpc_drift(today))

        # Signal 5: Gap Fill (if data available)
        if df is not None and len(df) >= 2:
            signals.extend(self._gap_fill(today, df))

        return signals

    def _late_month_bias(self, today: date) -> list:
        """
        SSRN IND_003 (p < 2.2e-16): Nifty peaks days 21-31.
        Enter LONG day 18-19, exit day 28-29.
        """
        signals = []
        dom = today.day

        if dom in (18, 19):
            signals.append({
                'signal_id': 'NSE_LATE_MONTH_ENTRY',
                'direction': 'LONG',
                'confidence': 0.72,
                'source': 'SSRN_IND_003',
                'hold_days': 10,
                'stop_pct': 0.015,
            })

        if dom in (28, 29):
            signals.append({
                'signal_id': 'NSE_LATE_MONTH_EXIT',
                'direction': 'EXIT',
                'confidence': 0.72,
                'source': 'SSRN_IND_003',
            })

        return signals

    def _expiry_week_effect(self, today: date) -> list:
        """
        SSRN IND_005: Elevated volatility near expiry.
        Reduce position size to 60% during expiry week.
        """
        if self.is_expiry_week(today):
            return [{
                'signal_id': 'NSE_EXPIRY_WEEK',
                'direction': 'REDUCE_SIZE',
                'size_scalar': 0.60,
                'confidence': 0.80,
                'source': 'SSRN_IND_005',
                'reason': 'Elevated vol near expiry',
            }]
        return []

    def _monthly_seasonality(self, today: date) -> list:
        """
        SSRN IND_006: March negative, Nov-Dec positive.
        """
        signals = []
        month = today.month
        dom = today.day

        if month == 3 and dom == 1:
            signals.append({
                'signal_id': 'NSE_MARCH_EFFECT',
                'direction': 'SHORT',
                'confidence': 0.62,
                'source': 'SSRN_IND_006',
                'hold_days': 20,
                'stop_pct': 0.02,
            })

        if month == 11 and dom == 1:
            signals.append({
                'signal_id': 'NSE_NOVDEC_ENTRY',
                'direction': 'LONG',
                'confidence': 0.65,
                'source': 'SSRN_IND_006',
                'hold_days': 45,
                'stop_pct': 0.02,
            })

        return signals

    def _rbi_mpc_drift(self, today: date) -> list:
        """
        Pre-MPC upward drift: enter LONG 3 days before, exit on day.
        """
        signals = []
        mpc_dates = self.get_rbi_mpc_dates(today.year)

        for mpc_date in mpc_dates:
            days_to = (mpc_date - today).days

            if days_to == 3:
                signals.append({
                    'signal_id': 'NSE_RBI_MPC_DRIFT',
                    'direction': 'LONG',
                    'confidence': 0.62,
                    'source': 'NSE_STRUCTURAL',
                    'hold_days': 3,
                    'stop_pct': 0.015,
                    'note': f'MPC on {mpc_date}',
                })

            if days_to == 0:
                signals.append({
                    'signal_id': 'NSE_RBI_MPC_EXIT',
                    'direction': 'EXIT',
                    'confidence': 0.80,
                    'source': 'NSE_STRUCTURAL',
                })

        return signals

    def _gap_fill(self, today: date, df: pd.DataFrame) -> list:
        """
        Large overnight gaps (>0.8%) fill 63% of the time.
        """
        signals = []
        if len(df) < 2:
            return signals

        prev_close = float(df.iloc[-2]['close'])
        today_open = float(df.iloc[-1]['open'])
        gap_pct = (today_open - prev_close) / prev_close

        if gap_pct > 0.008:
            signals.append({
                'signal_id': 'NSE_GAP_FILL_SHORT',
                'direction': 'SHORT',
                'confidence': 0.63,
                'source': 'NSE_STRUCTURAL',
                'hold_days': 1,
                'stop_pct': 0.008,
                'note': f'Gap up {gap_pct:.2%}',
            })

        elif gap_pct < -0.008:
            signals.append({
                'signal_id': 'NSE_GAP_FILL_LONG',
                'direction': 'LONG',
                'confidence': 0.63,
                'source': 'NSE_STRUCTURAL',
                'hold_days': 1,
                'stop_pct': 0.008,
                'note': f'Gap down {gap_pct:.2%}',
            })

        return signals

    @staticmethod
    def is_expiry_week(d: date) -> bool:
        """True if date falls in monthly expiry week (Mon-Thu of last Thursday)."""
        year, month = d.year, d.month
        last_day = calendar.monthrange(year, month)[1]
        last_thursday = max(
            day for day in range(1, last_day + 1)
            if date(year, month, day).weekday() == 3
        )
        expiry_date = date(year, month, last_thursday)
        expiry_monday = expiry_date - timedelta(days=3)
        return expiry_monday <= d <= expiry_date

    @staticmethod
    def get_rbi_mpc_dates(year: int) -> list:
        """Get RBI MPC announcement dates for given year."""
        dates_str = RBI_MPC_DATES.get(year, [])
        return [date.fromisoformat(d) for d in dates_str]

    def get_daily_summary(self, today: date) -> str:
        """One-line summary for Telegram digest."""
        signals = self.compute(today)
        if not signals:
            return "No structural signals today"

        active = [s['signal_id'].replace('NSE_', '') for s in signals
                  if s['direction'] not in ('EXIT', 'REDUCE_SIZE')]
        modifiers = [s['signal_id'].replace('NSE_', '') for s in signals
                     if s['direction'] in ('EXIT', 'REDUCE_SIZE')]

        parts = []
        if active:
            parts.append(f"Active: {', '.join(active)}")
        if modifiers:
            parts.append(f"Modifiers: {', '.join(modifiers)}")

        return ' | '.join(parts) if parts else "No structural signals today"
