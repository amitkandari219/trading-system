"""
RBI Monetary Policy Announcement Signal.

Exploits the predictable IV cycle around Reserve Bank of India (RBI)
monetary policy announcements, held 6 times per year (roughly bi-monthly:
Feb, Apr, Jun, Aug, Oct, Dec).

Signal logic:
    Phase 1 — PRE-POLICY (T-3 to T-1):
        IV expands as market prices in rate decision uncertainty.
        Signal: NEUTRAL overlay — reduce directional exposure.

    Phase 2 — POLICY DAY (T+0):
        Rate cut   → LONG  (bullish for equities, lower cost of capital)
        Rate hike  → SHORT (bearish, tighter liquidity)
        Hold (exp) → IV crush, direction neutral

        Proxy: Use day's close vs open to infer market reaction.
        Strong move up → likely cut / dovish hold → LONG
        Strong move down → likely hike / hawkish hold → SHORT

    Phase 3 — POST-POLICY (T+1):
        IV crush → premium sellers profit.
        Direction follows policy-day trend continuation.

    Filters:
        - india_vix column used for IV expansion detection
        - close, open columns required

Walk-forward parameters exposed as class constants.

Usage:
    from signals.seasonality.rbi_policy_day import RBIPolicyDaySignal

    sig = RBIPolicyDaySignal()
    result = sig.evaluate(df, date(2026, 4, 9))
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

SIGNAL_ID = 'RBI_POLICY_DAY'

# Historical RBI monetary policy announcement dates
# RBI publishes schedule in advance; these are actual/projected dates
RBI_POLICY_DATES: List[date] = [
    # 2023
    date(2023, 2, 8), date(2023, 4, 6), date(2023, 6, 8),
    date(2023, 8, 10), date(2023, 10, 6), date(2023, 12, 8),
    # 2024
    date(2024, 2, 8), date(2024, 4, 5), date(2024, 6, 7),
    date(2024, 8, 8), date(2024, 10, 9), date(2024, 12, 6),
    # 2025
    date(2025, 2, 7), date(2025, 4, 9), date(2025, 6, 6),
    date(2025, 8, 8), date(2025, 10, 8), date(2025, 12, 5),
    # 2026 (projected)
    date(2026, 2, 6), date(2026, 4, 9), date(2026, 6, 5),
    date(2026, 8, 7), date(2026, 10, 9), date(2026, 12, 4),
]

# Pre-policy window
PRE_POLICY_DAYS = 3
PRE_POLICY_IV_THRESHOLD = 1.10   # VIX 10%+ above 20-day mean

# Day-of thresholds
POLICY_MOVE_THRESHOLD_PCT = 0.3  # Minimum move to infer direction
CUT_PROXY_THRESHOLD_PCT = 0.5   # Open→close > 0.5% ≈ rate cut reaction
HIKE_PROXY_THRESHOLD_PCT = -0.5 # Open→close < -0.5% ≈ rate hike reaction
DAY_OF_BASE_STRENGTH = 0.65

# Post-policy
POST_POLICY_STRENGTH = 0.55

# Strength bounds
MIN_STRENGTH = 0.10
MAX_STRENGTH = 0.90
PRE_POLICY_STRENGTH = 0.50


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


def _find_nearest_policy_date(td: date, direction: str = 'both') -> Optional[date]:
    """Find the nearest RBI policy date to td.

    direction: 'future' (only future), 'past' (only past), 'both'.
    """
    best = None
    best_dist = float('inf')
    for pd_date in RBI_POLICY_DATES:
        dist = (pd_date - td).days
        if direction == 'future' and dist < 0:
            continue
        if direction == 'past' and dist > 0:
            continue
        if abs(dist) < best_dist:
            best_dist = abs(dist)
            best = pd_date
    return best


# ================================================================
# SIGNAL CLASS
# ================================================================

class RBIPolicyDaySignal:
    """
    RBI Monetary Policy announcement signal for Nifty.

    Fires across 3 phases:
        - PRE_POLICY (T-3 to T-1): IV expansion → reduce directional exposure
        - POLICY_DAY (T+0): direction based on market reaction
        - POST_POLICY (T+1): IV crush → trend continuation
    """

    SIGNAL_ID = SIGNAL_ID
    RBI_POLICY_DATES = RBI_POLICY_DATES

    # WF parameters
    PRE_POLICY_DAYS = PRE_POLICY_DAYS
    POLICY_MOVE_THRESHOLD_PCT = POLICY_MOVE_THRESHOLD_PCT
    PRE_POLICY_IV_THRESHOLD = PRE_POLICY_IV_THRESHOLD

    def __init__(self) -> None:
        self._last_fire_date: Optional[date] = None
        logger.info('RBIPolicyDaySignal initialised')

    # ----------------------------------------------------------
    # PUBLIC evaluate
    # ----------------------------------------------------------
    def evaluate(self, df: pd.DataFrame, trade_date: date) -> Optional[Dict]:
        """
        Evaluate RBI Policy Day signal.

        Parameters
        ----------
        df         : DataFrame with columns: date, open, close, india_vix (optional).
        trade_date : The date to evaluate.

        Returns
        -------
        dict with keys: signal_id, direction, strength, price, reason, metadata
        or None if no signal.
        """
        try:
            return self._evaluate_inner(df, trade_date)
        except Exception as e:
            logger.error('RBIPolicyDaySignal.evaluate error: %s', e, exc_info=True)
            return None

    def _evaluate_inner(self, df: pd.DataFrame, trade_date: date) -> Optional[Dict]:
        if df is None or df.empty:
            return None
        if 'date' not in df.columns:
            return None

        td = trade_date

        # Find relevant policy date
        next_policy = _find_nearest_policy_date(td, direction='future')
        prev_policy = _find_nearest_policy_date(td, direction='past')

        # Get current row
        row = df[df['date'] == pd.Timestamp(td)]
        if row.empty:
            return None
        row = row.iloc[-1]

        close = _safe_float(row.get('close'))
        open_price = _safe_float(row.get('open'))
        if math.isnan(close) or close <= 0:
            return None

        # ── Check if trade_date IS a policy date ─────────────────
        if td in RBI_POLICY_DATES:
            return self._policy_day_signal(df, row, td, open_price, close)

        # ── Check PRE-POLICY window ──────────────────────────────
        if next_policy is not None:
            days_before = (next_policy - td).days
            if 1 <= days_before <= self.PRE_POLICY_DAYS:
                return self._pre_policy_signal(df, row, td, close, days_before, next_policy)

        # ── Check POST-POLICY (1 trading day after) ──────────────
        if prev_policy is not None:
            days_after = (td - prev_policy).days
            if 1 <= days_after <= 3:
                # Verify it's the first trading day after policy
                policy_row = df[df['date'] == pd.Timestamp(prev_policy)]
                if not policy_row.empty:
                    after_rows = df[df['date'] > pd.Timestamp(prev_policy)].sort_values('date')
                    if not after_rows.empty and after_rows.iloc[0]['date'] == pd.Timestamp(td):
                        return self._post_policy_signal(df, row, td, close, prev_policy)

        return None

    # ----------------------------------------------------------
    # Phase signals
    # ----------------------------------------------------------
    def _pre_policy_signal(
        self, df: pd.DataFrame, row: pd.Series, td: date,
        close: float, days_before: int, policy_date: date,
    ) -> Optional[Dict]:
        """PRE-POLICY: IV expansion → reduce exposure."""
        strength = PRE_POLICY_STRENGTH

        # Check IV expansion via india_vix
        vix = _safe_float(row.get('india_vix'))
        iv_note = ''
        if not math.isnan(vix) and vix > 0:
            hist = df[df['date'] < pd.Timestamp(td)].tail(20)
            if 'india_vix' in hist.columns and len(hist) >= 10:
                mean_vix = hist['india_vix'].mean()
                if not math.isnan(mean_vix) and mean_vix > 0:
                    iv_ratio = vix / mean_vix
                    if iv_ratio >= self.PRE_POLICY_IV_THRESHOLD:
                        strength = min(MAX_STRENGTH, strength + 0.10)
                        iv_note = f'IV_ratio={iv_ratio:.2f}'
                    else:
                        iv_note = f'IV_ratio={iv_ratio:.2f}(normal)'

        # Closer to policy → stronger
        strength = min(MAX_STRENGTH, strength + (self.PRE_POLICY_DAYS - days_before) * 0.04)

        reason = (
            f"PRE_RBI_POLICY | {days_before}d before {policy_date} | "
            f"IV expansion — reduce exposure | {iv_note}"
        )

        logger.info('%s PRE_POLICY: %s days_before=%d strength=%.3f',
                     self.SIGNAL_ID, td, days_before, strength)

        return {
            'signal_id': self.SIGNAL_ID,
            'direction': 'NEUTRAL',
            'strength': round(max(MIN_STRENGTH, strength), 3),
            'price': round(close, 2),
            'reason': reason.strip(),
            'metadata': {
                'phase': 'PRE_POLICY',
                'days_before_policy': days_before,
                'policy_date': policy_date.isoformat(),
                'overlay': 'REDUCE_SIZE',
            },
        }

    def _policy_day_signal(
        self, df: pd.DataFrame, row: pd.Series, td: date,
        open_price: float, close: float,
    ) -> Optional[Dict]:
        """POLICY DAY: direction from market reaction."""
        if math.isnan(open_price) or open_price <= 0:
            return None

        move_pct = ((close - open_price) / open_price) * 100.0

        # Determine inferred action
        if move_pct >= CUT_PROXY_THRESHOLD_PCT:
            direction = 'LONG'
            inferred = 'RATE_CUT / DOVISH'
        elif move_pct <= HIKE_PROXY_THRESHOLD_PCT:
            direction = 'SHORT'
            inferred = 'RATE_HIKE / HAWKISH'
        elif abs(move_pct) < POLICY_MOVE_THRESHOLD_PCT:
            # Hold (as expected) → IV crush, no directional bias
            direction = 'NEUTRAL'
            inferred = 'HOLD (as expected) — IV crush'
        else:
            # Small directional move
            direction = 'LONG' if move_pct > 0 else 'SHORT'
            inferred = 'MILD_REACTION'

        strength = DAY_OF_BASE_STRENGTH + min(0.20, abs(move_pct) * 0.06)
        strength = min(MAX_STRENGTH, max(MIN_STRENGTH, strength))

        reason = (
            f"RBI_POLICY_DAY | {inferred} | Move={move_pct:+.2f}% | "
            f"Open={open_price:.2f} Close={close:.2f}"
        )

        logger.info('%s POLICY_DAY: %s %s move=%.2f%%',
                     self.SIGNAL_ID, direction, td, move_pct)

        return {
            'signal_id': self.SIGNAL_ID,
            'direction': direction,
            'strength': round(strength, 3),
            'price': round(close, 2),
            'reason': reason,
            'metadata': {
                'phase': 'POLICY_DAY',
                'move_pct': round(move_pct, 4),
                'inferred_action': inferred,
                'open': round(open_price, 2),
                'close': round(close, 2),
            },
        }

    def _post_policy_signal(
        self, df: pd.DataFrame, row: pd.Series, td: date,
        close: float, policy_date: date,
    ) -> Optional[Dict]:
        """POST-POLICY: IV crush → trend continuation."""
        strength = POST_POLICY_STRENGTH

        # Determine prior day's direction for continuation
        policy_row = df[df['date'] == pd.Timestamp(policy_date)]
        direction = 'LONG'  # Default bullish bias
        policy_note = ''
        if not policy_row.empty:
            p_open = _safe_float(policy_row.iloc[-1].get('open'))
            p_close = _safe_float(policy_row.iloc[-1].get('close'))
            if not math.isnan(p_open) and not math.isnan(p_close) and p_open > 0:
                p_move = ((p_close - p_open) / p_open) * 100.0
                direction = 'LONG' if p_move > 0 else 'SHORT'
                policy_note = f'PolicyDay_move={p_move:+.2f}%'

        # Check VIX drop
        vix = _safe_float(row.get('india_vix'))
        if not math.isnan(vix) and not policy_row.empty:
            p_vix = _safe_float(policy_row.iloc[-1].get('india_vix'))
            if not math.isnan(p_vix) and p_vix > 0:
                vix_drop = ((p_vix - vix) / p_vix) * 100.0
                if vix_drop > 5:
                    strength = min(MAX_STRENGTH, strength + 0.08)
                    policy_note += f' VIX_crush={vix_drop:.1f}%'

        reason = (
            f"POST_RBI_POLICY | After {policy_date} | "
            f"IV crush + trend continuation | {policy_note}"
        )

        logger.info('%s POST_POLICY: %s %s strength=%.3f',
                     self.SIGNAL_ID, direction, td, strength)

        return {
            'signal_id': self.SIGNAL_ID,
            'direction': direction,
            'strength': round(max(MIN_STRENGTH, strength), 3),
            'price': round(close, 2),
            'reason': reason.strip(),
            'metadata': {
                'phase': 'POST_POLICY',
                'policy_date': policy_date.isoformat(),
                'signal_type': 'IV_CRUSH_CONTINUATION',
            },
        }

    # ----------------------------------------------------------
    # Utility
    # ----------------------------------------------------------
    def reset(self) -> None:
        """Reset internal state for a fresh backtest run."""
        self._last_fire_date = None

    def __repr__(self) -> str:
        return f"RBIPolicyDaySignal(signal_id='{self.SIGNAL_ID}')"
