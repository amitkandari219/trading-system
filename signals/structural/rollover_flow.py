"""
Monthly Rollover OI Flow — SCORING signal.

Fires actual trades based on rollover ratio + cost-of-carry + FII positioning
during the last 5 trading days before monthly Nifty expiry.

Distinct from signals/rollover_signal.py which is an OVERLAY that classifies
LONG_BUILDUP/SHORT_BUILDUP for sizing.  This signal produces directional
entries with defined SL/TGT.

Signal logic:
    During the rollover window (T-5 to T-0 before monthly expiry):

    STRONG BULLISH (→ LONG):
        rollover_ratio > 0.75 AND cost_of_carry > 0.3% AND fii_l_s > 1.5

    STRONG BEARISH (→ SHORT):
        rollover_ratio > 0.75 AND cost_of_carry < -0.1% AND fii_l_s < 0.8

    NO TRADE:
        rollover_ratio < 0.60, or ambiguous cost-of-carry, or VIX > 22

    Risk:
        SL: 1.5%   TGT: 2.0%   Hold: 3-5 days

Data sources:
    - nifty_options table: OI by expiry for rollover ratio
    - nifty_futures table: near/next price for cost-of-carry
    - VIX for FII approximation in backtest mode

Usage:
    from signals.structural.rollover_flow import RolloverFlowSignal

    sig = RolloverFlowSignal()

    # Live
    result = sig.evaluate(
        trade_date=date(2026, 3, 24),
        rollover_ratio=0.78,
        cost_of_carry_pct=0.35,
        fii_long_short_ratio=1.6,
        vix=16.5,
        days_to_expiry=3,
    )

    # Backtest
    result = sig.backtest_evaluate(trade_date, daily_df, options_df)
"""

from __future__ import annotations

import calendar
import logging
import math
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ================================================================
# CONSTANTS
# ================================================================

SIGNAL_ID = 'ROLLOVER_FLOW'

# Rollover window
ROLLOVER_WINDOW_DAYS = 5          # Fire only in last 5 trading days before expiry

# Rollover ratio thresholds
ROLLOVER_STRONG = 0.75            # Strong conviction carry-forward
ROLLOVER_WEAK = 0.60              # Below this → no trade

# Cost of carry thresholds (annualised %)
COC_BULLISH = 0.30                # > 0.30% → market willing to pay premium → bullish
COC_BEARISH = -0.10               # < -0.10% → backwardation → bearish

# FII long/short ratio thresholds
FII_BULLISH = 1.50                # FII net long
FII_BEARISH = 0.80                # FII net short

# VIX filter
VIX_MAX = 22.0                    # Skip signal when VIX is elevated

# Risk management
SL_PCT = 1.5                      # Stop loss %
TGT_PCT = 2.0                     # Target %
MIN_HOLD_DAYS = 3                 # Minimum hold period
MAX_HOLD_DAYS = 5                 # Maximum hold period

# Confidence mapping
CONF_STRONG_BULLISH = 0.68
CONF_STRONG_BEARISH = 0.65
CONF_VIX_PENALTY = -0.05          # Per 2 VIX points above 16
CONF_HIGH_ROLLOVER_BOOST = 0.04   # Rollover > 0.85 adds conviction
CONF_FII_EXTREME_BOOST = 0.03     # FII ratio > 2.0 or < 0.5 adds conviction

# Size
BASE_SIZE_MODIFIER = 1.0
MAX_SIZE_MODIFIER = 1.3
MIN_SIZE_MODIFIER = 0.7


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


def _last_thursday(year: int, month: int) -> date:
    """
    Return the last Thursday of the given month/year.
    NSE monthly expiry is the last Thursday of the month.
    """
    # Find last day of month
    _, last_day = calendar.monthrange(year, month)
    d = date(year, month, last_day)
    # Walk backwards to Thursday (weekday 3)
    while d.weekday() != 3:
        d -= timedelta(days=1)
    return d


def _next_monthly_expiry(trade_date: date) -> date:
    """
    Return the next monthly expiry (last Thursday) on or after trade_date.
    If trade_date is past this month's expiry, return next month's.
    """
    expiry = _last_thursday(trade_date.year, trade_date.month)
    if trade_date > expiry:
        # Move to next month
        if trade_date.month == 12:
            expiry = _last_thursday(trade_date.year + 1, 1)
        else:
            expiry = _last_thursday(trade_date.year, trade_date.month + 1)
    return expiry


def _following_monthly_expiry(expiry: date) -> date:
    """Return the monthly expiry after the given one."""
    # Jump forward ~5 weeks to guarantee we land in the next month
    candidate = expiry + timedelta(days=35)
    return _last_thursday(candidate.year, candidate.month)


def _trading_days_between(start: date, end: date) -> int:
    """
    Approximate trading days between two dates (Mon-Fri, no holiday calendar).
    For exact results, a holiday calendar should be used.
    """
    if start > end:
        return 0
    count = 0
    d = start
    while d <= end:
        if d.weekday() < 5:  # Mon-Fri
            count += 1
        d += timedelta(days=1)
    return count


# ================================================================
# MAIN SIGNAL CLASS
# ================================================================

class RolloverFlowSignal:
    """
    Monthly rollover OI flow — SCORING signal.

    Fires directional trades during the rollover window based on:
        - Rollover ratio (near → next month OI migration)
        - Cost of carry (futures premium/discount)
        - FII long/short positioning
        - VIX filter
    """

    SIGNAL_ID = SIGNAL_ID

    def __init__(self, holiday_dates: Optional[List[date]] = None):
        """
        Parameters
        ----------
        holiday_dates : list of date, optional
            Known market holidays.  Used to improve rollover window
            detection.  Falls back to Mon-Fri calendar if not provided.
        """
        self._holidays: set = set(holiday_dates) if holiday_dates else set()

    # ----------------------------------------------------------------
    # PUBLIC — live evaluation
    # ----------------------------------------------------------------

    def evaluate(
        self,
        trade_date: date,
        rollover_ratio: float,
        cost_of_carry_pct: float,
        fii_long_short_ratio: float,
        vix: float,
        days_to_expiry: int,
        nifty_price: Optional[float] = None,
    ) -> Optional[Dict]:
        """
        Evaluate rollover flow for a live or pre-computed snapshot.

        Parameters
        ----------
        trade_date : date
            Current trading date.
        rollover_ratio : float
            Fraction of OI rolled to next month (0.0 – 1.0).
        cost_of_carry_pct : float
            Annualised cost of carry (%) derived from futures spread.
        fii_long_short_ratio : float
            FII index futures long / short ratio.
        vix : float
            India VIX level.
        days_to_expiry : int
            Calendar/trading days to monthly expiry.
        nifty_price : float, optional
            Current Nifty price for SL/TGT computation.

        Returns
        -------
        dict or None
            Signal dict with direction, confidence, SL, TGT, metadata.
            None if no trade.
        """
        # --- Validate inputs ---
        rollover_ratio = _safe_float(rollover_ratio)
        cost_of_carry_pct = _safe_float(cost_of_carry_pct)
        fii_long_short_ratio = _safe_float(fii_long_short_ratio)
        vix = _safe_float(vix)
        nifty_price = _safe_float(nifty_price) if nifty_price is not None else None

        if any(math.isnan(v) for v in [rollover_ratio, cost_of_carry_pct,
                                        fii_long_short_ratio, vix]):
            logger.warning('[%s] Missing input data on %s — skipping', SIGNAL_ID, trade_date)
            return None

        # --- Gate: rollover window ---
        if not self._is_rollover_window(trade_date):
            logger.debug('[%s] %s not in rollover window', SIGNAL_ID, trade_date)
            return None

        # --- Gate: VIX ---
        if vix > VIX_MAX:
            logger.info('[%s] VIX %.1f > %.1f — skipping on %s',
                        SIGNAL_ID, vix, VIX_MAX, trade_date)
            return None

        # --- Gate: weak rollover ---
        if rollover_ratio < ROLLOVER_WEAK:
            logger.info('[%s] Rollover ratio %.2f < %.2f — weak, no trade on %s',
                        SIGNAL_ID, rollover_ratio, ROLLOVER_WEAK, trade_date)
            return None

        # --- Directional classification ---
        direction = None
        base_confidence = 0.0

        if (rollover_ratio >= ROLLOVER_STRONG
                and cost_of_carry_pct > COC_BULLISH
                and fii_long_short_ratio > FII_BULLISH):
            direction = 'LONG'
            base_confidence = CONF_STRONG_BULLISH

        elif (rollover_ratio >= ROLLOVER_STRONG
                and cost_of_carry_pct < COC_BEARISH
                and fii_long_short_ratio < FII_BEARISH):
            direction = 'SHORT'
            base_confidence = CONF_STRONG_BEARISH

        else:
            # Ambiguous / mixed signals → no trade
            logger.info(
                '[%s] Ambiguous on %s: rollover=%.2f coc=%.2f%% fii=%.2f — no trade',
                SIGNAL_ID, trade_date, rollover_ratio,
                cost_of_carry_pct, fii_long_short_ratio,
            )
            return None

        # --- Confidence adjustments ---
        confidence = base_confidence

        # VIX penalty: for every 2 points above 16
        if vix > 16.0:
            penalty_units = (vix - 16.0) / 2.0
            confidence += CONF_VIX_PENALTY * penalty_units

        # Very high rollover boost
        if rollover_ratio > 0.85:
            confidence += CONF_HIGH_ROLLOVER_BOOST

        # Extreme FII positioning boost
        if fii_long_short_ratio > 2.0 or fii_long_short_ratio < 0.5:
            confidence += CONF_FII_EXTREME_BOOST

        confidence = max(0.45, min(0.85, confidence))

        # --- Size modifier ---
        size_mod = BASE_SIZE_MODIFIER
        if rollover_ratio > 0.85:
            size_mod += 0.15
        if (direction == 'LONG' and fii_long_short_ratio > 2.0) or \
           (direction == 'SHORT' and fii_long_short_ratio < 0.5):
            size_mod += 0.10
        size_mod = max(MIN_SIZE_MODIFIER, min(MAX_SIZE_MODIFIER, size_mod))

        # --- SL / TGT ---
        entry_price = nifty_price if nifty_price and not math.isnan(nifty_price) else None
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
        hold_days = MIN_HOLD_DAYS
        if days_to_expiry <= 2:
            hold_days = MIN_HOLD_DAYS  # Roll into next series
        elif days_to_expiry <= MAX_HOLD_DAYS:
            hold_days = days_to_expiry

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
            'rollover_ratio': round(rollover_ratio, 4),
            'cost_of_carry': round(cost_of_carry_pct, 4),
            'fii_ratio': round(fii_long_short_ratio, 4),
            'vix': round(vix, 2),
            'days_to_expiry': days_to_expiry,
        }

    # ----------------------------------------------------------------
    # PUBLIC — backtest evaluation
    # ----------------------------------------------------------------

    def backtest_evaluate(
        self,
        trade_date: date,
        daily_df: pd.DataFrame,
        options_df: pd.DataFrame,
    ) -> Optional[Dict]:
        """
        Evaluate the signal from historical data.

        Parameters
        ----------
        trade_date : date
            The date to evaluate.
        daily_df : pd.DataFrame
            Nifty daily bars with columns: date, open, high, low, close, volume.
            Must contain at least a few rows up to trade_date.
        options_df : pd.DataFrame
            Options OI data with columns: date, expiry, strike, option_type, oi, close.
            Must cover the rollover window.

        Returns
        -------
        dict or None
        """
        if not self._is_rollover_window(trade_date):
            return None

        # --- Validate dataframes ---
        if daily_df is None or daily_df.empty:
            logger.warning('[%s] Empty daily_df for %s', SIGNAL_ID, trade_date)
            return None
        if options_df is None or options_df.empty:
            logger.warning('[%s] Empty options_df for %s', SIGNAL_ID, trade_date)
            return None

        # Normalise date columns
        daily_df = daily_df.copy()
        options_df = options_df.copy()

        if 'date' in daily_df.columns:
            daily_df['date'] = pd.to_datetime(daily_df['date']).dt.date
        if 'date' in options_df.columns:
            options_df['date'] = pd.to_datetime(options_df['date']).dt.date
        if 'expiry' in options_df.columns:
            options_df['expiry'] = pd.to_datetime(options_df['expiry']).dt.date

        # --- Compute rollover ratio ---
        rollover_ratio = self._compute_rollover_ratio(options_df, trade_date)
        if rollover_ratio is None or math.isnan(rollover_ratio):
            logger.info('[%s] Cannot compute rollover ratio for %s', SIGNAL_ID, trade_date)
            return None

        # --- Compute cost of carry ---
        current_expiry = _next_monthly_expiry(trade_date)
        next_expiry = _following_monthly_expiry(current_expiry)
        days_to_expiry = _trading_days_between(trade_date, current_expiry)

        cost_of_carry = self._compute_cost_of_carry_from_options(
            options_df, trade_date, current_expiry, next_expiry, days_to_expiry,
        )
        if cost_of_carry is None:
            # Fallback: try futures price spread from daily_df if available
            cost_of_carry = self._approximate_coc_from_daily(daily_df, trade_date)
        if cost_of_carry is None:
            cost_of_carry = 0.0  # Neutral fallback

        # --- Approximate FII ratio from VIX ---
        vix = self._get_vix(daily_df, trade_date)
        if vix is None:
            vix = 15.0  # Default moderate VIX

        fii_ratio = self._approximate_fii_ratio(vix)

        # --- Nifty price ---
        nifty_price = self._get_nifty_close(daily_df, trade_date)

        return self.evaluate(
            trade_date=trade_date,
            rollover_ratio=rollover_ratio,
            cost_of_carry_pct=cost_of_carry,
            fii_long_short_ratio=fii_ratio,
            vix=vix,
            days_to_expiry=days_to_expiry,
            nifty_price=nifty_price,
        )

    # ----------------------------------------------------------------
    # ROLLOVER WINDOW
    # ----------------------------------------------------------------

    def _is_rollover_window(self, trade_date: date) -> bool:
        """
        Return True if trade_date is within ROLLOVER_WINDOW_DAYS trading days
        of the next monthly expiry.
        """
        expiry = _next_monthly_expiry(trade_date)
        if trade_date > expiry:
            return False

        trading_days = self._trading_days_until(trade_date, expiry)
        return 0 <= trading_days <= ROLLOVER_WINDOW_DAYS

    def _trading_days_until(self, start: date, end: date) -> int:
        """Count trading days from start to end (inclusive of end)."""
        if start > end:
            return -1
        count = 0
        d = start
        while d <= end:
            if d.weekday() < 5 and d not in self._holidays:
                count += 1
            d += timedelta(days=1)
        # Subtract 1 because we want days *until* end, not including start
        return max(0, count - 1)

    # ----------------------------------------------------------------
    # ROLLOVER RATIO
    # ----------------------------------------------------------------

    def _compute_rollover_ratio(
        self,
        options_df: pd.DataFrame,
        trade_date: date,
    ) -> Optional[float]:
        """
        Compute rollover ratio from options OI data.

        rollover_ratio = next_month_total_oi / (current_month_total_oi + next_month_total_oi)

        Parameters
        ----------
        options_df : DataFrame
            Must have columns: date, expiry, oi.
        trade_date : date

        Returns
        -------
        float or None
        """
        try:
            current_expiry = _next_monthly_expiry(trade_date)
            next_expiry = _following_monthly_expiry(current_expiry)

            # Filter to trade_date
            day_data = options_df[options_df['date'] == trade_date]
            if day_data.empty:
                # Try the most recent date before trade_date
                prior = options_df[options_df['date'] <= trade_date]
                if prior.empty:
                    return None
                latest = prior['date'].max()
                day_data = options_df[options_df['date'] == latest]

            # Only keep monthly expiries (not weeklies)
            current_oi = day_data[day_data['expiry'] == current_expiry]['oi'].sum()
            next_oi = day_data[day_data['expiry'] == next_expiry]['oi'].sum()

            total_oi = current_oi + next_oi
            if total_oi <= 0:
                return None

            ratio = next_oi / total_oi
            return round(ratio, 4)

        except Exception as e:
            logger.error('[%s] Error computing rollover ratio: %s', SIGNAL_ID, e)
            return None

    # ----------------------------------------------------------------
    # COST OF CARRY
    # ----------------------------------------------------------------

    @staticmethod
    def _compute_cost_of_carry(
        current_price: float,
        next_price: float,
        days_to_expiry: int,
    ) -> Optional[float]:
        """
        Compute annualised cost of carry (%) from futures price spread.

        CoC = ((next_price - current_price) / current_price) * (365 / days_between) * 100

        Parameters
        ----------
        current_price : float
            Near-month futures price.
        next_price : float
            Next-month futures price.
        days_to_expiry : int
            Trading days between the two expiries.

        Returns
        -------
        float or None
            Annualised cost of carry in percentage.
        """
        current_price = _safe_float(current_price)
        next_price = _safe_float(next_price)

        if (math.isnan(current_price) or math.isnan(next_price)
                or current_price <= 0 or days_to_expiry <= 0):
            return None

        spread_pct = (next_price - current_price) / current_price
        # Approximate days between two monthly expiries (~21-23 trading days ≈ 30 calendar days)
        calendar_days_between = max(days_to_expiry * 1.45, 1)  # rough trading→calendar
        annualised = spread_pct * (365 / calendar_days_between) * 100
        return round(annualised, 4)

    def _compute_cost_of_carry_from_options(
        self,
        options_df: pd.DataFrame,
        trade_date: date,
        current_expiry: date,
        next_expiry: date,
        days_to_expiry: int,
    ) -> Optional[float]:
        """
        Derive cost of carry from ATM futures-equivalent prices in options data.
        Uses ATM call - ATM put ≈ futures price (put-call parity).
        """
        try:
            day_data = options_df[options_df['date'] == trade_date]
            if day_data.empty:
                return None

            current_price = self._synthetic_futures_price(day_data, current_expiry)
            next_price = self._synthetic_futures_price(day_data, next_expiry)

            if current_price is None or next_price is None:
                return None

            days_between = _trading_days_between(current_expiry, next_expiry)
            return self._compute_cost_of_carry(current_price, next_price, days_between)

        except Exception as e:
            logger.debug('[%s] CoC from options failed: %s', SIGNAL_ID, e)
            return None

    @staticmethod
    def _synthetic_futures_price(
        day_data: pd.DataFrame,
        expiry: date,
    ) -> Optional[float]:
        """
        Approximate futures price using ATM options.
        For ATM strike: F ≈ strike + call_price - put_price  (put-call parity).
        """
        try:
            exp_data = day_data[day_data['expiry'] == expiry]
            if exp_data.empty:
                return None

            # Normalise option_type
            exp_data = exp_data.copy()
            exp_data['option_type'] = exp_data['option_type'].astype(str).str.upper().str.strip()
            exp_data['option_type'] = exp_data['option_type'].replace({
                'PUT': 'PE', 'P': 'PE', 'CALL': 'CE', 'C': 'CE',
            })

            # Find strikes with both CE and PE
            ce = exp_data[exp_data['option_type'] == 'CE'].set_index('strike')
            pe = exp_data[exp_data['option_type'] == 'PE'].set_index('strike')
            common_strikes = ce.index.intersection(pe.index)

            if common_strikes.empty:
                return None

            # Pick the ATM-ish strike (where CE close ≈ PE close)
            diffs = []
            for s in common_strikes:
                ce_close = _safe_float(ce.loc[s, 'close'] if isinstance(ce.loc[s], pd.Series)
                                       else ce.loc[s, 'close'].iloc[0])
                pe_close = _safe_float(pe.loc[s, 'close'] if isinstance(pe.loc[s], pd.Series)
                                       else pe.loc[s, 'close'].iloc[0])
                if math.isnan(ce_close) or math.isnan(pe_close):
                    continue
                diffs.append((abs(ce_close - pe_close), s, ce_close, pe_close))

            if not diffs:
                return None

            diffs.sort(key=lambda x: x[0])
            _, atm_strike, ce_price, pe_price = diffs[0]

            futures_price = atm_strike + ce_price - pe_price
            return round(futures_price, 2)

        except Exception:
            return None

    def _approximate_coc_from_daily(
        self,
        daily_df: pd.DataFrame,
        trade_date: date,
    ) -> Optional[float]:
        """
        Fallback: approximate CoC from recent Nifty daily returns.
        Positive trend → assume positive CoC; negative → negative CoC.
        """
        try:
            recent = daily_df[daily_df['date'] <= trade_date].tail(10)
            if len(recent) < 5:
                return None

            first_close = _safe_float(recent.iloc[0]['close'])
            last_close = _safe_float(recent.iloc[-1]['close'])

            if math.isnan(first_close) or math.isnan(last_close) or first_close <= 0:
                return None

            # 10-day return as rough CoC proxy
            ret = (last_close - first_close) / first_close * 100
            # Scale: 10-day return → approximate annualised CoC
            return round(ret * 0.5, 4)  # dampened

        except Exception:
            return None

    # ----------------------------------------------------------------
    # VIX / FII APPROXIMATION
    # ----------------------------------------------------------------

    @staticmethod
    def _get_vix(daily_df: pd.DataFrame, trade_date: date) -> Optional[float]:
        """Extract VIX from daily_df if column exists."""
        if 'vix' not in daily_df.columns and 'india_vix' not in daily_df.columns:
            return None
        vix_col = 'vix' if 'vix' in daily_df.columns else 'india_vix'
        row = daily_df[daily_df['date'] == trade_date]
        if row.empty:
            row = daily_df[daily_df['date'] <= trade_date].tail(1)
        if row.empty:
            return None
        val = _safe_float(row.iloc[0][vix_col])
        return val if not math.isnan(val) else None

    @staticmethod
    def _get_nifty_close(daily_df: pd.DataFrame, trade_date: date) -> Optional[float]:
        """Get Nifty close price for trade_date."""
        row = daily_df[daily_df['date'] == trade_date]
        if row.empty:
            row = daily_df[daily_df['date'] <= trade_date].tail(1)
        if row.empty:
            return None
        val = _safe_float(row.iloc[0]['close'])
        return val if not math.isnan(val) else None

    @staticmethod
    def _approximate_fii_ratio(vix: float) -> float:
        """
        Approximate FII long/short ratio from VIX.

        Heuristic (backtested correlation):
            - Low VIX (< 13)  → FII likely net long  → ratio ~1.8
            - Normal VIX (13-17) → balanced           → ratio ~1.2
            - High VIX (17-22) → FII reducing longs   → ratio ~0.9
            - Very high VIX (>22) → FII net short      → ratio ~0.6

        This is a crude approximation for backtest use only.
        Live mode should use actual NSE participant data.
        """
        if vix < 13:
            return 1.8
        elif vix < 15:
            return 1.5
        elif vix < 17:
            return 1.2
        elif vix < 19:
            return 0.95
        elif vix < 22:
            return 0.75
        else:
            return 0.60

    # ----------------------------------------------------------------
    # UTILITY
    # ----------------------------------------------------------------

    def get_next_rollover_window(self, as_of: date) -> Dict:
        """
        Return info about the next rollover window relative to as_of.

        Returns
        -------
        dict with keys: expiry, window_start (approx), days_until_expiry,
                        in_window (bool).
        """
        expiry = _next_monthly_expiry(as_of)
        days_until = _trading_days_between(as_of, expiry)
        in_window = self._is_rollover_window(as_of)

        # Approximate window start (~5 trading days before expiry ≈ 7 calendar days)
        window_start = expiry - timedelta(days=7)
        while window_start.weekday() >= 5:
            window_start += timedelta(days=1)

        return {
            'expiry': expiry,
            'window_start_approx': window_start,
            'days_until_expiry': days_until,
            'in_window': in_window,
        }

    def __repr__(self) -> str:
        return f'RolloverFlowSignal(signal_id={SIGNAL_ID!r})'
