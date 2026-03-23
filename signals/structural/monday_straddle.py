"""
Monday ATM Straddle Short — 0DTE theta capture on Mondays.

Nifty weekly options expire on TUESDAY (post-Sept 2025 rule change).
This signal sells the ATM straddle on Monday at 11:00 AM and exits by
2:30 PM Monday, capturing intraday theta decay on the 0DTE Tuesday
expiry contract.

Entry rules:
    - Only fires on Monday (weekday == 0). If Monday is a holiday, fire
      on the preceding Friday.
    - ATM strike = round(spot / 50) * 50
    - SELL ATM CE + PE → collect total_credit
    - VIX < 20, no overnight gap > 1 %, straddle premium > 150 pts
    - Block day before RBI policy / Budget (is_event_day flag)

Exit rules:
    - Profit target: straddle value drops to 40 % of credit (captured 60 %)
    - Stop loss: straddle value reaches 150 % of credit
    - Directional protection: Nifty moves > 1 % from entry → close losing leg
    - Time exit: 2:30 PM Monday

Interface:
    evaluate(trade_date, nifty_spot, atm_ce_premium, atm_pe_premium,
             vix, is_event_day) → dict | None

    backtest_evaluate(trade_date, spot, vix, option_chain_df) → dict | None

Usage:
    from signals.structural.monday_straddle import MondayStraddle
    sig = MondayStraddle()
    result = sig.evaluate(date.today(), 24500, 95, 88, 16.5, False)
"""

from __future__ import annotations

import logging
import math
from datetime import date, datetime, time, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ================================================================
# Constants
# ================================================================

SIGNAL_ID = "MONDAY_STRADDLE"

STRIKE_INTERVAL = 50
ENTRY_TIME = time(11, 0)           # 11:00 AM IST
EXIT_TIME = time(14, 30)           # 2:30 PM IST
PROFIT_TARGET_PCT = 0.40           # straddle value drops to 40 % of credit
STOP_LOSS_PCT = 1.50               # straddle reaches 150 % of credit
DIRECTIONAL_MOVE_PCT = 0.01        # 1 % Nifty move triggers leg closure
MIN_VIX = 0.0                      # no lower bound (we cap at upper)
MAX_VIX = 20.0                     # VIX must be below 20
MIN_COMBINED_PREMIUM = 150         # straddle premium > 150 pts
MAX_OVERNIGHT_GAP_PCT = 0.01       # 1 % gap filter
TRADING_HOURS = 6.25               # 9:15 AM to 3:30 PM = 6.25 hrs
ENTRY_HOUR_OFFSET = 1.75           # 11:00 AM is 1.75 hrs after open
REMAINING_HOURS_AT_ENTRY = TRADING_HOURS - ENTRY_HOUR_OFFSET  # ~4.5 hrs
TIME_EXIT_OFFSET = 5.25            # 2:30 PM = 5.25 hrs after open

# ================================================================
# Black-Scholes helpers (for backtest theta decay approximation)
# ================================================================

_RISK_FREE_RATE = 0.07             # annualised
_TRADING_DAYS_PER_YEAR = 252


def _norm_cdf(x: float) -> float:
    """Standard normal CDF via error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _bs_call(S: float, K: float, T: float, sigma: float, r: float) -> float:
    """European call price via Black-Scholes. Returns 0 if T <= 0."""
    if T <= 1e-10 or sigma <= 1e-10:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


def _bs_put(S: float, K: float, T: float, sigma: float, r: float) -> float:
    """European put price via Black-Scholes. Returns 0 if T <= 0."""
    if T <= 1e-10 or sigma <= 1e-10:
        return max(K - S, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def _bs_straddle(S: float, K: float, T: float, sigma: float, r: float) -> float:
    """ATM straddle price (call + put)."""
    return _bs_call(S, K, T, sigma, r) + _bs_put(S, K, T, sigma, r)


# ================================================================
# MondayStraddle Signal
# ================================================================

class MondayStraddle:
    """Monday ATM straddle short — 0DTE theta capture on Nifty weeklies."""

    SIGNAL_ID = SIGNAL_ID

    def __init__(
        self,
        max_vix: float = MAX_VIX,
        min_premium: float = MIN_COMBINED_PREMIUM,
        profit_target_pct: float = PROFIT_TARGET_PCT,
        stop_loss_pct: float = STOP_LOSS_PCT,
        directional_move_pct: float = DIRECTIONAL_MOVE_PCT,
        max_gap_pct: float = MAX_OVERNIGHT_GAP_PCT,
    ):
        self.max_vix = max_vix
        self.min_premium = min_premium
        self.profit_target_pct = profit_target_pct
        self.stop_loss_pct = stop_loss_pct
        self.directional_move_pct = directional_move_pct
        self.max_gap_pct = max_gap_pct

    # ----------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------

    @staticmethod
    def _atm_strike(spot: float) -> int:
        """Round spot to nearest STRIKE_INTERVAL multiple."""
        return int(round(spot / STRIKE_INTERVAL) * STRIKE_INTERVAL)

    @staticmethod
    def _is_monday(d: date) -> bool:
        return d.weekday() == 0

    @staticmethod
    def _is_friday(d: date) -> bool:
        return d.weekday() == 4

    def _is_valid_day(self, trade_date: date, is_event_day: bool) -> bool:
        """
        Valid if Monday, or Friday (acting as Monday substitute when
        Monday is a holiday).  Always blocked on event days.
        """
        if is_event_day:
            logger.debug("%s: blocked — event day", trade_date)
            return False
        if self._is_monday(trade_date):
            return True
        # Friday fires only when caller explicitly routes here
        # because the following Monday is a holiday.
        if self._is_friday(trade_date):
            return True
        return False

    def _check_overnight_gap(
        self, prev_close: Optional[float], current_open: Optional[float]
    ) -> bool:
        """Return True if gap is within limits (or data unavailable)."""
        if prev_close is None or current_open is None:
            return True  # missing data → allow (fail open)
        gap = abs(current_open - prev_close) / prev_close
        if gap > self.max_gap_pct:
            logger.debug("Overnight gap %.2f%% exceeds limit", gap * 100)
            return False
        return True

    # ----------------------------------------------------------------
    # Live evaluate
    # ----------------------------------------------------------------

    def evaluate(
        self,
        trade_date: date,
        nifty_spot: float,
        atm_ce_premium: float,
        atm_pe_premium: float,
        vix: float,
        is_event_day: bool = False,
        prev_close: Optional[float] = None,
        current_open: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluate whether to fire Monday straddle short.

        Parameters
        ----------
        trade_date : date
            Trading day.
        nifty_spot : float
            Current Nifty spot price.
        atm_ce_premium : float
            ATM call premium in points.
        atm_pe_premium : float
            ATM put premium in points.
        vix : float
            India VIX value.
        is_event_day : bool
            True if day before RBI / Budget.
        prev_close : float, optional
            Previous day Nifty close (for gap check).
        current_open : float, optional
            Today's Nifty open (for gap check).

        Returns
        -------
        dict or None
            Signal dict with entry details, or None if no trade.
        """
        # --- Day filter ---
        if not self._is_valid_day(trade_date, is_event_day):
            return None

        # --- VIX filter ---
        if vix is None or vix >= self.max_vix:
            logger.debug("%s: VIX %.1f >= %.1f — skip", trade_date, vix or 0, self.max_vix)
            return None

        # --- Gap filter ---
        if not self._check_overnight_gap(prev_close, current_open):
            return None

        # --- Premium filter ---
        if atm_ce_premium is None or atm_pe_premium is None:
            logger.debug("%s: missing premiums", trade_date)
            return None

        total_credit = atm_ce_premium + atm_pe_premium
        if total_credit < self.min_premium:
            logger.debug(
                "%s: straddle premium %.1f < %.1f — skip",
                trade_date, total_credit, self.min_premium,
            )
            return None

        atm_strike = self._atm_strike(nifty_spot)
        profit_target_value = total_credit * self.profit_target_pct
        stop_loss_value = total_credit * self.stop_loss_pct
        directional_trigger = nifty_spot * self.directional_move_pct

        return {
            "signal_id": SIGNAL_ID,
            "trade_date": trade_date,
            "direction": "SHORT",
            "instrument": "NIFTY_STRADDLE",
            "atm_strike": atm_strike,
            "spot_at_entry": nifty_spot,
            "ce_premium": atm_ce_premium,
            "pe_premium": atm_pe_premium,
            "total_credit": round(total_credit, 2),
            "profit_target_value": round(profit_target_value, 2),
            "stop_loss_value": round(stop_loss_value, 2),
            "directional_trigger_pts": round(directional_trigger, 2),
            "entry_time": ENTRY_TIME.isoformat(),
            "exit_time": EXIT_TIME.isoformat(),
            "vix": vix,
            "legs": [
                {
                    "action": "SELL",
                    "option_type": "CE",
                    "strike": atm_strike,
                    "premium": atm_ce_premium,
                },
                {
                    "action": "SELL",
                    "option_type": "PE",
                    "strike": atm_strike,
                    "premium": atm_pe_premium,
                },
            ],
        }

    # ----------------------------------------------------------------
    # Backtest evaluate
    # ----------------------------------------------------------------

    def backtest_evaluate(
        self,
        trade_date: date,
        spot: float,
        vix: float,
        option_chain_df: Optional[pd.DataFrame] = None,
        intraday_5min_df: Optional[pd.DataFrame] = None,
        prev_close: Optional[float] = None,
        current_open: Optional[float] = None,
        is_event_day: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Backtest version: simulates Monday straddle entry + outcome.

        Parameters
        ----------
        trade_date : date
            Must be a Monday (or Friday if Monday holiday).
        spot : float
            Nifty spot at ~11:00 AM.
        vix : float
            India VIX.
        option_chain_df : DataFrame, optional
            Option chain with columns: strike, option_type, ltp (or close),
            expiry_date.  Should contain Tuesday expiry.
        intraday_5min_df : DataFrame, optional
            5-min OHLC bars for trade_date with columns: timestamp, close.
            Used to detect >1 % directional moves after 11 AM.
        prev_close : float, optional
            Previous day close for gap filter.
        current_open : float, optional
            Open price for gap filter.
        is_event_day : bool
            Whether day before major event.

        Returns
        -------
        dict or None
            Backtest result with pnl, exit_reason, etc.
        """
        # --- Step 1: check it's Monday ---
        if not self._is_valid_day(trade_date, is_event_day):
            return None

        if vix is None or vix >= self.max_vix:
            return None

        if not self._check_overnight_gap(prev_close, current_open):
            return None

        # --- Step 2: find ATM strike, get CE + PE premiums ---
        atm_strike = self._atm_strike(spot)
        ce_prem, pe_prem = self._extract_premiums(atm_strike, option_chain_df)

        if ce_prem is None or pe_prem is None:
            logger.debug(
                "%s: could not find ATM premiums for strike %d",
                trade_date, atm_strike,
            )
            return None

        total_credit = ce_prem + pe_prem
        if total_credit < self.min_premium:
            return None

        # --- Step 3: simulate theta decay ---
        # Time to expiry at 11:00 AM Monday → Tuesday 3:30 PM
        # = rest of Monday (~4.5 hrs) + full Tuesday (~6.25 hrs) ≈ 10.75 hrs
        # In trading-day fractions: 10.75 / (252 * 6.25) ≈ 0.00683 years
        T_entry = (REMAINING_HOURS_AT_ENTRY + TRADING_HOURS) / (
            _TRADING_DAYS_PER_YEAR * TRADING_HOURS
        )

        # Time to expiry at 2:30 PM Monday → Tuesday 3:30 PM
        # = rest of Monday after 2:30 PM (~1 hr) + full Tuesday (~6.25 hrs) ≈ 7.25 hrs
        remaining_after_exit = 1.0 + TRADING_HOURS  # hrs
        T_exit = remaining_after_exit / (_TRADING_DAYS_PER_YEAR * TRADING_HOURS)

        sigma = vix / 100.0  # annualised vol proxy
        r = _RISK_FREE_RATE

        straddle_at_entry = _bs_straddle(spot, atm_strike, T_entry, sigma, r)
        straddle_at_exit = _bs_straddle(spot, atm_strike, T_exit, sigma, r)

        # --- Step 4: check directional move from 5-min bars ---
        directional_sl_hit = False
        max_move_pct = 0.0
        if intraday_5min_df is not None and not intraday_5min_df.empty:
            max_move_pct = self._check_directional_move(
                spot, intraday_5min_df, trade_date
            )
            if max_move_pct > self.directional_move_pct:
                directional_sl_hit = True

        # --- Step 5: determine outcome ---
        exit_reason, pnl = self._simulate_exit(
            total_credit,
            straddle_at_entry,
            straddle_at_exit,
            directional_sl_hit,
        )

        return {
            "signal_id": SIGNAL_ID,
            "trade_date": trade_date,
            "direction": "SHORT",
            "instrument": "NIFTY_STRADDLE",
            "atm_strike": atm_strike,
            "spot_at_entry": spot,
            "ce_premium": round(ce_prem, 2),
            "pe_premium": round(pe_prem, 2),
            "total_credit": round(total_credit, 2),
            "straddle_at_entry_bs": round(straddle_at_entry, 2),
            "straddle_at_exit_bs": round(straddle_at_exit, 2),
            "exit_reason": exit_reason,
            "pnl_pts": round(pnl, 2),
            "max_move_pct": round(max_move_pct * 100, 2),
            "vix": vix,
            "entry_time": ENTRY_TIME.isoformat(),
            "exit_time": EXIT_TIME.isoformat(),
        }

    # ----------------------------------------------------------------
    # Backtest internals
    # ----------------------------------------------------------------

    @staticmethod
    def _extract_premiums(
        atm_strike: int,
        option_chain_df: Optional[pd.DataFrame],
    ) -> tuple:
        """
        Pull CE and PE LTP for the given ATM strike from the option chain.

        Returns (ce_premium, pe_premium) — either may be None.
        """
        if option_chain_df is None or option_chain_df.empty:
            return None, None

        price_col = "ltp" if "ltp" in option_chain_df.columns else "close"
        if price_col not in option_chain_df.columns:
            return None, None

        ce_prem = None
        pe_prem = None

        try:
            ce_rows = option_chain_df[
                (option_chain_df["strike"] == atm_strike)
                & (option_chain_df["option_type"].str.upper() == "CE")
            ]
            if not ce_rows.empty:
                ce_prem = float(ce_rows.iloc[0][price_col])

            pe_rows = option_chain_df[
                (option_chain_df["strike"] == atm_strike)
                & (option_chain_df["option_type"].str.upper() == "PE")
            ]
            if not pe_rows.empty:
                pe_prem = float(pe_rows.iloc[0][price_col])
        except (KeyError, IndexError, TypeError) as exc:
            logger.warning("Error extracting premiums: %s", exc)

        return ce_prem, pe_prem

    @staticmethod
    def _check_directional_move(
        entry_spot: float,
        intraday_df: pd.DataFrame,
        trade_date: date,
    ) -> float:
        """
        Return the max absolute percentage move from entry_spot seen
        in 5-min bars after 11:00 AM on trade_date.
        """
        try:
            if "timestamp" in intraday_df.columns:
                ts_col = "timestamp"
            elif "datetime" in intraday_df.columns:
                ts_col = "datetime"
            else:
                return 0.0

            df = intraday_df.copy()
            df[ts_col] = pd.to_datetime(df[ts_col])

            entry_dt = datetime.combine(trade_date, ENTRY_TIME)
            exit_dt = datetime.combine(trade_date, EXIT_TIME)
            afternoon = df[(df[ts_col] >= entry_dt) & (df[ts_col] <= exit_dt)]

            if afternoon.empty:
                return 0.0

            price_col = "close" if "close" in afternoon.columns else "last"
            if price_col not in afternoon.columns:
                return 0.0

            prices = afternoon[price_col].astype(float)
            max_move = (prices - entry_spot).abs().max() / entry_spot
            return float(max_move)
        except Exception as exc:
            logger.warning("Error in directional move check: %s", exc)
            return 0.0

    def _simulate_exit(
        self,
        total_credit: float,
        straddle_at_entry: float,
        straddle_at_exit: float,
        directional_sl_hit: bool,
    ) -> tuple:
        """
        Determine exit reason and P&L.

        Uses BS-estimated straddle values to approximate whether
        profit target or stop loss was hit during the session.

        Returns (exit_reason, pnl_pts).
        """
        profit_target_value = total_credit * self.profit_target_pct
        stop_loss_value = total_credit * self.stop_loss_pct

        # Directional SL takes priority (happened intraday)
        if directional_sl_hit:
            # Approximate: when direction breaks, straddle roughly at
            # 120 % of credit (one leg balloons, other shrinks)
            estimated_exit = total_credit * 1.20
            pnl = total_credit - estimated_exit
            return "DIRECTIONAL_SL", round(pnl, 2)

        # Check if straddle dropped enough for profit target
        if straddle_at_exit <= profit_target_value:
            pnl = total_credit - straddle_at_exit
            return "PROFIT_TARGET", round(pnl, 2)

        # Check if straddle expanded to SL level
        if straddle_at_exit >= stop_loss_value:
            pnl = total_credit - stop_loss_value
            return "STOP_LOSS", round(pnl, 2)

        # Time exit at 2:30 PM — settle at BS-estimated value
        pnl = total_credit - straddle_at_exit
        return "TIME_EXIT", round(pnl, 2)
