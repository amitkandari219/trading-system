"""
Event IV Crush — sell premium before RBI MPC / Budget / FOMC events,
profit from post-event implied-volatility collapse.

The signal enters T-1 (day before event) at 11:00 AM by selling an ATM
straddle hedged with 200-pt wings (iron condor).  It exits on the event
day at a specified time per event type:
    - RBI MPC  → 1 hour post announcement (~11:00 AM)
    - Budget   → 2 hours post speech start (~1:00 PM)
    - FOMC     → next morning 9:30 AM IST (overnight US event)

Entry filters:
    - VIX > 15 (enough premium)
    - VIX risen > 10 % above 5-day average (event-fear priced in)
    - Nifty not moved > 2 % in last 3 trading days
    - ATM straddle premium > 250 pts combined

Interface:
    evaluate(trade_date, vix, vix_5d_avg, nifty_spot,
             atm_straddle_premium, event_type) → dict | None

    backtest_evaluate(trade_date, daily_df, option_chain_df) → dict | None

Usage:
    from signals.structural.event_iv_crush import EventIVCrush
    sig = EventIVCrush()
    result = sig.evaluate(date(2026,2,6), 17.2, 14.8, 24400, 310, 'RBI_MPC')
"""

from __future__ import annotations

import logging
import math
from datetime import date, datetime, time, timedelta
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ================================================================
# Event Calendar — 2026
# ================================================================

RBI_MPC_2026: List[date] = [
    date(2026, 2, 7),
    date(2026, 4, 9),
    date(2026, 6, 5),
    date(2026, 8, 6),
    date(2026, 10, 8),
    date(2026, 12, 5),
]

BUDGET_2026: List[date] = [
    date(2026, 2, 1),
]

FOMC_2026: List[date] = [
    date(2026, 1, 29),
    date(2026, 3, 19),
    date(2026, 5, 7),
    date(2026, 6, 18),
    date(2026, 7, 30),
    date(2026, 9, 17),
    date(2026, 11, 5),
    date(2026, 12, 17),
]

# Merged lookup: event_date → event_type
_ALL_EVENTS: Dict[date, str] = {}
for _d in RBI_MPC_2026:
    _ALL_EVENTS[_d] = "RBI_MPC"
for _d in BUDGET_2026:
    _ALL_EVENTS[_d] = "BUDGET"
for _d in FOMC_2026:
    _ALL_EVENTS[_d] = "FOMC"

# Exit timing per event type (IST)
EVENT_EXIT_TIME: Dict[str, time] = {
    "RBI_MPC": time(11, 0),    # 1 hr post 10 AM announcement
    "BUDGET":  time(13, 0),    # 2 hrs post 11 AM speech
    "FOMC":    time(9, 30),    # next morning (overnight event)
}

# ================================================================
# Constants
# ================================================================

SIGNAL_ID = "EVENT_IV_CRUSH"

STRIKE_INTERVAL = 50
WING_WIDTH = 200                   # iron condor wing width in points
ENTRY_TIME = time(11, 0)           # T-1 entry at 11:00 AM

MIN_VIX = 15.0                     # minimum VIX for adequate premium
VIX_SURGE_PCT = 0.10               # VIX must be > 10 % above 5d avg
MAX_NIFTY_3D_MOVE_PCT = 0.02       # Nifty < 2 % move in 3 trading days
MIN_STRADDLE_PREMIUM = 250         # combined ATM straddle > 250 pts

PROFIT_TARGET_PCT = 0.60           # straddle drops to 60 % of credit (40 % captured)
STOP_LOSS_PCT = 1.30               # straddle reaches 130 % of credit

_RISK_FREE_RATE = 0.07
_TRADING_DAYS_PER_YEAR = 252
_TRADING_HOURS = 6.25

# ================================================================
# Black-Scholes helpers
# ================================================================


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _bs_call(S: float, K: float, T: float, sigma: float, r: float) -> float:
    if T <= 1e-10 or sigma <= 1e-10:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


def _bs_put(S: float, K: float, T: float, sigma: float, r: float) -> float:
    if T <= 1e-10 or sigma <= 1e-10:
        return max(K - S, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def _bs_straddle(S: float, K: float, T: float, sigma: float, r: float) -> float:
    return _bs_call(S, K, T, sigma, r) + _bs_put(S, K, T, sigma, r)


def _bs_iron_condor_credit(
    S: float,
    K_atm: float,
    wing: int,
    T: float,
    sigma: float,
    r: float,
) -> float:
    """
    Net credit of short iron condor:
        sell ATM CE + PE, buy (ATM+wing) CE + (ATM-wing) PE.
    """
    short_straddle = _bs_straddle(S, K_atm, T, sigma, r)
    long_ce = _bs_call(S, K_atm + wing, T, sigma, r)
    long_pe = _bs_put(S, K_atm - wing, T, sigma, r)
    return short_straddle - long_ce - long_pe


# ================================================================
# Calendar helpers
# ================================================================


def get_event_on_date(d: date) -> Optional[str]:
    """Return event type if d is an event day, else None."""
    return _ALL_EVENTS.get(d)


def get_event_on_next_day(d: date) -> Optional[str]:
    """
    If the next calendar day is an event day, return its type.
    For FOMC (overnight), the "entry" day is T-1 in Indian calendar.
    """
    next_day = d + timedelta(days=1)
    return _ALL_EVENTS.get(next_day)


def find_next_event(d: date, max_lookahead: int = 7) -> Optional[tuple]:
    """
    Look ahead up to max_lookahead calendar days for the next event.
    Returns (event_date, event_type) or None.
    """
    for offset in range(1, max_lookahead + 1):
        candidate = d + timedelta(days=offset)
        etype = _ALL_EVENTS.get(candidate)
        if etype is not None:
            return candidate, etype
    return None


# ================================================================
# EventIVCrush Signal
# ================================================================

class EventIVCrush:
    """Sell premium T-1 before scheduled macro events; profit from IV crush."""

    SIGNAL_ID = SIGNAL_ID

    def __init__(
        self,
        min_vix: float = MIN_VIX,
        vix_surge_pct: float = VIX_SURGE_PCT,
        max_3d_move_pct: float = MAX_NIFTY_3D_MOVE_PCT,
        min_premium: float = MIN_STRADDLE_PREMIUM,
        profit_target_pct: float = PROFIT_TARGET_PCT,
        stop_loss_pct: float = STOP_LOSS_PCT,
        wing_width: int = WING_WIDTH,
    ):
        self.min_vix = min_vix
        self.vix_surge_pct = vix_surge_pct
        self.max_3d_move_pct = max_3d_move_pct
        self.min_premium = min_premium
        self.profit_target_pct = profit_target_pct
        self.stop_loss_pct = stop_loss_pct
        self.wing_width = wing_width

    # ----------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------

    @staticmethod
    def _atm_strike(spot: float) -> int:
        return int(round(spot / STRIKE_INTERVAL) * STRIKE_INTERVAL)

    def _passes_filters(
        self,
        trade_date: date,
        vix: float,
        vix_5d_avg: Optional[float],
        nifty_spot: float,
        atm_straddle_premium: float,
        nifty_3d_move_pct: Optional[float] = None,
    ) -> bool:
        """Run all pre-entry filters. Return True if all pass."""
        # VIX floor
        if vix < self.min_vix:
            logger.debug(
                "%s: VIX %.1f < %.1f — skip", trade_date, vix, self.min_vix
            )
            return False

        # VIX surge check
        if vix_5d_avg is not None and vix_5d_avg > 0:
            surge = (vix - vix_5d_avg) / vix_5d_avg
            if surge < self.vix_surge_pct:
                logger.debug(
                    "%s: VIX surge %.1f%% < %.1f%% — skip",
                    trade_date, surge * 100, self.vix_surge_pct * 100,
                )
                return False

        # Nifty 3-day move
        if nifty_3d_move_pct is not None:
            if abs(nifty_3d_move_pct) > self.max_3d_move_pct:
                logger.debug(
                    "%s: 3d Nifty move %.1f%% > %.1f%% — skip",
                    trade_date,
                    abs(nifty_3d_move_pct) * 100,
                    self.max_3d_move_pct * 100,
                )
                return False

        # Premium floor
        if atm_straddle_premium < self.min_premium:
            logger.debug(
                "%s: straddle premium %.1f < %.1f — skip",
                trade_date, atm_straddle_premium, self.min_premium,
            )
            return False

        return True

    # ----------------------------------------------------------------
    # Live evaluate
    # ----------------------------------------------------------------

    def evaluate(
        self,
        trade_date: date,
        vix: float,
        vix_5d_avg: Optional[float],
        nifty_spot: float,
        atm_straddle_premium: float,
        event_type: Optional[str] = None,
        nifty_3d_move_pct: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluate whether to sell premium ahead of a scheduled event.

        Parameters
        ----------
        trade_date : date
            Trading day (should be T-1 relative to event).
        vix : float
            India VIX.
        vix_5d_avg : float or None
            5-day average VIX.
        nifty_spot : float
            Current Nifty spot.
        atm_straddle_premium : float
            Combined ATM CE + PE premium.
        event_type : str, optional
            'RBI_MPC', 'BUDGET', or 'FOMC'.  Auto-detected if None.
        nifty_3d_move_pct : float, optional
            Nifty % change over last 3 trading days (signed).

        Returns
        -------
        dict or None
        """
        # --- Auto-detect event if not provided ---
        if event_type is None:
            result = find_next_event(trade_date, max_lookahead=1)
            if result is None:
                return None
            event_date, event_type = result
        else:
            event_date = trade_date + timedelta(days=1)

        # --- Filters ---
        if not self._passes_filters(
            trade_date, vix, vix_5d_avg, nifty_spot,
            atm_straddle_premium, nifty_3d_move_pct,
        ):
            return None

        atm_strike = self._atm_strike(nifty_spot)
        exit_time = EVENT_EXIT_TIME.get(event_type, time(11, 0))

        total_credit = atm_straddle_premium
        profit_target_value = total_credit * self.profit_target_pct
        stop_loss_value = total_credit * self.stop_loss_pct

        return {
            "signal_id": SIGNAL_ID,
            "trade_date": trade_date,
            "event_date": event_date,
            "event_type": event_type,
            "direction": "SHORT",
            "instrument": "NIFTY_IRON_CONDOR",
            "atm_strike": atm_strike,
            "wing_width": self.wing_width,
            "spot_at_entry": nifty_spot,
            "total_credit": round(total_credit, 2),
            "profit_target_value": round(profit_target_value, 2),
            "stop_loss_value": round(stop_loss_value, 2),
            "entry_time": ENTRY_TIME.isoformat(),
            "exit_time": exit_time.isoformat(),
            "vix": vix,
            "vix_5d_avg": round(vix_5d_avg, 2) if vix_5d_avg else None,
            "legs": [
                {"action": "SELL", "option_type": "CE", "strike": atm_strike},
                {"action": "SELL", "option_type": "PE", "strike": atm_strike},
                {"action": "BUY",  "option_type": "CE", "strike": atm_strike + self.wing_width},
                {"action": "BUY",  "option_type": "PE", "strike": atm_strike - self.wing_width},
            ],
        }

    # ----------------------------------------------------------------
    # Backtest evaluate
    # ----------------------------------------------------------------

    def backtest_evaluate(
        self,
        trade_date: date,
        daily_df: Optional[pd.DataFrame] = None,
        option_chain_df: Optional[pd.DataFrame] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Backtest version — checks if T+1 is an event day, extracts
        premiums from option chain, and simulates IV crush outcome.

        Parameters
        ----------
        trade_date : date
            Candidate entry day (T-1 relative to event).
        daily_df : DataFrame, optional
            Daily OHLCV + VIX with columns: date, close, vix.
            Must include at least 5 prior rows for VIX averaging and
            3 prior rows for Nifty move check.
        option_chain_df : DataFrame, optional
            Option chain with columns: strike, option_type, ltp/close.

        Returns
        -------
        dict or None
        """
        # --- Step 1: check if T+1 is an event day ---
        event_info = find_next_event(trade_date, max_lookahead=1)
        if event_info is None:
            return None
        event_date, event_type = event_info

        # --- Step 2: extract market data from daily_df ---
        spot, vix, vix_5d_avg, nifty_3d_move_pct = self._extract_daily_data(
            trade_date, daily_df
        )
        if spot is None or vix is None:
            logger.debug("%s: missing daily data", trade_date)
            return None

        # --- Step 3: get ATM straddle premium from option chain ---
        atm_strike = self._atm_strike(spot)
        atm_straddle_premium = self._extract_straddle_premium(
            atm_strike, option_chain_df
        )
        if atm_straddle_premium is None:
            logger.debug(
                "%s: could not extract straddle premium for strike %d",
                trade_date, atm_strike,
            )
            return None

        # --- Step 4: run filters ---
        if not self._passes_filters(
            trade_date, vix, vix_5d_avg, spot,
            atm_straddle_premium, nifty_3d_move_pct,
        ):
            return None

        # --- Step 5: simulate outcome ---
        exit_reason, pnl = self._simulate_event_outcome(
            trade_date=trade_date,
            event_date=event_date,
            event_type=event_type,
            spot=spot,
            atm_strike=atm_strike,
            total_credit=atm_straddle_premium,
            vix_entry=vix,
            daily_df=daily_df,
        )

        exit_time = EVENT_EXIT_TIME.get(event_type, time(11, 0))

        return {
            "signal_id": SIGNAL_ID,
            "trade_date": trade_date,
            "event_date": event_date,
            "event_type": event_type,
            "direction": "SHORT",
            "instrument": "NIFTY_IRON_CONDOR",
            "atm_strike": atm_strike,
            "wing_width": self.wing_width,
            "spot_at_entry": spot,
            "total_credit": round(atm_straddle_premium, 2),
            "vix_entry": round(vix, 2),
            "vix_5d_avg": round(vix_5d_avg, 2) if vix_5d_avg else None,
            "nifty_3d_move_pct": (
                round(nifty_3d_move_pct * 100, 2)
                if nifty_3d_move_pct is not None
                else None
            ),
            "exit_reason": exit_reason,
            "pnl_pts": round(pnl, 2),
            "entry_time": ENTRY_TIME.isoformat(),
            "exit_time": exit_time.isoformat(),
        }

    # ----------------------------------------------------------------
    # Backtest internals
    # ----------------------------------------------------------------

    @staticmethod
    def _extract_daily_data(
        trade_date: date,
        daily_df: Optional[pd.DataFrame],
    ) -> tuple:
        """
        Extract spot, vix, vix_5d_avg, nifty_3d_move_pct from daily_df.

        Returns (spot, vix, vix_5d_avg, nifty_3d_move_pct).
        Any value may be None if data is missing.
        """
        if daily_df is None or daily_df.empty:
            return None, None, None, None

        try:
            df = daily_df.copy()

            # Normalise date column
            date_col = "date" if "date" in df.columns else "trade_date"
            if date_col not in df.columns:
                return None, None, None, None

            df[date_col] = pd.to_datetime(df[date_col]).dt.date
            df = df.sort_values(date_col).reset_index(drop=True)

            row_mask = df[date_col] == trade_date
            if not row_mask.any():
                return None, None, None, None

            row_idx = df.index[row_mask][-1]  # latest match
            spot = float(df.loc[row_idx, "close"])

            # VIX
            vix_col = "vix" if "vix" in df.columns else "india_vix"
            vix = None
            if vix_col in df.columns:
                vix = float(df.loc[row_idx, vix_col])

            # VIX 5-day average
            vix_5d_avg = None
            if vix_col in df.columns and row_idx >= 5:
                vix_window = df.loc[row_idx - 5 : row_idx - 1, vix_col]
                vix_5d_avg = float(vix_window.mean())

            # Nifty 3-day move
            nifty_3d_move_pct = None
            if row_idx >= 3:
                close_3d_ago = float(df.loc[row_idx - 3, "close"])
                if close_3d_ago > 0:
                    nifty_3d_move_pct = (spot - close_3d_ago) / close_3d_ago

            return spot, vix, vix_5d_avg, nifty_3d_move_pct

        except (KeyError, IndexError, TypeError, ValueError) as exc:
            logger.warning("Error extracting daily data: %s", exc)
            return None, None, None, None

    @staticmethod
    def _extract_straddle_premium(
        atm_strike: int,
        option_chain_df: Optional[pd.DataFrame],
    ) -> Optional[float]:
        """Get combined ATM CE + PE premium from option chain."""
        if option_chain_df is None or option_chain_df.empty:
            return None

        price_col = "ltp" if "ltp" in option_chain_df.columns else "close"
        if price_col not in option_chain_df.columns:
            return None

        try:
            ce_rows = option_chain_df[
                (option_chain_df["strike"] == atm_strike)
                & (option_chain_df["option_type"].str.upper() == "CE")
            ]
            pe_rows = option_chain_df[
                (option_chain_df["strike"] == atm_strike)
                & (option_chain_df["option_type"].str.upper() == "PE")
            ]

            if ce_rows.empty or pe_rows.empty:
                return None

            ce_prem = float(ce_rows.iloc[0][price_col])
            pe_prem = float(pe_rows.iloc[0][price_col])

            if ce_prem <= 0 or pe_prem <= 0:
                return None

            return ce_prem + pe_prem

        except (KeyError, IndexError, TypeError) as exc:
            logger.warning("Error extracting straddle premium: %s", exc)
            return None

    def _simulate_event_outcome(
        self,
        trade_date: date,
        event_date: date,
        event_type: str,
        spot: float,
        atm_strike: int,
        total_credit: float,
        vix_entry: float,
        daily_df: Optional[pd.DataFrame],
    ) -> tuple:
        """
        Simulate post-event P&L using BS model with IV crush assumption.

        Logic:
        1. If event-day VIX dropped > 5 % from entry → IV crushed,
           straddle drops ~30-40 % → profit.
        2. If Nifty moved > 1.5 % on event day → directional risk,
           evaluate straddle at new spot.
        3. Otherwise, assume modest IV crush of ~20 %.

        Returns (exit_reason, pnl_pts).
        """
        # Try to get event-day data
        event_spot, event_vix = self._get_event_day_data(event_date, daily_df)

        sigma_entry = vix_entry / 100.0
        r = _RISK_FREE_RATE

        # Time to expiry for iron condor — assume nearest weekly expiry
        # T-1 at 11 AM: approximate ~2 trading days to expiry
        T_entry = 2.0 / _TRADING_DAYS_PER_YEAR

        # Event day exit: ~1 trading day to expiry
        T_exit = 1.0 / _TRADING_DAYS_PER_YEAR

        # --- Case 1: we have event-day data ---
        if event_spot is not None and event_vix is not None:
            sigma_exit = event_vix / 100.0
            vix_drop_pct = (vix_entry - event_vix) / vix_entry

            nifty_move_pct = abs(event_spot - spot) / spot

            # Compute BS straddle at exit
            straddle_at_exit = _bs_straddle(
                event_spot, atm_strike, T_exit, sigma_exit, r
            )

            # Iron condor adjustment: long wings reduce both credit and risk
            wing_at_exit = (
                _bs_call(event_spot, atm_strike + self.wing_width, T_exit, sigma_exit, r)
                + _bs_put(event_spot, atm_strike - self.wing_width, T_exit, sigma_exit, r)
            )
            ic_value_at_exit = straddle_at_exit - wing_at_exit

            # Iron condor credit at entry
            ic_credit = _bs_iron_condor_credit(
                spot, atm_strike, self.wing_width, T_entry, sigma_entry, r
            )
            # Use market premium if BS underestimates
            effective_credit = max(total_credit * 0.85, ic_credit)

            # Check SL
            if ic_value_at_exit >= effective_credit * self.stop_loss_pct:
                pnl = effective_credit - (effective_credit * self.stop_loss_pct)
                return "STOP_LOSS", pnl

            # Check profit target
            if ic_value_at_exit <= effective_credit * self.profit_target_pct:
                pnl = effective_credit - ic_value_at_exit
                return "PROFIT_TARGET", pnl

            # Determine exit reason
            if vix_drop_pct > 0.05:
                exit_reason = "IV_CRUSH"
            elif nifty_move_pct > 0.015:
                exit_reason = "DIRECTIONAL_MOVE"
            else:
                exit_reason = "TIME_EXIT"

            pnl = effective_credit - ic_value_at_exit
            return exit_reason, pnl

        # --- Case 2: no event-day data — use heuristic ---
        return self._heuristic_outcome(
            spot, atm_strike, total_credit, sigma_entry, T_entry, T_exit, r
        )

    def _heuristic_outcome(
        self,
        spot: float,
        atm_strike: int,
        total_credit: float,
        sigma_entry: float,
        T_entry: float,
        T_exit: float,
        r: float,
    ) -> tuple:
        """
        When event-day data is unavailable, assume a moderate IV crush
        scenario: VIX drops 15 % and Nifty stays within 0.5 % of entry.
        """
        sigma_exit = sigma_entry * 0.85  # 15 % IV crush
        event_spot = spot  # assume flat

        straddle_at_exit = _bs_straddle(
            event_spot, atm_strike, T_exit, sigma_exit, r
        )
        wing_at_exit = (
            _bs_call(event_spot, atm_strike + self.wing_width, T_exit, sigma_exit, r)
            + _bs_put(event_spot, atm_strike - self.wing_width, T_exit, sigma_exit, r)
        )
        ic_value_at_exit = straddle_at_exit - wing_at_exit

        ic_credit = _bs_iron_condor_credit(
            spot, atm_strike, self.wing_width, T_entry, sigma_entry, r
        )
        effective_credit = max(total_credit * 0.85, ic_credit)

        pnl = effective_credit - ic_value_at_exit
        return "IV_CRUSH_HEURISTIC", pnl

    @staticmethod
    def _get_event_day_data(
        event_date: date,
        daily_df: Optional[pd.DataFrame],
    ) -> tuple:
        """
        Retrieve (spot, vix) for the event day from daily_df.
        Returns (None, None) if unavailable.
        """
        if daily_df is None or daily_df.empty:
            return None, None

        try:
            df = daily_df
            date_col = "date" if "date" in df.columns else "trade_date"
            if date_col not in df.columns:
                return None, None

            dates = pd.to_datetime(df[date_col]).dt.date
            mask = dates == event_date
            if not mask.any():
                return None, None

            row = df.loc[mask].iloc[-1]
            spot = float(row["close"])

            vix_col = "vix" if "vix" in df.columns else "india_vix"
            vix = float(row[vix_col]) if vix_col in df.columns else None

            return spot, vix
        except (KeyError, IndexError, TypeError, ValueError):
            return None, None
