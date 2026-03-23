"""
Credit Spread Executor — builds and executes sell-premium strategies from L8.

Supports:
  BULL_PUT_SPREAD:  sell higher put, buy lower put → bullish credit
  BEAR_CALL_SPREAD: sell lower call, buy higher call → bearish credit
  IRON_CONDOR:      bear call spread + bull put spread → neutral credit
  SHORT_STRANGLE:   sell OTM call + sell OTM put → neutral, undefined risk

Execution order (REVERSE of debit spreads):
  Entry:  SHORT leg first (collect premium) → LONG leg (define risk)
  Exit:   BUY BACK short leg first → SELL long leg

CRITICAL SAFETY:
  If the long leg fails after the short leg fills, IMMEDIATELY close the
  short leg — NEVER leave a naked short position unhedged.

Usage:
    from execution.credit_spread_executor import CreditSpreadExecutor
    executor = CreditSpreadExecutor(kite_bridge, fill_monitor, paper_mode=True)
    spread = executor.build_bull_put_spread(
        spot=23500, sell_delta=0.20, width=100,
        instrument='NIFTY', vix=18.5, dte=5, lots=1,
    )
    fill = executor.execute_credit_spread(spread)
    exit_fill = executor.exit_credit_spread(spread, fill, 'TARGET')
"""

import logging
import math
import os
import random
import time
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, time as dt_time, timedelta
from typing import Dict, List, Optional, Tuple

from execution.spread_builder import (
    SpreadLeg,
    SpreadOrder,
    NIFTY_LOT_SIZE,
    BANKNIFTY_LOT_SIZE,
    NIFTY_STRIKE_INTERVAL,
    BANKNIFTY_STRIKE_INTERVAL,
    MIN_OI_FOR_SELL,
)
from execution.spread_executor import (
    LegFill,
    SpreadFill,
    BROKERAGE_PER_LOT_PER_LEG,
    STT_PCT,
    SLIPPAGE_PCT,
    LEG2_WAIT_SECONDS,
    EXIT_LEG_TIMEOUT,
)

logger = logging.getLogger(__name__)

EXECUTION_MODE = os.environ.get("EXECUTION_MODE", "PAPER").upper()

# ================================================================
# CREDIT STRATEGY TYPES
# ================================================================

BULL_PUT_SPREAD = "BULL_PUT_SPREAD"
BEAR_CALL_SPREAD = "BEAR_CALL_SPREAD"
IRON_CONDOR = "IRON_CONDOR"
SHORT_STRANGLE = "SHORT_STRANGLE"

FORCE_EXIT_TIME = dt_time(15, 20)


# ================================================================
# MARGIN CALCULATOR
# ================================================================

class MarginCalculator:
    """
    SPAN-style margin estimates for credit strategies.

    These are conservative approximations of NSE SPAN margin requirements.
    Actual SPAN margins depend on the full portfolio and are computed by
    the exchange — use these as pre-trade checks only.
    """

    @staticmethod
    def span_margin_naked(
        premium: float,
        spot: float,
        num_lots: int,
        lot_size: int,
    ) -> float:
        """
        Margin for a naked (uncovered) short option.

        Formula: max(premium + 15% of spot, premium + 10% of strike) × lot_size × lots

        For naked short options, the exchange requires a large margin because
        the risk is theoretically unlimited.

        Args:
            premium:  premium collected per unit
            spot:     current underlying price
            num_lots: number of lots
            lot_size: units per lot

        Returns:
            Estimated SPAN margin in rupees.
        """
        margin_a = (premium + 0.15 * spot) * lot_size * num_lots
        margin_b = (premium + 0.10 * spot) * lot_size * num_lots
        return round(max(margin_a, margin_b), 2)

    @staticmethod
    def span_margin_spread(
        max_loss_per_lot: float,
        num_lots: int,
        lot_size: int,
    ) -> float:
        """
        Margin for a defined-risk credit spread (bull put or bear call).

        For spreads, margin = max possible loss = width - net credit, per lot.

        Args:
            max_loss_per_lot: max loss per unit (width - net credit)
            num_lots:         number of lots
            lot_size:         units per lot

        Returns:
            Estimated SPAN margin in rupees.
        """
        return round(max_loss_per_lot * lot_size * num_lots, 2)

    @staticmethod
    def span_margin_iron_condor(
        put_spread_margin: float,
        call_spread_margin: float,
    ) -> float:
        """
        Margin for an iron condor = max of the two spread margins, NOT the sum.

        Only one side can be in-the-money at expiry, so the exchange charges
        the larger of the two wings.

        Args:
            put_spread_margin:  margin for the bull put spread wing
            call_spread_margin: margin for the bear call spread wing

        Returns:
            Estimated SPAN margin in rupees.
        """
        return round(max(put_spread_margin, call_spread_margin), 2)

    @staticmethod
    def check_margin_available(required: float, equity: float) -> bool:
        """
        Pre-trade margin check: required must be less than 50% of equity.

        We cap at 50% to leave headroom for MTM swings and additional trades.

        Args:
            required: estimated margin requirement
            equity:   current account equity

        Returns:
            True if margin is available, False otherwise.
        """
        return required < 0.50 * equity


# ================================================================
# CREDIT SPREAD EXECUTOR
# ================================================================

class CreditSpreadExecutor:
    """
    Builds and executes credit spread strategies (sell premium).

    In PAPER mode, simulates fills at midpoint with minor slippage.
    In LIVE mode, places real orders leg-by-leg through KiteBridge.

    Execution order is the REVERSE of SpreadExecutor for debit spreads:
      Entry: SHORT leg first → LONG leg
      Exit:  BUY BACK short first → SELL long
    """

    def __init__(
        self,
        kite_bridge=None,
        fill_monitor=None,
        paper_mode: bool = True,
        alerter=None,
    ):
        """
        Args:
            kite_bridge:  KiteBridge instance for order placement
            fill_monitor: FillMonitor instance for fill tracking
            paper_mode:   True for simulated fills
            alerter:      TelegramAlerter instance
        """
        self.bridge = kite_bridge
        self.monitor = fill_monitor
        self.paper_mode = paper_mode or (EXECUTION_MODE == "PAPER")
        self.alerter = alerter
        self.margin_calc = MarginCalculator()

    # ================================================================
    # CONSTRUCTION: Bull Put Spread
    # ================================================================

    def build_bull_put_spread(
        self,
        spot: float,
        sell_delta: float,
        width: int,
        instrument: str,
        vix: float,
        dte: int,
        lots: int = 1,
        atr: Optional[float] = None,
        signal_id: str = "CREDIT_BULL_PUT",
        expiry_date: Optional[str] = None,
    ) -> Optional[SpreadOrder]:
        """
        Build a bull put spread: sell higher-strike put, buy lower-strike put.

        Bullish bias — profits if underlying stays above the short put strike.

        Args:
            spot:       current underlying price
            sell_delta: target delta for the short put (e.g. 0.20 for 20-delta)
            width:      spread width in points (e.g. 100, 200)
            instrument: 'NIFTY' or 'BANKNIFTY'
            vix:        India VIX
            dte:        days to expiry
            lots:       number of lots
            atr:        daily ATR (for delta approximation). If None, estimated.
            signal_id:  signal identifier for logging
            expiry_date: option expiry in 'YYMMDD' format

        Returns:
            SpreadOrder or None if construction fails.
        """
        lot_size = NIFTY_LOT_SIZE if instrument == "NIFTY" else BANKNIFTY_LOT_SIZE
        interval = NIFTY_STRIKE_INTERVAL if instrument == "NIFTY" else BANKNIFTY_STRIKE_INTERVAL

        if atr is None:
            atr = self._estimate_atr(spot, vix)

        # Sell put strike: OTM put below spot
        sell_strike = self._delta_to_strike(
            spot, sell_delta, "PE", instrument, atr
        )
        # Buy put strike: further OTM (lower)
        buy_strike = self._snap_strike(sell_strike - width, instrument)

        # Ensure buy is below sell for a bull put spread
        if buy_strike >= sell_strike:
            buy_strike = sell_strike - interval

        expiry_str = expiry_date or self._next_weekly_expiry()

        # Estimate premiums
        sell_premium = self._estimate_otm_premium(spot, sell_strike, vix, dte, atr)
        buy_premium = self._estimate_otm_premium(spot, buy_strike, vix, dte, atr)

        sell_symbol = f"{instrument}{expiry_str}{sell_strike}PE"
        buy_symbol = f"{instrument}{expiry_str}{buy_strike}PE"

        # Build legs
        sell_leg = SpreadLeg(
            tradingsymbol=sell_symbol,
            instrument=instrument,
            option_type="PE",
            strike=sell_strike,
            transaction_type="SELL",
            premium=round(sell_premium, 2),
            lot_size=lot_size,
            lots=lots,
            quantity=lots * lot_size,
        )

        buy_leg = SpreadLeg(
            tradingsymbol=buy_symbol,
            instrument=instrument,
            option_type="PE",
            strike=buy_strike,
            transaction_type="BUY",
            premium=round(buy_premium, 2),
            lot_size=lot_size,
            lots=lots,
            quantity=lots * lot_size,
        )

        # Credit and risk
        net_credit = sell_premium - buy_premium
        net_credit = max(net_credit, 0.05)
        spread_width_pts = sell_strike - buy_strike
        max_loss_per_unit = spread_width_pts - net_credit
        max_profit_per_unit = net_credit
        quantity = lots * lot_size

        # Breakeven: sell_strike - net_credit (below this → loss)
        breakeven = sell_strike - net_credit

        # Exit thresholds for credit spreads:
        # SL: spread value rises to 2x credit received (losing money)
        # TGT: spread value drops to 20% of credit (captured 80% of premium)
        sl_value = round(net_credit * 2.0, 2)
        tgt_value = round(net_credit * 0.20, 2)

        # Greeks
        net_delta, net_gamma, net_theta, net_vega = self._compute_credit_greeks(
            sell_leg, buy_leg, sell_delta
        )

        spread = SpreadOrder(
            signal_id=signal_id,
            strategy=BULL_PUT_SPREAD,
            direction="SHORT",
            instrument=instrument,
            expiry_str=expiry_str,
            buy_leg=buy_leg,
            sell_leg=sell_leg,
            lots=lots,
            lot_size=lot_size,
            net_debit=round(-net_credit, 2),  # negative = credit received
            max_loss=round(max_loss_per_unit * quantity, 2),
            max_profit=round(max_profit_per_unit * quantity, 2),
            breakeven=round(breakeven, 2),
            sl_value=sl_value,
            tgt_value=tgt_value,
            net_delta=net_delta,
            net_gamma=net_gamma,
            net_theta=net_theta,
            net_vega=net_vega,
            naked_cost=0.0,
            premium_saved=0.0,
            spread_width=spread_width_pts // interval,
            entry_time=datetime.now().strftime("%H:%M:%S"),
            underlying_ltp=spot,
            vix_at_entry=vix,
        )

        # Margin check
        margin_req = self.margin_calc.span_margin_spread(
            max_loss_per_unit, lots, lot_size
        )

        logger.info(
            f"CreditSpread: {BULL_PUT_SPREAD} {signal_id} | "
            f"sell={sell_symbol}@{sell_premium:.2f} buy={buy_symbol}@{buy_premium:.2f} | "
            f"credit={net_credit:.2f} max_loss={spread.max_loss:,.0f} "
            f"max_profit={spread.max_profit:,.0f} margin={margin_req:,.0f} | "
            f"VIX={vix:.1f} DTE={dte}"
        )

        return spread

    # ================================================================
    # CONSTRUCTION: Bear Call Spread
    # ================================================================

    def build_bear_call_spread(
        self,
        spot: float,
        sell_delta: float,
        width: int,
        instrument: str,
        vix: float,
        dte: int,
        lots: int = 1,
        atr: Optional[float] = None,
        signal_id: str = "CREDIT_BEAR_CALL",
        expiry_date: Optional[str] = None,
    ) -> Optional[SpreadOrder]:
        """
        Build a bear call spread: sell lower-strike call, buy higher-strike call.

        Bearish bias — profits if underlying stays below the short call strike.

        Args:
            spot:       current underlying price
            sell_delta: target delta for the short call (e.g. 0.20 for 20-delta)
            width:      spread width in points (e.g. 100, 200)
            instrument: 'NIFTY' or 'BANKNIFTY'
            vix:        India VIX
            dte:        days to expiry
            lots:       number of lots
            atr:        daily ATR. If None, estimated from VIX.
            signal_id:  signal identifier
            expiry_date: expiry in 'YYMMDD' format

        Returns:
            SpreadOrder or None if construction fails.
        """
        lot_size = NIFTY_LOT_SIZE if instrument == "NIFTY" else BANKNIFTY_LOT_SIZE
        interval = NIFTY_STRIKE_INTERVAL if instrument == "NIFTY" else BANKNIFTY_STRIKE_INTERVAL

        if atr is None:
            atr = self._estimate_atr(spot, vix)

        # Sell call strike: OTM call above spot
        sell_strike = self._delta_to_strike(
            spot, sell_delta, "CE", instrument, atr
        )
        # Buy call strike: further OTM (higher)
        buy_strike = self._snap_strike(sell_strike + width, instrument)

        # Ensure buy is above sell for a bear call spread
        if buy_strike <= sell_strike:
            buy_strike = sell_strike + interval

        expiry_str = expiry_date or self._next_weekly_expiry()

        # Estimate premiums
        sell_premium = self._estimate_otm_premium(spot, sell_strike, vix, dte, atr)
        buy_premium = self._estimate_otm_premium(spot, buy_strike, vix, dte, atr)

        sell_symbol = f"{instrument}{expiry_str}{sell_strike}CE"
        buy_symbol = f"{instrument}{expiry_str}{buy_strike}CE"

        sell_leg = SpreadLeg(
            tradingsymbol=sell_symbol,
            instrument=instrument,
            option_type="CE",
            strike=sell_strike,
            transaction_type="SELL",
            premium=round(sell_premium, 2),
            lot_size=lot_size,
            lots=lots,
            quantity=lots * lot_size,
        )

        buy_leg = SpreadLeg(
            tradingsymbol=buy_symbol,
            instrument=instrument,
            option_type="CE",
            strike=buy_strike,
            transaction_type="BUY",
            premium=round(buy_premium, 2),
            lot_size=lot_size,
            lots=lots,
            quantity=lots * lot_size,
        )

        net_credit = sell_premium - buy_premium
        net_credit = max(net_credit, 0.05)
        spread_width_pts = buy_strike - sell_strike
        max_loss_per_unit = spread_width_pts - net_credit
        max_profit_per_unit = net_credit
        quantity = lots * lot_size

        # Breakeven: sell_strike + net_credit (above this → loss)
        breakeven = sell_strike + net_credit

        sl_value = round(net_credit * 2.0, 2)
        tgt_value = round(net_credit * 0.20, 2)

        net_delta, net_gamma, net_theta, net_vega = self._compute_credit_greeks(
            sell_leg, buy_leg, sell_delta
        )

        spread = SpreadOrder(
            signal_id=signal_id,
            strategy=BEAR_CALL_SPREAD,
            direction="SHORT",
            instrument=instrument,
            expiry_str=expiry_str,
            buy_leg=buy_leg,
            sell_leg=sell_leg,
            lots=lots,
            lot_size=lot_size,
            net_debit=round(-net_credit, 2),
            max_loss=round(max_loss_per_unit * quantity, 2),
            max_profit=round(max_profit_per_unit * quantity, 2),
            breakeven=round(breakeven, 2),
            sl_value=sl_value,
            tgt_value=tgt_value,
            net_delta=net_delta,
            net_gamma=net_gamma,
            net_theta=net_theta,
            net_vega=net_vega,
            naked_cost=0.0,
            premium_saved=0.0,
            spread_width=spread_width_pts // interval,
            entry_time=datetime.now().strftime("%H:%M:%S"),
            underlying_ltp=spot,
            vix_at_entry=vix,
        )

        margin_req = self.margin_calc.span_margin_spread(
            max_loss_per_unit, lots, lot_size
        )

        logger.info(
            f"CreditSpread: {BEAR_CALL_SPREAD} {signal_id} | "
            f"sell={sell_symbol}@{sell_premium:.2f} buy={buy_symbol}@{buy_premium:.2f} | "
            f"credit={net_credit:.2f} max_loss={spread.max_loss:,.0f} "
            f"max_profit={spread.max_profit:,.0f} margin={margin_req:,.0f} | "
            f"VIX={vix:.1f} DTE={dte}"
        )

        return spread

    # ================================================================
    # CONSTRUCTION: Iron Condor
    # ================================================================

    def build_iron_condor(
        self,
        spot: float,
        call_delta: float,
        put_delta: float,
        wing_offset: int,
        instrument: str,
        vix: float,
        dte: int,
        lots: int = 1,
        atr: Optional[float] = None,
        signal_id: str = "CREDIT_IRON_CONDOR",
        expiry_date: Optional[str] = None,
    ) -> Optional[SpreadOrder]:
        """
        Build an iron condor: bear call spread + bull put spread.

        Neutral strategy — profits if underlying stays between the two short
        strikes. Defined risk on both sides.

        Args:
            spot:        current underlying price
            call_delta:  target delta for the short call (e.g. 0.20)
            put_delta:   target delta for the short put (e.g. 0.20)
            wing_offset: width of each wing in points (e.g. 100)
            instrument:  'NIFTY' or 'BANKNIFTY'
            vix:         India VIX
            dte:         days to expiry
            lots:        number of lots
            atr:         daily ATR. If None, estimated.
            signal_id:   signal identifier
            expiry_date: expiry in 'YYMMDD' format

        Returns:
            SpreadOrder representing the full iron condor, or None.
        """
        lot_size = NIFTY_LOT_SIZE if instrument == "NIFTY" else BANKNIFTY_LOT_SIZE
        interval = NIFTY_STRIKE_INTERVAL if instrument == "NIFTY" else BANKNIFTY_STRIKE_INTERVAL

        if atr is None:
            atr = self._estimate_atr(spot, vix)

        expiry_str = expiry_date or self._next_weekly_expiry()

        # ── Call side (bear call spread) ──
        sell_call_strike = self._delta_to_strike(spot, call_delta, "CE", instrument, atr)
        buy_call_strike = self._snap_strike(sell_call_strike + wing_offset, instrument)
        if buy_call_strike <= sell_call_strike:
            buy_call_strike = sell_call_strike + interval

        # ── Put side (bull put spread) ──
        sell_put_strike = self._delta_to_strike(spot, put_delta, "PE", instrument, atr)
        buy_put_strike = self._snap_strike(sell_put_strike - wing_offset, instrument)
        if buy_put_strike >= sell_put_strike:
            buy_put_strike = sell_put_strike - interval

        # Estimate premiums for all four legs
        sell_call_prem = self._estimate_otm_premium(spot, sell_call_strike, vix, dte, atr)
        buy_call_prem = self._estimate_otm_premium(spot, buy_call_strike, vix, dte, atr)
        sell_put_prem = self._estimate_otm_premium(spot, sell_put_strike, vix, dte, atr)
        buy_put_prem = self._estimate_otm_premium(spot, buy_put_strike, vix, dte, atr)

        # Net credit = (sell_call - buy_call) + (sell_put - buy_put)
        call_credit = sell_call_prem - buy_call_prem
        put_credit = sell_put_prem - buy_put_prem
        net_credit = call_credit + put_credit
        net_credit = max(net_credit, 0.10)

        # Max loss = wider wing width - net credit
        call_width = buy_call_strike - sell_call_strike
        put_width = sell_put_strike - buy_put_strike
        max_wing_width = max(call_width, put_width)
        max_loss_per_unit = max_wing_width - net_credit
        max_profit_per_unit = net_credit
        quantity = lots * lot_size

        # For SpreadOrder, we use the short PUT as the sell_leg and short CALL info
        # is embedded. The buy_leg is the long PUT (protective).
        # This is a simplification — the SpreadOrder dataclass has two legs,
        # so we represent the iron condor with the put-side legs and log the
        # call-side details.

        sell_put_symbol = f"{instrument}{expiry_str}{sell_put_strike}PE"
        buy_put_symbol = f"{instrument}{expiry_str}{buy_put_strike}PE"
        sell_call_symbol = f"{instrument}{expiry_str}{sell_call_strike}CE"
        buy_call_symbol = f"{instrument}{expiry_str}{buy_call_strike}CE"

        # Primary legs (put side — used for SpreadOrder structure)
        sell_leg = SpreadLeg(
            tradingsymbol=sell_put_symbol,
            instrument=instrument,
            option_type="PE",
            strike=sell_put_strike,
            transaction_type="SELL",
            premium=round(sell_put_prem, 2),
            lot_size=lot_size,
            lots=lots,
            quantity=quantity,
        )

        buy_leg = SpreadLeg(
            tradingsymbol=buy_put_symbol,
            instrument=instrument,
            option_type="PE",
            strike=buy_put_strike,
            transaction_type="BUY",
            premium=round(buy_put_prem, 2),
            lot_size=lot_size,
            lots=lots,
            quantity=quantity,
        )

        # Breakevens: sell_put - net_credit (lower) and sell_call + net_credit (upper)
        lower_be = sell_put_strike - net_credit
        upper_be = sell_call_strike + net_credit

        sl_value = round(net_credit * 2.0, 2)
        tgt_value = round(net_credit * 0.20, 2)

        # Net greeks: short delta from both sides should roughly cancel
        net_delta = 0.0  # iron condor is approximately delta-neutral
        net_gamma = round(-0.002 * 2, 6)  # short gamma on both sides
        net_theta = round(10.0 * 2, 2)    # positive theta from two short legs
        net_vega = round(-5.0 * 2, 2)     # short vega

        spread = SpreadOrder(
            signal_id=signal_id,
            strategy=IRON_CONDOR,
            direction="SHORT",
            instrument=instrument,
            expiry_str=expiry_str,
            buy_leg=buy_leg,
            sell_leg=sell_leg,
            lots=lots,
            lot_size=lot_size,
            net_debit=round(-net_credit, 2),
            max_loss=round(max_loss_per_unit * quantity, 2),
            max_profit=round(max_profit_per_unit * quantity, 2),
            breakeven=round(lower_be, 2),  # lower breakeven (primary)
            sl_value=sl_value,
            tgt_value=tgt_value,
            net_delta=net_delta,
            net_gamma=net_gamma,
            net_theta=net_theta,
            net_vega=net_vega,
            naked_cost=0.0,
            premium_saved=0.0,
            spread_width=max_wing_width // interval,
            entry_time=datetime.now().strftime("%H:%M:%S"),
            underlying_ltp=spot,
            vix_at_entry=vix,
        )

        # Margin for iron condor: max of the two wings
        put_margin = self.margin_calc.span_margin_spread(
            put_width - put_credit, lots, lot_size
        )
        call_margin = self.margin_calc.span_margin_spread(
            call_width - call_credit, lots, lot_size
        )
        margin_req = self.margin_calc.span_margin_iron_condor(put_margin, call_margin)

        logger.info(
            f"CreditSpread: {IRON_CONDOR} {signal_id} | "
            f"sell_call={sell_call_symbol}@{sell_call_prem:.2f} "
            f"buy_call={buy_call_symbol}@{buy_call_prem:.2f} | "
            f"sell_put={sell_put_symbol}@{sell_put_prem:.2f} "
            f"buy_put={buy_put_symbol}@{buy_put_prem:.2f} | "
            f"credit={net_credit:.2f} max_loss={spread.max_loss:,.0f} "
            f"margin={margin_req:,.0f} | BE=[{lower_be:.0f}, {upper_be:.0f}] "
            f"VIX={vix:.1f} DTE={dte}"
        )

        # Store call-side info for execution
        spread._call_sell_leg = SpreadLeg(
            tradingsymbol=sell_call_symbol,
            instrument=instrument,
            option_type="CE",
            strike=sell_call_strike,
            transaction_type="SELL",
            premium=round(sell_call_prem, 2),
            lot_size=lot_size,
            lots=lots,
            quantity=quantity,
        )
        spread._call_buy_leg = SpreadLeg(
            tradingsymbol=buy_call_symbol,
            instrument=instrument,
            option_type="CE",
            strike=buy_call_strike,
            transaction_type="BUY",
            premium=round(buy_call_prem, 2),
            lot_size=lot_size,
            lots=lots,
            quantity=quantity,
        )
        spread._upper_breakeven = round(upper_be, 2)
        spread._margin_required = margin_req

        return spread

    # ================================================================
    # CONSTRUCTION: Short Strangle
    # ================================================================

    def build_short_strangle(
        self,
        spot: float,
        call_delta: float,
        put_delta: float,
        instrument: str,
        vix: float,
        dte: int,
        lots: int = 1,
        atr: Optional[float] = None,
        signal_id: str = "CREDIT_SHORT_STRANGLE",
        expiry_date: Optional[str] = None,
    ) -> Optional[SpreadOrder]:
        """
        Build a short strangle: sell OTM call + sell OTM put (no long legs).

        WARNING: Undefined risk on both sides. Requires large margin.

        Args:
            spot:       current underlying price
            call_delta: target delta for the short call (e.g. 0.15)
            put_delta:  target delta for the short put (e.g. 0.15)
            instrument: 'NIFTY' or 'BANKNIFTY'
            vix:        India VIX
            dte:        days to expiry
            lots:       number of lots
            atr:        daily ATR. If None, estimated.
            signal_id:  signal identifier
            expiry_date: expiry in 'YYMMDD' format

        Returns:
            SpreadOrder or None if construction fails.
        """
        lot_size = NIFTY_LOT_SIZE if instrument == "NIFTY" else BANKNIFTY_LOT_SIZE
        interval = NIFTY_STRIKE_INTERVAL if instrument == "NIFTY" else BANKNIFTY_STRIKE_INTERVAL

        if atr is None:
            atr = self._estimate_atr(spot, vix)

        expiry_str = expiry_date or self._next_weekly_expiry()

        sell_call_strike = self._delta_to_strike(spot, call_delta, "CE", instrument, atr)
        sell_put_strike = self._delta_to_strike(spot, put_delta, "PE", instrument, atr)

        sell_call_prem = self._estimate_otm_premium(spot, sell_call_strike, vix, dte, atr)
        sell_put_prem = self._estimate_otm_premium(spot, sell_put_strike, vix, dte, atr)

        net_credit = sell_call_prem + sell_put_prem
        quantity = lots * lot_size

        sell_call_symbol = f"{instrument}{expiry_str}{sell_call_strike}CE"
        sell_put_symbol = f"{instrument}{expiry_str}{sell_put_strike}PE"

        # Short strangle: sell_leg is the put, buy_leg placeholder is the call
        # (both are short — no long leg for protection)
        sell_leg = SpreadLeg(
            tradingsymbol=sell_put_symbol,
            instrument=instrument,
            option_type="PE",
            strike=sell_put_strike,
            transaction_type="SELL",
            premium=round(sell_put_prem, 2),
            lot_size=lot_size,
            lots=lots,
            quantity=quantity,
        )

        # Use the call as the "buy_leg" slot — but it is also SELL
        buy_leg = SpreadLeg(
            tradingsymbol=sell_call_symbol,
            instrument=instrument,
            option_type="CE",
            strike=sell_call_strike,
            transaction_type="SELL",
            premium=round(sell_call_prem, 2),
            lot_size=lot_size,
            lots=lots,
            quantity=quantity,
        )

        # Breakevens
        upper_be = sell_call_strike + net_credit
        lower_be = sell_put_strike - net_credit

        # SL: net value doubles (loss = credit received)
        sl_value = round(net_credit * 2.0, 2)
        tgt_value = round(net_credit * 0.20, 2)

        # Unlimited risk — use naked margin for both legs, take max
        call_naked_margin = self.margin_calc.span_margin_naked(
            sell_call_prem, spot, lots, lot_size
        )
        put_naked_margin = self.margin_calc.span_margin_naked(
            sell_put_prem, spot, lots, lot_size
        )
        # Strangle margin: higher of the two naked margins + the other premium
        margin_req = max(call_naked_margin, put_naked_margin) + min(
            sell_call_prem, sell_put_prem
        ) * lot_size * lots

        spread = SpreadOrder(
            signal_id=signal_id,
            strategy=SHORT_STRANGLE,
            direction="SHORT",
            instrument=instrument,
            expiry_str=expiry_str,
            buy_leg=buy_leg,      # actually short call (no protective leg)
            sell_leg=sell_leg,     # short put
            lots=lots,
            lot_size=lot_size,
            net_debit=round(-net_credit, 2),
            max_loss=round(margin_req, 2),  # undefined, use margin as proxy
            max_profit=round(net_credit * quantity, 2),
            breakeven=round(lower_be, 2),
            sl_value=sl_value,
            tgt_value=tgt_value,
            net_delta=0.0,
            net_gamma=round(-0.002 * 2, 6),
            net_theta=round(10.0 * 2, 2),
            net_vega=round(-5.0 * 2, 2),
            naked_cost=0.0,
            premium_saved=0.0,
            spread_width=0,
            entry_time=datetime.now().strftime("%H:%M:%S"),
            underlying_ltp=spot,
            vix_at_entry=vix,
        )

        spread._upper_breakeven = round(upper_be, 2)
        spread._margin_required = margin_req

        logger.info(
            f"CreditSpread: {SHORT_STRANGLE} {signal_id} | "
            f"sell_call={sell_call_symbol}@{sell_call_prem:.2f} "
            f"sell_put={sell_put_symbol}@{sell_put_prem:.2f} | "
            f"credit={net_credit:.2f} margin={margin_req:,.0f} | "
            f"BE=[{lower_be:.0f}, {upper_be:.0f}] VIX={vix:.1f} DTE={dte}"
        )

        return spread

    # ================================================================
    # EXECUTION: Entry (SHORT leg first, then LONG leg)
    # ================================================================

    def execute_credit_spread(
        self,
        spread: SpreadOrder,
        equity: Optional[float] = None,
    ) -> SpreadFill:
        """
        Execute a credit spread: SHORT leg first, then LONG leg.

        CRITICAL SAFETY: If the long leg fails after the short leg fills,
        IMMEDIATELY close the short leg to prevent naked exposure.

        Args:
            spread:  SpreadOrder from build_* methods
            equity:  current account equity for margin check

        Returns:
            SpreadFill with fill details and actual costs.
        """
        signal_id = spread.signal_id
        strategy = spread.strategy
        logger.info(
            f"Executing credit spread {strategy} for {signal_id}: "
            f"sell={spread.sell_leg.tradingsymbol} "
            f"buy={spread.buy_leg.tradingsymbol}"
        )

        # ── Pre-trade margin check ──
        if equity is not None:
            margin_req = getattr(spread, '_margin_required', None)
            if margin_req is None:
                margin_req = self.margin_calc.span_margin_spread(
                    spread.max_loss / max(spread.buy_leg.quantity, 1),
                    spread.lots,
                    spread.lot_size,
                )
            if not self.margin_calc.check_margin_available(margin_req, equity):
                logger.error(
                    f"MARGIN CHECK FAILED for {signal_id}: "
                    f"required={margin_req:,.0f} equity={equity:,.0f} "
                    f"(limit=50%={equity * 0.50:,.0f})"
                )
                return SpreadFill(
                    signal_id=signal_id,
                    strategy=strategy,
                    buy_fill=None,
                    sell_fill=None,
                    net_entry_debit=0.0,
                    total_cost=0.0,
                    transaction_costs=0.0,
                    status="FAILED",
                    notes="Margin check failed — insufficient equity",
                )

        # ── Step 1: Place SHORT leg FIRST (collect premium) ──
        sell_fill = self._place_leg(spread.sell_leg, signal_id)
        if sell_fill is None or sell_fill.status == "FAILED":
            logger.error(f"Short leg FAILED for {signal_id} — aborting credit spread")
            return SpreadFill(
                signal_id=signal_id,
                strategy=strategy,
                buy_fill=None,
                sell_fill=sell_fill,
                net_entry_debit=0.0,
                total_cost=0.0,
                transaction_costs=0.0,
                status="FAILED",
                notes="Short leg failed — no position opened",
            )

        # ── Step 2: Place LONG leg (define risk) ──
        buy_fill = self._place_leg(spread.buy_leg, signal_id)

        if buy_fill is None or buy_fill.status == "FAILED":
            # CRITICAL SAFETY: long leg failed — we have a naked short!
            logger.warning(
                f"LONG LEG FAILED for {signal_id} after short filled — "
                f"NAKED SHORT EXPOSURE! Retrying with MARKET order..."
            )

            # Retry long leg ONCE with MARKET order
            buy_fill = self._retry_market_order(spread.buy_leg, signal_id)

            if buy_fill is None or buy_fill.status == "FAILED":
                # EMERGENCY: close the short leg immediately
                logger.error(
                    f"MARKET retry FAILED for long leg — "
                    f"IMMEDIATELY closing short leg {spread.sell_leg.tradingsymbol}"
                )
                self._emergency_close_short(spread.sell_leg, sell_fill, signal_id)

                return SpreadFill(
                    signal_id=signal_id,
                    strategy=strategy,
                    buy_fill=buy_fill,
                    sell_fill=sell_fill,
                    net_entry_debit=0.0,
                    total_cost=0.0,
                    transaction_costs=0.0,
                    status="FAILED",
                    notes="Long leg failed — emergency closed short leg (no naked exposure)",
                )

        # ── Step 3: Compute actual costs ──
        sell_price = sell_fill.fill_price if sell_fill else spread.sell_leg.premium
        buy_price = buy_fill.fill_price if buy_fill else spread.buy_leg.premium

        net_credit = sell_price - buy_price  # positive = credit received
        quantity = spread.sell_leg.quantity
        total_credit = net_credit * quantity

        tx_costs = self._calculate_transaction_costs(spread, buy_fill, sell_fill)

        # Determine status
        if (sell_fill and sell_fill.status == "FILLED" and
                buy_fill and buy_fill.status == "FILLED"):
            status = "FILLED"
            degraded = ""
        elif buy_fill and buy_fill.status == "PARTIAL":
            status = "PARTIAL"
            degraded = ""
        else:
            status = "DEGRADED"
            degraded = "NAKED_SHORT"

        fill = SpreadFill(
            signal_id=signal_id,
            strategy=strategy,
            buy_fill=buy_fill,
            sell_fill=sell_fill,
            net_entry_debit=round(-net_credit, 2),  # negative = credit
            total_cost=round(-total_credit, 2),      # negative = received
            transaction_costs=round(tx_costs, 2),
            status=status,
            degraded_to=degraded,
            fill_time=datetime.now().strftime("%H:%M:%S"),
        )

        logger.info(
            f"CreditSpreadFill: {status} | net_credit={net_credit:.2f}/unit "
            f"total_credit={total_credit:,.0f} tx_costs={tx_costs:,.0f} "
            f"max_risk={spread.max_loss:,.0f} "
            f"{'DEGRADED to ' + degraded if degraded else ''}"
        )

        return fill

    # ================================================================
    # EXIT: Buy back short first, then sell long
    # ================================================================

    def exit_credit_spread(
        self,
        spread: SpreadOrder,
        spread_fill: SpreadFill,
        reason: str,
        current_sell_premium: Optional[float] = None,
        current_buy_premium: Optional[float] = None,
    ) -> SpreadFill:
        """
        Exit a credit spread: buy back SHORT leg first, then sell LONG leg.

        At FORCE_EXIT_TIME (15:20): uses MARKET orders.

        Args:
            spread:               original SpreadOrder
            spread_fill:          the entry SpreadFill
            reason:               'STOP_LOSS', 'TARGET', 'FORCE_EXIT', 'DTE_EXIT', 'EMERGENCY'
            current_sell_premium: current price of the short leg
            current_buy_premium:  current price of the long leg

        Returns:
            SpreadFill representing the exit.
        """
        signal_id = spread.signal_id
        is_time_exit = reason in ("FORCE_EXIT", "EMERGENCY")
        logger.info(f"Exiting credit spread {signal_id}: reason={reason}")

        exit_sell_fill = None  # buying back the short
        exit_buy_fill = None   # selling the long

        # ── Step 1: BUY BACK the short leg FIRST ──
        if spread.sell_leg is not None and spread_fill.sell_fill is not None:
            close_short_leg = SpreadLeg(
                tradingsymbol=spread.sell_leg.tradingsymbol,
                instrument=spread.sell_leg.instrument,
                option_type=spread.sell_leg.option_type,
                strike=spread.sell_leg.strike,
                transaction_type="BUY",  # buy back to close short
                premium=current_sell_premium or spread.sell_leg.premium,
                lot_size=spread.sell_leg.lot_size,
                lots=spread.sell_leg.lots,
                quantity=spread.sell_leg.quantity,
            )

            if is_time_exit:
                exit_sell_fill = self._place_market_leg(close_short_leg, signal_id)
            else:
                exit_sell_fill = self._place_leg(close_short_leg, signal_id)

            if exit_sell_fill is None or exit_sell_fill.status == "FAILED":
                # CRITICAL: must close short — retry with MARKET
                logger.warning(
                    f"Short leg buyback FAILED — retrying MARKET for safety"
                )
                exit_sell_fill = self._retry_market_order(close_short_leg, signal_id)

        # ── Step 2: SELL the long leg ──
        if spread_fill.buy_fill is not None:
            close_long_leg = SpreadLeg(
                tradingsymbol=spread.buy_leg.tradingsymbol,
                instrument=spread.buy_leg.instrument,
                option_type=spread.buy_leg.option_type,
                strike=spread.buy_leg.strike,
                transaction_type="SELL",  # sell to close long
                premium=current_buy_premium or spread.buy_leg.premium,
                lot_size=spread.buy_leg.lot_size,
                lots=spread.buy_leg.lots,
                quantity=spread.buy_leg.quantity,
            )

            if is_time_exit:
                exit_buy_fill = self._place_market_leg(close_long_leg, signal_id)
            else:
                exit_buy_fill = self._place_leg(close_long_leg, signal_id)

        # ── Step 3: Handle iron condor call-side exit ──
        if spread.strategy == IRON_CONDOR and hasattr(spread, '_call_sell_leg'):
            self._exit_iron_condor_call_side(spread, spread_fill, reason, signal_id)

        # ── Step 4: Compute exit cost ──
        buyback_price = (
            exit_sell_fill.fill_price
            if exit_sell_fill and exit_sell_fill.status == "FILLED"
            else current_sell_premium or 0.0
        )
        sell_long_price = (
            exit_buy_fill.fill_price
            if exit_buy_fill and exit_buy_fill.status == "FILLED"
            else current_buy_premium or 0.0
        )

        # Net exit debit: what we pay to close short minus what we receive from long
        net_exit_debit = buyback_price - sell_long_price
        quantity = spread.sell_leg.quantity
        total_exit = net_exit_debit * quantity

        tx_costs = self._calculate_exit_costs(spread, exit_buy_fill, exit_sell_fill)

        status = "FILLED"
        if exit_sell_fill is None or exit_sell_fill.status == "FAILED":
            status = "PARTIAL"
        if exit_buy_fill is not None and exit_buy_fill.status == "FAILED":
            status = "PARTIAL"

        exit_result = SpreadFill(
            signal_id=signal_id,
            strategy=spread.strategy,
            buy_fill=exit_buy_fill,     # closing the long (SELL)
            sell_fill=exit_sell_fill,    # closing the short (BUY back)
            net_entry_debit=round(net_exit_debit, 2),
            total_cost=round(total_exit, 2),
            transaction_costs=round(tx_costs, 2),
            status=status,
            fill_time=datetime.now().strftime("%H:%M:%S"),
            notes=f"EXIT:{reason}",
        )

        logger.info(
            f"Credit spread exit {signal_id}: {reason} | "
            f"buyback={buyback_price:.2f} sell_long={sell_long_price:.2f} | "
            f"net_exit_debit={net_exit_debit:.2f}/unit total={total_exit:,.0f} "
            f"tx={tx_costs:,.0f}"
        )

        return exit_result

    # ================================================================
    # PRIVATE: Emergency close
    # ================================================================

    def _emergency_close_short(
        self,
        sell_leg: SpreadLeg,
        sell_fill: LegFill,
        signal_id: str,
    ):
        """
        EMERGENCY: immediately close a naked short position.

        Uses MARKET order. Logs critically. Alerts via Telegram.
        """
        logger.error(
            f"EMERGENCY CLOSE: buying back {sell_leg.tradingsymbol} "
            f"to eliminate naked short exposure for {signal_id}"
        )

        close_leg = SpreadLeg(
            tradingsymbol=sell_leg.tradingsymbol,
            instrument=sell_leg.instrument,
            option_type=sell_leg.option_type,
            strike=sell_leg.strike,
            transaction_type="BUY",
            premium=sell_fill.fill_price * 1.05,  # willing to pay 5% more
            lot_size=sell_leg.lot_size,
            lots=sell_leg.lots,
            quantity=sell_leg.quantity,
        )

        close_fill = self._place_market_leg(close_leg, signal_id)

        if close_fill is None or close_fill.status == "FAILED":
            # This is a critical failure — alert human intervention needed
            logger.critical(
                f"CRITICAL: FAILED to close naked short {sell_leg.tradingsymbol} "
                f"for {signal_id} — MANUAL INTERVENTION REQUIRED"
            )
            if self.alerter:
                self.alerter.send(
                    "CRITICAL",
                    f"NAKED SHORT EXPOSURE: Failed to close {sell_leg.tradingsymbol}. "
                    f"Manual intervention required!",
                    signal_id=signal_id,
                )
        else:
            logger.info(
                f"Emergency close SUCCESS: {sell_leg.tradingsymbol} "
                f"bought back @ {close_fill.fill_price:.2f}"
            )
            if self.alerter:
                self.alerter.send(
                    "WARNING",
                    f"Emergency closed short {sell_leg.tradingsymbol} "
                    f"@ {close_fill.fill_price:.2f} — spread aborted",
                    signal_id=signal_id,
                )

    def _exit_iron_condor_call_side(
        self,
        spread: SpreadOrder,
        spread_fill: SpreadFill,
        reason: str,
        signal_id: str,
    ):
        """Exit the call-side legs of an iron condor."""
        is_time_exit = reason in ("FORCE_EXIT", "EMERGENCY")

        call_sell_leg = getattr(spread, '_call_sell_leg', None)
        call_buy_leg = getattr(spread, '_call_buy_leg', None)

        if call_sell_leg is None:
            return

        # Buy back short call
        close_call_short = SpreadLeg(
            tradingsymbol=call_sell_leg.tradingsymbol,
            instrument=call_sell_leg.instrument,
            option_type=call_sell_leg.option_type,
            strike=call_sell_leg.strike,
            transaction_type="BUY",
            premium=call_sell_leg.premium,
            lot_size=call_sell_leg.lot_size,
            lots=call_sell_leg.lots,
            quantity=call_sell_leg.quantity,
        )

        if is_time_exit:
            fill = self._place_market_leg(close_call_short, signal_id)
        else:
            fill = self._place_leg(close_call_short, signal_id)

        if fill is None or fill.status == "FAILED":
            logger.warning(f"Call-side short buyback failed — retrying MARKET")
            fill = self._retry_market_order(close_call_short, signal_id)

        # Sell long call
        if call_buy_leg is not None:
            close_call_long = SpreadLeg(
                tradingsymbol=call_buy_leg.tradingsymbol,
                instrument=call_buy_leg.instrument,
                option_type=call_buy_leg.option_type,
                strike=call_buy_leg.strike,
                transaction_type="SELL",
                premium=call_buy_leg.premium,
                lot_size=call_buy_leg.lot_size,
                lots=call_buy_leg.lots,
                quantity=call_buy_leg.quantity,
            )

            if is_time_exit:
                self._place_market_leg(close_call_long, signal_id)
            else:
                self._place_leg(close_call_long, signal_id)

    # ================================================================
    # PRIVATE: Strike selection & premium estimation
    # ================================================================

    def _delta_to_strike(
        self,
        spot: float,
        target_delta: float,
        option_type: str,
        instrument: str,
        atr: float,
    ) -> int:
        """
        Approximate strike from target delta using ATR-based model.

        For NIFTY:
            0.20 delta ~ ATM +/- 1.5 * ATR
            0.30 delta ~ ATM +/- 1.0 * ATR
            0.15 delta ~ ATM +/- 2.0 * ATR

        Falls back to spot +/- fixed offset if ATR is unavailable.
        """
        if atr <= 0:
            # Fallback: fixed offset based on delta
            offset = spot * (0.50 - target_delta) * 0.10
            offset = max(offset, 100)
        else:
            # Delta-to-ATR multiplier: linear interpolation
            # 0.50 delta = ATM (0 ATR offset)
            # 0.30 delta = 1.0 ATR offset
            # 0.20 delta = 1.5 ATR offset
            # 0.15 delta = 2.0 ATR offset
            # 0.10 delta = 2.5 ATR offset
            if target_delta >= 0.50:
                atr_mult = 0.0
            elif target_delta >= 0.30:
                # Linear from 0.50→0.0 to 0.30→1.0
                atr_mult = (0.50 - target_delta) / 0.20 * 1.0
            elif target_delta >= 0.20:
                # Linear from 0.30→1.0 to 0.20→1.5
                atr_mult = 1.0 + (0.30 - target_delta) / 0.10 * 0.5
            elif target_delta >= 0.10:
                # Linear from 0.20→1.5 to 0.10→2.5
                atr_mult = 1.5 + (0.20 - target_delta) / 0.10 * 1.0
            else:
                atr_mult = 2.5 + (0.10 - target_delta) / 0.05 * 0.5

            offset = atr * atr_mult

        if option_type == "CE":
            raw_strike = spot + offset
        else:
            raw_strike = spot - offset

        return self._snap_strike(raw_strike, instrument)

    def _snap_strike(self, raw_strike: float, instrument: str) -> int:
        """Round to the nearest valid strike interval."""
        interval = (
            NIFTY_STRIKE_INTERVAL if instrument == "NIFTY"
            else BANKNIFTY_STRIKE_INTERVAL
        )
        return int(round(raw_strike / interval) * interval)

    def _estimate_atr(self, spot: float, vix: float) -> float:
        """
        Estimate daily ATR from VIX.

        ATR ~ spot * VIX/100 * sqrt(1/252) ~ spot * VIX / 1587
        """
        return spot * max(vix, 8.0) / 1587.0

    def _estimate_otm_premium(
        self,
        spot: float,
        strike: int,
        vix: float,
        dte: int,
        atr: float,
    ) -> float:
        """
        Estimate OTM option premium using VIX and time-decay model.

        Premium = intrinsic (0 for OTM) + time value
        Time value ~ spot * VIX/100 * sqrt(DTE/365) * decay(distance)
        """
        distance = abs(strike - spot)
        moneyness = distance / spot

        # Base time value: spot * VIX/100 * sqrt(DTE/365)
        time_value = spot * (max(vix, 8.0) / 100.0) * math.sqrt(max(dte, 1) / 365.0)

        # OTM decay: exponential with moneyness
        decay = math.exp(-6.0 * moneyness)
        premium = time_value * decay

        # Floor: ₹1 minimum for tradability
        premium = max(premium, 1.0)

        return round(premium, 2)

    def _compute_credit_greeks(
        self,
        sell_leg: SpreadLeg,
        buy_leg: SpreadLeg,
        sell_delta: float,
    ) -> Tuple[float, float, float, float]:
        """
        Compute net greeks for a credit spread.

        Short leg has higher absolute delta (closer to ATM).
        Long leg has lower absolute delta (further OTM).
        """
        # Short leg delta (negative because we are short)
        sell_leg.delta = -sell_delta
        sell_leg.gamma = -0.002
        sell_leg.theta = 10.0    # positive theta (we collect time decay)
        sell_leg.vega = -5.0     # short vega (we want vol to drop)

        # Long leg: lower delta (further OTM)
        buy_delta = sell_delta * 0.5
        buy_leg.delta = buy_delta
        buy_leg.gamma = 0.001
        buy_leg.theta = -5.0
        buy_leg.vega = 3.0

        # Net: short delta spread, positive theta, short vega
        net_delta = round(-sell_delta + buy_delta, 4)
        net_gamma = round(-0.002 + 0.001, 6)
        net_theta = round(10.0 - 5.0, 2)
        net_vega = round(-5.0 + 3.0, 2)

        return net_delta, net_gamma, net_theta, net_vega

    # ================================================================
    # PRIVATE: Order placement (mirrors SpreadExecutor patterns)
    # ================================================================

    def _place_leg(self, leg: SpreadLeg, signal_id: str) -> Optional[LegFill]:
        """Place a single leg order and wait for fill."""
        if self.paper_mode:
            return self._simulate_paper_fill(leg, signal_id)

        if self.bridge is None:
            logger.error("No KiteBridge configured — cannot place live orders")
            return None

        order_dict = {
            "signal_id": signal_id,
            "tradingsymbol": leg.tradingsymbol,
            "transaction_type": leg.transaction_type,
            "lots": leg.lots,
            "lot_size": leg.lot_size,
            "quantity": leg.quantity,
            "limit_price": leg.premium,
            "instrument": leg.instrument,
            "option_type": leg.option_type,
            "strike": leg.strike,
            "exchange": "NFO",
            "product": "MIS",
            "order_type": "LIMIT",
        }

        order_id = self.bridge.place_order(order_dict)
        if order_id is None:
            return LegFill(
                tradingsymbol=leg.tradingsymbol,
                transaction_type=leg.transaction_type,
                order_id="",
                fill_price=0.0,
                fill_quantity=0,
                fill_time="",
                slippage=0.0,
                status="FAILED",
            )

        result = self.monitor.monitor_fill(
            order_id=order_id,
            timeout_seconds=LEG2_WAIT_SECONDS,
            expected_price=leg.premium,
            tradingsymbol=leg.tradingsymbol,
            signal_id=signal_id,
        )

        if result is None:
            return LegFill(
                tradingsymbol=leg.tradingsymbol,
                transaction_type=leg.transaction_type,
                order_id=order_id,
                fill_price=0.0,
                fill_quantity=0,
                fill_time="",
                slippage=0.0,
                status="FAILED",
            )

        return LegFill(
            tradingsymbol=leg.tradingsymbol,
            transaction_type=leg.transaction_type,
            order_id=order_id,
            fill_price=result.get("fill_price", 0.0),
            fill_quantity=result.get("fill_quantity", leg.quantity),
            fill_time=result.get("fill_time", ""),
            slippage=result.get("slippage", 0.0),
            status=result.get("status", "FILLED"),
        )

    def _place_market_leg(
        self, leg: SpreadLeg, signal_id: str
    ) -> Optional[LegFill]:
        """Place a MARKET order for time-critical exits."""
        if self.paper_mode:
            return self._simulate_paper_fill(leg, signal_id, market=True)

        if self.bridge is None:
            return None

        order_dict = {
            "signal_id": signal_id,
            "tradingsymbol": leg.tradingsymbol,
            "transaction_type": leg.transaction_type,
            "lots": leg.lots,
            "lot_size": leg.lot_size,
            "quantity": leg.quantity,
            "limit_price": leg.premium,
            "instrument": leg.instrument,
            "option_type": leg.option_type,
            "strike": leg.strike,
            "exchange": "NFO",
            "product": "MIS",
            "order_type": "MARKET",
        }

        order_id = self.bridge.place_order(order_dict)
        if order_id is None:
            return LegFill(
                tradingsymbol=leg.tradingsymbol,
                transaction_type=leg.transaction_type,
                order_id="",
                fill_price=0.0,
                fill_quantity=0,
                fill_time="",
                slippage=0.0,
                status="FAILED",
            )

        result = self.monitor.monitor_fill(
            order_id=order_id,
            timeout_seconds=EXIT_LEG_TIMEOUT,
            expected_price=leg.premium,
            tradingsymbol=leg.tradingsymbol,
            signal_id=signal_id,
        )

        if result is None:
            return LegFill(
                tradingsymbol=leg.tradingsymbol,
                transaction_type=leg.transaction_type,
                order_id=order_id,
                fill_price=0.0,
                fill_quantity=0,
                fill_time="",
                slippage=0.0,
                status="FAILED",
            )

        return LegFill(
            tradingsymbol=leg.tradingsymbol,
            transaction_type=leg.transaction_type,
            order_id=order_id,
            fill_price=result.get("fill_price", 0.0),
            fill_quantity=result.get("fill_quantity", leg.quantity),
            fill_time=result.get("fill_time", ""),
            slippage=result.get("slippage", 0.0),
            status=result.get("status", "FILLED"),
        )

    def _retry_market_order(
        self, leg: SpreadLeg, signal_id: str
    ) -> Optional[LegFill]:
        """Retry a failed leg with MARKET order type."""
        logger.info(f"MARKET retry: {leg.transaction_type} {leg.tradingsymbol}")
        return self._place_market_leg(leg, signal_id)

    # ================================================================
    # PRIVATE: Paper mode simulation
    # ================================================================

    def _simulate_paper_fill(
        self,
        leg: SpreadLeg,
        signal_id: str,
        market: bool = False,
    ) -> LegFill:
        """Simulate a fill at midpoint with minor slippage."""
        if market:
            slippage_pct = random.uniform(0.001, 0.008)
        else:
            slippage_pct = random.uniform(-0.002, 0.005)

        if leg.transaction_type == "BUY":
            fill_price = leg.premium * (1 + slippage_pct)
        else:
            fill_price = leg.premium * (1 - slippage_pct)

        # Tick-align to 0.05
        fill_price = round(fill_price / 0.05) * 0.05
        fill_price = max(fill_price, 0.05)
        fill_price = round(fill_price, 2)

        order_id = f"PAPER-{uuid.uuid4().hex[:12]}"

        logger.info(
            f"PAPER FILL | {leg.transaction_type} {leg.tradingsymbol} "
            f"x{leg.quantity} @ {fill_price:.2f} "
            f"(expected {leg.premium:.2f}, slippage {slippage_pct:+.2%})"
        )

        return LegFill(
            tradingsymbol=leg.tradingsymbol,
            transaction_type=leg.transaction_type,
            order_id=order_id,
            fill_price=fill_price,
            fill_quantity=leg.quantity,
            fill_time=datetime.now().strftime("%H:%M:%S"),
            slippage=round(slippage_pct, 4),
            status="FILLED",
        )

    # ================================================================
    # PRIVATE: Cost calculation
    # ================================================================

    def _calculate_transaction_costs(
        self,
        spread: SpreadOrder,
        buy_fill: Optional[LegFill],
        sell_fill: Optional[LegFill],
    ) -> float:
        """Calculate entry transaction costs for a credit spread."""
        lots = spread.lots
        n_legs = 0
        if buy_fill and buy_fill.status == "FILLED":
            n_legs += 1
        if sell_fill and sell_fill.status == "FILLED":
            n_legs += 1

        brokerage = BROKERAGE_PER_LOT_PER_LEG * lots * n_legs

        # STT: 0.05% on sell-side premium x quantity
        stt = 0.0
        if sell_fill and sell_fill.status == "FILLED":
            stt = sell_fill.fill_price * spread.sell_leg.quantity * STT_PCT

        # Slippage on total premium
        total_premium = 0.0
        if buy_fill and buy_fill.status == "FILLED":
            total_premium += buy_fill.fill_price * spread.buy_leg.quantity
        if sell_fill and sell_fill.status == "FILLED":
            total_premium += sell_fill.fill_price * spread.sell_leg.quantity
        slippage = total_premium * SLIPPAGE_PCT

        return brokerage + stt + slippage

    def _calculate_exit_costs(
        self,
        spread: SpreadOrder,
        exit_buy_fill: Optional[LegFill],
        exit_sell_fill: Optional[LegFill],
    ) -> float:
        """Calculate exit transaction costs for a credit spread."""
        lots = spread.lots
        n_legs = 0
        if exit_buy_fill and exit_buy_fill.status == "FILLED":
            n_legs += 1
        if exit_sell_fill and exit_sell_fill.status == "FILLED":
            n_legs += 1

        brokerage = BROKERAGE_PER_LOT_PER_LEG * lots * n_legs

        # STT on sell-side (selling the long leg at exit)
        stt = 0.0
        if exit_buy_fill and exit_buy_fill.status == "FILLED":
            stt = exit_buy_fill.fill_price * spread.buy_leg.quantity * STT_PCT

        total_premium = 0.0
        if exit_buy_fill and exit_buy_fill.status == "FILLED":
            total_premium += exit_buy_fill.fill_price * spread.buy_leg.quantity
        if exit_sell_fill and exit_sell_fill.status == "FILLED":
            total_premium += exit_sell_fill.fill_price * spread.sell_leg.quantity
        slippage = total_premium * SLIPPAGE_PCT

        return brokerage + stt + slippage

    # ================================================================
    # PRIVATE: Expiry helper
    # ================================================================

    def _next_weekly_expiry(self) -> str:
        """Estimate next weekly expiry in YYMMDD format."""
        today = date.today()
        days_ahead = (3 - today.weekday()) % 7
        if days_ahead == 0 and datetime.now().time() > dt_time(15, 30):
            days_ahead = 7
        expiry = today + timedelta(days=days_ahead)
        return expiry.strftime("%y%m%d")
