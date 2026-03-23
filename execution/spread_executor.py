"""
Spread Executor — places multi-leg option orders with safety guarantees.

Entry sequence: BUY leg first → wait for fill → SELL leg.
Exit sequence:  close SELL leg first (buyback) → close BUY leg.

Safety invariant: NEVER leave a naked short option position.
If the sell leg fails on entry, keep the buy leg as a naked long (safe).
If the buy-back fails on exit, retry with MARKET order.

Usage:
    from execution.spread_executor import SpreadExecutor
    executor = SpreadExecutor(kite_bridge, fill_monitor, paper_mode=True)
    fill = executor.execute_spread(spread_order)
    exit_fill = executor.exit_spread(spread_order, positions, 'TARGET')
"""

import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, time as dt_time
from typing import Dict, List, Optional

from execution.spread_builder import SpreadOrder, SpreadLeg, NAKED_BUY

logger = logging.getLogger(__name__)

EXECUTION_MODE = os.environ.get("EXECUTION_MODE", "PAPER").upper()

# Timing
LEG2_WAIT_SECONDS = 60    # max wait for leg1 fill before placing leg2
EXIT_LEG_TIMEOUT = 30     # timeout for exit leg fills
FORCE_EXIT_TIME = dt_time(15, 20)

# Transaction costs (from options_backtest.py)
BROKERAGE_PER_LOT_PER_LEG = 40   # ₹40 per lot per leg
STT_PCT = 0.0005                  # 0.05% on sell-side premium
SLIPPAGE_PCT = 0.001              # 0.1% of premium


# ================================================================
# DATACLASS
# ================================================================

@dataclass
class LegFill:
    """Fill result for a single leg."""
    tradingsymbol: str
    transaction_type: str       # BUY / SELL
    order_id: str
    fill_price: float
    fill_quantity: int
    fill_time: str
    slippage: float
    status: str                 # FILLED / PARTIAL / FAILED


@dataclass
class SpreadFill:
    """Fill result for the complete spread."""
    signal_id: str
    strategy: str
    buy_fill: Optional[LegFill]
    sell_fill: Optional[LegFill]

    # Net fill
    net_entry_debit: float      # actual net premium paid per unit
    total_cost: float           # total capital deployed (net × quantity)
    transaction_costs: float    # brokerage + STT + slippage

    # Status
    status: str                 # FILLED / PARTIAL / DEGRADED / FAILED
    degraded_to: str = ""       # if spread degraded, what it became
    fill_time: str = ""
    notes: str = ""


# ================================================================
# SPREAD EXECUTOR
# ================================================================

class SpreadExecutor:
    """
    Executes multi-leg option spreads through KiteBridge and FillMonitor.

    In PAPER mode, simulates fills at midpoint with minor slippage.
    In LIVE mode, places real orders leg-by-leg with safety fallbacks.
    """

    def __init__(self, kite_bridge=None, fill_monitor=None,
                 paper_mode: bool = True, alerter=None):
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

    # ================================================================
    # ENTRY
    # ================================================================

    def execute_spread(self, spread: SpreadOrder) -> SpreadFill:
        """
        Execute a spread order: BUY leg first, then SELL leg.

        If NAKED_BUY, only places the buy leg.
        If sell leg fails, degrades to NAKED_BUY (never naked short).

        Args:
            spread: SpreadOrder from SpreadBuilder

        Returns:
            SpreadFill with fill details and actual costs.
        """
        signal_id = spread.signal_id
        logger.info(
            f"Executing {spread.strategy} for {signal_id}: "
            f"buy={spread.buy_leg.tradingsymbol} "
            f"{'sell=' + spread.sell_leg.tradingsymbol if spread.sell_leg else 'naked'}"
        )

        # ── Step 1: Place BUY leg ──
        buy_fill = self._place_leg(spread.buy_leg, signal_id)
        if buy_fill is None or buy_fill.status == "FAILED":
            logger.error(f"Buy leg FAILED for {signal_id} — aborting spread")
            return SpreadFill(
                signal_id=signal_id,
                strategy=spread.strategy,
                buy_fill=buy_fill,
                sell_fill=None,
                net_entry_debit=0.0,
                total_cost=0.0,
                transaction_costs=0.0,
                status="FAILED",
                notes="Buy leg failed — no position opened",
            )

        # ── Step 2: Place SELL leg (if spread) ──
        sell_fill = None
        if spread.sell_leg is not None and spread.strategy != NAKED_BUY:
            sell_fill = self._place_leg(spread.sell_leg, signal_id)

            if sell_fill is None or sell_fill.status == "FAILED":
                # SAFETY: sell leg failed — degrade to naked long (safe)
                logger.warning(
                    f"Sell leg FAILED for {signal_id} — degrading to NAKED_BUY "
                    f"(keeping long {spread.buy_leg.tradingsymbol})"
                )

                # Retry sell with MARKET order (one attempt)
                sell_fill = self._retry_market_order(spread.sell_leg, signal_id)

                if sell_fill is None or sell_fill.status == "FAILED":
                    logger.warning(
                        f"MARKET retry also failed for sell leg — "
                        f"position is naked long"
                    )
                    sell_fill = None  # confirmed degraded

        # ── Step 3: Compute actual costs ──
        buy_price = buy_fill.fill_price if buy_fill else spread.buy_leg.premium
        sell_price = sell_fill.fill_price if sell_fill else 0.0

        net_debit = buy_price - sell_price
        quantity = spread.buy_leg.quantity
        total_cost = net_debit * quantity

        tx_costs = self._calculate_transaction_costs(
            spread, buy_fill, sell_fill
        )

        # Determine status
        if spread.strategy == NAKED_BUY:
            status = "FILLED" if buy_fill and buy_fill.status == "FILLED" else "PARTIAL"
            degraded = ""
        elif sell_fill and sell_fill.status == "FILLED":
            status = "FILLED"
            degraded = ""
        elif sell_fill and sell_fill.status == "PARTIAL":
            status = "PARTIAL"
            degraded = ""
        else:
            status = "DEGRADED"
            degraded = "NAKED_BUY"

        fill = SpreadFill(
            signal_id=signal_id,
            strategy=spread.strategy,
            buy_fill=buy_fill,
            sell_fill=sell_fill,
            net_entry_debit=round(net_debit, 2),
            total_cost=round(total_cost, 2),
            transaction_costs=round(tx_costs, 2),
            status=status,
            degraded_to=degraded,
            fill_time=datetime.now().strftime("%H:%M:%S"),
        )

        logger.info(
            f"SpreadFill: {status} | net_debit={net_debit:.2f}/unit "
            f"total={total_cost:,.0f} tx_costs={tx_costs:,.0f} "
            f"{'DEGRADED to ' + degraded if degraded else ''}"
        )

        return fill

    # ================================================================
    # CREDIT SPREAD ENTRY (Layer 8)
    # ================================================================

    def execute_credit_spread(self, spread: SpreadOrder) -> SpreadFill:
        """
        Execute a credit spread: SELL leg first (collect premium), then BUY leg (define risk).

        If buy leg (hedge) fails after sell fills -> IMMEDIATELY close sell leg.
        This is the reverse of execute_spread() which places buy first.

        Supports: BULL_PUT_SPREAD, BEAR_CALL_SPREAD, SHORT_STRANGLE,
                  SHORT_STRADDLE, IRON_CONDOR
        """
        signal_id = spread.signal_id
        logger.info(
            f"Executing CREDIT {spread.strategy} for {signal_id}: "
            f"sell={spread.sell_leg.tradingsymbol if spread.sell_leg else 'none'} "
            f"buy={spread.buy_leg.tradingsymbol}"
        )

        if spread.sell_leg is None:
            logger.error(f"Credit spread requires a sell leg — aborting {signal_id}")
            return SpreadFill(
                signal_id=signal_id, strategy=spread.strategy,
                buy_fill=None, sell_fill=None,
                net_entry_debit=0.0, total_cost=0.0, transaction_costs=0.0,
                status="FAILED", notes="No sell leg for credit spread",
            )

        # ── Step 1: Place SELL leg first (collect premium) ──
        sell_fill = self._place_leg(spread.sell_leg, signal_id)
        if sell_fill is None or sell_fill.status == "FAILED":
            logger.error(f"Sell leg FAILED for credit spread {signal_id} — aborting")
            return SpreadFill(
                signal_id=signal_id, strategy=spread.strategy,
                buy_fill=None, sell_fill=sell_fill,
                net_entry_debit=0.0, total_cost=0.0, transaction_costs=0.0,
                status="FAILED", notes="Sell leg failed — no position opened",
            )

        # ── Step 2: Place BUY leg (define risk / hedge) ──
        buy_fill = self._place_leg(spread.buy_leg, signal_id)

        if buy_fill is None or buy_fill.status == "FAILED":
            # CRITICAL: buy hedge failed — MUST close sell leg (no naked risk)
            logger.warning(
                f"Buy hedge FAILED for {signal_id} — retrying with MARKET"
            )
            buy_fill = self._retry_market_order(spread.buy_leg, signal_id)

            if buy_fill is None or buy_fill.status == "FAILED":
                # EMERGENCY: close sell leg immediately
                logger.error(
                    f"EMERGENCY: closing sell leg for {signal_id} — "
                    f"hedge could not be placed"
                )
                close_sell = SpreadLeg(
                    tradingsymbol=spread.sell_leg.tradingsymbol,
                    instrument=spread.sell_leg.instrument,
                    option_type=spread.sell_leg.option_type,
                    strike=spread.sell_leg.strike,
                    transaction_type="BUY",  # buyback
                    premium=sell_fill.fill_price,
                    lot_size=spread.sell_leg.lot_size,
                    lots=spread.sell_leg.lots,
                    quantity=spread.sell_leg.quantity,
                )
                self._retry_market_order(close_sell, f"EMERG_{signal_id}")

                return SpreadFill(
                    signal_id=signal_id, strategy=spread.strategy,
                    buy_fill=None, sell_fill=sell_fill,
                    net_entry_debit=0.0, total_cost=0.0, transaction_costs=0.0,
                    status="FAILED",
                    notes="Hedge failed — sell leg closed immediately",
                )

        # ── Step 3: Compute actual costs ──
        sell_price = sell_fill.fill_price
        buy_price = buy_fill.fill_price
        net_credit = sell_price - buy_price  # positive for credit
        quantity = spread.sell_leg.quantity
        total_cost = net_credit * quantity  # negative = credit received

        tx_costs = self._calculate_transaction_costs(spread, buy_fill, sell_fill)

        status = "FILLED"
        if buy_fill.status == "PARTIAL" or sell_fill.status == "PARTIAL":
            status = "PARTIAL"

        fill = SpreadFill(
            signal_id=signal_id,
            strategy=spread.strategy,
            buy_fill=buy_fill,
            sell_fill=sell_fill,
            net_entry_debit=round(-net_credit, 2),  # negative debit = credit
            total_cost=round(total_cost, 2),
            transaction_costs=round(tx_costs, 2),
            status=status,
            fill_time=datetime.now().strftime("%H:%M:%S"),
            notes=f"Credit received: {net_credit:.2f}/unit",
        )

        logger.info(
            f"CreditSpreadFill: {status} | net_credit={net_credit:.2f}/unit "
            f"total={abs(total_cost):,.0f} tx_costs={tx_costs:,.0f}"
        )

        return fill

    # ================================================================
    # EXIT
    # ================================================================

    def exit_spread(
        self,
        spread: SpreadOrder,
        spread_fill: SpreadFill,
        reason: str,
        current_buy_premium: Optional[float] = None,
        current_sell_premium: Optional[float] = None,
    ) -> SpreadFill:
        """
        Exit a spread position: close SELL leg first (buyback), then BUY leg.

        At FORCE_EXIT_TIME (15:20): uses MARKET orders.

        Args:
            spread:              original SpreadOrder
            spread_fill:         the entry SpreadFill
            reason:              'STOP_LOSS', 'TARGET', 'FORCE_EXIT', 'EMERGENCY'
            current_buy_premium: current price of buy leg (for paper mode)
            current_sell_premium: current price of sell leg

        Returns:
            SpreadFill representing the exit.
        """
        signal_id = spread.signal_id
        is_time_exit = reason in ("FORCE_EXIT", "EMERGENCY")
        logger.info(f"Exiting spread {signal_id}: reason={reason}")

        exit_sell_fill = None
        exit_buy_fill = None

        # ── Step 1: Close SELL leg (buyback) — do this FIRST ──
        if spread.sell_leg is not None and spread_fill.sell_fill is not None:
            close_sell_leg = SpreadLeg(
                tradingsymbol=spread.sell_leg.tradingsymbol,
                instrument=spread.sell_leg.instrument,
                option_type=spread.sell_leg.option_type,
                strike=spread.sell_leg.strike,
                transaction_type="BUY",  # buyback
                premium=current_sell_premium or spread.sell_leg.premium,
                lot_size=spread.sell_leg.lot_size,
                lots=spread.sell_leg.lots,
                quantity=spread.sell_leg.quantity,
            )

            if is_time_exit:
                exit_sell_fill = self._place_market_leg(close_sell_leg, signal_id)
            else:
                exit_sell_fill = self._place_leg(close_sell_leg, signal_id)

            if exit_sell_fill is None or exit_sell_fill.status == "FAILED":
                # CRITICAL: retry with MARKET — we must close short
                logger.warning(
                    f"Sell leg buyback FAILED — retrying MARKET for safety"
                )
                exit_sell_fill = self._retry_market_order(
                    close_sell_leg, signal_id
                )

        # ── Step 2: Close BUY leg ──
        close_buy_leg = SpreadLeg(
            tradingsymbol=spread.buy_leg.tradingsymbol,
            instrument=spread.buy_leg.instrument,
            option_type=spread.buy_leg.option_type,
            strike=spread.buy_leg.strike,
            transaction_type="SELL",  # sell to close
            premium=current_buy_premium or spread.buy_leg.premium,
            lot_size=spread.buy_leg.lot_size,
            lots=spread.buy_leg.lots,
            quantity=spread.buy_leg.quantity,
        )

        if is_time_exit:
            exit_buy_fill = self._place_market_leg(close_buy_leg, signal_id)
        else:
            exit_buy_fill = self._place_leg(close_buy_leg, signal_id)

        # ── Step 3: Compute exit P&L ──
        exit_buy_price = (
            exit_buy_fill.fill_price if exit_buy_fill and exit_buy_fill.status == "FILLED"
            else current_buy_premium or 0.0
        )
        exit_sell_price = (
            exit_sell_fill.fill_price if exit_sell_fill and exit_sell_fill.status == "FILLED"
            else current_sell_premium or 0.0
        )

        # Net exit credit: what we receive (sell buy_leg) minus what we pay (buyback sell_leg)
        net_exit_credit = exit_buy_price - exit_sell_price
        quantity = spread.buy_leg.quantity
        total_exit = net_exit_credit * quantity

        tx_costs = self._calculate_exit_costs(spread, exit_buy_fill, exit_sell_fill)

        status = "FILLED"
        if exit_buy_fill is None or exit_buy_fill.status == "FAILED":
            status = "PARTIAL"
        if exit_sell_fill is not None and exit_sell_fill.status == "FAILED":
            status = "PARTIAL"

        exit_result = SpreadFill(
            signal_id=signal_id,
            strategy=spread.strategy,
            buy_fill=exit_buy_fill,
            sell_fill=exit_sell_fill,
            net_entry_debit=round(net_exit_credit, 2),  # reused field for exit credit
            total_cost=round(total_exit, 2),
            transaction_costs=round(tx_costs, 2),
            status=status,
            fill_time=datetime.now().strftime("%H:%M:%S"),
            notes=f"EXIT:{reason}",
        )

        logger.info(
            f"Spread exit {signal_id}: {reason} | exit_credit={net_exit_credit:.2f}/unit "
            f"total={total_exit:,.0f} tx={tx_costs:,.0f}"
        )

        return exit_result

    # ================================================================
    # P&L CALCULATION
    # ================================================================

    def calculate_spread_pnl(
        self,
        entry_fill: SpreadFill,
        exit_fill: SpreadFill,
        quantity: int,
    ) -> Dict:
        """
        Calculate full P&L for a spread trade including all costs.

        Returns:
            dict with gross_pnl, costs, net_pnl, pnl_pct
        """
        entry_debit = entry_fill.net_entry_debit  # per unit
        exit_credit = exit_fill.net_entry_debit    # per unit (reused field)

        gross_pnl = (exit_credit - entry_debit) * quantity
        total_costs = entry_fill.transaction_costs + exit_fill.transaction_costs
        net_pnl = gross_pnl - total_costs

        pnl_pct = net_pnl / (entry_debit * quantity) if entry_debit * quantity > 0 else 0

        return {
            "entry_debit_per_unit": round(entry_debit, 2),
            "exit_credit_per_unit": round(exit_credit, 2),
            "quantity": quantity,
            "gross_pnl": round(gross_pnl, 2),
            "entry_costs": round(entry_fill.transaction_costs, 2),
            "exit_costs": round(exit_fill.transaction_costs, 2),
            "total_costs": round(total_costs, 2),
            "net_pnl": round(net_pnl, 2),
            "pnl_pct": round(pnl_pct, 4),
        }

    # ================================================================
    # PRIVATE: Order placement
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

        # Monitor fill
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
        import random

        # Midpoint fill with slippage
        if market:
            # Market orders: wider slippage
            slippage_pct = random.uniform(0.001, 0.008)
        else:
            slippage_pct = random.uniform(-0.002, 0.005)

        if leg.transaction_type == "BUY":
            fill_price = leg.premium * (1 + slippage_pct)
        else:
            fill_price = leg.premium * (1 - slippage_pct)

        fill_price = round(fill_price / 0.05) * 0.05  # tick-align
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
        """Calculate entry transaction costs."""
        lots = spread.lots
        n_legs = 1 + (1 if sell_fill and sell_fill.status == "FILLED" else 0)

        # Brokerage: ₹40 per lot per leg
        brokerage = BROKERAGE_PER_LOT_PER_LEG * lots * n_legs

        # STT: 0.05% on sell-side premium × quantity
        stt = 0.0
        if sell_fill and sell_fill.status == "FILLED":
            stt = sell_fill.fill_price * spread.sell_leg.quantity * STT_PCT

        # Slippage: 0.1% on total premium
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
        """Calculate exit transaction costs."""
        lots = spread.lots
        n_legs = 0
        if exit_buy_fill and exit_buy_fill.status == "FILLED":
            n_legs += 1
        if exit_sell_fill and exit_sell_fill.status == "FILLED":
            n_legs += 1

        brokerage = BROKERAGE_PER_LOT_PER_LEG * lots * n_legs

        # STT on sell-side (selling the buy leg at exit)
        stt = 0.0
        if exit_buy_fill and exit_buy_fill.status == "FILLED":
            # exit_buy_fill is actually selling the buy leg (SELL to close)
            stt = exit_buy_fill.fill_price * spread.buy_leg.quantity * STT_PCT

        total_premium = 0.0
        if exit_buy_fill and exit_buy_fill.status == "FILLED":
            total_premium += exit_buy_fill.fill_price * spread.buy_leg.quantity
        if exit_sell_fill and exit_sell_fill.status == "FILLED":
            total_premium += exit_sell_fill.fill_price * spread.sell_leg.quantity
        slippage = total_premium * SLIPPAGE_PCT

        return brokerage + stt + slippage
