"""
Spread Monitor — tracks live spread positions and checks exit conditions.

Exit conditions:
  SL:   spread value drops 40% from entry (net_debit × 0.60)
  TGT:  spread value reaches 80% of max spread value
  TIME: 15:20 IST force exit

Usage:
    from execution.spread_monitor import SpreadMonitor
    monitor = SpreadMonitor()
    monitor.add_spread(signal_id, spread_order, spread_fill)
    exits = monitor.check_exits(current_time, price_feeds)
    live_pnl = monitor.get_live_pnl()
"""

import logging
from dataclasses import dataclass
from datetime import datetime, time as dt_time
from typing import Dict, List, Optional, Tuple

from execution.spread_builder import SpreadOrder
from execution.spread_executor import SpreadFill

logger = logging.getLogger(__name__)

FORCE_EXIT_TIME = dt_time(15, 20)

# Exit thresholds (as fraction of entry spread value)
SL_LOSS_PCT = 0.40      # exit if 40% of spread value lost
TGT_PROFIT_PCT = 0.80   # exit at 80% of max spread profit


# ================================================================
# DATACLASS
# ================================================================

@dataclass
class LiveSpread:
    """A tracked live spread position."""
    signal_id: str
    spread_order: SpreadOrder
    spread_fill: SpreadFill

    # Entry values
    entry_debit: float          # net debit per unit at entry
    entry_time: datetime

    # Computed thresholds
    sl_value: float             # exit if spread value drops to this (per unit)
    tgt_value: float            # exit if spread value rises to this (per unit)

    # Live tracking
    current_value: float = 0.0  # current spread value per unit
    unrealized_pnl: float = 0.0
    bars_held: int = 0
    underlying_entry: float = 0.0


@dataclass
class ExitSignal:
    """Signal to exit a spread."""
    signal_id: str
    reason: str                 # STOP_LOSS / TARGET / FORCE_EXIT
    current_spread_value: float
    entry_spread_value: float
    pnl_pct: float
    current_buy_premium: float
    current_sell_premium: float


# ================================================================
# SPREAD MONITOR
# ================================================================

class SpreadMonitor:
    """
    Monitors live spread positions and generates exit signals.
    Thread-safe for single-threaded bar-loop use.
    """

    def __init__(self):
        self._spreads: Dict[str, LiveSpread] = {}  # signal_id -> LiveSpread
        self._closed_pnl: float = 0.0
        self._total_premium_saved: float = 0.0

    # ================================================================
    # PUBLIC API
    # ================================================================

    def add_spread(
        self,
        signal_id: str,
        spread_order: SpreadOrder,
        spread_fill: SpreadFill,
        underlying_price: float = 0.0,
    ):
        """
        Register a new spread position for monitoring.

        Args:
            signal_id:        unique signal identifier
            spread_order:     the SpreadOrder from SpreadBuilder
            spread_fill:      the SpreadFill from SpreadExecutor
            underlying_price: current underlying price at entry
        """
        if signal_id in self._spreads:
            logger.warning(f"Spread {signal_id} already tracked — overwriting")

        entry_debit = spread_fill.net_entry_debit

        live = LiveSpread(
            signal_id=signal_id,
            spread_order=spread_order,
            spread_fill=spread_fill,
            entry_debit=entry_debit,
            entry_time=datetime.now(),
            sl_value=spread_order.sl_value,
            tgt_value=spread_order.tgt_value,
            current_value=entry_debit,
            underlying_entry=underlying_price or spread_order.underlying_ltp,
        )

        self._spreads[signal_id] = live
        self._total_premium_saved += spread_order.premium_saved * spread_order.buy_leg.quantity

        logger.info(
            f"SpreadMonitor: tracking {signal_id} | "
            f"entry={entry_debit:.2f} SL={live.sl_value:.2f} "
            f"TGT={live.tgt_value:.2f} | saved={spread_order.premium_saved:.2f}/unit"
        )

    def remove_spread(self, signal_id: str):
        """Remove a spread from monitoring (after exit)."""
        if signal_id in self._spreads:
            del self._spreads[signal_id]

    def check_exits(
        self,
        current_time: datetime,
        price_feeds: Optional[Dict[str, float]] = None,
    ) -> List[ExitSignal]:
        """
        Check all tracked spreads for exit conditions.

        Args:
            current_time: current datetime
            price_feeds:  dict of {tradingsymbol: current_premium} for live pricing.
                          If None, estimates from underlying movement.

        Returns:
            List of ExitSignal for spreads that should be closed.
        """
        exits = []
        now_time = current_time.time() if isinstance(current_time, datetime) else current_time

        for signal_id, live in list(self._spreads.items()):
            live.bars_held += 1
            spread = live.spread_order

            # ── Get current premiums ──
            buy_premium, sell_premium = self._get_current_premiums(
                live, price_feeds
            )

            # Current spread value per unit
            current_value = buy_premium - sell_premium
            current_value = max(current_value, 0.0)
            live.current_value = current_value

            quantity = spread.buy_leg.quantity
            live.unrealized_pnl = (current_value - live.entry_debit) * quantity

            exit_reason = None

            # ── TIME EXIT (15:20) ──
            if now_time >= FORCE_EXIT_TIME:
                exit_reason = "FORCE_EXIT"

            # ── STOP LOSS: spread value dropped 40% ──
            elif current_value <= live.sl_value:
                exit_reason = "STOP_LOSS"

            # ── TARGET: spread value hit 80% of max profit ──
            elif current_value >= live.tgt_value:
                exit_reason = "TARGET"

            if exit_reason:
                pnl_pct = (
                    (current_value - live.entry_debit) / live.entry_debit
                    if live.entry_debit > 0 else 0.0
                )

                exits.append(ExitSignal(
                    signal_id=signal_id,
                    reason=exit_reason,
                    current_spread_value=round(current_value, 2),
                    entry_spread_value=round(live.entry_debit, 2),
                    pnl_pct=round(pnl_pct, 4),
                    current_buy_premium=round(buy_premium, 2),
                    current_sell_premium=round(sell_premium, 2),
                ))

                logger.info(
                    f"SpreadMonitor EXIT: {signal_id} {exit_reason} | "
                    f"spread={current_value:.2f} (entry={live.entry_debit:.2f}) | "
                    f"pnl={pnl_pct:+.1%} | bars={live.bars_held}"
                )

        return exits

    def get_live_pnl(self) -> Dict:
        """
        Get unrealized P&L summary for all tracked spreads.

        Returns:
            dict with total_unrealized, positions count, per-spread detail
        """
        total_unrealized = 0.0
        details = []

        for signal_id, live in self._spreads.items():
            total_unrealized += live.unrealized_pnl
            details.append({
                "signal_id": signal_id,
                "strategy": live.spread_order.strategy,
                "entry_debit": live.entry_debit,
                "current_value": live.current_value,
                "unrealized_pnl": round(live.unrealized_pnl, 2),
                "bars_held": live.bars_held,
            })

        return {
            "total_unrealized": round(total_unrealized, 2),
            "positions": len(self._spreads),
            "total_premium_saved": round(self._total_premium_saved, 2),
            "details": details,
        }

    def get_spread(self, signal_id: str) -> Optional[LiveSpread]:
        """Get a tracked spread by signal_id."""
        return self._spreads.get(signal_id)

    @property
    def active_count(self) -> int:
        """Number of active spreads."""
        return len(self._spreads)

    @property
    def total_premium_saved(self) -> float:
        """Cumulative premium saved by using spreads vs naked buys."""
        return self._total_premium_saved

    # ================================================================
    # PRIVATE
    # ================================================================

    def _get_current_premiums(
        self,
        live: LiveSpread,
        price_feeds: Optional[Dict[str, float]],
    ) -> Tuple[float, float]:
        """
        Get current buy and sell leg premiums.

        If price_feeds available, use direct quotes.
        Otherwise, estimate from underlying delta movement.
        """
        spread = live.spread_order

        # Try price feeds first
        if price_feeds:
            buy_sym = spread.buy_leg.tradingsymbol
            buy_premium = price_feeds.get(buy_sym)
            if buy_premium is not None:
                sell_premium = 0.0
                if spread.sell_leg:
                    sell_sym = spread.sell_leg.tradingsymbol
                    sell_premium = price_feeds.get(sell_sym, 0.0)
                return buy_premium, sell_premium

        # Estimate from underlying movement (delta ≈ 0.50 for ATM)
        # This is a simplified model — in live mode, use actual quotes
        entry_buy = live.spread_fill.buy_fill.fill_price if live.spread_fill.buy_fill else spread.buy_leg.premium
        entry_sell = live.spread_fill.sell_fill.fill_price if live.spread_fill.sell_fill else 0.0

        # Without live quotes, use entry prices (no movement estimated)
        # The intraday_runner will provide underlying movement via price_feeds
        buy_premium = entry_buy
        sell_premium = entry_sell

        return buy_premium, sell_premium

    def update_from_underlying(
        self,
        signal_id: str,
        underlying_price: float,
    ):
        """
        Update spread value estimate from underlying price movement.
        Uses delta approximation.

        Args:
            signal_id:        signal to update
            underlying_price: current underlying price
        """
        live = self._spreads.get(signal_id)
        if live is None:
            return

        spread = live.spread_order
        delta_move = underlying_price - live.underlying_entry

        # Direction adjustment
        if spread.direction == "LONG":
            premium_change = delta_move * 0.50  # ATM delta
        else:
            premium_change = -delta_move * 0.50

        entry_buy = (
            live.spread_fill.buy_fill.fill_price
            if live.spread_fill.buy_fill
            else spread.buy_leg.premium
        )
        entry_sell = (
            live.spread_fill.sell_fill.fill_price
            if live.spread_fill.sell_fill
            else 0.0
        )

        # Buy leg moves more (ATM), sell leg moves less (OTM)
        current_buy = max(entry_buy + premium_change, 0.05)
        if spread.sell_leg:
            otm_delta_factor = 0.3  # OTM delta is lower
            current_sell = max(entry_sell + premium_change * otm_delta_factor, 0.05)
        else:
            current_sell = 0.0

        current_value = current_buy - current_sell
        current_value = max(current_value, 0.0)

        live.current_value = current_value
        quantity = spread.buy_leg.quantity
        live.unrealized_pnl = (current_value - live.entry_debit) * quantity
