"""
Realistic transaction cost model for Nifty F&O backtesting.

Implements the complete Zerodha cost structure for Nifty F&O trading
as of 2024, including:

- Brokerage: ₹20 per executed order (flat fee, both sides)
- STT (Securities Transaction Tax):
  - Futures sell side: 0.0125%
  - Options buy (premium): 0.1%
  - Options sell (premium): 0.0625%
- Exchange charges: 0.00173% per side (NSE futures), 0.0035% (options)
- GST: 18% on brokerage + exchange charges
- SEBI charges: ₹10 per crore traded
- Stamp duty: 0.002% on buy side
- Slippage: 0.02% base, VIX-scaled when VIX > 20

NIFTY_LOT_SIZE = 25 (post-2024 NSE change)

Usage in backtests:
    model = TransactionCostModel()
    costs = model.compute_futures_round_trip(entry=24000, exit=24100, lots=1)
    pnl_adjusted = gross_pnl - costs.total
    cost_bps = model.cost_as_pct(24000, 24100) * 10000  # in basis points
"""

from dataclasses import dataclass
from typing import Optional, Dict
import numpy as np


@dataclass
class CostConfig:
    """Transaction cost configuration for Zerodha Nifty F&O.

    All monetary values are in rupees (₹), percentages are 0-1 decimals.
    These reflect the official Zerodha fee schedule as of 2024.

    Attributes:
        brokerage_per_order: Flat brokerage per executed order (₹20).
        stt_futures_sell: STT rate on futures sell side (0.0125%).
        stt_options_buy_sell: STT on options bought/sold premium (0.1%/0.0625%).
        exchange_charge_futures: Exchange charges on futures (0.00173% per side).
        exchange_charge_options: Exchange charges on options (0.0035%).
        gst_rate: GST on brokerage + exchange charges (18%).
        sebi_per_crore: SEBI charges (₹10 per crore notional).
        stamp_duty_rate: Stamp duty on buy side (0.002%).
        base_slippage_pct: Base slippage for liquid Nifty futures (0.02%).
        vix_slippage_threshold: VIX level above which slippage increases.
        vix_slippage_multiplier: Slippage multiplier when VIX > threshold.
        lot_size: Nifty futures lot size post-2024 (25 contracts).
        default_lots: Default number of lots for cost calculations.
    """

    # Brokerage: ₹20/order × 1.18 (incl GST) = ₹23.60/order
    brokerage_per_order: float = 20.0
    brokerage_gst_inclusive: float = 23.60  # ₹20 × 1.18 GST

    # STT rates (Securities Transaction Tax)
    stt_futures_sell: float = 0.000125      # 0.0125% on sell side
    stt_options_buy_sell: float = 0.001     # 0.1% on sell premium (options buying)
    stt_options_sell_sell: float = 0.000625 # 0.0625% on sell premium (options selling)

    # Exchange charges: 0.0495% on turnover (both sides)
    exchange_charge_futures: float = 0.000495   # 0.0495%
    exchange_charge_options: float = 0.000495   # 0.0495%

    # GST: 18% on (brokerage + exchange charges)
    gst_rate: float = 0.18

    # SEBI charges: 0.0001% on turnover (₹10 per crore)
    sebi_per_crore: float = 10.0
    sebi_rate: float = 0.000001  # 0.0001%

    # Stamp duty: 0.003% on buy side
    stamp_duty_rate: float = 0.00003  # 0.003%

    # Slippage model
    base_slippage_pct: float = 0.0002  # 0.02% per side for Nifty futures
    vix_slippage_threshold: float = 20.0  # VIX level above which slippage scales
    vix_slippage_multiplier: float = 2.0  # 2x slippage when VIX > threshold

    # Lot size and position sizing
    lot_size: int = 25  # Nifty lot size post-2024
    default_lots: int = 1  # Default position size


@dataclass
class TradeCosts:
    """Itemized breakdown of transaction costs for a single trade.

    Attributes:
        brokerage: Flat brokerage fee (₹).
        stt: Securities Transaction Tax (₹).
        exchange_charges: Exchange fees (₹).
        gst: GST on brokerage + exchange (₹).
        sebi_charges: SEBI regulatory fees (₹).
        stamp_duty: Stamp duty on entry (₹).
        slippage: Estimated slippage on entry + exit (₹).
    """

    brokerage: float = 0.0
    stt: float = 0.0
    exchange_charges: float = 0.0
    gst: float = 0.0
    sebi_charges: float = 0.0
    stamp_duty: float = 0.0
    slippage: float = 0.0

    @property
    def total(self) -> float:
        """Total cost across all components (₹)."""
        return (self.brokerage + self.stt + self.exchange_charges +
                self.gst + self.sebi_charges + self.stamp_duty + self.slippage)

    @property
    def total_pct(self) -> float:
        """Total cost as fraction of notional. Must be set externally."""
        return 0.0

    def __repr__(self) -> str:
        """Human-readable representation of costs."""
        return (f"TradeCosts(total=₹{self.total:.2f}, "
                f"brok=₹{self.brokerage:.2f}, stt=₹{self.stt:.2f}, "
                f"exch=₹{self.exchange_charges:.2f}, gst=₹{self.gst:.2f}, "
                f"slip=₹{self.slippage:.2f})")


class TransactionCostModel:
    """Computes realistic transaction costs for Nifty F&O trades.

    Implements complete Zerodha cost structure with Nifty-specific defaults.
    Provides methods to compute costs for futures round-trips, convert to
    Nifty points for backtest P&L adjustment, and generate cost summaries.

    Attributes:
        config: CostConfig instance with fee structure.
    """

    def __init__(self, config: Optional[CostConfig] = None):
        """Initialize the transaction cost model.

        Args:
            config: CostConfig instance. If None, uses Zerodha defaults.
        """
        self.config = config or CostConfig()

    def compute_futures_round_trip(self, entry_price: float, exit_price: float,
                                    lots: Optional[int] = None,
                                    vix: Optional[float] = None) -> TradeCosts:
        """Compute all costs for a futures round-trip (entry + exit).

        Calculates itemized costs including brokerage, STT, exchange charges,
        GST, SEBI fees, stamp duty, and slippage. All costs are in rupees.

        Args:
            entry_price: Nifty level at entry (e.g., 24000).
            exit_price: Nifty level at exit (e.g., 24100).
            lots: Number of lots traded. Defaults to config.default_lots.
            vix: India VIX at time of trade. Used to scale slippage.

        Returns:
            TradeCosts with itemized breakdown. Access .total for aggregate cost.

        Example:
            >>> model = TransactionCostModel()
            >>> costs = model.compute_futures_round_trip(24000, 24100, lots=1)
            >>> print(f"Total cost: ₹{costs.total:.2f}")
            >>> print(f"As points: {costs.total / 25:.3f}")
        """
        lots = lots or self.config.default_lots
        lot_size = self.config.lot_size

        # Notional values (contract value = price × lot_size × lots)
        entry_notional = entry_price * lot_size * lots
        exit_notional = exit_price * lot_size * lots

        costs = TradeCosts()

        # Brokerage: ₹20/order × 1.18 GST = ₹23.60/order × 2 (entry + exit)
        costs.brokerage = self.config.brokerage_gst_inclusive * 2

        # STT: on sell side only (exit)
        costs.stt = exit_notional * self.config.stt_futures_sell

        # Exchange charges: 0.0495% on turnover (both sides)
        costs.exchange_charges = (
            entry_notional * self.config.exchange_charge_futures +
            exit_notional * self.config.exchange_charge_futures
        )

        # GST: 18% on (exchange charges only; brokerage GST already included)
        costs.gst = costs.exchange_charges * self.config.gst_rate

        # SEBI charges: 0.0001% on turnover
        total_notional = entry_notional + exit_notional
        costs.sebi_charges = total_notional * self.config.sebi_rate

        # Stamp duty: 0.003% on buy side only (entry)
        costs.stamp_duty = entry_notional * self.config.stamp_duty_rate

        # Slippage: VIX-scaled, applied to both entry and exit
        slippage_pct = self.config.base_slippage_pct
        if vix and vix > self.config.vix_slippage_threshold:
            # Scale slippage based on VIX excess above threshold
            vix_excess = (vix - self.config.vix_slippage_threshold) / 10.0
            slippage_pct *= (1.0 + vix_excess * (self.config.vix_slippage_multiplier - 1.0))
        # Slippage on both entry and exit notional
        costs.slippage = (entry_notional + exit_notional) * slippage_pct

        return costs

    def cost_in_points(self, entry_price: float, exit_price: float,
                       lots: Optional[int] = None,
                       vix: Optional[float] = None) -> float:
        """Return total round-trip cost in Nifty points.

        Converts rupee costs to Nifty index points, useful for subtracting
        from P&L in backtest engines that work in points.

        Formula: cost_points = total_cost_rupees / (lot_size × lots)

        Args:
            entry_price: Nifty entry level.
            exit_price: Nifty exit level.
            lots: Number of lots (default from config).
            vix: India VIX at trade time.

        Returns:
            Total cost in Nifty points (e.g., 7.5 points).

        Example:
            >>> cost_pts = model.cost_in_points(24000, 24100, lots=1, vix=18)
            >>> print(f"Cost: {cost_pts:.3f} points")
        """
        costs = self.compute_futures_round_trip(entry_price, exit_price, lots, vix)
        lot_size = self.config.lot_size
        lots_used = lots or self.config.default_lots
        # Convert ₹ cost to points: cost / (lot_size × lots)
        return costs.total / (lot_size * lots_used)

    def cost_as_pct(self, entry_price: float, exit_price: float,
                    lots: Optional[int] = None,
                    vix: Optional[float] = None) -> float:
        """Return total cost as percentage of notional value.

        Useful for comparing costs across different price levels and
        expressing costs in basis points or percentage terms.

        Args:
            entry_price: Nifty entry level.
            exit_price: Nifty exit level.
            lots: Number of lots.
            vix: India VIX at trade time.

        Returns:
            Cost as decimal (e.g., 0.00035 = 3.5 basis points).

        Example:
            >>> cost_pct = model.cost_as_pct(24000, 24100, lots=1)
            >>> print(f"Cost: {cost_pct * 10000:.1f} bps")  # basis points
        """
        cost_pts = self.cost_in_points(entry_price, exit_price, lots, vix)
        avg_price = (entry_price + exit_price) / 2
        return cost_pts / avg_price if avg_price > 0 else 0.0

    def summary_stats(self, entry_price: float = 24000,
                      lots: int = 1) -> Dict:
        """Generate summary statistics for a typical Nifty trade.

        Useful for understanding the cost structure at a glance.
        Computes costs for a flat round-trip (entry = exit price).

        Args:
            entry_price: Nifty level for the trade (default 24000).
            lots: Number of lots (default 1).

        Returns:
            Dict with keys:
                'notional': Total notional value in rupees
                'total_cost': Total cost in rupees
                'cost_pct': Cost as percentage (0-1 decimal)
                'cost_points': Cost in Nifty points
                'breakdown': Dict with individual cost components

        Example:
            >>> stats = model.summary_stats(entry_price=24000, lots=1)
            >>> print(f"Cost for 1 lot at 24000: ₹{stats['total_cost']:.0f}")
            >>> print(f"  {stats['cost_pct'] * 100:.3f}% of notional")
        """
        exit_price = entry_price  # Flat trade for cost estimation
        costs = self.compute_futures_round_trip(entry_price, exit_price, lots)
        notional = entry_price * self.config.lot_size * lots
        return {
            'notional': notional,
            'total_cost': costs.total,
            'cost_pct': costs.total / notional * 100 if notional > 0 else 0.0,
            'cost_points': self.cost_in_points(entry_price, exit_price, lots),
            'breakdown': {
                'brokerage': costs.brokerage,
                'stt': costs.stt,
                'exchange_charges': costs.exchange_charges,
                'gst': costs.gst,
                'sebi_charges': costs.sebi_charges,
                'stamp_duty': costs.stamp_duty,
                'slippage': costs.slippage,
            }
        }
