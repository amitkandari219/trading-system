"""
Spread Builder — selects multi-leg options strategy based on VIX regime.

Decision matrix:
  VIX < 14           → NAKED_BUY (cheap premiums, buy outright)
  VIX 14-20 + dir    → BULL_CALL / BEAR_PUT spread (2-3 strikes wide)
  VIX 14-20 + MR     → tighter spread (1-2 strikes wide)
  VIX 20-25          → wider spread (3-4 strikes wide)
  VIX > 25           → tightest spread (1 strike wide)
  GAMMA signals      → always NAKED_BUY

Safety invariant: every SELL leg must have a paired BUY leg.

Usage:
    from execution.spread_builder import SpreadBuilder
    builder = SpreadBuilder(kite=kite, paper_mode=True)
    spread = builder.select_strategy(
        signal_id='KAUFMAN_BB_MR', direction='LONG',
        instrument='NIFTY', price=23500, atr=300,
        vix=16.5, regime='NORMAL', equity=200_000,
    )
"""

import logging
import math
import os
from dataclasses import dataclass, field
from datetime import date, datetime, time as dt_time
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

EXECUTION_MODE = os.environ.get("EXECUTION_MODE", "PAPER").upper()

# ================================================================
# CONSTANTS
# ================================================================

NIFTY_LOT_SIZE = 25
BANKNIFTY_LOT_SIZE = 15

NIFTY_STRIKE_INTERVAL = 50
BANKNIFTY_STRIKE_INTERVAL = 100

# OI liquidity floor — do not sell strikes with OI below this
MIN_OI_FOR_SELL = 10_000

# Spread width bounds (in strike intervals)
MIN_SPREAD_WIDTH = 1
MAX_SPREAD_WIDTH = 5

# Strategy types — debit
NAKED_BUY = "NAKED_BUY"
BULL_CALL_SPREAD = "BULL_CALL_SPREAD"
BEAR_PUT_SPREAD = "BEAR_PUT_SPREAD"

# Strategy types — credit (Layer 8)
BULL_PUT_SPREAD = "BULL_PUT_SPREAD"
BEAR_CALL_SPREAD = "BEAR_CALL_SPREAD"
SHORT_STRANGLE = "SHORT_STRANGLE"
SHORT_STRADDLE = "SHORT_STRADDLE"
IRON_CONDOR = "IRON_CONDOR"

# Gamma signal IDs that always get naked buys
GAMMA_SIGNAL_IDS = frozenset({
    "GAMMA_SCALP", "GAMMA_SQUEEZE", "GAMMA_FLIP",
    "GAMMA_PIN", "GAMMA_EXPOSURE",
})

# Mean-reversion signal IDs (get tighter spreads)
MR_SIGNAL_IDS = frozenset({
    "KAUFMAN_BB_MR", "GUJRAL_RANGE", "BN_KAUFMAN_BB_MR", "BN_GUJRAL_RANGE",
})


# ================================================================
# DATACLASS
# ================================================================

@dataclass
class SpreadLeg:
    """Single leg of a spread."""
    tradingsymbol: str
    instrument: str
    option_type: str        # CE / PE
    strike: int
    transaction_type: str   # BUY / SELL
    premium: float          # estimated premium per unit
    lot_size: int
    lots: int
    quantity: int           # lots × lot_size
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    oi: int = 0


@dataclass
class SpreadOrder:
    """Complete spread order with both legs and risk metrics."""
    signal_id: str
    strategy: str           # NAKED_BUY / BULL_CALL_SPREAD / BEAR_PUT_SPREAD
    direction: str          # LONG / SHORT (from signal)
    instrument: str
    expiry_str: str         # YYMMDD

    buy_leg: SpreadLeg
    sell_leg: Optional[SpreadLeg]  # None for NAKED_BUY

    lots: int
    lot_size: int

    # Cost / risk
    net_debit: float        # net premium paid per unit (buy - sell credit)
    max_loss: float         # total max loss (net_debit × quantity for spreads)
    max_profit: float       # total max profit
    breakeven: float        # breakeven underlying price

    # Stop / target on spread value
    sl_value: float         # exit if spread value drops to this (per unit)
    tgt_value: float        # exit if spread value rises to this (per unit)

    # Greeks (combined)
    net_delta: float = 0.0
    net_gamma: float = 0.0
    net_theta: float = 0.0
    net_vega: float = 0.0

    # Premium savings vs naked buy
    naked_cost: float = 0.0
    premium_saved: float = 0.0

    # Metadata
    spread_width: int = 0   # in strike intervals
    entry_time: str = ""
    underlying_ltp: float = 0.0
    vix_at_entry: float = 0.0


# ================================================================
# SPREAD BUILDER
# ================================================================

class SpreadBuilder:
    """
    Selects the optimal options strategy based on VIX, regime, and signal type.
    Builds SpreadOrder with both legs, greeks, and risk parameters.
    """

    def __init__(self, kite=None, paper_mode: bool = True):
        self.kite = kite
        self.paper_mode = paper_mode or (EXECUTION_MODE == "PAPER")
        self._oi_cache: Dict[str, int] = {}  # strike_key -> OI

    # ================================================================
    # PUBLIC API
    # ================================================================

    def select_strategy(
        self,
        signal_id: str,
        direction: str,
        instrument: str,
        price: float,
        atr: float,
        vix: float,
        regime: str,
        equity: float,
        expiry_date: Optional[str] = None,
        lots: int = 1,
    ) -> Optional[SpreadOrder]:
        """
        Select the best options strategy for this signal and market context.

        Args:
            signal_id:  e.g. 'KAUFMAN_BB_MR'
            direction:  'LONG' or 'SHORT'
            instrument: 'NIFTY' or 'BANKNIFTY'
            price:      current underlying price
            atr:        daily ATR
            vix:        India VIX
            regime:     regime string from DynamicRegimeManager
            equity:     current account equity
            expiry_date: option expiry in 'YYMMDD' format
            lots:       number of lots (from compound sizer)

        Returns:
            SpreadOrder or None if construction fails.
        """
        direction = direction.upper()
        if direction not in ("LONG", "SHORT"):
            logger.error(f"Invalid direction: {direction}")
            return None

        # Determine strategy type and spread width
        strategy, width = self._decide_strategy(signal_id, direction, vix, regime)

        # ATM strike
        atm = self._find_atm_strike(price, instrument)
        lot_size = NIFTY_LOT_SIZE if instrument == "NIFTY" else BANKNIFTY_LOT_SIZE
        interval = NIFTY_STRIKE_INTERVAL if instrument == "NIFTY" else BANKNIFTY_STRIKE_INTERVAL

        # Option type from signal direction
        option_type = "CE" if direction == "LONG" else "PE"

        # Expiry
        expiry_str = expiry_date or self._next_weekly_expiry()

        # Estimate premiums
        atm_premium = self._estimate_premium(price, atr, vix, 0)

        # Build buy leg (always ATM)
        buy_strike = atm
        buy_symbol = f"{instrument}{expiry_str}{buy_strike}{option_type}"
        buy_premium = atm_premium

        buy_leg = SpreadLeg(
            tradingsymbol=buy_symbol,
            instrument=instrument,
            option_type=option_type,
            strike=buy_strike,
            transaction_type="BUY",
            premium=round(buy_premium, 2),
            lot_size=lot_size,
            lots=lots,
            quantity=lots * lot_size,
        )

        sell_leg = None
        net_debit = buy_premium
        naked_cost = buy_premium

        if strategy == NAKED_BUY:
            # No sell leg
            max_loss_per_unit = buy_premium
            # Theoretical max profit is unlimited for calls, capped for puts
            max_profit_per_unit = buy_premium * 2.0  # conservative estimate

            sl_value = round(buy_premium * 0.60, 2)   # 40% loss
            tgt_value = round(buy_premium * 1.80, 2)   # 80% gain

            breakeven = (
                buy_strike + buy_premium if option_type == "CE"
                else buy_strike - buy_premium
            )

        else:
            # Build sell leg (OTM)
            sell_strike = self._select_otm_strike(
                atm, option_type, direction, width, instrument
            )

            # Validate OI on sell strike
            sell_oi = self._get_oi(instrument, sell_strike, option_type, expiry_str)
            if sell_oi < MIN_OI_FOR_SELL and not self.paper_mode:
                logger.warning(
                    f"OI too low for sell strike {sell_strike} "
                    f"(OI={sell_oi} < {MIN_OI_FOR_SELL}) — falling back to NAKED_BUY"
                )
                return self._build_naked_order(
                    signal_id, direction, instrument, buy_leg,
                    buy_premium, expiry_str, lots, lot_size,
                    price, vix, atm_premium,
                )

            sell_premium = self._estimate_premium(
                price, atr, vix, abs(sell_strike - atm)
            )
            sell_symbol = f"{instrument}{expiry_str}{sell_strike}{option_type}"

            sell_leg = SpreadLeg(
                tradingsymbol=sell_symbol,
                instrument=instrument,
                option_type=option_type,
                strike=sell_strike,
                transaction_type="SELL",
                premium=round(sell_premium, 2),
                lot_size=lot_size,
                lots=lots,
                quantity=lots * lot_size,
                oi=sell_oi,
            )

            net_debit = buy_premium - sell_premium
            net_debit = max(net_debit, 0.05)  # floor

            spread_pts = abs(buy_strike - sell_strike)
            max_loss_per_unit = net_debit
            max_profit_per_unit = spread_pts - net_debit
            max_profit_per_unit = max(max_profit_per_unit, 0.0)

            # SL: 40% loss on spread value
            sl_value = round(net_debit * 0.60, 2)
            # TGT: 80% of max spread value
            tgt_value = round(net_debit + max_profit_per_unit * 0.80, 2)

            if option_type == "CE":
                breakeven = buy_strike + net_debit
            else:
                breakeven = buy_strike - net_debit

        # Compute combined greeks
        net_delta, net_gamma, net_theta, net_vega = self._compute_spread_greeks(
            buy_leg, sell_leg
        )

        quantity = lots * lot_size
        premium_saved = (naked_cost - net_debit) if sell_leg else 0.0

        spread = SpreadOrder(
            signal_id=signal_id,
            strategy=strategy,
            direction=direction,
            instrument=instrument,
            expiry_str=expiry_str,
            buy_leg=buy_leg,
            sell_leg=sell_leg,
            lots=lots,
            lot_size=lot_size,
            net_debit=round(net_debit, 2),
            max_loss=round(max_loss_per_unit * quantity, 2),
            max_profit=round(max_profit_per_unit * quantity, 2),
            breakeven=round(breakeven, 2),
            sl_value=sl_value,
            tgt_value=tgt_value,
            net_delta=net_delta,
            net_gamma=net_gamma,
            net_theta=net_theta,
            net_vega=net_vega,
            naked_cost=round(naked_cost, 2),
            premium_saved=round(premium_saved, 2),
            spread_width=abs(buy_leg.strike - sell_leg.strike) // interval if sell_leg else 0,
            entry_time=datetime.now().strftime("%H:%M:%S"),
            underlying_ltp=price,
            vix_at_entry=vix,
        )

        logger.info(
            f"SpreadBuilder: {strategy} for {signal_id} {direction} | "
            f"buy={buy_leg.tradingsymbol}@{buy_leg.premium} "
            f"{'sell=' + sell_leg.tradingsymbol + '@' + str(sell_leg.premium) if sell_leg else ''} | "
            f"net_debit={net_debit:.2f} max_loss={spread.max_loss:,.0f} "
            f"max_profit={spread.max_profit:,.0f} | "
            f"saved={premium_saved:.2f}/unit VIX={vix:.1f}"
        )

        return spread

    # ================================================================
    # STRATEGY DECISION
    # ================================================================

    def _decide_strategy(
        self,
        signal_id: str,
        direction: str,
        vix: float,
        regime: str,
    ) -> Tuple[str, int]:
        """
        Decide strategy type and spread width based on VIX and signal.

        Returns:
            (strategy_type, width_in_strike_intervals)
        """
        # Gamma signals always naked buy
        if signal_id in GAMMA_SIGNAL_IDS or signal_id.startswith("GAMMA_"):
            return NAKED_BUY, 0

        is_mr = signal_id in MR_SIGNAL_IDS

        # VIX < 14: cheap premiums, naked buy
        if vix < 14:
            return NAKED_BUY, 0

        # VIX 14-20
        if 14 <= vix < 20:
            if is_mr:
                width = 1 if vix < 17 else 2
            else:
                width = 2 if vix < 17 else 3

        # VIX 20-25: wider spreads for premium protection
        elif 20 <= vix < 25:
            width = 3 if is_mr else 4

        # VIX > 25: tightest spread to limit cost
        else:
            width = 1

        # Clamp
        width = max(MIN_SPREAD_WIDTH, min(width, MAX_SPREAD_WIDTH))

        # Strategy type based on direction
        if direction == "LONG":
            strategy = BULL_CALL_SPREAD
        else:
            strategy = BEAR_PUT_SPREAD

        return strategy, width

    # ================================================================
    # STRIKE SELECTION
    # ================================================================

    def _find_atm_strike(self, price: float, instrument: str) -> int:
        """Round to nearest strike interval."""
        interval = (
            NIFTY_STRIKE_INTERVAL if instrument == "NIFTY"
            else BANKNIFTY_STRIKE_INTERVAL
        )
        return int(round(price / interval) * interval)

    def _select_otm_strike(
        self,
        atm: int,
        option_type: str,
        direction: str,
        width: int,
        instrument: str,
    ) -> int:
        """
        Select the OTM strike for the sell leg.

        For BULL_CALL_SPREAD (LONG + CE): sell OTM call above ATM
        For BEAR_PUT_SPREAD (SHORT + PE): sell OTM put below ATM
        """
        interval = (
            NIFTY_STRIKE_INTERVAL if instrument == "NIFTY"
            else BANKNIFTY_STRIKE_INTERVAL
        )
        offset = width * interval

        if option_type == "CE":
            # Sell higher strike call
            return atm + offset
        else:
            # Sell lower strike put
            return atm - offset

    # ================================================================
    # PREMIUM & GREEKS ESTIMATION
    # ================================================================

    def _estimate_premium(
        self,
        underlying: float,
        atr: float,
        vix: float,
        distance_from_atm: float,
    ) -> float:
        """
        Estimate option premium. ATM uses VIX-scaled model.
        OTM decays exponentially with distance.
        """
        # ATM premium ≈ underlying × VIX/100 × sqrt(7/365) for weekly
        # Simplified: scale off ATR
        base = max(50, min(120 + 0.8 * atr, 500))

        # VIX adjustment
        vix_factor = max(0.7, vix / 15.0)
        base *= vix_factor

        if distance_from_atm <= 0:
            return round(base, 2)

        # OTM decay: exponential falloff with moneyness
        moneyness = distance_from_atm / underlying
        decay = math.exp(-8.0 * moneyness)
        otm_premium = base * decay
        otm_premium = max(otm_premium, 1.0)  # floor at ₹1

        return round(otm_premium, 2)

    def _compute_spread_greeks(
        self,
        buy_leg: SpreadLeg,
        sell_leg: Optional[SpreadLeg],
    ) -> Tuple[float, float, float, float]:
        """
        Compute net greeks for the spread.
        Uses simple estimates since live greeks may not be available.
        """
        # ATM buy: delta ~0.50
        buy_delta = 0.50
        buy_gamma = 0.002
        buy_theta = -10.0
        buy_vega = 5.0

        buy_leg.delta = buy_delta
        buy_leg.gamma = buy_gamma
        buy_leg.theta = buy_theta
        buy_leg.vega = buy_vega

        if sell_leg is None:
            return buy_delta, buy_gamma, buy_theta, buy_vega

        # OTM sell: lower delta
        distance_ratio = abs(sell_leg.strike - buy_leg.strike) / buy_leg.strike
        sell_delta = max(0.10, 0.50 - distance_ratio * 10)
        sell_gamma = buy_gamma * 0.6
        sell_theta = buy_theta * 0.5  # less theta decay on OTM
        sell_vega = buy_vega * 0.7

        sell_leg.delta = -sell_delta  # short
        sell_leg.gamma = -sell_gamma
        sell_leg.theta = -sell_theta  # positive theta from short
        sell_leg.vega = -sell_vega

        net_delta = buy_delta - sell_delta
        net_gamma = buy_gamma - sell_gamma
        net_theta = buy_theta + abs(sell_theta)  # short theta offsets
        net_vega = buy_vega - sell_vega

        return (
            round(net_delta, 4),
            round(net_gamma, 6),
            round(net_theta, 2),
            round(net_vega, 2),
        )

    # ================================================================
    # OI / OPTION CHAIN
    # ================================================================

    def _get_oi(
        self,
        instrument: str,
        strike: int,
        option_type: str,
        expiry_str: str,
    ) -> int:
        """
        Get open interest for a strike. In paper mode, returns a simulated value.
        In live mode, fetches from Kite quote.
        """
        cache_key = f"{instrument}:{strike}:{option_type}:{expiry_str}"
        if cache_key in self._oi_cache:
            return self._oi_cache[cache_key]

        if self.paper_mode:
            # Simulate: ATM has high OI, decays OTM
            oi = 50_000  # default high enough
            self._oi_cache[cache_key] = oi
            return oi

        if self.kite is None:
            return 50_000  # safe default

        try:
            symbol = f"{instrument}{expiry_str}{strike}{option_type}"
            quote_key = f"NFO:{symbol}"
            quote = self.kite.quote(quote_key)
            oi = int(quote.get(quote_key, {}).get("oi", 0))
            self._oi_cache[cache_key] = oi
            return oi
        except Exception as e:
            logger.warning(f"OI fetch failed for {cache_key}: {e}")
            return 0

    def _get_option_chain(
        self,
        instrument: str,
        expiry_str: str,
    ) -> Optional[List[Dict]]:
        """
        Fetch option chain from Kite LTP or DB.
        Returns list of dicts with strike, option_type, ltp, oi.
        """
        if self.paper_mode:
            return None  # use estimates

        if self.kite is None:
            return None

        # In live mode, this would query Kite for the full chain
        # For now, we rely on per-strike lookups in _get_oi
        return None

    # ================================================================
    # HELPERS
    # ================================================================

    def _build_naked_order(
        self,
        signal_id: str,
        direction: str,
        instrument: str,
        buy_leg: SpreadLeg,
        buy_premium: float,
        expiry_str: str,
        lots: int,
        lot_size: int,
        price: float,
        vix: float,
        naked_cost: float,
    ) -> SpreadOrder:
        """Build a NAKED_BUY SpreadOrder (fallback when sell leg fails OI check)."""
        option_type = buy_leg.option_type
        quantity = lots * lot_size
        max_loss_per_unit = buy_premium
        max_profit_per_unit = buy_premium * 2.0

        sl_value = round(buy_premium * 0.60, 2)
        tgt_value = round(buy_premium * 1.80, 2)

        breakeven = (
            buy_leg.strike + buy_premium if option_type == "CE"
            else buy_leg.strike - buy_premium
        )

        return SpreadOrder(
            signal_id=signal_id,
            strategy=NAKED_BUY,
            direction=direction,
            instrument=instrument,
            expiry_str=expiry_str,
            buy_leg=buy_leg,
            sell_leg=None,
            lots=lots,
            lot_size=lot_size,
            net_debit=round(buy_premium, 2),
            max_loss=round(max_loss_per_unit * quantity, 2),
            max_profit=round(max_profit_per_unit * quantity, 2),
            breakeven=round(breakeven, 2),
            sl_value=sl_value,
            tgt_value=tgt_value,
            naked_cost=round(naked_cost, 2),
            premium_saved=0.0,
            spread_width=0,
            entry_time=datetime.now().strftime("%H:%M:%S"),
            underlying_ltp=price,
            vix_at_entry=vix,
        )

    def _next_weekly_expiry(self) -> str:
        """Estimate next weekly expiry in YYMMDD format."""
        from datetime import timedelta
        today = date.today()
        days_ahead = (3 - today.weekday()) % 7
        if days_ahead == 0 and datetime.now().time() > dt_time(15, 30):
            days_ahead = 7
        expiry = today + timedelta(days=days_ahead)
        return expiry.strftime("%y%m%d")
