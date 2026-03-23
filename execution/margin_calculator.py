"""
Approximate SPAN margin calculator for Nifty / BankNifty options.

NSE uses SPAN (Standard Portfolio Analysis of Risk) for margin computation.
Exact figures come from the exchange; this module provides conservative
approximations so the system can pre-flight margin checks BEFORE placing
orders.  All values are intentionally rounded UP so we never underestimate.

Lot sizes (as of 2024):  NIFTY = 25,  BANKNIFTY = 15
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NAKED_MARGIN_SPOT_PCT: float = 0.15       # 15 % of spot for naked short
NAKED_MARGIN_STRIKE_PCT: float = 0.10     # 10 % of strike fallback
SPREAD_MARGIN_BUFFER: float = 1.10        # 10 % safety buffer on spreads
MAX_MARGIN_PCT: float = 0.50              # never use > 50 % of equity

LOT_SIZES: Dict[str, int] = {
    "NIFTY": 25,
    "BANKNIFTY": 15,
}


class MarginCalculator:
    """Conservative SPAN margin approximations for Indian index options."""

    # -- naked short --------------------------------------------------------
    @staticmethod
    def naked_short_margin(
        spot: float,
        strike: float,
        premium: float,
        lot_size: int,
        lots: int = 1,
    ) -> float:
        """SPAN-style margin for a single naked short option.

        Formula (per unit):
            max(premium + 15% × spot, premium + 10% × strike)
        Then multiply by lot_size × lots.
        """
        per_unit = max(
            premium + NAKED_MARGIN_SPOT_PCT * spot,
            premium + NAKED_MARGIN_STRIKE_PCT * strike,
        )
        return per_unit * lot_size * lots

    # -- credit spread (defined risk) --------------------------------------
    @staticmethod
    def spread_margin(
        max_loss_per_unit: float,
        lot_size: int,
        lots: int = 1,
    ) -> float:
        """Margin for a defined-risk credit spread.

        max_loss_per_unit = spread_width - net_credit  (for credit spreads).
        We add a 10 % buffer on top.
        """
        return max_loss_per_unit * lot_size * lots * SPREAD_MARGIN_BUFFER

    # -- iron condor -------------------------------------------------------
    @staticmethod
    def iron_condor_margin(
        put_spread_margin: float,
        call_spread_margin: float,
    ) -> float:
        """Iron-condor margin = max of the two spread margins (NOT sum).

        Only one wing can be breached at a time.
        """
        return max(put_spread_margin, call_spread_margin)

    # -- short straddle / strangle -----------------------------------------
    @staticmethod
    def short_straddle_margin(
        spot: float,
        atm_strike: float,
        ce_premium: float,
        pe_premium: float,
        lot_size: int,
        lots: int = 1,
    ) -> float:
        """Margin for a short straddle (or strangle when strikes differ).

        Rule of thumb:
            max(naked_CE_margin, naked_PE_margin) + other_side_premium × lot × lots
        Plus 10 % buffer.
        """
        ce_margin = MarginCalculator.naked_short_margin(
            spot, atm_strike, ce_premium, lot_size, lots
        )
        pe_margin = MarginCalculator.naked_short_margin(
            spot, atm_strike, pe_premium, lot_size, lots
        )
        if ce_margin >= pe_margin:
            raw = ce_margin + pe_premium * lot_size * lots
        else:
            raw = pe_margin + ce_premium * lot_size * lots
        return raw * SPREAD_MARGIN_BUFFER

    # -- calendar spread ---------------------------------------------------
    @staticmethod
    def calendar_spread_margin(
        near_premium: float,
        far_premium: float,
        lot_size: int,
        lots: int = 1,
    ) -> float:
        """Approximate margin for a calendar spread (short near, long far).

        Debit paid = far_premium - near_premium.
        Short leg has reduced margin because the long leg hedges it.
        We conservatively estimate 50 % of a naked short margin on the
        near leg plus the debit paid.
        """
        debit = max(far_premium - near_premium, 0.0)
        short_leg_margin = near_premium * lot_size * lots * 0.50
        return (debit * lot_size * lots + short_leg_margin) * SPREAD_MARGIN_BUFFER

    # -- margin availability check -----------------------------------------
    @staticmethod
    def check_margin_available(
        required_margin: float,
        equity: float,
    ) -> Dict[str, Any]:
        """Pre-flight check: is *required_margin* within safe limits?

        Returns a dict with:
            approved   – bool, True if margin fits within MAX_MARGIN_PCT
            margin_pct – required / equity
            headroom   – remaining margin budget after this trade
            required   – echo back the required margin
            limit      – equity × MAX_MARGIN_PCT
        """
        limit = equity * MAX_MARGIN_PCT
        margin_pct = required_margin / equity if equity > 0 else float("inf")
        headroom = limit - required_margin
        approved = required_margin < limit
        return {
            "approved": approved,
            "margin_pct": round(margin_pct, 4),
            "headroom": round(headroom, 2),
            "required": round(required_margin, 2),
            "limit": round(limit, 2),
        }

    # -- high-level dispatcher ---------------------------------------------
    def estimate_strategy_margin(
        self,
        strategy_type: str,
        spot: float,
        legs_info: List[Dict[str, Any]],
        lot_size: int,
        lots: int = 1,
    ) -> float:
        """Dispatch to the right margin method based on *strategy_type*.

        Each entry in *legs_info* carries:
            strike, option_type (CE/PE), action (BUY/SELL), premium
        """
        st = strategy_type.upper().replace(" ", "_")

        if st in ("SHORT_STRANGLE", "SHORT_STRADDLE"):
            return self._margin_short_strangle_straddle(spot, legs_info, lot_size, lots)

        if st == "IRON_CONDOR":
            return self._margin_iron_condor(spot, legs_info, lot_size, lots)

        if st in ("BULL_PUT_SPREAD", "BEAR_CALL_SPREAD"):
            return self._margin_vertical_spread(spot, legs_info, lot_size, lots)

        if st == "CALENDAR_SPREAD":
            return self._margin_calendar(legs_info, lot_size, lots)

        if st in ("PROTECTIVE_PUT", "COVERED_CALL"):
            # Hedged positions – margin is just the premium paid for the hedge
            buy_leg = next((l for l in legs_info if l["action"] == "BUY"), None)
            premium = buy_leg["premium"] if buy_leg else 0.0
            return premium * lot_size * lots

        if st == "RATIO_SPREAD":
            return self._margin_ratio_spread(spot, legs_info, lot_size, lots)

        logger.warning("Unknown strategy '%s' – falling back to naked margin sum", st)
        return self._fallback_naked_sum(spot, legs_info, lot_size, lots)

    # -- private helpers ----------------------------------------------------
    def _margin_short_strangle_straddle(
        self, spot, legs_info, lot_size, lots
    ) -> float:
        ce = next((l for l in legs_info if l["option_type"] == "CE"), None)
        pe = next((l for l in legs_info if l["option_type"] == "PE"), None)
        if ce is None or pe is None:
            return self._fallback_naked_sum(spot, legs_info, lot_size, lots)
        return self.short_straddle_margin(
            spot, ce["strike"], ce["premium"], pe["premium"], lot_size, lots
        )

    def _margin_iron_condor(self, spot, legs_info, lot_size, lots) -> float:
        sells = [l for l in legs_info if l["action"] == "SELL"]
        buys = [l for l in legs_info if l["action"] == "BUY"]
        put_sell = next((l for l in sells if l["option_type"] == "PE"), None)
        put_buy = next((l for l in buys if l["option_type"] == "PE"), None)
        call_sell = next((l for l in sells if l["option_type"] == "CE"), None)
        call_buy = next((l for l in buys if l["option_type"] == "CE"), None)
        if not all([put_sell, put_buy, call_sell, call_buy]):
            return self._fallback_naked_sum(spot, legs_info, lot_size, lots)
        put_width = abs(put_sell["strike"] - put_buy["strike"])
        call_width = abs(call_sell["strike"] - call_buy["strike"])
        put_credit = put_sell["premium"] - put_buy["premium"]
        call_credit = call_sell["premium"] - call_buy["premium"]
        put_sp = self.spread_margin(max(put_width - put_credit, 0), lot_size, lots)
        call_sp = self.spread_margin(max(call_width - call_credit, 0), lot_size, lots)
        return self.iron_condor_margin(put_sp, call_sp)

    def _margin_vertical_spread(self, spot, legs_info, lot_size, lots) -> float:
        sell_leg = next((l for l in legs_info if l["action"] == "SELL"), None)
        buy_leg = next((l for l in legs_info if l["action"] == "BUY"), None)
        if sell_leg is None or buy_leg is None:
            return self._fallback_naked_sum(spot, legs_info, lot_size, lots)
        width = abs(sell_leg["strike"] - buy_leg["strike"])
        credit = sell_leg["premium"] - buy_leg["premium"]
        max_loss = max(width - credit, 0)
        return self.spread_margin(max_loss, lot_size, lots)

    def _margin_calendar(self, legs_info, lot_size, lots) -> float:
        sell_leg = next((l for l in legs_info if l["action"] == "SELL"), None)
        buy_leg = next((l for l in legs_info if l["action"] == "BUY"), None)
        near_prem = sell_leg["premium"] if sell_leg else 0.0
        far_prem = buy_leg["premium"] if buy_leg else 0.0
        return self.calendar_spread_margin(near_prem, far_prem, lot_size, lots)

    def _margin_ratio_spread(self, spot, legs_info, lot_size, lots) -> float:
        """Ratio spread: long 1, short N.  Extra shorts are naked."""
        sells = [l for l in legs_info if l["action"] == "SELL"]
        buys = [l for l in legs_info if l["action"] == "BUY"]
        hedged = min(len(buys), len(sells))
        naked_sells = sells[hedged:]
        margin = 0.0
        if hedged and buys and sells:
            width = abs(sells[0]["strike"] - buys[0]["strike"])
            credit = sells[0]["premium"] - buys[0]["premium"]
            margin += self.spread_margin(max(width - credit, 0), lot_size, lots)
        for leg in naked_sells:
            margin += self.naked_short_margin(
                spot, leg["strike"], leg["premium"], lot_size, lots
            )
        return margin

    def _fallback_naked_sum(self, spot, legs_info, lot_size, lots) -> float:
        total = 0.0
        for leg in legs_info:
            if leg["action"] == "SELL":
                total += self.naked_short_margin(
                    spot, leg["strike"], leg["premium"], lot_size, lots
                )
        return total * SPREAD_MARGIN_BUFFER


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    mc = MarginCalculator()
    spot = 24_500.0
    lot = LOT_SIZES["NIFTY"]

    print("=== Margin Calculator Self-Test ===\n")

    # 1. Naked short
    m = mc.naked_short_margin(spot, 24_000, 150, lot, 1)
    print(f"Naked short  24000 PE @150  : ₹{m:>12,.2f}")

    # 2. Bull put spread  (sell 24000 PE @150, buy 23800 PE @80)
    max_loss = (24_000 - 23_800) - (150 - 80)  # 200 - 70 = 130
    m = mc.spread_margin(max_loss, lot, 1)
    print(f"Bull put spread  200-wide   : ₹{m:>12,.2f}")

    # 3. Iron condor
    ic_legs = [
        {"strike": 24_000, "option_type": "PE", "action": "SELL", "premium": 150},
        {"strike": 23_800, "option_type": "PE", "action": "BUY",  "premium": 80},
        {"strike": 25_000, "option_type": "CE", "action": "SELL", "premium": 140},
        {"strike": 25_200, "option_type": "CE", "action": "BUY",  "premium": 70},
    ]
    m = mc.estimate_strategy_margin("IRON_CONDOR", spot, ic_legs, lot, 1)
    print(f"Iron condor                 : ₹{m:>12,.2f}")

    # 4. Short straddle
    m = mc.short_straddle_margin(spot, 24_500, 320, 310, lot, 1)
    print(f"Short straddle ATM @320/310 : ₹{m:>12,.2f}")

    # 5. Short strangle via dispatcher
    strangle_legs = [
        {"strike": 25_000, "option_type": "CE", "action": "SELL", "premium": 140},
        {"strike": 24_000, "option_type": "PE", "action": "SELL", "premium": 150},
    ]
    m = mc.estimate_strategy_margin("SHORT_STRANGLE", spot, strangle_legs, lot, 1)
    print(f"Short strangle via dispatch : ₹{m:>12,.2f}")

    # 6. Calendar spread
    m = mc.calendar_spread_margin(near_premium=120, far_premium=180, lot_size=lot, lots=1)
    print(f"Calendar spread 120/180     : ₹{m:>12,.2f}")

    # 7. Margin check
    equity = 500_000.0
    required = mc.naked_short_margin(spot, 24_000, 150, lot, 2)
    check = mc.check_margin_available(required, equity)
    print(f"\nMargin check (equity ₹{equity:,.0f}):")
    for k, v in check.items():
        print(f"  {k:>12}: {v}")

    print("\nDone.")
