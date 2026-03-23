"""
India Risk Guard — India-specific pre-order risk checks.

Enforces two categories of risk controls:
  1. NSE circuit breaker proximity — blocks entries when price is within 1%
     of any circuit level (5%, 10%, 15%, 20% from previous close)
  2. Consecutive loss tracking — halts trading after streaks of losses
     (5 consecutive → 24h halt, 8 consecutive → 1 week halt + decay scan)

Runs before every order placement.

Usage:
    from risk.india_risk_guard import IndiaRiskGuard
    guard = IndiaRiskGuard()

    # Before placing any order
    check = guard.can_trade(current_price=22350, prev_close=22000)
    if not check['allowed']:
        print(f"Blocked: {check['reasons']}")

    # After each trade closes
    guard.record_trade_result(pnl=-1500)
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# NSE circuit breaker levels (percentage from previous close)
CIRCUIT_LEVELS_PCT = [5.0, 10.0, 15.0, 20.0]

# Block new entries if within this percentage of any circuit level
CIRCUIT_PROXIMITY_PCT = 1.0

# Consecutive loss thresholds
CONSECUTIVE_LOSS_HALT_24H = 5
CONSECUTIVE_LOSS_HALT_1WEEK = 8


class IndiaRiskGuard:
    """
    India-specific risk checks: circuit breaker proximity and consecutive loss tracking.
    Must pass before every order placement.
    """

    def __init__(self):
        # Circuit breaker levels (fractions)
        self.circuit_levels = [lvl / 100.0 for lvl in CIRCUIT_LEVELS_PCT]
        self.proximity_threshold = CIRCUIT_PROXIMITY_PCT / 100.0

        # Consecutive loss tracking
        self._consecutive_losses = 0
        self._trade_results: List[float] = []

        # Halt state
        self._halted = False
        self._halt_until: Optional[datetime] = None
        self._halt_reason: Optional[str] = None
        self._trigger_decay_scan = False

    # ==================================================================
    # PUBLIC: Circuit breaker proximity check
    # ==================================================================

    def check_circuit_proximity(
        self, current_price: float, prev_close: float
    ) -> Dict:
        """
        Check how close the current price is to any NSE circuit breaker level.

        Args:
            current_price: current market price of the underlying
            prev_close:    previous day's closing price

        Returns:
            {allowed: bool, nearest_circuit: str, distance_pct: float, reason: str}
        """
        if prev_close <= 0:
            return {
                "allowed": True,
                "nearest_circuit": "N/A",
                "distance_pct": 100.0,
                "reason": "Invalid prev_close, skipping circuit check",
            }

        change_pct = abs(current_price - prev_close) / prev_close

        nearest_circuit = None
        nearest_distance = float("inf")

        for level in self.circuit_levels:
            # Upper circuit: prev_close * (1 + level)
            upper_circuit_price = prev_close * (1.0 + level)
            upper_distance = abs(current_price - upper_circuit_price) / prev_close

            # Lower circuit: prev_close * (1 - level)
            lower_circuit_price = prev_close * (1.0 - level)
            lower_distance = abs(current_price - lower_circuit_price) / prev_close

            # Track nearest
            if upper_distance < nearest_distance:
                nearest_distance = upper_distance
                nearest_circuit = f"+{level * 100:.0f}% (₹{upper_circuit_price:,.2f})"

            if lower_distance < nearest_distance:
                nearest_distance = lower_distance
                nearest_circuit = f"-{level * 100:.0f}% (₹{lower_circuit_price:,.2f})"

        # Block if within proximity threshold of any circuit level
        blocked = nearest_distance <= self.proximity_threshold
        reason = ""

        if blocked:
            reason = (
                f"Price ₹{current_price:,.2f} is within {nearest_distance * 100:.2f}% "
                f"of circuit level {nearest_circuit}. New entries BLOCKED."
            )
            logger.warning(f"Circuit proximity block: {reason}")

        return {
            "allowed": not blocked,
            "nearest_circuit": nearest_circuit or "N/A",
            "distance_pct": round(nearest_distance * 100, 2),
            "reason": reason,
        }

    # ==================================================================
    # PUBLIC: Consecutive loss tracking
    # ==================================================================

    def record_trade_result(self, pnl: float) -> Dict:
        """
        Record a trade result and update consecutive loss counter.

        Args:
            pnl: realized P&L for the trade (positive = win, negative = loss)

        Returns:
            {consecutive_losses: int, halt_24h: bool, halt_1week: bool,
             trigger_decay_scan: bool}
        """
        self._trade_results.append(pnl)

        if pnl < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0
            # Winning trade clears any pending decay scan
            self._trigger_decay_scan = False

        result = {
            "consecutive_losses": self._consecutive_losses,
            "halt_24h": False,
            "halt_1week": False,
            "trigger_decay_scan": False,
        }

        # Check thresholds (check higher threshold first)
        if self._consecutive_losses >= CONSECUTIVE_LOSS_HALT_1WEEK:
            halt_until = datetime.now() + timedelta(weeks=1)
            self._halted = True
            self._halt_until = halt_until
            self._halt_reason = (
                f"{self._consecutive_losses} consecutive losses: "
                f"trading halted until {halt_until.strftime('%Y-%m-%d %H:%M')}"
            )
            self._trigger_decay_scan = True
            result["halt_1week"] = True
            result["trigger_decay_scan"] = True
            logger.critical(
                f"HALT 1 WEEK: {self._consecutive_losses} consecutive losses. "
                f"Halted until {halt_until.isoformat()}"
            )

        elif self._consecutive_losses >= CONSECUTIVE_LOSS_HALT_24H:
            halt_until = datetime.now() + timedelta(hours=24)
            self._halted = True
            self._halt_until = halt_until
            self._halt_reason = (
                f"{self._consecutive_losses} consecutive losses: "
                f"trading halted until {halt_until.strftime('%Y-%m-%d %H:%M')}"
            )
            result["halt_24h"] = True
            logger.warning(
                f"HALT 24H: {self._consecutive_losses} consecutive losses. "
                f"Halted until {halt_until.isoformat()}"
            )

        return result

    def check_consecutive_losses(self) -> Dict:
        """
        Check current consecutive loss state.

        Returns:
            {consecutive_losses: int, halted: bool, halt_until: datetime|None, reason: str}
        """
        # Check if halt has expired
        if self._halted and self._halt_until:
            if datetime.now() >= self._halt_until:
                logger.info(
                    f"Halt expired at {self._halt_until.isoformat()}. "
                    f"Trading resumed."
                )
                self._halted = False
                self._halt_until = None
                self._halt_reason = None

        return {
            "consecutive_losses": self._consecutive_losses,
            "halted": self._halted,
            "halt_until": self._halt_until,
            "reason": self._halt_reason or "",
        }

    # ==================================================================
    # PUBLIC: Combined pre-order check
    # ==================================================================

    def can_trade(
        self, current_price: float = None, prev_close: float = None
    ) -> Dict:
        """
        Combined pre-order check: circuit proximity + consecutive losses + active halt.

        Args:
            current_price: current market price (optional, skips circuit check if None)
            prev_close:    previous day's close (optional, skips circuit check if None)

        Returns:
            {allowed: bool, reasons: list}
        """
        reasons = []
        allowed = True

        # Check 1: Active halt from consecutive losses
        loss_status = self.check_consecutive_losses()
        if loss_status["halted"]:
            allowed = False
            reasons.append(loss_status["reason"])

        # Check 2: Circuit breaker proximity
        if current_price is not None and prev_close is not None:
            circuit_check = self.check_circuit_proximity(current_price, prev_close)
            if not circuit_check["allowed"]:
                allowed = False
                reasons.append(circuit_check["reason"])

        if allowed:
            logger.debug("IndiaRiskGuard: all checks passed")

        return {"allowed": allowed, "reasons": reasons}

    # ==================================================================
    # PUBLIC: Status
    # ==================================================================

    def get_status(self) -> Dict:
        """Return full risk guard state."""
        loss_status = self.check_consecutive_losses()
        return {
            "consecutive_losses": self._consecutive_losses,
            "total_trades_tracked": len(self._trade_results),
            "halted": loss_status["halted"],
            "halt_until": (
                self._halt_until.isoformat() if self._halt_until else None
            ),
            "halt_reason": self._halt_reason,
            "trigger_decay_scan": self._trigger_decay_scan,
            "circuit_levels_pct": CIRCUIT_LEVELS_PCT,
            "proximity_threshold_pct": CIRCUIT_PROXIMITY_PCT,
            "halt_threshold_24h": CONSECUTIVE_LOSS_HALT_24H,
            "halt_threshold_1week": CONSECUTIVE_LOSS_HALT_1WEEK,
        }


# ================================================================
# MAIN — self-test
# ================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s"
    )

    print("=" * 65)
    print("  INDIA RISK GUARD — Self-Test")
    print("=" * 65)

    guard = IndiaRiskGuard()

    # Test circuit proximity
    print("\n--- Circuit Proximity Checks ---")
    prev_close = 22000.0
    test_prices = [22000, 23050, 23100, 19850, 17650, 21500]
    for price in test_prices:
        result = guard.check_circuit_proximity(price, prev_close)
        status = "BLOCKED" if not result["allowed"] else "OK"
        print(
            f"  Price ₹{price:,.0f} (prev_close ₹{prev_close:,.0f}): "
            f"{status} | nearest={result['nearest_circuit']} "
            f"distance={result['distance_pct']:.2f}%"
        )

    # Test consecutive losses
    print("\n--- Consecutive Loss Tracking ---")
    trades = [-500, -300, -200, -100, -400, 200, -100, -200, -300, -400, -500, -600, -700, -800]
    for i, pnl in enumerate(trades):
        result = guard.record_trade_result(pnl)
        print(
            f"  Trade {i + 1}: ₹{pnl:+,.0f} | "
            f"streak={result['consecutive_losses']} "
            f"halt_24h={result['halt_24h']} "
            f"halt_1week={result['halt_1week']}"
        )
        if result["halt_1week"]:
            print("  >>> 1 WEEK HALT + DECAY SCAN TRIGGERED <<<")
            break
        elif result["halt_24h"]:
            print("  >>> 24H HALT <<<")

    # Test combined check
    print("\n--- Combined can_trade() ---")
    check = guard.can_trade(current_price=22000, prev_close=22000)
    print(f"  Halted state: allowed={check['allowed']} reasons={check['reasons']}")

    print(f"\n--- Status ---")
    for k, v in guard.get_status().items():
        print(f"  {k}: {v}")
