"""
Daily Loss Limiter — hard 5% daily loss cap with cascading circuit breakers.

Three tiers:
  Tier 1 (3% loss): reduce position size to 50%, alert Telegram
  Tier 2 (5% loss): HALT all new entries for the day, close open positions
  Tier 3 (12% weekly): HALT trading for remainder of week

Integrates with:
- OptionsExecutor: blocks signal_to_orders() when limit hit
- CompoundSizer: forces lot count to 0
- ExecutionEngine: skips ORDER_QUEUE processing
- PnlTracker: reads realized + unrealized P&L

Usage:
    from risk.daily_loss_limiter import DailyLossLimiter
    limiter = DailyLossLimiter(equity=200_000)

    # Before every trade entry
    if limiter.can_trade():
        order = executor.signal_to_orders(signal, ...)

    # After every trade exit
    limiter.record_trade(pnl=-3500)

    # Periodic check (unrealized P&L from live positions)
    limiter.update_unrealized(-2000)
"""

import logging
from datetime import date, datetime, time as dt_time
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# Tier thresholds (as fraction of start-of-day equity)
TIER_1_LOSS_PCT = 0.03   # 3% → reduce size
TIER_2_LOSS_PCT = 0.05   # 5% → halt entries
WEEKLY_LOSS_PCT = 0.12   # 12% → halt for week

# Size reduction in Tier 1
TIER_1_SIZE_FACTOR = 0.50


class DailyLossLimiter:
    """
    Hard daily/weekly loss limits with tiered circuit breakers.
    """

    def __init__(self, equity: float):
        self.equity = equity
        self._start_of_day_equity = equity
        self._start_of_week_equity = equity

        # Daily tracking
        self._daily_realized_pnl = 0.0
        self._daily_unrealized_pnl = 0.0
        self._daily_trades = []
        self._last_date = None

        # Weekly tracking
        self._weekly_realized_pnl = 0.0
        self._last_week = None

        # State
        self.tier = 0           # 0=normal, 1=reduced, 2=halted
        self.weekly_halt = False
        self._alerts_sent = set()  # track which alerts sent today

    @property
    def daily_pnl(self) -> float:
        """Total daily P&L (realized + unrealized)."""
        return self._daily_realized_pnl + self._daily_unrealized_pnl

    @property
    def daily_loss_pct(self) -> float:
        """Daily loss as fraction of start-of-day equity."""
        if self._start_of_day_equity <= 0:
            return 0
        return -self.daily_pnl / self._start_of_day_equity

    @property
    def weekly_loss_pct(self) -> float:
        """Weekly loss as fraction of start-of-week equity."""
        if self._start_of_week_equity <= 0:
            return 0
        return -self._weekly_realized_pnl / self._start_of_week_equity

    @property
    def size_factor(self) -> float:
        """Position size multiplier based on current tier."""
        if self.tier >= 2 or self.weekly_halt:
            return 0.0
        if self.tier == 1:
            return TIER_1_SIZE_FACTOR
        return 1.0

    def can_trade(self, today: Optional[date] = None) -> bool:
        """
        Check if new entries are allowed.

        Returns:
            True if trading is allowed, False if halted.
        """
        today = today or date.today()
        self._maybe_reset(today)

        if self.weekly_halt:
            return False
        if self.tier >= 2:
            return False
        return True

    def record_trade(self, pnl: float, today: Optional[date] = None):
        """
        Record a completed trade and check limits.

        Args:
            pnl: realized P&L for this trade (positive or negative)
        """
        today = today or date.today()
        self._maybe_reset(today)

        self._daily_realized_pnl += pnl
        self._weekly_realized_pnl += pnl
        self.equity += pnl

        self._daily_trades.append({
            'time': datetime.now().strftime('%H:%M:%S'),
            'pnl': round(pnl, 2),
            'cum_pnl': round(self._daily_realized_pnl, 2),
        })

        # Check tiers
        self._check_tiers()

    def update_unrealized(self, unrealized_pnl: float):
        """
        Update unrealized P&L from open positions.
        Call this periodically (e.g., every 5 min bar).
        """
        self._daily_unrealized_pnl = unrealized_pnl
        self._check_tiers()

    def _check_tiers(self):
        """Evaluate circuit breaker tiers based on current P&L."""
        loss_pct = self.daily_loss_pct

        # Tier 2: 5% daily loss → HALT
        if loss_pct >= TIER_2_LOSS_PCT and self.tier < 2:
            self.tier = 2
            self._alert('TIER_2',
                        f"DAILY LOSS LIMIT HIT: {loss_pct:.1%} "
                        f"(₹{self.daily_pnl:,.0f}). ALL ENTRIES HALTED.")

        # Tier 1: 3% daily loss → reduce size
        elif loss_pct >= TIER_1_LOSS_PCT and self.tier < 1:
            self.tier = 1
            self._alert('TIER_1',
                        f"Daily loss warning: {loss_pct:.1%} "
                        f"(₹{self.daily_pnl:,.0f}). Position size reduced to 50%.")

        # Weekly halt: 12% weekly loss
        if self.weekly_loss_pct >= WEEKLY_LOSS_PCT and not self.weekly_halt:
            self.weekly_halt = True
            self._alert('WEEKLY_HALT',
                        f"WEEKLY LOSS LIMIT HIT: {self.weekly_loss_pct:.1%} "
                        f"(₹{self._weekly_realized_pnl:,.0f}). TRADING HALTED FOR WEEK.")

    def _alert(self, alert_type: str, message: str):
        """Send alert (logs + can be extended to Telegram)."""
        if alert_type in self._alerts_sent:
            return
        self._alerts_sent.add(alert_type)

        if 'HALT' in alert_type or 'TIER_2' in alert_type:
            logger.critical(message)
        else:
            logger.warning(message)

    def _maybe_reset(self, today: date):
        """Reset daily/weekly counters at session boundaries."""
        # Daily reset
        if self._last_date != today:
            if self._last_date is not None:
                logger.info(
                    f"Session close {self._last_date}: "
                    f"P&L ₹{self._daily_realized_pnl:,.0f} "
                    f"({len(self._daily_trades)} trades)"
                )
            self._start_of_day_equity = self.equity
            self._daily_realized_pnl = 0.0
            self._daily_unrealized_pnl = 0.0
            self._daily_trades = []
            self.tier = 0
            self._alerts_sent = set()
            self._last_date = today

        # Weekly reset (Monday)
        week_num = today.isocalendar()[1]
        if self._last_week != week_num:
            if self._last_week is not None:
                logger.info(
                    f"Week {self._last_week} close: "
                    f"weekly P&L ₹{self._weekly_realized_pnl:,.0f}"
                )
            self._start_of_week_equity = self.equity
            self._weekly_realized_pnl = 0.0
            self.weekly_halt = False
            self._last_week = week_num

    def get_status(self) -> Dict:
        """Return current risk state."""
        return {
            'equity': self.equity,
            'start_of_day_equity': self._start_of_day_equity,
            'daily_realized_pnl': round(self._daily_realized_pnl, 2),
            'daily_unrealized_pnl': round(self._daily_unrealized_pnl, 2),
            'daily_total_pnl': round(self.daily_pnl, 2),
            'daily_loss_pct': round(self.daily_loss_pct, 4),
            'weekly_pnl': round(self._weekly_realized_pnl, 2),
            'weekly_loss_pct': round(self.weekly_loss_pct, 4),
            'tier': self.tier,
            'size_factor': self.size_factor,
            'weekly_halt': self.weekly_halt,
            'can_trade': self.tier < 2 and not self.weekly_halt,
            'trades_today': len(self._daily_trades),
        }

    def force_close_positions(self) -> List[str]:
        """
        Called when Tier 2 is hit — returns list of position IDs to close.
        Actual closing is done by ExecutionEngine.
        """
        logger.critical("FORCE CLOSE: Tier 2 triggered — all positions must be closed")
        return ['ALL']  # Signal to execution engine to close everything


# ================================================================
# MAIN — self-test
# ================================================================

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s %(message)s')

    print("=" * 65)
    print("  DAILY LOSS LIMITER — Self-Test")
    print("=" * 65)

    limiter = DailyLossLimiter(equity=200_000)
    today = date(2026, 3, 23)

    print(f"\nEquity: ₹{limiter.equity:,.0f}")
    print(f"Tier 1 limit: ₹{limiter._start_of_day_equity * TIER_1_LOSS_PCT:,.0f} (3%)")
    print(f"Tier 2 limit: ₹{limiter._start_of_day_equity * TIER_2_LOSS_PCT:,.0f} (5%)")

    # Simulate trades
    trades = [
        (-2000, "Small loss"),
        (-1500, "Another loss"),
        (1000,  "Small win"),
        (-3000, "Bad trade — should trigger Tier 1"),
        (-2500, "More losses — should trigger Tier 2"),
    ]

    for pnl, desc in trades:
        print(f"\n  Trade: {desc} (₹{pnl:+,.0f})")
        limiter.record_trade(pnl, today=today)
        status = limiter.get_status()
        print(f"    Day P&L: ₹{status['daily_realized_pnl']:,.0f} "
              f"({status['daily_loss_pct']:.1%} loss)")
        print(f"    Tier: {status['tier']} | Size factor: {status['size_factor']} "
              f"| Can trade: {status['can_trade']}")

        if not limiter.can_trade(today):
            print(f"    >>> TRADING HALTED <<<")
            break

    print(f"\n{'─' * 65}")
    print(f"  Final status:")
    for k, v in limiter.get_status().items():
        print(f"    {k}: {v}")
