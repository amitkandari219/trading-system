"""
Compound Position Sizer — equity-proportional lot sizing with weekly ratchet.

Core idea: as equity grows via compounding, deploy more lots proportionally.
Lots = floor(equity × deploy_fraction / cost_per_lot)

Safety features:
- Weekly ratchet: lots only increase at week boundaries (no intra-week whipsaw)
- Drawdown reducer: if equity drops >8% from peak, halve position size
- Gradual recovery: after drawdown, increase lots by 1 per profitable week
- Hard caps: absolute max lots regardless of equity

Usage:
    from risk.compound_sizer import CompoundSizer
    sizer = CompoundSizer(initial_equity=200_000)
    lots = sizer.get_lots(instrument='NIFTY', premium=200)
    sizer.update_equity(new_equity)  # call after each trade
"""

import logging
from datetime import date, timedelta
from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from risk.behavioral_overlay import BehavioralOverlay, OverlayContext, OverlayResult

logger = logging.getLogger(__name__)


# Instrument lot sizes
LOT_SIZES = {
    'NIFTY': 25,
    'BANKNIFTY': 15,
}

# Absolute caps
MAX_LOTS = {
    'NIFTY': 60,        # 60 × 25 = 1500 units max
    'BANKNIFTY': 80,    # 80 × 15 = 1200 units max
}

# Deploy fraction by equity tier
# As equity grows, we can afford to be slightly more aggressive
EQUITY_TIERS = [
    (0,       200_000,  0.45),   # < 2L: conservative 45%
    (200_000, 500_000,  0.50),   # 2-5L: standard 50%
    (500_000, 1_000_000, 0.55),  # 5-10L: moderate 55%
    (1_000_000, float('inf'), 0.50),  # >10L: back to 50% (capital preservation)
]

# Drawdown thresholds
DD_REDUCE_PCT = 0.08    # reduce size at 8% drawdown from peak
DD_HALT_PCT = 0.15      # halt trading at 15% drawdown
DD_SIZE_FACTOR = 0.50   # halve position size during drawdown

# SPAN margin approximation (per lot, in rupees)
# NSE SPAN margin for Nifty futures ~ Rs 1.2L/lot, options ~ Rs 0.8-1.5L/lot
SPAN_MARGIN_PER_LOT = {
    'NIFTY': 120_000,
    'BANKNIFTY': 100_000,
}
# Max fraction of equity usable as margin (safety buffer)
MAX_MARGIN_FRACTION = 0.50


class CompoundSizer:
    """
    Equity-proportional position sizer with weekly compounding ratchet.
    """

    def __init__(self, initial_equity: float, deploy_fraction: Optional[float] = None,
                 sizing_rule: str = 'COMPOUND_TIERED'):
        self.equity = initial_equity
        self.initial_equity = initial_equity
        self._custom_deploy = deploy_fraction

        # Peak tracking
        self.peak_equity = initial_equity
        self.in_drawdown = False

        # Weekly ratchet
        self._current_lots = {}    # instrument -> lots (locked for the week)
        self._week_start_equity = initial_equity
        self._last_ratchet_date = None

        # Recovery tracking
        self._consecutive_profit_weeks = 0
        self._dd_recovery_mode = False

        # Behavioral overlay integration (Layer 7)
        self._behavioral_overlay: Optional['BehavioralOverlay'] = None

        # Core sizing rule (switchable via A/B test results)
        # Options: 'COMPOUND_TIERED', 'FIXED_FRACTIONAL', 'VOLATILITY_SCALED',
        #          'HALF_KELLY', 'SIGNAL_CONFIDENCE'
        self.sizing_rule = sizing_rule

        logger.info(
            f"CompoundSizer initialized: equity=₹{initial_equity:,.0f} "
            f"rule={sizing_rule}"
        )

    @property
    def deploy_fraction(self) -> float:
        """Get deploy fraction based on equity tier."""
        if self._custom_deploy is not None:
            return self._custom_deploy

        for low, high, frac in EQUITY_TIERS:
            if low <= self.equity < high:
                return frac
        return 0.50

    @property
    def drawdown_pct(self) -> float:
        """Current drawdown from peak."""
        if self.peak_equity <= 0:
            return 0
        return (self.peak_equity - self.equity) / self.peak_equity

    def get_lots(self, instrument: str = 'NIFTY', premium: float = 200,
                 today: Optional[date] = None) -> int:
        """
        Get number of lots to trade.

        Uses weekly ratchet: lots are recalculated only at week boundaries.
        Within a week, lot count is locked.

        Args:
            instrument: 'NIFTY' or 'BANKNIFTY'
            premium: current ATM option premium per unit
            today: date for ratchet check (defaults to today)

        Returns:
            Number of lots to trade.
        """
        today = today or date.today()

        # Check drawdown halt
        if self.drawdown_pct >= DD_HALT_PCT:
            logger.warning(f"TRADING HALTED: drawdown {self.drawdown_pct:.1%} >= {DD_HALT_PCT:.0%}")
            return 0

        # Weekly ratchet check
        self._maybe_ratchet(today, instrument, premium)

        # Return locked lots
        lots = self._current_lots.get(instrument, 0)

        # Apply drawdown reducer
        if self.in_drawdown:
            lots = max(1, int(lots * DD_SIZE_FACTOR))

        # SPAN margin cap: ensure total margin doesn't exceed 50% of equity
        span_per_lot = SPAN_MARGIN_PER_LOT.get(instrument, 120_000)
        max_margin = self.equity * MAX_MARGIN_FRACTION
        margin_lots = int(max_margin / span_per_lot) if span_per_lot > 0 else lots
        if lots > margin_lots:
            logger.info(
                f"SPAN margin cap: {lots} -> {margin_lots} lots "
                f"(margin {lots * span_per_lot:,.0f} > limit {max_margin:,.0f})"
            )
            lots = margin_lots

        return lots

    def get_spread_lots(
        self,
        instrument: str = 'NIFTY',
        net_debit: float = 100.0,
        max_loss_per_lot: float = 0.0,
        today: Optional[date] = None,
    ) -> int:
        """
        Get number of lots for a spread trade, capping risk at 2% of equity.

        The key constraint: max_loss from spread <= 2% of equity.
        If max_loss_per_lot is provided, lots = floor(0.02 * equity / max_loss_per_lot).
        Otherwise falls back to standard get_lots using net_debit as premium.

        Args:
            instrument:        'NIFTY' or 'BANKNIFTY'
            net_debit:         net premium paid per unit (buy - sell credit)
            max_loss_per_lot:  maximum loss per lot (net_debit × lot_size)
            today:             date for ratchet check

        Returns:
            Number of lots, subject to weekly ratchet and drawdown rules.
        """
        today = today or date.today()

        # Check drawdown halt
        if self.drawdown_pct >= DD_HALT_PCT:
            logger.warning(
                f"TRADING HALTED: drawdown {self.drawdown_pct:.1%} >= {DD_HALT_PCT:.0%}"
            )
            return 0

        # Risk cap: max loss from spread <= 2% of equity
        risk_cap_pct = 0.02
        max_risk = self.equity * risk_cap_pct

        if max_loss_per_lot > 0:
            risk_lots = int(max_risk / max_loss_per_lot)
        else:
            # Fallback: use net_debit × lot_size as max loss estimate
            lot_size = LOT_SIZES.get(instrument, 25)
            max_loss_per_lot = net_debit * lot_size
            if max_loss_per_lot <= 0:
                return 0
            risk_lots = int(max_risk / max_loss_per_lot)

        # Also get standard lots from regular sizer
        standard_lots = self.get_lots(
            instrument=instrument, premium=net_debit, today=today
        )

        # Take the more conservative (minimum)
        lots = min(risk_lots, standard_lots)
        max_lots = MAX_LOTS.get(instrument, 40)
        lots = min(lots, max_lots)
        lots = max(lots, 0)

        # Apply drawdown reducer
        if self.in_drawdown:
            lots = max(1, int(lots * DD_SIZE_FACTOR))

        logger.info(
            f"Spread lots: {instrument} risk_lots={risk_lots} "
            f"standard_lots={standard_lots} -> {lots} lots "
            f"(max_loss/lot={max_loss_per_lot:,.0f} risk_cap={max_risk:,.0f})"
        )
        return lots

    def update_equity(self, new_equity: float, today: Optional[date] = None):
        """
        Update equity after trade completion.
        Call this after every trade exit.
        """
        old = self.equity
        self.equity = new_equity

        # Update peak
        if new_equity > self.peak_equity:
            self.peak_equity = new_equity
            if self.in_drawdown:
                logger.info("Recovery complete — new equity high")
                self.in_drawdown = False
                self._dd_recovery_mode = False
                self._consecutive_profit_weeks = 0

        # Check drawdown entry
        if not self.in_drawdown and self.drawdown_pct >= DD_REDUCE_PCT:
            self.in_drawdown = True
            self._dd_recovery_mode = True
            logger.warning(
                f"DRAWDOWN MODE: equity ₹{new_equity:,.0f} "
                f"({self.drawdown_pct:.1%} from peak ₹{self.peak_equity:,.0f})"
            )

        change = new_equity - old
        emoji = "+" if change >= 0 else ""
        logger.info(f"Equity: ₹{old:,.0f} → ₹{new_equity:,.0f} ({emoji}₹{change:,.0f})")

    def _maybe_ratchet(self, today: date, instrument: str, premium: float):
        """Recalculate lots at week boundaries."""
        # Monday = 0. We ratchet on Monday or first trading day of week.
        is_new_week = (
            self._last_ratchet_date is None or
            today.isocalendar()[1] != self._last_ratchet_date.isocalendar()[1] or
            today.year != self._last_ratchet_date.year
        )

        if not is_new_week and instrument in self._current_lots:
            return  # Already locked for this week

        # Week transition: check if previous week was profitable
        if self._last_ratchet_date is not None:
            week_pnl = self.equity - self._week_start_equity
            if week_pnl > 0:
                self._consecutive_profit_weeks += 1
            else:
                self._consecutive_profit_weeks = 0

        # Calculate new lots
        lot_size = LOT_SIZES.get(instrument, 25)
        max_lots = MAX_LOTS.get(instrument, 40)
        deploy = self.deploy_fraction
        cost_per_lot = premium * lot_size

        if cost_per_lot <= 0:
            self._current_lots[instrument] = 0
            return

        raw_lots = int((self.equity * deploy) / cost_per_lot)

        # In recovery mode: gradual increase
        if self._dd_recovery_mode:
            prev_lots = self._current_lots.get(instrument, raw_lots)
            # Add 1 lot per consecutive profitable week
            recovery_lots = prev_lots + self._consecutive_profit_weeks
            raw_lots = min(raw_lots, recovery_lots)

        lots = min(raw_lots, max_lots)
        lots = max(lots, 0)

        self._current_lots[instrument] = lots
        self._week_start_equity = self.equity
        self._last_ratchet_date = today

        logger.info(
            f"Ratchet {today}: {instrument} = {lots} lots "
            f"(equity ₹{self.equity:,.0f} × {deploy:.0%} / ₹{cost_per_lot:,.0f})"
        )

    # ── Behavioral Overlay Integration (Layer 7) ──

    def set_behavioral_overlay(self, overlay: 'BehavioralOverlay'):
        """Attach a BehavioralOverlay instance for bias correction."""
        self._behavioral_overlay = overlay
        logger.info("BehavioralOverlay attached to CompoundSizer")

    def apply_behavioral_overlay(
        self, lots: int, overlay_ctx: 'OverlayContext'
    ) -> tuple:
        """
        Apply behavioral overlay adjustments to lot count.

        Args:
            lots: base lot count from sizer
            overlay_ctx: OverlayContext with trade history, position, etc.

        Returns:
            (adjusted_lots, overlay_result) — adjusted lots and full result
        """
        if self._behavioral_overlay is None:
            return lots, None

        result = self._behavioral_overlay.apply_all(overlay_ctx)
        adjusted_lots = max(1, int(lots * result.size_multiplier))

        if result.overlays_triggered:
            logger.info(
                f"Behavioral overlay: {lots} -> {adjusted_lots} lots "
                f"(mult={result.size_multiplier:.2f} "
                f"triggers={result.overlays_triggered})"
            )

        return adjusted_lots, result

    def get_status(self) -> Dict:
        """Return current sizer state."""
        return {
            'equity': self.equity,
            'peak_equity': self.peak_equity,
            'drawdown_pct': round(self.drawdown_pct, 4),
            'in_drawdown': self.in_drawdown,
            'deploy_fraction': self.deploy_fraction,
            'deployable': self.equity * self.deploy_fraction,
            'current_lots': dict(self._current_lots),
            'consecutive_profit_weeks': self._consecutive_profit_weeks,
            'recovery_mode': self._dd_recovery_mode,
        }

    def project_growth(self, weeks: int = 26, avg_daily_pnl_pct: float = 0.02,
                       premium: float = 200) -> list:
        """
        Project equity and lot growth over N weeks.

        Args:
            weeks: projection horizon
            avg_daily_pnl_pct: average daily return as fraction of equity
            premium: assumed ATM option premium

        Returns:
            List of weekly snapshots.
        """
        snapshots = []
        eq = self.equity
        days_per_week = 5

        for w in range(1, weeks + 1):
            # Weekly P&L
            weekly_pnl = eq * avg_daily_pnl_pct * days_per_week
            eq += weekly_pnl

            lots_nifty = min(int(eq * 0.50 / (premium * 25)), MAX_LOTS['NIFTY'])
            lots_bn = min(int(eq * 0.50 / (premium * 15)), MAX_LOTS['BANKNIFTY'])
            daily_target = eq * avg_daily_pnl_pct

            snapshots.append({
                'week': w,
                'equity': round(eq),
                'nifty_lots': lots_nifty,
                'bn_lots': lots_bn,
                'daily_target': round(daily_target),
                'weekly_pnl': round(weekly_pnl),
            })

        return snapshots


# ================================================================
# MAIN — self-test and growth projection
# ================================================================

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    print("=" * 70)
    print("  COMPOUND SIZER — Self-Test & Projection")
    print("=" * 70)

    sizer = CompoundSizer(initial_equity=200_000)
    d = date(2026, 3, 23)  # Monday

    print(f"\n--- Week 1: Starting equity ₹{sizer.equity:,.0f} ---")
    lots = sizer.get_lots('NIFTY', premium=200, today=d)
    print(f"  Nifty lots: {lots}")
    lots_bn = sizer.get_lots('BANKNIFTY', premium=300, today=d)
    print(f"  BankNifty lots: {lots_bn}")

    # Simulate profitable week
    sizer.update_equity(220_000)
    d2 = date(2026, 3, 30)
    print(f"\n--- Week 2: Equity ₹{sizer.equity:,.0f} ---")
    lots2 = sizer.get_lots('NIFTY', premium=200, today=d2)
    print(f"  Nifty lots: {lots2}")

    # Simulate drawdown
    sizer.update_equity(195_000)
    print(f"\n--- Drawdown: Equity ₹{sizer.equity:,.0f} (DD={sizer.drawdown_pct:.1%}) ---")
    d3 = date(2026, 4, 6)
    lots3 = sizer.get_lots('NIFTY', premium=200, today=d3)
    print(f"  Nifty lots (drawdown mode): {lots3}")

    # Growth projection
    print(f"\n{'─' * 70}")
    print(f"  GROWTH PROJECTION (₹2L start, 2% daily return)")
    print(f"{'─' * 70}")
    print(f"  {'Week':>4s} {'Equity':>12s} {'Nifty':>6s} {'BN':>6s} {'Daily':>10s}")

    proj = CompoundSizer(200_000).project_growth(weeks=20, avg_daily_pnl_pct=0.02)
    for p in proj:
        marker = " ← ₹10K/day" if p['daily_target'] >= 10000 and (
            len([x for x in proj if x['daily_target'] >= 10000]) == 0 or
            p == [x for x in proj if x['daily_target'] >= 10000][0]
        ) else ""
        print(f"  {p['week']:4d} ₹{p['equity']:>10,} {p['nifty_lots']:>6d} {p['bn_lots']:>6d} ₹{p['daily_target']:>8,}{marker}")
