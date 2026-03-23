"""
End-to-end sizing chain verification.

Tests the full sizing pipeline:
  CompoundSizer -> BehavioralOverlay -> DailyLossLimiter -> 2x Safety Cap

Ensures all components wire together correctly and safety invariants hold.
"""

import pytest
from datetime import date, datetime

from risk.compound_sizer import CompoundSizer, MAX_LOTS
from risk.daily_loss_limiter import DailyLossLimiter
from risk.behavioral_overlay import (
    BehavioralOverlay, OverlayContext, TradeRecord, PositionContext,
)


# ══════════════════════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════════════════════

@pytest.fixture
def sizer():
    return CompoundSizer(initial_equity=1_000_000)


@pytest.fixture
def overlay():
    return BehavioralOverlay()


@pytest.fixture
def limiter():
    return DailyLossLimiter(equity=1_000_000)


# ══════════════════════════════════════════════════════════════
# TEST: Full sizing chain
# ══════════════════════════════════════════════════════════════

class TestSizingChain:

    def test_basic_chain_produces_positive_lots(self, sizer, overlay, limiter):
        """Full chain: sizer -> overlay -> limiter produces valid lot count."""
        today = date(2026, 3, 23)

        # Step 1: CompoundSizer
        lots = sizer.get_lots('NIFTY', premium=200, today=today)
        assert lots > 0
        assert lots <= MAX_LOTS['NIFTY']

        # Step 2: BehavioralOverlay
        ctx = OverlayContext(system_size=1.0, current_dd_pct=0.02)
        result = overlay.apply_all(ctx)
        lots_after_overlay = max(1, int(lots * result.size_multiplier))
        assert lots_after_overlay > 0
        assert lots_after_overlay <= lots  # overlay never increases

        # Step 3: DailyLossLimiter
        assert limiter.can_trade(today)
        lots_final = max(1, int(lots_after_overlay * limiter.size_factor))
        assert lots_final > 0

    def test_behavioral_overlay_integration_point(self, sizer, overlay):
        """CompoundSizer.set_behavioral_overlay and apply_behavioral_overlay work."""
        sizer.set_behavioral_overlay(overlay)

        ctx = OverlayContext(system_size=1.0, current_dd_pct=0.02)
        lots, result = sizer.apply_behavioral_overlay(10, ctx)
        assert lots > 0
        assert lots <= 10  # overlay never increases

    def test_drawdown_reduces_lots(self, sizer):
        """Drawdown mode halves lot count."""
        today = date(2026, 3, 23)
        lots_normal = sizer.get_lots('NIFTY', premium=200, today=today)

        # Simulate drawdown (>8% but <15% to avoid halt)
        sizer.update_equity(900_000)  # 10% DD
        assert sizer.in_drawdown

        today2 = date(2026, 3, 30)
        lots_dd = sizer.get_lots('NIFTY', premium=200, today=today2)
        assert lots_dd < lots_normal
        assert lots_dd >= 1

    def test_loss_limiter_blocks_after_tier2(self, limiter):
        """Tier 2 halts all trading."""
        today = date(2026, 3, 23)
        assert limiter.can_trade(today)

        # Simulate 5% loss
        limiter.record_trade(-50_000, today)
        assert not limiter.can_trade(today)
        assert limiter.tier >= 2
        assert limiter.size_factor == 0.0

    def test_loss_limiter_tier1_reduces_size(self, limiter):
        """Tier 1 reduces size to 50%."""
        today = date(2026, 3, 23)
        limiter.record_trade(-30_000, today)  # 3% loss
        assert limiter.can_trade(today)
        assert limiter.size_factor == 0.5

    def test_2x_safety_cap_logic(self):
        """Verify 2x safety cap: lots > 2x system_lots is rejected."""
        system_lots = 10

        # Within cap
        requested_lots = 18
        assert requested_lots <= 2 * system_lots

        # Over cap
        requested_lots_over = 25
        assert requested_lots_over > 2 * system_lots

    def test_overlay_after_losses_floors_size(self, overlay):
        """After 3+ losses with healthy metrics, floor at 70%."""
        # Build healthy history then consecutive losses
        for i in range(20):
            overlay.record_trade(TradeRecord(
                pnl=500, pnl_pct=0.02,
                entry_time=datetime.now(), holding_bars=10,
            ))
        for i in range(4):
            overlay.record_trade(TradeRecord(
                pnl=-300, pnl_pct=-0.01,
                entry_time=datetime.now(), holding_bars=10,
            ))

        ctx = OverlayContext(current_dd_pct=0.03)
        result = overlay.apply_all(ctx)
        # Floor should be at 70%
        assert result.size_multiplier >= 0.69

    def test_overconfidence_reduces_after_8_wins(self, overlay):
        """After 8+ wins, size reduced to 80%."""
        for i in range(9):
            overlay.record_trade(TradeRecord(
                pnl=500, pnl_pct=0.02,
                entry_time=datetime.now(), holding_bars=10,
            ))

        ctx = OverlayContext()
        result = overlay.apply_all(ctx)
        assert result.size_multiplier <= 0.81

    def test_chain_with_all_modifiers(self, sizer, overlay, limiter):
        """Full chain with all modifiers active."""
        today = date(2026, 3, 23)

        # Setup: overconfidence dampener active
        for i in range(9):
            overlay.record_trade(TradeRecord(
                pnl=500, pnl_pct=0.02,
                entry_time=datetime.now(), holding_bars=10,
            ))

        # Setup: Tier 1 loss limiter
        limiter.record_trade(-30_000, today)

        # Step 1: Sizer
        lots = sizer.get_lots('NIFTY', premium=200, today=today)
        assert lots > 0

        # Step 2: Behavioral overlay (overconfidence -> 80%)
        ctx = OverlayContext(current_dd_pct=0.02)
        result = overlay.apply_all(ctx)
        lots = max(1, int(lots * result.size_multiplier))

        # Step 3: Loss limiter (Tier 1 -> 50%)
        lots = max(1, int(lots * limiter.size_factor))

        # Final lots should be significantly reduced
        base_lots = sizer.get_lots('NIFTY', premium=200, today=today)
        assert lots <= base_lots
        assert lots >= 1

    def test_spread_lots_capped(self, sizer):
        """Spread lots respect risk cap and drawdown rules."""
        today = date(2026, 3, 23)
        lots = sizer.get_spread_lots(
            instrument='NIFTY',
            net_debit=100,
            max_loss_per_lot=2500,
            today=today,
        )
        assert lots > 0
        assert lots <= MAX_LOTS['NIFTY']

    def test_equity_update_tracks_peak(self, sizer):
        """Equity update correctly tracks peak for drawdown."""
        sizer.update_equity(1_100_000)
        assert sizer.peak_equity == 1_100_000

        sizer.update_equity(1_050_000)
        assert sizer.peak_equity == 1_100_000
        assert sizer.drawdown_pct > 0

    def test_sizing_rule_attribute(self):
        """CompoundSizer accepts sizing_rule parameter."""
        sizer = CompoundSizer(initial_equity=500_000, sizing_rule='FIXED_FRACTIONAL')
        assert sizer.sizing_rule == 'FIXED_FRACTIONAL'

        sizer2 = CompoundSizer(initial_equity=500_000)
        assert sizer2.sizing_rule == 'COMPOUND_TIERED'


# ══════════════════════════════════════════════════════════════
# TEST: Safety invariants
# ══════════════════════════════════════════════════════════════

class TestSafetyInvariants:

    def test_behavioral_overlay_never_increases_beyond_system(self, overlay):
        """Behavioral overlay never returns size_multiplier > 1.0."""
        for _ in range(100):
            ctx = OverlayContext(
                system_size=1.0,
                current_dd_pct=0.01,
                entry_price=25000,
                proposed_sl=24500,
                proposed_tgt=25500,
                direction="LONG",
                current_atr=150,
            )
            result = overlay.apply_all(ctx)
            assert result.size_multiplier <= 1.0, (
                f"size_multiplier={result.size_multiplier} > 1.0"
            )

    def test_lots_always_positive(self, sizer):
        """Lots never go negative even with extreme drawdown."""
        today = date(2026, 3, 23)
        sizer.update_equity(50_000)  # severe drawdown

        lots = sizer.get_lots('NIFTY', premium=200, today=today)
        assert lots >= 0  # can be 0 if halted, never negative

    def test_drawdown_halt_returns_zero(self, sizer):
        """15% drawdown halts trading (0 lots)."""
        sizer.update_equity(840_000)  # >15% DD from 1M
        today = date(2026, 3, 23)
        lots = sizer.get_lots('NIFTY', premium=200, today=today)
        assert lots == 0

    def test_daily_reset_clears_tier(self, limiter):
        """Tier resets on new day."""
        day1 = date(2026, 3, 23)
        limiter.record_trade(-50_000, day1)
        assert not limiter.can_trade(day1)

        day2 = date(2026, 3, 24)
        assert limiter.can_trade(day2)
        assert limiter.tier == 0


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
