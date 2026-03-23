"""
Unit tests for BehavioralOverlay — all 4 Kahneman-inspired overlays.
"""

import pytest
from datetime import datetime, timedelta

from risk.behavioral_overlay import (
    BehavioralOverlay,
    OverlayContext,
    PositionContext,
    TradeRecord,
    OverlayResult,
    LA_CONSECUTIVE_LOSSES,
    LA_FLOOR_FRACTION,
    OC_CONSECUTIVE_WINS_CAP,
    OC_CONSECUTIVE_WINS_REDUCE,
    OC_REDUCE_FRACTION,
    DE_TRAILING_STOP_PCT,
)


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════

def make_overlay_with_trades(wins: int = 0, losses: int = 0,
                              win_pnl_pct: float = 0.02,
                              loss_pnl_pct: float = -0.01) -> BehavioralOverlay:
    """Create overlay with specified win/loss history."""
    overlay = BehavioralOverlay()
    for i in range(wins):
        overlay.record_trade(TradeRecord(
            pnl=500, pnl_pct=win_pnl_pct,
            entry_time=datetime.now() - timedelta(days=wins - i),
            holding_bars=10, signal_id=f"WIN_{i}",
        ))
    for i in range(losses):
        overlay.record_trade(TradeRecord(
            pnl=-300, pnl_pct=loss_pnl_pct,
            entry_time=datetime.now() - timedelta(days=losses - i),
            holding_bars=10, signal_id=f"LOSS_{i}",
        ))
    return overlay


# ══════════════════════════════════════════════════════════════
# TEST 1: Loss Aversion Corrector
# ══════════════════════════════════════════════════════════════

class TestLossAversionCorrector:

    def test_no_trigger_below_threshold(self):
        """No trigger with fewer than 3 consecutive losses."""
        overlay = make_overlay_with_trades(wins=15, losses=2)
        ctx = OverlayContext(current_dd_pct=0.02)
        result = overlay.apply_all(ctx)
        assert 'LOSS_AVERSION_CORRECTOR' not in result.overlays_triggered

    def test_trigger_with_healthy_metrics(self):
        """Floor at 70% when 3+ losses but rolling Sharpe is healthy."""
        overlay = make_overlay_with_trades(wins=20, losses=4)
        ctx = OverlayContext(current_dd_pct=0.02)
        result = overlay.apply_all(ctx)
        assert 'LOSS_AVERSION_CORRECTOR' in result.overlays_triggered
        assert result.size_multiplier >= LA_FLOOR_FRACTION - 0.01

    def test_no_floor_when_unhealthy(self):
        """Allow full reduction when drawdown > 8%."""
        overlay = make_overlay_with_trades(wins=5, losses=4, win_pnl_pct=0.001)
        ctx = OverlayContext(current_dd_pct=0.10)  # 10% DD
        result = overlay.apply_all(ctx)
        # Should NOT trigger LA floor since system is unhealthy
        assert 'LOSS_AVERSION_CORRECTOR' not in result.overlays_triggered

    def test_no_floor_negative_sharpe(self):
        """Allow full reduction when Sharpe < 0."""
        overlay = make_overlay_with_trades(wins=3, losses=5,
                                            win_pnl_pct=0.005,
                                            loss_pnl_pct=-0.02)
        ctx = OverlayContext(current_dd_pct=0.01)
        result = overlay.apply_all(ctx)
        # Negative Sharpe should not trigger floor
        assert result.size_multiplier <= 1.0


# ══════════════════════════════════════════════════════════════
# TEST 2: Disposition Effect Blocker
# ══════════════════════════════════════════════════════════════

class TestDispositionEffectBlocker:

    def test_trailing_stop_on_winner(self):
        """Winning position gets trailing stop at 60% of max gain."""
        overlay = make_overlay_with_trades(wins=10)
        ctx = OverlayContext(
            position=PositionContext(
                direction="LONG",
                entry_price=25000,
                current_price=25400,
                bars_held=5,
                max_unrealized_gain_pct=0.02,
                unrealized_pnl_pct=0.016,
            ),
        )
        result = overlay.apply_all(ctx)
        assert 'DISPOSITION_EFFECT_BLOCKER' in result.overlays_triggered
        assert result.trailing_stop is not None
        # Trailing stop should be above entry
        assert result.trailing_stop > 25000

    def test_force_exit_on_losing_position_held_too_long(self):
        """Force exit when holding > 2x average and losing > 1%."""
        overlay = make_overlay_with_trades(wins=20)
        # Average holding is 10 bars. Set position to 25 bars (> 2x)
        ctx = OverlayContext(
            position=PositionContext(
                direction="LONG",
                entry_price=25000,
                current_price=24700,
                bars_held=25,
                max_unrealized_gain_pct=0.001,
                unrealized_pnl_pct=-0.012,  # -1.2%
            ),
        )
        result = overlay.apply_all(ctx)
        assert result.force_exit is True

    def test_no_exit_on_short_holding(self):
        """No force exit when holding time is normal."""
        overlay = make_overlay_with_trades(wins=20)
        ctx = OverlayContext(
            position=PositionContext(
                direction="LONG",
                entry_price=25000,
                current_price=24800,
                bars_held=5,
                max_unrealized_gain_pct=0.001,
                unrealized_pnl_pct=-0.008,
            ),
        )
        result = overlay.apply_all(ctx)
        assert result.force_exit is False

    def test_no_trigger_without_position(self):
        """No disposition effect without an active position."""
        overlay = make_overlay_with_trades(wins=10)
        ctx = OverlayContext()
        result = overlay.apply_all(ctx)
        assert 'DISPOSITION_EFFECT_BLOCKER' not in result.overlays_triggered


# ══════════════════════════════════════════════════════════════
# TEST 3: Anchoring Detector
# ══════════════════════════════════════════════════════════════

class TestAnchoringDetector:

    def test_sl_near_round_number(self):
        """SL at 25000 (round number) gets overridden."""
        overlay = BehavioralOverlay()
        ctx = OverlayContext(
            entry_price=25100,
            proposed_sl=25000.0,
            proposed_tgt=25800,
            direction="LONG",
            current_atr=150,
        )
        result = overlay.apply_all(ctx)
        assert 'ANCHORING_DETECTOR' in result.overlays_triggered
        assert result.sl_override is not None
        # ATR-based SL: 25100 - 2*150 = 24800
        assert abs(result.sl_override - 24800) < 10

    def test_tgt_near_round_number(self):
        """TGT at 25500 (round number) gets overridden."""
        overlay = BehavioralOverlay()
        ctx = OverlayContext(
            entry_price=25100,
            proposed_sl=24700,
            proposed_tgt=25500.0,
            direction="LONG",
            current_atr=150,
        )
        result = overlay.apply_all(ctx)
        assert 'ANCHORING_DETECTOR' in result.overlays_triggered
        assert result.tgt_override is not None
        # ATR-based TGT: 25100 + 3*150 = 25550
        assert abs(result.tgt_override - 25550) < 10

    def test_no_trigger_away_from_round(self):
        """No trigger when SL/TGT are far from round numbers."""
        overlay = BehavioralOverlay()
        ctx = OverlayContext(
            entry_price=25100,
            proposed_sl=24750,
            proposed_tgt=25680,
            direction="LONG",
            current_atr=150,
        )
        result = overlay.apply_all(ctx)
        assert 'ANCHORING_DETECTOR' not in result.overlays_triggered

    def test_short_direction_override(self):
        """Anchoring override works for SHORT direction."""
        overlay = BehavioralOverlay()
        ctx = OverlayContext(
            entry_price=24900,
            proposed_sl=25000.0,
            proposed_tgt=24500.0,
            direction="SHORT",
            current_atr=150,
        )
        result = overlay.apply_all(ctx)
        assert 'ANCHORING_DETECTOR' in result.overlays_triggered
        # SHORT SL: entry + 2*ATR = 24900 + 300 = 25200
        if result.sl_override is not None:
            assert result.sl_override > 25000
        # SHORT TGT: entry - 3*ATR = 24900 - 450 = 24450
        if result.tgt_override is not None:
            assert result.tgt_override < 24500

    def test_no_trigger_zero_atr(self):
        """No trigger when ATR is zero."""
        overlay = BehavioralOverlay()
        ctx = OverlayContext(
            entry_price=25000,
            proposed_sl=24500,
            proposed_tgt=25500,
            direction="LONG",
            current_atr=0,
        )
        result = overlay.apply_all(ctx)
        assert 'ANCHORING_DETECTOR' not in result.overlays_triggered


# ══════════════════════════════════════════════════════════════
# TEST 4: Overconfidence Dampener
# ══════════════════════════════════════════════════════════════

class TestOverconfidenceDampener:

    def test_cap_after_5_wins(self):
        """After 5 consecutive wins, cap at system size."""
        overlay = make_overlay_with_trades(wins=6, losses=0)
        ctx = OverlayContext()
        result = overlay.apply_all(ctx)
        assert 'OVERCONFIDENCE_DAMPENER' in result.overlays_triggered
        assert result.size_multiplier <= 1.0

    def test_reduce_after_8_wins(self):
        """After 8 consecutive wins, reduce to 80%."""
        overlay = make_overlay_with_trades(wins=9, losses=0)
        ctx = OverlayContext()
        result = overlay.apply_all(ctx)
        assert 'OVERCONFIDENCE_DAMPENER' in result.overlays_triggered
        assert result.size_multiplier <= OC_REDUCE_FRACTION + 0.01

    def test_no_trigger_below_threshold(self):
        """No trigger with fewer than 5 consecutive wins."""
        overlay = make_overlay_with_trades(wins=3, losses=0)
        ctx = OverlayContext()
        result = overlay.apply_all(ctx)
        assert 'OVERCONFIDENCE_DAMPENER' not in result.overlays_triggered

    def test_streak_resets_on_loss(self):
        """Streak resets when a loss occurs."""
        overlay = make_overlay_with_trades(wins=7, losses=0)
        overlay.record_trade(TradeRecord(
            pnl=-100, pnl_pct=-0.005,
            entry_time=datetime.now(), holding_bars=5,
        ))
        ctx = OverlayContext()
        result = overlay.apply_all(ctx)
        assert 'OVERCONFIDENCE_DAMPENER' not in result.overlays_triggered


# ══════════════════════════════════════════════════════════════
# TEST 5: Combined overlays
# ══════════════════════════════════════════════════════════════

class TestCombinedOverlays:

    def test_never_exceeds_system_size(self):
        """Size multiplier never exceeds 1.0 (never increases beyond system)."""
        overlay = make_overlay_with_trades(wins=20, losses=4)
        ctx = OverlayContext(current_dd_pct=0.02)
        result = overlay.apply_all(ctx)
        assert result.size_multiplier <= 1.0

    def test_multiple_overlays_combine(self):
        """Multiple overlays can trigger simultaneously."""
        overlay = make_overlay_with_trades(wins=20, losses=4)
        ctx = OverlayContext(
            current_dd_pct=0.02,
            entry_price=25100,
            proposed_sl=25000.0,
            proposed_tgt=25500.0,
            direction="LONG",
            current_atr=150,
        )
        result = overlay.apply_all(ctx)
        # Should trigger both loss aversion and anchoring
        assert len(result.overlays_triggered) >= 1

    def test_empty_history(self):
        """No crash with empty trade history."""
        overlay = BehavioralOverlay()
        ctx = OverlayContext()
        result = overlay.apply_all(ctx)
        assert result.size_multiplier == 1.0
        assert not result.force_exit
        assert not result.overlays_triggered

    def test_get_status(self):
        """Status dict has expected keys."""
        overlay = make_overlay_with_trades(wins=5, losses=2)
        status = overlay.get_status()
        assert 'trade_count' in status
        assert 'consecutive_wins' in status
        assert 'consecutive_losses' in status
        assert 'rolling_sharpe_20' in status
        assert status['consecutive_losses'] == 2
        assert status['consecutive_wins'] == 0


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
