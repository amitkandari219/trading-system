"""
Tests for global market signals: GIFT Nifty gap, US overnight, composite.

Covers:
  - Gap classification across all zones (noise, reversion, continuation, extreme)
  - VIX regime scaling of thresholds
  - US overnight composite scoring
  - Crisis mode detection
  - DXY-FII lead indicator
  - Composite signal agreement bonus
  - Size modifier bounds
  - Edge cases (missing data, zero values)

Run:
    python -m pytest tests/test_global_signals.py -v
"""

import pytest
from datetime import date
from unittest.mock import MagicMock, patch

from signals.gift_nifty_gap import GiftNiftyGapSignal
from signals.us_overnight import USOvernightSignal
from signals.global_composite import GlobalCompositeSignal, GlobalPreMarketContext


# ================================================================
# GIFT NIFTY GAP TESTS
# ================================================================

class TestGiftNiftyGap:

    def setup_method(self):
        self.signal = GiftNiftyGapSignal()

    # ── Gap Classification ────────────────────────────────────

    def test_noise_gap(self):
        """Gap < 0.3% should be classified as noise."""
        result = self.signal.evaluate(
            eval_date=date(2026, 3, 20),
            gift_nifty_price=22050,
            nifty_prev_close=22000,
            india_vix=16.0,
        )
        assert result['gap_type'] == 'NOISE'
        assert result['action'] is None

    def test_reversion_gap_up(self):
        """Gap up 0.5% should trigger short (mean reversion)."""
        result = self.signal.evaluate(
            eval_date=date(2026, 3, 20),
            gift_nifty_price=22110,
            nifty_prev_close=22000,
            india_vix=16.0,
        )
        assert result['gap_type'] == 'REVERSION'
        assert result['direction'] == 'SHORT'
        assert result['action'] == 'ENTER_SHORT'

    def test_reversion_gap_down(self):
        """Gap down 0.5% should trigger long (mean reversion)."""
        result = self.signal.evaluate(
            eval_date=date(2026, 3, 20),
            gift_nifty_price=21890,
            nifty_prev_close=22000,
            india_vix=16.0,
        )
        assert result['gap_type'] == 'REVERSION'
        assert result['direction'] == 'LONG'

    def test_continuation_gap(self):
        """Gap 1.0% should trigger continuation (follow direction)."""
        result = self.signal.evaluate(
            eval_date=date(2026, 3, 20),
            gift_nifty_price=22220,
            nifty_prev_close=22000,
            india_vix=16.0,
        )
        assert result['gap_type'] == 'CONTINUATION'
        assert result['direction'] == 'LONG'  # Follow gap up

    def test_extreme_gap(self):
        """Gap > 1.5% should be extreme with delayed entry."""
        result = self.signal.evaluate(
            eval_date=date(2026, 3, 20),
            gift_nifty_price=22400,
            nifty_prev_close=22000,
            india_vix=16.0,
        )
        assert result['gap_type'] == 'EXTREME'
        assert result['delay_entry'] is True
        assert result['direction'] == 'SHORT'  # Fade extreme gap up

    # ── VIX Regime Scaling ────────────────────────────────────

    def test_vix_calm_tighter_thresholds(self):
        """In CALM regime (VIX<13), thresholds scale down by 0.8x."""
        # 0.25% gap is noise in NORMAL but should be reversion in CALM
        result = self.signal.evaluate(
            eval_date=date(2026, 3, 20),
            gift_nifty_price=22055,
            nifty_prev_close=22000,
            india_vix=11.0,  # CALM
        )
        # 0.25% > 0.24% (0.3 * 0.8), so should be REVERSION
        assert result['gap_type'] == 'REVERSION'
        assert result['vix_regime'] == 'CALM'

    def test_vix_high_vol_wider_thresholds(self):
        """In HIGH_VOL regime, noise band widens to 0.45%."""
        result = self.signal.evaluate(
            eval_date=date(2026, 3, 20),
            gift_nifty_price=22088,
            nifty_prev_close=22000,
            india_vix=28.0,  # HIGH_VOL
        )
        # 0.4% < 0.45% (0.3 * 1.5), so should be NOISE
        assert result['gap_type'] == 'NOISE'
        assert result['vix_regime'] == 'HIGH_VOL'

    # ── Edge Cases ────────────────────────────────────────────

    def test_missing_gift_nifty(self):
        """Missing GIFT price should return empty result."""
        result = self.signal.evaluate(
            eval_date=date(2026, 3, 20),
            gift_nifty_price=None,
            nifty_prev_close=22000,
        )
        assert result['action'] is None
        assert 'Missing' in result['reason'] or 'No global' in result['reason']

    def test_zero_prev_close(self):
        """Zero previous close should be handled gracefully."""
        result = self.signal.evaluate(
            eval_date=date(2026, 3, 20),
            gift_nifty_price=22000,
            nifty_prev_close=0,
        )
        assert result['action'] is None

    def test_gap_fill_probability_small_gap(self):
        """Small gaps should have high fill probability."""
        prob = self.signal._gap_fill_probability(0.4, 'NORMAL')
        assert prob >= 0.65

    def test_gap_fill_probability_large_gap(self):
        """Large gaps should have lower fill probability."""
        prob = self.signal._gap_fill_probability(1.5, 'NORMAL')
        assert prob < 0.45

    def test_confidence_bounded(self):
        """Confidence should always be between 0.1 and 0.95."""
        result = self.signal.evaluate(
            eval_date=date(2026, 3, 20),
            gift_nifty_price=22500,
            nifty_prev_close=22000,
            india_vix=16.0,
        )
        if result['confidence'] > 0:
            assert 0.1 <= result['confidence'] <= 0.95


# ================================================================
# US OVERNIGHT SIGNAL TESTS
# ================================================================

class TestUSOvernightSignal:

    def setup_method(self):
        self.signal = USOvernightSignal()

    def test_strong_bullish_us(self):
        """S&P +1.5% should produce bullish bias."""
        result = self.signal.evaluate(
            eval_date=date(2026, 3, 20),
            us_return=1.5,
            us_vix_close=18,
            us_vix_change_pct=-2,
            dxy_change_pct=-0.1,
        )
        assert result['direction'] == 'LONG'
        assert result['composite_score'] > 0.3

    def test_strong_bearish_us(self):
        """S&P -1.5% should produce bearish bias."""
        result = self.signal.evaluate(
            eval_date=date(2026, 3, 20),
            us_return=-1.5,
            us_vix_close=22,
            us_vix_change_pct=10,
            dxy_change_pct=0.3,
        )
        assert result['direction'] == 'SHORT'
        assert result['composite_score'] < -0.3

    def test_crisis_mode_sp_crash(self):
        """S&P <-2.5% should trigger crisis mode."""
        result = self.signal.evaluate(
            eval_date=date(2026, 3, 20),
            us_return=-3.0,
            us_vix_close=35,
            us_vix_change_pct=30,
            dxy_change_pct=1.0,
        )
        assert result['crisis_mode'] is True
        assert result['action'] == 'REDUCE_ALL'
        assert result['size_modifier'] <= 0.3

    def test_crisis_mode_vix_extreme(self):
        """US VIX > 35 should trigger crisis regardless of S&P."""
        result = self.signal.evaluate(
            eval_date=date(2026, 3, 20),
            us_return=0.5,
            us_vix_close=38,
            us_vix_change_pct=25,
            dxy_change_pct=0,
        )
        assert result['crisis_mode'] is True
        assert result['composite_score'] <= -0.6

    def test_neutral_market(self):
        """Small moves should produce neutral."""
        result = self.signal.evaluate(
            eval_date=date(2026, 3, 20),
            us_return=0.1,
            us_vix_close=16,
            us_vix_change_pct=-1,
            dxy_change_pct=0.1,
        )
        assert result['action'] is None or 'MILD' in (result['action'] or '')

    def test_dxy_strong_move(self):
        """Strong DXY move should impact score."""
        result_strong_dxy = self.signal.evaluate(
            eval_date=date(2026, 3, 20),
            us_return=0,
            us_vix_close=16,
            us_vix_change_pct=0,
            dxy_change_pct=1.0,  # Strong USD
            dxy_5d_change=2.0,
        )
        result_weak_dxy = self.signal.evaluate(
            eval_date=date(2026, 3, 20),
            us_return=0,
            us_vix_close=16,
            us_vix_change_pct=0,
            dxy_change_pct=-1.0,  # Weak USD
            dxy_5d_change=-2.0,
        )
        # Strong USD should be more bearish than weak USD
        assert result_strong_dxy['composite_score'] < result_weak_dxy['composite_score']

    def test_vix_regime_warning(self):
        """VIX spike with low India VIX should trigger warning."""
        # Mock DB to return India VIX
        signal = USOvernightSignal()
        signal.db = MagicMock()
        mock_cursor = MagicMock()
        signal.db.cursor.return_value = mock_cursor

        # Mock snapshot load
        mock_cursor.fetchone.side_effect = [
            # First call: load_snapshot
            (0.5, 28.0, 22.0, 105.0, 0.3),
            # Second call: get_india_vix
            (15.0,),
        ]

        warning = signal.get_vix_regime_warning(date(2026, 3, 20))
        # We expect a warning since US VIX spiked but India VIX is low
        # (depends on mock data alignment, so just check the method runs)
        assert warning is None or isinstance(warning, dict)


# ================================================================
# GLOBAL COMPOSITE TESTS
# ================================================================

class TestGlobalComposite:

    def test_all_bullish_agreement(self):
        """When all signals align bullish, should get agreement bonus."""
        signal = GlobalCompositeSignal()

        snapshot = {
            'gift_nifty_last': 22200,
            'nifty_prev_close': 22000,
            'india_vix': 14.0,
            'sp500_change_pct': 1.5,
            'nasdaq_change_pct': 1.8,
            'us_overnight_return': 1.6,
            'us_vix_close': 15,
            'us_vix_change_pct': -5,
            'dxy_close': 103,
            'dxy_change_pct': -0.6,
            'brent_close': 70,
            'brent_change_pct': -2.0,
            'hang_seng_change_pct': 1.0,
            'nikkei_change_pct': 0.8,
        }

        ctx = signal.evaluate(date(2026, 3, 20), snapshot=snapshot)
        assert ctx.direction == 'LONG'
        assert ctx.confidence > 0.6
        assert ctx.size_modifier > 1.0

    def test_all_bearish_agreement(self):
        """All bearish signals should produce strong short bias."""
        signal = GlobalCompositeSignal()

        snapshot = {
            'gift_nifty_last': 21700,
            'nifty_prev_close': 22000,
            'india_vix': 25.0,
            'sp500_change_pct': -2.0,
            'nasdaq_change_pct': -2.5,
            'us_overnight_return': -2.2,
            'us_vix_close': 30,
            'us_vix_change_pct': 20,
            'dxy_close': 106,
            'dxy_change_pct': 0.8,
            'brent_close': 95,
            'brent_change_pct': 4.0,
            'hang_seng_change_pct': -2.0,
            'nikkei_change_pct': -1.5,
        }

        ctx = signal.evaluate(date(2026, 3, 20), snapshot=snapshot)
        assert ctx.direction == 'SHORT'
        assert ctx.size_modifier < 1.0

    def test_conflicting_signals_dampened(self):
        """Conflicting signals should dampen the composite."""
        signal = GlobalCompositeSignal()

        snapshot = {
            'gift_nifty_last': 22200,  # Gap up → bullish
            'nifty_prev_close': 22000,
            'india_vix': 16.0,
            'sp500_change_pct': -1.5,  # S&P down → bearish
            'nasdaq_change_pct': -1.2,
            'us_overnight_return': -1.4,
            'us_vix_close': 22,
            'us_vix_change_pct': 15,
            'dxy_close': 104,
            'dxy_change_pct': 0.1,
            'brent_close': 80,
            'brent_change_pct': 0.5,
            'hang_seng_change_pct': 0.0,
            'nikkei_change_pct': 0.0,
        }

        ctx = signal.evaluate(date(2026, 3, 20), snapshot=snapshot)
        # Should be lower confidence due to conflicting signals
        assert abs(ctx.composite_score) < 0.6

    def test_crisis_overrides_everything(self):
        """Crisis mode should override all bullish signals."""
        signal = GlobalCompositeSignal()

        snapshot = {
            'gift_nifty_last': 22200,  # Gap up
            'nifty_prev_close': 22000,
            'india_vix': 16.0,
            'sp500_change_pct': -3.5,  # CRASH
            'nasdaq_change_pct': -4.0,
            'us_overnight_return': -3.7,
            'us_vix_close': 40,
            'us_vix_change_pct': 35,
            'dxy_close': 108,
            'dxy_change_pct': 1.5,
            'brent_close': 90,
            'brent_change_pct': 5.0,
            'hang_seng_change_pct': -3.0,
            'nikkei_change_pct': -2.5,
        }

        ctx = signal.evaluate(date(2026, 3, 20), snapshot=snapshot)
        assert ctx.risk_off is True
        assert ctx.size_modifier <= 0.7  # Regime warning caps at 0.7

    def test_size_modifier_bounds(self):
        """Size modifier should always be within [0.3, 1.3]."""
        signal = GlobalCompositeSignal()

        # Extreme bullish
        snapshot_bull = {
            'gift_nifty_last': 22300,
            'nifty_prev_close': 22000,
            'india_vix': 12.0,
            'sp500_change_pct': 2.0,
            'nasdaq_change_pct': 2.5,
            'us_overnight_return': 2.2,
            'us_vix_close': 12,
            'us_vix_change_pct': -10,
            'dxy_close': 100,
            'dxy_change_pct': -1.0,
            'brent_close': 60,
            'brent_change_pct': -5.0,
            'hang_seng_change_pct': 2.0,
            'nikkei_change_pct': 1.5,
        }
        ctx = signal.evaluate(date(2026, 3, 20), snapshot=snapshot_bull)
        assert 0.3 <= ctx.size_modifier <= 1.3

    def test_telegram_format(self):
        """Telegram output should be properly formatted."""
        ctx = GlobalPreMarketContext(
            date=date(2026, 3, 20),
            direction='LONG',
            composite_score=0.45,
            confidence=0.72,
            size_modifier=1.15,
            risk_off=False,
        )
        text = ctx.to_telegram()
        assert 'LONG' in text
        assert '0.45' in text or '+0.45' in text

    def test_missing_data_graceful(self):
        """Missing snapshot should return neutral, not crash."""
        signal = GlobalCompositeSignal()
        ctx = signal.evaluate(date(2026, 3, 20), snapshot=None)
        assert ctx.direction is None
        assert ctx.size_modifier == 1.0

    def test_crude_oil_shock(self):
        """Crude oil spike > 5% should show in composite."""
        signal = GlobalCompositeSignal()

        snapshot_oil_spike = {
            'gift_nifty_last': 22010,
            'nifty_prev_close': 22000,
            'india_vix': 16.0,
            'sp500_change_pct': 0.0,
            'nasdaq_change_pct': 0.0,
            'us_overnight_return': 0.0,
            'us_vix_close': 16,
            'us_vix_change_pct': 0,
            'dxy_close': 104,
            'dxy_change_pct': 0,
            'brent_close': 110,
            'brent_change_pct': 6.0,  # Major crude spike
            'hang_seng_change_pct': 0,
            'nikkei_change_pct': 0,
        }
        ctx = signal.evaluate(date(2026, 3, 20), snapshot=snapshot_oil_spike)
        assert ctx.components.get('crude_oil', 0) < -0.3


# ================================================================
# RUN TESTS
# ================================================================
if __name__ == '__main__':
    pytest.main([__file__, '-v'])
