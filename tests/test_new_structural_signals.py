"""
Tests for 8 structural signals:
    - EODInstitutionalFlowSignal  (8 tests)
    - GammaSqueezeSignal          (8 tests)
    - FIIDivergenceSignal         (7 tests)
    - OpeningCandleSignal         (8 tests)
    - SIPFlowSignal               (8 tests)
    - SkewReversalSignal          (7 tests)
    - ThursdayPinSetupSignal      (7 tests)
    - RBIDriftSignal              (7 tests)

Total: 60 tests.
"""

import pytest
from datetime import date

from signals.structural.eod_institutional_flow import EODInstitutionalFlowSignal
from signals.structural.gamma_squeeze import GammaSqueezeSignal
from signals.structural.fii_divergence import FIIDivergenceSignal
from signals.structural.opening_candle import OpeningCandleSignal
from signals.structural.sip_flow import SIPFlowSignal
from signals.structural.skew_reversal import SkewReversalSignal
from signals.structural.thursday_pin_setup import ThursdayPinSetupSignal
from signals.structural.rbi_drift import RBIDriftSignal


# ===================================================================
#  EOD INSTITUTIONAL FLOW TESTS (8)
# ===================================================================

class TestEODInstitutionalFlow:
    """Tests for EODInstitutionalFlowSignal."""

    def setup_method(self):
        self.sig = EODInstitutionalFlowSignal()

    def test_strong_buying(self):
        """3x volume surge + 50% dominance + 0.6% return + 55% delivery = score 8 STRONG LONG."""
        result = self.sig.evaluate({
            'last_hour_volume': 3000000,
            'avg_last_hour_volume_20d': 1000000,
            'morning_volume': 6000000,
            'last_hour_open': 24000.0,
            'last_hour_close': 24150.0,   # +0.625%
            'day_of_week': 2,             # Wednesday
            'delivery_pct': 55.0,
        })
        assert result is not None
        assert result['direction'] == 'LONG'
        assert result['strength'] == 'STRONG'
        assert result['composite_score'] >= 8
        assert result['signal_id'] == 'EOD_INSTITUTIONAL_FLOW'

    def test_strong_selling(self):
        """Strong selling signal: large volume + return down."""
        result = self.sig.evaluate({
            'last_hour_volume': 3200000,
            'avg_last_hour_volume_20d': 1000000,
            'morning_volume': 5000000,
            'last_hour_open': 24200.0,
            'last_hour_close': 24050.0,   # -0.62%
            'day_of_week': 1,             # Tuesday
            'delivery_pct': 60.0,
        })
        assert result is not None
        assert result['direction'] == 'SHORT'
        assert result['strength'] == 'STRONG'

    def test_friday_multiplier(self):
        """Friday boosts composite_score +1 and hold_days = 2."""
        result = self.sig.evaluate({
            'last_hour_volume': 3000000,
            'avg_last_hour_volume_20d': 1000000,
            'morning_volume': 6000000,
            'last_hour_open': 24000.0,
            'last_hour_close': 24150.0,
            'day_of_week': 4,             # Friday
            'delivery_pct': 55.0,
        })
        assert result is not None
        assert result['is_friday'] is True
        assert result['hold_days'] == 2
        # Friday adds +1 to score
        assert result['composite_score'] >= 9

    def test_weak_no_signal(self):
        """Low volume + small move = score below threshold = no signal."""
        result = self.sig.evaluate({
            'last_hour_volume': 800000,
            'avg_last_hour_volume_20d': 1000000,
            'morning_volume': 5000000,
            'last_hour_open': 24000.0,
            'last_hour_close': 24010.0,   # +0.04%
            'day_of_week': 1,
        })
        assert result is None

    def test_delivery_boost(self):
        """Delivery > 50% adds +1 to score."""
        # Without delivery: vol_surge=2, dominance=0, return=2 = 4 (WEAK)
        base = self.sig.evaluate({
            'last_hour_volume': 2000000,
            'avg_last_hour_volume_20d': 1000000,
            'morning_volume': 10000000,
            'last_hour_open': 24000.0,
            'last_hour_close': 24150.0,
            'day_of_week': 2,
        })
        # With delivery: same + delivery_score=1 = 5 (WEAK)
        with_delivery = self.sig.evaluate({
            'last_hour_volume': 2000000,
            'avg_last_hour_volume_20d': 1000000,
            'morning_volume': 10000000,
            'last_hour_open': 24000.0,
            'last_hour_close': 24150.0,
            'day_of_week': 2,
            'delivery_pct': 55.0,
        })
        assert with_delivery is not None
        if base is not None:
            assert with_delivery['composite_score'] > base['composite_score']
        assert with_delivery['delivery_score'] == 1

    def test_hold_2_friday(self):
        """Friday signals hold for 2 days."""
        result = self.sig.evaluate({
            'last_hour_volume': 3000000,
            'avg_last_hour_volume_20d': 1000000,
            'morning_volume': 5000000,
            'last_hour_open': 24000.0,
            'last_hour_close': 24200.0,
            'day_of_week': 4,
            'delivery_pct': 55.0,
        })
        assert result is not None
        assert result['hold_days'] == 2

    def test_hold_1_weekday(self):
        """Non-Friday signals hold for 1 day."""
        result = self.sig.evaluate({
            'last_hour_volume': 3000000,
            'avg_last_hour_volume_20d': 1000000,
            'morning_volume': 5000000,
            'last_hour_open': 24000.0,
            'last_hour_close': 24200.0,
            'day_of_week': 2,
            'delivery_pct': 55.0,
        })
        assert result is not None
        assert result['hold_days'] == 1

    def test_neutral_low_score(self):
        """Score 3 (below WEAK=4 threshold) returns None."""
        # vol_surge=1 (1.5x), dominance=0 (low ratio), return=1 (0.3%), delivery=1 = 3
        result = self.sig.evaluate({
            'last_hour_volume': 1500000,
            'avg_last_hour_volume_20d': 1000000,
            'morning_volume': 20000000,     # very low dominance
            'last_hour_open': 24000.0,
            'last_hour_close': 24080.0,     # 0.33%
            'day_of_week': 1,
            'delivery_pct': 55.0,
        })
        assert result is None


# ===================================================================
#  GAMMA SQUEEZE TESTS (8)
# ===================================================================

class TestGammaSqueeze:
    """Tests for GammaSqueezeSignal."""

    def setup_method(self):
        self.sig = GammaSqueezeSignal()

    def test_bullish_squeeze(self):
        """Expiry day + 1.2% up move + high ATM OI + acceleration = STRONG LONG."""
        result = self.sig.evaluate({
            'day_of_week': 1,            # Tuesday (expiry)
            'days_to_weekly_expiry': 0,
            'day_open': 24000.0,
            'current_price': 24360.0,    # +1.5%
            'atm_oi_pct_of_total': 0.22, # extreme OI concentration
            'first_30min_move_pct': 0.008,  # +0.8%
            'next_45min_move_pct': 0.010,   # +1.0% acceleration
        })
        assert result is not None
        assert result['direction'] == 'LONG'
        assert result['strength'] == 'STRONG'
        assert result['signal_id'] == 'GAMMA_SQUEEZE'

    def test_bearish_squeeze(self):
        """Expiry day + down move + OI + acceleration = SHORT."""
        result = self.sig.evaluate({
            'day_of_week': 1,
            'days_to_weekly_expiry': 0,
            'day_open': 24000.0,
            'current_price': 23640.0,    # -1.5%
            'atm_oi_pct_of_total': 0.15,
            'first_30min_move_pct': -0.008,
            'next_45min_move_pct': -0.009,
        })
        assert result is not None
        assert result['direction'] == 'SHORT'

    def test_no_signal_outside_expiry(self):
        """days_to_weekly_expiry > 1 returns None."""
        result = self.sig.evaluate({
            'day_of_week': 3,            # Thursday
            'days_to_weekly_expiry': 5,  # far from expiry
            'day_open': 24000.0,
            'current_price': 24300.0,
            'atm_oi_pct_of_total': 0.15,
            'first_30min_move_pct': 0.008,
            'next_45min_move_pct': 0.005,
        })
        assert result is None

    def test_small_move(self):
        """Move < 0.5% returns None."""
        result = self.sig.evaluate({
            'day_of_week': 0,
            'days_to_weekly_expiry': 1,
            'day_open': 24000.0,
            'current_price': 24050.0,    # +0.21% < 0.5%
            'atm_oi_pct_of_total': 0.15,
            'first_30min_move_pct': 0.001,
            'next_45min_move_pct': 0.001,
        })
        assert result is None

    def test_low_gamma(self):
        """Low ATM OI concentration scores poorly (oi_score=0)."""
        result = self.sig.evaluate({
            'day_of_week': 0,
            'days_to_weekly_expiry': 1,
            'day_open': 24000.0,
            'current_price': 24300.0,    # +1.25%
            'atm_oi_pct_of_total': 0.05, # below 10% threshold
            'first_30min_move_pct': 0.006,
            'next_45min_move_pct': 0.001, # weak acceleration
        })
        # May or may not fire depending on total score
        if result is not None:
            assert result['oi_score'] == 0

    def test_direction_reversal(self):
        """Opposite acceleration direction = accel_score 0."""
        result = self.sig.evaluate({
            'day_of_week': 0,
            'days_to_weekly_expiry': 1,
            'day_open': 24000.0,
            'current_price': 24200.0,    # +0.83%
            'atm_oi_pct_of_total': 0.12,
            'first_30min_move_pct': 0.005,
            'next_45min_move_pct': -0.003,  # reversal
        })
        if result is not None:
            assert result['accel_score'] == 0

    def test_expiry_bonus(self):
        """Expiry day (DTE=0) gets +2 bonus to composite_score."""
        result = self.sig.evaluate({
            'day_of_week': 1,
            'days_to_weekly_expiry': 0,   # expiry day
            'day_open': 24000.0,
            'current_price': 24300.0,     # +1.25%
            'atm_oi_pct_of_total': 0.12,
            'first_30min_move_pct': 0.006,
            'next_45min_move_pct': 0.007,
        })
        assert result is not None
        assert result['expiry_bonus'] == 2

    def test_oi_scoring(self):
        """ATM OI 20%+ of total = oi_score 2, 10-20% = 1, <10% = 0."""
        # Extreme OI
        r1 = self.sig.evaluate({
            'day_of_week': 1, 'days_to_weekly_expiry': 0,
            'day_open': 24000.0, 'current_price': 24400.0,
            'atm_oi_pct_of_total': 0.25,
            'first_30min_move_pct': 0.01, 'next_45min_move_pct': 0.01,
        })
        assert r1 is not None
        assert r1['oi_score'] == 2

        # High OI
        r2 = self.sig.evaluate({
            'day_of_week': 1, 'days_to_weekly_expiry': 0,
            'day_open': 24000.0, 'current_price': 24400.0,
            'atm_oi_pct_of_total': 0.12,
            'first_30min_move_pct': 0.01, 'next_45min_move_pct': 0.01,
        })
        assert r2 is not None
        assert r2['oi_score'] == 1


# ===================================================================
#  FII DIVERGENCE TESTS (7)
# ===================================================================

class TestFIIDivergence:
    """Tests for FIIDivergenceSignal."""

    def setup_method(self):
        self.sig = FIIDivergenceSignal()

    def test_hedged_bullish(self):
        """Long futures + long puts = HEDGED_BULLISH, size 1.15."""
        result = self.sig.evaluate({
            'fii_index_fut_net_contracts': 50000,
            'fii_index_ce_net_oi': 0,
            'fii_index_pe_net_oi': 20000,
        })
        assert result is not None
        assert result['regime'] == 'HEDGED_BULLISH'
        assert result['size_multiplier'] == 1.15
        assert result['direction'] == 'LONG'
        assert result['signal_id'] == 'FII_DIVERGENCE'

    def test_aggressive_bullish(self):
        """Long futures + short calls = AGGRESSIVE_BULLISH, size 1.05."""
        result = self.sig.evaluate({
            'fii_index_fut_net_contracts': 30000,
            'fii_index_ce_net_oi': -5000,  # short calls
            'fii_index_pe_net_oi': 0,
        })
        assert result is not None
        assert result['regime'] == 'AGGRESSIVE_BULLISH'
        assert result['size_multiplier'] == 1.05
        assert result['direction'] == 'LONG'

    def test_hedged_bearish(self):
        """Short futures + long calls = HEDGED_BEARISH, size 0.85."""
        result = self.sig.evaluate({
            'fii_index_fut_net_contracts': -40000,
            'fii_index_ce_net_oi': 15000,  # long calls (hedge)
            'fii_index_pe_net_oi': 0,
        })
        assert result is not None
        assert result['regime'] == 'HEDGED_BEARISH'
        assert result['size_multiplier'] == 0.85
        assert result['direction'] == 'SHORT'

    def test_neutral(self):
        """Small positions near zero = NEUTRAL, size 1.0."""
        result = self.sig.evaluate({
            'fii_index_fut_net_contracts': 500,   # below 1000 threshold
            'fii_index_ce_net_oi': 200,
            'fii_index_pe_net_oi': 100,
        })
        assert result is not None
        assert result['regime'] == 'NEUTRAL'
        assert result['size_multiplier'] == 1.0

    def test_proxy_mode(self):
        """When FII data unavailable, uses PCR + VIX proxy."""
        result = self.sig.evaluate({
            'put_call_ratio': 1.4,      # high PCR
            'india_vix': 22.0,          # high VIX
            'nifty_daily_change_pct': -0.5,
        })
        assert result is not None
        assert result['using_proxy'] is True
        assert result['regime'] == 'HEDGED_BULLISH'
        assert result['direction'] == 'LONG'

    def test_position_change_boost(self):
        """Large FII futures change boosts confidence by 0.05."""
        r1 = self.sig.evaluate({
            'fii_index_fut_net_contracts': 50000,
            'fii_index_ce_net_oi': 0,
            'fii_index_pe_net_oi': 20000,
            'fii_fut_net_change': 100,   # small change
        })
        r2 = self.sig.evaluate({
            'fii_index_fut_net_contracts': 50000,
            'fii_index_ce_net_oi': 0,
            'fii_index_pe_net_oi': 20000,
            'fii_fut_net_change': 10000, # large change
        })
        assert r1 is not None and r2 is not None
        assert r2['confidence'] > r1['confidence']

    def test_size_modifier_range(self):
        """All size_multiplier values between 0.85 and 1.15."""
        test_cases = [
            {'fii_index_fut_net_contracts': 50000, 'fii_index_ce_net_oi': 0,
             'fii_index_pe_net_oi': 20000},
            {'fii_index_fut_net_contracts': -40000, 'fii_index_ce_net_oi': 15000,
             'fii_index_pe_net_oi': 0},
            {'fii_index_fut_net_contracts': 500, 'fii_index_ce_net_oi': 200,
             'fii_index_pe_net_oi': 100},
            {'put_call_ratio': 1.4, 'india_vix': 22.0},
        ]
        for md in test_cases:
            result = self.sig.evaluate(md)
            assert result is not None
            assert 0.85 <= result['size_multiplier'] <= 1.15


# ===================================================================
#  OPENING CANDLE TESTS (8)
# ===================================================================

class TestOpeningCandle:
    """Tests for OpeningCandleSignal."""

    def setup_method(self):
        self.sig = OpeningCandleSignal()

    def test_large_bullish(self):
        """Large body + high volume = bullish signal."""
        result = self.sig.evaluate({
            'first_15min_open': 24000.0,
            'first_15min_close': 24200.0,    # +0.83% body
            'first_15min_high': 24220.0,
            'first_15min_low': 23990.0,
            'first_15min_volume': 2000000,
            'avg_first_15min_volume_20d': 800000,
            'prev_close': 24000.0,           # no gap
        })
        assert result is not None
        assert result['direction'] == 'LONG'
        assert result['signal_id'] == 'OPENING_CANDLE'
        assert result['body_pct'] > 0.3

    def test_gap_continuation(self):
        """Gap up + bullish candle = GAP_CONTINUATION."""
        result = self.sig.evaluate({
            'first_15min_open': 24100.0,      # gap up from 24000 = +0.42%
            'first_15min_close': 24250.0,     # bullish candle
            'first_15min_high': 24260.0,
            'first_15min_low': 24090.0,
            'first_15min_volume': 1800000,
            'avg_first_15min_volume_20d': 800000,
            'prev_close': 24000.0,
        })
        assert result is not None
        assert result['candle_type'] == 'GAP_CONTINUATION'
        assert result['direction'] == 'LONG'

    def test_gap_fade(self):
        """Gap up + bearish candle = GAP_FADE with shorter hold."""
        result = self.sig.evaluate({
            'first_15min_open': 24100.0,      # gap up from 24000
            'first_15min_close': 24000.0,     # bearish candle (fading gap)
            'first_15min_high': 24120.0,
            'first_15min_low': 23990.0,
            'first_15min_volume': 1800000,
            'avg_first_15min_volume_20d': 800000,
            'prev_close': 24000.0,
        })
        assert result is not None
        assert result['candle_type'] == 'GAP_FADE'
        assert result['direction'] == 'SHORT'
        assert result['hold_minutes'] == 120

    def test_small_candle(self):
        """Candle body < 0.3% returns None."""
        result = self.sig.evaluate({
            'first_15min_open': 24000.0,
            'first_15min_close': 24020.0,     # +0.08% body
            'first_15min_high': 24030.0,
            'first_15min_low': 23990.0,
            'first_15min_volume': 1500000,
            'avg_first_15min_volume_20d': 800000,
            'prev_close': 24000.0,
        })
        assert result is None

    def test_low_volume(self):
        """Volume < 1.5x average returns None."""
        result = self.sig.evaluate({
            'first_15min_open': 24000.0,
            'first_15min_close': 24200.0,
            'first_15min_high': 24220.0,
            'first_15min_low': 23990.0,
            'first_15min_volume': 800000,     # ratio = 1.0 < 1.5
            'avg_first_15min_volume_20d': 800000,
            'prev_close': 24000.0,
        })
        assert result is None

    def test_body_ratio(self):
        """Extreme body (1%+) boosts confidence more than strong body (0.6%+)."""
        r_strong = self.sig.evaluate({
            'first_15min_open': 24000.0,
            'first_15min_close': 24160.0,     # +0.67% body (strong)
            'first_15min_high': 24170.0,
            'first_15min_low': 23990.0,
            'first_15min_volume': 1600000,
            'avg_first_15min_volume_20d': 800000,
            'prev_close': 24000.0,
        })
        r_extreme = self.sig.evaluate({
            'first_15min_open': 24000.0,
            'first_15min_close': 24300.0,     # +1.25% body (extreme)
            'first_15min_high': 24310.0,
            'first_15min_low': 23990.0,
            'first_15min_volume': 1600000,
            'avg_first_15min_volume_20d': 800000,
            'prev_close': 24000.0,
        })
        assert r_strong is not None and r_extreme is not None
        assert r_extreme['confidence'] > r_strong['confidence']

    def test_fade_shorter_hold(self):
        """GAP_FADE hold = 120 min, others = 240 min."""
        # Gap continuation
        r_cont = self.sig.evaluate({
            'first_15min_open': 24100.0,
            'first_15min_close': 24250.0,
            'first_15min_high': 24260.0,
            'first_15min_low': 24090.0,
            'first_15min_volume': 1800000,
            'avg_first_15min_volume_20d': 800000,
            'prev_close': 24000.0,
        })
        assert r_cont is not None
        assert r_cont['hold_minutes'] == 240

    def test_target_by_pattern(self):
        """GAP_FADE has gap_fill_target = prev_close; others don't."""
        # Gap fade
        r_fade = self.sig.evaluate({
            'first_15min_open': 24100.0,
            'first_15min_close': 24000.0,
            'first_15min_high': 24120.0,
            'first_15min_low': 23990.0,
            'first_15min_volume': 1800000,
            'avg_first_15min_volume_20d': 800000,
            'prev_close': 24000.0,
        })
        assert r_fade is not None
        assert r_fade['gap_fill_target'] == 24000.0

        # Fresh move (no gap)
        r_fresh = self.sig.evaluate({
            'first_15min_open': 24000.0,
            'first_15min_close': 24200.0,
            'first_15min_high': 24220.0,
            'first_15min_low': 23990.0,
            'first_15min_volume': 1800000,
            'avg_first_15min_volume_20d': 800000,
            'prev_close': 24000.0,
        })
        assert r_fresh is not None
        assert r_fresh['gap_fill_target'] is None


# ===================================================================
#  SIP FLOW TESTS (8)
# ===================================================================

class TestSIPFlow:
    """Tests for SIPFlowSignal."""

    def setup_method(self):
        self.sig = SIPFlowSignal()

    def test_primary_date(self):
        """Primary SIP date (5th) with falling market + high VIX fires LONG."""
        result = self.sig.evaluate({
            'date': date(2026, 3, 5),
            'nifty_5d_return_pct': -2.5,
            'india_vix': 23.0,
            'monthly_sip_flow_crore': 23000,
        })
        assert result is not None
        assert result['direction'] == 'LONG'
        assert result['date_type'] == 'PRIMARY'
        assert result['signal_id'] == 'SIP_FLOW'
        # date_score=3, return_score=2, vix_score=2, flow_score=1 = 8
        assert result['composite_score'] >= 5

    def test_secondary_date(self):
        """Secondary SIP date (15th) with enough supporting factors fires."""
        result = self.sig.evaluate({
            'date': date(2026, 3, 15),
            'nifty_5d_return_pct': -1.5,
            'india_vix': 19.0,
            'monthly_sip_flow_crore': 23000,
        })
        assert result is not None
        assert result['date_type'] == 'SECONDARY'
        assert result['direction'] == 'LONG'
        # date_score=2, return_score=1, vix_score=1, flow_score=1 = 5

    def test_non_sip_date(self):
        """Non-SIP date (12th) returns None."""
        result = self.sig.evaluate({
            'date': date(2026, 3, 12),
            'nifty_5d_return_pct': -3.0,
            'india_vix': 25.0,
        })
        assert result is None

    def test_falling_market_boost(self):
        """5d return < -1% boosts confidence."""
        r_flat = self.sig.evaluate({
            'date': date(2026, 3, 1),
            'nifty_5d_return_pct': 0.5,
            'india_vix': 19.0,
            'monthly_sip_flow_crore': 23000,
        })
        r_falling = self.sig.evaluate({
            'date': date(2026, 3, 1),
            'nifty_5d_return_pct': -2.5,
            'india_vix': 19.0,
            'monthly_sip_flow_crore': 23000,
        })
        assert r_falling is not None
        if r_flat is not None:
            assert r_falling['confidence'] > r_flat['confidence']

    def test_high_vix_boost(self):
        """VIX > 18 boosts confidence."""
        r_low_vix = self.sig.evaluate({
            'date': date(2026, 3, 5),
            'nifty_5d_return_pct': -2.0,
            'india_vix': 14.0,
            'monthly_sip_flow_crore': 23000,
        })
        r_high_vix = self.sig.evaluate({
            'date': date(2026, 3, 5),
            'nifty_5d_return_pct': -2.0,
            'india_vix': 23.0,
            'monthly_sip_flow_crore': 23000,
        })
        assert r_high_vix is not None
        if r_low_vix is not None:
            assert r_high_vix['confidence'] > r_low_vix['confidence']

    def test_always_bullish(self):
        """SIP flow is always LONG (never SHORT)."""
        result = self.sig.evaluate({
            'date': date(2026, 3, 5),
            'nifty_5d_return_pct': -3.0,
            'india_vix': 25.0,
            'monthly_sip_flow_crore': 23000,
        })
        assert result is not None
        assert result['direction'] == 'LONG'

    def test_weak_filtered(self):
        """Secondary date with no supporting factors (score < 5) returns None."""
        result = self.sig.evaluate({
            'date': date(2026, 3, 15),
            'nifty_5d_return_pct': 1.0,      # positive return
            'india_vix': 12.0,               # low VIX
            'monthly_sip_flow_crore': 15000, # low SIP flow
        })
        # date_score=2, return_score=0, vix_score=0, flow_score=0 = 2 < 5
        assert result is None

    def test_spillover(self):
        """Date with next day also SIP date gets spillover boost."""
        # March 7 is SIP, March 8 is not. March 9 is not SIP, March 10 is SIP.
        r_no_spill = self.sig.evaluate({
            'date': date(2026, 3, 7),
            'nifty_5d_return_pct': -2.0,
            'india_vix': 19.0,
            'monthly_sip_flow_crore': 23000,
        })
        assert r_no_spill is not None
        assert r_no_spill['spillover'] is False


# ===================================================================
#  SKEW REVERSAL TESTS (7)
# ===================================================================

class TestSkewReversal:
    """Tests for SkewReversalSignal."""

    def setup_method(self):
        self.sig = SkewReversalSignal()

    def test_put_skew_reversion(self):
        """Extreme put skew reverting = LONG."""
        result = self.sig.evaluate({
            'atm_put_iv': 18.0,
            'atm_call_iv': 14.0,       # skew = +4 (extreme put)
            'prev_day_skew': 4.5,
            'skew_5d_ago': 6.0,        # was 6, now 4 = 2 pts reversion
            'avg_skew_20d': 1.5,
        })
        assert result is not None
        assert result['direction'] == 'LONG'
        assert result['extreme_type'] == 'EXTREME_PUT_SKEW'
        assert result['proxy_mode'] is False
        assert result['signal_id'] == 'SKEW_REVERSAL'

    def test_call_skew_reversion(self):
        """Extreme call skew reverting = SHORT."""
        result = self.sig.evaluate({
            'atm_put_iv': 13.0,
            'atm_call_iv': 16.0,       # skew = -3 (extreme call)
            'prev_day_skew': -3.5,
            'skew_5d_ago': -4.5,       # was -4.5, now -3 = 1.5 pts reversion
            'avg_skew_20d': 1.0,
        })
        assert result is not None
        assert result['direction'] == 'SHORT'
        assert result['extreme_type'] == 'EXTREME_CALL_SKEW'

    def test_extreme_no_reversion(self):
        """Extreme skew but no reversion (< 1 pt) = None."""
        result = self.sig.evaluate({
            'atm_put_iv': 18.0,
            'atm_call_iv': 14.0,       # skew = +4 (extreme put)
            'prev_day_skew': 4.2,
            'skew_5d_ago': 4.5,        # was 4.5, now 4 = 0.5 pts < 1 minimum
            'avg_skew_20d': 1.5,
        })
        assert result is None

    def test_no_extreme(self):
        """Skew within normal range (not extreme) = None."""
        result = self.sig.evaluate({
            'atm_put_iv': 15.5,
            'atm_call_iv': 14.5,       # skew = +1 (normal)
            'prev_day_skew': 1.2,
            'skew_5d_ago': 1.5,
            'avg_skew_20d': 1.0,
        })
        assert result is None

    def test_proxy(self):
        """Proxy mode fires when IV unavailable but VIX + PCR indicate extreme."""
        result = self.sig.evaluate({
            'india_vix': 22.0,
            'prev_india_vix': 23.0,         # VIX dropping (reverting)
            'put_call_ratio': 1.5,
            'prev_put_call_ratio': 1.6,     # PCR dropping (reverting)
        })
        assert result is not None
        assert result['proxy_mode'] is True
        assert result['direction'] == 'LONG'
        assert result['extreme_type'] == 'EXTREME_PUT_SKEW'

    def test_confidence_scaling(self):
        """Larger reversion and distance from mean = higher confidence."""
        r_small = self.sig.evaluate({
            'atm_put_iv': 17.5,
            'atm_call_iv': 14.0,       # skew = +3.5
            'prev_day_skew': 3.8,
            'skew_5d_ago': 4.6,        # 1.1 pts reversion
            'avg_skew_20d': 1.5,
        })
        r_large = self.sig.evaluate({
            'atm_put_iv': 19.0,
            'atm_call_iv': 14.0,       # skew = +5
            'prev_day_skew': 5.5,
            'skew_5d_ago': 8.0,        # 3 pts reversion
            'avg_skew_20d': 1.5,
        })
        assert r_small is not None and r_large is not None
        assert r_large['confidence'] > r_small['confidence']

    def test_max_hold(self):
        """Max hold bars = 48."""
        result = self.sig.evaluate({
            'atm_put_iv': 18.0,
            'atm_call_iv': 14.0,
            'prev_day_skew': 4.5,
            'skew_5d_ago': 6.0,
            'avg_skew_20d': 1.5,
        })
        assert result is not None
        assert result['max_hold_bars'] == 48


# ===================================================================
#  THURSDAY PIN SETUP TESTS (7)
# ===================================================================

class TestThursdayPinSetup:
    """Tests for ThursdayPinSetupSignal."""

    def setup_method(self):
        self.sig = ThursdayPinSetupSignal()

    def test_near_put_floor(self):
        """Spot near max put OI buildup strike = LONG."""
        result = self.sig.evaluate({
            'day_of_week': 3,              # Thursday
            'spot_price': 24050.0,
            'next_week_put_oi_by_strike': {24000: 500000, 24100: 200000, 23900: 100000},
            'next_week_call_oi_by_strike': {24500: 600000, 24400: 200000},
            'prev_next_week_put_oi_by_strike': {24000: 200000, 24100: 100000, 23900: 50000},
            'prev_next_week_call_oi_by_strike': {24500: 300000, 24400: 100000},
        })
        assert result is not None
        assert result['direction'] == 'LONG'
        assert result['near_support'] is True
        assert result['signal_id'] == 'THURSDAY_PIN_SETUP'

    def test_near_call_ceiling(self):
        """Spot near max call OI buildup strike = SHORT."""
        result = self.sig.evaluate({
            'day_of_week': 4,              # Friday
            'spot_price': 24480.0,
            'next_week_put_oi_by_strike': {24000: 500000, 24100: 200000},
            'next_week_call_oi_by_strike': {24500: 600000, 24400: 200000},
            'prev_next_week_put_oi_by_strike': {24000: 200000, 24100: 100000},
            'prev_next_week_call_oi_by_strike': {24500: 250000, 24400: 100000},
        })
        assert result is not None
        assert result['direction'] == 'SHORT'
        assert result['near_resistance'] is True

    def test_not_thursday(self):
        """Monday (day_of_week=0) returns None."""
        result = self.sig.evaluate({
            'day_of_week': 0,
            'spot_price': 24050.0,
            'next_week_put_oi_by_strike': {24000: 500000},
            'next_week_call_oi_by_strike': {24500: 600000},
        })
        assert result is None

    def test_no_oi_proxy(self):
        """Proxy mode with max_put_oi_strike / max_call_oi_strike."""
        result = self.sig.evaluate({
            'day_of_week': 3,
            'spot_price': 24050.0,
            'max_put_oi_strike': 24000,
            'max_call_oi_strike': 24500,
        })
        assert result is not None
        assert result['proxy_mode'] is True
        assert result['direction'] == 'LONG'

    def test_concentration(self):
        """Concentrated OI (max > 2x second highest) boosts confidence."""
        r_spread = self.sig.evaluate({
            'day_of_week': 3,
            'spot_price': 24050.0,
            'next_week_put_oi_by_strike': {24000: 500000, 24100: 400000, 23900: 300000},
            'next_week_call_oi_by_strike': {24500: 600000},
            'prev_next_week_put_oi_by_strike': {24000: 100000, 24100: 100000, 23900: 100000},
            'prev_next_week_call_oi_by_strike': {24500: 200000},
        })
        r_concentrated = self.sig.evaluate({
            'day_of_week': 3,
            'spot_price': 24050.0,
            'next_week_put_oi_by_strike': {24000: 500000, 24100: 100000, 23900: 50000},
            'next_week_call_oi_by_strike': {24500: 600000},
            'prev_next_week_put_oi_by_strike': {24000: 100000, 24100: 50000, 23900: 25000},
            'prev_next_week_call_oi_by_strike': {24500: 200000},
        })
        assert r_spread is not None and r_concentrated is not None
        assert r_concentrated['concentrated'] is True
        assert r_concentrated['confidence'] >= r_spread['confidence']

    def test_size_mod_support(self):
        """Near support = size_modifier 1.10."""
        result = self.sig.evaluate({
            'day_of_week': 3,
            'spot_price': 24050.0,
            'next_week_put_oi_by_strike': {24000: 500000},
            'next_week_call_oi_by_strike': {24500: 600000},
            'prev_next_week_put_oi_by_strike': {24000: 100000},
            'prev_next_week_call_oi_by_strike': {24500: 200000},
        })
        assert result is not None
        assert result['size_modifier'] == 1.10

    def test_size_mod_resistance(self):
        """Near resistance = size_modifier 0.90."""
        result = self.sig.evaluate({
            'day_of_week': 3,
            'spot_price': 24480.0,
            'next_week_put_oi_by_strike': {24000: 500000},
            'next_week_call_oi_by_strike': {24500: 600000},
            'prev_next_week_put_oi_by_strike': {24000: 100000},
            'prev_next_week_call_oi_by_strike': {24500: 200000},
        })
        assert result is not None
        assert result['size_modifier'] == 0.90


# ===================================================================
#  RBI DRIFT TESTS (7)
# ===================================================================

class TestRBIDrift:
    """Tests for RBIDriftSignal."""

    def setup_method(self):
        self.sig = RBIDriftSignal()

    def test_mpc_decision_day(self):
        """MPC decision date fires with event_type=MPC_DECISION."""
        result = self.sig.evaluate({
            'date': date(2026, 2, 7),
            'price_at_915': 24100.0,
            'price_at_930': 24130.0,    # +0.124% drift up
            'prev_close': 24050.0,
            'india_vix': 16.5,
            'rbi_consensus': 'HOLD',
        })
        assert result is not None
        assert result['event_type'] == 'MPC_DECISION'
        assert result['direction'] == 'LONG'
        assert result['signal_id'] == 'RBI_DRIFT'

    def test_cut_bullish(self):
        """Consensus CUT + upward drift = LONG, not contrarian."""
        result = self.sig.evaluate({
            'date': date(2026, 4, 9),
            'price_at_915': 24200.0,
            'price_at_930': 24250.0,    # +0.207% drift up
            'prev_close': 24150.0,
            'india_vix': 17.0,
            'rbi_consensus': 'CUT',
        })
        assert result is not None
        assert result['direction'] == 'LONG'
        assert result['is_contrarian'] is False

    def test_hike_bearish(self):
        """Consensus HIKE + downward drift = SHORT, not contrarian."""
        result = self.sig.evaluate({
            'date': date(2026, 6, 5),
            'price_at_915': 24300.0,
            'price_at_930': 24240.0,    # -0.247% drift down
            'prev_close': 24350.0,
            'india_vix': 18.5,
            'rbi_consensus': 'HIKE',
        })
        assert result is not None
        assert result['direction'] == 'SHORT'
        assert result['is_contrarian'] is False

    def test_hold_bullish(self):
        """Consensus HOLD (default) + upward drift = LONG."""
        result = self.sig.evaluate({
            'date': date(2026, 8, 6),
            'price_at_915': 24500.0,
            'price_at_930': 24530.0,
            'prev_close': 24450.0,
            'india_vix': 15.0,
        })
        assert result is not None
        assert result['direction'] == 'LONG'
        assert result['rbi_consensus'] == 'HOLD'

    def test_contrarian(self):
        """Drift opposite to consensus = contrarian, lower confidence."""
        r_aligned = self.sig.evaluate({
            'date': date(2026, 2, 7),
            'price_at_915': 24100.0,
            'price_at_930': 24130.0,
            'prev_close': 24050.0,
            'india_vix': 16.0,
            'rbi_consensus': 'HOLD',     # expects LONG
        })
        r_contrarian = self.sig.evaluate({
            'date': date(2026, 2, 7),
            'price_at_915': 24100.0,
            'price_at_930': 24070.0,     # drift DOWN
            'prev_close': 24050.0,
            'india_vix': 16.0,
            'rbi_consensus': 'HOLD',     # expects LONG, but drift is SHORT
        })
        assert r_aligned is not None and r_contrarian is not None
        assert r_contrarian['is_contrarian'] is True
        assert r_contrarian['confidence'] < r_aligned['confidence']

    def test_non_rbi(self):
        """Non-RBI date returns None."""
        result = self.sig.evaluate({
            'date': date(2026, 3, 15),   # not an MPC date
            'price_at_915': 24100.0,
            'price_at_930': 24130.0,
            'prev_close': 24050.0,
            'india_vix': 16.0,
        })
        assert result is None

    def test_minutes_lower_conf(self):
        """RBI minutes day fires but with lower base confidence."""
        r_mpc = self.sig.evaluate({
            'date': date(2026, 2, 7),
            'price_at_915': 24100.0,
            'price_at_930': 24130.0,
            'prev_close': 24050.0,
            'india_vix': 16.0,
            'rbi_consensus': 'HOLD',
        })
        r_minutes = self.sig.evaluate({
            'date': date(2026, 2, 21),   # minutes date
            'price_at_915': 24100.0,
            'price_at_930': 24130.0,
            'prev_close': 24050.0,
            'india_vix': 16.0,
            'rbi_consensus': 'HOLD',
            'is_rbi_minutes_day': True,
        })
        assert r_mpc is not None and r_minutes is not None
        assert r_minutes['event_type'] == 'MPC_MINUTES'
        assert r_minutes['confidence'] < r_mpc['confidence']
