"""
Tests for Tier-3 structural signals:
    - VIXTransmissionSignal  (8 tests)
    - RBIInterventionSignal  (8 tests)
    - PreOpenAuctionSignal   (8 tests)

Total: 24 tests.
"""

import pytest

from signals.structural.vix_transmission import VIXTransmissionSignal
from signals.structural.rbi_intervention import RBIInterventionSignal
from signals.structural.preopen_auction import PreOpenAuctionSignal


# ===================================================================
#  VIX TRANSMISSION TESTS (8)
# ===================================================================

class TestVIXTransmission:
    """Tests for VIXTransmissionSignal."""

    def setup_method(self):
        self.sig = VIXTransmissionSignal()

    def test_extreme_spike(self):
        """US VIX +30% -> size 0.70, bias BEARISH."""
        result = self.sig.evaluate({
            'us_vix_current': 26.0,
            'us_vix_prev': 20.0,   # +30%
            'india_vix_current': 18.0,
            'india_vix_prev': 15.0,
        })
        assert result['spike_category'] == 'EXTREME_SPIKE'
        assert result['size_modifier'] == 0.70
        assert result['bias'] == 'BEARISH'
        assert result['signal_id'] == 'VIX_TRANSMISSION'

    def test_major_spike(self):
        """US VIX +18% -> size 0.85."""
        result = self.sig.evaluate({
            'us_vix_current': 23.6,
            'us_vix_prev': 20.0,   # +18%
            'india_vix_current': 16.0,
            'india_vix_prev': 14.5,
        })
        assert result['spike_category'] == 'MAJOR_SPIKE'
        assert result['size_modifier'] == 0.85
        assert result['bias'] == 'BEARISH'

    def test_vix_crush(self):
        """US VIX -15% -> size 1.10, bias BULLISH."""
        result = self.sig.evaluate({
            'us_vix_current': 17.0,
            'us_vix_prev': 20.0,   # -15%
            'india_vix_current': 13.5,
            'india_vix_prev': 15.0,
        })
        assert result['spike_category'] == 'VIX_CRUSH'
        assert result['size_modifier'] == 1.10
        assert result['bias'] == 'BULLISH'

    def test_normal(self):
        """US VIX +3% -> size 1.0, bias NEUTRAL."""
        result = self.sig.evaluate({
            'us_vix_current': 20.6,
            'us_vix_prev': 20.0,   # +3%
            'india_vix_current': 15.2,
            'india_vix_prev': 15.0,
        })
        assert result['spike_category'] == 'NORMAL'
        assert result['size_modifier'] == 1.0
        assert result['bias'] == 'NEUTRAL'

    def test_size_modifier_defensive(self):
        """Moderate spike should have defensive but not extreme modifier."""
        result = self.sig.evaluate({
            'us_vix_current': 22.4,
            'us_vix_prev': 20.0,   # +12%
            'india_vix_current': 16.0,
            'india_vix_prev': 15.0,
        })
        assert result['spike_category'] == 'MODERATE_SPIKE'
        assert result['size_modifier'] == 0.95
        # Moderate is more defensive than normal but less than major
        assert 0.85 < result['size_modifier'] < 1.0

    def test_transmission_complete(self):
        """India VIX moved >= 60% of expected -> transmission_complete True."""
        # US VIX +20% -> expected India move = 20 * 0.65 = 13%
        # India moved from 15 to 17.1 = +14% -> 14/13 > 60% -> complete
        result = self.sig.evaluate({
            'us_vix_current': 24.0,
            'us_vix_prev': 20.0,   # +20%
            'india_vix_current': 17.1,
            'india_vix_prev': 15.0,
        })
        assert result['transmission_complete'] is True
        assert result['spike_category'] == 'MAJOR_SPIKE'

    def test_bias_bearish_on_spike(self):
        """Any spike category (EXTREME/MAJOR/MODERATE/ELEVATED) -> BEARISH."""
        for us_current, expected_cat in [
            (26.0, 'EXTREME_SPIKE'),   # +30%
            (23.6, 'MAJOR_SPIKE'),     # +18%
            (22.4, 'MODERATE_SPIKE'),  # +12%
            (21.2, 'ELEVATED'),        # +6%
        ]:
            result = self.sig.evaluate({
                'us_vix_current': us_current,
                'us_vix_prev': 20.0,
                'india_vix_current': 16.0,
                'india_vix_prev': 15.0,
            })
            assert result['spike_category'] == expected_cat, (
                f'Expected {expected_cat} for us_current={us_current}'
            )
            assert result['bias'] == 'BEARISH', (
                f'Expected BEARISH for {expected_cat}'
            )

    def test_missing_data_neutral(self):
        """Missing or None inputs -> neutral context with size_modifier=1.0."""
        # All None
        result = self.sig.evaluate({})
        assert result['bias'] == 'NEUTRAL'
        assert result['size_modifier'] == 1.0

        # Partial None
        result = self.sig.evaluate({
            'us_vix_current': 20.0,
            'us_vix_prev': None,
            'india_vix_current': 15.0,
            'india_vix_prev': 14.0,
        })
        assert result['bias'] == 'NEUTRAL'
        assert result['size_modifier'] == 1.0

        # NaN value
        result = self.sig.evaluate({
            'us_vix_current': float('nan'),
            'us_vix_prev': 20.0,
            'india_vix_current': 15.0,
            'india_vix_prev': 14.0,
        })
        assert result['bias'] == 'NEUTRAL'
        assert result['size_modifier'] == 1.0


# ===================================================================
#  RBI INTERVENTION TESTS (8)
# ===================================================================

class TestRBIIntervention:
    """Tests for RBIInterventionSignal."""

    def setup_method(self):
        self.sig = RBIInterventionSignal()

    def test_heavy_intervention(self):
        """Touches high, reverses > 0.2% -> HEAVY_INTERVENTION."""
        result = self.sig.evaluate({
            'usdinr_spot': 84.70,
            'usdinr_prev': 84.90,
            'usdinr_high': 85.10,   # reversal from high = (85.10-84.70)/84.90 = 0.47%
            'usdinr_low': 84.65,
        })
        assert result['intervention_category'] == 'HEAVY_INTERVENTION'
        assert result['signal_id'] == 'RBI_INTERVENTION'

    def test_defense_mode_round_number(self):
        """USDINR at 85.05 -> near round 85 -> DEFENSE_MODE."""
        result = self.sig.evaluate({
            'usdinr_spot': 85.05,
            'usdinr_prev': 85.00,
            'usdinr_high': 85.10,
            'usdinr_low': 84.95,
        })
        assert result['intervention_category'] == 'DEFENSE_MODE'
        assert result['at_round_number'] is True
        assert result['nearest_round_number'] == 85.0

    def test_accumulation_bullish(self):
        """Reserve increase -> ACCUMULATION -> size 1.05."""
        result = self.sig.evaluate({
            'usdinr_spot': 83.50,
            'usdinr_prev': 83.55,
            'usdinr_high': 83.60,
            'usdinr_low': 83.45,
            'reserve_change_usd_bn': 2.0,
        })
        assert result['reserve_category'] == 'ACCUMULATION'
        assert result['size_modifier'] == 1.05
        assert result['bias'] == 'INR_BULLISH'

    def test_no_intervention_neutral(self):
        """Normal range, not near round number -> NO_INTERVENTION."""
        result = self.sig.evaluate({
            'usdinr_spot': 83.50,
            'usdinr_prev': 83.48,
            'usdinr_high': 83.55,
            'usdinr_low': 83.45,
        })
        assert result['intervention_category'] == 'NO_INTERVENTION'
        assert result['bias'] == 'NEUTRAL'

    def test_size_modifier_stress(self):
        """HEAVY_DEPLOY + DEFENSE_MODE -> 0.80."""
        result = self.sig.evaluate({
            'usdinr_spot': 85.05,
            'usdinr_prev': 85.00,
            'usdinr_high': 85.10,
            'usdinr_low': 84.95,
            'reserve_change_usd_bn': -4.0,  # heavy deploy
        })
        assert result['reserve_category'] == 'HEAVY_DEPLOY'
        assert result['intervention_category'] == 'DEFENSE_MODE'
        assert result['size_modifier'] == 0.80

    def test_reserve_change_classification(self):
        """Reserve change categories are correctly classified."""
        sig = self.sig

        # Heavy deploy
        r = sig.evaluate({
            'usdinr_spot': 83.50, 'usdinr_prev': 83.48,
            'usdinr_high': 83.55, 'usdinr_low': 83.45,
            'reserve_change_usd_bn': -4.5,
        })
        assert r['reserve_category'] == 'HEAVY_DEPLOY'

        # Moderate deploy
        r = sig.evaluate({
            'usdinr_spot': 83.50, 'usdinr_prev': 83.48,
            'usdinr_high': 83.55, 'usdinr_low': 83.45,
            'reserve_change_usd_bn': -2.0,
        })
        assert r['reserve_category'] == 'MODERATE_DEPLOY'

        # Light deploy
        r = sig.evaluate({
            'usdinr_spot': 83.50, 'usdinr_prev': 83.48,
            'usdinr_high': 83.55, 'usdinr_low': 83.45,
            'reserve_change_usd_bn': -0.5,
        })
        assert r['reserve_category'] == 'LIGHT_DEPLOY'

        # Accumulation
        r = sig.evaluate({
            'usdinr_spot': 83.50, 'usdinr_prev': 83.48,
            'usdinr_high': 83.55, 'usdinr_low': 83.45,
            'reserve_change_usd_bn': 1.5,
        })
        assert r['reserve_category'] == 'ACCUMULATION'

    def test_round_number_detection(self):
        """Round number detection: 83.1, 84.0, 85.15 True; 83.5 False."""
        from signals.structural.rbi_intervention import _at_round_number

        assert _at_round_number(83.1) is True    # within 0.15 of 83
        assert _at_round_number(84.0) is True    # exactly 84
        assert _at_round_number(85.14) is True   # within 0.15 of 85
        assert _at_round_number(83.5) is False   # 0.5 away from 83 and 84

    def test_missing_reserve_data(self):
        """Missing reserve data -> UNKNOWN category, size unaffected."""
        result = self.sig.evaluate({
            'usdinr_spot': 83.50,
            'usdinr_prev': 83.48,
            'usdinr_high': 83.55,
            'usdinr_low': 83.45,
            # no reserve_change_usd_bn
        })
        assert result['reserve_category'] == 'UNKNOWN'
        assert result['size_modifier'] == 1.0  # NO_INTERVENTION + UNKNOWN


# ===================================================================
#  PREOPEN AUCTION TESTS (8)
# ===================================================================

class TestPreOpenAuction:
    """Tests for PreOpenAuctionSignal."""

    def setup_method(self):
        self.sig = PreOpenAuctionSignal()

    def test_strong_buy(self):
        """Domestic adjustment > 0.15% -> STRONG_BUY."""
        # auction=22540, gift=22500, prev=22500
        # domestic_adj = (22540-22500)/22500*100 = 0.1778%
        result = self.sig.evaluate({
            'auction_equilibrium': 22540,
            'prev_close': 22500,
            'gift_implied_open': 22500,
            'auction_volume': 1000000,
            'avg_auction_volume': 800000,
        })
        assert result['signal_classification'] == 'STRONG_BUY'
        assert result['bias'] == 'BULLISH'
        assert result['size_modifier'] in (1.05, 1.10)

    def test_strong_sell(self):
        """Domestic adjustment < -0.15% -> STRONG_SELL."""
        # auction=22460, gift=22500, prev=22500
        # domestic_adj = (22460-22500)/22500*100 = -0.1778%
        result = self.sig.evaluate({
            'auction_equilibrium': 22460,
            'prev_close': 22500,
            'gift_implied_open': 22500,
            'auction_volume': 1000000,
            'avg_auction_volume': 800000,
        })
        assert result['signal_classification'] == 'STRONG_SELL'
        assert result['bias'] == 'BEARISH'
        assert result['size_modifier'] in (0.90, 0.95)

    def test_neutral(self):
        """Adjustment near zero -> NEUTRAL."""
        # auction=22502, gift=22500, prev=22500
        # domestic_adj = (22502-22500)/22500*100 = 0.0089%
        result = self.sig.evaluate({
            'auction_equilibrium': 22502,
            'prev_close': 22500,
            'gift_implied_open': 22500,
            'auction_volume': 900000,
            'avg_auction_volume': 800000,
        })
        assert result['signal_classification'] == 'NEUTRAL'
        assert result['size_modifier'] == 1.0
        assert result['bias'] == 'NEUTRAL'

    def test_domestic_adjustment_calc(self):
        """Verify domestic_adjustment_pct is computed correctly."""
        result = self.sig.evaluate({
            'auction_equilibrium': 22550,
            'prev_close': 22500,
            'gift_implied_open': 22530,
            'auction_volume': 900000,
            'avg_auction_volume': 800000,
        })
        # domestic_adj = (22550 - 22530) / 22500 * 100 = 0.0889%
        assert result['domestic_adjustment_pct'] is not None
        assert abs(result['domestic_adjustment_pct'] - 0.0889) < 0.01
        assert result['signal_classification'] == 'MILD_BUY'

    def test_low_volume_overrides_to_neutral(self):
        """LOW volume overrides any classification to NEUTRAL."""
        # Without low volume this would be STRONG_BUY
        result = self.sig.evaluate({
            'auction_equilibrium': 22540,
            'prev_close': 22500,
            'gift_implied_open': 22500,
            'auction_volume': 400000,       # low
            'avg_auction_volume': 800000,   # ratio = 0.5 < 0.7
        })
        assert result['signal_classification'] == 'NEUTRAL'
        assert result['size_modifier'] == 1.0

    def test_high_volume_amplifies(self):
        """HIGH volume with STRONG_BUY -> size 1.10."""
        result = self.sig.evaluate({
            'auction_equilibrium': 22540,
            'prev_close': 22500,
            'gift_implied_open': 22500,
            'auction_volume': 1500000,      # high
            'avg_auction_volume': 800000,   # ratio = 1.875 > 1.5
        })
        assert result['signal_classification'] == 'STRONG_BUY'
        assert result['volume_classification'] == 'HIGH'
        assert result['size_modifier'] == 1.10

    def test_size_modifier_range(self):
        """All size modifiers must be between 0.90 and 1.10."""
        test_cases = [
            # Strong buy + high volume
            {'auction_equilibrium': 22540, 'prev_close': 22500,
             'gift_implied_open': 22500, 'auction_volume': 1500000,
             'avg_auction_volume': 800000},
            # Strong sell + high volume
            {'auction_equilibrium': 22460, 'prev_close': 22500,
             'gift_implied_open': 22500, 'auction_volume': 1500000,
             'avg_auction_volume': 800000},
            # Neutral
            {'auction_equilibrium': 22501, 'prev_close': 22500,
             'gift_implied_open': 22500, 'auction_volume': 900000,
             'avg_auction_volume': 800000},
            # Mild buy
            {'auction_equilibrium': 22520, 'prev_close': 22500,
             'gift_implied_open': 22510, 'auction_volume': 900000,
             'avg_auction_volume': 800000},
            # Low volume override
            {'auction_equilibrium': 22540, 'prev_close': 22500,
             'gift_implied_open': 22500, 'auction_volume': 400000,
             'avg_auction_volume': 800000},
        ]

        for md in test_cases:
            result = self.sig.evaluate(md)
            assert 0.90 <= result['size_modifier'] <= 1.10, (
                f"size_modifier {result['size_modifier']} out of range "
                f"for {result['signal_classification']}"
            )

    def test_missing_gift_data(self):
        """Missing GIFT data -> uses preopen_deviation_pct for classification."""
        # auction=22540, prev=22500 -> deviation = 0.1778%
        result = self.sig.evaluate({
            'auction_equilibrium': 22540,
            'prev_close': 22500,
            # no gift_implied_open
            'auction_volume': 1000000,
            'avg_auction_volume': 800000,
        })
        assert result['domestic_adjustment_pct'] is None
        assert result['preopen_deviation_pct'] > 0.15
        assert result['signal_classification'] == 'STRONG_BUY'
        assert result['signal_id'] == 'PREOPEN_AUCTION'
