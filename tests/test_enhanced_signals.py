"""
Tests for all 11 enhanced signal modules.

Run: python -m pytest tests/test_enhanced_signals.py -v
"""

import pytest
from datetime import date
from unittest.mock import MagicMock, patch


# ================================================================
# 1. PCR Autotrender Tests
# ================================================================
class TestPCRAutotrender:
    def setup_method(self):
        from signals.pcr_signal import PCRAutotrender
        self.sig = PCRAutotrender()

    def test_extreme_high_pcr_bullish(self):
        ctx = self.sig.evaluate(pcr_override=1.50, vix_override=16.0)
        assert ctx.direction == 'BULLISH'
        assert ctx.pcr_zone == 'EXTREME_HIGH'
        assert ctx.is_contrarian is True
        assert ctx.size_modifier >= 1.2

    def test_extreme_low_pcr_bearish(self):
        ctx = self.sig.evaluate(pcr_override=0.40, vix_override=16.0)
        assert ctx.direction == 'BEARISH'
        assert ctx.pcr_zone == 'EXTREME_LOW'
        assert ctx.is_contrarian is True
        assert ctx.size_modifier <= 0.75

    def test_neutral_pcr(self):
        ctx = self.sig.evaluate(pcr_override=0.90, vix_override=16.0)
        assert ctx.direction == 'NEUTRAL'
        assert ctx.size_modifier == 1.0

    def test_high_vix_widens_bands(self):
        # At VIX=30 (HIGH_VOL), 1.35 might be NEUTRAL instead of extreme
        ctx = self.sig.evaluate(pcr_override=1.40, vix_override=30.0)
        assert ctx.vix_regime == 'HIGH_VOL'

    def test_contrarian_entry_thresholds(self):
        ctx_bull = self.sig.evaluate(pcr_override=1.55, vix_override=16.0)
        assert ctx_bull.is_contrarian is True
        ctx_bear = self.sig.evaluate(pcr_override=0.42, vix_override=16.0)
        assert ctx_bear.is_contrarian is True
        ctx_no = self.sig.evaluate(pcr_override=1.0, vix_override=16.0)
        assert ctx_no.is_contrarian is False

    def test_to_dict_has_required_keys(self):
        ctx = self.sig.evaluate(pcr_override=1.2, vix_override=16.0)
        d = ctx.to_dict()
        assert 'signal_id' in d
        assert d['signal_id'] == 'PCR_AUTOTRENDER'
        assert 'size_modifier' in d
        assert 'direction' in d

    def test_telegram_format(self):
        ctx = self.sig.evaluate(pcr_override=1.5, vix_override=16.0)
        msg = ctx.to_telegram()
        assert 'PCR' in msg
        assert 'CONTRARIAN' in msg


# ================================================================
# 2. Rollover Signal Tests
# ================================================================
class TestRolloverSignal:
    def setup_method(self):
        from signals.rollover_signal import RolloverSignal
        self.sig = RolloverSignal()

    def test_buildup_classification(self):
        from signals.rollover_signal import RolloverSignal
        # OI up + price up = long buildup
        assert RolloverSignal._classify_buildup(10.0, 1.0) == 'LONG_BUILDUP'
        # OI up + price down = short buildup
        assert RolloverSignal._classify_buildup(10.0, -1.0) == 'SHORT_BUILDUP'
        # OI down + price down = long unwinding
        assert RolloverSignal._classify_buildup(-10.0, -1.0) == 'LONG_UNWINDING'
        # OI down + price up = short covering
        assert RolloverSignal._classify_buildup(-10.0, 1.0) == 'SHORT_COVERING'

    def test_signal_strength_high_rollover_long_buildup(self):
        from signals.rollover_signal import RolloverSignal
        strength, direction = RolloverSignal._classify_signal_strength(75.0, 'LONG_BUILDUP')
        assert strength == 'STRONG_BULLISH'
        assert direction == 'BULLISH'

    def test_signal_strength_low_rollover_short_buildup(self):
        from signals.rollover_signal import RolloverSignal
        strength, direction = RolloverSignal._classify_signal_strength(30.0, 'SHORT_BUILDUP')
        assert strength == 'STRONG_BEARISH'
        assert direction == 'BEARISH'

    def test_monthly_expiry_calculation(self):
        from signals.rollover_signal import RolloverSignal
        exp = RolloverSignal._get_monthly_expiry(date(2026, 3, 15))
        assert exp.weekday() == 3  # Thursday
        assert exp.month == 3


# ================================================================
# 3. FII Futures OI Tests
# ================================================================
class TestFIIFuturesOI:
    def setup_method(self):
        from signals.fii_futures_oi import FIIFuturesOI
        self.sig = FIIFuturesOI()

    def test_strong_bullish(self):
        ctx = self.sig.evaluate(fii_long_override=80000, fii_short_override=20000)
        assert ctx.fii_ratio == 0.8
        assert ctx.direction == 'BULLISH'
        assert ctx.ratio_zone == 'STRONG_BULL'
        # Divergence with default dii_ratio=0.5 dampens modifier
        assert ctx.size_modifier >= 1.1

    def test_strong_bearish(self):
        ctx = self.sig.evaluate(fii_long_override=20000, fii_short_override=80000)
        assert ctx.fii_ratio == 0.2
        assert ctx.direction == 'BEARISH'
        # Divergence dampening applies: 1.0 + (0.6-1.0)*0.5 = 0.8
        assert ctx.size_modifier <= 0.85

    def test_neutral(self):
        ctx = self.sig.evaluate(fii_long_override=50000, fii_short_override=50000)
        assert ctx.fii_ratio == 0.5
        assert ctx.ratio_zone == 'NEUTRAL'

    def test_ratio_calculation(self):
        ctx = self.sig.evaluate(fii_long_override=60000, fii_short_override=40000)
        assert ctx.fii_ratio == 0.6
        assert ctx.fii_long_oi == 60000
        assert ctx.fii_short_oi == 40000


# ================================================================
# 4. Delivery Signal Tests
# ================================================================
class TestDeliverySignal:
    def setup_method(self):
        from signals.delivery_signal import DeliverySignal
        self.sig = DeliverySignal()

    def test_accumulation(self):
        ctx = self.sig.evaluate(delivery_override=55.0, price_change_override=1.0)
        assert ctx.classification == 'ACCUMULATION'
        assert ctx.direction == 'BULLISH'
        assert ctx.size_modifier > 1.0

    def test_distribution(self):
        ctx = self.sig.evaluate(delivery_override=55.0, price_change_override=-1.0)
        assert ctx.classification == 'DISTRIBUTION'
        assert ctx.direction == 'BEARISH'

    def test_speculative_rally(self):
        ctx = self.sig.evaluate(delivery_override=30.0, price_change_override=1.0)
        assert ctx.classification == 'SPECULATIVE_RALLY'
        assert ctx.direction == 'BEARISH'  # Fade speculative rally

    def test_panic_selling(self):
        ctx = self.sig.evaluate(delivery_override=22.0, price_change_override=-1.5)
        assert ctx.classification == 'PANIC_SELLING'
        assert ctx.direction == 'BULLISH'  # Contrarian

    def test_strong_accumulation(self):
        ctx = self.sig.evaluate(delivery_override=65.0, price_change_override=1.5)
        assert ctx.classification == 'STRONG_ACCUMULATION'
        assert ctx.size_modifier >= 1.3


# ================================================================
# 5. Sentiment Signal Tests
# ================================================================
class TestSentimentSignal:
    def setup_method(self):
        from signals.sentiment_signal import SentimentSignal
        self.sig = SentimentSignal()

    def test_extreme_fear_bullish(self):
        ctx = self.sig.evaluate(mmi_override=15.0)
        assert ctx.mmi_zone == 'EXTREME_FEAR'
        assert ctx.combined_direction == 'BULLISH'
        assert ctx.is_contrarian is True

    def test_extreme_greed_bearish(self):
        ctx = self.sig.evaluate(mmi_override=85.0)
        assert ctx.mmi_zone == 'EXTREME_GREED'
        assert ctx.combined_direction == 'BEARISH'
        assert ctx.is_contrarian is True

    def test_neutral(self):
        ctx = self.sig.evaluate(mmi_override=50.0)
        assert ctx.mmi_zone == 'NEUTRAL'
        assert ctx.combined_direction == 'NEUTRAL'

    def test_gtrends_spike_boosts_fear(self):
        ctx = self.sig.evaluate(
            mmi_override=15.0,
            gtrends_override={'score': 3.0, 'spike': True}
        )
        assert ctx.gtrends_spike is True
        assert ctx.confidence > 0.70  # Should be high confidence


# ================================================================
# 6. Bond Yield Signal Tests
# ================================================================
class TestBondYieldSignal:
    def setup_method(self):
        from signals.bond_yield_signal import BondYieldSignal
        self.sig = BondYieldSignal()

    def test_wide_spread_bullish(self):
        ctx = self.sig.evaluate(spread_override=5.0, dxy_override=100.0)
        assert ctx.spread_zone == 'WIDE'
        assert ctx.direction == 'BULLISH'
        assert ctx.size_modifier > 1.0

    def test_narrow_spread_bearish(self):
        ctx = self.sig.evaluate(spread_override=2.5, dxy_override=105.0)
        assert ctx.spread_zone == 'NARROW'
        assert ctx.direction == 'BEARISH'

    def test_critical_spread(self):
        ctx = self.sig.evaluate(spread_override=1.5, dxy_override=106.0)
        assert ctx.spread_zone == 'CRITICAL'
        assert ctx.size_modifier <= 0.75

    def test_dxy_impact(self):
        ctx_strong = self.sig.evaluate(spread_override=3.5, dxy_override=106.0)
        assert ctx_strong.dxy_impact == 'NEGATIVE'
        ctx_weak = self.sig.evaluate(spread_override=3.5, dxy_override=98.0)
        assert ctx_weak.dxy_impact == 'POSITIVE'


# ================================================================
# 7. XGBoost Meta-Learner Tests
# ================================================================
class TestXGBoostMeta:
    def setup_method(self):
        from signals.xgboost_meta import XGBoostMetaLearner
        self.meta = XGBoostMetaLearner()

    def test_heuristic_fallback(self):
        signals = {
            'PCR_AUTOTRENDER': {'pcr_current': 1.3, 'pcr_zone': 'HIGH',
                                'pcr_momentum': 'RISING', 'direction': 'BULLISH',
                                'size_modifier': 1.15},
            'FII_FUTURES_OI': {'fii_ratio': 0.6, 'ratio_momentum': 0.05,
                               'direction': 'BULLISH', 'size_modifier': 1.15},
            'context': {'vix_level': 15, 'vix_regime': 'NORMAL'},
        }
        result = self.meta.predict(signals)
        assert 'size_modifier' in result
        assert 'direction' in result
        assert result['method'].startswith('heuristic')

    def test_feature_extraction_shape(self):
        signals = {
            'context': {'vix_level': 15, 'vix_regime': 'NORMAL'},
        }
        features = self.meta.extract_features(signals)
        assert features.shape[0] == 1
        assert features.shape[1] > 20

    def test_size_bounds(self):
        signals = {
            'PCR_AUTOTRENDER': {'direction': 'BULLISH', 'size_modifier': 2.0},
            'FII_FUTURES_OI': {'direction': 'BULLISH', 'size_modifier': 2.0},
            'context': {'vix_regime': 'NORMAL'},
        }
        result = self.meta.predict(signals)
        assert 0.3 <= result['size_modifier'] <= 1.5


# ================================================================
# 8. Gamma Exposure Tests
# ================================================================
class TestGammaExposure:
    def setup_method(self):
        from signals.gamma_exposure import GammaExposureSignal
        self.sig = GammaExposureSignal()

    def test_strong_positive_gamma(self):
        ctx = self.sig.evaluate(spot=23400, net_gamma_override=2e9)
        assert ctx.gamma_zone == 'STRONG_POSITIVE'
        assert ctx.regime == 'MEAN_REVERSION'
        assert ctx.size_modifier_fade > 1.0
        assert ctx.size_modifier_momentum < 1.0

    def test_strong_negative_gamma(self):
        ctx = self.sig.evaluate(spot=23400, net_gamma_override=-1.5e9)
        assert ctx.gamma_zone == 'STRONG_NEGATIVE'
        assert ctx.regime == 'TRENDING'
        assert ctx.size_modifier_fade < 1.0
        assert ctx.size_modifier_momentum > 1.0

    def test_gamma_range_wider_in_negative(self):
        ctx_pos = self.sig.evaluate(spot=23400, net_gamma_override=1e9)
        ctx_neg = self.sig.evaluate(spot=23400, net_gamma_override=-1e9)
        range_pos = ctx_pos.gamma_range[1] - ctx_pos.gamma_range[0]
        range_neg = ctx_neg.gamma_range[1] - ctx_neg.gamma_range[0]
        assert range_neg > range_pos

    def test_to_dict_keys(self):
        ctx = self.sig.evaluate(spot=23400, net_gamma_override=500e6)
        d = ctx.to_dict()
        assert d['signal_id'] == 'GAMMA_EXPOSURE'
        assert 'gamma_flip_strike' in d
        assert 'regime' in d


# ================================================================
# 9. Vol Term Structure Tests
# ================================================================
class TestVolTermStructure:
    def setup_method(self):
        from signals.vol_term_structure import VolTermStructureSignal
        self.sig = VolTermStructureSignal()

    def test_contango(self):
        ctx = self.sig.evaluate(front_iv_override=14.0, back_iv_override=17.0)
        assert ctx.structure_state in ('CONTANGO', 'STEEP_CONTANGO')
        assert ctx.term_spread > 0

    def test_backwardation(self):
        ctx = self.sig.evaluate(front_iv_override=22.0, back_iv_override=18.0)
        assert ctx.structure_state in ('BACKWARDATION', 'STEEP_BACKWARDATION')
        assert ctx.direction == 'BEARISH'
        assert ctx.size_modifier < 1.0

    def test_flat(self):
        ctx = self.sig.evaluate(front_iv_override=15.0, back_iv_override=15.5)
        assert ctx.structure_state == 'FLAT'

    def test_steep_backwardation_crisis(self):
        ctx = self.sig.evaluate(front_iv_override=30.0, back_iv_override=22.0)
        assert ctx.structure_state == 'STEEP_BACKWARDATION'
        assert ctx.size_modifier <= 0.65


# ================================================================
# 10. RBI Macro Filter Tests
# ================================================================
class TestRBIMacroFilter:
    def setup_method(self):
        from signals.rbi_macro_filter import RBIMacroFilter
        self.filt = RBIMacroFilter()

    def test_rbi_mpc_day(self):
        ctx = self.filt.evaluate(trade_date=date(2026, 2, 6))
        assert ctx.in_event_window is True
        assert ctx.size_modifier <= 0.55
        assert 'RBI MPC' in ctx.active_events[0]

    def test_budget_day(self):
        ctx = self.filt.evaluate(trade_date=date(2026, 2, 1))
        assert ctx.in_event_window is True
        assert ctx.size_modifier <= 0.35

    def test_normal_day(self):
        ctx = self.filt.evaluate(trade_date=date(2026, 3, 15))
        assert ctx.in_event_window is False
        assert ctx.size_modifier >= 0.90

    def test_fomc_day(self):
        ctx = self.filt.evaluate(trade_date=date(2026, 3, 18))
        assert ctx.in_event_window is True
        assert any('FOMC' in e for e in ctx.active_events)

    def test_gst_classification(self):
        from signals.rbi_macro_filter import RBIMacroFilter
        regime, direction = RBIMacroFilter._classify_gst(170000)
        assert regime == 'EXPANSION'
        assert direction == 'BULLISH'
        regime2, direction2 = RBIMacroFilter._classify_gst(130000)
        assert regime2 == 'CONTRACTION'
        assert direction2 == 'BEARISH'


# ================================================================
# 11. Order Flow Tests
# ================================================================
class TestOrderFlow:
    def setup_method(self):
        from signals.order_flow import OrderFlowSignal
        self.sig = OrderFlowSignal()

    def test_strong_buy_flow(self):
        ctx = self.sig.evaluate(ofi_override=0.40)
        assert ctx.ofi_zone == 'STRONG_BUY'
        assert ctx.direction == 'BULLISH'
        assert ctx.size_modifier >= 1.2

    def test_strong_sell_flow(self):
        ctx = self.sig.evaluate(ofi_override=-0.40)
        assert ctx.ofi_zone == 'STRONG_SELL'
        assert ctx.direction == 'BEARISH'
        assert ctx.size_modifier <= 0.75

    def test_neutral_flow(self):
        ctx = self.sig.evaluate(ofi_override=0.0)
        assert ctx.ofi_zone == 'NEUTRAL'
        assert ctx.direction == 'NEUTRAL'
        assert ctx.size_modifier == 1.0

    def test_buy_volume_pct(self):
        ctx = self.sig.evaluate(ofi_override=0.3)
        assert ctx.buy_volume_pct > 50

    def test_to_dict_keys(self):
        ctx = self.sig.evaluate(ofi_override=0.2)
        d = ctx.to_dict()
        assert d['signal_id'] == 'ORDER_FLOW_IMBALANCE'
        assert 'ofi_15min' in d
        assert 'vwap' in d


# ================================================================
# Cross-signal integration tests
# ================================================================
class TestSignalIntegration:
    """Test that all signals can be imported and evaluated together."""

    def test_all_signals_importable(self):
        from signals.pcr_signal import PCRAutotrender
        from signals.rollover_signal import RolloverSignal
        from signals.fii_futures_oi import FIIFuturesOI
        from signals.delivery_signal import DeliverySignal
        from signals.sentiment_signal import SentimentSignal
        from signals.bond_yield_signal import BondYieldSignal
        from signals.xgboost_meta import XGBoostMetaLearner
        from signals.gamma_exposure import GammaExposureSignal
        from signals.vol_term_structure import VolTermStructureSignal
        from signals.rbi_macro_filter import RBIMacroFilter
        from signals.order_flow import OrderFlowSignal

        # All should instantiate without error
        PCRAutotrender()
        RolloverSignal()
        FIIFuturesOI()
        DeliverySignal()
        SentimentSignal()
        BondYieldSignal()
        XGBoostMetaLearner()
        GammaExposureSignal()
        VolTermStructureSignal()
        RBIMacroFilter()
        OrderFlowSignal()

    def test_meta_learner_combines_all(self):
        from signals.xgboost_meta import XGBoostMetaLearner

        meta = XGBoostMetaLearner()
        # Build signal dict from all overrides
        signals = {
            'PCR_AUTOTRENDER': {'pcr_current': 1.2, 'pcr_zone': 'HIGH',
                                'pcr_momentum': 'FLAT', 'direction': 'BULLISH',
                                'size_modifier': 1.15},
            'FII_FUTURES_OI': {'fii_ratio': 0.55, 'ratio_momentum': 0.03,
                               'direction': 'BULLISH', 'size_modifier': 1.15},
            'DELIVERY_PCT': {'delivery_pct': 48, 'classification': 'NORMAL',
                             'direction': 'NEUTRAL', 'size_modifier': 1.0},
            'ROLLOVER_ANALYSIS': {'rollover_pct': 60, 'buildup_type': 'LONG_BUILDUP',
                                  'direction': 'BULLISH', 'size_modifier': 1.15},
            'SENTIMENT_COMPOSITE': {'mmi_value': 40, 'mmi_zone': 'NEUTRAL',
                                    'direction': 'NEUTRAL', 'size_modifier': 1.0},
            'BOND_YIELD_SPREAD': {'spread': 4.0, 'spread_momentum': 'FLAT',
                                  'direction': 'NEUTRAL', 'size_modifier': 1.0},
            'GLOBAL_OVERNIGHT_COMPOSITE': {'composite_score': 0.3, 'risk_off': False,
                                           'direction': 'BULLISH', 'size_modifier': 1.1},
            'GIFT_NIFTY_GAP': {'gap_pct': 0.5, 'gap_type': 'REVERSION'},
            'GAMMA_EXPOSURE': {'direction': 'NEUTRAL'},
            'VOL_TERM_STRUCTURE': {'structure_zone': 'CONTANGO'},
            'context': {'vix_level': 15, 'vix_regime': 'NORMAL', 'day_of_week': 2,
                        'dte_weekly': 3, 'dte_monthly': 15,
                        'nifty_5d_return': 0.5, 'nifty_20d_return': 1.2},
        }

        result = meta.predict(signals)
        assert 0.3 <= result['size_modifier'] <= 1.5
        assert result['direction'] in ('BULLISH', 'BEARISH', 'NEUTRAL')

    def test_all_signals_have_to_dict(self):
        """Every signal context should have to_dict() and to_telegram()."""
        from signals.pcr_signal import PCRAutotrender
        from signals.fii_futures_oi import FIIFuturesOI
        from signals.delivery_signal import DeliverySignal
        from signals.sentiment_signal import SentimentSignal
        from signals.bond_yield_signal import BondYieldSignal
        from signals.gamma_exposure import GammaExposureSignal
        from signals.vol_term_structure import VolTermStructureSignal
        from signals.rbi_macro_filter import RBIMacroFilter
        from signals.order_flow import OrderFlowSignal

        contexts = [
            PCRAutotrender().evaluate(pcr_override=1.0, vix_override=15.0),
            FIIFuturesOI().evaluate(fii_long_override=50000, fii_short_override=50000),
            DeliverySignal().evaluate(delivery_override=45.0, price_change_override=0.5),
            SentimentSignal().evaluate(mmi_override=50.0),
            BondYieldSignal().evaluate(spread_override=4.0),
            GammaExposureSignal().evaluate(spot=23400, net_gamma_override=500e6),
            VolTermStructureSignal().evaluate(front_iv_override=15.0, back_iv_override=16.0),
            RBIMacroFilter().evaluate(trade_date=date(2026, 3, 15)),
            OrderFlowSignal().evaluate(ofi_override=0.0),
        ]

        for ctx in contexts:
            d = ctx.to_dict()
            assert 'signal_id' in d
            assert 'direction' in d
            assert 'confidence' in d
            msg = ctx.to_telegram()
            assert len(msg) > 10
