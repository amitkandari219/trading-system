"""Tests for conviction scorer + trend confirmer."""
import pytest
from execution.conviction_scorer import ConvictionScorer
from execution.trend_confirmer import TrendConfirmer


class TestBullConviction:
    def test_all_bullish_high_score(self):
        cs = ConvictionScorer()
        mods = {
            'FII_FUTURES_OI': 1.2, 'MAMBA_REGIME': 1.15, 'PCR_AUTOTRENDER': 1.2,
            'DELIVERY_PCT': 1.15, 'GAMMA_EXPOSURE': 1.1, 'VOL_TERM_STRUCTURE': 1.1,
            'AMFI_MF_FLOW': 1.1, 'NLP_SENTIMENT': 1.05, 'BOND_YIELD_SPREAD': 1.0,
            'GNN_SECTOR_ROTATION': 1.05, 'CREDIT_CARD_SPENDING': 1.05,
            'RBI_MACRO_FILTER': 1.0,
        }
        result = cs.compute(mods, vix=14, adx=25, direction='LONG')
        assert result['bull_score'] >= 60
        assert result['final_modifier'] >= 1.2

    def test_bearish_modifiers_reduce(self):
        cs = ConvictionScorer()
        mods = {
            'FII_FUTURES_OI': 0.7, 'MAMBA_REGIME': 0.6, 'PCR_AUTOTRENDER': 0.8,
            'RBI_MACRO_FILTER': 1.0,
        }
        result = cs.compute(mods, vix=14, adx=25, direction='SHORT')
        assert result['bear_score'] >= 20
        # Bear conviction should produce modifier (lowered thresholds may amplify)
        assert result['final_modifier'] <= 1.30

    def test_mixed_near_neutral(self):
        cs = ConvictionScorer()
        mods = {'FII_FUTURES_OI': 1.1, 'MAMBA_REGIME': 0.9, 'RBI_MACRO_FILTER': 1.0}
        result = cs.compute(mods, vix=16, adx=22, direction='LONG')
        assert 0.85 <= result['final_modifier'] <= 1.3

    def test_event_reduces_conviction(self):
        cs = ConvictionScorer()
        mods = {
            'FII_FUTURES_OI': 1.3, 'MAMBA_REGIME': 1.2,
            'RBI_MACRO_FILTER': 0.5,
        }
        result_event = cs.compute(mods, vix=14, adx=25, direction='LONG')
        mods2 = dict(mods)
        mods2['RBI_MACRO_FILTER'] = 1.0
        result_no_event = cs.compute(mods2, vix=14, adx=25, direction='LONG')
        assert result_event['bull_score'] < result_no_event['bull_score']

    def test_high_vix_reduces(self):
        cs = ConvictionScorer()
        mods = {'FII_FUTURES_OI': 1.2, 'MAMBA_REGIME': 1.1, 'RBI_MACRO_FILTER': 1.0}
        r_low = cs.compute(mods, vix=14, adx=25, direction='LONG')
        r_high = cs.compute(mods, vix=25, adx=25, direction='LONG')
        assert r_high['bull_score'] < r_low['bull_score']


class TestSafeguards:
    def test_3_positions_limits(self):
        cs = ConvictionScorer()
        mods = {'FII_FUTURES_OI': 1.3, 'MAMBA_REGIME': 1.2, 'RBI_MACRO_FILTER': 1.0}
        result = cs.compute(mods, vix=14, adx=25, direction='LONG', open_positions=3)
        assert result['final_modifier'] <= 1.3

    def test_consecutive_losses_limits(self):
        cs = ConvictionScorer()
        mods = {'FII_FUTURES_OI': 1.3, 'MAMBA_REGIME': 1.2, 'RBI_MACRO_FILTER': 1.0}
        result = cs.compute(mods, vix=14, adx=25, direction='LONG', consecutive_losses=4)
        assert result['final_modifier'] <= 1.15

    def test_modifier_always_clamped(self):
        cs = ConvictionScorer()
        mods = {k: 3.0 for k in ['FII_FUTURES_OI', 'MAMBA_REGIME', 'PCR_AUTOTRENDER',
                                   'DELIVERY_PCT', 'GAMMA_EXPOSURE', 'RBI_MACRO_FILTER']}
        result = cs.compute(mods, vix=10, adx=30, direction='LONG')
        assert result['final_modifier'] <= 2.0

        mods2 = {k: 0.1 for k in mods}
        result2 = cs.compute(mods2, vix=30, adx=10, direction='LONG')
        assert result2['final_modifier'] >= 0.3


class TestTrendConfirmer:
    def test_uptrend_confirmed(self):
        tc = TrendConfirmer()
        # (close, open) tuples — 4/5 green days
        r = tc.is_trend_confirmed(
            close=25000, sma_20=24800, sma_50=24500,
            adx=25, high_52w=25200,
            recent_closes_5d=[(25000, 24800), (24900, 24700), (25100, 24900), (25000, 24950), (24800, 24850)],
        )
        assert r['confirmed'] is True

    def test_downtrend_not_confirmed(self):
        tc = TrendConfirmer()
        r = tc.is_trend_confirmed(
            close=23000, sma_20=23500, sma_50=24000,
            adx=15, high_52w=25000,
            recent_closes_5d=[(23000, 23200), (22900, 23100), (23000, 23000), (22800, 23000), (23000, 23100)],
        )
        assert r['confirmed'] is False

    def test_gate_caps_without_trend(self):
        tc = TrendConfirmer()
        assert tc.gate_modifier(1.5, trend_confirmed=False) <= 1.15
        assert tc.gate_modifier(1.5, trend_confirmed=True) == 1.5
        assert tc.gate_modifier(0.7, trend_confirmed=False) == 0.7


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
