"""Tests for Kelly optimizer and adaptive Kelly."""
import pytest
from execution.adaptive_kelly import AdaptiveKelly


class TestKellyFormula:
    def test_standard_kelly(self):
        """60% WR, avg_win/avg_loss=1.5 → f* = 0.60 - 0.40/1.5 = 0.333"""
        ak = AdaptiveKelly()
        trades = [{'pnl_pct': 0.015}] * 60 + [{'pnl_pct': -0.010}] * 40
        f = ak.compute_kelly_from_trades(trades)
        assert 0.25 < f < 0.45  # ~0.333

    def test_low_wr_low_kelly(self):
        """40% WR → low Kelly"""
        ak = AdaptiveKelly()
        trades = [{'pnl_pct': 0.02}] * 40 + [{'pnl_pct': -0.015}] * 60
        f = ak.compute_kelly_from_trades(trades)
        assert f < 0.30

    def test_few_trades_default(self):
        ak = AdaptiveKelly()
        f = ak.compute_kelly_from_trades([{'pnl_pct': 0.01}] * 5)
        assert f == 0.50  # default for < 20 trades


class TestAdaptiveKelly:
    def test_normal_conditions(self):
        ak = AdaptiveKelly(base_fraction=0.75)
        r = ak.get_fraction(drawdown_pct=0.02, recent_wr=0.50, vix=15)
        assert r['fraction'] == 0.75
        assert r['gear'] == 'NORMAL'

    def test_drawdown_reduces(self):
        ak = AdaptiveKelly(base_fraction=0.75)
        r = ak.get_fraction(drawdown_pct=0.12)
        assert r['fraction'] < 0.75
        assert r['fraction'] <= 0.50

    def test_hot_streak_increases(self):
        ak = AdaptiveKelly(base_fraction=0.75)
        r = ak.get_fraction(recent_wr=0.65)
        assert r['fraction'] >= 0.80

    def test_crisis_reduces(self):
        ak = AdaptiveKelly(base_fraction=0.75)
        r = ak.get_fraction(vix=28, regime='VOLATILE_BEAR')
        assert r['fraction'] < 0.50

    def test_clamp_low(self):
        ak = AdaptiveKelly(base_fraction=0.30)
        r = ak.get_fraction(drawdown_pct=0.15, consecutive_losers=5, vix=30)
        assert r['fraction'] >= 0.20

    def test_clamp_high(self):
        ak = AdaptiveKelly(base_fraction=0.95)
        r = ak.get_fraction(recent_wr=0.70)
        assert r['fraction'] <= 1.00

    def test_consecutive_losers(self):
        ak = AdaptiveKelly(base_fraction=0.75)
        r = ak.get_fraction(consecutive_losers=5)
        assert r['fraction'] <= 0.60


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
