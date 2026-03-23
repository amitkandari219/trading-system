"""Tests for lot-based position sizer."""

import pytest
from datetime import date
from execution.lot_sizer import LotSizer, get_lot_size


class TestLotSize:
    def test_pre_jul2023(self):
        assert get_lot_size(date(2023, 6, 30)) == 75

    def test_post_jul2023(self):
        assert get_lot_size(date(2023, 7, 1)) == 25

    def test_current(self):
        assert get_lot_size(date(2026, 1, 1)) == 25


class TestBasicComputation:
    def test_basic_lots(self):
        """₹10L equity, 2% risk, 50pt stop, lot=25 → risk_budget=20K, risk/lot=1250 → 16 lots."""
        s = LotSizer(equity=1_000_000)
        r = s.compute(stop_loss_pts=50, nifty_price=24000, trade_date=date(2026, 1, 1))
        assert r['lots'] >= 1
        assert r['lots'] <= 20
        assert r['lot_size'] == 25

    def test_large_stop_few_lots(self):
        """Large SL → fewer lots."""
        s = LotSizer(equity=1_000_000)
        r = s.compute(stop_loss_pts=500, nifty_price=24000, trade_date=date(2026, 1, 1))
        assert r['lots'] < 5

    def test_small_stop_more_lots(self):
        """Small SL → more lots (capped by margin)."""
        s = LotSizer(equity=1_000_000)
        r = s.compute(stop_loss_pts=10, nifty_price=24000, trade_date=date(2026, 1, 1))
        assert r['lots'] >= 5


class TestOverlayModifiers:
    def test_overlays_change_lots(self):
        """Overlay modifiers must actually change lot count."""
        s = LotSizer(equity=1_000_000)
        base = s.compute(stop_loss_pts=100, nifty_price=24000)

        bullish = s.compute(stop_loss_pts=100, nifty_price=24000,
                            overlay_modifiers={'MAMBA_REGIME': 1.3, 'FII_FUTURES_OI': 1.2})
        bearish = s.compute(stop_loss_pts=100, nifty_price=24000,
                            overlay_modifiers={'MAMBA_REGIME': 0.5, 'CRISIS_SHORT': 0.4})

        assert bullish['lots'] >= base['lots']
        assert bearish['lots'] <= base['lots']
        assert bullish['composite_modifier'] > 1.0
        assert bearish['composite_modifier'] < 1.0

    def test_no_overlays_modifier_is_1(self):
        s = LotSizer(equity=1_000_000)
        r = s.compute(stop_loss_pts=100, nifty_price=24000)
        assert r['composite_modifier'] == 1.0


class TestConstraints:
    def test_min_1_lot(self):
        """Always at least 1 lot."""
        s = LotSizer(equity=50_000)  # tiny account
        r = s.compute(stop_loss_pts=1000, nifty_price=24000)
        assert r['lots'] >= 1

    def test_margin_cap(self):
        """Lots capped by margin availability."""
        s = LotSizer(equity=500_000, max_exposure_pct=0.30)
        r = s.compute(stop_loss_pts=10, nifty_price=24000)
        assert r['margin_required'] <= 500_000 * 0.30

    def test_hard_cap_20(self):
        """Hard cap at 20 lots."""
        s = LotSizer(equity=10_000_000, max_lots_cap=20)
        r = s.compute(stop_loss_pts=10, nifty_price=24000)
        assert r['lots'] <= 20

    def test_composite_clamped_low(self):
        """All overlays bearish → clamped at 0.3."""
        s = LotSizer(equity=1_000_000)
        mods = {k: 0.3 for k in ['MAMBA_REGIME', 'CRISIS_SHORT', 'FII_FUTURES_OI',
                                   'SENTIMENT_COMPOSITE', 'PCR_AUTOTRENDER']}
        r = s.compute(stop_loss_pts=100, nifty_price=24000, overlay_modifiers=mods)
        assert r['composite_modifier'] >= 0.3

    def test_composite_clamped_high(self):
        """All overlays bullish → clamped at 2.0."""
        s = LotSizer(equity=1_000_000)
        mods = {k: 2.0 for k in ['MAMBA_REGIME', 'FII_FUTURES_OI',
                                   'SENTIMENT_COMPOSITE', 'PCR_AUTOTRENDER']}
        r = s.compute(stop_loss_pts=100, nifty_price=24000, overlay_modifiers=mods)
        assert r['composite_modifier'] <= 2.0


class TestPreJul2023:
    def test_lot_size_75(self):
        s = LotSizer(equity=1_000_000)
        r = s.compute(stop_loss_pts=100, nifty_price=15000, trade_date=date(2020, 1, 1))
        assert r['lot_size'] == 75

    def test_fewer_lots_with_75(self):
        """75-lot size = 3x exposure per lot → fewer lots for same risk."""
        s = LotSizer(equity=1_000_000)
        r75 = s.compute(stop_loss_pts=100, nifty_price=15000, trade_date=date(2020, 1, 1))
        r25 = s.compute(stop_loss_pts=100, nifty_price=15000, trade_date=date(2024, 1, 1))
        # 75-lot trades should have fewer lots than 25-lot for same risk
        assert r75['lots'] < r25['lots'] or r75['lots'] == r25['lots'] == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
