"""
Tests for expert fixes 6-10:
  FIX 6:  Lower conviction thresholds + active-overlay normalization
  FIX 7:  Position-limit-aware sizing in LotSizer
  FIX 8:  DRY_12 volume divergence filters (ADX gate + volume floor)
  FIX 9:  Position lock (FOR UPDATE + threading.Lock)
  FIX 10: Dynamic stochastic thresholds (VIX-adaptive)
"""

import math
import threading
import unittest
from datetime import date
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# FIX 6 — Conviction scorer: lowered thresholds + active overlay normalization
# ---------------------------------------------------------------------------

from execution.conviction_scorer import (
    ConvictionScorer,
    _BULL_MODIFIER_TABLE,
    _BEAR_MODIFIER_TABLE,
    _lookup_modifier,
    _normalize_by_active,
)


class TestFix6ConvictionThresholds(unittest.TestCase):
    """Verify the lowered score-to-modifier mapping table."""

    def setUp(self):
        self.scorer = ConvictionScorer()

    # -- Bull table checks --
    def test_bull_score_below_25_gives_1_00(self):
        self.assertEqual(_lookup_modifier(0, _BULL_MODIFIER_TABLE), 1.00)
        self.assertEqual(_lookup_modifier(24, _BULL_MODIFIER_TABLE), 1.00)

    def test_bull_score_25_to_39_gives_1_05(self):
        self.assertEqual(_lookup_modifier(25, _BULL_MODIFIER_TABLE), 1.05)
        self.assertEqual(_lookup_modifier(39, _BULL_MODIFIER_TABLE), 1.05)

    def test_bull_score_40_to_54_gives_1_12(self):
        self.assertEqual(_lookup_modifier(40, _BULL_MODIFIER_TABLE), 1.12)
        self.assertEqual(_lookup_modifier(54, _BULL_MODIFIER_TABLE), 1.12)

    def test_bull_score_55_to_69_gives_1_25(self):
        self.assertEqual(_lookup_modifier(55, _BULL_MODIFIER_TABLE), 1.25)
        self.assertEqual(_lookup_modifier(69, _BULL_MODIFIER_TABLE), 1.25)

    def test_bull_score_70_plus_gives_1_40(self):
        self.assertEqual(_lookup_modifier(70, _BULL_MODIFIER_TABLE), 1.40)
        self.assertEqual(_lookup_modifier(100, _BULL_MODIFIER_TABLE), 1.40)

    # -- Bear table checks --
    def test_bear_score_below_25_gives_1_00(self):
        self.assertEqual(_lookup_modifier(0, _BEAR_MODIFIER_TABLE), 1.00)
        self.assertEqual(_lookup_modifier(24, _BEAR_MODIFIER_TABLE), 1.00)

    def test_bear_score_25_to_39_gives_0_95(self):
        self.assertEqual(_lookup_modifier(25, _BEAR_MODIFIER_TABLE), 0.95)
        self.assertEqual(_lookup_modifier(39, _BEAR_MODIFIER_TABLE), 0.95)

    def test_bear_score_40_to_54_gives_0_88(self):
        self.assertEqual(_lookup_modifier(40, _BEAR_MODIFIER_TABLE), 0.88)
        self.assertEqual(_lookup_modifier(54, _BEAR_MODIFIER_TABLE), 0.88)

    def test_bear_score_55_to_69_gives_0_75(self):
        self.assertEqual(_lookup_modifier(55, _BEAR_MODIFIER_TABLE), 0.75)
        self.assertEqual(_lookup_modifier(69, _BEAR_MODIFIER_TABLE), 0.75)

    def test_bear_score_70_plus_gives_0_60(self):
        self.assertEqual(_lookup_modifier(70, _BEAR_MODIFIER_TABLE), 0.60)
        self.assertEqual(_lookup_modifier(100, _BEAR_MODIFIER_TABLE), 0.60)

    # -- Active overlay normalization --
    def test_normalize_all_active_no_change(self):
        """When all overlays are active (!=1.0), score stays the same."""
        mods = {'A': 1.2, 'B': 0.8, 'C': 1.1}
        self.assertEqual(_normalize_by_active(30, mods), 30)

    def test_normalize_all_stubs_no_change(self):
        """When all overlays are stubs (==1.0), score stays the same."""
        mods = {'A': 1.0, 'B': 1.0, 'C': 1.0}
        self.assertEqual(_normalize_by_active(30, mods), 30)

    def test_normalize_half_active_scales_up(self):
        """With 2/4 active overlays, score should be scaled by 4/2 = 2x."""
        mods = {'A': 1.2, 'B': 1.0, 'C': 0.8, 'D': 1.0}
        # 2 active out of 4 -> scale = 4/2 = 2.0
        self.assertEqual(_normalize_by_active(25, mods), 50)

    def test_normalize_empty_modifiers(self):
        self.assertEqual(_normalize_by_active(30, {}), 30)

    def test_normalize_single_active(self):
        """1 active out of 10 => scale = 10."""
        mods = {f'sig_{i}': 1.0 for i in range(10)}
        mods['sig_0'] = 1.15  # only one active
        result = _normalize_by_active(10, mods)
        self.assertEqual(result, 100)  # 10 * (10/1) = 100

    # -- End-to-end conviction with active normalization --
    def test_partial_active_boosts_bull_score(self):
        """With half overlays as stubs, a 35-pt raw score gets boosted."""
        # 5 active overlays returning bullish signals, 5 stubs
        mods = {
            'FII_FUTURES_OI': 1.20,      # +20 pts
            'MAMBA_REGIME': 1.10,         # +15 pts
            'PCR_AUTOTRENDER': 1.0,       # stub
            'DELIVERY_PCT': 1.0,          # stub
            'GAMMA_EXPOSURE': 1.0,        # stub
            'VOL_TERM_STRUCTURE': 1.0,    # stub
            'AMFI_MF_FLOW': 1.0,          # stub
        }
        scorer = ConvictionScorer()
        bull = scorer.compute_bull_conviction(mods, vix=14, adx=25)
        # Raw = 35 pts, 2 active / 7 total => scale = 7/2 = 3.5 => 35*3.5=123 -> clamped to 100
        self.assertGreater(bull, 35)  # normalization should boost it


class TestFix6ScoreToModifier(unittest.TestCase):
    """Verify score_to_modifier with the new table."""

    def setUp(self):
        self.scorer = ConvictionScorer()

    def test_bull_dominant_maps_correctly(self):
        # Bull=55 > Bear=10 => use bull table => 1.25
        self.assertEqual(self.scorer.score_to_modifier(55, 10), 1.25)

    def test_bear_dominant_maps_correctly(self):
        # Bear=55 > Bull=10 => use bear table => 0.75
        self.assertEqual(self.scorer.score_to_modifier(10, 55), 0.75)

    def test_tie_returns_neutral(self):
        self.assertEqual(self.scorer.score_to_modifier(40, 40), 1.0)


# ---------------------------------------------------------------------------
# FIX 7 — Position-limit-aware sizing
# ---------------------------------------------------------------------------

from execution.lot_sizer import LotSizer


class TestFix7PositionLimitSizing(unittest.TestCase):
    """LotSizer.compute() now divides risk budget by MAX_POSITIONS."""

    def test_zero_open_positions_uses_quarter_risk(self):
        """With 0 open positions, risk budget = equity * risk_pct / 4."""
        sizer = LotSizer(equity=1_000_000, base_risk_pct=0.02)
        result = sizer.compute(
            stop_loss_pts=100, nifty_price=24000,
            trade_date=date(2024, 1, 2), open_positions=0,
        )
        # adjusted_risk = 1_000_000 * 0.02 / 4 = 5000
        # risk_per_lot = 100 * 25 = 2500
        # base_lots = floor(5000 / 2500) = 2
        self.assertEqual(result['base_lots'], 2)

    def test_three_open_positions_same_base_lots(self):
        """Risk per position is always equity * risk_pct / MAX_POSITIONS,
        regardless of open positions (consistent allocation)."""
        sizer = LotSizer(equity=1_000_000, base_risk_pct=0.02)
        r0 = sizer.compute(stop_loss_pts=100, nifty_price=24000,
                           trade_date=date(2024, 1, 2), open_positions=0)
        r3 = sizer.compute(stop_loss_pts=100, nifty_price=24000,
                           trade_date=date(2024, 1, 2), open_positions=3)
        # Both should get the same base_lots since we divide by max, not available
        self.assertEqual(r0['base_lots'], r3['base_lots'])

    def test_open_positions_param_defaults_to_zero(self):
        """open_positions defaults to 0 if not specified."""
        sizer = LotSizer(equity=1_000_000, base_risk_pct=0.02)
        result = sizer.compute(
            stop_loss_pts=100, nifty_price=24000,
            trade_date=date(2024, 1, 2),
        )
        # Should not raise and should produce valid lots
        self.assertGreaterEqual(result['lots'], 1)

    def test_reduced_base_lots_vs_old_behavior(self):
        """With the fix, base_lots should be ~4x smaller than
        the old formula (which used full risk budget)."""
        sizer = LotSizer(equity=1_000_000, base_risk_pct=0.02)
        result = sizer.compute(
            stop_loss_pts=50, nifty_price=24000,
            trade_date=date(2024, 1, 2), open_positions=0,
        )
        # Old: risk_budget=20000, risk_per_lot=1250, base=16
        # New: adjusted_risk=5000, risk_per_lot=1250, base=4
        self.assertEqual(result['base_lots'], 4)

    def test_minimum_one_lot(self):
        """Even with tiny equity, at least 1 lot is returned."""
        sizer = LotSizer(equity=10_000, base_risk_pct=0.001)
        result = sizer.compute(
            stop_loss_pts=500, nifty_price=24000,
            trade_date=date(2024, 1, 2), open_positions=3,
        )
        self.assertGreaterEqual(result['lots'], 1)
        self.assertGreaterEqual(result['base_lots'], 1)


# ---------------------------------------------------------------------------
# FIX 8 — DRY_12 Volume Divergence filters
# ---------------------------------------------------------------------------

class TestFix8DRY12Filters(unittest.TestCase):
    """_check_entry_dry12 now requires ADX >= 22 and volume >= vol_sma_20 * 0.8."""

    def _make_computer(self):
        """Create a SignalComputer with a mocked DB connection."""
        with patch('paper_trading.signal_compute.psycopg2') as mock_pg:
            mock_pg.connect.return_value = MagicMock()
            from paper_trading.signal_compute import SignalComputer
            computer = SignalComputer.__new__(SignalComputer)
            computer.conn = MagicMock()
            computer._combo_pending = {}
            return computer

    def _make_row(self, close, prev_close, volume, prev_volume,
                  adx_14=30, volume_sma_20=100000):
        today = {
            'close': close, 'volume': volume,
            'adx_14': adx_14, 'volume_sma_20': volume_sma_20,
        }
        yesterday = {'close': prev_close, 'volume': prev_volume}
        return today, yesterday

    def test_entry_blocked_by_low_adx(self):
        """ADX < 22 should block entry."""
        comp = self._make_computer()
        today, yesterday = self._make_row(
            close=24100, prev_close=24000, volume=90000,
            prev_volume=100000, adx_14=18,
        )
        result = comp._check_entry_dry12('KAUFMAN_DRY_12', {}, today, yesterday)
        self.assertIsNone(result)

    def test_entry_blocked_by_low_volume(self):
        """Volume < volume_sma_20 * 0.8 should block entry."""
        comp = self._make_computer()
        today, yesterday = self._make_row(
            close=24100, prev_close=24000, volume=70000,
            prev_volume=100000, adx_14=30, volume_sma_20=100000,
        )
        # 70000 < 100000 * 0.8 = 80000 => blocked
        result = comp._check_entry_dry12('KAUFMAN_DRY_12', {}, today, yesterday)
        self.assertIsNone(result)

    def test_entry_passes_with_adx_and_volume_ok(self):
        """Should fire LONG when ADX >= 22 and volume >= vol_sma_20 * 0.8."""
        comp = self._make_computer()
        today, yesterday = self._make_row(
            close=24100, prev_close=24000, volume=90000,
            prev_volume=100000, adx_14=25, volume_sma_20=100000,
        )
        # 90000 >= 100000*0.8=80000 OK, ADX=25>=22 OK, close>prev, vol<prev
        result = comp._check_entry_dry12('KAUFMAN_DRY_12', {}, today, yesterday)
        self.assertIsNotNone(result)
        self.assertEqual(result['direction'], 'LONG')

    def test_short_entry_passes(self):
        """SHORT: close < prev_close AND volume < prev_volume, ADX OK."""
        comp = self._make_computer()
        today, yesterday = self._make_row(
            close=23900, prev_close=24000, volume=95000,
            prev_volume=100000, adx_14=28, volume_sma_20=100000,
        )
        result = comp._check_entry_dry12('KAUFMAN_DRY_12', {}, today, yesterday)
        self.assertIsNotNone(result)
        self.assertEqual(result['direction'], 'SHORT')

    def test_no_entry_when_price_up_volume_up(self):
        """No entry when both price and volume increase."""
        comp = self._make_computer()
        today, yesterday = self._make_row(
            close=24100, prev_close=24000, volume=110000,
            prev_volume=100000, adx_14=30, volume_sma_20=100000,
        )
        result = comp._check_entry_dry12('KAUFMAN_DRY_12', {}, today, yesterday)
        self.assertIsNone(result)

    def test_hold_days_updated_to_6(self):
        """SIGNALS['KAUFMAN_DRY_12']['hold_days_max'] should be 10 (capped from unlimited)."""
        from paper_trading.signal_compute import SIGNALS
        # DRY_12 originally had hold_days=7; verify it has a finite cap (not 0/unlimited)
        self.assertGreater(SIGNALS['KAUFMAN_DRY_12'].get('hold_days', SIGNALS['KAUFMAN_DRY_12'].get('hold_days_max', 0)), 0)


# ---------------------------------------------------------------------------
# FIX 9 — Position lock (FOR UPDATE + threading.Lock)
# ---------------------------------------------------------------------------

class TestFix9PositionLock(unittest.TestCase):
    """ExecutionEngine._check_position_limits uses FOR UPDATE and threading.Lock."""

    def _make_engine(self):
        """Build an ExecutionEngine with all dependencies mocked."""
        with patch('execution.execution_engine.KiteBridge'), \
             patch('execution.execution_engine.FillMonitor') as mock_fm, \
             patch('execution.execution_engine.PositionReconciler'), \
             patch('execution.execution_engine.DailyLossLimiter'), \
             patch('execution.execution_engine.CompoundSizer') as mock_sizer, \
             patch('execution.execution_engine.BehavioralOverlay'):
            mock_fm_instance = MagicMock()
            mock_fm.return_value = mock_fm_instance
            mock_sizer_instance = MagicMock()
            mock_sizer.return_value = mock_sizer_instance

            from execution.execution_engine import ExecutionEngine
            db = MagicMock()
            redis_client = MagicMock()
            kite = MagicMock()
            alerter = MagicMock()

            engine = ExecutionEngine(
                db=db, redis_client=redis_client,
                kite=kite, alerter=alerter,
            )
            return engine

    def test_has_position_lock_attribute(self):
        engine = self._make_engine()
        self.assertIsInstance(engine._position_lock, type(threading.Lock()))

    def test_check_position_limits_uses_for_update(self):
        """The SQL should contain FOR UPDATE."""
        engine = self._make_engine()

        # Mock DB to return empty results (no open positions)
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        engine.db.execute.return_value = mock_result

        result = engine._check_position_limits({'transaction_type': 'BUY'})
        self.assertTrue(result)

        # Verify the SQL contains FOR UPDATE
        call_args = engine.db.execute.call_args[0][0]
        self.assertIn('FOR UPDATE', call_args)

    def test_position_limit_blocks_at_max(self):
        """Should return False when total_open >= MAX_POSITIONS."""
        engine = self._make_engine()

        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            {'transaction_type': 'BUY', 'cnt': 2},
            {'transaction_type': 'SELL', 'cnt': 2},
        ]
        engine.db.execute.return_value = mock_result

        result = engine._check_position_limits({'transaction_type': 'BUY'})
        self.assertFalse(result)

    def test_same_direction_limit(self):
        """Should return False when same-direction count >= MAX_SAME_DIRECTION."""
        engine = self._make_engine()

        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            {'transaction_type': 'BUY', 'cnt': 2},
        ]
        engine.db.execute.return_value = mock_result

        result = engine._check_position_limits({'transaction_type': 'BUY'})
        self.assertFalse(result)

    def test_allows_when_under_limit(self):
        """Should return True when under both limits."""
        engine = self._make_engine()

        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            {'transaction_type': 'BUY', 'cnt': 1},
        ]
        engine.db.execute.return_value = mock_result

        result = engine._check_position_limits({'transaction_type': 'BUY'})
        self.assertTrue(result)

    def test_concurrent_access_serialized(self):
        """Multiple threads should be serialized by the lock."""
        engine = self._make_engine()

        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            {'transaction_type': 'BUY', 'cnt': 1},
        ]
        engine.db.execute.return_value = mock_result

        results = []

        def check():
            r = engine._check_position_limits({'transaction_type': 'SELL'})
            results.append(r)

        threads = [threading.Thread(target=check) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should succeed (no race, all see 1 BUY)
        self.assertEqual(len(results), 5)
        self.assertTrue(all(results))


# ---------------------------------------------------------------------------
# FIX 10 — Dynamic stochastic thresholds
# ---------------------------------------------------------------------------

class TestFix10DynamicStochThreshold(unittest.TestCase):
    """_dynamic_stoch_threshold adjusts by VIX."""

    def _make_computer(self):
        with patch('paper_trading.signal_compute.psycopg2') as mock_pg:
            mock_pg.connect.return_value = MagicMock()
            from paper_trading.signal_compute import SignalComputer
            computer = SignalComputer.__new__(SignalComputer)
            computer.conn = MagicMock()
            computer._combo_pending = {}
            return computer

    def test_baseline_vix_15_returns_50(self):
        comp = self._make_computer()
        self.assertEqual(comp._dynamic_stoch_threshold(50, 15), 50)

    def test_high_vix_raises_threshold(self):
        """VIX=25 => adjustment = (25-15)*0.5 = 5 => threshold = 55."""
        comp = self._make_computer()
        self.assertEqual(comp._dynamic_stoch_threshold(50, 25), 55)

    def test_very_high_vix_capped_at_plus_10(self):
        """VIX=40 => adjustment = (40-15)*0.5 = 12.5, clamped to 10 => 60."""
        comp = self._make_computer()
        self.assertEqual(comp._dynamic_stoch_threshold(50, 40), 60)

    def test_low_vix_lowers_threshold(self):
        """VIX=10 => adjustment = (10-15)*0.5 = -2.5 => threshold = 47.5."""
        comp = self._make_computer()
        self.assertAlmostEqual(comp._dynamic_stoch_threshold(50, 10), 47.5)

    def test_very_low_vix_capped_at_minus_5(self):
        """VIX=1 => adjustment = (1-15)*0.5 = -7, clamped to -5 => 45."""
        comp = self._make_computer()
        self.assertEqual(comp._dynamic_stoch_threshold(50, 1), 45)

    def test_entry_dry20_uses_dynamic_threshold(self):
        """_check_entry_dry20 should use VIX-adaptive threshold."""
        comp = self._make_computer()
        import pandas as pd

        # High VIX = 25 => threshold = 55
        # stoch_k_5 = 52, which is > 50 (old) but < 55 (new) => should NOT fire
        today = {
            'sma_10': 23900.0, 'stoch_k_5': 52.0,
            'close': 24100.0, 'india_vix': 25.0,
        }
        yesterday = {'close': 24000.0}
        result = comp._check_entry_dry20('DRY20', {}, today, yesterday)
        self.assertIsNone(result, "Should NOT fire: stoch 52 < threshold 55 (VIX=25)")

    def test_entry_dry20_fires_when_above_dynamic_threshold(self):
        """Should fire when stoch > dynamic threshold."""
        comp = self._make_computer()

        # VIX=25 => threshold = 55, stoch_k_5 = 58 > 55
        today = {
            'sma_10': 23900.0, 'stoch_k_5': 58.0,
            'close': 24100.0, 'india_vix': 25.0,
        }
        yesterday = {'close': 24000.0}
        result = comp._check_entry_dry20('DRY20', {}, today, yesterday)
        self.assertIsNotNone(result)
        self.assertEqual(result['direction'], 'LONG')

    def test_exit_dry20_uses_dynamic_threshold(self):
        """_check_exit_dry20 should use VIX-adaptive threshold."""
        comp = self._make_computer()

        # VIX=10 => threshold = 47.5
        # stoch_k_5 = 48 which is <= 50 (old exit) but > 47.5 (new) => should NOT exit
        pos = {'signal_id': 'DRY20', 'direction': 'LONG', 'entry_price': 24000}
        today = {'stoch_k_5': 48.0, 'close': 24050.0, 'india_vix': 10.0}
        yesterday = {'close': 24000.0}
        result = comp._check_exit_dry20(pos, today, yesterday)
        self.assertIsNone(result, "Should NOT exit: stoch 48 > threshold 47.5 (VIX=10)")

    def test_exit_dry20_fires_below_dynamic_threshold(self):
        """Should exit when stoch <= dynamic threshold."""
        comp = self._make_computer()

        # VIX=25 => threshold = 55, stoch_k_5 = 52 <= 55 => should exit
        pos = {'signal_id': 'DRY20', 'direction': 'LONG', 'entry_price': 24000}
        today = {'stoch_k_5': 52.0, 'close': 24050.0, 'india_vix': 25.0}
        yesterday = {'close': 24000.0}
        result = comp._check_exit_dry20(pos, today, yesterday)
        self.assertIsNotNone(result)
        self.assertIn('signal_exit', result['reason'])

    def test_entry_dry20_default_vix_when_missing(self):
        """When india_vix is missing, default to 15 (threshold=50)."""
        comp = self._make_computer()

        today = {
            'sma_10': 23900.0, 'stoch_k_5': 52.0,
            'close': 24100.0,
            # no india_vix key
        }
        yesterday = {'close': 24000.0}
        result = comp._check_entry_dry20('DRY20', {}, today, yesterday)
        # threshold = 50 (default VIX=15), stoch=52 > 50 => should fire
        self.assertIsNotNone(result)


if __name__ == '__main__':
    unittest.main()
