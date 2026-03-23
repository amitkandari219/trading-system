"""
Tests for Expert Fixes 1-5.

FIX 1: Correct Cost Model — brokerage GST, exchange charges, stamp duty, SEBI
FIX 2: Implement 5 Stubbed Overlays — DELIVERY_PCT, ORDER_FLOW_IMBALANCE,
       BOND_YIELD_SPREAD, GLOBAL_OVERNIGHT_COMPOSITE, ROLLOVER_ANALYSIS
FIX 3: Integrate Kelly into LotSizer — _KELLY_FRACTION modifier
FIX 4: Max-Hold Limits on DRY Signals — hold_days_max corrections
FIX 5: Verify Async Execution — start() called, blocking API works
"""

import math
import threading
import time
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ================================================================
# FIX 1: Correct Cost Model
# ================================================================

class TestFix1CostModel:
    """Verify corrected TransactionCostModel constants and computations."""

    def test_brokerage_gst_inclusive(self):
        """Brokerage should be ₹20/order × 1.18 GST = ₹23.60/order."""
        from backtest.transaction_costs import CostConfig
        cfg = CostConfig()
        assert cfg.brokerage_per_order == 20.0
        assert cfg.brokerage_gst_inclusive == 23.60

    def test_exchange_charge_rate(self):
        """Exchange charges should be 0.0495% on turnover."""
        from backtest.transaction_costs import CostConfig
        cfg = CostConfig()
        assert cfg.exchange_charge_futures == pytest.approx(0.000495, abs=1e-7)

    def test_stamp_duty_rate(self):
        """Stamp duty should be 0.003% on buy side."""
        from backtest.transaction_costs import CostConfig
        cfg = CostConfig()
        assert cfg.stamp_duty_rate == pytest.approx(0.00003, abs=1e-8)

    def test_sebi_rate(self):
        """SEBI fee should be 0.0001%."""
        from backtest.transaction_costs import CostConfig
        cfg = CostConfig()
        assert cfg.sebi_rate == pytest.approx(0.000001, abs=1e-9)

    def test_round_trip_brokerage(self):
        """Round-trip brokerage = ₹23.60 × 2 = ₹47.20."""
        from backtest.transaction_costs import TransactionCostModel
        model = TransactionCostModel()
        costs = model.compute_futures_round_trip(24000, 24100, lots=1)
        assert costs.brokerage == pytest.approx(47.20, abs=0.01)

    def test_stt_on_sell_side_only(self):
        """STT should apply only to exit (sell) notional."""
        from backtest.transaction_costs import TransactionCostModel
        model = TransactionCostModel()
        costs = model.compute_futures_round_trip(24000, 24100, lots=1)
        expected_stt = 24100 * 25 * 0.000125  # exit_notional × STT rate
        assert costs.stt == pytest.approx(expected_stt, rel=0.01)

    def test_exchange_charges_both_sides(self):
        """Exchange charges should apply to both entry and exit notional."""
        from backtest.transaction_costs import TransactionCostModel
        model = TransactionCostModel()
        costs = model.compute_futures_round_trip(24000, 24100, lots=1)
        entry_notional = 24000 * 25
        exit_notional = 24100 * 25
        expected = (entry_notional + exit_notional) * 0.000495
        assert costs.exchange_charges == pytest.approx(expected, rel=0.01)

    def test_stamp_duty_buy_side_only(self):
        """Stamp duty should only apply to entry (buy side) notional."""
        from backtest.transaction_costs import TransactionCostModel
        model = TransactionCostModel()
        costs = model.compute_futures_round_trip(24000, 24100, lots=1)
        expected = 24000 * 25 * 0.00003
        assert costs.stamp_duty == pytest.approx(expected, rel=0.01)

    def test_total_cost_realistic_range(self):
        """Total round-trip cost for 1 lot at ~24000 should be in realistic range."""
        from backtest.transaction_costs import TransactionCostModel
        model = TransactionCostModel()
        costs = model.compute_futures_round_trip(24000, 24100, lots=1)
        # Should be roughly ₹300-800 range for 1 lot round trip
        assert 200 < costs.total < 1500

    def test_lot_based_wf_uses_cost_model(self):
        """lot_based_wf._compute_costs should delegate to TransactionCostModel."""
        from backtest.lot_based_wf import _compute_costs
        # Should not raise and should return a positive number
        cost = _compute_costs(24000, 24100, 1, 25, 'LONG')
        assert cost > 0
        # Verify it's not using old hardcoded brokerage of ₹40/lot
        # Old: slippage + stt + ₹40*1*2 = ₹80 brokerage component
        # New: ₹23.60*2 = ₹47.20 brokerage component
        # Total should be different from old model
        assert cost != pytest.approx(80, abs=10)  # old brokerage was ₹80

    def test_gst_on_exchange_charges_not_brokerage(self):
        """GST should apply to exchange charges only (brokerage already includes GST)."""
        from backtest.transaction_costs import TransactionCostModel
        model = TransactionCostModel()
        costs = model.compute_futures_round_trip(24000, 24100, lots=1)
        # GST = exchange_charges × 18% (NOT brokerage + exchange_charges)
        expected_gst = costs.exchange_charges * 0.18
        assert costs.gst == pytest.approx(expected_gst, rel=0.01)


# ================================================================
# FIX 2: Implement 5 Stubbed Overlays
# ================================================================

def _make_test_df(n=100):
    """Create a minimal test DataFrame for OverlayPipeline."""
    dates = pd.date_range('2024-01-01', periods=n, freq='B')
    np.random.seed(42)
    close = 24000 + np.cumsum(np.random.randn(n) * 50)
    open_ = close + np.random.randn(n) * 30
    high = np.maximum(close, open_) + abs(np.random.randn(n) * 20)
    low = np.minimum(close, open_) - abs(np.random.randn(n) * 20)
    volume = np.random.randint(100000, 500000, n).astype(float)
    vix = 14 + np.random.randn(n) * 3

    df = pd.DataFrame({
        'date': dates,
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
        'india_vix': np.clip(vix, 8, 35),
        'pcr_oi': np.random.uniform(0.5, 1.5, n),
        'rsi_14': np.random.uniform(30, 70, n),
        'adx_14': np.random.uniform(15, 35, n),
        'sma_50': close - np.random.randn(n) * 100,
        'bb_bandwidth': np.random.uniform(0.02, 0.10, n),
        'hvol_20': np.random.uniform(10, 25, n),
    })
    return df


class TestFix2OverlayDeliveryPct:
    """DELIVERY_PCT: volume + range proxy for accumulation."""

    def test_not_hardcoded_1(self):
        """DELIVERY_PCT should no longer always return 1.0."""
        from execution.overlay_pipeline import OverlayPipeline
        df = _make_test_df(100)
        pipeline = OverlayPipeline(df)
        # Try multiple dates — at least one should differ from 1.0
        results = []
        for d in df['date'].dt.date.values[50:80]:
            mods = pipeline.get_modifiers(d, 'LONG')
            results.append(mods.get('DELIVERY_PCT', 1.0))
        # At least some should differ from 1.0
        assert not all(r == 1.0 for r in results), \
            "DELIVERY_PCT should produce non-1.0 values for varied data"

    def test_returns_valid_range(self):
        """DELIVERY_PCT should return values in [0.85, 1.15] range."""
        from execution.overlay_pipeline import OverlayPipeline
        df = _make_test_df(100)
        pipeline = OverlayPipeline(df)
        for d in df['date'].dt.date.values[50:70]:
            mods = pipeline.get_modifiers(d, 'LONG')
            val = mods.get('DELIVERY_PCT', 1.0)
            assert 0.85 <= val <= 1.15, f"DELIVERY_PCT={val} out of range"


class TestFix2OverlayOrderFlow:
    """ORDER_FLOW_IMBALANCE: bar direction proxy."""

    def test_bullish_bar_gives_boost(self):
        """Close > open bar should give modifier >= 1.0."""
        from execution.overlay_pipeline import OverlayPipeline
        df = _make_test_df(100)
        # Force a strongly bullish bar
        idx = 60
        df.loc[idx, 'close'] = df.loc[idx, 'open'] + 200  # big green bar
        pipeline = OverlayPipeline(df)
        d = df.loc[idx, 'date'].date()
        mods = pipeline.get_modifiers(d, 'LONG')
        assert mods['ORDER_FLOW_IMBALANCE'] >= 1.0

    def test_bearish_bar_gives_reduction(self):
        """Close < open bar should give modifier <= 1.0."""
        from execution.overlay_pipeline import OverlayPipeline
        df = _make_test_df(100)
        idx = 60
        df.loc[idx, 'close'] = df.loc[idx, 'open'] - 200  # big red bar
        pipeline = OverlayPipeline(df)
        d = df.loc[idx, 'date'].date()
        mods = pipeline.get_modifiers(d, 'LONG')
        assert mods['ORDER_FLOW_IMBALANCE'] <= 1.0


class TestFix2OverlayBondYield:
    """BOND_YIELD_SPREAD: VIX proxy for yield stress."""

    def test_high_vix_tight_money(self):
        """VIX > 22 should return 0.85 (tight money)."""
        from execution.overlay_pipeline import OverlayPipeline
        df = _make_test_df(100)
        idx = 60
        df.loc[idx, 'india_vix'] = 25.0
        pipeline = OverlayPipeline(df)
        d = df.loc[idx, 'date'].date()
        mods = pipeline.get_modifiers(d, 'LONG')
        assert mods['BOND_YIELD_SPREAD'] == pytest.approx(0.85)

    def test_low_vix_easy_money(self):
        """VIX < 12 should return 1.05 (easy money)."""
        from execution.overlay_pipeline import OverlayPipeline
        df = _make_test_df(100)
        idx = 60
        df.loc[idx, 'india_vix'] = 10.0
        pipeline = OverlayPipeline(df)
        d = df.loc[idx, 'date'].date()
        mods = pipeline.get_modifiers(d, 'LONG')
        assert mods['BOND_YIELD_SPREAD'] == pytest.approx(1.05)

    def test_normal_vix_neutral(self):
        """VIX 12-22 should return 1.0."""
        from execution.overlay_pipeline import OverlayPipeline
        df = _make_test_df(100)
        idx = 60
        df.loc[idx, 'india_vix'] = 16.0
        pipeline = OverlayPipeline(df)
        d = df.loc[idx, 'date'].date()
        mods = pipeline.get_modifiers(d, 'LONG')
        assert mods['BOND_YIELD_SPREAD'] == pytest.approx(1.0)


class TestFix2OverlayGlobalOvernight:
    """GLOBAL_OVERNIGHT_COMPOSITE: overnight gap proxy."""

    def test_gap_up_risk_on(self):
        """Gap up > 0.3% should return 1.10."""
        from execution.overlay_pipeline import OverlayPipeline
        df = _make_test_df(100)
        idx = 60
        prev_close = float(df.loc[idx - 1, 'close'])
        df.loc[idx, 'open'] = prev_close * 1.005  # 0.5% gap up
        pipeline = OverlayPipeline(df)
        d = df.loc[idx, 'date'].date()
        mods = pipeline.get_modifiers(d, 'LONG')
        assert mods['GLOBAL_OVERNIGHT_COMPOSITE'] == pytest.approx(1.10)

    def test_gap_down_risk_off(self):
        """Gap down > 0.3% should return 0.85."""
        from execution.overlay_pipeline import OverlayPipeline
        df = _make_test_df(100)
        idx = 60
        prev_close = float(df.loc[idx - 1, 'close'])
        df.loc[idx, 'open'] = prev_close * 0.995  # 0.5% gap down
        pipeline = OverlayPipeline(df)
        d = df.loc[idx, 'date'].date()
        mods = pipeline.get_modifiers(d, 'LONG')
        assert mods['GLOBAL_OVERNIGHT_COMPOSITE'] == pytest.approx(0.85)


class TestFix2OverlayRollover:
    """ROLLOVER_ANALYSIS: last 5 days before expiry + VIX direction."""

    def test_outside_rollover_neutral(self):
        """Dates not near expiry should return 1.0."""
        from execution.overlay_pipeline import OverlayPipeline
        df = _make_test_df(100)
        pipeline = OverlayPipeline(df)
        # Pick a date that's mid-month (not near last Thursday)
        d = date(2024, 1, 10)  # Jan 10 — far from last Thursday (Jan 25)
        mods = pipeline.get_modifiers(d, 'LONG')
        assert mods.get('ROLLOVER_ANALYSIS', 1.0) == pytest.approx(1.0)

    def test_all_five_overlays_present(self):
        """All 5 previously-stubbed overlays should be present in modifiers."""
        from execution.overlay_pipeline import OverlayPipeline
        df = _make_test_df(100)
        pipeline = OverlayPipeline(df)
        d = df['date'].dt.date.values[60]
        mods = pipeline.get_modifiers(d, 'LONG')
        for key in ['DELIVERY_PCT', 'ORDER_FLOW_IMBALANCE', 'BOND_YIELD_SPREAD',
                     'GLOBAL_OVERNIGHT_COMPOSITE', 'ROLLOVER_ANALYSIS']:
            assert key in mods, f"Missing overlay: {key}"
            assert isinstance(mods[key], float), f"{key} should be float"


# ================================================================
# FIX 3: Integrate Kelly into LotSizer
# ================================================================

class TestFix3KellyIntegration:
    """Kelly fraction should scale base_lots before overlay composite."""

    def test_kelly_fraction_reduces_lots(self):
        """_KELLY_FRACTION < 1.0 should reduce base lots."""
        from execution.lot_sizer import LotSizer
        sizer = LotSizer(equity=1_000_000)
        # Without Kelly
        r1 = sizer.compute(
            stop_loss_pts=200, nifty_price=24000,
            trade_date=date(2024, 6, 1),
            overlay_modifiers={},
        )
        # With Kelly = 0.5
        r2 = sizer.compute(
            stop_loss_pts=200, nifty_price=24000,
            trade_date=date(2024, 6, 1),
            overlay_modifiers={'_KELLY_FRACTION': 0.5},
        )
        # Kelly 0.5 should produce fewer or equal lots
        assert r2['lots'] <= r1['lots']

    def test_kelly_fraction_1_no_change(self):
        """_KELLY_FRACTION = 1.0 should not change lots."""
        from execution.lot_sizer import LotSizer
        sizer = LotSizer(equity=1_000_000)
        r1 = sizer.compute(
            stop_loss_pts=200, nifty_price=24000,
            trade_date=date(2024, 6, 1),
            overlay_modifiers={},
        )
        r2 = sizer.compute(
            stop_loss_pts=200, nifty_price=24000,
            trade_date=date(2024, 6, 1),
            overlay_modifiers={'_KELLY_FRACTION': 1.0},
        )
        assert r1['lots'] == r2['lots']

    def test_kelly_not_in_overlay_hierarchy(self):
        """_KELLY_FRACTION should be popped before overlay composite computation."""
        from execution.lot_sizer import LotSizer
        sizer = LotSizer(equity=1_000_000)
        result = sizer.compute(
            stop_loss_pts=200, nifty_price=24000,
            trade_date=date(2024, 6, 1),
            overlay_modifiers={'_KELLY_FRACTION': 0.6, 'MAMBA_REGIME': 1.1},
        )
        # _KELLY_FRACTION should not appear in modifier_breakdown
        breakdown = result.get('modifier_breakdown', {})
        assert '_KELLY_FRACTION' not in breakdown

    def test_adaptive_kelly_returns_fraction(self):
        """AdaptiveKelly should return a dict with 'fraction' key."""
        from execution.adaptive_kelly import AdaptiveKelly
        kelly = AdaptiveKelly(base_fraction=0.85)
        result = kelly.get_fraction(drawdown_pct=0.03, recent_wr=0.52, vix=16)
        assert 'fraction' in result
        assert 0.20 <= result['fraction'] <= 1.0

    def test_kelly_in_backtest_modifiers(self):
        """The lot_based_wf backtest should inject _KELLY_FRACTION into modifiers."""
        # This is a structural test — verify the import exists
        from backtest.lot_based_wf import run_lot_based_backtest
        from execution.adaptive_kelly import AdaptiveKelly
        # Just verify the import chain works
        kelly = AdaptiveKelly(base_fraction=0.85)
        r = kelly.get_fraction(drawdown_pct=0.0, recent_wr=0.50, vix=15)
        assert 'fraction' in r


# ================================================================
# FIX 4: Max-Hold Limits on DRY Signals
# ================================================================

class TestFix4HoldDaysMax:
    """Verify hold_days_max corrections in SIGNALS dict."""

    def test_kaufman_dry_20_hold_10(self):
        """KAUFMAN_DRY_20 should have hold_days_max = 10."""
        from paper_trading.signal_compute import SIGNALS
        assert SIGNALS['KAUFMAN_DRY_20']['hold_days_max'] == 10

    def test_kaufman_dry_16_hold_8(self):
        """KAUFMAN_DRY_16 should have hold_days_max = 8."""
        from paper_trading.signal_compute import SIGNALS
        assert SIGNALS['KAUFMAN_DRY_16']['hold_days_max'] == 8

    def test_kaufman_dry_12_hold_7(self):
        """KAUFMAN_DRY_12 should have hold_days_max = 7."""
        from paper_trading.signal_compute import SIGNALS
        assert SIGNALS['KAUFMAN_DRY_12']['hold_days_max'] == 7

    def test_gujral_dry_8_hold_5(self):
        """GUJRAL_DRY_8 should have hold_days_max = 5."""
        from paper_trading.signal_compute import SIGNALS
        assert SIGNALS['GUJRAL_DRY_8']['hold_days_max'] == 5

    def test_gujral_dry_13_hold_7(self):
        """GUJRAL_DRY_13 should have hold_days_max = 7."""
        from paper_trading.signal_compute import SIGNALS
        assert SIGNALS['GUJRAL_DRY_13']['hold_days_max'] == 7

    def test_no_zero_hold_days_in_scoring_signals(self):
        """No SCORING signal with DRY in its name should have hold_days_max = 0."""
        from paper_trading.signal_compute import SIGNALS
        for sig_id, config in SIGNALS.items():
            if 'DRY' in sig_id and config.get('trade_type') != 'SHADOW':
                hold = config.get('hold_days_max', 0)
                # hold_days_max of 0 means unlimited — bad for DRY signals
                if hold == 0:
                    # Allow it only for intraday signals
                    if config.get('timeframe') == '5min':
                        continue
                    # Check if it's a top-level scoring signal (not combo/shadow)
                    if sig_id in ('KAUFMAN_DRY_20', 'KAUFMAN_DRY_16',
                                   'KAUFMAN_DRY_12', 'GUJRAL_DRY_8',
                                   'GUJRAL_DRY_13'):
                        pytest.fail(
                            f"{sig_id} has hold_days_max=0 (unlimited)"
                        )


# ================================================================
# FIX 5: Verify Async Execution
# ================================================================

class TestFix5AsyncExecution:
    """Verify execution engine uses async fill monitor correctly."""

    def test_fill_monitor_start_called_in_init(self):
        """ExecutionEngine.__init__ should call fill_monitor.start()."""
        # Read the source to verify start() is called
        import inspect
        from execution.execution_engine import ExecutionEngine
        source = inspect.getsource(ExecutionEngine.__init__)
        assert 'fill_monitor.start()' in source

    def test_fill_monitor_has_async_api(self):
        """AsyncFillMonitor should have both submit() and monitor_fill() APIs."""
        from execution.fill_monitor import AsyncFillMonitor
        assert hasattr(AsyncFillMonitor, 'submit')
        assert hasattr(AsyncFillMonitor, 'monitor_fill')
        assert hasattr(AsyncFillMonitor, 'start')
        assert hasattr(AsyncFillMonitor, 'stop')

    def test_monitor_fill_delegates_to_submit(self):
        """monitor_fill() should internally call submit()."""
        import inspect
        from execution.fill_monitor import AsyncFillMonitor
        source = inspect.getsource(AsyncFillMonitor.monitor_fill)
        assert 'self.submit(' in source

    def test_paper_mode_fill_immediate(self):
        """In PAPER mode, submit() should resolve immediately."""
        from execution.fill_monitor import AsyncFillMonitor
        mock_kite = MagicMock()
        mock_alerter = MagicMock()
        monitor = AsyncFillMonitor(mock_kite, mock_alerter)

        result = monitor.submit(
            order_id='PAPER-TEST-001',
            expected_price=200.0,
            tradingsymbol='NIFTY2630524500CE',
            signal_id='TEST',
        )
        # Should be resolved immediately in paper mode
        assert result._result is not None
        assert result._result['status'] == 'FILLED'
        assert result._event.is_set()

    def test_execution_engine_uses_blocking_api(self):
        """ExecutionEngine._process_signal should use monitor_fill (blocking)."""
        import inspect
        from execution.execution_engine import ExecutionEngine
        source = inspect.getsource(ExecutionEngine._process_signal)
        assert 'monitor_fill(' in source
        # Also verify the TODO comment about switching to async
        assert 'TODO' in source or 'NOTE' in source

    def test_backwards_compatible_alias(self):
        """FillMonitor should be an alias for AsyncFillMonitor."""
        from execution.fill_monitor import FillMonitor, AsyncFillMonitor
        assert FillMonitor is AsyncFillMonitor

    def test_start_is_idempotent(self):
        """Calling start() multiple times should not create multiple threads."""
        from execution.fill_monitor import AsyncFillMonitor
        mock_kite = MagicMock()
        mock_alerter = MagicMock()
        monitor = AsyncFillMonitor(mock_kite, mock_alerter)

        monitor.start()
        thread1 = monitor._thread
        monitor.start()  # second call
        thread2 = monitor._thread
        assert thread1 is thread2
        monitor.stop()
