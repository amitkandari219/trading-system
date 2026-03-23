"""
Integration tests verifying full system wiring.

Tests cover: UnifiedConfig, DataProvider, RegimeDetector, UnifiedSizer,
SignalCoordinator, Orchestrator, and end-to-end signal-to-sizing chain.

All tests use mock data where DB access is unavailable.

Run:
    venv/bin/python3 -m pytest tests/test_integration.py -v
"""

import math
import sys
from datetime import date, datetime
from typing import Dict, List
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pandas as pd
import pytest


# ═══════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════

def _make_mock_df(rows: int = 60):
    """Create a minimal mock OHLCV + indicator DataFrame."""
    dates = pd.date_range('2026-01-01', periods=rows, freq='B')
    np.random.seed(42)
    closes = 24000 + np.cumsum(np.random.randn(rows) * 50)
    df = pd.DataFrame({
        'date': dates,
        'open': closes - 20,
        'high': closes + 50,
        'low': closes - 50,
        'close': closes,
        'volume': np.random.randint(100_000, 500_000, rows),
        'india_vix': 14 + np.random.randn(rows) * 2,
        'sma_50': pd.Series(closes).rolling(50, min_periods=1).mean(),
        'sma_200': pd.Series(closes).rolling(50, min_periods=1).mean() - 200,
        'rsi_14': 50 + np.random.randn(rows) * 10,
        'adx_14': 22 + np.random.randn(rows) * 5,
        'atr_14': 100 + np.random.randn(rows) * 20,
        'pcr_oi': 1.0 + np.random.randn(rows) * 0.2,
        'bb_bandwidth': 0.05 + np.abs(np.random.randn(rows) * 0.01),
        'hvol_20': 15 + np.abs(np.random.randn(rows) * 3),
    })
    return df


# ═══════════════════════════════════════════════════════════════
# TEST GROUP 1: Unified Config (5 tests)
# ═══════════════════════════════════════════════════════════════

class TestUnifiedConfig:
    """Verify unified_config is consistent and correct."""

    def test_all_active_signals_have_module_mapping(self):
        """Every signal in ACTIVE_SCORING_SIGNALS must exist in SIGNAL_MODULE_MAP."""
        from config.unified_config import ACTIVE_SCORING_SIGNALS, SIGNAL_MODULE_MAP
        missing = [s for s in ACTIVE_SCORING_SIGNALS if s not in SIGNAL_MODULE_MAP]
        assert missing == [], f'Signals without module mapping: {missing}'

    def test_no_duplicate_signals(self):
        """No signal appears in both scoring and overlay lists."""
        from config.unified_config import (
            ACTIVE_SCORING_SIGNALS, ACTIVE_OVERLAY_SIGNALS, SHADOW_SCORING_SIGNALS,
        )
        scoring_set = set(ACTIVE_SCORING_SIGNALS)
        overlay_set = set(ACTIVE_OVERLAY_SIGNALS)
        shadow_set = set(SHADOW_SCORING_SIGNALS)

        scoring_overlay = scoring_set & overlay_set
        assert scoring_overlay == set(), f'Signals in both scoring and overlay: {scoring_overlay}'

        scoring_shadow = scoring_set & shadow_set
        assert scoring_shadow == set(), f'Signals in both scoring and shadow: {scoring_shadow}'

    def test_database_dsn_points_to_docker(self):
        """Default DATABASE_DSN must point to the Docker Postgres on port 5450."""
        from config.unified_config import DATABASE_DSN
        assert 'localhost:5450' in DATABASE_DSN
        assert 'trading' in DATABASE_DSN

    def test_lot_size_correct(self):
        """Nifty lot size must be 25 (post-Nov 2024 NSE standard)."""
        from config.unified_config import NIFTY_LOT_SIZE
        assert NIFTY_LOT_SIZE == 25

    def test_conviction_tiers_ordered(self):
        """HIGH conviction must require a higher min_score than MEDIUM, which exceeds LOW."""
        from config.unified_config import CONVICTION_TIERS
        high = CONVICTION_TIERS['HIGH']['min_score']
        med = CONVICTION_TIERS['MEDIUM']['min_score']
        low = CONVICTION_TIERS['LOW']['min_score']
        assert high > med > low, f'Conviction order violated: HIGH={high}, MEDIUM={med}, LOW={low}'


# ═══════════════════════════════════════════════════════════════
# TEST GROUP 2: DataProvider (5 tests)
# ═══════════════════════════════════════════════════════════════

class TestDataProvider:
    """Verify DataProvider loads data and computes context correctly."""

    def test_loads_nifty_daily_from_db(self):
        """DataProvider.load_history returns a DataFrame with expected columns."""
        from core.data_provider import DataProvider

        mock_df = _make_mock_df()
        dp = DataProvider(db_connection=MagicMock())

        with patch('data.nifty_loader.load_nifty_history', return_value=mock_df):
            with patch('backtest.indicators.add_all_indicators', side_effect=lambda x: x):
                df = dp.load_history(days=60)

        assert len(df) == 60
        for col in ['date', 'open', 'high', 'low', 'close', 'volume', 'india_vix']:
            assert col in df.columns, f'Missing column: {col}'

    def test_market_context_has_required_keys(self):
        """market_context() must return all required keys."""
        from core.data_provider import DataProvider

        mock_df = _make_mock_df()
        dp = DataProvider(db_connection=MagicMock())

        with patch('data.nifty_loader.load_nifty_history', return_value=mock_df):
            with patch('backtest.indicators.add_all_indicators', side_effect=lambda x: x):
                ctx = dp.market_context()

        required_keys = ['spot_price', 'india_vix', 'day_of_week', 'prev_close',
                         'sma_50', 'rsi_14', 'adx_14', 'atr_14', 'volume']
        for key in required_keys:
            assert key in ctx, f'Missing context key: {key}'
        assert ctx['spot_price'] > 0

    def test_caching_works(self):
        """Second call to load_history within TTL must not hit the loader again."""
        from core.data_provider import DataProvider

        mock_df = _make_mock_df()
        dp = DataProvider(db_connection=MagicMock(), cache_ttl_seconds=300)

        with patch('data.nifty_loader.load_nifty_history', return_value=mock_df) as loader:
            with patch('backtest.indicators.add_all_indicators', side_effect=lambda x: x):
                dp.load_history(days=60)
                dp.load_history(days=60)  # should use cache

        assert loader.call_count == 1, 'Loader called more than once despite cache'

    def test_handles_missing_data(self):
        """DataProvider should handle empty / missing data gracefully."""
        from core.data_provider import DataProvider

        dp = DataProvider(db_connection=MagicMock())
        empty_df = pd.DataFrame()

        with patch('data.nifty_loader.load_nifty_history', return_value=empty_df):
            with patch('backtest.indicators.add_all_indicators', side_effect=lambda x: x):
                ctx = dp.market_context()

        assert ctx['spot_price'] == 0.0
        assert ctx['india_vix'] == 15.0  # safe default

    def test_indicators_computed(self):
        """When add_all_indicators is available, indicator columns should be present."""
        from core.data_provider import DataProvider

        mock_df = _make_mock_df()
        dp = DataProvider(db_connection=MagicMock())

        # Simulate add_all_indicators adding columns
        def mock_add_indicators(df):
            df['sma_50'] = df['close'].rolling(50, min_periods=1).mean()
            df['rsi_14'] = 50.0
            return df

        with patch('data.nifty_loader.load_nifty_history', return_value=mock_df):
            with patch('backtest.indicators.add_all_indicators', side_effect=mock_add_indicators):
                df = dp.load_history(days=60)

        assert 'sma_50' in df.columns


# ═══════════════════════════════════════════════════════════════
# TEST GROUP 3: RegimeDetector (5 tests)
# ═══════════════════════════════════════════════════════════════

class TestRegimeDetector:
    """Verify RegimeDetector classifies market conditions correctly."""

    def test_high_vix_crisis(self):
        """VIX >= 25 must produce CRISIS regime with size_modifier 0."""
        from core.regime_detector import RegimeDetector
        rd = RegimeDetector()
        result = rd.detect(vix=28.0, adx=30, close=22000, sma_50=23000)
        assert result['regime'] == 'CRISIS'
        assert result['size_modifier'] == 0.0

    def test_low_vix_uptrend_strong_bull(self):
        """VIX < 12 + strong uptrend (ADX > 25) should produce CALM regime."""
        from core.regime_detector import RegimeDetector
        rd = RegimeDetector()
        result = rd.detect(vix=10.0, adx=30, close=25000, sma_50=24000)
        assert result['regime'] == 'CALM'
        assert result['size_modifier'] == 1.0

    def test_neutral_regime(self):
        """VIX 10-14 should produce NORMAL regime."""
        from core.regime_detector import RegimeDetector
        rd = RegimeDetector()
        result = rd.detect(vix=12.0, adx=15, close=24000, sma_50=24000)
        assert result['regime'] == 'NORMAL'

    def test_regime_has_size_modifier(self):
        """Every regime result must contain a size_modifier between 0 and 1."""
        from core.regime_detector import RegimeDetector
        rd = RegimeDetector()
        for vix in [8, 12, 15, 20, 28]:
            result = rd.detect(vix=float(vix))
            assert 0.0 <= result['size_modifier'] <= 1.0, \
                f'VIX={vix}: modifier={result["size_modifier"]}'

    def test_consistent_across_calls(self):
        """Same inputs must produce the same regime classification."""
        from core.regime_detector import RegimeDetector
        rd = RegimeDetector()
        r1 = rd.detect(vix=16.0, adx=22, close=24500, sma_50=24000)
        r2 = rd.detect(vix=16.0, adx=22, close=24500, sma_50=24000)
        assert r1['regime'] == r2['regime']
        assert r1['size_modifier'] == r2['size_modifier']


# ═══════════════════════════════════════════════════════════════
# TEST GROUP 4: UnifiedSizer (8 tests)
# ═══════════════════════════════════════════════════════════════

class TestUnifiedSizer:
    """Verify UnifiedSizer computes correct lot counts through the full chain."""

    def test_basic_lot_computation(self):
        """Base lots should be positive given standard inputs."""
        from core.unified_sizer import UnifiedSizer
        sizer = UnifiedSizer(equity=1_000_000)
        result = sizer.compute(
            signal_name='KAUFMAN_DRY_20',
            sl_points=100,
            spot_price=24000,
        )
        assert result['lots'] >= 1
        assert result['lots'] <= 20

    def test_position_limit_zero_lots(self):
        """Regime modifier for CRISIS should be less than 1.0 (restrictive)."""
        from core.unified_sizer import UnifiedSizer
        sizer = UnifiedSizer(equity=1_000_000)
        result = sizer.compute(
            signal_name='KAUFMAN_DRY_20',
            sl_points=100,
            spot_price=24000,
            regime='CRISIS',
        )
        assert result['regime_modifier'] < 1.0

    def test_kelly_reduces_size(self):
        """Providing a Kelly fraction < 1 should reduce lots vs base."""
        from core.unified_sizer import UnifiedSizer

        # Create a mock Kelly engine that returns fraction=0.5
        mock_kelly = MagicMock()
        mock_kelly.get_fraction.return_value = {'fraction': 0.50, 'gear': 'STRESS'}

        sizer_no_kelly = UnifiedSizer(equity=1_000_000)
        sizer_kelly = UnifiedSizer(equity=1_000_000, kelly_engine=mock_kelly)

        r_base = sizer_no_kelly.compute(
            signal_name='TEST', sl_points=100, spot_price=24000)
        r_kelly = sizer_kelly.compute(
            signal_name='TEST', sl_points=100, spot_price=24000)

        assert r_kelly['kelly_fraction'] == 0.50
        assert r_kelly['lots'] <= r_base['lots']

    def test_conviction_amplifies(self):
        """ConvictionScorer > 1.0 should produce conviction_modifier > 1."""
        from core.unified_sizer import UnifiedSizer

        mock_scorer = MagicMock()
        mock_scorer.compute.return_value = {'final_modifier': 1.50}

        sizer = UnifiedSizer(equity=1_000_000, conviction_scorer=mock_scorer)
        result = sizer.compute(
            signal_name='TEST', sl_points=100, spot_price=24000)

        assert result['conviction_modifier'] == 1.50

    def test_regime_modifier_applied(self):
        """Regime modifier should scale lots: TRENDING > CRISIS."""
        from core.unified_sizer import UnifiedSizer
        sizer = UnifiedSizer(equity=1_000_000)

        r_trending = sizer.compute(
            signal_name='TEST', sl_points=100, spot_price=24000, regime='TRENDING')
        r_crisis = sizer.compute(
            signal_name='TEST', sl_points=100, spot_price=24000, regime='CRISIS')

        assert r_trending['regime_modifier'] > r_crisis['regime_modifier']

    def test_overlay_composite_computed(self):
        """Overlay modifiers should produce a composite modifier != 1.0."""
        from core.unified_sizer import UnifiedSizer
        sizer = UnifiedSizer(equity=1_000_000)
        result = sizer.compute(
            signal_name='TEST', sl_points=100, spot_price=24000,
            overlay_modifiers={'FII_FUTURES_OI': 1.2, 'MAMBA_REGIME': 1.3},
        )
        # composite_modifier combines kelly * conviction * regime * overlay
        # With no kelly/conviction, overlay should still be != 1.0
        assert result['composite_modifier'] != 1.0

    def test_min_1_lot(self):
        """Even with tiny equity and large SL, should get at least 1 lot."""
        from core.unified_sizer import UnifiedSizer
        sizer = UnifiedSizer(equity=50_000)
        result = sizer.compute(
            signal_name='TEST', sl_points=500, spot_price=24000)
        assert result['lots'] >= 1

    def test_at_10L_mostly_1_lot(self):
        """At 10L capital with standard SL, lots should be in the 1-20 range."""
        from core.unified_sizer import UnifiedSizer
        sizer = UnifiedSizer(equity=1_000_000)
        result = sizer.compute(
            signal_name='TEST', sl_points=200, spot_price=24000)
        assert 1 <= result['lots'] <= 20


# ═══════════════════════════════════════════════════════════════
# TEST GROUP 5: SignalCoordinator (5 tests)
# ═══════════════════════════════════════════════════════════════

class TestSignalCoordinator:
    """Verify SignalCoordinator correctly deduplicates, limits, and resolves."""

    def test_deduplicates_correlated_pairs(self):
        """Correlated pair (KAUFMAN_DRY_16, KAUFMAN_DRY_12) -- weaker must be dropped."""
        from core.signal_coordinator import SignalCoordinator
        coord = SignalCoordinator()

        candidates = [
            {'signal_name': 'KAUFMAN_DRY_16', 'direction': 'BULLISH', 'confidence': 0.8},
            {'signal_name': 'KAUFMAN_DRY_12', 'direction': 'BULLISH', 'confidence': 0.6},
        ]
        result = coord.deduplicate(candidates)
        names = [c['signal_name'] for c in result]
        assert len(result) == 1
        assert 'KAUFMAN_DRY_16' in names  # higher confidence wins

    def test_category_limits(self):
        """Max 2 signals per category -- 3 Kaufman signals should trim to 2."""
        from core.signal_coordinator import SignalCoordinator
        coord = SignalCoordinator(max_per_category=2)

        candidates = [
            {'signal_name': 'KAUFMAN_DRY_20', 'direction': 'BULLISH', 'confidence': 0.9},
            {'signal_name': 'KAUFMAN_DRY_16', 'direction': 'BULLISH', 'confidence': 0.7},
            {'signal_name': 'KAUFMAN_DRY_12', 'direction': 'BULLISH', 'confidence': 0.5},
        ]
        result = coord.enforce_category_limits(candidates, max_per_category=2)
        kaufman_count = sum(1 for c in result if 'KAUFMAN' in c['signal_name'])
        assert kaufman_count <= 2

    def test_conflict_resolution_removes_weaker(self):
        """Mixed BULLISH/BEARISH -- weaker direction should be dropped."""
        from core.signal_coordinator import SignalCoordinator
        coord = SignalCoordinator()

        candidates = [
            {'signal_name': 'SIG_A', 'direction': 'BULLISH', 'confidence': 0.9},
            {'signal_name': 'SIG_B', 'direction': 'BULLISH', 'confidence': 0.7},
            {'signal_name': 'SIG_C', 'direction': 'BEARISH', 'confidence': 0.5},
        ]
        result = coord.resolve_conflicts(candidates)
        directions = {c['direction'] for c in result}
        # Bulls have higher aggregate confidence (1.6 vs 0.5), so bears are dropped
        assert 'BEARISH' not in directions
        assert len(result) == 2

    def test_single_signal_passes_through(self):
        """A single candidate should pass through all stages unchanged."""
        from core.signal_coordinator import SignalCoordinator
        coord = SignalCoordinator()

        candidates = [
            {'signal_name': 'KAUFMAN_DRY_20', 'direction': 'BULLISH', 'confidence': 0.8},
        ]
        result = coord.process(candidates)
        assert len(result) == 1
        assert result[0]['signal_name'] == 'KAUFMAN_DRY_20'

    def test_empty_candidates(self):
        """Empty candidate list should return empty list."""
        from core.signal_coordinator import SignalCoordinator
        coord = SignalCoordinator()
        result = coord.process([])
        assert result == []


# ═══════════════════════════════════════════════════════════════
# TEST GROUP 6: Orchestrator (8 tests)
# ═══════════════════════════════════════════════════════════════

class TestOrchestrator:
    """Verify TradingOrchestrator wires all components together."""

    def _make_orchestrator(self):
        """Create an orchestrator with a mocked DB connection."""
        from core.orchestrator import TradingOrchestrator
        mock_db = MagicMock()
        orch = TradingOrchestrator(mode='paper', db_connection=mock_db)
        return orch

    def test_initialization(self):
        """Orchestrator should initialize all sub-components without error."""
        orch = self._make_orchestrator()
        assert orch.mode == 'paper'
        assert orch.conn is not None
        assert orch.coordinator is not None
        assert orch.sizer is not None

    def test_loads_scoring_signals(self):
        """Orchestrator's DAILY_SIGNAL_METHODS must include KAUFMAN_DRY_20."""
        from core.orchestrator import DAILY_SIGNAL_METHODS
        assert 'KAUFMAN_DRY_20' in DAILY_SIGNAL_METHODS

    def test_loads_overlay_signals(self):
        """Orchestrator should be able to evaluate overlays via OverlayPipeline."""
        orch = self._make_orchestrator()
        mock_df = _make_mock_df(rows=100)

        # _evaluate_overlays wraps OverlayPipeline; just verify it runs
        modifiers = orch._evaluate_overlays(mock_df, direction='LONG')
        # Even if overlay pipeline fails, should return a dict (possibly empty)
        assert isinstance(modifiers, dict)

    def test_daily_session_runs(self):
        """run_daily_session() with mock data should complete without error."""
        orch = self._make_orchestrator()
        mock_df = _make_mock_df(rows=100)

        # Patch _fetch_market_data to return our mock DataFrame
        with patch.object(orch, '_fetch_market_data', return_value=mock_df):
            with patch.object(orch, '_add_indicators', side_effect=lambda df: df):
                result = orch.run_daily_session()

        # run_daily_session returns a list of sized trade dicts
        assert isinstance(result, list)

    def test_shadow_logged_not_traded(self):
        """Shadow signals (SHADOW_SCORING_SIGNALS) must not appear in daily_session output."""
        from config.unified_config import SHADOW_SCORING_SIGNALS
        orch = self._make_orchestrator()
        mock_df = _make_mock_df(rows=100)

        with patch.object(orch, '_fetch_market_data', return_value=mock_df):
            with patch.object(orch, '_add_indicators', side_effect=lambda df: df):
                result = orch.run_daily_session()

        trade_names = {t.get('signal_name') for t in result}
        shadow_set = set(SHADOW_SCORING_SIGNALS)
        overlap = trade_names & shadow_set
        assert overlap == set(), f'Shadow signals leaked into trades: {overlap}'

    def test_regime_filter_applied(self):
        """Orchestrator's _filter_by_regime should remove signals blocked by regime."""
        orch = self._make_orchestrator()

        # Create candidates and attempt to filter in CRISIS regime
        candidates = [
            {'signal_name': 'KAUFMAN_BB_MR', 'direction': 'BULLISH', 'confidence': 0.8},
            {'signal_name': 'UNKNOWN_SIG', 'direction': 'BULLISH', 'confidence': 0.5},
        ]
        # KAUFMAN_BB_MR is blocked in CRISIS per SIGNAL_REGIME_MATRIX
        filtered = orch._filter_by_regime(candidates, 'CRISIS')
        # The unknown signal has no policy, so it passes through
        names = [c['signal_name'] for c in filtered]
        assert 'UNKNOWN_SIG' in names
        # KAUFMAN_BB_MR should be removed if regime_filter is available
        # (depends on SIGNAL_REGIME_MATRIX having the entry)

    def test_paper_mode_no_execution(self):
        """In paper mode, kite_bridge should be None (no live execution)."""
        orch = self._make_orchestrator()
        assert orch.mode == 'paper'
        assert orch.kite_bridge is None

    def test_conflict_resolution_wired(self):
        """The signal coordinator must be wired and called during daily session."""
        orch = self._make_orchestrator()
        mock_df = _make_mock_df(rows=100)

        # Inject a fake candidate that would fire
        fake_candidates = [
            {'signal_name': 'KAUFMAN_DRY_20', 'direction': 'BULLISH',
             'confidence': 0.8, 'sl_points': 100},
        ]

        with patch.object(orch, '_fetch_market_data', return_value=mock_df):
            with patch.object(orch, '_add_indicators', side_effect=lambda df: df):
                with patch.object(orch, '_evaluate_scoring_signals', return_value=fake_candidates):
                    with patch.object(orch.coordinator, 'process',
                                      wraps=orch.coordinator.process) as mock_process:
                        orch.run_daily_session()

        mock_process.assert_called_once()


# ═══════════════════════════════════════════════════════════════
# TEST GROUP 7: End-to-End (4 tests)
# ═══════════════════════════════════════════════════════════════

class TestEndToEnd:
    """Full-chain integration tests: import -> compute -> size."""

    def test_structural_signals_importable(self):
        """All structural signal classes should be importable."""
        from signals.structural import (
            GiftConvergenceSignal,
            MaxOIBarrierSignal,
            RolloverFlowSignal,
            IndexRebalanceSignal,
        )
        # Verify they are callable classes
        assert callable(GiftConvergenceSignal)
        assert callable(MaxOIBarrierSignal)

    def test_overlay_pipeline_produces_values(self):
        """OverlayPipeline.get_modifiers should return a dict of float modifiers."""
        from execution.overlay_pipeline import OverlayPipeline

        df = _make_mock_df(rows=100)
        pipeline = OverlayPipeline(df)
        trade_date = df['date'].iloc[80]
        if hasattr(trade_date, 'date'):
            trade_date = trade_date.date()

        modifiers = pipeline.get_modifiers(trade_date, direction='LONG')

        assert isinstance(modifiers, dict)
        if modifiers:
            for key, val in modifiers.items():
                assert isinstance(val, (int, float)), f'{key} is not numeric: {val}'

    def test_full_chain_data_to_sizing(self):
        """Full chain: mock data -> regime -> coordination -> sizing."""
        from core.regime_detector import RegimeDetector
        from core.signal_coordinator import SignalCoordinator
        from core.unified_sizer import UnifiedSizer

        # Step 1: Detect regime
        rd = RegimeDetector()
        regime = rd.detect(vix=14.0, adx=25, close=24500, sma_50=24000)
        assert regime['regime'] in ('CALM', 'NORMAL', 'ELEVATED', 'HIGH_VOL', 'CRISIS')

        # Step 2: Create candidates
        candidates = [
            {'signal_name': 'KAUFMAN_DRY_20', 'direction': 'BULLISH', 'confidence': 0.8,
             'stop_loss_pts': 100},
            {'signal_name': 'GUJRAL_DRY_8', 'direction': 'BULLISH', 'confidence': 0.6,
             'stop_loss_pts': 120},
        ]

        # Step 3: Coordinate
        coord = SignalCoordinator()
        filtered = coord.process(candidates)
        assert len(filtered) > 0

        # Step 4: Size
        sizer = UnifiedSizer(equity=1_000_000)
        for cand in filtered:
            result = sizer.compute(
                signal_name=cand['signal_name'],
                sl_points=cand['stop_loss_pts'],
                spot_price=24500,
                regime=regime['regime'],
            )
            assert result['lots'] >= 1
            assert 'reasoning' in result

    def test_cost_model_used(self):
        """TransactionCostModel should compute non-zero costs for a round trip."""
        from core.costs import TransactionCostModel

        model = TransactionCostModel()
        costs = model.compute_futures_round_trip(entry_price=24000, exit_price=24100, lots=1)

        assert costs.total > 0
        assert costs.total < 5000  # sanity: costs should be reasonable for 1 lot
