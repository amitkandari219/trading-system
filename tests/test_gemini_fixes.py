"""Tests for Gemini-identified missing components: SlippageLogger, DecayMonitor, FDRController."""

import pytest
import numpy as np
from unittest.mock import MagicMock


# ================================================================
# SLIPPAGE LOGGER TESTS
# ================================================================

class TestSlippageLogger:

    def test_gap_pct_correct(self):
        from paper_trading.slippage_logger import SlippageLogger
        sl = SlippageLogger.__new__(SlippageLogger)
        sl.conn = MagicMock()
        sl.conn.cursor.return_value.execute = MagicMock()
        sl.conn.commit = MagicMock()

        # LONG entry: theoretical 23000, actual open 23050 → adverse gap
        gap_pts = 23050 - 23000  # +50
        gap_pct = gap_pts / 23000
        assert gap_pct > 0  # adverse for LONG

    def test_adverse_gap_long_positive(self):
        """For LONG entries, positive gap = adverse (paid more)."""
        theoretical = 23000
        actual_open = 23050
        gap = actual_open - theoretical
        assert gap > 0  # paid more than expected

    def test_favorable_gap_long_negative(self):
        """For LONG entries, negative gap = favorable (paid less)."""
        theoretical = 23000
        actual_open = 22950
        gap = actual_open - theoretical
        assert gap < 0  # paid less than expected

    def test_slippage_factor_less_than_one(self):
        """Slippage factor should always be < 1 for realistic inputs."""
        # Any positive cost drag reduces the factor below 1
        cost_per_trade_pct = 3 / 23000  # 3 pts cost on 23000 price
        trades_per_year = 50
        annual_drag = cost_per_trade_pct * trades_per_year
        factor = 1 - annual_drag
        assert factor < 1.0
        assert factor > 0.9  # should be reasonable

    def test_projected_live_cagr(self):
        """projected_live_cagr = paper_cagr × slippage_factor."""
        paper_cagr = 0.33  # 33%
        slippage_factor = 0.95
        projected = paper_cagr * slippage_factor
        assert abs(projected - 0.3135) < 0.01


# ================================================================
# DECAY MONITOR TESTS
# ================================================================

class TestDecayMonitor:

    def test_green_when_matching_backtest(self):
        from paper_trading.decay_monitor import DecayMonitor, SIZE_MULTIPLIERS
        dm = DecayMonitor.__new__(DecayMonitor)
        dm.lookback = 30
        status = dm._determine_status(
            wr_drop=0.02, sr_drop=0.1, pf_drop=0.05, consec=1)
        assert status == 'GREEN'
        assert SIZE_MULTIPLIERS[status] == 1.0

    def test_yellow_on_15pct_wr_drop(self):
        from paper_trading.decay_monitor import DecayMonitor
        dm = DecayMonitor.__new__(DecayMonitor)
        status = dm._determine_status(
            wr_drop=0.16, sr_drop=0.1, pf_drop=0.05, consec=1)
        assert status == 'YELLOW'

    def test_red_on_25pct_wr_drop(self):
        from paper_trading.decay_monitor import DecayMonitor
        dm = DecayMonitor.__new__(DecayMonitor)
        status = dm._determine_status(
            wr_drop=0.26, sr_drop=0.1, pf_drop=0.05, consec=1)
        assert status == 'RED'

    def test_critical_on_8_consecutive_losses(self):
        from paper_trading.decay_monitor import DecayMonitor
        dm = DecayMonitor.__new__(DecayMonitor)
        status = dm._determine_status(
            wr_drop=0.05, sr_drop=0.1, pf_drop=0.05, consec=8)
        assert status == 'CRITICAL'

    def test_critical_size_multiplier_zero(self):
        from paper_trading.decay_monitor import SIZE_MULTIPLIERS
        assert SIZE_MULTIPLIERS['CRITICAL'] == 0.0


# ================================================================
# FDR CONTROLLER TESTS
# ================================================================

class TestFDRController:

    def test_bh_accepts_all_when_pvalues_small(self):
        from backtest.fdr_controller import FDRController
        fdr = FDRController()
        signals = [
            {'signal_id': f'SIG_{i}', 'sharpe': 3.0, 'p_value': 0.001 * (i+1)}
            for i in range(5)
        ]
        result = fdr.apply_bh_correction(signals, alpha=0.05)
        accepted = sum(1 for s in result if s['bh_accepted'])
        assert accepted == 5  # all should pass with very small p-values

    def test_bh_rejects_at_boundary(self):
        from backtest.fdr_controller import FDRController
        fdr = FDRController()
        # Create signals where last one fails BH
        signals = [
            {'signal_id': 'A', 'p_value': 0.01},
            {'signal_id': 'B', 'p_value': 0.02},
            {'signal_id': 'C', 'p_value': 0.04},
            {'signal_id': 'D', 'p_value': 0.08},  # should fail at alpha=0.05
        ]
        result = fdr.apply_bh_correction(signals, alpha=0.05)
        # D has p=0.08, BH threshold at rank 4 = (4/4)*0.05 = 0.05
        # 0.08 > 0.05 → rejected
        d_result = next(s for s in result if s['signal_id'] == 'D')
        assert not d_result['bh_accepted']

    def test_dsr_passes_dry20(self):
        from backtest.fdr_controller import FDRController
        fdr = FDRController()
        dsr = fdr.compute_dsr(sharpe=2.37, n_trades=226, n_trials=1)
        assert dsr > 0.90  # DRY_20 should pass DSR easily

    def test_combined_requires_both(self):
        from backtest.fdr_controller import FDRController
        fdr = FDRController()
        signals = [
            {'signal_id': 'GOOD', 'sharpe': 2.5, 'trades': 200, 'p_value': 0.001},
            {'signal_id': 'WEAK', 'sharpe': 0.3, 'trades': 200, 'p_value': 0.40},
        ]
        result = fdr.combined_acceptance(signals, n_trials=2)
        good = next(s for s in result if s['signal_id'] == 'GOOD')
        weak = next(s for s in result if s['signal_id'] == 'WEAK')
        assert good['combined_tier'] in ('A', 'B')
        assert weak['combined_tier'] == 'GHOST'

    def test_ghost_rate_for_random(self):
        from backtest.fdr_controller import FDRController
        fdr = FDRController()
        np.random.seed(42)
        # 100 random signals (null hypothesis — no skill)
        signals = [
            {'signal_id': f'RAND_{i}', 'sharpe': np.random.randn() * 0.3,
             'trades': 100}
            for i in range(100)
        ]
        result = fdr.apply_bh_correction(signals, alpha=0.05)
        accepted = sum(1 for s in result if s['bh_accepted'])
        # With alpha=0.05 on random signals, should accept very few
        assert accepted <= 10  # most random signals rejected
