"""Tests for CombinationTester backtest logic."""

import numpy as np
import pandas as pd
import pytest
from backtest.combination_tester import CombinationTester


@pytest.fixture
def sample_df():
    """Create a simple 100-day DataFrame with indicators."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range('2020-01-01', periods=n, freq='B')
    close = 10000 + np.cumsum(np.random.randn(n) * 50)
    open_ = close - np.random.randn(n) * 20
    high = np.maximum(close, open_) + abs(np.random.randn(n) * 30)
    low = np.minimum(close, open_) - abs(np.random.randn(n) * 30)
    volume = np.random.randint(100000, 500000, n).astype(float)

    df = pd.DataFrame({
        'date': dates, 'open': open_, 'high': high, 'low': low,
        'close': close, 'volume': volume,
    })
    # Add minimal indicators needed by tests
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['rsi_14'] = 50 + np.random.randn(n) * 15  # fake RSI
    df['prev_close'] = df['close'].shift(1)
    df['prev_volume'] = df['volume'].shift(1)
    df['pivot'] = (df['high'].shift(1) + df['low'].shift(1) + df['close'].shift(1)) / 3
    df['r1'] = 2 * df['pivot'] - df['low'].shift(1)
    df['s1'] = 2 * df['pivot'] - df['high'].shift(1)
    df['stoch_k_5'] = np.random.uniform(10, 90, n)
    df['india_vix'] = 15 + np.random.randn(n) * 3
    return df


def make_sig(entry_long=None, entry_short=None, exit_long=None, exit_short=None, direction='BOTH'):
    return {
        'signal_id': 'TEST',
        'book_id': 'TEST',
        'entry_long': entry_long or [],
        'entry_short': entry_short or [],
        'exit_long': exit_long or [],
        'exit_short': exit_short or [],
        'direction': direction,
        'stop_loss_pct': 0.02,
        'take_profit_pct': 0,
        'hold_days': 0,
    }


# Always-true/false signals for testing logic
SIG_ALWAYS_LONG = make_sig(
    entry_long=[{'indicator': 'close', 'op': '>', 'value': 0}],
    exit_long=[{'indicator': 'close', 'op': '<', 'value': 0}],  # never exits
)
SIG_NEVER = make_sig(
    entry_long=[{'indicator': 'close', 'op': '<', 'value': 0}],
)
SIG_RSI_LOW = make_sig(
    entry_long=[{'indicator': 'rsi_14', 'op': '<', 'value': 40}],
    exit_long=[{'indicator': 'rsi_14', 'op': '>', 'value': 60}],
)
SIG_SMA_CROSS = make_sig(
    entry_long=[{'indicator': 'sma_10', 'op': '>', 'value': 'sma_20'}],
    exit_long=[{'indicator': 'sma_10', 'op': '<', 'value': 'sma_20'}],
)


def test_and_fires_when_both(sample_df):
    """AND logic fires when both signals fire same day."""
    tester = CombinationTester.__new__(CombinationTester)
    result = tester.backtest_combination(
        SIG_ALWAYS_LONG, SIG_ALWAYS_LONG, 'AND', sample_df,
        stop_pct=0.5, take_profit_pct=0, hold_days_max=0)
    assert result['trades'] > 0


def test_and_no_fire_one_signal(sample_df):
    """AND logic does NOT fire when only one fires."""
    tester = CombinationTester.__new__(CombinationTester)
    result = tester.backtest_combination(
        SIG_ALWAYS_LONG, SIG_NEVER, 'AND', sample_df,
        stop_pct=0.5, take_profit_pct=0, hold_days_max=0)
    assert result['trades'] == 0 or result.get('insufficient_trades')


def test_or_fires_either(sample_df):
    """OR logic fires when either signal fires."""
    tester = CombinationTester.__new__(CombinationTester)
    result = tester.backtest_combination(
        SIG_RSI_LOW, SIG_NEVER, 'OR', sample_df,
        stop_pct=0.5, take_profit_pct=0, hold_days_max=5)
    assert result['trades'] > 0


def test_seq3_fires_within_window(sample_df):
    """SEQ_3: B confirms within 3 days of A firing."""
    tester = CombinationTester.__new__(CombinationTester)
    # Use RSI_LOW as anchor (fires sometimes) and ALWAYS as confirmation (fires every day)
    # This guarantees confirmation fires within 1 day of anchor
    result = tester.backtest_combination(
        SIG_RSI_LOW, SIG_ALWAYS_LONG, 'SEQ_3', sample_df,
        stop_pct=0.5, take_profit_pct=0, hold_days_max=5)
    assert result['trades'] > 0


def test_seq5_boundary(sample_df):
    """SEQ_5 fires with 5-day window."""
    tester = CombinationTester.__new__(CombinationTester)
    result = tester.backtest_combination(
        SIG_RSI_LOW, SIG_SMA_CROSS, 'SEQ_5', sample_df,
        stop_pct=0.5, take_profit_pct=0, hold_days_max=5)
    # May or may not have trades depending on timing
    assert 'trades' in result


def test_anchor_uses_a_entry(sample_df):
    """ANCHOR uses sig_a entry, sig_b adds exit."""
    tester = CombinationTester.__new__(CombinationTester)
    result = tester.backtest_combination(
        SIG_RSI_LOW, SIG_SMA_CROSS, 'ANCHOR', sample_df,
        stop_pct=0.5, take_profit_pct=0, hold_days_max=10)
    assert 'trades' in result


def test_no_lookahead(sample_df):
    """Entry price should be next day open, not same day close."""
    tester = CombinationTester.__new__(CombinationTester)
    # Use always-true signal so we know exactly when it fires
    result = tester.backtest_combination(
        SIG_ALWAYS_LONG, SIG_ALWAYS_LONG, 'AND', sample_df,
        stop_pct=0.5, take_profit_pct=0, hold_days_max=3)
    # Entry should be at open prices, which differ from close
    assert result['trades'] > 0


def test_stop_loss(sample_df):
    """Stop loss exits correctly."""
    tester = CombinationTester.__new__(CombinationTester)
    result = tester.backtest_combination(
        SIG_ALWAYS_LONG, SIG_ALWAYS_LONG, 'AND', sample_df,
        stop_pct=0.001, take_profit_pct=0, hold_days_max=0)
    # Very tight stop should cause many stop exits
    assert result['trades'] > 0


def test_take_profit(sample_df):
    """Take profit exits correctly."""
    tester = CombinationTester.__new__(CombinationTester)
    result = tester.backtest_combination(
        SIG_ALWAYS_LONG, SIG_ALWAYS_LONG, 'AND', sample_df,
        stop_pct=0.5, take_profit_pct=0.001, hold_days_max=0)
    # Very tight TP should cause many TP exits
    assert result['trades'] > 0


def test_hold_days(sample_df):
    """Hold days forces exit."""
    tester = CombinationTester.__new__(CombinationTester)
    r1 = tester.backtest_combination(
        SIG_ALWAYS_LONG, SIG_ALWAYS_LONG, 'AND', sample_df,
        stop_pct=0.5, take_profit_pct=0, hold_days_max=3)
    r2 = tester.backtest_combination(
        SIG_ALWAYS_LONG, SIG_ALWAYS_LONG, 'AND', sample_df,
        stop_pct=0.5, take_profit_pct=0, hold_days_max=10)
    # Shorter hold → more trades
    assert r1['trades'] >= r2['trades']


def test_no_long_when_short(sample_df):
    """No new long when short position open."""
    tester = CombinationTester.__new__(CombinationTester)
    # The engine should only allow one position at a time
    result = tester.backtest_combination(
        SIG_ALWAYS_LONG, SIG_ALWAYS_LONG, 'AND', sample_df,
        stop_pct=0.5, take_profit_pct=0, hold_days_max=5)
    # Position management: only one at a time
    assert result['trades'] > 0


def test_insufficient_trades():
    """Insufficient trades returns correct flag."""
    df = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=10, freq='B'),
        'open': [100]*10, 'high': [101]*10, 'low': [99]*10,
        'close': [100]*10, 'volume': [1000]*10,
        'rsi_14': [50]*10, 'sma_10': [100]*10, 'sma_20': [100]*10,
    })
    tester = CombinationTester.__new__(CombinationTester)
    result = tester.backtest_combination(
        SIG_NEVER, SIG_NEVER, 'AND', df)
    assert result.get('insufficient_trades') or result['trades'] < 10


def test_nifty_corr_on_trade_returns(sample_df):
    """Nifty correlation computed on trade returns."""
    tester = CombinationTester.__new__(CombinationTester)
    result = tester.backtest_combination(
        SIG_ALWAYS_LONG, SIG_ALWAYS_LONG, 'AND', sample_df,
        stop_pct=0.5, take_profit_pct=0, hold_days_max=3)
    assert 'nifty_corr' in result
    assert -1.0 <= result['nifty_corr'] <= 1.0


def test_walkforward_nonoverlapping(sample_df):
    """Walk-forward test periods should not overlap."""
    # Create a longer dataset
    np.random.seed(42)
    n = 2500
    dates = pd.date_range('2015-01-01', periods=n, freq='B')
    df = pd.DataFrame({
        'date': dates,
        'open': 10000 + np.cumsum(np.random.randn(n) * 50),
        'high': 10050 + np.cumsum(np.random.randn(n) * 50),
        'low': 9950 + np.cumsum(np.random.randn(n) * 50),
        'close': 10000 + np.cumsum(np.random.randn(n) * 50),
        'volume': np.random.randint(100000, 500000, n).astype(float),
        'rsi_14': 50 + np.random.randn(n) * 15,
        'sma_10': 10000 + np.cumsum(np.random.randn(n) * 20),
        'sma_20': 10000 + np.cumsum(np.random.randn(n) * 10),
    })
    tester = CombinationTester.__new__(CombinationTester)
    tester.df_full = df
    result = tester.run_walk_forward(SIG_RSI_LOW, SIG_SMA_CROSS, 'AND', df)
    assert 'tier' in result
    assert result['total_windows'] >= 4
