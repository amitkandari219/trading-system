"""Tests for PCR contrarian signals."""

import pytest
import pandas as pd
import numpy as np
from paper_trading.pcr_signals import PCRSignals


@pytest.fixture
def pcr():
    return PCRSignals()


def make_df(pcr_oi, pcr_zscore=None):
    d = {'pcr_oi': [pcr_oi]}
    if pcr_zscore is not None:
        d['pcr_zscore'] = [pcr_zscore]
    return pd.DataFrame(d)


def test_extreme_fear_long(pcr):
    """PCR > 1.5 with high z-score generates LONG."""
    df = make_df(pcr_oi=1.6, pcr_zscore=2.0)
    signals = pcr.compute(df)
    assert len(signals) == 1
    assert signals[0]['signal_id'] == 'PCR_EXTREME_FEAR'
    assert signals[0]['direction'] == 'LONG'


def test_extreme_greed_short(pcr):
    """PCR < 0.7 with low z-score generates SHORT."""
    df = make_df(pcr_oi=0.6, pcr_zscore=-2.0)
    signals = pcr.compute(df)
    assert len(signals) == 1
    assert signals[0]['signal_id'] == 'PCR_EXTREME_GREED'
    assert signals[0]['direction'] == 'SHORT'


def test_neutral_no_signal(pcr):
    """PCR in neutral zone (0.8-1.3) generates nothing."""
    df = make_df(pcr_oi=1.0, pcr_zscore=0.5)
    signals = pcr.compute(df)
    assert len(signals) == 0


def test_zscore_filters(pcr):
    """High PCR but low z-score = no signal (not extreme enough)."""
    df = make_df(pcr_oi=1.6, pcr_zscore=0.5)  # PCR high but z-score low
    signals = pcr.compute(df)
    assert len(signals) == 0


def test_required_fields(pcr):
    """PCR signal has all required fields."""
    df = make_df(pcr_oi=1.6, pcr_zscore=2.0)
    signals = pcr.compute(df)
    for s in signals:
        assert 'signal_id' in s
        assert 'direction' in s
        assert 'confidence' in s
        assert 'source' in s
        assert 'stop_pct' in s


def test_no_pcr_column(pcr):
    """No signal when pcr_oi column missing."""
    df = pd.DataFrame({'close': [23000]})
    signals = pcr.compute(df)
    assert len(signals) == 0


def test_nan_pcr_returns_empty(pcr):
    """NaN pcr_oi returns empty list."""
    df = make_df(pcr_oi=float('nan'), pcr_zscore=2.0)
    signals = pcr.compute(df)
    assert len(signals) == 0


def test_bhavcopy_parses_nifty_options():
    """Bhavcopy download parses NIFTY options correctly."""
    from data.pcr_loader import parse_bhavcopy_df

    # Simulate raw bhavcopy data
    rows = [
        {'SYMBOL': 'NIFTY', 'OPTION_TYP': 'CE', 'OPEN_INT': 1000000},
        {'SYMBOL': 'NIFTY', 'OPTION_TYP': 'CE', 'OPEN_INT': 500000},
        {'SYMBOL': 'NIFTY', 'OPTION_TYP': 'PE', 'OPEN_INT': 800000},
        {'SYMBOL': 'NIFTY', 'OPTION_TYP': 'PE', 'OPEN_INT': 700000},
        {'SYMBOL': 'BANKNIFTY', 'OPTION_TYP': 'CE', 'OPEN_INT': 2000000},
        {'SYMBOL': 'BANKNIFTY', 'OPTION_TYP': 'PE', 'OPEN_INT': 3000000},
    ]
    df = pd.DataFrame(rows)
    result = parse_bhavcopy_df(df)
    assert result is not None
    # PCR = (800000 + 700000) / (1000000 + 500000) = 1500000 / 1500000 = 1.0
    assert result['pcr_oi'] == 1.0
    assert result['total_pe_oi'] == 1500000
    assert result['total_ce_oi'] == 1500000
