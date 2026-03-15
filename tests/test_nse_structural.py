"""Tests for NSE structural signals."""

import pytest
from datetime import date
from paper_trading.nse_structural import NSEStructuralSignals


@pytest.fixture
def nse():
    return NSEStructuralSignals()


def test_expiry_week_correct_thursday(nse):
    """Last Thursday of March 2026 = 26th. Expiry week = 23-26."""
    assert nse.is_expiry_week(date(2026, 3, 26))  # Thursday
    assert nse.is_expiry_week(date(2026, 3, 23))  # Monday of expiry week


def test_expiry_week_false_non_expiry(nse):
    """First week of month is NOT expiry week."""
    assert not nse.is_expiry_week(date(2026, 3, 2))
    assert not nse.is_expiry_week(date(2026, 3, 10))


def test_rbi_mpc_dates_2026(nse):
    """Returns 6 MPC dates for 2026."""
    dates = nse.get_rbi_mpc_dates(2026)
    assert len(dates) == 6
    assert dates[0] == date(2026, 2, 6)


def test_late_month_fires_day_18_19(nse):
    """Late month entry fires only on day 18-19."""
    signals_18 = nse._late_month_bias(date(2026, 3, 18))
    signals_19 = nse._late_month_bias(date(2026, 3, 19))
    signals_17 = nse._late_month_bias(date(2026, 3, 17))
    signals_20 = nse._late_month_bias(date(2026, 3, 20))

    assert any(s['signal_id'] == 'NSE_LATE_MONTH_ENTRY' for s in signals_18)
    assert any(s['signal_id'] == 'NSE_LATE_MONTH_ENTRY' for s in signals_19)
    assert not any(s['signal_id'] == 'NSE_LATE_MONTH_ENTRY' for s in signals_17)
    assert not any(s['signal_id'] == 'NSE_LATE_MONTH_ENTRY' for s in signals_20)


def test_march_effect_fires_march_1(nse):
    """March effect fires only on March 1."""
    signals = nse._monthly_seasonality(date(2026, 3, 1))
    assert any(s['signal_id'] == 'NSE_MARCH_EFFECT' for s in signals)

    signals_feb = nse._monthly_seasonality(date(2026, 2, 1))
    assert not any(s['signal_id'] == 'NSE_MARCH_EFFECT' for s in signals_feb)


def test_gap_fill_requires_threshold(nse):
    """Gap fill requires >0.8% gap."""
    import pandas as pd
    # Small gap — no signal
    df_small = pd.DataFrame({
        'close': [23000, 23000],
        'open': [23000, 23050],  # 0.2% gap — too small
    })
    signals = nse._gap_fill(date(2026, 3, 15), df_small)
    assert len(signals) == 0

    # Large gap — signal fires
    df_large = pd.DataFrame({
        'close': [23000, 23000],
        'open': [23000, 23200],  # 0.87% gap
    })
    signals = nse._gap_fill(date(2026, 3, 15), df_large)
    assert len(signals) == 1
    assert signals[0]['signal_id'] == 'NSE_GAP_FILL_SHORT'


def test_all_signals_have_required_fields(nse):
    """Every signal must have direction, confidence, source."""
    # Test on a day that fires multiple signals
    signals = nse.compute(date(2026, 3, 18))  # day 18 = late month
    for s in signals:
        assert 'signal_id' in s
        assert 'direction' in s
        assert 'confidence' in s
        assert 'source' in s


def test_rbi_mpc_drift_fires_3_days_before(nse):
    """RBI MPC drift fires exactly 3 days before MPC date."""
    # 2026-02-06 is MPC date, so 2026-02-03 should fire
    signals = nse._rbi_mpc_drift(date(2026, 2, 3))
    assert any(s['signal_id'] == 'NSE_RBI_MPC_DRIFT' for s in signals)

    # Day of MPC should fire EXIT
    signals_day = nse._rbi_mpc_drift(date(2026, 2, 6))
    assert any(s['signal_id'] == 'NSE_RBI_MPC_EXIT' for s in signals_day)
