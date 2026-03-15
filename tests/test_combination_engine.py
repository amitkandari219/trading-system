"""Tests for CombinationEngine (GRIMES_DRY_3_2 + KAUFMAN_DRY_12 SEQ_5)."""

import pytest
from datetime import date, timedelta
import pandas as pd
import numpy as np

from paper_trading.combination_engine import CombinationEngine


def make_row(high=100, low=98, close=99, volume=1000,
             prev_high=99, prev_low=97, prev_close=98,
             prev_volume=1100, adx_14=30, regime='TRENDING'):
    """Create a mock row Series."""
    return pd.Series({
        'high': high, 'low': low, 'close': close, 'volume': volume,
        'prev_high': prev_high, 'prev_low': prev_low,
        'prev_close': prev_close, 'prev_volume': prev_volume,
        'adx_14': adx_14, 'regime': regime,
    })


# Grimes LONG: high > prev_high AND low > prev_low AND adx > 25 AND TRENDING
GRIMES_LONG_ROW = make_row(high=101, low=99, adx_14=30, regime='TRENDING')
GRIMES_LONG_PREV = make_row(high=100, low=98)

# Kaufman LONG confirm: close > prev_close AND volume < prev_volume
KAUFMAN_LONG_ROW = make_row(close=101, prev_close=99, volume=900, prev_volume=1100)
KAUFMAN_LONG_PREV = make_row(close=99)

# Grimes SHORT: low < prev_low AND high < prev_high AND adx > 25 AND TRENDING
GRIMES_SHORT_ROW = make_row(high=99, low=96, prev_high=100, prev_low=97, adx_14=30, regime='TRENDING')
GRIMES_SHORT_PREV = make_row(high=100, low=97)

# Kaufman SHORT confirm: close < prev_close AND volume > prev_volume
KAUFMAN_SHORT_ROW = make_row(close=97, prev_close=99, volume=1200, prev_volume=1000)

# Neutral row (no signal fires)
NEUTRAL_ROW = make_row(high=100, low=98.5, close=99, volume=1050,
                        prev_high=100, prev_low=98, prev_close=99, prev_volume=1050)
NEUTRAL_PREV = make_row()


def test_grimes_long_kaufman_confirms_day2():
    """Grimes fires long, Kaufman confirms day 2 → ENTER_LONG."""
    engine = CombinationEngine()
    day1 = date(2025, 1, 1)
    day2 = date(2025, 1, 2)

    r1 = engine.update(day1, GRIMES_LONG_ROW, GRIMES_LONG_PREV)
    assert r1['grimes_fired'] is True
    assert r1['action'] is None  # no entry yet

    r2 = engine.update(day2, KAUFMAN_LONG_ROW, KAUFMAN_LONG_PREV)
    assert r2['action'] == 'ENTER_LONG'
    assert r2['kaufman_confirmed'] is True


def test_grimes_long_kaufman_confirms_day5_boundary():
    """Grimes fires long, Kaufman confirms day 5 → ENTER_LONG (boundary)."""
    engine = CombinationEngine()
    day1 = date(2025, 1, 1)

    engine.update(day1, GRIMES_LONG_ROW, GRIMES_LONG_PREV)

    # Days 2-4: neutral
    for d in range(2, 5):
        engine.update(date(2025, 1, d), NEUTRAL_ROW, NEUTRAL_PREV)

    # Day 5: confirm (5 - 1 = 4 days after, within 5-day window)
    r = engine.update(date(2025, 1, 5), KAUFMAN_LONG_ROW, KAUFMAN_LONG_PREV)
    assert r['action'] == 'ENTER_LONG'


def test_grimes_long_kaufman_confirms_day6_expired():
    """Grimes fires long, Kaufman confirms day 7 → NO ENTRY (expired)."""
    engine = CombinationEngine()
    day1 = date(2025, 1, 1)

    engine.update(day1, GRIMES_LONG_ROW, GRIMES_LONG_PREV)

    # Days 2-6: neutral
    for d in range(2, 7):
        engine.update(date(2025, 1, d), NEUTRAL_ROW, NEUTRAL_PREV)

    # Day 7: too late
    r = engine.update(date(2025, 1, 7), KAUFMAN_LONG_ROW, KAUFMAN_LONG_PREV)
    assert r['action'] is None


def test_grimes_long_kaufman_short_no_entry():
    """Grimes fires long, Kaufman fires short → NO ENTRY (direction mismatch)."""
    engine = CombinationEngine()
    day1 = date(2025, 1, 1)
    day2 = date(2025, 1, 2)

    engine.update(day1, GRIMES_LONG_ROW, GRIMES_LONG_PREV)
    r = engine.update(day2, KAUFMAN_SHORT_ROW, NEUTRAL_PREV)
    assert r['action'] is None


def test_grimes_short_kaufman_confirms():
    """Grimes fires short, Kaufman confirms → ENTER_SHORT."""
    engine = CombinationEngine()
    day1 = date(2025, 1, 1)
    day2 = date(2025, 1, 2)

    engine.update(day1, GRIMES_SHORT_ROW, GRIMES_SHORT_PREV)
    r = engine.update(day2, KAUFMAN_SHORT_ROW, NEUTRAL_PREV)
    assert r['action'] == 'ENTER_SHORT'


def test_structure_violation_exits_long():
    """Structure violation (low < prev_low) exits long position."""
    engine = CombinationEngine()
    day1 = date(2025, 1, 1)
    day2 = date(2025, 1, 2)
    day3 = date(2025, 1, 3)

    engine.update(day1, GRIMES_LONG_ROW, GRIMES_LONG_PREV)
    engine.update(day2, KAUFMAN_LONG_ROW, KAUFMAN_LONG_PREV)  # enter
    assert engine.position is not None

    # Day 3: low < prev_low → structure violation
    violation_row = make_row(high=101, low=96, prev_high=100, prev_low=98)
    violation_prev = make_row(high=100, low=98)
    r = engine.update(day3, violation_row, violation_prev)
    assert r['action'] == 'EXIT'
    assert 'structure_violation' in r['reason']
    assert engine.position is None


def test_structure_violation_exits_short():
    """Structure violation (high > prev_high) exits short position."""
    engine = CombinationEngine()
    day1 = date(2025, 1, 1)
    day2 = date(2025, 1, 2)
    day3 = date(2025, 1, 3)

    engine.update(day1, GRIMES_SHORT_ROW, GRIMES_SHORT_PREV)
    engine.update(day2, KAUFMAN_SHORT_ROW, NEUTRAL_PREV)  # enter short
    assert engine.position is not None

    # Day 3: high > prev_high → exit short
    violation_row = make_row(high=102, low=98, prev_high=100, prev_low=97)
    violation_prev = make_row(high=100, low=97)
    r = engine.update(day3, violation_row, violation_prev)
    assert r['action'] == 'EXIT'
    assert 'structure_violation' in r['reason']


def test_hold_max_10_days():
    """Hold max 10 days → forced exit."""
    engine = CombinationEngine()
    day1 = date(2025, 1, 1)
    day2 = date(2025, 1, 2)

    engine.update(day1, GRIMES_LONG_ROW, GRIMES_LONG_PREV)
    engine.update(day2, KAUFMAN_LONG_ROW, KAUFMAN_LONG_PREV)  # enter

    # Hold for 10 days with no structure violation
    hold_row = make_row(high=102, low=100, prev_high=101, prev_low=99)  # always higher
    hold_prev = make_row(high=101, low=99)
    for d in range(3, 13):
        r = engine.update(date(2025, 1, d), hold_row, hold_prev)

    # Day 13 = 11 days from entry, days_held=10 → exit
    assert r['action'] == 'EXIT'
    assert 'hold_max' in r['reason']


def test_pending_long_cancelled_by_short():
    """Pending long cancelled when Grimes fires short."""
    engine = CombinationEngine()
    day1 = date(2025, 1, 1)
    day2 = date(2025, 1, 2)

    engine.update(day1, GRIMES_LONG_ROW, GRIMES_LONG_PREV)
    assert engine.pending_long is not None

    engine.update(day2, GRIMES_SHORT_ROW, GRIMES_SHORT_PREV)
    assert engine.pending_long is None
    assert engine.pending_short is not None


def test_no_entry_when_position_open():
    """No new entry when position already open."""
    engine = CombinationEngine()
    day1 = date(2025, 1, 1)
    day2 = date(2025, 1, 2)
    day3 = date(2025, 1, 3)
    day4 = date(2025, 1, 4)

    engine.update(day1, GRIMES_LONG_ROW, GRIMES_LONG_PREV)
    engine.update(day2, KAUFMAN_LONG_ROW, KAUFMAN_LONG_PREV)  # enter
    assert engine.position is not None

    # Another Grimes fires — should NOT create new pending
    r = engine.update(day3, GRIMES_LONG_ROW, GRIMES_LONG_PREV)
    assert engine.pending_long is None  # not set because position open


def test_state_export_import():
    """State persists correctly via get_state."""
    engine = CombinationEngine()
    engine.pending_long = date(2025, 1, 5)
    engine.position = {'direction': 'LONG', 'entry_date': date(2025, 1, 6), 'days_held': 3}

    state = engine.get_state()
    assert state['pending_long'] == '2025-01-05'
    assert state['position_open'] is True
    assert state['position_dir'] == 'LONG'
    assert state['days_held'] == 3


def test_regime_filter_ranging():
    """No Grimes signal in RANGING regime."""
    engine = CombinationEngine()
    row = make_row(high=101, low=99, adx_14=30, regime='RANGING')
    prev = make_row(high=100, low=98)
    r = engine.update(date(2025, 1, 1), row, prev)
    assert r['grimes_fired'] is False
    assert engine.pending_long is None


def test_regime_filter_high_vol():
    """No Grimes signal in HIGH_VOL regime."""
    engine = CombinationEngine()
    row = make_row(high=101, low=99, adx_14=30, regime='HIGH_VOL')
    prev = make_row(high=100, low=98)
    r = engine.update(date(2025, 1, 1), row, prev)
    assert r['grimes_fired'] is False


def test_full_5day_sequence():
    """Full 5-day sequence: Grimes fires day 1, neutral 2-4, confirm day 5."""
    engine = CombinationEngine()
    days = [date(2025, 1, d) for d in range(1, 7)]

    # Day 1: Grimes fires
    r = engine.update(days[0], GRIMES_LONG_ROW, GRIMES_LONG_PREV)
    assert r['grimes_fired'] is True
    assert r['action'] is None

    # Days 2-4: neutral
    for d in days[1:4]:
        r = engine.update(d, NEUTRAL_ROW, NEUTRAL_PREV)
        assert r['action'] is None
        assert r['pending_days_remaining'] > 0

    # Day 5: Kaufman confirms
    r = engine.update(days[4], KAUFMAN_LONG_ROW, KAUFMAN_LONG_PREV)
    assert r['action'] == 'ENTER_LONG'


def test_backdate_sep2025_no_fires():
    """In HIGH_VOL/correction market, combination should not fire."""
    engine = CombinationEngine()
    # Simulate HIGH_VOL regime for 30 days
    for d in range(30):
        day = date(2025, 10, 1) + timedelta(days=d)
        row = make_row(high=100 - d*0.5, low=98 - d*0.5,
                       prev_high=100 - (d-1)*0.5, prev_low=98 - (d-1)*0.5,
                       adx_14=30, regime='HIGH_VOL')
        prev = make_row(high=100 - (d-1)*0.5, low=98 - (d-1)*0.5)
        r = engine.update(day, row, prev)
        assert r['action'] is None or r['action'] == 'EXIT'
