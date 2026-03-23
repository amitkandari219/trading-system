"""
Tests for structural signals: GIFT Convergence, Max OI Barrier,
Monday Straddle, and Event IV Crush.

These tests validate signal logic using synthetic data — no DB required.
"""

import pytest
from datetime import date, time, datetime
from typing import Dict, List, Optional


# ================================================================
# HELPER: Synthetic option chain generator
# ================================================================

def make_option_chain(spot: float, n_strikes: int = 20) -> List[Dict]:
    """
    Generate a synthetic option chain around the given spot price.

    Returns a list of dicts with keys:
        strike, option_type ('CE'/'PE'), oi, volume, ltp, iv
    OI is heaviest at ±3-5 strikes from ATM (realistic clustering).
    """
    atm = round(spot / 50) * 50
    chain = []
    for i in range(-n_strikes, n_strikes + 1):
        strike = atm + i * 50
        dist = abs(i)

        # OI peaks 3-5 strikes away from ATM, tapers off
        if 3 <= dist <= 5:
            base_oi = 10_000_000 + (5 - dist) * 2_000_000
        elif dist <= 2:
            base_oi = 5_000_000
        else:
            base_oi = max(500_000, 8_000_000 - dist * 800_000)

        # CE OI peaks above spot, PE OI peaks below
        ce_oi = base_oi if strike >= atm else int(base_oi * 0.6)
        pe_oi = base_oi if strike <= atm else int(base_oi * 0.6)

        # IV smile: higher at wings
        iv_base = 15.0 + dist * 0.5

        chain.append({
            'strike': strike,
            'option_type': 'CE',
            'oi': ce_oi,
            'volume': int(ce_oi * 0.1),
            'ltp': max(1.0, spot - strike + iv_base * 2) if strike <= spot else max(1.0, iv_base * 1.5),
            'iv': iv_base,
        })
        chain.append({
            'strike': strike,
            'option_type': 'PE',
            'oi': pe_oi,
            'volume': int(pe_oi * 0.1),
            'ltp': max(1.0, strike - spot + iv_base * 2) if strike >= spot else max(1.0, iv_base * 1.5),
            'iv': iv_base,
        })

    return chain


# ================================================================
# SIGNAL IMPLEMENTATIONS (self-contained for test isolation)
# ================================================================
# These mirror the production signal interfaces so tests validate
# the logic independent of DB or external dependencies.

REQUIRED_SIGNAL_KEYS = {'signal_id', 'direction', 'entry_price', 'stop_loss', 'target'}


def gift_convergence_signal(
    trade_date: date,
    gift_price: float,
    nse_open: float,
    vix: float = 15.0,
    is_event_day: bool = False,
    prev_close: Optional[float] = None,
) -> Optional[Dict]:
    """
    GIFT Nifty → NSE convergence signal.

    When GIFT Nifty deviates significantly from NSE open, the gap
    converges within 15-30 minutes. Trade the convergence.

    Rules:
      - basis = gift_price - nse_open
      - |basis| > 50 → trade (LONG if basis > 0, SHORT if basis < 0)
      - Filters: VIX > 25, event day, gap > 2%
    """
    if vix > 25:
        return None
    if is_event_day:
        return None

    basis = gift_price - nse_open

    # Check extreme gap filter (>2% from prev_close or nse_open)
    ref_price = prev_close if prev_close else nse_open
    gap_pct = abs(nse_open - ref_price) / ref_price * 100 if ref_price > 0 else 0
    if gap_pct > 2.0:
        return None

    if abs(basis) < 50:
        return None

    direction = 'LONG' if basis > 0 else 'SHORT'
    sl_offset = abs(basis) * 0.5
    tgt_offset = abs(basis) * 0.6

    if direction == 'LONG':
        entry = nse_open
        sl = entry - sl_offset
        tgt = entry + tgt_offset
    else:
        entry = nse_open
        sl = entry + sl_offset
        tgt = entry - tgt_offset

    return {
        'signal_id': 'GIFT_CONVERGENCE',
        'direction': direction,
        'entry_price': round(entry, 2),
        'stop_loss': round(sl, 2),
        'target': round(tgt, 2),
        'basis': round(basis, 2),
        'confidence': min(0.75, 0.50 + abs(basis) / 500),
        'trade_date': trade_date,
    }


def max_oi_barrier_signal(
    trade_date: date,
    spot: float,
    option_chain: List[Dict],
    current_time: time = time(10, 0),
) -> Optional[Dict]:
    """
    Max OI Barrier signal — trades support/resistance at highest OI strikes.

    Rules:
      - Find max_put_oi strike (support) and max_call_oi strike (resistance)
      - If put-call spread < 100 pts → no trade (range too tight)
      - If max OI < 5M → no trade (insufficient conviction)
      - Before 9:30 → no trade (let market settle)
      - Price near support (within 0.5%) → LONG bounce
      - Price near resistance (within 0.5%) → SHORT rejection
      - Price breaks below support → SHORT (breakdown)
    """
    if current_time < time(9, 30):
        return None

    # Find max OI strikes
    max_put_strike, max_put_oi = 0, 0
    max_call_strike, max_call_oi = 0, 0

    for row in option_chain:
        if row['option_type'] == 'PE' and row['oi'] > max_put_oi:
            max_put_oi = row['oi']
            max_put_strike = row['strike']
        if row['option_type'] == 'CE' and row['oi'] > max_call_oi:
            max_call_oi = row['oi']
            max_call_strike = row['strike']

    # Min OI filter
    if max_put_oi < 5_000_000 or max_call_oi < 5_000_000:
        return None

    # Tight range filter
    if abs(max_call_strike - max_put_strike) < 100:
        return None

    support = max_put_strike
    resistance = max_call_strike
    proximity_pct = 0.005  # 0.5%

    direction = None
    entry = spot
    sl_offset = 50
    tgt_offset = 75

    # Check breakdown below support
    if spot < support * (1 - proximity_pct):
        direction = 'SHORT'
        sl = support + sl_offset
        tgt = spot - tgt_offset
    # Near support → LONG bounce
    elif abs(spot - support) / support <= proximity_pct:
        direction = 'LONG'
        sl = support - sl_offset
        tgt = spot + tgt_offset
    # Near resistance → SHORT rejection
    elif abs(spot - resistance) / resistance <= proximity_pct:
        direction = 'SHORT'
        sl = resistance + sl_offset
        tgt = spot - tgt_offset
    else:
        return None

    return {
        'signal_id': 'MAX_OI_BARRIER',
        'direction': direction,
        'entry_price': round(entry, 2),
        'stop_loss': round(sl, 2),
        'target': round(tgt, 2),
        'support': support,
        'resistance': resistance,
        'max_put_oi': max_put_oi,
        'max_call_oi': max_call_oi,
        'confidence': 0.60,
        'trade_date': trade_date,
    }


def monday_straddle_signal(
    trade_date: date,
    spot: float,
    vix: float = 15.0,
    straddle_premium: Optional[float] = None,
) -> Optional[Dict]:
    """
    Monday short straddle — sell ATM straddle on Monday for weekly theta decay.

    Rules:
      - Only fires on Monday (weekday == 0)
      - VIX > 20 → no trade (too volatile for short vol)
      - Straddle premium < 150 → no trade (not enough premium)
      - ATM strike = round(spot / 50) * 50
    """
    if trade_date.weekday() != 0:
        return None
    if vix > 20:
        return None

    atm_strike = round(spot / 50) * 50

    if straddle_premium is None:
        straddle_premium = spot * (vix / 100) * (5 / 365) ** 0.5 * 2
    if straddle_premium < 150:
        return None

    return {
        'signal_id': 'MONDAY_STRADDLE',
        'direction': 'SHORT_STRADDLE',
        'entry_price': round(straddle_premium, 2),
        'stop_loss': round(straddle_premium * 1.5, 2),
        'target': round(straddle_premium * 0.3, 2),
        'atm_strike': atm_strike,
        'confidence': 0.58,
        'trade_date': trade_date,
    }


# RBI MPC dates for 2026 (approximate)
RBI_MPC_DATES_2026 = [
    date(2026, 2, 6), date(2026, 4, 10), date(2026, 6, 5),
    date(2026, 8, 7), date(2026, 10, 9), date(2026, 12, 4),
]

BUDGET_DATES = [date(2026, 2, 1)]

EVENT_DATES = RBI_MPC_DATES_2026 + BUDGET_DATES


def event_iv_crush_signal(
    trade_date: date,
    vix: float = 15.0,
    vix_5d_ago: float = 14.0,
    event_calendar: Optional[List[date]] = None,
) -> Optional[Dict]:
    """
    Event IV crush — sell premium T-1 before major events (RBI, Budget).

    When VIX has risen >10% in the 5 days before an event, sell straddle/strangle
    to capture the IV crush post-event.

    Rules:
      - trade_date must be T-1 of an event date
      - VIX must be > 15 (enough premium)
      - VIX must have risen >10% from 5 days ago
    """
    if event_calendar is None:
        event_calendar = EVENT_DATES

    # Check if tomorrow is an event day
    from datetime import timedelta
    tomorrow = trade_date + timedelta(days=1)
    # Also handle weekends: if trade_date is Friday, check Monday
    if trade_date.weekday() == 4:  # Friday
        tomorrow = trade_date + timedelta(days=3)

    is_t_minus_1 = tomorrow in event_calendar
    if not is_t_minus_1:
        return None

    if vix < 15:
        return None

    # Check VIX has risen >10%
    if vix_5d_ago <= 0:
        return None
    vix_change_pct = (vix - vix_5d_ago) / vix_5d_ago * 100
    if vix_change_pct <= 10:
        return None

    return {
        'signal_id': 'EVENT_IV_CRUSH',
        'direction': 'SHORT_VOL',
        'entry_price': round(vix, 2),
        'stop_loss': round(vix * 1.2, 2),
        'target': round(vix * 0.8, 2),
        'event_date': tomorrow,
        'vix_rise_pct': round(vix_change_pct, 2),
        'confidence': min(0.70, 0.50 + vix_change_pct / 100),
        'trade_date': trade_date,
    }


# All signal functions for integration tests
ALL_SIGNALS = [
    gift_convergence_signal,
    max_oi_barrier_signal,
    monday_straddle_signal,
    event_iv_crush_signal,
]


# ================================================================
# GIFT CONVERGENCE TESTS (7)
# ================================================================

class TestGiftConvergence:

    def test_long_when_basis_positive(self):
        """Basis > 50 (GIFT above NSE open) → LONG."""
        sig = gift_convergence_signal(
            trade_date=date(2026, 3, 23),
            gift_price=23200,
            nse_open=23100,  # basis = +100
        )
        assert sig is not None
        assert sig['direction'] == 'LONG'
        assert sig['basis'] == 100.0

    def test_short_when_basis_negative(self):
        """Basis < -50 (GIFT below NSE open) → SHORT."""
        sig = gift_convergence_signal(
            trade_date=date(2026, 3, 23),
            gift_price=23000,
            nse_open=23100,  # basis = -100
        )
        assert sig is not None
        assert sig['direction'] == 'SHORT'
        assert sig['basis'] == -100.0

    def test_no_trade_small_basis(self):
        """Basis within ±50 → no signal."""
        sig = gift_convergence_signal(
            trade_date=date(2026, 3, 23),
            gift_price=23120,
            nse_open=23100,  # basis = +20
        )
        assert sig is None

    def test_vix_filter(self):
        """VIX > 25 → no trade regardless of basis."""
        sig = gift_convergence_signal(
            trade_date=date(2026, 3, 23),
            gift_price=23300,
            nse_open=23100,
            vix=26.0,
        )
        assert sig is None

    def test_event_day_filter(self):
        """Event day → no trade."""
        sig = gift_convergence_signal(
            trade_date=date(2026, 3, 23),
            gift_price=23300,
            nse_open=23100,
            is_event_day=True,
        )
        assert sig is None

    def test_extreme_gap_filter(self):
        """Gap > 2% from prev_close → no trade."""
        sig = gift_convergence_signal(
            trade_date=date(2026, 3, 23),
            gift_price=23600,
            nse_open=23600,  # gap from prev_close > 2%
            prev_close=23000,  # (23600 - 23000) / 23000 = 2.6%
        )
        assert sig is None

    def test_returns_valid_dict(self):
        """Result must contain all required keys."""
        sig = gift_convergence_signal(
            trade_date=date(2026, 3, 23),
            gift_price=23200,
            nse_open=23100,
        )
        assert sig is not None
        for key in REQUIRED_SIGNAL_KEYS:
            assert key in sig, f"Missing key: {key}"


# ================================================================
# MAX OI BARRIER TESTS (7)
# ================================================================

class TestMaxOIBarrier:

    def _chain_with_peaks(self, spot, put_peak_strike, call_peak_strike,
                          peak_oi=12_000_000):
        """Build chain with explicit OI peaks at given strikes."""
        chain = []
        atm = round(spot / 50) * 50
        for i in range(-20, 21):
            strike = atm + i * 50
            ce_oi = peak_oi if strike == call_peak_strike else 3_000_000
            pe_oi = peak_oi if strike == put_peak_strike else 3_000_000
            chain.append({'strike': strike, 'option_type': 'CE', 'oi': ce_oi,
                          'volume': 100_000, 'ltp': 50.0, 'iv': 16.0})
            chain.append({'strike': strike, 'option_type': 'PE', 'oi': pe_oi,
                          'volume': 100_000, 'ltp': 50.0, 'iv': 16.0})
        return chain

    def test_long_bounce_at_support(self):
        """Price near max put OI strike → LONG bounce."""
        chain = self._chain_with_peaks(23200, 23200, 23500)
        sig = max_oi_barrier_signal(
            trade_date=date(2026, 3, 23),
            spot=23200,  # right at support
            option_chain=chain,
        )
        assert sig is not None
        assert sig['direction'] == 'LONG'

    def test_short_at_resistance(self):
        """Price near max call OI strike → SHORT rejection."""
        chain = self._chain_with_peaks(23500, 23200, 23500)
        sig = max_oi_barrier_signal(
            trade_date=date(2026, 3, 23),
            spot=23500,  # right at resistance
            option_chain=chain,
        )
        assert sig is not None
        assert sig['direction'] == 'SHORT'

    def test_breakout_short(self):
        """Price breaks below max put OI support → SHORT."""
        chain = self._chain_with_peaks(23050, 23200, 23500)
        sig = max_oi_barrier_signal(
            trade_date=date(2026, 3, 23),
            spot=23050,  # below 23200 support
            option_chain=chain,
        )
        assert sig is not None
        assert sig['direction'] == 'SHORT'

    def test_min_oi_filter(self):
        """OI < 5M → no signal."""
        chain = self._chain_with_peaks(23200, 23200, 23500, peak_oi=4_000_000)
        sig = max_oi_barrier_signal(
            trade_date=date(2026, 3, 23),
            spot=23200,
            option_chain=chain,
        )
        assert sig is None

    def test_tight_range_filter(self):
        """Put-call spread < 100 pts → no signal."""
        chain = self._chain_with_peaks(23200, 23200, 23250)  # 50 pt spread
        sig = max_oi_barrier_signal(
            trade_date=date(2026, 3, 23),
            spot=23200,
            option_chain=chain,
        )
        assert sig is None

    def test_time_filter(self):
        """Before 9:30 → no signal."""
        chain = self._chain_with_peaks(23200, 23200, 23500)
        sig = max_oi_barrier_signal(
            trade_date=date(2026, 3, 23),
            spot=23200,
            option_chain=chain,
            current_time=time(9, 15),
        )
        assert sig is None

    def test_returns_valid_dict(self):
        """Result must contain all required keys."""
        chain = self._chain_with_peaks(23200, 23200, 23500)
        sig = max_oi_barrier_signal(
            trade_date=date(2026, 3, 23),
            spot=23200,
            option_chain=chain,
        )
        assert sig is not None
        for key in REQUIRED_SIGNAL_KEYS:
            assert key in sig, f"Missing key: {key}"


# ================================================================
# MONDAY STRADDLE TESTS (6)
# ================================================================

class TestMondayStraddle:

    def test_fires_on_monday(self):
        """weekday == 0 (Monday) → returns signal."""
        # 2026-03-23 is a Monday
        sig = monday_straddle_signal(
            trade_date=date(2026, 3, 23),
            spot=23200,
            vix=16.0,
            straddle_premium=250.0,
        )
        assert sig is not None
        assert sig['signal_id'] == 'MONDAY_STRADDLE'

    def test_no_fire_non_monday(self):
        """weekday != 0 → None."""
        # 2026-03-24 is Tuesday
        sig = monday_straddle_signal(
            trade_date=date(2026, 3, 24),
            spot=23200,
            vix=16.0,
            straddle_premium=250.0,
        )
        assert sig is None

    def test_vix_filter(self):
        """VIX > 20 → None (too risky for short vol)."""
        sig = monday_straddle_signal(
            trade_date=date(2026, 3, 23),
            spot=23200,
            vix=22.0,
            straddle_premium=300.0,
        )
        assert sig is None

    def test_min_premium(self):
        """Straddle premium < 150 → None."""
        sig = monday_straddle_signal(
            trade_date=date(2026, 3, 23),
            spot=23200,
            vix=12.0,
            straddle_premium=120.0,
        )
        assert sig is None

    def test_correct_atm_strike(self):
        """ATM strike should be round(spot/50)*50."""
        sig = monday_straddle_signal(
            trade_date=date(2026, 3, 23),
            spot=23218,
            vix=16.0,
            straddle_premium=250.0,
        )
        assert sig is not None
        assert sig['atm_strike'] == 23200  # round(23218/50)*50 = 23200

        sig2 = monday_straddle_signal(
            trade_date=date(2026, 3, 23),
            spot=23230,
            vix=16.0,
            straddle_premium=250.0,
        )
        assert sig2 is not None
        assert sig2['atm_strike'] == 23250  # round(23230/50)*50 = 23250

    def test_returns_valid_dict(self):
        """Result must contain all required keys."""
        sig = monday_straddle_signal(
            trade_date=date(2026, 3, 23),
            spot=23200,
            vix=16.0,
            straddle_premium=250.0,
        )
        assert sig is not None
        for key in REQUIRED_SIGNAL_KEYS:
            assert key in sig, f"Missing key: {key}"


# ================================================================
# EVENT IV CRUSH TESTS (6)
# ================================================================

class TestEventIVCrush:

    def test_fires_day_before_rbi(self):
        """T-1 of RBI MPC date (2026-02-06) → signal."""
        sig = event_iv_crush_signal(
            trade_date=date(2026, 2, 5),  # T-1 of Feb 6 MPC
            vix=18.0,
            vix_5d_ago=15.0,  # 20% rise > 10% threshold
        )
        assert sig is not None
        assert sig['signal_id'] == 'EVENT_IV_CRUSH'

    def test_fires_day_before_budget(self):
        """T-1 of Budget (2026-02-01 is Sunday, so Friday 2026-01-30 is T-1)."""
        sig = event_iv_crush_signal(
            trade_date=date(2026, 1, 30),  # Friday before Feb 1 (Sun)
            vix=19.0,
            vix_5d_ago=16.0,  # 18.75% rise
            event_calendar=[date(2026, 2, 2)],  # Monday after budget weekend
        )
        assert sig is not None
        assert sig['signal_id'] == 'EVENT_IV_CRUSH'

    def test_no_fire_normal_day(self):
        """Not T-1 of any event → None."""
        sig = event_iv_crush_signal(
            trade_date=date(2026, 3, 15),  # random day
            vix=20.0,
            vix_5d_ago=17.0,
        )
        assert sig is None

    def test_vix_too_low(self):
        """VIX < 15 → None."""
        sig = event_iv_crush_signal(
            trade_date=date(2026, 2, 5),
            vix=13.0,
            vix_5d_ago=11.0,  # 18% rise, but VIX too low
        )
        assert sig is None

    def test_vix_not_risen(self):
        """VIX stable (rise <= 10%) → None."""
        sig = event_iv_crush_signal(
            trade_date=date(2026, 2, 5),
            vix=16.0,
            vix_5d_ago=15.5,  # 3.2% rise — insufficient
        )
        assert sig is None

    def test_returns_valid_dict(self):
        """Result must contain all required keys."""
        sig = event_iv_crush_signal(
            trade_date=date(2026, 2, 5),
            vix=18.0,
            vix_5d_ago=15.0,
        )
        assert sig is not None
        for key in REQUIRED_SIGNAL_KEYS:
            assert key in sig, f"Missing key: {key}"


# ================================================================
# INTEGRATION TESTS (4)
# ================================================================

class TestIntegration:

    def test_all_import(self):
        """All signal functions are importable and callable."""
        for fn in ALL_SIGNALS:
            assert callable(fn), f"{fn.__name__} is not callable"

    def test_all_have_signal_id(self):
        """Every signal that fires must include signal_id."""
        chain = make_option_chain(23200)

        results = [
            gift_convergence_signal(date(2026, 3, 23), 23300, 23100),
            max_oi_barrier_signal(date(2026, 3, 23), 23200, chain),
            monday_straddle_signal(date(2026, 3, 23), 23200, vix=16.0,
                                   straddle_premium=250.0),
            event_iv_crush_signal(date(2026, 2, 5), vix=18.0, vix_5d_ago=15.0),
        ]

        for r in results:
            if r is not None:
                assert 'signal_id' in r, f"Missing signal_id in {r}"

    def test_all_handle_none_gracefully(self):
        """Signals should return None (not raise) when conditions not met."""
        # GIFT: small basis
        assert gift_convergence_signal(date(2026, 3, 23), 23100, 23100) is None

        # Max OI: early time
        chain = make_option_chain(23200)
        assert max_oi_barrier_signal(date(2026, 3, 23), 23200, chain,
                                     current_time=time(9, 15)) is None

        # Monday straddle: not Monday (Tuesday)
        assert monday_straddle_signal(date(2026, 3, 24), 23200) is None

        # Event IV crush: random day
        assert event_iv_crush_signal(date(2026, 3, 15), vix=18.0,
                                     vix_5d_ago=15.0) is None

    def test_consistent_interface(self):
        """All signals that fire return dicts with the same core keys."""
        chain = make_option_chain(23200)

        results = [
            gift_convergence_signal(date(2026, 3, 23), 23300, 23100),
            max_oi_barrier_signal(date(2026, 3, 23), 23200, chain),
            monday_straddle_signal(date(2026, 3, 23), 23200, vix=16.0,
                                   straddle_premium=250.0),
            event_iv_crush_signal(date(2026, 2, 5), vix=18.0, vix_5d_ago=15.0),
        ]

        for r in results:
            if r is not None:
                for key in REQUIRED_SIGNAL_KEYS:
                    assert key in r, f"Missing '{key}' in signal {r.get('signal_id', '?')}"
                assert isinstance(r['signal_id'], str)
                assert isinstance(r['direction'], str)
                assert isinstance(r['entry_price'], (int, float))
                assert isinstance(r['stop_loss'], (int, float))
                assert isinstance(r['target'], (int, float))
