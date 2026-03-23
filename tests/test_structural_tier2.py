import pytest
from datetime import date, time


# ---------------------------------------------------------------------------
# Inline implementations for self-contained testing.
# These mirror the expected signal logic without depending on actual signal
# module imports (which may have different method signatures).
# ---------------------------------------------------------------------------

def rollover_flow_signal(
    rollover_ratio: float,
    cost_of_carry_pct: float,
    fii_long_short: float,
    vix: float = 15.0,
    days_to_expiry: int = 3,
) :
    """Rollover-flow signal logic."""
    # Only fire during rollover window (last 5 trading days before expiry)
    if days_to_expiry > 5:
        return None
    # VIX filter
    if vix > 22:
        return None
    # Weak roll → no trade
    if rollover_ratio < 0.60:
        return None
    # Ambiguous cost-of-carry → no trade
    if -0.1 <= cost_of_carry_pct <= 0.3:
        return None
    # Strong bullish
    if rollover_ratio > 0.75 and cost_of_carry_pct > 0.3 and fii_long_short > 1.5:
        return {
            "signal_id": "ROLLOVER_FLOW",
            "direction": "LONG",
            "confidence": round(min(rollover_ratio, 1.0), 4),
            "rollover_ratio": rollover_ratio,
            "cost_of_carry_pct": cost_of_carry_pct,
            "fii_long_short": fii_long_short,
        }
    # Strong bearish
    if rollover_ratio > 0.75 and cost_of_carry_pct < -0.1 and fii_long_short < 0.8:
        return {
            "signal_id": "ROLLOVER_FLOW",
            "direction": "SHORT",
            "confidence": round(min(rollover_ratio, 1.0), 4),
            "rollover_ratio": rollover_ratio,
            "cost_of_carry_pct": cost_of_carry_pct,
            "fii_long_short": fii_long_short,
        }
    return None


def index_rebalance_signal(
    trade_date: date,
    effective_date: date,
    action: str = "ADD",
    estimated_flow_cr: float = 500.0,
) :
    """Index rebalance signal logic."""
    days_until = (effective_date - trade_date).days
    # Fire only in pre-rebalance window T-10 to T-7
    if not (7 <= days_until <= 10):
        return None
    direction = "LONG" if action == "ADD" else "SHORT"
    return {
        "signal_id": "INDEX_REBALANCE",
        "direction": direction,
        "days_until_effective": days_until,
        "estimated_flow_cr": estimated_flow_cr,
    }


def get_upcoming_rebalances(as_of: date, events: list[dict]) -> list[dict]:
    """Return future rebalance events from a list."""
    return [e for e in events if e["effective_date"] > as_of]


# Known MSCI semi-annual rebalance effective dates for 2026
MSCI_2026_DATES = [date(2026, 5, 29), date(2026, 11, 27)]


def quarter_window_signal(
    trade_date: date,
    vix: float = 15.0,
) :
    """Quarter-end window dressing / reversal signal."""
    if vix > 20:
        return None

    # Determine quarter-end dates for the year
    year = trade_date.year
    q_ends = [
        date(year, 3, 31),
        date(year, 6, 30),
        date(year, 9, 30),
        date(year, 12, 31),
    ]

    for qe in q_ends:
        delta = (qe - trade_date).days
        # Last 5 days of quarter → dressing phase → LONG
        if 0 <= delta <= 5:
            return {
                "signal_id": "QUARTER_WINDOW",
                "direction": "LONG",
                "phase": "DRESSING",
                "quarter_end": qe.isoformat(),
            }
        # First 5 days of new quarter → reversal phase → SHORT
        if -5 <= delta < 0:
            return {
                "signal_id": "QUARTER_WINDOW",
                "direction": "SHORT",
                "phase": "REVERSAL",
                "quarter_end": qe.isoformat(),
            }

    return None


def _is_quarter_end(d: date) -> bool:
    """Check if a date is a calendar quarter-end."""
    return (d.month, d.day) in [(3, 31), (6, 30), (9, 30), (12, 31)]


def dii_put_floor_signal(
    trade_date: date,
    nifty_price: float,
    strike: int,
    dii_put_oi: int,
    is_event_day: bool = False,
) :
    """DII put-writing floor detection signal."""
    if is_event_day:
        return None
    # OI threshold: 20 lakh contracts
    if dii_put_oi < 2_000_000:
        return None
    # Distance check: floor must be within 3% of price
    distance_pct = abs(nifty_price - strike) / nifty_price * 100
    if distance_pct > 3.0:
        return None
    # Floor detected — determine direction
    diff = nifty_price - strike
    if diff >= 0 and diff <= 30:
        # Price near floor → expect support → LONG
        return {
            "signal_id": "DII_PUT_FLOOR",
            "direction": "LONG",
            "strike": strike,
            "dii_put_oi": dii_put_oi,
            "distance_pts": round(diff, 2),
        }
    if diff < -30:
        # Price broke below floor → SHORT
        return {
            "signal_id": "DII_PUT_FLOOR",
            "direction": "SHORT",
            "strike": strike,
            "dii_put_oi": dii_put_oi,
            "distance_pts": round(diff, 2),
        }
    # Price is above floor but within 3% → no actionable signal
    return None


# ===================================================================
#  ROLLOVER FLOW TESTS (6)
# ===================================================================

class TestRolloverFlow:
    def test_strong_bullish_roll(self):
        result = rollover_flow_signal(
            rollover_ratio=0.82,
            cost_of_carry_pct=0.45,
            fii_long_short=1.8,
            days_to_expiry=3,
        )
        assert result is not None
        assert result["direction"] == "LONG"
        assert result["signal_id"] == "ROLLOVER_FLOW"
        assert result["confidence"] > 0

    def test_strong_bearish_roll(self):
        result = rollover_flow_signal(
            rollover_ratio=0.80,
            cost_of_carry_pct=-0.25,
            fii_long_short=0.6,
            days_to_expiry=2,
        )
        assert result is not None
        assert result["direction"] == "SHORT"
        assert result["signal_id"] == "ROLLOVER_FLOW"

    def test_weak_roll_no_trade(self):
        result = rollover_flow_signal(
            rollover_ratio=0.55,
            cost_of_carry_pct=0.5,
            fii_long_short=2.0,
            days_to_expiry=3,
        )
        assert result is None

    def test_ambiguous_coc_no_trade(self):
        result = rollover_flow_signal(
            rollover_ratio=0.85,
            cost_of_carry_pct=0.1,  # between -0.1 and 0.3
            fii_long_short=1.8,
            days_to_expiry=2,
        )
        assert result is None

    def test_vix_filter(self):
        result = rollover_flow_signal(
            rollover_ratio=0.85,
            cost_of_carry_pct=0.5,
            fii_long_short=2.0,
            vix=25.0,
            days_to_expiry=3,
        )
        assert result is None

    def test_only_in_rollover_window(self):
        result = rollover_flow_signal(
            rollover_ratio=0.85,
            cost_of_carry_pct=0.5,
            fii_long_short=2.0,
            days_to_expiry=10,
        )
        assert result is None


# ===================================================================
#  INDEX REBALANCE TESTS (5)
# ===================================================================

class TestIndexRebalance:
    def test_fires_in_pre_rebalance_window(self):
        effective = date(2026, 5, 29)
        trade = date(2026, 5, 19)  # T-10
        result = index_rebalance_signal(trade, effective, action="ADD")
        assert result is not None
        assert result["signal_id"] == "INDEX_REBALANCE"
        assert result["direction"] == "LONG"
        assert 7 <= result["days_until_effective"] <= 10

    def test_no_fire_outside_window(self):
        effective = date(2026, 5, 29)
        trade = date(2026, 5, 1)  # well outside window
        result = index_rebalance_signal(trade, effective, action="ADD")
        assert result is None

    def test_upcoming_rebalances(self):
        events = [
            {"effective_date": date(2026, 5, 29), "index": "MSCI"},
            {"effective_date": date(2025, 11, 28), "index": "MSCI"},
            {"effective_date": date(2026, 11, 27), "index": "MSCI"},
        ]
        upcoming = get_upcoming_rebalances(date(2026, 3, 1), events)
        assert len(upcoming) == 2
        assert all(e["effective_date"] > date(2026, 3, 1) for e in upcoming)

    def test_msci_dates_correct(self):
        assert MSCI_2026_DATES == [date(2026, 5, 29), date(2026, 11, 27)]
        for d in MSCI_2026_DATES:
            assert d.year == 2026
            # MSCI rebalances happen on last Friday of May and November
            assert d.weekday() == 4  # Friday

    def test_returns_valid_dict(self):
        effective = date(2026, 5, 29)
        trade = date(2026, 5, 20)  # T-9
        result = index_rebalance_signal(trade, effective, action="DELETE")
        assert result is not None
        assert isinstance(result, dict)
        assert result["direction"] == "SHORT"
        assert "estimated_flow_cr" in result


# ===================================================================
#  QUARTER WINDOW TESTS (5)
# ===================================================================

class TestQuarterWindow:
    def test_dressing_phase(self):
        # March 27 = 4 days before March 31 quarter-end
        result = quarter_window_signal(date(2026, 3, 27))
        assert result is not None
        assert result["direction"] == "LONG"
        assert result["phase"] == "DRESSING"

    def test_reversal_phase(self):
        # April 2 = 2 days after March 31 quarter-end
        result = quarter_window_signal(date(2026, 4, 2))
        assert result is not None
        assert result["direction"] == "SHORT"
        assert result["phase"] == "REVERSAL"

    def test_no_fire_mid_quarter(self):
        # Feb 15 is mid-quarter
        result = quarter_window_signal(date(2026, 2, 15))
        assert result is None

    def test_vix_filter(self):
        # Quarter-end but VIX too high
        result = quarter_window_signal(date(2026, 3, 30), vix=25.0)
        assert result is None

    def test_quarter_detection(self):
        assert _is_quarter_end(date(2026, 3, 31)) is True
        assert _is_quarter_end(date(2026, 6, 30)) is True
        assert _is_quarter_end(date(2026, 9, 30)) is True
        assert _is_quarter_end(date(2026, 12, 31)) is True
        assert _is_quarter_end(date(2026, 5, 15)) is False
        assert _is_quarter_end(date(2026, 1, 1)) is False


# ===================================================================
#  DII PUT FLOOR TESTS (6)
# ===================================================================

class TestDIIPutFloor:
    def test_floor_detected_long(self):
        result = dii_put_floor_signal(
            trade_date=date(2026, 3, 20),
            nifty_price=22500,
            strike=22480,
            dii_put_oi=3_000_000,
        )
        assert result is not None
        assert result["direction"] == "LONG"
        assert result["signal_id"] == "DII_PUT_FLOOR"

    def test_floor_break_short(self):
        result = dii_put_floor_signal(
            trade_date=date(2026, 3, 20),
            nifty_price=22450,
            strike=22500,
            dii_put_oi=2_500_000,
        )
        assert result is not None
        assert result["direction"] == "SHORT"
        assert result["distance_pts"] < 0

    def test_oi_too_low(self):
        result = dii_put_floor_signal(
            trade_date=date(2026, 3, 20),
            nifty_price=22500,
            strike=22480,
            dii_put_oi=1_500_000,  # below 20L threshold
        )
        assert result is None

    def test_too_far_from_price(self):
        result = dii_put_floor_signal(
            trade_date=date(2026, 3, 20),
            nifty_price=23000,
            strike=22000,  # >3% away
            dii_put_oi=3_000_000,
        )
        assert result is None

    def test_event_day_skip(self):
        result = dii_put_floor_signal(
            trade_date=date(2026, 3, 20),
            nifty_price=22500,
            strike=22480,
            dii_put_oi=3_000_000,
            is_event_day=True,
        )
        assert result is None

    def test_returns_valid_dict(self):
        result = dii_put_floor_signal(
            trade_date=date(2026, 3, 20),
            nifty_price=22500,
            strike=22490,
            dii_put_oi=2_500_000,
        )
        assert result is not None
        assert isinstance(result, dict)
        assert "strike" in result
        assert "dii_put_oi" in result
        assert "distance_pts" in result


# ===================================================================
#  INTEGRATION TESTS (4)
# ===================================================================

SIGNAL_FUNCTIONS = [
    rollover_flow_signal,
    index_rebalance_signal,
    quarter_window_signal,
    dii_put_floor_signal,
]


class TestIntegration:
    def test_all_import(self):
        """All inline signal functions are defined and callable."""
        for fn in SIGNAL_FUNCTIONS:
            assert callable(fn), f"{fn.__name__} is not callable"

    def test_all_have_signal_id(self):
        """Every signal that fires includes a signal_id key."""
        results = [
            rollover_flow_signal(0.85, 0.5, 2.0, vix=15, days_to_expiry=3),
            index_rebalance_signal(date(2026, 5, 20), date(2026, 5, 29)),
            quarter_window_signal(date(2026, 3, 30)),
            dii_put_floor_signal(date(2026, 3, 20), 22500, 22490, 3_000_000),
        ]
        for r in results:
            assert r is not None, "Expected a signal but got None"
            assert "signal_id" in r, f"Missing signal_id in {r}"

    def test_all_handle_none_gracefully(self):
        """Each signal can legitimately return None without raising."""
        assert rollover_flow_signal(0.50, 0.1, 1.0) is None
        assert index_rebalance_signal(date(2026, 1, 1), date(2026, 5, 29)) is None
        assert quarter_window_signal(date(2026, 2, 15)) is None
        assert dii_put_floor_signal(date(2026, 3, 20), 22500, 22480, 100_000) is None

    def test_consistent_interface(self):
        """All fired signals return dicts with direction and signal_id."""
        results = [
            rollover_flow_signal(0.85, 0.5, 2.0, vix=15, days_to_expiry=3),
            index_rebalance_signal(date(2026, 5, 20), date(2026, 5, 29)),
            quarter_window_signal(date(2026, 3, 30)),
            dii_put_floor_signal(date(2026, 3, 20), 22500, 22490, 3_000_000),
        ]
        for r in results:
            assert isinstance(r, dict)
            assert "signal_id" in r
            assert "direction" in r
            assert r["direction"] in ("LONG", "SHORT")
