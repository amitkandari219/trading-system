"""Tests for the ScoringEngine – 10 scenarios covering score rules,
position sizing, exits, and conflict handling."""

import pytest

from paper_trading.scoring_engine import ScoringEngine


def _make_signals(dry20=None, dry12=None, dry16=None) -> dict:
    """Helper to build a signals dict."""
    return {
        "DRY_20": {"action": dry20},
        "DRY_12": {"action": dry12},
        "DRY_16": {"action": dry16},
    }


# 1. DRY_20 alone fires -> score = 2 -> LONG half size
def test_dry20_alone_long_half():
    engine = ScoringEngine()
    result = engine.update(_make_signals(dry20="ENTER_LONG"))
    assert result["score"] == 2
    assert result["action"] == "ENTER_LONG"
    assert result["size"] == 0.5


# 2. DRY_20 + DRY_12 agree LONG -> score = 3 -> LONG full size
def test_dry20_and_dry12_long_full():
    engine = ScoringEngine()
    result = engine.update(
        _make_signals(dry20="ENTER_LONG", dry12="ENTER_LONG")
    )
    assert result["score"] == 3
    assert result["action"] == "ENTER_LONG"
    assert result["size"] == 1.0


# 3. DRY_20 LONG + DRY_12 SHORT -> score = 1 -> no action
def test_dry20_long_dry12_short_no_action():
    engine = ScoringEngine()
    result = engine.update(
        _make_signals(dry20="ENTER_LONG", dry12="ENTER_SHORT")
    )
    assert result["score"] == 1
    assert result["action"] is None


# 4. DRY_16 SHORT alone -> score = -1 -> no action
def test_dry16_short_alone_no_action():
    engine = ScoringEngine()
    result = engine.update(_make_signals(dry16="ENTER_SHORT"))
    assert result["score"] == -1
    assert result["action"] is None


# 5. DRY_16 SHORT + DRY_12 SHORT -> score = -2 -> SHORT half
def test_dry16_and_dry12_short_half():
    engine = ScoringEngine()
    result = engine.update(
        _make_signals(dry12="ENTER_SHORT", dry16="ENTER_SHORT")
    )
    assert result["score"] == -2
    assert result["action"] == "ENTER_SHORT"
    assert result["size"] == 0.5


# 6. DRY_20 EXIT -> score resets to 0
def test_dry20_exit_resets_score():
    engine = ScoringEngine()
    # First enter a position
    engine.update(_make_signals(dry20="ENTER_LONG", dry12="ENTER_LONG"))
    assert engine.score == 3
    # DRY_20 EXIT resets
    result = engine.update(_make_signals(dry20="EXIT"))
    assert result["score"] == 0
    assert result["action"] == "EXIT"


# 7. Already LONG, SHORT signal fires -> no new SHORT (position conflict)
def test_already_long_no_short():
    engine = ScoringEngine()
    # Enter LONG
    engine.update(_make_signals(dry20="ENTER_LONG"))
    assert engine.position == "LONG"
    # Now signals want SHORT – but we're already LONG, and score = -2
    # which would normally trigger SHORT, but conflict prevents it.
    # First we need the LONG to exit. Let's set up the conflict scenario:
    # We force the engine into LONG, then feed SHORT signals.
    engine.score = 0  # reset for clean test
    engine.position = "LONG"
    engine.entry_threshold = 2
    engine.size = 0.5
    result = engine.update(
        _make_signals(dry12="ENTER_SHORT", dry16="ENTER_SHORT")
    )
    # score = -2 would normally SHORT, but already LONG -> EXIT first
    assert result["action"] == "EXIT"


# 8. Score drops below entry threshold -> EXIT
def test_score_drops_below_threshold_exit():
    engine = ScoringEngine()
    # Enter LONG full at score 3 (threshold = 3)
    result = engine.update(
        _make_signals(dry20="ENTER_LONG", dry12="ENTER_LONG")
    )
    assert result["action"] == "ENTER_LONG"
    assert result["size"] == 1.0
    assert engine.entry_threshold == 3
    # Next day: only DRY_20 fires -> score = 2, below threshold 3 -> EXIT
    result = engine.update(_make_signals(dry20="ENTER_LONG"))
    assert result["score"] == 2
    assert result["action"] == "EXIT"


# 9. All three agree LONG -> score = 4 -> LONG full size (capped at 1.0x)
def test_all_three_long_full_capped():
    engine = ScoringEngine()
    result = engine.update(
        _make_signals(
            dry20="ENTER_LONG", dry12="ENTER_LONG", dry16="ENTER_LONG"
        )
    )
    assert result["score"] == 4
    assert result["action"] == "ENTER_LONG"
    assert result["size"] == 1.0


# 10. Full sequence: enter on score 3, score drops to 1, exits
def test_full_sequence_enter_drop_exit():
    engine = ScoringEngine()

    # Day 1: DRY_20 + DRY_12 LONG -> score 3 -> ENTER_LONG full
    r1 = engine.update(
        _make_signals(dry20="ENTER_LONG", dry12="ENTER_LONG")
    )
    assert r1["score"] == 3
    assert r1["action"] == "ENTER_LONG"
    assert r1["size"] == 1.0
    assert engine.position == "LONG"
    assert engine.entry_threshold == 3

    # Day 2: only DRY_12 fires LONG -> score 1 -> below threshold 3 -> EXIT
    r2 = engine.update(_make_signals(dry12="ENTER_LONG"))
    assert r2["score"] == 1
    assert r2["prev_score"] == 3
    assert r2["action"] == "EXIT"
    assert engine.position is None

    # Day 3: no signals -> score 0 -> no action, flat
    r3 = engine.update(_make_signals())
    assert r3["score"] == 0
    assert r3["action"] is None
    summary = engine.get_daily_summary()
    assert "FLAT" in summary
