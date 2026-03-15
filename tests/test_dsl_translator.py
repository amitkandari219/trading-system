"""
Tests for the DSL translation pipeline.

Tests validation, compilation, and structural correctness.
LLM-dependent tests (actual translation) are in test 10.
"""

import pytest

from extraction.dsl_schema import DSLSignalRule, DSLCondition, INDICATOR_SET, OPERATOR_SET
from extraction.dsl_validator import DSLValidator, ValidationResult
from extraction.dsl_to_backtest import DSLToBacktest


@pytest.fixture
def validator():
    return DSLValidator()


@pytest.fixture
def compiler():
    return DSLToBacktest()


# --- Test 1: Valid RSI oversold + trend signal → PASS ---

def test_valid_rsi_trend_signal(validator, compiler):
    """Valid RSI oversold + trend signal → PASS, compiles correctly."""
    rule = DSLSignalRule(
        signal_id="TEST_001",
        entry_long=[
            DSLCondition(left="rsi_14", operator="<", right="30"),
            DSLCondition(left="close", operator=">", right="sma_50"),
        ],
        exit_long=[
            DSLCondition(left="rsi_14", operator=">", right="70"),
        ],
        stop_loss_pct=2.0,
        hold_days_max=10,
        direction="LONG",
        target_regime=["TRENDING_UP", "RANGING"],
    )

    result = validator.validate(rule)
    assert result.passed, f"Should pass, but got issues: {result.issues}"

    compiled = compiler.compile(rule)
    assert compiled["backtestable"] is True
    assert len(compiled["entry_long"]) == 2
    assert compiled["entry_long"][0]["indicator"] == "rsi_14"
    assert compiled["entry_long"][0]["op"] == "<"
    assert compiled["entry_long"][0]["value"] == 30.0
    assert compiled["entry_long"][1]["value"] == "sma_50"
    assert compiled["direction"] == "LONG"
    assert compiled["regime_filter"] == ["TRENDING_UP", "RANGING"]


# --- Test 2: Self-comparison → validator catches it ---

def test_self_comparison(validator):
    """Self-comparison sma_50 > sma_50 → validator catches it."""
    rule = DSLSignalRule(
        signal_id="TEST_002",
        entry_long=[
            DSLCondition(left="sma_50", operator=">", right="sma_50"),
        ],
        exit_long=[
            DSLCondition(left="rsi_14", operator=">", right="70"),
        ],
        stop_loss_pct=2.0,
        hold_days_max=10,
        direction="LONG",
    )

    result = validator.validate(rule)
    assert not result.passed
    assert any("self-comparison" in i for i in result.issues)


# --- Test 3: Phantom condition → validator catches it ---

def test_phantom_condition(validator):
    """Phantom condition close > 0 → validator catches it."""
    rule = DSLSignalRule(
        signal_id="TEST_003",
        entry_long=[
            DSLCondition(left="close", operator=">", right="0"),
            DSLCondition(left="rsi_14", operator="<", right="30"),
        ],
        exit_long=[
            DSLCondition(left="rsi_14", operator=">", right="70"),
        ],
        stop_loss_pct=2.0,
        hold_days_max=10,
        direction="LONG",
    )

    result = validator.validate(rule)
    assert not result.passed
    assert any("phantom" in i for i in result.issues)


# --- Test 4: Unknown indicator → untranslatable ---

def test_unknown_indicator_untranslatable():
    """Unknown indicator PRIOR_SWING_HIGH → should be caught by validator."""
    rule = DSLSignalRule(
        signal_id="TEST_004",
        entry_long=[
            DSLCondition(left="close", operator=">", right="PRIOR_SWING_HIGH"),
        ],
        exit_long=[
            DSLCondition(left="rsi_14", operator=">", right="70"),
        ],
        stop_loss_pct=2.0,
        hold_days_max=10,
        direction="LONG",
    )

    validator = DSLValidator()
    result = validator.validate(rule)
    assert not result.passed
    assert any("unknown" in i for i in result.issues)


# --- Test 5: crosses_above compiles correctly ---

def test_crosses_above_compiles(compiler):
    """crosses_above compiles to correct backtest format with shift(1)."""
    rule = DSLSignalRule(
        signal_id="TEST_005",
        entry_long=[
            DSLCondition(left="close", operator="crosses_above", right="sma_50"),
        ],
        exit_long=[
            DSLCondition(left="close", operator="crosses_below", right="sma_50"),
        ],
        stop_loss_pct=2.0,
        hold_days_max=10,
        direction="LONG",
    )

    compiled = compiler.compile(rule)
    entry = compiled["entry_long"][0]
    assert entry["indicator"] == "close"
    assert entry["op"] == "crosses_above"
    assert entry["value"] == "sma_50"

    # The backtest engine's _eval_condition handles crosses_above natively
    # by checking prev_row <= target and current_row > target
    exit_cond = compiled["exit_long"][0]
    assert exit_cond["op"] == "crosses_below"


# --- Test 6: Regime condition ---

def test_regime_target(validator, compiler):
    """Regime condition compiles to string comparison."""
    rule = DSLSignalRule(
        signal_id="TEST_006",
        entry_long=[
            DSLCondition(left="rsi_14", operator="<", right="30"),
        ],
        exit_long=[
            DSLCondition(left="rsi_14", operator=">", right="70"),
        ],
        stop_loss_pct=2.0,
        hold_days_max=10,
        direction="LONG",
        target_regime=["TRENDING_UP"],
    )

    result = validator.validate(rule)
    assert result.passed

    compiled = compiler.compile(rule)
    assert compiled["regime_filter"] == ["TRENDING_UP"]


# --- Test 7: BOTH direction signal ---

def test_both_direction(validator):
    """BOTH direction signal has both entry_long and entry_short."""
    rule = DSLSignalRule(
        signal_id="TEST_007",
        entry_long=[
            DSLCondition(left="rsi_14", operator="<", right="30"),
        ],
        entry_short=[
            DSLCondition(left="rsi_14", operator=">", right="70"),
        ],
        exit_long=[
            DSLCondition(left="rsi_14", operator=">", right="50"),
        ],
        exit_short=[
            DSLCondition(left="rsi_14", operator="<", right="50"),
        ],
        stop_loss_pct=2.0,
        hold_days_max=10,
        direction="BOTH",
    )

    result = validator.validate(rule)
    assert result.passed


# --- Test 8: Missing exit with hold_days_max set → PASS ---

def test_missing_exit_with_hold_days(validator):
    """Missing exit with hold_days_max set → validator PASS."""
    rule = DSLSignalRule(
        signal_id="TEST_008",
        entry_long=[
            DSLCondition(left="close", operator="crosses_above", right="sma_200"),
        ],
        stop_loss_pct=2.0,
        hold_days_max=5,
        direction="LONG",
    )

    result = validator.validate(rule)
    assert result.passed, f"Should pass with hold_days_max set, but: {result.issues}"


# --- Test 9: Missing exit without hold_days_max → FAIL ---

def test_missing_exit_without_hold_days(validator):
    """Missing exit without hold_days_max → validator FAIL."""
    rule = DSLSignalRule(
        signal_id="TEST_009",
        entry_long=[
            DSLCondition(left="close", operator="crosses_above", right="sma_200"),
        ],
        stop_loss_pct=2.0,
        hold_days_max=0,  # no hold days
        direction="LONG",
    )

    result = validator.validate(rule)
    assert not result.passed
    assert any("hold_days_max" in i or "no exit" in i for i in result.issues)


# --- Test 10: Full end-to-end ---

def test_full_end_to_end(validator, compiler):
    """Full pipeline: DSLSignalRule → validate → compile → no errors."""
    # Simulate what Haiku would produce for a Kaufman-style signal
    rule = DSLSignalRule(
        signal_id="KAUFMAN_TEST",
        entry_long=[
            DSLCondition(left="sma_10", operator="<", right="prev_close"),
            DSLCondition(left="stoch_k_5", operator=">", right="50"),
        ],
        exit_long=[
            DSLCondition(left="stoch_k_5", operator="<=", right="50"),
        ],
        stop_loss_pct=2.0,
        hold_days_max=10,
        direction="LONG",
        target_regime=["ANY"],
        translation_notes="Kaufman genetic algo rule: 10-day SMA below prev close, 5-day stoch > 50",
    )

    # Validate
    result = validator.validate(rule)
    assert result.passed, f"Validation failed: {result.issues}"

    # Compile
    compiled = compiler.compile(rule)
    assert compiled["backtestable"] is True
    assert len(compiled["entry_long"]) == 2
    assert compiled["entry_long"][0]["indicator"] == "sma_10"
    assert compiled["entry_long"][0]["value"] == "prev_close"
    assert compiled["entry_long"][1]["indicator"] == "stoch_k_5"
    assert compiled["entry_long"][1]["value"] == 50.0
    assert compiled["exit_long"][0]["indicator"] == "stoch_k_5"
    assert compiled["stop_loss_pct"] == 0.02  # 2% converted to fraction
    assert compiled["hold_days"] == 10
    assert compiled["direction"] == "LONG"
    assert compiled["regime_filter"] == []  # ANY means no filter


# --- Additional edge case tests ---

def test_direction_long_with_short_entries_fails(validator):
    """Direction LONG but entry_short defined → FAIL."""
    rule = DSLSignalRule(
        signal_id="TEST_EDGE_1",
        entry_long=[DSLCondition(left="rsi_14", operator="<", right="30")],
        entry_short=[DSLCondition(left="rsi_14", operator=">", right="70")],
        exit_long=[DSLCondition(left="rsi_14", operator=">", right="50")],
        stop_loss_pct=2.0,
        hold_days_max=10,
        direction="LONG",
    )

    result = validator.validate(rule)
    assert not result.passed
    assert any("direction is LONG but entry_short" in i for i in result.issues)


def test_stop_loss_out_of_range(validator):
    """Stop loss outside valid range → caught."""
    rule = DSLSignalRule(
        signal_id="TEST_EDGE_2",
        entry_long=[DSLCondition(left="rsi_14", operator="<", right="30")],
        exit_long=[DSLCondition(left="rsi_14", operator=">", right="70")],
        stop_loss_pct=0.1,  # too low
        hold_days_max=10,
        direction="LONG",
    )

    result = validator.validate(rule)
    assert not result.passed
    assert any("stop_loss_pct" in i for i in result.issues)


def test_impossible_rsi_value(validator):
    """RSI > 200 is impossible → caught."""
    rule = DSLSignalRule(
        signal_id="TEST_EDGE_3",
        entry_long=[DSLCondition(left="rsi_14", operator=">", right="200")],
        exit_long=[DSLCondition(left="rsi_14", operator="<", right="50")],
        stop_loss_pct=2.0,
        hold_days_max=10,
        direction="LONG",
    )

    result = validator.validate(rule)
    assert not result.passed
    assert any("impossible" in i for i in result.issues)


def test_untranslatable_rule_compiles_raises(compiler):
    """Untranslatable rule raises ValueError on compile."""
    rule = DSLSignalRule(
        signal_id="TEST_EDGE_4",
        untranslatable=True,
        untranslatable_reason="Concept requires intraday data",
    )

    with pytest.raises(ValueError, match="untranslatable"):
        compiler.compile(rule)


def test_all_indicators_in_set():
    """Verify all indicators in schema are valid."""
    from extraction.dsl_schema import ALLOWED_INDICATORS
    assert len(ALLOWED_INDICATORS) == len(set(ALLOWED_INDICATORS)), "Duplicate indicators"
    assert "close" in INDICATOR_SET
    assert "prev_close" in INDICATOR_SET
    assert "stoch_k_5" in INDICATOR_SET
    assert "PRIOR_SWING_HIGH" not in INDICATOR_SET


def test_all_operators_in_set():
    """Verify all operators in schema are valid."""
    from extraction.dsl_schema import ALLOWED_OPERATORS
    assert "crosses_above" in OPERATOR_SET
    assert "crosses_below" in OPERATOR_SET
    assert "is" in OPERATOR_SET
    assert "!=" not in OPERATOR_SET
