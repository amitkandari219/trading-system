"""
DSL Validator: validates DSLSignalRule objects for structural correctness.

Catches the three failure modes that broke the original translator:
1. Unknown indicators (structurally impossible with DSL, but validates anyway)
2. Phantom conditions (always true)
3. Self-comparisons (always false)

Plus additional structural checks for completeness.
"""

from dataclasses import dataclass, field
from typing import List

from extraction.dsl_schema import (
    DSLSignalRule, DSLCondition,
    INDICATOR_SET, OPERATOR_SET, ALLOWED_DIRECTIONS, ALLOWED_REGIMES,
)


# Indicators that are always positive in normal markets
ALWAYS_POSITIVE = {
    'atr_7', 'atr_14', 'atr_20', 'true_range',
    'volume', 'vol_ratio_20',
    'adx_14',
    'bb_bandwidth',
    'hvol_20', 'india_vix',
    'close', 'open', 'high', 'low',
    'prev_close', 'prev_high', 'prev_low', 'prev_open',
    'upper_wick', 'lower_wick', 'range',
}

# Indicators with known bounded ranges
BOUNDED = {
    'rsi_7': (0, 100), 'rsi_14': (0, 100), 'rsi_21': (0, 100),
    'stoch_k': (0, 100), 'stoch_d': (0, 100),
    'stoch_k_5': (0, 100), 'stoch_d_5': (0, 100),
    'adx_14': (0, 100),
    'price_pos_20': (0, 1),
}


@dataclass
class ValidationResult:
    passed: bool
    issues: List[str] = field(default_factory=list)


def _is_numeric(s: str) -> bool:
    """Check if a string represents a number."""
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def _is_phantom(cond: DSLCondition) -> bool:
    """Check if a condition is always true (phantom)."""
    if not _is_numeric(cond.right):
        return False

    val = float(cond.right)
    ind = cond.left
    op = cond.operator

    # indicator > 0 for always-positive indicators
    if ind in ALWAYS_POSITIVE and op in ('>', '>=') and val <= 0:
        return True

    # Bounded indicators: rsi > 0, stoch > 0, etc.
    if ind in BOUNDED:
        lo, hi = BOUNDED[ind]
        if op in ('>', '>=') and val <= lo:
            return True
        if op in ('<', '<=') and val >= hi:
            return True

    return False


def _is_impossible(cond: DSLCondition) -> bool:
    """Check if a condition is always false."""
    # Self-comparison with strict inequality
    if cond.left == cond.right and cond.operator in ('<', '>'):
        return True

    if not _is_numeric(cond.right):
        return False

    val = float(cond.right)
    ind = cond.left

    # Bounded indicators: impossible values
    if ind in BOUNDED:
        lo, hi = BOUNDED[ind]
        if cond.operator in ('>', '>=') and val > hi:
            return True
        if cond.operator in ('<', '<=') and val < lo:
            return True

    # Negative comparisons for always-positive
    if ind in ALWAYS_POSITIVE and cond.operator in ('<', '<=') and val <= 0:
        return True

    return False


class DSLValidator:
    """Validates DSLSignalRule objects for structural correctness."""

    def validate(self, rule: DSLSignalRule) -> ValidationResult:
        """
        Validate a DSLSignalRule.

        Returns:
            ValidationResult(passed=bool, issues=list[str])
        """
        if rule.untranslatable:
            return ValidationResult(passed=True, issues=[])

        issues = []

        # (a) All left/right fields are valid indicators or numbers
        all_conditions = (
            [('entry_long', c) for c in rule.entry_long] +
            [('entry_short', c) for c in rule.entry_short] +
            [('exit_long', c) for c in rule.exit_long] +
            [('exit_short', c) for c in rule.exit_short]
        )

        for side, cond in all_conditions:
            if cond.left not in INDICATOR_SET:
                issues.append(f"{side}: unknown indicator '{cond.left}'")
            if (cond.right not in INDICATOR_SET and
                    not _is_numeric(cond.right) and
                    cond.right not in ALLOWED_REGIMES):
                issues.append(f"{side}: unknown right value '{cond.right}'")

        # (b) All operators are valid
        for side, cond in all_conditions:
            if cond.operator not in OPERATOR_SET:
                issues.append(f"{side}: unknown operator '{cond.operator}'")

        # (c) No self-comparisons
        for side, cond in all_conditions:
            if cond.left == cond.right and cond.operator in ('<', '>'):
                issues.append(f"{side}: self-comparison '{cond.left} {cond.operator} {cond.right}'")

        # (d) No phantom conditions
        for side, cond in all_conditions:
            if _is_phantom(cond):
                issues.append(
                    f"{side}: phantom condition '{cond.left} {cond.operator} {cond.right}' "
                    f"is always true"
                )

        # Check for impossible conditions too
        for side, cond in all_conditions:
            if _is_impossible(cond):
                issues.append(
                    f"{side}: impossible condition '{cond.left} {cond.operator} {cond.right}' "
                    f"is always false"
                )

        # (e) entry_long OR entry_short must have at least 1 condition
        if not rule.entry_long and not rule.entry_short:
            issues.append("no entry conditions defined")

        # (f) exit_long must have at least 1 condition (or hold_days_max must be set)
        has_exit = bool(rule.exit_long or rule.exit_short)
        has_hold = rule.hold_days_max and rule.hold_days_max > 0
        has_stop = rule.stop_loss_pct and rule.stop_loss_pct > 0

        if not has_exit and not has_hold:
            if has_stop:
                issues.append("no exit conditions or hold_days_max; relies on stop_loss only")
            else:
                issues.append("no exit conditions, no hold_days_max, no stop_loss")

        # (g) stop_loss_pct between 0.5 and 10.0
        if rule.stop_loss_pct < 0.5 or rule.stop_loss_pct > 10.0:
            issues.append(f"stop_loss_pct={rule.stop_loss_pct} outside range [0.5, 10.0]")

        # (h) hold_days_max between 1 and 30
        if rule.hold_days_max < 1 or rule.hold_days_max > 30:
            issues.append(f"hold_days_max={rule.hold_days_max} outside range [1, 30]")

        # (i) direction is valid
        if rule.direction not in ALLOWED_DIRECTIONS:
            issues.append(f"invalid direction '{rule.direction}'")

        # (j) If direction==LONG, entry_short must be empty
        if rule.direction == 'LONG' and rule.entry_short:
            issues.append("direction is LONG but entry_short has conditions")

        # (k) If direction==SHORT, entry_long must be empty
        if rule.direction == 'SHORT' and rule.entry_long:
            issues.append("direction is SHORT but entry_long has conditions")

        return ValidationResult(passed=len(issues) == 0, issues=issues)
