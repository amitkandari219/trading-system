"""
DSL to Backtest compiler.

Converts DSLSignalRule into the dict format expected by
backtest/generic_backtest.py: run_generic_backtest(rules, history_df, regime_labels).

The backtest engine expects conditions as:
  {"indicator": str, "op": str, "value": str_or_number}

This compiler maps DSLCondition → backtest condition format,
handling crosses_above/crosses_below and regime comparisons.
"""

from extraction.dsl_schema import DSLSignalRule, DSLCondition, INDICATOR_SET


def _is_numeric(s: str) -> bool:
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def _compile_condition(cond: DSLCondition) -> dict:
    """
    Convert DSLCondition to backtest engine condition dict.

    DSL: {"left": "rsi_14", "operator": "<", "right": "30"}
    Backtest: {"indicator": "rsi_14", "op": "<", "value": 30}

    DSL: {"left": "close", "operator": ">", "right": "sma_50"}
    Backtest: {"indicator": "close", "op": ">", "value": "sma_50"}
    """
    # Determine value: numeric or indicator reference
    if _is_numeric(cond.right):
        value = float(cond.right)
    else:
        value = cond.right  # indicator name or regime string

    return {
        "indicator": cond.left,
        "op": cond.operator,
        "value": value,
    }


def _compile_conditions(conditions: list) -> list:
    """Compile a list of DSLConditions to backtest format."""
    return [_compile_condition(c) for c in conditions]


class DSLToBacktest:
    """Compiles DSLSignalRule into backtest engine format."""

    def compile(self, rule: DSLSignalRule) -> dict:
        """
        Convert DSLSignalRule to backtest rules dict.

        Args:
            rule: validated DSLSignalRule

        Returns:
            dict compatible with run_generic_backtest()

        Raises:
            ValueError if rule is untranslatable
        """
        if rule.untranslatable:
            raise ValueError(
                f"Cannot compile untranslatable rule: {rule.untranslatable_reason}"
            )

        # Compile conditions
        entry_long = _compile_conditions(rule.entry_long)
        entry_short = _compile_conditions(rule.entry_short)
        exit_long = _compile_conditions(rule.exit_long)
        exit_short = _compile_conditions(rule.exit_short)

        # Map regime filter
        regime_filter = []
        if rule.target_regime and rule.target_regime != ['ANY']:
            regime_filter = [r for r in rule.target_regime if r != 'ANY']

        return {
            "backtestable": True,
            "entry_long": entry_long,
            "entry_short": entry_short,
            "exit_long": exit_long,
            "exit_short": exit_short,
            "regime_filter": regime_filter,
            "hold_days": rule.hold_days_max,
            "stop_loss_pct": rule.stop_loss_pct / 100.0,  # DSL uses percent, engine uses fraction
            "take_profit_pct": 0,
            "direction": rule.direction,
            "reason": rule.translation_notes or f"DSL translation of {rule.signal_id}",
        }
