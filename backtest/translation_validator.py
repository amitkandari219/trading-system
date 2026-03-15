"""
Translation Validator: catches broken translations before backtesting.

Detects three failure modes from Haiku's signal translation:
1. PHANTOM — conditions that are always true (indicator > 0 for always-positive indicators)
2. BROKEN_EXIT — self-comparisons that are always false (exit never fires)
3. OVERSIMPLIFIED — too few real entry conditions to constitute a genuine signal

Also detects:
4. UNKNOWN_INDICATOR — references columns that don't exist in the indicator set
5. IMPOSSIBLE_CONDITION — logically impossible bounds (rsi > 100, price_pos_20 > 1)
6. REDUNDANT_EXIT — exit conditions identical to entry (signal can never hold)
7. NO_EXIT — no exit conditions and no hold_days (position held forever until stop)
"""

import json
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# All valid indicator column names produced by add_all_indicators()
VALID_INDICATORS = {
    'open', 'high', 'low', 'close', 'volume', 'date',
    # Moving averages
    *[f'sma_{p}' for p in [5, 10, 20, 40, 50, 80, 100, 200]],
    *[f'ema_{p}' for p in [5, 10, 20, 50, 100, 200]],
    # RSI
    'rsi_7', 'rsi_14', 'rsi_21',
    # MACD
    'macd', 'macd_signal', 'macd_hist',
    # ATR
    'atr_7', 'atr_14', 'atr_20', 'true_range',
    # Bollinger
    'bb_upper', 'bb_middle', 'bb_lower', 'bb_pct_b', 'bb_bandwidth',
    # Stochastic
    'stoch_k', 'stoch_d', 'stoch_k_5', 'stoch_d_5',
    # ADX
    'adx_14',
    # Donchian
    'dc_upper', 'dc_lower', 'dc_middle',
    # Volume
    'vol_ratio_20',
    # Volatility
    'hvol_6', 'hvol_20', 'hvol_100', 'india_vix',
    # Pivots
    'pivot', 'r1', 's1', 'r2', 's2', 'r3', 's3',
    # Price position
    'price_pos_20',
    # Previous bar
    'prev_close', 'prev_high', 'prev_low', 'prev_open', 'prev_volume',
    # Returns
    'returns', 'log_returns',
    # Bar properties
    'body', 'body_pct', 'upper_wick', 'lower_wick', 'range',
    # Regime (string column)
    'regime',
}

# Indicators that are always positive (or non-negative) in normal markets.
# Comparing these > 0 or >= 0 is a phantom condition.
ALWAYS_POSITIVE_INDICATORS = {
    'atr_7', 'atr_14', 'atr_20', 'true_range',
    'volume', 'vol_ratio_20',
    'adx_14',
    'bb_bandwidth',
    'hvol_6', 'hvol_20', 'hvol_100', 'india_vix',
    'close', 'open', 'high', 'low',
    'upper_wick', 'lower_wick', 'range',
}

# Indicators with known bounded ranges
INDICATOR_BOUNDS = {
    'rsi_7': (0, 100), 'rsi_14': (0, 100), 'rsi_21': (0, 100),
    'stoch_k': (0, 100), 'stoch_d': (0, 100),
    'adx_14': (0, 100),
    'price_pos_20': (0, 1),
    'bb_pct_b': (-0.5, 1.5),  # can exceed 0-1 but not by huge amounts
}


def _is_phantom(cond: dict) -> bool:
    """Check if a condition is always true (phantom)."""
    indicator = cond.get('indicator', '')
    op = cond.get('op', '')
    value = cond.get('value')

    if isinstance(value, str):
        return False  # comparing to another indicator — not trivially phantom

    if value is None or not isinstance(value, (int, float)):
        return False

    value = float(value)

    # indicator > 0 or >= 0 for always-positive indicators
    if indicator in ALWAYS_POSITIVE_INDICATORS:
        if op in ('>', '>=') and value <= 0:
            return True
        if op == '>' and value == 0 and indicator != 'volume':
            # volume can be 0 on holidays, but atr/adx/close are always > 0
            return True

    # rsi > 0, adx > 0, stoch > 0 — always true for bounded indicators
    if indicator in INDICATOR_BOUNDS:
        lo, hi = INDICATOR_BOUNDS[indicator]
        if op in ('>', '>=') and value <= lo:
            return True
        if op in ('<', '<=') and value >= hi:
            return True

    return False


def _is_impossible(cond: dict) -> Optional[str]:
    """Check if a condition is logically impossible (always false)."""
    indicator = cond.get('indicator', '')
    op = cond.get('op', '')
    value = cond.get('value')

    if isinstance(value, str):
        # Self-comparison: indicator op indicator (always false for < or >)
        if value == indicator and op in ('<', '>'):
            return f"self-comparison '{indicator} {op} {indicator}' is always false"
        return None

    if value is None or not isinstance(value, (int, float)):
        return None

    value = float(value)

    # Check bounded indicators for impossible values
    if indicator in INDICATOR_BOUNDS:
        lo, hi = INDICATOR_BOUNDS[indicator]
        if op in ('>', '>=') and value > hi:
            return f"'{indicator} {op} {value}' impossible (max is {hi})"
        if op in ('<', '<=') and value < lo:
            return f"'{indicator} {op} {value}' impossible (min is {lo})"

    # Negative comparisons for always-positive indicators
    if indicator in ALWAYS_POSITIVE_INDICATORS:
        if op in ('<', '<=') and value <= 0:
            return f"'{indicator} {op} {value}' impossible (always positive)"

    return None


def _is_unknown_indicator(cond: dict) -> Optional[str]:
    """Check if a condition references an unknown indicator."""
    indicator = cond.get('indicator', '')
    value = cond.get('value')

    issues = []
    if indicator and indicator not in VALID_INDICATORS and indicator not in ('True', 'False'):
        issues.append(f"unknown indicator '{indicator}'")
    if isinstance(value, str) and value not in VALID_INDICATORS:
        issues.append(f"unknown value column '{value}'")

    return '; '.join(issues) if issues else None


def validate_signal(translated: dict) -> List[Dict[str, str]]:
    """
    Validate a single translated signal's rules.

    Args:
        translated: dict with 'signal_id', 'backtestable', 'rules' keys
                    where rules contains entry_long/short, exit_long/short, etc.

    Returns:
        List of issue dicts: [{"type": "PHANTOM", "severity": "HIGH", "detail": "..."}]
    """
    if not translated.get('backtestable') or not translated.get('rules'):
        return []

    rules = translated['rules']
    issues = []

    entry_long = rules.get('entry_long', [])
    entry_short = rules.get('entry_short', [])
    exit_long = rules.get('exit_long', [])
    exit_short = rules.get('exit_short', [])
    hold_days = rules.get('hold_days', 0)
    direction = rules.get('direction', 'BOTH')

    # Collect all conditions for cross-checks
    all_entries = []
    if direction in ('BOTH', 'LONG', 'CONTEXT_DEPENDENT'):
        all_entries.extend(('entry_long', c) for c in entry_long)
    if direction in ('BOTH', 'SHORT', 'CONTEXT_DEPENDENT'):
        all_entries.extend(('entry_short', c) for c in entry_short)

    all_exits = []
    if direction in ('BOTH', 'LONG', 'CONTEXT_DEPENDENT'):
        all_exits.extend(('exit_long', c) for c in exit_long)
    if direction in ('BOTH', 'SHORT', 'CONTEXT_DEPENDENT'):
        all_exits.extend(('exit_short', c) for c in exit_short)

    all_conditions = [(side, c) for side, c in all_entries] + [(side, c) for side, c in all_exits]

    # --- CHECK 1: PHANTOM CONDITIONS (always true) ---
    phantom_count = 0
    for side, cond in all_conditions:
        if _is_phantom(cond):
            indicator = cond.get('indicator', '')
            op = cond.get('op', '')
            value = cond.get('value', '')
            issues.append({
                'type': 'PHANTOM',
                'severity': 'HIGH',
                'detail': f"{side}: '{indicator} {op} {value}' is always true",
            })
            phantom_count += 1

    # --- CHECK 2: BROKEN CONDITIONS (always false / impossible) ---
    for side, cond in all_conditions:
        reason = _is_impossible(cond)
        if reason:
            issues.append({
                'type': 'BROKEN_EXIT' if 'exit' in side else 'BROKEN_ENTRY',
                'severity': 'CRITICAL',
                'detail': f"{side}: {reason}",
            })

    # --- CHECK 3: OVERSIMPLIFICATION (too few real entry conditions) ---
    for side_name, conditions in [('entry_long', entry_long), ('entry_short', entry_short)]:
        if not conditions:
            continue
        # Skip if this direction isn't used
        if side_name == 'entry_long' and direction in ('SHORT',):
            continue
        if side_name == 'entry_short' and direction in ('LONG',):
            continue

        real_conditions = [c for c in conditions if not _is_phantom(c)]
        if len(real_conditions) < 2:
            issues.append({
                'type': 'OVERSIMPLIFIED',
                'severity': 'MEDIUM',
                'detail': f"{side_name}: only {len(real_conditions)} real condition(s) "
                          f"(total {len(conditions)}, {len(conditions) - len(real_conditions)} phantom)",
            })

    # --- CHECK 4: UNKNOWN INDICATORS ---
    for side, cond in all_conditions:
        reason = _is_unknown_indicator(cond)
        if reason:
            issues.append({
                'type': 'UNKNOWN_INDICATOR',
                'severity': 'CRITICAL',
                'detail': f"{side}: {reason}",
            })

    # --- CHECK 5: NO EXIT (no exit conditions AND no hold_days) ---
    has_signal_exit = bool(exit_long or exit_short)
    has_hold_days = hold_days and hold_days > 0
    stop_loss = rules.get('stop_loss_pct', 0)
    take_profit = rules.get('take_profit_pct', 0)

    if not has_signal_exit and not has_hold_days:
        # Only stop_loss/take_profit as exit — might be intentional but worth flagging
        if stop_loss > 0 or take_profit > 0:
            issues.append({
                'type': 'NO_SIGNAL_EXIT',
                'severity': 'LOW',
                'detail': f"no exit conditions or hold_days; relies entirely on "
                          f"stop_loss={stop_loss}/take_profit={take_profit}",
            })
        else:
            issues.append({
                'type': 'NO_EXIT',
                'severity': 'CRITICAL',
                'detail': "no exit conditions, no hold_days, no stop_loss — position held forever",
            })

    # --- CHECK 6: REDUNDANT EXIT (exit == entry, signal never holds) ---
    for entry_side, exit_side, entry_conds, exit_conds in [
        ('entry_long', 'exit_long', entry_long, exit_long),
        ('entry_short', 'exit_short', entry_short, exit_short),
    ]:
        if not entry_conds or not exit_conds:
            continue
        # Check if any exit condition directly contradicts an entry condition
        # (same indicator, opposite direction → fires on same bar)
        for ec in entry_conds:
            for xc in exit_conds:
                if (ec.get('indicator') == xc.get('indicator') and
                        ec.get('value') == xc.get('value')):
                    e_op = ec.get('op', '')
                    x_op = xc.get('op', '')
                    # Same indicator, same value, same direction = exit fires immediately
                    if e_op == x_op:
                        issues.append({
                            'type': 'REDUNDANT_EXIT',
                            'severity': 'HIGH',
                            'detail': f"{exit_side} has same condition as {entry_side}: "
                                      f"'{ec.get('indicator')} {e_op} {ec.get('value')}'",
                        })

    # --- CHECK 7: ALL CONDITIONS PHANTOM (entire signal is meaningless) ---
    for side_name, conditions in [('entry_long', entry_long), ('entry_short', entry_short)]:
        if not conditions:
            continue
        if all(_is_phantom(c) for c in conditions):
            issues.append({
                'type': 'ALL_PHANTOM',
                'severity': 'CRITICAL',
                'detail': f"{side_name}: ALL {len(conditions)} conditions are phantom — "
                          f"signal enters on every bar",
            })

    return issues


def validate_all(translated_signals: List[dict]) -> Dict[str, Any]:
    """
    Run validation on all translated signals.

    Returns:
        {
            "total_backtestable": int,
            "signals_with_issues": int,
            "signals_clean": int,
            "issue_counts": {"PHANTOM": N, "BROKEN_EXIT": N, ...},
            "severity_counts": {"CRITICAL": N, "HIGH": N, "MEDIUM": N, "LOW": N},
            "results": [{"signal_id": ..., "issues": [...], "verdict": "PASS"|"FAIL"}, ...]
        }
    """
    backtestable = [s for s in translated_signals if s.get('backtestable') and s.get('rules')]

    results = []
    issue_counts = {}
    severity_counts = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
    signals_with_issues = 0

    for signal in backtestable:
        issues = validate_signal(signal)
        has_critical = any(i['severity'] == 'CRITICAL' for i in issues)
        has_high = any(i['severity'] == 'HIGH' for i in issues)

        # Separate UNKNOWN_INDICATOR (fixable syntax errors) from true failures.
        # A signal whose ONLY critical issues are UNKNOWN_INDICATOR gets FIXABLE,
        # not FAIL — the concepts are real, just the column syntax is wrong.
        has_unknown = any(i['type'] == 'UNKNOWN_INDICATOR' for i in issues)
        non_unknown_critical = any(
            i['severity'] == 'CRITICAL' and i['type'] != 'UNKNOWN_INDICATOR'
            for i in issues
        )

        if non_unknown_critical:
            verdict = 'FAIL'
        elif has_unknown and has_critical:
            # Only critical issues are UNKNOWN_INDICATOR — fixable
            verdict = 'FIXABLE'
        elif has_high:
            verdict = 'WARN'
        elif issues:
            verdict = 'REVIEW'
        else:
            verdict = 'PASS'

        if issues:
            signals_with_issues += 1

        for issue in issues:
            issue_counts[issue['type']] = issue_counts.get(issue['type'], 0) + 1
            severity_counts[issue['severity']] += 1

        results.append({
            'signal_id': signal['signal_id'],
            'verdict': verdict,
            'issue_count': len(issues),
            'issues': issues,
        })

    return {
        'total_backtestable': len(backtestable),
        'signals_with_issues': signals_with_issues,
        'signals_clean': len(backtestable) - signals_with_issues,
        'issue_counts': issue_counts,
        'severity_counts': severity_counts,
        'results': results,
    }
