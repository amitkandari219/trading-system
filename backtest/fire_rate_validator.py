"""
Fire-rate validator for DSL signals.

Flags signals that fire on < 2% of bars as OVER_CONSTRAINED.
Optionally auto-relaxes by dropping the tightest AND condition
or removing signal-level regime filters.

Usage:
    from backtest.fire_rate_validator import validate_fire_rate, auto_relax

    result = validate_fire_rate(rules, df)
    if result['fire_rate'] < 0.02:
        relaxed = auto_relax(rules, df)
"""

import logging
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backtest.generic_backtest import run_generic_backtest, _eval_condition
from backtest.indicators import add_all_indicators

logger = logging.getLogger(__name__)

MIN_FIRE_RATE = 0.02       # 2% of bars
MIN_TRADES_VIABLE = 15     # minimum trades over full history


def validate_fire_rate(rules: dict, df: pd.DataFrame) -> dict:
    """
    Compute fire rate and classify signal health.

    Returns:
        {
            'fire_rate': float,        # % of bars where entry fires
            'trade_count': int,        # actual trades in backtest
            'status': str,             # VIABLE, OVER_CONSTRAINED, INCOMPLETE, BROKEN
            'issues': list[str],       # specific problems found
            'per_condition_rates': list # fire rate of each individual condition
        }
    """
    issues = []

    # Check for missing entry conditions
    entry_long = rules.get('entry_long', [])
    entry_short = rules.get('entry_short', [])
    direction = rules.get('direction', 'BOTH')

    if direction in ('BOTH', 'LONG') and not entry_long:
        issues.append('INCOMPLETE: direction includes LONG but entry_long is empty')
    if direction in ('BOTH', 'SHORT') and not entry_short:
        issues.append('INCOMPLETE: direction includes SHORT but entry_short is empty')

    if not entry_long and not entry_short:
        return {
            'fire_rate': 0.0,
            'trade_count': 0,
            'status': 'BROKEN',
            'issues': ['No entry conditions defined'],
            'per_condition_rates': [],
        }

    # Compute indicators
    df_ind = add_all_indicators(df)
    n_bars = len(df_ind)

    # Compute per-condition fire rates
    per_cond_rates = []
    for side_name, conditions in [('entry_long', entry_long), ('entry_short', entry_short)]:
        for i, cond in enumerate(conditions):
            fires = 0
            for idx in range(1, n_bars):
                row = df_ind.iloc[idx]
                prev = df_ind.iloc[idx - 1]
                try:
                    if _eval_condition(row, prev, cond):
                        fires += 1
                except Exception:
                    pass
            rate = fires / n_bars if n_bars > 0 else 0
            per_cond_rates.append({
                'side': side_name,
                'index': i,
                'condition': cond,
                'fire_rate': round(rate, 4),
                'fires': fires,
            })

    # Compute combined fire rate (all AND conditions met)
    combined_fires_long = 0
    combined_fires_short = 0

    for idx in range(1, n_bars):
        row = df_ind.iloc[idx]
        prev = df_ind.iloc[idx - 1]

        if entry_long:
            all_met = True
            for cond in entry_long:
                try:
                    if not _eval_condition(row, prev, cond):
                        all_met = False
                        break
                except Exception:
                    all_met = False
                    break
            if all_met:
                combined_fires_long += 1

        if entry_short:
            all_met = True
            for cond in entry_short:
                try:
                    if not _eval_condition(row, prev, cond):
                        all_met = False
                        break
                except Exception:
                    all_met = False
                    break
            if all_met:
                combined_fires_short += 1

    total_fires = combined_fires_long + combined_fires_short
    fire_rate = total_fires / n_bars if n_bars > 0 else 0

    # Run actual backtest for trade count
    try:
        result = run_generic_backtest(rules, df, {})
        trade_count = result.trade_count
    except Exception:
        trade_count = 0

    # Classify
    if issues and any('INCOMPLETE' in i for i in issues):
        status = 'INCOMPLETE'
    elif fire_rate < 0.001:
        status = 'BROKEN'
        issues.append(f'Fire rate {fire_rate:.4f} = virtually never fires')
    elif fire_rate < MIN_FIRE_RATE:
        status = 'OVER_CONSTRAINED'
        issues.append(f'Fire rate {fire_rate:.3f} < {MIN_FIRE_RATE} threshold')
    elif trade_count < MIN_TRADES_VIABLE:
        status = 'LOW_TRADES'
        issues.append(f'Only {trade_count} trades (need {MIN_TRADES_VIABLE}+)')
    else:
        status = 'VIABLE'

    # Flag specific problems
    if len(entry_long) >= 3 or len(entry_short) >= 3:
        issues.append(f'AND-overfit: {len(entry_long)} long + {len(entry_short)} short conditions')

    # Check for redundant regime filter
    regime_filter = rules.get('regime_filter', [])
    if regime_filter:
        has_adx = any(c.get('indicator', c.get('left', '')) == 'adx_14'
                      for c in entry_long + entry_short)
        if has_adx and any('TRENDING' in r for r in regime_filter):
            issues.append('Regime conflict: ADX condition + TRENDING regime filter = double-filter')

    return {
        'fire_rate': round(fire_rate, 4),
        'trade_count': trade_count,
        'status': status,
        'issues': issues,
        'per_condition_rates': per_cond_rates,
    }


def auto_relax(rules: dict, df: pd.DataFrame) -> Optional[dict]:
    """
    Auto-relax an OVER_CONSTRAINED signal by:
    1. Removing regime filter (if present with ADX condition)
    2. Dropping the tightest AND condition (lowest individual fire rate)
    3. Relaxing extreme thresholds

    Returns relaxed rules, or None if can't be fixed.
    """
    relaxed = deepcopy(rules)
    changes = []

    # Step 1: Remove regime filter if it has ADX condition
    regime_filter = relaxed.get('regime_filter', [])
    if regime_filter:
        has_adx = any(c.get('indicator', c.get('left', '')) == 'adx_14'
                      for c in relaxed.get('entry_long', []) + relaxed.get('entry_short', []))
        if has_adx:
            relaxed['regime_filter'] = []
            changes.append('Removed regime filter (ADX already checks trend)')

    # Check if that fixed it
    result = validate_fire_rate(relaxed, df)
    if result['status'] == 'VIABLE':
        relaxed['_relaxation_notes'] = changes
        return relaxed

    # Step 2: Drop tightest condition for sides with 3+ conditions
    for side in ['entry_long', 'entry_short']:
        conditions = relaxed.get(side, [])
        if len(conditions) < 3:
            continue

        # Find per-condition fire rates
        rates = []
        df_ind = add_all_indicators(df)
        n_bars = len(df_ind)

        for i, cond in enumerate(conditions):
            fires = 0
            for idx in range(1, n_bars):
                row = df_ind.iloc[idx]
                prev = df_ind.iloc[idx - 1]
                try:
                    if _eval_condition(row, prev, cond):
                        fires += 1
                except Exception:
                    pass
            rates.append((i, fires / n_bars if n_bars > 0 else 0, cond))

        # Drop the condition with lowest fire rate
        rates.sort(key=lambda x: x[1])
        drop_idx, drop_rate, drop_cond = rates[0]
        conditions.pop(drop_idx)
        changes.append(f'Dropped tightest {side} condition: {drop_cond} (fire rate {drop_rate:.3f})')

    # Step 3: Relax extreme thresholds
    for side in ['entry_long', 'entry_short']:
        for cond in relaxed.get(side, []):
            indicator = cond.get('indicator', cond.get('left', ''))
            value = cond.get('value', cond.get('right', ''))
            op = cond.get('op', cond.get('operator', ''))

            # dc_upper → dc_middle
            if isinstance(value, str) and value == 'dc_upper':
                cond['value'] = 'dc_middle'
                changes.append(f'{side}: dc_upper → dc_middle')

            # bb_pct_b < 0 → bb_pct_b < 0.1
            if indicator in ('bb_pct_b', 'bb_pct_b_30'):
                try:
                    v = float(value)
                    if op in ('<', '<=') and v < 0.05:
                        cond['value'] = 0.1
                        changes.append(f'{side}: {indicator} {op} {v} → 0.1')
                    elif op in ('>', '>=') and v > 0.95:
                        cond['value'] = 0.9
                        changes.append(f'{side}: {indicator} {op} {v} → 0.9')
                except (ValueError, TypeError):
                    pass

    if changes:
        relaxed['_relaxation_notes'] = changes
        return relaxed

    return None
