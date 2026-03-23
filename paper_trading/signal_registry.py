"""
Table-driven signal registry for paper trading.

Replaces hardcoded _check_entry_* and _check_exit_* methods in signal_compute.py
with a declarative signal definition system. Each signal is defined as a dict
of conditions (same format as the backtest engine's DSL rules), making it trivial
to add new signals without modifying signal_compute.py.

Usage:
    registry = SignalRegistry()
    registry.register_from_dsl('KAUFMAN_DRY_20', rules_dict)
    entry = registry.check_entry('KAUFMAN_DRY_20', today_row, yesterday_row)
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SignalDefinition:
    """A single signal's complete trading rules."""

    def __init__(self, signal_id: str, config: dict):
        self.signal_id = signal_id
        self.direction = config.get('direction', 'BOTH')
        self.stop_loss_pct = config.get('stop_loss_pct', 0.02)
        self.take_profit_pct = config.get('take_profit_pct', 0.0)
        self.hold_days_max = config.get('hold_days_max', 0)
        self.trade_type = config.get('trade_type', 'PAPER')
        self.instrument_type = config.get('instrument_type', 'FUTURES')

        # DSL-style conditions (list of condition dicts)
        self.entry_long = config.get('entry_long', [])
        self.entry_short = config.get('entry_short', [])
        self.exit_long = config.get('exit_long', [])
        self.exit_short = config.get('exit_short', [])

        # Optional: regime filter
        self.regime_filter = config.get('regime_filter', [])

        # Optional: ADX filter for adaptive variant
        self.adx_filter = config.get('adx_filter', None)  # e.g., {'max_adx': 25}

    def __repr__(self):
        return f"SignalDefinition({self.signal_id}, dir={self.direction}, type={self.trade_type})"


def _eval_condition(row: pd.Series, prev_row: pd.Series, cond: dict) -> bool:
    """Evaluate a single DSL condition against indicator rows."""
    indicator = cond.get('indicator', '')
    op = cond.get('op', '>')
    value = cond.get('value')

    if indicator == 'True':
        return True
    if indicator == 'False':
        return False

    if indicator not in row.index:
        return False
    ind_val = row[indicator]
    if pd.isna(ind_val):
        return False

    # Value can be a number or indicator reference
    if isinstance(value, str) and value in row.index:
        target = row[value]
        if pd.isna(target):
            return False
    elif value is None:
        return False
    else:
        try:
            target = float(value)
        except (ValueError, TypeError):
            return False

    if op == '>': return ind_val > target
    elif op == '<': return ind_val < target
    elif op == '>=': return ind_val >= target
    elif op == '<=': return ind_val <= target
    elif op == '==': return abs(ind_val - target) < 1e-9
    elif op == 'crosses_above':
        if prev_row is None:
            return False
        prev_ind = prev_row.get(indicator, np.nan)
        prev_target = prev_row.get(value, target) if isinstance(value, str) else target
        if pd.isna(prev_ind) or pd.isna(prev_target):
            return False
        return prev_ind <= prev_target and ind_val > target
    elif op == 'crosses_below':
        if prev_row is None:
            return False
        prev_ind = prev_row.get(indicator, np.nan)
        prev_target = prev_row.get(value, target) if isinstance(value, str) else target
        if pd.isna(prev_ind) or pd.isna(prev_target):
            return False
        return prev_ind >= prev_target and ind_val < target
    return False


def _eval_conditions(row: pd.Series, prev_row: pd.Series, conditions: list) -> bool:
    """All conditions must be true (AND logic)."""
    if not conditions:
        return False
    return all(_eval_condition(row, prev_row, c) for c in conditions)


class SignalRegistry:
    """
    Central registry of all trading signals.

    Provides table-driven entry/exit checking instead of hardcoded methods.
    """

    def __init__(self):
        self.signals: Dict[str, SignalDefinition] = {}
        self._register_confirmed_signals()

    def register(self, signal_id: str, config: dict):
        """Register a signal definition."""
        self.signals[signal_id] = SignalDefinition(signal_id, config)
        logger.debug(f"Registered signal: {signal_id}")

    def register_from_dsl(self, signal_id: str, rules: dict,
                           trade_type: str = 'PAPER', **overrides):
        """Register a signal from DSL backtest rules."""
        config = {
            'direction': rules.get('direction', 'BOTH'),
            'stop_loss_pct': rules.get('stop_loss_pct', 0.02),
            'take_profit_pct': rules.get('take_profit_pct', 0.0),
            'hold_days_max': rules.get('hold_days', 0),
            'entry_long': rules.get('entry_long', []),
            'entry_short': rules.get('entry_short', []),
            'exit_long': rules.get('exit_long', []),
            'exit_short': rules.get('exit_short', []),
            'regime_filter': rules.get('regime_filter', []),
            'trade_type': trade_type,
            **overrides,
        }
        self.register(signal_id, config)

    def check_entry(self, signal_id: str, today: pd.Series,
                     yesterday: pd.Series, regime: str = None,
                     adx_filtered: bool = False) -> Optional[Dict]:
        """
        Check if a signal fires an entry today.

        Args:
            signal_id: Signal to check
            today: Today's indicator row
            yesterday: Yesterday's indicator row
            regime: Current regime (for regime filtering)
            adx_filtered: If True, apply ADX < 25 filter (for adaptive variants)

        Returns:
            {'signal_id': str, 'direction': str, 'price': float} or None
        """
        sig = self.signals.get(signal_id)
        if sig is None:
            logger.warning(f"Signal {signal_id} not in registry")
            return None

        # Regime filter
        if sig.regime_filter and regime:
            if regime not in sig.regime_filter:
                return None

        # ADX filter (adaptive variant)
        if adx_filtered:
            adx = today.get('adx_14', 30)
            if pd.notna(adx) and float(adx) >= 25:
                return None

        price = float(today['close'])

        # Check LONG entry
        if sig.direction in ('BOTH', 'LONG'):
            if sig.entry_long and _eval_conditions(today, yesterday, sig.entry_long):
                return {
                    'signal_id': signal_id,
                    'direction': 'LONG',
                    'price': price,
                    'reason': f'entry_long ({signal_id})',
                }

        # Check SHORT entry
        if sig.direction in ('BOTH', 'SHORT'):
            if sig.entry_short and _eval_conditions(today, yesterday, sig.entry_short):
                return {
                    'signal_id': signal_id,
                    'direction': 'SHORT',
                    'price': price,
                    'reason': f'entry_short ({signal_id})',
                }

        return None

    def check_exit(self, signal_id: str, position: dict,
                    today: pd.Series, yesterday: pd.Series,
                    as_of_date=None) -> Optional[Dict]:
        """
        Check if an open position should be exited.

        Args:
            signal_id: Signal that opened the position
            position: {'direction': str, 'entry_price': float, 'entry_date': date}
            today: Today's indicator row
            yesterday: Yesterday's indicator row
            as_of_date: Current date (for hold days calculation)

        Returns:
            {'signal_id': str, 'direction': str, 'price': float, 'reason': str} or None
        """
        sig = self.signals.get(signal_id)
        if sig is None:
            return None

        price = float(today['close'])
        entry_price = position['entry_price']
        direction = position['direction']

        # Stop loss
        if sig.stop_loss_pct > 0:
            if direction == 'LONG':
                loss_pct = (entry_price - price) / entry_price
            else:
                loss_pct = (price - entry_price) / entry_price

            if loss_pct >= sig.stop_loss_pct:
                return {
                    'signal_id': signal_id,
                    'direction': direction,
                    'price': price,
                    'reason': f'stop_loss ({loss_pct:.1%})',
                    'pnl': (price - entry_price) if direction == 'LONG' else (entry_price - price),
                }

        # Take profit
        if sig.take_profit_pct > 0:
            if direction == 'LONG':
                gain_pct = (price - entry_price) / entry_price
            else:
                gain_pct = (entry_price - price) / entry_price

            if gain_pct >= sig.take_profit_pct:
                return {
                    'signal_id': signal_id,
                    'direction': direction,
                    'price': price,
                    'reason': f'take_profit ({gain_pct:.1%})',
                    'pnl': (price - entry_price) if direction == 'LONG' else (entry_price - price),
                }

        # Hold days
        if sig.hold_days_max > 0 and as_of_date and position.get('entry_date'):
            days_held = (as_of_date - position['entry_date']).days
            if days_held >= sig.hold_days_max:
                pnl = (price - entry_price) if direction == 'LONG' else (entry_price - price)
                return {
                    'signal_id': signal_id,
                    'direction': direction,
                    'price': price,
                    'reason': f'hold_days ({days_held}d >= {sig.hold_days_max}d)',
                    'pnl': pnl,
                }

        # Signal-based exit
        exit_conditions = sig.exit_long if direction == 'LONG' else sig.exit_short
        if exit_conditions and _eval_conditions(today, yesterday, exit_conditions):
            pnl = (price - entry_price) if direction == 'LONG' else (entry_price - price)
            return {
                'signal_id': signal_id,
                'direction': direction,
                'price': price,
                'reason': f'signal_exit ({signal_id})',
                'pnl': pnl,
            }

        return None

    def get_all_signals(self, trade_type: str = None) -> Dict[str, SignalDefinition]:
        """Get all registered signals, optionally filtered by trade type."""
        if trade_type:
            return {sid: s for sid, s in self.signals.items()
                    if s.trade_type == trade_type}
        return self.signals

    def _register_confirmed_signals(self):
        """Register the 3 confirmed Kaufman signals + overlays + shadows."""

        # ===== CONFIRMED SCORING SIGNALS =====
        self.register('KAUFMAN_DRY_20', {
            'direction': 'LONG',
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.0,
            'hold_days_max': 0,
            'trade_type': 'PAPER',
            'entry_long': [
                {'indicator': 'sma_10', 'op': '<', 'value': 'prev_close'},
                {'indicator': 'stoch_k_5', 'op': '>', 'value': 50},
            ],
            'exit_long': [
                {'indicator': 'stoch_k_5', 'op': '<=', 'value': 50},
            ],
        })

        self.register('KAUFMAN_DRY_16', {
            'direction': 'BOTH',
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.03,
            'hold_days_max': 0,
            'trade_type': 'PAPER',
            'entry_long': [
                {'indicator': 'close', 'op': '>', 'value': 'r1'},
                {'indicator': 'low', 'op': '>=', 'value': 'pivot'},
                {'indicator': 'hvol_6', 'op': '<', 'value': 'hvol_100'},
            ],
            'exit_long': [
                {'indicator': 'low', 'op': '<', 'value': 'pivot'},
            ],
            'entry_short': [
                {'indicator': 'close', 'op': '<', 'value': 's1'},
                {'indicator': 'hvol_6', 'op': '<', 'value': 'hvol_100'},
            ],
            'exit_short': [
                {'indicator': 'high', 'op': '>', 'value': 'r1'},
            ],
        })

        self.register('KAUFMAN_DRY_12', {
            'direction': 'BOTH',
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.03,
            'hold_days_max': 7,
            'trade_type': 'PAPER',
            'entry_long': [
                {'indicator': 'close', 'op': '>', 'value': 'prev_close'},
                {'indicator': 'volume', 'op': '<', 'value': 'prev_volume'},
            ],
            'exit_long': [
                {'indicator': 'close', 'op': '<', 'value': 'prev_close'},
            ],
            'entry_short': [
                {'indicator': 'close', 'op': '<', 'value': 'prev_close'},
                {'indicator': 'volume', 'op': '<', 'value': 'prev_volume'},
            ],
            'exit_short': [
                {'indicator': 'close', 'op': '>', 'value': 'prev_close'},
            ],
        })

        # ===== OVERLAY SIGNALS =====
        self.register('GUJRAL_DRY_7', {
            'direction': 'BOTH',
            'stop_loss_pct': 0.02,
            'hold_days_max': 0,
            'trade_type': 'OVERLAY',
            'entry_long': [
                {'indicator': 'close', 'op': '>', 'value': 'sma_50'},
                {'indicator': 'adx_14', 'op': '>', 'value': 25},
            ],
            'exit_long': [
                {'indicator': 'close', 'op': '<', 'value': 'sma_50'},
            ],
            'entry_short': [
                {'indicator': 'close', 'op': '<', 'value': 'sma_50'},
                {'indicator': 'adx_14', 'op': '>', 'value': 25},
            ],
            'exit_short': [
                {'indicator': 'close', 'op': '>', 'value': 'sma_50'},
            ],
        })

        self.register('CRISIS_SHORT', {
            'direction': 'SHORT',
            'stop_loss_pct': 0.04,
            'hold_days_max': 15,
            'trade_type': 'OVERLAY',
            'entry_short': [
                {'indicator': 'india_vix', 'op': '>', 'value': 25},
                {'indicator': 'close', 'op': '<', 'value': 'sma_50'},
                {'indicator': 'adx_14', 'op': '>', 'value': 25},
            ],
            'exit_short': [
                {'indicator': 'india_vix', 'op': '<', 'value': 20},
            ],
        })

        logger.info(f"Signal registry initialized with {len(self.signals)} signals")
