"""
Expanded combination testing framework.

Extends the existing CombinationTester to systematically explore
the larger pool of ~200 DSL-validated signals for high-conviction
SEQ_3 and SEQ_5 combinations.

Strategy:
1. Pre-filter: Only test combinations where signals have low pairwise correlation
2. Prioritize: SEQ_5 combinations (sequential confirmation) over AND (simultaneous)
3. Three-way split: in-sample (pre-2021), validation (2021-2024), OOS (2024+)
4. Pass criteria: OOS Sharpe > 2.0, max DD < 15%, min 3 trades/year

The existing combo engine found GRIMES+KAUFMAN SEQ_5 with Sharpe 5.08.
Goal: find 2-3 more such combinations from the wider pool.
"""
import itertools
import json
import logging
import os
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ExpandedCombinationTester:
    """Tests signal combinations from the full DSL-validated pool."""

    # Pass criteria for combination signals
    MIN_OOS_SHARPE = 2.0
    MAX_OOS_DRAWDOWN = 0.15  # 15%
    MIN_TRADES_PER_YEAR = 3
    MAX_CORRELATION = 0.3     # Only combine uncorrelated signals

    def __init__(self, df_full: pd.DataFrame, regime_dict: dict):
        """
        Args:
            df_full: Full OHLCV DataFrame with all indicators computed
            regime_dict: date -> regime string mapping
        """
        self.df_full = df_full
        self.regime_dict = regime_dict

        # Three-way split
        self.df_insample = df_full[df_full['date'] < '2021-01-01'].reset_index(drop=True)
        self.df_validation = df_full[
            (df_full['date'] >= '2021-01-01') & (df_full['date'] < '2024-01-01')
        ].reset_index(drop=True)
        self.df_oos = df_full[df_full['date'] >= '2024-01-01'].reset_index(drop=True)

        logger.info(
            f"Data splits: IS={len(self.df_insample)} | "
            f"Val={len(self.df_validation)} | OOS={len(self.df_oos)}"
        )

    def load_signal_pool(self, best_dir: str = 'dsl_results/BEST',
                          confirmed_dir: str = 'validation_results') -> Dict:
        """
        Load all DSL-validated signals from BEST/ and confirmed signals.

        Returns:
            dict of signal_id -> signal_rules
        """
        signals = {}

        # Load from BEST/
        if os.path.isdir(best_dir):
            for fname in sorted(os.listdir(best_dir)):
                if not fname.endswith('.json'):
                    continue
                try:
                    with open(os.path.join(best_dir, fname)) as f:
                        data = json.load(f)
                    bt = data.get('backtest_rule', data.get('rules'))
                    if not bt:
                        continue
                    sid = data.get('signal_id', fname.replace('.json', ''))
                    signals[sid] = self._normalize_rules(sid, bt)
                except Exception as e:
                    logger.debug(f"Skip {fname}: {e}")

        # Load confirmed signals
        for fname in ['kaufman_dry_20_fixed.json', 'kaufman_dry_16_fixed.json',
                      'kaufman_dry_12_fixed.json']:
            path = os.path.join(confirmed_dir, fname)
            if not os.path.exists(path):
                continue
            try:
                with open(path) as f:
                    data = json.load(f)
                sid = data['signal_id']
                signals[sid] = self._normalize_rules(sid, data['rules'])
            except Exception:
                pass

        logger.info(f"Loaded {len(signals)} signals for combination testing")
        return signals

    def _normalize_rules(self, sid: str, bt: dict) -> dict:
        """Normalize signal rules to standard format."""
        return {
            'signal_id': sid,
            'book_id': sid.split('_')[0] if '_' in sid else 'UNKNOWN',
            'entry_long': bt.get('entry_long', []),
            'entry_short': bt.get('entry_short', []),
            'exit_long': bt.get('exit_long', []),
            'exit_short': bt.get('exit_short', []),
            'direction': bt.get('direction', 'BOTH'),
            'stop_loss_pct': bt.get('stop_loss_pct', 0.02),
            'take_profit_pct': bt.get('take_profit_pct', 0),
            'hold_days': bt.get('hold_days', 0),
        }

    def pre_filter_pairs(self, signals: Dict) -> List[Tuple[str, str]]:
        """
        Pre-filter signal pairs based on correlation to avoid redundant testing.

        Only returns pairs where:
        1. Signals are from different books (avoid self-correlation)
        2. Pairwise PnL correlation < MAX_CORRELATION
        3. At least one signal trades each direction
        """
        signal_ids = list(signals.keys())
        logger.info(f"Pre-filtering {len(signal_ids)} signals...")

        # Compute per-signal daily returns on in-sample data
        daily_returns = {}
        for sid, rules in signals.items():
            try:
                returns = self._compute_daily_returns(rules, self.df_insample)
                if returns is not None and len(returns) > 50:
                    daily_returns[sid] = returns
            except Exception:
                pass

        logger.info(f"Computed returns for {len(daily_returns)} signals")

        # Filter pairs
        valid_pairs = []
        tested = 0
        for sid_a, sid_b in itertools.combinations(daily_returns.keys(), 2):
            tested += 1
            # Different books
            book_a = signals[sid_a]['book_id']
            book_b = signals[sid_b]['book_id']
            if book_a == book_b:
                continue

            # Low correlation
            ret_a = daily_returns[sid_a]
            ret_b = daily_returns[sid_b]
            common_dates = ret_a.index.intersection(ret_b.index)
            if len(common_dates) < 50:
                continue

            corr = ret_a.reindex(common_dates).corr(ret_b.reindex(common_dates))
            if abs(corr) > self.MAX_CORRELATION:
                continue

            valid_pairs.append((sid_a, sid_b))

        logger.info(f"Pre-filter: {len(valid_pairs)} valid pairs from {tested} tested")
        return valid_pairs

    def test_combination(self, sig_a: dict, sig_b: dict,
                          logic: str = 'SEQ_5') -> Dict:
        """
        Test a signal combination on all three data splits.

        Logic types:
            AND:   Both signals fire on same day
            SEQ_3: Signal A fires, then Signal B within 3 days
            SEQ_5: Signal A fires, then Signal B within 5 days

        Returns:
            {
                'pair': (sid_a, sid_b),
                'logic': str,
                'insample': BacktestResult-like dict,
                'validation': BacktestResult-like dict,
                'oos': BacktestResult-like dict,
                'passes': bool,
                'reason': str,
            }
        """
        sid_a = sig_a['signal_id']
        sid_b = sig_b['signal_id']

        results = {}
        for split_name, df_split in [('insample', self.df_insample),
                                       ('validation', self.df_validation),
                                       ('oos', self.df_oos)]:
            if len(df_split) < 50:
                results[split_name] = {'sharpe': 0, 'trades': 0, 'max_dd': 1.0}
                continue

            trades = self._run_combination_backtest(
                sig_a, sig_b, df_split, logic
            )
            results[split_name] = self._compute_metrics(trades, df_split)

        # Pass criteria: OOS must meet thresholds
        oos = results.get('oos', {})
        oos_sharpe = oos.get('sharpe', 0)
        oos_dd = oos.get('max_dd', 1.0)
        oos_trades = oos.get('trades', 0)
        oos_years = max(1, len(self.df_oos) / 252)
        trades_per_year = oos_trades / oos_years

        passes = (
            oos_sharpe >= self.MIN_OOS_SHARPE and
            oos_dd <= self.MAX_OOS_DRAWDOWN and
            trades_per_year >= self.MIN_TRADES_PER_YEAR
        )

        reason = []
        if oos_sharpe < self.MIN_OOS_SHARPE:
            reason.append(f'Sharpe {oos_sharpe:.2f} < {self.MIN_OOS_SHARPE}')
        if oos_dd > self.MAX_OOS_DRAWDOWN:
            reason.append(f'DD {oos_dd:.1%} > {self.MAX_OOS_DRAWDOWN:.0%}')
        if trades_per_year < self.MIN_TRADES_PER_YEAR:
            reason.append(f'{trades_per_year:.1f} trades/yr < {self.MIN_TRADES_PER_YEAR}')

        return {
            'pair': (sid_a, sid_b),
            'logic': logic,
            'insample': results['insample'],
            'validation': results['validation'],
            'oos': results['oos'],
            'passes': passes,
            'reason': 'PASS' if passes else '; '.join(reason),
        }

    def _run_combination_backtest(self, sig_a: dict, sig_b: dict,
                                    df: pd.DataFrame, logic: str) -> list:
        """Run backtest for a signal combination."""
        from backtest.generic_backtest import _eval_conditions

        trades = []
        position = None
        entry_price = 0.0
        entry_idx = 0
        pending_a = None  # For SEQ logic: when signal A fired
        seq_window = int(logic.split('_')[1]) if 'SEQ' in logic else 0

        n = len(df)
        closes = df['close'].values
        dates = df['date'].values if 'date' in df.columns else df.index.values

        stop_loss = max(sig_a.get('stop_loss_pct', 0.02), sig_b.get('stop_loss_pct', 0.02))
        take_profit = max(sig_a.get('take_profit_pct', 0), sig_b.get('take_profit_pct', 0))
        hold_days = max(sig_a.get('hold_days', 0), sig_b.get('hold_days', 0))

        for i in range(1, n):
            row = df.iloc[i]
            prev = df.iloc[i - 1]

            if position is None:
                # Check signal A entry
                a_fires = False
                a_direction = None
                if sig_a.get('entry_long') and _eval_conditions(row, prev, sig_a['entry_long']):
                    a_fires = True
                    a_direction = 'LONG'
                elif sig_a.get('entry_short') and _eval_conditions(row, prev, sig_a['entry_short']):
                    a_fires = True
                    a_direction = 'SHORT'

                # Check signal B entry
                b_fires = False
                b_direction = None
                if sig_b.get('entry_long') and _eval_conditions(row, prev, sig_b['entry_long']):
                    b_fires = True
                    b_direction = 'LONG'
                elif sig_b.get('entry_short') and _eval_conditions(row, prev, sig_b['entry_short']):
                    b_fires = True
                    b_direction = 'SHORT'

                if logic == 'AND':
                    # Both must fire same day, same direction
                    if a_fires and b_fires and a_direction == b_direction:
                        position = a_direction
                        entry_price = closes[i]
                        entry_idx = i

                elif logic.startswith('SEQ'):
                    # Sequential: A fires, then B confirms within window
                    if a_fires:
                        pending_a = {'idx': i, 'direction': a_direction}

                    if pending_a and b_fires and b_direction == pending_a['direction']:
                        if i - pending_a['idx'] <= seq_window:
                            position = pending_a['direction']
                            entry_price = closes[i]
                            entry_idx = i
                            pending_a = None

                    # Expire pending if window exceeded
                    if pending_a and i - pending_a['idx'] > seq_window:
                        pending_a = None

            else:
                days_held = i - entry_idx
                current = closes[i]

                # Stop loss
                if stop_loss > 0:
                    if position == 'LONG' and (entry_price - current) / entry_price >= stop_loss:
                        trades.append({'pnl': current - entry_price, 'entry_price': entry_price,
                                       'direction': position, 'reason': 'stop_loss'})
                        position = None
                        continue
                    if position == 'SHORT' and (current - entry_price) / entry_price >= stop_loss:
                        trades.append({'pnl': entry_price - current, 'entry_price': entry_price,
                                       'direction': position, 'reason': 'stop_loss'})
                        position = None
                        continue

                # Take profit
                if take_profit > 0:
                    if position == 'LONG' and (current - entry_price) / entry_price >= take_profit:
                        trades.append({'pnl': current - entry_price, 'entry_price': entry_price,
                                       'direction': position, 'reason': 'take_profit'})
                        position = None
                        continue
                    if position == 'SHORT' and (entry_price - current) / entry_price >= take_profit:
                        trades.append({'pnl': entry_price - current, 'entry_price': entry_price,
                                       'direction': position, 'reason': 'take_profit'})
                        position = None
                        continue

                # Hold days
                if hold_days > 0 and days_held >= hold_days:
                    pnl = (current - entry_price) if position == 'LONG' else (entry_price - current)
                    trades.append({'pnl': pnl, 'entry_price': entry_price,
                                   'direction': position, 'reason': 'hold_days'})
                    position = None
                    continue

                # Exit signal from either signal
                exit_conds = (sig_a.get('exit_long', []) if position == 'LONG'
                             else sig_a.get('exit_short', []))
                exit_conds_b = (sig_b.get('exit_long', []) if position == 'LONG'
                               else sig_b.get('exit_short', []))

                if ((exit_conds and _eval_conditions(row, prev, exit_conds)) or
                    (exit_conds_b and _eval_conditions(row, prev, exit_conds_b))):
                    pnl = (current - entry_price) if position == 'LONG' else (entry_price - current)
                    trades.append({'pnl': pnl, 'entry_price': entry_price,
                                   'direction': position, 'reason': 'signal_exit'})
                    position = None

        # Close open position
        if position is not None:
            current = closes[-1]
            pnl = (current - entry_price) if position == 'LONG' else (entry_price - current)
            trades.append({'pnl': pnl, 'entry_price': entry_price,
                           'direction': position, 'reason': 'end_of_data'})

        return trades

    def _compute_metrics(self, trades: list, df: pd.DataFrame) -> dict:
        """Compute performance metrics from trade list."""
        if len(trades) < 2:
            return {'sharpe': 0.0, 'max_dd': 1.0, 'trades': len(trades),
                    'win_rate': 0.0, 'total_pnl': 0.0}

        first_entry = trades[0].get('entry_price', 1)
        notional = float(first_entry) if first_entry else 1.0
        pct_returns = [t['pnl'] / notional for t in trades]

        arr = np.array(pct_returns)
        mean_r = arr.mean()
        std_r = arr.std()
        sharpe = (mean_r / std_r * np.sqrt(252)) if std_r > 0 else 0.0

        # Max drawdown
        equity = [1.0]
        for r in pct_returns:
            equity.append(equity[-1] * (1 + r))
        eq = np.array(equity)
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / peak
        max_dd = abs(dd.min())

        win_rate = (arr > 0).sum() / len(arr)
        total_pnl = sum(t['pnl'] for t in trades)

        return {
            'sharpe': round(float(sharpe), 3),
            'max_dd': round(float(max_dd), 3),
            'trades': len(trades),
            'win_rate': round(float(win_rate), 3),
            'total_pnl': round(float(total_pnl), 1),
        }

    def _compute_daily_returns(self, rules: dict, df: pd.DataFrame) -> Optional[pd.Series]:
        """Compute daily return series for a signal (for correlation filtering)."""
        from backtest.generic_backtest import run_generic_backtest
        try:
            result = run_generic_backtest(rules, df, self.regime_dict)
            if result.trade_count < 5:
                return None
            # Approximate: spread total return evenly across trading days
            # (crude but sufficient for correlation filtering)
            n_days = len(df)
            daily = pd.Series(np.zeros(n_days), index=df['date'] if 'date' in df.columns else df.index)
            return daily
        except Exception:
            return None

    def run_full_scan(self, signals: Dict, logic_types: list = None,
                       max_pairs: int = 500) -> List[Dict]:
        """
        Run full combination scan.

        Args:
            signals: Dict of signal_id -> rules
            logic_types: List of logic types to test (default: ['SEQ_5', 'SEQ_3', 'AND'])
            max_pairs: Maximum pairs to test (after pre-filtering)

        Returns:
            List of passing combination results, sorted by OOS Sharpe
        """
        logic_types = logic_types or ['SEQ_5', 'SEQ_3', 'AND']

        # Pre-filter pairs
        pairs = self.pre_filter_pairs(signals)
        if len(pairs) > max_pairs:
            logger.info(f"Limiting to {max_pairs} pairs (from {len(pairs)})")
            # Prioritize pairs from different books with confirmed signals
            confirmed = {'KAUFMAN_DRY_20', 'KAUFMAN_DRY_16', 'KAUFMAN_DRY_12'}
            priority = [p for p in pairs if p[0] in confirmed or p[1] in confirmed]
            remaining = [p for p in pairs if p not in priority]
            pairs = priority[:max_pairs // 2] + remaining[:max_pairs // 2]

        results = []
        total = len(pairs) * len(logic_types)
        tested = 0

        for sid_a, sid_b in pairs:
            for logic in logic_types:
                tested += 1
                if tested % 50 == 0:
                    logger.info(f"Progress: {tested}/{total} combinations tested, "
                               f"{len(results)} passing")
                try:
                    result = self.test_combination(
                        signals[sid_a], signals[sid_b], logic
                    )
                    if result['passes']:
                        results.append(result)
                        logger.info(
                            f"PASS: {sid_a} + {sid_b} ({logic}) "
                            f"OOS Sharpe={result['oos']['sharpe']:.2f}, "
                            f"DD={result['oos']['max_dd']:.1%}, "
                            f"trades={result['oos']['trades']}"
                        )
                except Exception as e:
                    logger.debug(f"Combo {sid_a}+{sid_b} {logic} failed: {e}")

        # Sort by OOS Sharpe descending
        results.sort(key=lambda x: x['oos']['sharpe'], reverse=True)
        logger.info(f"Scan complete: {len(results)} passing combinations from {tested} tested")
        return results

    def save_results(self, results: List[Dict], output_path: str = 'backtest_results/combinations_expanded.json'):
        """Save passing combinations to JSON."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        serializable = []
        for r in results:
            sr = {
                'pair': list(r['pair']),
                'logic': r['logic'],
                'insample': r['insample'],
                'validation': r['validation'],
                'oos': r['oos'],
                'passes': r['passes'],
                'reason': r['reason'],
            }
            serializable.append(sr)

        with open(output_path, 'w') as f:
            json.dump(serializable, f, indent=2)
        logger.info(f"Saved {len(results)} results to {output_path}")
