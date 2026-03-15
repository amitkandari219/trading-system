"""
Systematic combination testing framework.

Tests all pairwise combinations of DSL-validated signals using
AND, SEQ_3, SEQ_5 logic types with three-way data split validation.
"""

import itertools
import json
import os
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import psycopg2

from backtest.indicators import add_all_indicators, historical_volatility
from backtest.types import harmonic_mean_sharpe
from config.settings import DATABASE_DSN


class CombinationTester:

    def __init__(self):
        print("Loading market data...", flush=True)
        conn = psycopg2.connect(DATABASE_DSN)
        df_raw = pd.read_sql(
            "SELECT date, open, high, low, close, volume, india_vix "
            "FROM nifty_daily ORDER BY date", conn)
        conn.close()
        df_raw['date'] = pd.to_datetime(df_raw['date'])

        df = add_all_indicators(df_raw)
        df['hvol_6'] = historical_volatility(df['close'], period=6)
        df['hvol_100'] = historical_volatility(df['close'], period=100)
        df['date'] = df_raw['date']
        df['india_vix'] = df_raw['india_vix']

        # Regime labels
        from regime_labeler import RegimeLabeler
        labeler = RegimeLabeler()
        regime_dict = labeler.label_full_history(df_raw)
        df['regime'] = df['date'].map(regime_dict).fillna('UNKNOWN')

        self.df_full = df
        self.df_insample = df[df['date'] < '2021-01-01'].reset_index(drop=True)
        self.df_validation = df[(df['date'] >= '2021-01-01') & (df['date'] < '2024-01-01')].reset_index(drop=True)
        self.df_oos = df[df['date'] >= '2024-01-01'].reset_index(drop=True)
        self.df_recent = df[df['date'] >= '2025-09-01'].reset_index(drop=True)

        n_indicators = len([c for c in df.columns if c not in ('date', 'regime')])
        print(f"  Loaded {len(df)} days, computed {n_indicators} indicators", flush=True)
        print(f"  In-sample: {len(self.df_insample)} | Validation: {len(self.df_validation)} | "
              f"OOS: {len(self.df_oos)} | Recent: {len(self.df_recent)}", flush=True)

    def load_all_signals(self) -> dict:
        """Load all PASS signals from BEST/ and confirmed signals."""
        signals = {}

        # Load from BEST/ (only PASS signals — have backtest_rule)
        best_dir = 'dsl_results/BEST'
        for fname in sorted(os.listdir(best_dir)):
            if not fname.endswith('.json'):
                continue
            with open(os.path.join(best_dir, fname)) as f:
                data = json.load(f)

            bt = data.get('backtest_rule', data.get('rules'))
            if not bt or not bt.get('backtestable', bt.get('entry_long')):
                continue

            sid = data.get('signal_id', fname.replace('.json', ''))
            book_id = sid.split('_')[0]

            signals[sid] = {
                'signal_id': sid,
                'book_id': book_id,
                'entry_long': bt.get('entry_long', []),
                'entry_short': bt.get('entry_short', []),
                'exit_long': bt.get('exit_long', []),
                'exit_short': bt.get('exit_short', []),
                'direction': bt.get('direction', 'BOTH'),
                'stop_loss_pct': bt.get('stop_loss_pct', 0.02),
                'take_profit_pct': bt.get('take_profit_pct', 0),
                'hold_days': bt.get('hold_days', 0),
            }

        # Load confirmed signals
        for fname in ['kaufman_dry_20_fixed.json', 'kaufman_dry_16_fixed.json', 'kaufman_dry_12_fixed.json']:
            path = os.path.join('validation_results', fname)
            if not os.path.exists(path):
                continue
            with open(path) as f:
                data = json.load(f)
            sid = data['signal_id']
            bt = data['rules']
            signals[sid] = {
                'signal_id': sid,
                'book_id': 'KAUFMAN',
                'entry_long': bt.get('entry_long', []),
                'entry_short': bt.get('entry_short', []),
                'exit_long': bt.get('exit_long', []),
                'exit_short': bt.get('exit_short', []),
                'direction': bt.get('direction', 'BOTH'),
                'stop_loss_pct': bt.get('stop_loss_pct', 0.02),
                'take_profit_pct': bt.get('take_profit_pct', 0),
                'hold_days': bt.get('hold_days', 0),
            }

        # Count per book
        books = defaultdict(int)
        for s in signals.values():
            books[s['book_id']] += 1
        book_str = ', '.join(f"{k} {v}" for k, v in sorted(books.items(), key=lambda x: -x[1]))
        print(f"  Loaded {len(signals)} signals ({book_str})", flush=True)

        return signals

    # ================================================================
    # CONDITION EVALUATION
    # ================================================================

    @staticmethod
    def _eval_cond(row, prev, cond):
        indicator = cond.get('indicator', '')
        op = cond.get('op', '>')
        value = cond.get('value')

        if indicator not in row.index:
            return False
        ind_val = row[indicator]
        if pd.isna(ind_val):
            return False

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
            if prev is None: return False
            pi = prev.get(indicator, np.nan)
            pt = prev.get(value, target) if isinstance(value, str) else target
            if pd.isna(pi) or pd.isna(pt): return False
            return pi <= pt and ind_val > target
        elif op == 'crosses_below':
            if prev is None: return False
            pi = prev.get(indicator, np.nan)
            pt = prev.get(value, target) if isinstance(value, str) else target
            if pd.isna(pi) or pd.isna(pt): return False
            return pi >= pt and ind_val < target
        return False

    @staticmethod
    def _eval_conds(row, prev, conds):
        if not conds:
            return False
        return all(CombinationTester._eval_cond(row, prev, c) for c in conds)

    # ================================================================
    # CORE BACKTEST
    # ================================================================

    def backtest_combination(self, sig_a, sig_b, logic, df,
                              stop_pct=0.02, take_profit_pct=0.03,
                              hold_days_max=20):
        n = len(df)
        if n < 50:
            return {'insufficient_trades': True, 'trades': 0}

        closes = df['close'].values
        opens = df['open'].values
        dates = df['date'].values

        trades = []
        position = None
        entry_price = 0.0
        entry_idx = 0

        # For SEQ logic: track pending anchor fires
        pending_a_long = -999
        pending_a_short = -999

        for i in range(2, n):
            row = df.iloc[i]
            prev = df.iloc[i - 1]
            prev2 = df.iloc[i - 2]

            # Evaluate individual signal conditions
            a_long = self._eval_conds(row, prev, sig_a.get('entry_long', []))
            a_short = self._eval_conds(row, prev, sig_a.get('entry_short', []))
            b_long = self._eval_conds(row, prev, sig_b.get('entry_long', []))
            b_short = self._eval_conds(row, prev, sig_b.get('entry_short', []))

            a_exit_long = self._eval_conds(row, prev, sig_a.get('exit_long', []))
            a_exit_short = self._eval_conds(row, prev, sig_a.get('exit_short', []))
            b_exit_long = self._eval_conds(row, prev, sig_b.get('exit_long', []))
            b_exit_short = self._eval_conds(row, prev, sig_b.get('exit_short', []))

            # Track anchor fires for SEQ logic
            if a_long:
                pending_a_long = i
            if a_short:
                pending_a_short = i

            # Determine entry signal based on logic
            entry_long = False
            entry_short = False

            if logic == 'AND':
                entry_long = a_long and b_long
                entry_short = a_short and b_short
            elif logic == 'OR':
                entry_long = a_long or b_long
                entry_short = a_short or b_short
            elif logic in ('SEQ_3', 'SEQ_5'):
                window = 3 if logic == 'SEQ_3' else 5
                entry_long = b_long and (i - pending_a_long) <= window and pending_a_long != i
                entry_short = b_short and (i - pending_a_short) <= window and pending_a_short != i
            elif logic == 'ANCHOR':
                entry_long = a_long
                entry_short = a_short

            # Determine exit signal
            if logic == 'AND':
                exit_long = a_exit_long or b_exit_long
                exit_short = a_exit_short or b_exit_short
            elif logic == 'OR':
                exit_long = a_exit_long and b_exit_long
                exit_short = a_exit_short and b_exit_short
            elif logic in ('SEQ_3', 'SEQ_5', 'ANCHOR'):
                exit_long = a_exit_long or b_exit_long
                exit_short = a_exit_short or b_exit_short

            if position is None:
                if i + 1 >= n:
                    continue
                next_open = opens[i + 1]

                if entry_long:
                    position = 'LONG'
                    entry_price = next_open
                    entry_idx = i + 1
                elif entry_short:
                    position = 'SHORT'
                    entry_price = next_open
                    entry_idx = i + 1
            else:
                price = closes[i]
                days_held = i - entry_idx

                # Stop loss
                if position == 'LONG':
                    loss_pct = (entry_price - price) / entry_price
                else:
                    loss_pct = (price - entry_price) / entry_price
                if loss_pct >= stop_pct:
                    pnl = (price - entry_price) if position == 'LONG' else (entry_price - price)
                    trades.append({'pnl': pnl, 'entry_price': entry_price,
                                   'direction': position, 'exit_reason': 'stop',
                                   'entry_date': dates[entry_idx], 'exit_date': dates[i]})
                    position = None
                    continue

                # Take profit
                if take_profit_pct > 0:
                    if position == 'LONG':
                        gain = (price - entry_price) / entry_price
                    else:
                        gain = (entry_price - price) / entry_price
                    if gain >= take_profit_pct:
                        pnl = gain * entry_price
                        trades.append({'pnl': pnl, 'entry_price': entry_price,
                                       'direction': position, 'exit_reason': 'tp',
                                       'entry_date': dates[entry_idx], 'exit_date': dates[i]})
                        position = None
                        continue

                # Hold days
                if hold_days_max > 0 and days_held >= hold_days_max:
                    pnl = (price - entry_price) if position == 'LONG' else (entry_price - price)
                    trades.append({'pnl': pnl, 'entry_price': entry_price,
                                   'direction': position, 'exit_reason': 'hold',
                                   'entry_date': dates[entry_idx], 'exit_date': dates[i]})
                    position = None
                    continue

                # Signal exit
                should_exit = exit_long if position == 'LONG' else exit_short
                if should_exit:
                    pnl = (price - entry_price) if position == 'LONG' else (entry_price - price)
                    trades.append({'pnl': pnl, 'entry_price': entry_price,
                                   'direction': position, 'exit_reason': 'signal',
                                   'entry_date': dates[entry_idx], 'exit_date': dates[i]})
                    position = None

        # Close open position
        if position is not None and n > 0:
            price = closes[-1]
            pnl = (price - entry_price) if position == 'LONG' else (entry_price - price)
            trades.append({'pnl': pnl, 'entry_price': entry_price,
                           'direction': position, 'exit_reason': 'eod',
                           'entry_date': dates[entry_idx], 'exit_date': dates[-1]})

        return self._compute_metrics(trades, df)

    def _compute_metrics(self, trades, df):
        if len(trades) < 5:
            return {'insufficient_trades': True, 'trades': len(trades),
                    'sharpe': 0, 'win_rate': 0, 'profit_factor': 0,
                    'max_drawdown': 1.0, 'nifty_corr': 0, 'total_pnl_pts': 0,
                    'trades_per_year': 0, 'long_trades': 0, 'long_win_rate': 0,
                    'short_trades': 0, 'short_win_rate': 0}

        notional = abs(trades[0]['entry_price'])
        if notional == 0:
            notional = 1.0
        pnls = [t['pnl'] for t in trades]
        pct_ret = [p / notional for p in pnls]
        wins = [r for r in pct_ret if r > 0]

        ret = pd.Series(pct_ret)
        std = ret.std()
        sharpe = (ret.mean() / std * np.sqrt(252)) if std > 0 else 0

        eq = [1.0]
        for r in pct_ret:
            eq.append(eq[-1] * (1 + r))
        eqs = pd.Series(eq)
        dd = (eqs - eqs.cummax()) / eqs.cummax()
        max_dd = abs(dd.min())

        gw = sum(p for p in pnls if p > 0)
        gl = abs(sum(p for p in pnls if p < 0))
        pf = gw / gl if gl > 0 else 99

        # Nifty correlation
        trade_ret = {}
        for t in trades:
            d = t['exit_date']
            trade_ret[d] = trade_ret.get(d, 0) + t['pnl'] / notional
        nifty_ret = df.set_index('date')['close'].pct_change()
        strat_ret = pd.Series(trade_ret)
        common = strat_ret.index.intersection(nifty_ret.index)
        corr = float(strat_ret.reindex(common).corr(nifty_ret.reindex(common))) if len(common) > 10 else 0
        if np.isnan(corr):
            corr = 0

        date_range = (pd.Timestamp(df['date'].max()) - pd.Timestamp(df['date'].min())).days
        years = max(0.5, date_range / 365.25)

        long_trades = [t for t in trades if t['direction'] == 'LONG']
        short_trades = [t for t in trades if t['direction'] == 'SHORT']
        long_wr = sum(1 for t in long_trades if t['pnl'] > 0) / len(long_trades) if long_trades else 0
        short_wr = sum(1 for t in short_trades if t['pnl'] > 0) / len(short_trades) if short_trades else 0

        return {
            'trades': len(trades),
            'trades_per_year': round(len(trades) / years, 1),
            'win_rate': round(len(wins) / len(pct_ret), 3),
            'long_trades': len(long_trades),
            'long_win_rate': round(long_wr, 3),
            'short_trades': len(short_trades),
            'short_win_rate': round(short_wr, 3),
            'sharpe': round(sharpe, 3),
            'profit_factor': round(min(pf, 99), 2),
            'max_drawdown': round(max_dd, 3),
            'nifty_corr': round(corr, 3),
            'total_pnl_pts': round(sum(pnls), 1),
            'insufficient_trades': False,
        }

    # ================================================================
    # SCREENING
    # ================================================================

    def screen_all_pairs(self, signals, df, logics=None,
                          min_trades_yr=8.0, min_win_rate=0.48,
                          min_sharpe=1.5, max_corr=0.5, max_dd=0.25,
                          limit=None):
        logics = logics or ['AND', 'SEQ_3', 'SEQ_5']
        pairs = list(itertools.combinations(signals.keys(), 2))
        if limit:
            pairs = pairs[:limit]

        total = len(pairs) * len(logics)
        survivors = []
        tested = 0

        print(f"  Screening {len(pairs)} pairs × {len(logics)} logics = {total} combos", flush=True)

        for sig_a_id, sig_b_id in pairs:
            for logic in logics:
                tested += 1
                result = self.backtest_combination(
                    signals[sig_a_id], signals[sig_b_id],
                    logic, df)

                if result.get('insufficient_trades'):
                    continue

                passes = (
                    result['trades_per_year'] >= min_trades_yr and
                    result['win_rate'] >= min_win_rate and
                    result['sharpe'] >= min_sharpe and
                    abs(result['nifty_corr']) <= max_corr and
                    result['max_drawdown'] <= max_dd
                )

                if passes:
                    same_book = signals[sig_a_id]['book_id'] == signals[sig_b_id]['book_id']
                    survivors.append({
                        'sig_a': sig_a_id,
                        'sig_b': sig_b_id,
                        'book_a': signals[sig_a_id]['book_id'],
                        'book_b': signals[sig_b_id]['book_id'],
                        'logic': logic,
                        'same_book': same_book,
                        **result,
                    })

                if tested % 500 == 0:
                    print(f"  [{tested}/{total}] survivors: {len(survivors)}", flush=True)

        survivors.sort(key=lambda x: -x['sharpe'])

        # Save checkpoint
        os.makedirs('combination_results', exist_ok=True)
        with open('combination_results/screen_checkpoint.json', 'w') as f:
            json.dump(survivors, f, indent=2, default=str)

        print(f"  Screen complete: {len(survivors)} survivors from {tested} combos", flush=True)
        return survivors

    # ================================================================
    # WALK-FORWARD
    # ================================================================

    def run_walk_forward(self, sig_a, sig_b, logic, df=None):
        df = df if df is not None else self.df_full
        if len(df) < 200:
            return {'tier': 'DROP', 'reason': 'insufficient_data'}

        # Generate windows: 36 train, 12 test, 3 step
        dates = df['date']
        min_date = dates.min()
        max_date = dates.max()

        from dateutil.relativedelta import relativedelta
        windows = []
        train_start = min_date
        while True:
            train_end = train_start + relativedelta(months=36)
            test_start = train_end
            test_end = test_start + relativedelta(months=12)
            if test_end > max_date:
                break
            windows.append({
                'train_start': train_start, 'train_end': train_end,
                'test_start': test_start, 'test_end': test_end,
            })
            train_start += relativedelta(months=3)

        if len(windows) < 4:
            return {'tier': 'DROP', 'reason': 'insufficient_windows',
                    'total_windows': len(windows)}

        window_results = []
        for w in windows:
            test_df = df[(df['date'] >= w['test_start']) & (df['date'] < w['test_end'])].reset_index(drop=True)
            if len(test_df) < 20:
                window_results.append({'passed': False, 'sharpe': 0, 'trades': 0})
                continue

            result = self.backtest_combination(sig_a, sig_b, logic, test_df)
            passed = (not result.get('insufficient_trades') and
                      result.get('sharpe', 0) >= 0.5 and
                      result.get('trades', 0) >= 3)
            window_results.append({
                'passed': passed,
                'sharpe': result.get('sharpe', 0),
                'trades': result.get('trades', 0),
                'win_rate': result.get('win_rate', 0),
                'max_dd': result.get('max_drawdown', 0),
            })

        total = len(window_results)
        passed = sum(1 for w in window_results if w['passed'])
        pass_rate = passed / total
        last4 = window_results[-4:]
        l4p = sum(1 for w in last4 if w['passed'])
        l4r = l4p / 4

        sharpes = [w['sharpe'] for w in window_results if w['sharpe'] != 0]
        agg_sharpe = np.mean(sharpes) if sharpes else 0

        if pass_rate >= 0.60 and agg_sharpe >= 1.5 and l4r >= 0.75:
            tier = 'TIER_A'
        elif pass_rate >= 0.40 and agg_sharpe >= 1.0 and l4r >= 0.50:
            tier = 'TIER_B'
        else:
            tier = 'DROP'

        return {
            'tier': tier,
            'windows_passed': passed,
            'total_windows': total,
            'pass_rate': round(pass_rate, 3),
            'last4_passed': l4p,
            'last4_rate': round(l4r, 3),
            'aggregate_sharpe': round(agg_sharpe, 3),
            'window_details': window_results,
        }

    # ================================================================
    # PARAMETER SWEEP
    # ================================================================

    def run_parameter_sweep(self, sig_a, sig_b, logic, n_samples=150):
        results = []
        for _ in range(n_samples):
            stop = np.random.uniform(0.5, 5.0) / 100
            tp_choices = [0, 0.015, 0.02, 0.03, 0.04, 0.05]
            tp = np.random.choice(tp_choices)
            hold_choices = [0, 5, 10, 15, 20, 30]
            hold = int(np.random.choice(hold_choices))

            val = self.backtest_combination(sig_a, sig_b, logic, self.df_validation,
                                            stop, tp, hold)
            ins = self.backtest_combination(sig_a, sig_b, logic, self.df_insample,
                                            stop, tp, hold)

            if val.get('insufficient_trades') or ins.get('insufficient_trades'):
                continue

            overfit = val['sharpe'] > ins['sharpe'] * 2.0 if ins['sharpe'] > 0 else False

            results.append({
                'stop_pct': round(stop, 4),
                'tp_pct': round(tp, 4),
                'hold_days': hold,
                'val_sharpe': val['sharpe'],
                'ins_sharpe': ins['sharpe'],
                'val_wr': val['win_rate'],
                'val_trades': val['trades'],
                'overfit': overfit,
            })

        non_overfit = [r for r in results if not r['overfit']]
        non_overfit.sort(key=lambda x: -x['val_sharpe'])
        return non_overfit[:5]

    # ================================================================
    # OOS VALIDATION
    # ================================================================

    def validate_oos(self, sig_a, sig_b, logic, params=None):
        params = params or {}
        stop = params.get('stop_pct', 0.02)
        tp = params.get('tp_pct', 0.03)
        hold = params.get('hold_days', 20)

        oos = self.backtest_combination(sig_a, sig_b, logic, self.df_oos, stop, tp, hold)
        recent = self.backtest_combination(sig_a, sig_b, logic, self.df_recent, stop, tp, hold)
        ins = self.backtest_combination(sig_a, sig_b, logic, self.df_insample, stop, tp, hold)

        ins_sharpe = ins.get('sharpe', 0)
        oos_sharpe = oos.get('sharpe', 0)

        if oos.get('insufficient_trades'):
            verdict = 'INSUFFICIENT_TRADES'
        elif ins_sharpe > 0 and oos_sharpe < ins_sharpe * 0.4:
            verdict = 'OVERFIT'
        else:
            verdict = 'VALIDATED'

        return {
            'oos_sharpe': oos.get('sharpe', 0),
            'oos_trades': oos.get('trades', 0),
            'oos_win_rate': oos.get('win_rate', 0),
            'oos_pnl': oos.get('total_pnl_pts', 0),
            'oos_max_dd': oos.get('max_drawdown', 0),
            'recent_sharpe': recent.get('sharpe', 0),
            'recent_pnl': recent.get('total_pnl_pts', 0),
            'recent_trades': recent.get('trades', 0),
            'ins_sharpe': ins_sharpe,
            'verdict': verdict,
        }

    # ================================================================
    # CROSS-BOOK ANALYSIS
    # ================================================================

    def cross_book_analysis(self, signals, screen_results):
        same_book = [r for r in screen_results if r['same_book']]
        cross_book = [r for r in screen_results if not r['same_book']]

        book_pairs = defaultdict(list)
        for r in cross_book:
            pair = tuple(sorted([r['book_a'], r['book_b']]))
            book_pairs[pair].append(r)

        return {
            'same_book_count': len(same_book),
            'cross_book_count': len(cross_book),
            'book_pair_stats': {
                f"{p[0]}+{p[1]}": {
                    'count': len(results),
                    'avg_sharpe': round(np.mean([r['sharpe'] for r in results]), 2),
                    'best': max(results, key=lambda x: x['sharpe']),
                }
                for p, results in sorted(book_pairs.items(), key=lambda x: -len(x[1]))
            }
        }

    # ================================================================
    # REPORT
    # ================================================================

    def generate_report(self, all_results, output_path):
        screen = all_results.get('screen', [])
        wf = all_results.get('walkforward', [])
        oos = all_results.get('oos', [])

        tier_a = [r for r in wf if r.get('tier') == 'TIER_A']
        tier_b = [r for r in wf if r.get('tier') == 'TIER_B']
        validated = [r for r in (oos or []) if r.get('verdict') == 'VALIDATED']

        lines = [
            f"# Combination Testing Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            "## Executive Summary",
            f"- Signals tested: {len(set(r['sig_a'] for r in screen) | set(r['sig_b'] for r in screen)) if screen else 0}",
            f"- Combinations screened: {len(screen)} survivors",
            f"- Walk-forward tested: {len(wf)}",
            f"- Tier A: {len(tier_a)}",
            f"- Tier B: {len(tier_b)}",
            f"- OOS validated: {len(validated)}",
            "",
        ]

        if tier_a or tier_b:
            lines.append("## Tier A/B Combinations")
            lines.append("")
            lines.append("| Sig A | Sig B | Logic | Sharpe | WR | WF% | L4 | Tier |")
            lines.append("|-------|-------|-------|--------|-----|-----|-----|------|")
            for r in tier_a + tier_b:
                lines.append(
                    f"| {r['sig_a']} | {r['sig_b']} | {r['logic']} | "
                    f"{r.get('aggregate_sharpe', r.get('sharpe', 0)):.2f} | "
                    f"{r.get('win_rate', 0):.0%} | {r.get('pass_rate', 0):.0%} | "
                    f"{r.get('last4_passed', 0)}/4 | {r['tier']} |"
                )
            lines.append("")

        if validated:
            lines.append("## OOS Validated")
            lines.append("")
            for r in validated:
                lines.append(f"- **{r['sig_a']} + {r['sig_b']}** ({r['logic']}): "
                             f"OOS Sharpe={r['oos_sharpe']:.2f}, "
                             f"Recent P&L={r['recent_pnl']:+.0f}")
            lines.append("")

        lines.append("## Baselines")
        lines.append("- DRY_20 alone: Sharpe 2.60, WR 47%, MaxDD 24.3%, Recent +60pts")
        lines.append("- SCORING system: Sharpe 2.30, MaxDD 12.2%, Recent +340pts")
        lines.append("")

        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))
        print(f"Report saved to {output_path}")
