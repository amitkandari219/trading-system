"""
Confirmation filter screen: test DSL pool signals as entry filters for DRY_20.

Steps:
1. Screen DSL pool for confirmation candidates
2. Test each as DRY_20 confirmation filter
3. Test non-Kaufman signals specifically
4. Multi-book combinations
5. Failed signals as regime indicators
"""

import json
import os
from collections import defaultdict
from datetime import date

import numpy as np
import pandas as pd
import psycopg2

from backtest.indicators import add_all_indicators, historical_volatility
from config.settings import DATABASE_DSN

RESULTS_DIR = 'backtest_results'


# ================================================================
# DRY_20 SIGNAL LOGIC
# ================================================================

def dry20_entry_long(row, prev):
    if pd.isna(row['sma_10']) or pd.isna(row['stoch_k_5']):
        return False
    return row['sma_10'] < prev['close'] and row['stoch_k_5'] > 50

def dry20_exit_long(row, prev):
    if pd.isna(row['stoch_k_5']):
        return False
    return row['stoch_k_5'] <= 50


# ================================================================
# GENERIC CONDITION EVALUATOR (for DSL rules)
# ================================================================

def eval_condition(row, prev, cond):
    """Evaluate a single backtest condition dict."""
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
        prev_ind = prev.get(indicator, np.nan)
        if isinstance(value, str) and value in prev.index:
            prev_target = prev.get(value, np.nan)
        else:
            prev_target = target
        if pd.isna(prev_ind) or pd.isna(prev_target):
            return False
        return prev_ind <= prev_target and ind_val > target
    elif op == 'crosses_below':
        if prev is None: return False
        prev_ind = prev.get(indicator, np.nan)
        if isinstance(value, str) and value in prev.index:
            prev_target = prev.get(value, np.nan)
        else:
            prev_target = target
        if pd.isna(prev_ind) or pd.isna(prev_target):
            return False
        return prev_ind >= prev_target and ind_val < target
    return False


def eval_conditions(row, prev, conditions):
    """All conditions must be true (AND logic)."""
    if not conditions:
        return False
    return all(eval_condition(row, prev, c) for c in conditions)


# ================================================================
# BACKTEST ENGINE
# ================================================================

def run_backtest(df, entry_fn, exit_fn, stop_pct=0.02, tp_pct=0.03):
    """Simple backtest returning trades list."""
    closes = df['close'].values
    dates = df['date'].values
    n = len(df)
    trades = []
    position = None
    entry_price = 0.0
    entry_idx = 0

    for i in range(1, n):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        price = closes[i]

        if position is None:
            if entry_fn(row, prev):
                position = 'LONG'
                entry_price = price
                entry_idx = i
        else:
            loss_pct = (entry_price - price) / entry_price
            if loss_pct >= stop_pct:
                trades.append({'pnl': price - entry_price, 'entry_price': entry_price,
                               'entry_date': dates[entry_idx], 'exit_date': dates[i]})
                position = None
                continue

            if tp_pct > 0:
                gain_pct = (price - entry_price) / entry_price
                if gain_pct >= tp_pct:
                    trades.append({'pnl': price - entry_price, 'entry_price': entry_price,
                                   'entry_date': dates[entry_idx], 'exit_date': dates[i]})
                    position = None
                    continue

            if exit_fn(row, prev):
                trades.append({'pnl': price - entry_price, 'entry_price': entry_price,
                               'entry_date': dates[entry_idx], 'exit_date': dates[i]})
                position = None

    if position is not None:
        trades.append({'pnl': closes[-1] - entry_price, 'entry_price': entry_price,
                       'entry_date': dates[entry_idx], 'exit_date': dates[-1]})

    return trades


def compute_metrics(trades, df):
    if len(trades) < 5:
        return {'trades': len(trades), 'sharpe': 0, 'win_rate': 0, 'max_dd': 1.0,
                'pnl_pts': sum(t['pnl'] for t in trades) if trades else 0,
                'profit_factor': 0, 'nifty_corr': 0}

    notional = trades[0]['entry_price']
    pnls = [t['pnl'] for t in trades]
    pct_returns = [p / notional for p in pnls]
    wins = [r for r in pct_returns if r > 0]

    ret = pd.Series(pct_returns)
    std = ret.std()
    sharpe = (ret.mean() / std * np.sqrt(252)) if std > 0 else 0

    equity = [1.0]
    for r in pct_returns:
        equity.append(equity[-1] * (1 + r))
    eq = pd.Series(equity)
    dd = (eq - eq.cummax()) / eq.cummax()
    max_dd = abs(dd.min())

    gw = sum(p for p in pnls if p > 0)
    gl = abs(sum(p for p in pnls if p < 0))
    pf = gw / gl if gl > 0 else 99

    trade_ret = {}
    for t in trades:
        d = t['exit_date']
        trade_ret[d] = trade_ret.get(d, 0) + t['pnl'] / notional
    nifty_ret = df.set_index('date')['close'].pct_change()
    strat_ret = pd.Series(trade_ret)
    common = strat_ret.index.intersection(nifty_ret.index)
    corr = float(strat_ret.reindex(common).corr(nifty_ret.reindex(common))) if len(common) > 10 else 0
    if np.isnan(corr): corr = 0

    return {
        'trades': len(trades),
        'sharpe': round(sharpe, 3),
        'win_rate': round(len(wins) / len(pct_returns), 3),
        'max_dd': round(max_dd, 3),
        'profit_factor': round(min(pf, 99), 2),
        'nifty_corr': round(corr, 3),
        'pnl_pts': round(sum(pnls), 1),
    }


# ================================================================
# MAIN
# ================================================================

def main():
    print("Loading data...", flush=True)
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
    print(f"  {len(df)} trading days", flush=True)

    df_6mo = df[df['date'] >= '2025-09-15'].reset_index(drop=True)
    years = (df['date'].max() - df['date'].min()).days / 365.25

    # DRY_20 baseline
    print("\nDRY_20 baseline...", flush=True)
    dry20_trades = run_backtest(df, dry20_entry_long, dry20_exit_long, 0.02, 0.0)
    dry20_m = compute_metrics(dry20_trades, df)
    dry20_6mo = run_backtest(df_6mo, dry20_entry_long, dry20_exit_long, 0.02, 0.0)
    dry20_6m = compute_metrics(dry20_6mo, df_6mo)
    print(f"  10yr: Sh={dry20_m['sharpe']:.2f} WR={dry20_m['win_rate']:.0%} "
          f"Tr={dry20_m['trades']} MaxDD={dry20_m['max_dd']:.1%}")
    print(f"  6mo:  Sh={dry20_6m['sharpe']:.2f} P&L={dry20_6m['pnl_pts']:+.0f}")

    # Pre-compute DRY_20 entry days for overlap calculation
    dry20_entry_days = set()
    for i in range(1, len(df)):
        if dry20_entry_long(df.iloc[i], df.iloc[i-1]):
            dry20_entry_days.add(df.iloc[i]['date'])
    print(f"  DRY_20 fires on {len(dry20_entry_days)} days over {years:.1f} years")

    # ================================================================
    # STEP 1: Screen DSL pool
    # ================================================================
    print(f"\n{'='*80}")
    print("STEP 1: Screen DSL pool for confirmation candidates")
    print(f"{'='*80}")

    dsl_dir = 'dsl_results/PASS'
    exclude = {'KAUFMAN_DRY_20', 'KAUFMAN_DRY_16', 'KAUFMAN_DRY_12'}
    candidates = []

    for fname in sorted(os.listdir(dsl_dir)):
        if not fname.endswith('.json'):
            continue
        with open(os.path.join(dsl_dir, fname)) as f:
            data = json.load(f)
        sid = data['signal_id']
        if sid in exclude:
            continue

        rules = data.get('backtest_rule', {})
        entry_long = rules.get('entry_long', [])
        if not entry_long:
            continue

        # Count how many days this signal fires LONG
        fire_days = set()
        for i in range(1, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i-1]
            if eval_conditions(row, prev, entry_long):
                fire_days.add(row['date'])

        fires_per_year = len(fire_days) / years
        overlap_days = fire_days & dry20_entry_days
        overlap_rate = len(overlap_days) / len(dry20_entry_days) if dry20_entry_days else 0

        # Quick backtest for correlation
        def make_entry(el):
            def fn(row, prev):
                return eval_conditions(row, prev, el)
            return fn

        entry_fn = make_entry(entry_long)
        quick_trades = run_backtest(df, entry_fn, lambda r, p: False, 0.02, 0.03)
        if len(quick_trades) < 10:
            continue

        notional = quick_trades[0]['entry_price']
        trade_ret = {}
        for t in quick_trades:
            d = t['exit_date']
            trade_ret[d] = trade_ret.get(d, 0) + t['pnl'] / notional
        nifty_ret = df.set_index('date')['close'].pct_change()
        strat_ret = pd.Series(trade_ret)
        common = strat_ret.index.intersection(nifty_ret.index)
        corr = float(strat_ret.reindex(common).corr(nifty_ret.reindex(common))) if len(common) > 10 else 0
        if np.isnan(corr): corr = 0

        # Filter criteria
        if abs(corr) >= 0.5:
            continue
        if fires_per_year < 10:
            continue

        book_id = sid.split('_')[0]
        candidates.append({
            'signal_id': sid,
            'book_id': book_id,
            'entry_long': entry_long,
            'fires_per_year': round(fires_per_year, 1),
            'overlap_days': len(overlap_days),
            'overlap_rate': round(overlap_rate, 3),
            'nifty_corr': round(corr, 3),
            'filename': fname,
        })

    print(f"  Candidates passing screen: {len(candidates)}")
    candidates.sort(key=lambda x: -x['overlap_rate'])
    for c in candidates[:10]:
        print(f"    {c['signal_id']:25s} book={c['book_id']:10s} "
              f"fires={c['fires_per_year']:5.1f}/yr overlap={c['overlap_rate']:.0%} "
              f"corr={c['nifty_corr']:+.3f}")

    # ================================================================
    # STEP 2: Test each as DRY_20 confirmation
    # ================================================================
    print(f"\n{'='*80}")
    print("STEP 2: Test each candidate as DRY_20 confirmation filter")
    print(f"{'='*80}")

    useful_filters = []
    all_confirmation_results = []

    for c in candidates:
        sid = c['signal_id']
        entry_long_conds = c['entry_long']

        def make_combined_entry(conds):
            def fn(row, prev):
                return dry20_entry_long(row, prev) and eval_conditions(row, prev, conds)
            return fn

        combined_entry = make_combined_entry(entry_long_conds)
        trades_10y = run_backtest(df, combined_entry, dry20_exit_long, 0.02, 0.03)
        m10 = compute_metrics(trades_10y, df)

        trades_6mo = run_backtest(df_6mo, combined_entry, dry20_exit_long, 0.02, 0.03)
        m6 = compute_metrics(trades_6mo, df_6mo)

        result = {
            'signal_id': sid,
            'book_id': c['book_id'],
            'overlap_rate': c['overlap_rate'],
            '10yr': m10,
            '6mo': m6,
        }
        all_confirmation_results.append(result)

        # Check USEFUL_FILTER criteria
        is_useful = (
            m10['win_rate'] >= 0.52 and
            m10['sharpe'] >= 2.0 and
            m10['max_dd'] <= 0.20 and
            c['overlap_rate'] >= 0.20 and
            m6['pnl_pts'] >= 60
        )

        if is_useful:
            useful_filters.append(result)

    # Sort by 10yr Sharpe
    all_confirmation_results.sort(key=lambda x: -x['10yr']['sharpe'])

    print(f"\n  Results (top 15 by Sharpe):")
    print(f"  {'Signal':25s} {'Tr':>4s} {'WR':>5s} {'Sh':>6s} {'DD':>5s} {'Ovlp':>5s} {'6mPL':>7s} {'Flag':>8s}")
    print(f"  {'-'*25} {'-'*4} {'-'*5} {'-'*6} {'-'*5} {'-'*5} {'-'*7} {'-'*8}")
    print(f"  {'DRY_20 baseline':25s} {dry20_m['trades']:4d} {dry20_m['win_rate']:4.0%} "
          f"{dry20_m['sharpe']:6.2f} {dry20_m['max_dd']:4.1%} {'100%':>5s} "
          f"{dry20_6m['pnl_pts']:+7.0f}")
    for r in all_confirmation_results[:15]:
        m = r['10yr']
        m6 = r['6mo']
        flag = 'USEFUL' if r in useful_filters else ''
        print(f"  {r['signal_id']:25s} {m['trades']:4d} {m['win_rate']:4.0%} "
              f"{m['sharpe']:6.2f} {m['max_dd']:4.1%} {r['overlap_rate']:4.0%} "
              f"{m6['pnl_pts']:+7.0f} {flag:>8s}")

    print(f"\n  USEFUL_FILTER signals: {len(useful_filters)}")

    # ================================================================
    # STEP 3: Non-Kaufman signals
    # ================================================================
    print(f"\n{'='*80}")
    print("STEP 3: Non-Kaufman confirmation signals")
    print(f"{'='*80}")

    non_kaufman_books = {'GUJRAL', 'MCMILLAN', 'GRIMES', 'HILPISCH'}
    non_kaufman = [r for r in all_confirmation_results if r['book_id'] in non_kaufman_books]

    if non_kaufman:
        # Top 3 per book by overlap rate
        by_book = defaultdict(list)
        for r in non_kaufman:
            by_book[r['book_id']].append(r)

        print(f"  Non-Kaufman candidates tested: {len(non_kaufman)}")
        for book_id, book_results in sorted(by_book.items()):
            book_results.sort(key=lambda x: -x['overlap_rate'])
            print(f"\n  {book_id} (top 3 by overlap):")
            for r in book_results[:3]:
                m = r['10yr']
                m6 = r['6mo']
                flag = 'USEFUL' if r in useful_filters else ''
                print(f"    {r['signal_id']:25s} Tr={m['trades']:3d} WR={m['win_rate']:4.0%} "
                      f"Sh={m['sharpe']:5.2f} DD={m['max_dd']:4.1%} "
                      f"Ovlp={r['overlap_rate']:.0%} 6m={m6['pnl_pts']:+.0f} {flag}")

        best_non_kaufman = max(non_kaufman, key=lambda x: x['10yr']['sharpe'])
        print(f"\n  Best non-Kaufman by Sharpe: {best_non_kaufman['signal_id']} "
              f"(Sh={best_non_kaufman['10yr']['sharpe']:.2f})")
    else:
        print("  No non-Kaufman candidates found")
        best_non_kaufman = None

    # ================================================================
    # STEP 4: Multi-book combination
    # ================================================================
    print(f"\n{'='*80}")
    print("STEP 4: Multi-book combinations")
    print(f"{'='*80}")

    if useful_filters and best_non_kaufman:
        best_kaufman_filter = max(
            [r for r in useful_filters if r['book_id'] == 'KAUFMAN'],
            key=lambda x: x['10yr']['sharpe'],
            default=None
        )

        if best_kaufman_filter:
            kf_conds = None
            nkf_conds = None
            # Reload conditions
            for c in candidates:
                if c['signal_id'] == best_kaufman_filter['signal_id']:
                    kf_conds = c['entry_long']
                if c['signal_id'] == best_non_kaufman['signal_id']:
                    nkf_conds = c['entry_long']

            if kf_conds and nkf_conds:
                def triple_entry(row, prev):
                    return (dry20_entry_long(row, prev) and
                            eval_conditions(row, prev, kf_conds) and
                            eval_conditions(row, prev, nkf_conds))

                t_10y = run_backtest(df, triple_entry, dry20_exit_long, 0.02, 0.03)
                t_6mo = run_backtest(df_6mo, triple_entry, dry20_exit_long, 0.02, 0.03)
                m10 = compute_metrics(t_10y, df)
                m6 = compute_metrics(t_6mo, df_6mo)

                print(f"  DRY_20 + {best_kaufman_filter['signal_id']} + {best_non_kaufman['signal_id']}:")
                print(f"    10yr: Tr={m10['trades']} WR={m10['win_rate']:.0%} "
                      f"Sh={m10['sharpe']:.2f} DD={m10['max_dd']:.1%}")
                print(f"    6mo:  Tr={m6['trades']} P&L={m6['pnl_pts']:+.0f}")
            else:
                print("  Could not load conditions for multi-book test")
        else:
            print("  No Kaufman filter among useful filters")
    else:
        print("  Insufficient useful filters for multi-book test")

    # ================================================================
    # STEP 5: Failed signals as regime indicators
    # ================================================================
    print(f"\n{'='*80}")
    print("STEP 5: Failed signals as regime indicators")
    print(f"{'='*80}")

    # Get top 10 signals by full-period Sharpe that weren't in the useful filters
    # These are signals with some predictive power but failed walk-forward
    from backtest.generic_backtest import run_generic_backtest

    # Load all DSL signals and find high-Sharpe non-survivors
    regime_candidates = []
    for fname in sorted(os.listdir(dsl_dir)):
        if not fname.endswith('.json'):
            continue
        with open(os.path.join(dsl_dir, fname)) as f:
            data = json.load(f)
        sid = data['signal_id']
        if sid in exclude:
            continue

        rules = data.get('backtest_rule', {})
        entry_long = rules.get('entry_long', [])
        if not entry_long:
            continue

        # Quick full-period evaluation
        def make_e(el):
            def fn(row, prev):
                return eval_conditions(row, prev, el)
            return fn

        tr = run_backtest(df, make_e(entry_long), lambda r, p: False, 0.02, 0.03)
        if len(tr) < 20:
            continue

        notional = tr[0]['entry_price']
        pnls = [t['pnl'] / notional for t in tr]
        std = pd.Series(pnls).std()
        sharpe = (pd.Series(pnls).mean() / std * np.sqrt(252)) if std > 0 else 0

        regime_candidates.append({
            'signal_id': sid,
            'entry_long': entry_long,
            'sharpe': sharpe,
            'trades': tr,
        })

    regime_candidates.sort(key=lambda x: -x['sharpe'])
    top_regime = regime_candidates[:10]

    print(f"  Testing top {len(top_regime)} signals as regime indicators")

    regime_indicators = []
    for rc in top_regime:
        sid = rc['signal_id']
        sig_trades = rc['trades']

        # Build streak tracker: for each date, how many of the last 5 signals were winners
        win_dates = set()
        streak_dates = {}  # date -> streak count

        sorted_trades = sorted(sig_trades, key=lambda t: t['exit_date'])
        recent_outcomes = []
        for t in sorted_trades:
            recent_outcomes.append(1 if t['pnl'] > 0 else 0)
            if len(recent_outcomes) > 5:
                recent_outcomes.pop(0)
            streak = sum(recent_outcomes[-3:]) if len(recent_outcomes) >= 3 else 0
            streak_dates[t['exit_date']] = streak

        # For each DRY_20 trade, check if regime indicator was favorable
        favorable_trades = []
        unfavorable_trades = []

        for dt in dry20_trades:
            entry_d = dt['entry_date']
            # Find most recent streak value before this entry
            best_streak = 0
            for sd, sv in streak_dates.items():
                if sd <= entry_d:
                    best_streak = sv

            if best_streak >= 2:
                favorable_trades.append(dt)
            else:
                unfavorable_trades.append(dt)

        if len(favorable_trades) >= 5 and len(unfavorable_trades) >= 5:
            fav_pnls = [t['pnl'] / dry20_trades[0]['entry_price'] for t in favorable_trades]
            unfav_pnls = [t['pnl'] / dry20_trades[0]['entry_price'] for t in unfavorable_trades]

            fav_std = pd.Series(fav_pnls).std()
            unfav_std = pd.Series(unfav_pnls).std()
            fav_sharpe = (pd.Series(fav_pnls).mean() / fav_std * np.sqrt(252)) if fav_std > 0 else 0
            unfav_sharpe = (pd.Series(unfav_pnls).mean() / unfav_std * np.sqrt(252)) if unfav_std > 0 else 0

            improvement = fav_sharpe / unfav_sharpe if unfav_sharpe > 0 else 0

            is_regime_indicator = fav_sharpe >= 2 * unfav_sharpe and fav_sharpe > 0

            regime_indicators.append({
                'signal_id': sid,
                'favorable_trades': len(favorable_trades),
                'unfavorable_trades': len(unfavorable_trades),
                'favorable_sharpe': round(fav_sharpe, 2),
                'unfavorable_sharpe': round(unfav_sharpe, 2),
                'improvement_ratio': round(improvement, 2),
                'is_regime_indicator': is_regime_indicator,
            })

    regime_indicators.sort(key=lambda x: -x['improvement_ratio'])

    print(f"\n  {'Signal':25s} {'Fav Tr':>6s} {'Fav Sh':>7s} {'Unfav Tr':>8s} {'Unfav Sh':>8s} {'Impr':>6s} {'Flag':>10s}")
    print(f"  {'-'*25} {'-'*6} {'-'*7} {'-'*8} {'-'*8} {'-'*6} {'-'*10}")
    for ri in regime_indicators:
        flag = 'REGIME_IND' if ri['is_regime_indicator'] else ''
        print(f"  {ri['signal_id']:25s} {ri['favorable_trades']:6d} {ri['favorable_sharpe']:7.2f} "
              f"{ri['unfavorable_trades']:8d} {ri['unfavorable_sharpe']:8.2f} "
              f"{ri['improvement_ratio']:5.1f}x {flag:>10s}")

    # ================================================================
    # SAVE
    # ================================================================
    os.makedirs(RESULTS_DIR, exist_ok=True)

    output = {
        'dry20_baseline_10yr': dry20_m,
        'dry20_baseline_6mo': dry20_6m,
        'step1_candidates': len(candidates),
        'step2_all_results': [{
            'signal_id': r['signal_id'],
            'book_id': r['book_id'],
            'overlap_rate': r['overlap_rate'],
            '10yr': r['10yr'],
            '6mo': r['6mo'],
        } for r in all_confirmation_results],
        'step2_useful_filters': [{
            'signal_id': r['signal_id'],
            'book_id': r['book_id'],
            'overlap_rate': r['overlap_rate'],
            '10yr': r['10yr'],
            '6mo': r['6mo'],
        } for r in useful_filters],
        'step5_regime_indicators': regime_indicators,
    }

    with open(os.path.join(RESULTS_DIR, 'confirmation_screen_results.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)

    # Final recommendation
    print(f"\n{'='*80}")
    print("FINAL RECOMMENDATION")
    print(f"{'='*80}")

    if useful_filters:
        best = max(useful_filters, key=lambda x: x['10yr']['sharpe'])
        print(f"  Best USEFUL_FILTER: {best['signal_id']}")
        print(f"    10yr: Sh={best['10yr']['sharpe']:.2f} WR={best['10yr']['win_rate']:.0%} "
              f"DD={best['10yr']['max_dd']:.1%} Tr={best['10yr']['trades']}")
        print(f"    6mo:  P&L={best['6mo']['pnl_pts']:+.0f}")
        print(f"    Overlap: {best['overlap_rate']:.0%} of DRY_20 entries confirmed")
    else:
        print("  No USEFUL_FILTER found — DRY_20 alone is the best option")
        # Find the closest miss
        if all_confirmation_results:
            best_combo = max(all_confirmation_results, key=lambda x: x['10yr']['sharpe'])
            m = best_combo['10yr']
            print(f"\n  Closest miss: {best_combo['signal_id']}")
            print(f"    10yr: Sh={m['sharpe']:.2f} WR={m['win_rate']:.0%} DD={m['max_dd']:.1%}")
            fails = []
            if m['win_rate'] < 0.52: fails.append(f"WR={m['win_rate']:.0%}<52%")
            if m['sharpe'] < 2.0: fails.append(f"Sh={m['sharpe']:.2f}<2.0")
            if m['max_dd'] > 0.20: fails.append(f"DD={m['max_dd']:.1%}>20%")
            if best_combo['overlap_rate'] < 0.20: fails.append(f"overlap={best_combo['overlap_rate']:.0%}<20%")
            if best_combo['6mo']['pnl_pts'] < 60: fails.append(f"6mo={best_combo['6mo']['pnl_pts']:+.0f}<+60")
            print(f"    Failed criteria: {', '.join(fails)}")

    ri_found = [r for r in regime_indicators if r['is_regime_indicator']]
    if ri_found:
        print(f"\n  REGIME_INDICATORS found: {len(ri_found)}")
        for ri in ri_found:
            print(f"    {ri['signal_id']}: {ri['improvement_ratio']:.1f}x improvement "
                  f"({ri['favorable_sharpe']:.2f} vs {ri['unfavorable_sharpe']:.2f})")
    else:
        print(f"\n  No REGIME_INDICATORS found (no signal's winning streak predicts DRY_20 at 2x)")

    print(f"\n  Results saved to {RESULTS_DIR}/confirmation_screen_results.json")


if __name__ == '__main__':
    main()
