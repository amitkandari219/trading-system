"""
Round 2 Backtest: Walk-forward on 211 DSL-validated signals.

Pipeline:
1. Load DSL-translated signals from dsl_results/PASS/
2. Full-dataset backtest → correlation filter (|corr| > 0.5 → reject)
3. Walk-forward on survivors (36/12/3 windows)
4. Tier classification (A/B/DROP)
5. Condition frequency analysis across survivors

Criteria:
  Tier A: pass_rate >= 60%, agg_sharpe >= 1.5, last4 >= 75%, worst_dd <= 20%
  Tier B: pass_rate >= 40%, agg_sharpe >= 1.0, last4 >= 50%
  Min trades per window: 20
  Correlation filter: |corr| > 0.5 → reject
"""

import json
import os
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import psycopg2

from backtest.generic_backtest import run_generic_backtest, make_generic_backtest_fn
from backtest.indicators import sma, adx
from backtest.walk_forward import WalkForwardEngine
from backtest.types import harmonic_mean_sharpe
from config.settings import DATABASE_DSN

RESULTS_DIR = 'round2_results'
DSL_PASS_DIR = 'dsl_results/PASS'
MAX_NIFTY_CORR = 0.50
MIN_TRADES_PER_WINDOW = 20


def classify_regime(history_df, start_date, end_date):
    mask = (history_df['date'] >= start_date) & (history_df['date'] <= end_date)
    window_df = history_df[mask].copy()
    if len(window_df) < 20:
        return 'UNKNOWN'
    close = window_df['close']
    sma_200 = sma(history_df['close'], 200)
    window_sma = sma_200[mask]
    above_sma = (close > window_sma).mean() if window_sma.notna().sum() > 0 else 0.5
    adx_vals = adx(history_df)
    window_adx = adx_vals[mask]
    mean_adx = window_adx.mean() if window_adx.notna().sum() > 0 else 20
    vix = window_df.get('india_vix')
    mean_vix = vix.mean() if vix is not None and vix.notna().sum() > 0 else 15
    net_return = (close.iloc[-1] / close.iloc[0] - 1) if len(close) > 1 else 0
    if mean_vix > 22:
        return 'HIGH_VOL'
    elif mean_adx > 25:
        return 'TRENDING_UP' if net_return > 0.05 and above_sma > 0.6 else \
               'TRENDING_DOWN' if net_return < -0.05 and above_sma < 0.4 else \
               ('TRENDING_UP' if above_sma > 0.5 else 'TRENDING_DOWN')
    return 'RANGING'


def tier_signal(window_details):
    total = len(window_details)
    if total < 4:
        return 'DROP', 'INSUFFICIENT_WINDOWS'
    passed = sum(1 for w in window_details if w['passed'])
    pass_rate = passed / total
    last4 = window_details[-4:]
    last4_passed = sum(1 for w in last4 if w['passed'])
    last4_rate = last4_passed / 4
    agg_sharpe = harmonic_mean_sharpe(
        [{'result': w['result']} for w in window_details]
    )
    zero_windows = sum(1 for w in window_details if w['result'].trade_count == 0)
    zero_pct = zero_windows / total
    active_dds = [w['result'].max_drawdown for w in window_details
                  if w['result'].trade_count > 0]
    worst_dd = max(active_dds) if active_dds else 1.0

    if zero_pct > 0.6:
        return 'DROP', f'ZERO_TRADES_{zero_pct:.0%}_OF_WINDOWS'
    if last4_rate == 0:
        return 'DROP', 'LAST_4_ALL_FAIL'
    if passed <= 2:
        return 'DROP', f'ONLY_{passed}_WINDOWS_PASS'

    if (pass_rate >= 0.60 and agg_sharpe >= 1.5 and
            last4_rate >= 0.75 and worst_dd <= 0.20):
        return 'TIER_A', None
    if (pass_rate >= 0.40 and agg_sharpe >= 1.0 and
            last4_rate >= 0.50):
        return 'TIER_B', None

    reasons = []
    if pass_rate < 0.40:
        reasons.append(f'pass_rate={pass_rate:.0%}<40%')
    if agg_sharpe < 1.0:
        reasons.append(f'sharpe={agg_sharpe:.2f}<1.0')
    if last4_rate < 0.50:
        reasons.append(f'last4={last4_rate:.0%}<50%')
    return 'DROP', '; '.join(reasons) if reasons else 'BELOW_TIER_B'


def regime_profile(window_details):
    regime_stats = {}
    for w in window_details:
        regime = w.get('regime', 'UNKNOWN')
        if regime not in regime_stats:
            regime_stats[regime] = {'total': 0, 'passed': 0, 'sharpes': [], 'trades': 0}
        regime_stats[regime]['total'] += 1
        if w['passed']:
            regime_stats[regime]['passed'] += 1
        if w['result'].sharpe != 0:
            regime_stats[regime]['sharpes'].append(w['result'].sharpe)
        regime_stats[regime]['trades'] += w['result'].trade_count
    profile = {}
    for regime, s in regime_stats.items():
        avg_sharpe = np.mean(s['sharpes']) if s['sharpes'] else 0
        profile[regime] = {
            'windows': s['total'],
            'passed': s['passed'],
            'pass_rate': round(s['passed'] / s['total'], 2) if s['total'] > 0 else 0,
            'avg_sharpe': round(avg_sharpe, 2),
            'total_trades': s['trades'],
        }
    return profile


def extract_conditions(rules):
    """Extract all conditions from a rules dict for frequency analysis."""
    conditions = []
    for side in ['entry_long', 'entry_short', 'exit_long', 'exit_short']:
        for cond in rules.get(side, []):
            indicator = cond.get('indicator', '')
            op = cond.get('op', '')
            value = cond.get('value', '')
            # Normalize: use indicator name and value type
            if isinstance(value, str):
                conditions.append(f"{side}:{indicator} {op} {value}")
            else:
                conditions.append(f"{side}:{indicator} {op} <numeric>")
    return conditions


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ================================================================
    # LOAD DATA
    # ================================================================
    print("Loading market data...", flush=True)
    conn = psycopg2.connect(DATABASE_DSN)
    df = pd.read_sql(
        "SELECT date, open, high, low, close, volume, india_vix "
        "FROM nifty_daily ORDER BY date", conn
    )
    cal = pd.read_sql(
        "SELECT trading_date AS date, is_trading_day "
        "FROM market_calendar ORDER BY trading_date", conn
    )
    conn.close()
    df['date'] = pd.to_datetime(df['date'])
    cal['date'] = pd.to_datetime(cal['date'])
    print(f"  {len(df)} trading days", flush=True)

    from regime_labeler import RegimeLabeler
    regime_labels = RegimeLabeler().label_full_history(df)

    # ================================================================
    # LOAD DSL SIGNALS
    # ================================================================
    print("\nLoading DSL-translated signals...", flush=True)
    signals = []
    for fname in sorted(os.listdir(DSL_PASS_DIR)):
        if not fname.endswith('.json'):
            continue
        with open(os.path.join(DSL_PASS_DIR, fname)) as f:
            data = json.load(f)
        rules = data.get('backtest_rule')
        if rules:
            signals.append({
                'signal_id': data['signal_id'],
                'rules': rules,
                'filename': fname,
            })
    print(f"  Loaded {len(signals)} DSL signals", flush=True)

    # ================================================================
    # STEP 1: FULL-DATASET SCREEN + CORRELATION FILTER
    # ================================================================
    print(f"\n{'='*90}")
    print(f"STEP 1: FULL-DATASET BACKTEST + CORRELATION FILTER (|corr| > {MAX_NIFTY_CORR})")
    print(f"{'='*90}")

    alpha_candidates = []
    rejected_corr = 0
    rejected_no_trades = 0
    rejected_error = 0

    start = time.time()
    for i, sig in enumerate(signals):
        sid = sig['signal_id']
        rules = sig['rules']

        try:
            result = run_generic_backtest(rules, df, {})
        except Exception as e:
            rejected_error += 1
            continue

        if result.trade_count < 15:
            rejected_no_trades += 1
            continue

        if abs(result.nifty_correlation) > MAX_NIFTY_CORR:
            rejected_corr += 1
            continue

        alpha_candidates.append({
            'signal_id': sid,
            'rules': rules,
            'full_result': result,
            'filename': sig['filename'],
        })

        done = i + 1
        if done % 25 == 0 or done == len(signals):
            elapsed = time.time() - start
            print(f"  [{done}/{len(signals)}] ({elapsed:.0f}s) "
                  f"Alpha:{len(alpha_candidates)} CorrKill:{rejected_corr} "
                  f"NoTrade:{rejected_no_trades} Err:{rejected_error}",
                  flush=True)

    print(f"\n  ALPHA SCREEN RESULTS:")
    print(f"    Input DSL signals:    {len(signals)}")
    print(f"    Rejected (corr):      {rejected_corr}")
    print(f"    Rejected (trades):    {rejected_no_trades}")
    print(f"    Rejected (errors):    {rejected_error}")
    print(f"    ALPHA CANDIDATES:     {len(alpha_candidates)}")

    if not alpha_candidates:
        print("\n  NO ALPHA CANDIDATES. Exiting.")
        return

    # ================================================================
    # STEP 2: WALK-FORWARD
    # ================================================================
    print(f"\n{'='*90}")
    print(f"STEP 2: WALK-FORWARD ON {len(alpha_candidates)} CANDIDATES")
    print(f"{'='*90}")

    wfe = WalkForwardEngine(cal)
    for criteria in wfe.CRITERIA.values():
        criteria['min_trades'] = MIN_TRADES_PER_WINDOW
        criteria['min_sharpe'] = 0.8
        if 'min_calmar' in criteria:
            del criteria['min_calmar']
        if 'min_profit_factor' in criteria:
            del criteria['min_profit_factor']

    test_windows = wfe._generate_windows(df)
    window_regimes = {i: classify_regime(df, w['test_start'], w['test_end'])
                      for i, w in enumerate(test_windows)}

    results = {'TIER_A': [], 'TIER_B': [], 'DROP': []}
    condition_freq_tierab = Counter()  # condition frequency in Tier A/B signals
    condition_freq_all = Counter()     # condition frequency in all signals

    for idx, cand in enumerate(alpha_candidates):
        signal_id = cand['signal_id']
        rules = cand['rules']
        full_result = cand['full_result']
        backtest_fn = make_generic_backtest_fn(rules)

        # Track conditions for frequency analysis
        conditions = extract_conditions(rules)
        for c in conditions:
            condition_freq_all[c] += 1

        try:
            wf = wfe.run(signal_id, backtest_fn, df, regime_labels, rules, 'FUTURES')
        except Exception as e:
            results['DROP'].append({
                'signal_id': signal_id,
                'tier': 'DROP',
                'reason': f'ERROR: {str(e)[:100]}',
            })
            continue

        for wd in wf['window_details']:
            wd['regime'] = window_regimes.get(wd['window_index'], 'UNKNOWN')

        tier, reason = tier_signal(wf['window_details'])
        profile = regime_profile(wf['window_details'])
        last4 = wf['window_details'][-4:]
        last4_passed = sum(1 for w in last4 if w['passed'])
        active_dds = [w['result'].max_drawdown for w in wf['window_details']
                      if w['result'].trade_count > 0]

        entry = {
            'signal_id': signal_id,
            'tier': tier,
            'reason': reason,
            'nifty_correlation': full_result.nifty_correlation,
            'full_sharpe': full_result.sharpe,
            'full_trades': full_result.trade_count,
            'full_win_rate': full_result.win_rate,
            'full_max_dd': full_result.max_drawdown,
            'full_annual_return': full_result.annual_return,
            'windows_passed': wf['windows_passed'],
            'total_windows': wf['total_windows'],
            'pass_rate': wf['pass_rate'],
            'last4_passed': last4_passed,
            'last4_rate': last4_passed / 4,
            'aggregate_sharpe': wf['aggregate_sharpe'],
            'worst_dd': round(max(active_dds), 3) if active_dds else 1.0,
            'regime_profile': profile,
            'rules': rules,
            'per_window': [{
                'idx': w['window_index'],
                'regime': w.get('regime', '?'),
                'sharpe': w['result'].sharpe,
                'max_dd': w['result'].max_drawdown,
                'win_rate': w['result'].win_rate,
                'trades': w['result'].trade_count,
                'passed': w['passed'],
            } for w in wf['window_details']],
        }
        results[tier].append(entry)

        # Track conditions for Tier A/B signals
        if tier in ('TIER_A', 'TIER_B'):
            for c in conditions:
                condition_freq_tierab[c] += 1

        done = idx + 1
        if done % 10 == 0 or done == len(alpha_candidates):
            print(f"  [{done}/{len(alpha_candidates)}] "
                  f"A:{len(results['TIER_A'])} B:{len(results['TIER_B'])} "
                  f"Drop:{len(results['DROP'])}", flush=True)

    # ================================================================
    # RESULTS
    # ================================================================
    for tier in ['TIER_A', 'TIER_B']:
        results[tier].sort(key=lambda x: x['aggregate_sharpe'], reverse=True)

    print(f"\n{'='*90}")
    print(f"ROUND 2 WALK-FORWARD RESULTS")
    print(f"{'='*90}")
    print(f"  Input DSL signals:     {len(signals)}")
    print(f"  Alpha candidates:      {len(alpha_candidates)} (|corr| <= {MAX_NIFTY_CORR})")
    print(f"  TIER A:                {len(results['TIER_A'])}")
    print(f"  TIER B:                {len(results['TIER_B'])}")
    print(f"  DROPPED:               {len(results['DROP'])}")

    for tier_name in ['TIER_A', 'TIER_B']:
        if results[tier_name]:
            print(f"\n{'='*90}")
            print(f"{tier_name} SIGNALS")
            print(f"{'='*90}")
            print(f"{'Signal':<25} {'AggSh':>6} {'NCorr':>6} {'Pass':>6} {'L4':>4} "
                  f"{'WDD':>5} {'Trades':>6} {'WR':>5} {'AnnRet':>7}")
            print(f"{'-'*25} {'-'*6} {'-'*6} {'-'*6} {'-'*4} "
                  f"{'-'*5} {'-'*6} {'-'*5} {'-'*7}")
            for s in results[tier_name]:
                regime_str = ' | '.join(
                    f"{r}:{p['passed']}/{p['windows']}"
                    for r, p in sorted(s['regime_profile'].items())
                    if p['windows'] > 0
                )
                print(f"{s['signal_id']:<25} {s['aggregate_sharpe']:>6.2f} "
                      f"{s['nifty_correlation']:>6.3f} {s['pass_rate']:>5.0%} "
                      f"{s['last4_passed']:>3}/4 {s['worst_dd']:>4.1%} "
                      f"{s['full_trades']:>5} {s['full_win_rate']:>5.1%} "
                      f"{s['full_annual_return']:>6.2%}")
                print(f"  {'Regimes:':>25} {regime_str}")

    # ================================================================
    # CONDITION FREQUENCY ANALYSIS
    # ================================================================
    if condition_freq_tierab:
        print(f"\n{'='*90}")
        print(f"CONDITION FREQUENCY IN TIER A/B SIGNALS")
        print(f"{'='*90}")
        total_tierab = len(results['TIER_A']) + len(results['TIER_B'])
        for cond, count in condition_freq_tierab.most_common(20):
            pct = count / total_tierab * 100
            print(f"  {count:3d}/{total_tierab} ({pct:4.0f}%)  {cond}")

    # Also show conditions common in all alpha candidates for comparison
    if condition_freq_all:
        print(f"\n{'='*90}")
        print(f"CONDITION FREQUENCY IN ALL ALPHA CANDIDATES (top 15)")
        print(f"{'='*90}")
        total_all = len(alpha_candidates)
        for cond, count in condition_freq_all.most_common(15):
            pct = count / total_all * 100
            # Mark if also in tier A/B
            in_tierab = condition_freq_tierab.get(cond, 0)
            marker = f"  ← {in_tierab} in Tier A/B" if in_tierab > 0 else ""
            print(f"  {count:3d}/{total_all} ({pct:4.0f}%)  {cond}{marker}")

    # ================================================================
    # NEAR-MISSES (close to Tier B)
    # ================================================================
    near_misses = []
    for s in results['DROP']:
        if (s.get('pass_rate', 0) >= 0.30 and
                s.get('aggregate_sharpe', 0) >= 0.5):
            near_misses.append(s)
    near_misses.sort(key=lambda x: x.get('aggregate_sharpe', 0), reverse=True)

    if near_misses:
        print(f"\n{'='*90}")
        print(f"NEAR-MISSES (pass_rate >= 30%, sharpe >= 0.5) — {len(near_misses)} signals")
        print(f"{'='*90}")
        print(f"{'Signal':<25} {'AggSh':>6} {'NCorr':>6} {'Pass':>6} {'L4':>4} {'Reason'}")
        print(f"{'-'*25} {'-'*6} {'-'*6} {'-'*6} {'-'*4} {'-'*30}")
        for s in near_misses[:15]:
            print(f"{s['signal_id']:<25} {s.get('aggregate_sharpe',0):>6.2f} "
                  f"{s.get('nifty_correlation',0):>6.3f} "
                  f"{s.get('pass_rate',0):>5.0%} {s.get('last4_passed',0):>3}/4 "
                  f"{s.get('reason','')}")

    # ================================================================
    # SAVE
    # ================================================================
    output = {
        'filters': {
            'max_nifty_correlation': MAX_NIFTY_CORR,
            'min_trades_per_window': MIN_TRADES_PER_WINDOW,
        },
        'summary': {
            'input_dsl_signals': len(signals),
            'alpha_candidates': len(alpha_candidates),
            'rejected_corr': rejected_corr,
            'rejected_no_trades': rejected_no_trades,
            'rejected_error': rejected_error,
            'tier_a': len(results['TIER_A']),
            'tier_b': len(results['TIER_B']),
            'dropped': len(results['DROP']),
        },
        'tier_a': results['TIER_A'],
        'tier_b': results['TIER_B'],
        'dropped': results['DROP'],
        'condition_frequency_tierab': dict(condition_freq_tierab.most_common(30)),
        'condition_frequency_all': dict(condition_freq_all.most_common(30)),
        'near_misses': near_misses[:20],
    }

    output_path = os.path.join(RESULTS_DIR, 'round2_results.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n  Results saved to: {output_path}")

    print(f"\n{'='*90}")
    print(f"ROUND 2 COMPLETE")
    print(f"{'='*90}")
    print(f"  TIER A: {len(results['TIER_A'])} signals")
    print(f"  TIER B: {len(results['TIER_B'])} signals")
    print(f"  Near-misses: {len(near_misses)} signals")
    print(f"  Paper trading candidates: {len(results['TIER_A'])} (Tier A)")


if __name__ == '__main__':
    main()
