"""
Walk-forward the full DSL PASS pool.

Reads all entry/exit DSL PASS files from dsl_results/PASS/,
runs each through the tiered walk-forward engine, and saves results.

Skips POSITION_SIZING files (sizing rules, not entry/exit signals).

Usage:
    python run_wf_dsl_pool.py                    # run all
    python run_wf_dsl_pool.py --limit 50         # test first 50
    python run_wf_dsl_pool.py --books ARONSON    # specific book
    python run_wf_dsl_pool.py --resume           # skip already tested
"""

import argparse
import json
import os
import time
from collections import Counter

import numpy as np
import pandas as pd
import psycopg2

from backtest.generic_backtest import run_generic_backtest, make_generic_backtest_fn
from backtest.walk_forward import WalkForwardEngine
from backtest.types import harmonic_mean_sharpe
from config.settings import DATABASE_DSN
from run_wf_tiered import classify_regime, evaluate_window, tier_signal, regime_profile

DSL_PASS_DIR = 'dsl_results/PASS'
RESULTS_DIR = 'backtest_results'
OUTPUT_PATH = os.path.join(RESULTS_DIR, '_WF_DSL_POOL.json')  # overridden in main() for wide windows


def load_dsl_pass_signals(books=None, limit=0):
    """Load all entry/exit DSL PASS files. Skip POSITION_SIZING."""
    signals = {}

    for fname in sorted(os.listdir(DSL_PASS_DIR)):
        if not fname.endswith('.json'):
            continue

        path = os.path.join(DSL_PASS_DIR, fname)
        with open(path) as f:
            data = json.load(f)

        # Skip POSITION_SIZING files
        if data.get('signal_type') == 'POSITION_SIZING':
            continue

        # Get backtest rules
        rules = data.get('backtest_rule')
        if not rules or not rules.get('backtestable'):
            continue

        sid = data.get('signal_id', fname.replace('.json', ''))
        book_id = data.get('book_id', sid.split('_DRY_')[0] if '_DRY_' in sid else sid.split('_')[0])

        # Book filter
        if books and book_id not in books:
            continue

        # Keep best variant per signal_id (most conditions = most specific)
        n_conds = sum(len(rules.get(k, [])) for k in
                      ['entry_long', 'entry_short', 'exit_long', 'exit_short'])

        if sid not in signals or n_conds > signals[sid]['_n_conds']:
            signals[sid] = {
                'signal_id': sid,
                'book_id': book_id,
                'rules': rules,
                '_n_conds': n_conds,
            }

    result = list(signals.values())
    if limit > 0:
        result = result[:limit]
    return result


def main():
    parser = argparse.ArgumentParser(description='Walk-forward DSL PASS pool')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--books', nargs='+', default=None)
    parser.add_argument('--resume', action='store_true',
                        help='Skip signals already in results file')
    parser.add_argument('--wide-windows', action='store_true',
                        help='Use 24-month test windows for low-frequency signals')
    parser.add_argument('--zero-trade-only', action='store_true',
                        help='Only test signals that had zero trades in previous run')
    args = parser.parse_args()

    book_set = set(b.upper() for b in args.books) if args.books else None

    # Load signals
    signals = load_dsl_pass_signals(books=book_set, limit=args.limit)

    # Filter to zero-trade signals if requested
    if args.zero_trade_only:
        zt_path = os.path.join(RESULTS_DIR, '_ZERO_TRADE_IDS.json')
        if os.path.exists(zt_path):
            with open(zt_path) as f:
                zero_ids = set(json.load(f))
            signals = [s for s in signals if s['signal_id'] in zero_ids]
            print(f"Filtered to {len(signals)} zero-trade signals")
    print(f"Loaded {len(signals)} entry/exit DSL PASS signals")

    # Use separate output file for wide-window runs
    global OUTPUT_PATH
    if args.wide_windows:
        OUTPUT_PATH = os.path.join(RESULTS_DIR, '_WF_DSL_POOL_WIDE.json')

    if not signals:
        print("No signals to test. Check dsl_results/PASS/ directory.")
        return

    # Book breakdown
    book_counts = Counter(s['book_id'] for s in signals)
    for book, count in book_counts.most_common():
        print(f"  {book:<18s} {count:>4d}")

    # Resume: skip already tested
    already_tested = set()
    if args.resume and os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH) as f:
            prev = json.load(f)
        for tier in ['tier_a', 'tier_b', 'dropped']:
            for entry in prev.get(tier, []):
                already_tested.add(entry['signal_id'])
        signals = [s for s in signals if s['signal_id'] not in already_tested]
        print(f"Resume: skipping {len(already_tested)} already tested, {len(signals)} remaining")

    # Load market data
    print("\nLoading market data...", flush=True)
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
    print(f"  {len(df)} trading days, {len(cal)} calendar days", flush=True)

    # Regime labels
    from regime_labeler import RegimeLabeler
    labeler = RegimeLabeler()
    regime_labels = labeler.label_full_history(df)
    print(f"  {len(regime_labels)} regime labels", flush=True)

    # Pre-filter: quick backtest + fire-rate validation + auto-relax
    from backtest.fire_rate_validator import auto_relax
    print("\nPre-filtering signals (backtest + fire-rate + auto-relax)...", flush=True)
    viable = []
    skipped = 0
    relaxed_count = 0
    for s in signals:
        rules = s['rules']
        try:
            result = run_generic_backtest(rules, df, {})
            if result.trade_count >= 15:
                viable.append(s)
                continue

            # Try auto-relaxation for zero/low-trade signals
            relaxed_rules = auto_relax(rules, df)
            if relaxed_rules:
                result2 = run_generic_backtest(relaxed_rules, df, {})
                if result2.trade_count >= 15:
                    s['rules'] = relaxed_rules
                    s['relaxed'] = True
                    viable.append(s)
                    relaxed_count += 1
                    continue

            skipped += 1
        except Exception:
            skipped += 1

    print(f"  {len(viable)} viable (>=15 trades), {skipped} skipped")
    print(f"  {relaxed_count} rescued via auto-relaxation")

    # Walk-forward engine
    wfe = WalkForwardEngine(cal)

    # Wide windows: 24-month test (vs default 12) for low-frequency signals
    if args.wide_windows:
        wfe.TEST_MONTHS = 24
        wfe.STEP_MONTHS = 6
        print(f"  Wide windows: {wfe.TRAIN_MONTHS}mo train / {wfe.TEST_MONTHS}mo test / {wfe.STEP_MONTHS}mo step")

    for criteria in wfe.CRITERIA.values():
        criteria['min_trades'] = 5 if args.wide_windows else 10
        criteria['min_sharpe'] = 0.8
        if 'min_calmar' in criteria:
            del criteria['min_calmar']
        if 'min_profit_factor' in criteria:
            del criteria['min_profit_factor']

    # Pre-compute window regime tags
    print("\nPre-computing window regimes...", flush=True)
    test_windows = wfe._generate_windows(df)
    window_regimes = {}
    for i, w in enumerate(test_windows):
        window_regimes[i] = classify_regime(df, w['test_start'], w['test_end'])
    regime_counts = Counter(window_regimes.values())
    print(f"  Window regimes: {dict(regime_counts)}")

    # Run walk-forward
    print(f"\n{'='*90}")
    print(f"WALK-FORWARD ON {len(viable)} DSL PASS SIGNALS")
    print(f"{'='*90}\n")

    results = {'TIER_A': [], 'TIER_B': [], 'DROP': []}
    start = time.time()

    for idx, s in enumerate(viable):
        signal_id = s['signal_id']
        rules = s['rules']
        backtest_fn = make_generic_backtest_fn(rules)

        try:
            wf = wfe.run(signal_id, backtest_fn, df, regime_labels, rules, 'FUTURES')
        except Exception as e:
            results['DROP'].append({
                'signal_id': signal_id,
                'book_id': s['book_id'],
                'tier': 'DROP',
                'reason': f'ERROR: {str(e)[:100]}',
            })
            continue

        # Tag windows with regime
        for wd in wf['window_details']:
            wd['regime'] = window_regimes.get(wd['window_index'], 'UNKNOWN')

        tier, reason = tier_signal(wf['window_details'])
        profile = regime_profile(wf['window_details'])

        last4 = wf['window_details'][-4:]
        last4_passed = sum(1 for w in last4 if w['passed'])

        active_windows = [w for w in wf['window_details'] if w['result'].trade_count > 0]
        active_sharpes = [w['result'].sharpe for w in active_windows]
        active_dds = [w['result'].max_drawdown for w in active_windows]

        entry = {
            'signal_id': signal_id,
            'book_id': s['book_id'],
            'tier': tier,
            'reason': reason,
            'windows_passed': wf['windows_passed'],
            'total_windows': wf['total_windows'],
            'pass_rate': wf['pass_rate'],
            'last4_passed': last4_passed,
            'last4_rate': last4_passed / 4,
            'aggregate_sharpe': wf['aggregate_sharpe'],
            'active_windows': len(active_windows),
            'mean_sharpe': round(np.mean(active_sharpes), 2) if active_sharpes else 0,
            'median_sharpe': round(np.median(active_sharpes), 2) if active_sharpes else 0,
            'worst_dd': round(max(active_dds), 3) if active_dds else 1.0,
            'regime_profile': profile,
            'rules': rules,
            'per_window': [{
                'idx': w['window_index'],
                'regime': w.get('regime', '?'),
                'test_start': str(w['window']['test_start'])[:10],
                'test_end': str(w['window']['test_end'])[:10],
                'sharpe': w['result'].sharpe,
                'max_dd': w['result'].max_drawdown,
                'win_rate': w['result'].win_rate,
                'trades': w['result'].trade_count,
                'passed': w['passed'],
            } for w in wf['window_details']],
        }

        results[tier].append(entry)

        done = idx + 1
        elapsed = time.time() - start
        if done % 10 == 0 or done == len(viable):
            print(f"  [{done}/{len(viable)}] ({elapsed:.0f}s) "
                  f"A:{len(results['TIER_A'])} B:{len(results['TIER_B'])} "
                  f"Drop:{len(results['DROP'])}", flush=True)

    # Merge with previous results if resuming
    if args.resume and os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH) as f:
            prev = json.load(f)
        for tier_key, out_key in [('TIER_A', 'tier_a'), ('TIER_B', 'tier_b'), ('DROP', 'dropped')]:
            for entry in prev.get(out_key, []):
                if entry['signal_id'] not in {e['signal_id'] for e in results[tier_key]}:
                    results[tier_key].append(entry)

    # Sort by Sharpe
    for tier in ['TIER_A', 'TIER_B']:
        results[tier].sort(key=lambda x: x.get('aggregate_sharpe', 0), reverse=True)

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    total_tested = len(viable) + len(already_tested)
    output = {
        'summary': {
            'total_tested': total_tested,
            'tier_a': len(results['TIER_A']),
            'tier_b': len(results['TIER_B']),
            'dropped': len(results['DROP']),
            'window_regimes': dict(regime_counts),
        },
        'tier_a': results['TIER_A'],
        'tier_b': results['TIER_B'],
        'dropped': results['DROP'],
    }

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    # Report
    print(f"\n{'='*90}")
    print(f"WALK-FORWARD DSL POOL RESULTS")
    print(f"{'='*90}")
    print(f"  Total tested:  {total_tested}")
    print(f"  TIER A:        {len(results['TIER_A'])} (strong)")
    print(f"  TIER B:        {len(results['TIER_B'])} (regime-dependent)")
    print(f"  DROPPED:       {len(results['DROP'])}")

    for tier_name, tier_list in [('TIER A', results['TIER_A']), ('TIER B', results['TIER_B'])]:
        if tier_list:
            print(f"\n{'='*90}")
            print(f"{tier_name} SIGNALS")
            print(f"{'='*90}")
            print(f"{'Signal':<30} {'Book':<15} {'Sharpe':>7} {'Pass':>6} {'L4':>4} "
                  f"{'WrstDD':>7}")
            print(f"{'-'*30} {'-'*15} {'-'*7} {'-'*6} {'-'*4} {'-'*7}")
            for s in tier_list:
                print(f"{s['signal_id']:<30} {s.get('book_id','?'):<15} "
                      f"{s['aggregate_sharpe']:>7.2f} {s['pass_rate']:>5.0%} "
                      f"{s['last4_passed']:>3}/4 {s['worst_dd']:>6.1%}")

    # Drop reasons
    drop_reasons = Counter()
    for d in results['DROP']:
        r = d.get('reason', 'unknown')
        if r:
            drop_reasons[r.split(';')[0].strip()] += 1
    print(f"\n{'='*90}")
    print(f"DROP REASONS")
    print(f"{'='*90}")
    for reason, count in drop_reasons.most_common(15):
        print(f"  {count:>3}  {reason}")

    # Per-book summary
    book_results = {}
    for tier, entries in results.items():
        for e in entries:
            bid = e.get('book_id', '?')
            if bid not in book_results:
                book_results[bid] = {'A': 0, 'B': 0, 'DROP': 0}
            if tier == 'TIER_A':
                book_results[bid]['A'] += 1
            elif tier == 'TIER_B':
                book_results[bid]['B'] += 1
            else:
                book_results[bid]['DROP'] += 1

    print(f"\n{'='*90}")
    print(f"PER-BOOK RESULTS")
    print(f"{'='*90}")
    print(f"{'Book':<18} {'Tier A':>6} {'Tier B':>6} {'Drop':>6}")
    print(f"{'-'*18} {'-'*6} {'-'*6} {'-'*6}")
    for bid in sorted(book_results.keys()):
        br = book_results[bid]
        print(f"{bid:<18} {br['A']:>6} {br['B']:>6} {br['DROP']:>6}")

    print(f"\n  Results saved to: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
