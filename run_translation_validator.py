"""
Path B+ Steps 1 & 2: Translation validator + Nifty correlation filter.

Step 1: Validate all 615 translated signals for broken translations.
        Buckets: PASS, FIXABLE (syntax-only errors), WARN, REVIEW, FAIL
Step 2: Cross-reference PASS signals against alpha run (which already
        computed Nifty correlation correctly via full-dataset backtest).

Outputs:
  - validation_results/validator_report.json    — full per-signal results
  - validation_results/validator_summary.json   — aggregate stats + signal buckets
  - validation_results/step2_survivors.json     — signals passing both filters
"""

import json
import os
import re
from collections import Counter

from backtest.translation_validator import validate_all

TRANSLATED_PATH = 'extraction_results/translated_signals.json'
ALPHA_RESULTS_PATH = 'backtest_results/_WF_ALPHA.json'
OUTPUT_DIR = 'validation_results'
NIFTY_CORR_THRESHOLD = 0.5


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ================================================================
    # STEP 1: Translation Validator
    # ================================================================
    print("=" * 70)
    print("STEP 1: TRANSLATION VALIDATOR")
    print("=" * 70)

    print("Loading translated signals...")
    with open(TRANSLATED_PATH) as f:
        signals = json.load(f)

    total = len(signals)
    backtestable = [s for s in signals if s.get('backtestable') and s.get('rules')]
    print(f"Total signals: {total}")
    print(f"Backtestable: {len(backtestable)}")
    print()

    print("Running translation validator...")
    report = validate_all(signals)

    # Console summary
    print()
    print(f"Total backtestable signals: {report['total_backtestable']}")
    print(f"Signals with issues:        {report['signals_with_issues']} "
          f"({report['signals_with_issues'] / report['total_backtestable'] * 100:.1f}%)")
    print(f"Signals clean (PASS):       {report['signals_clean']} "
          f"({report['signals_clean'] / report['total_backtestable'] * 100:.1f}%)")
    print()

    print("Issue type breakdown:")
    for issue_type, count in sorted(report['issue_counts'].items(),
                                     key=lambda x: -x[1]):
        print(f"  {issue_type:25s} {count:4d}")
    print()

    # Verdict distribution
    verdicts = Counter(r['verdict'] for r in report['results'])
    print("Verdict distribution:")
    for verdict in ['PASS', 'FIXABLE', 'REVIEW', 'WARN', 'FAIL']:
        count = verdicts.get(verdict, 0)
        pct = count / report['total_backtestable'] * 100 if report['total_backtestable'] else 0
        print(f"  {verdict:8s} {count:4d}  ({pct:.1f}%)")
    print()

    # Sample issues
    print("-" * 70)
    print("SAMPLE ISSUES (first 3 per type):")
    print("-" * 70)
    shown = {}
    for r in report['results']:
        for issue in r['issues']:
            itype = issue['type']
            if itype not in shown:
                shown[itype] = []
            if len(shown[itype]) < 3:
                shown[itype].append(f"  {r['signal_id']}: {issue['detail']}")
    for itype, examples in sorted(shown.items()):
        print(f"\n{itype}:")
        for ex in examples:
            print(ex)
    print()

    # Save full report
    report_path = os.path.join(OUTPUT_DIR, 'validator_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Full report saved to: {report_path}")

    # Build buckets (deduplicate to unique signal_ids)
    pass_ids = list(dict.fromkeys(
        r['signal_id'] for r in report['results'] if r['verdict'] == 'PASS'))
    fixable_ids = list(dict.fromkeys(
        r['signal_id'] for r in report['results'] if r['verdict'] == 'FIXABLE'))
    warn_ids = list(dict.fromkeys(
        r['signal_id'] for r in report['results'] if r['verdict'] == 'WARN'))
    review_ids = list(dict.fromkeys(
        r['signal_id'] for r in report['results'] if r['verdict'] == 'REVIEW'))
    fail_ids = list(dict.fromkeys(
        r['signal_id'] for r in report['results'] if r['verdict'] == 'FAIL'))

    # Count variants per verdict
    pass_variants = sum(1 for r in report['results'] if r['verdict'] == 'PASS')
    fixable_variants = sum(1 for r in report['results'] if r['verdict'] == 'FIXABLE')

    print(f"\nPASS:    {len(pass_ids)} unique signal IDs ({pass_variants} variants)")
    print(f"FIXABLE: {len(fixable_ids)} unique signal IDs ({fixable_variants} variants)")

    # Extract unknown column patterns in FIXABLE signals
    unknown_columns = set()
    for r in report['results']:
        if r['verdict'] == 'FIXABLE':
            for issue in r['issues']:
                if issue['type'] == 'UNKNOWN_INDICATOR':
                    for match in re.findall(r"unknown (?:indicator|value column) '([^']+)'",
                                            issue['detail']):
                        unknown_columns.add(match)

    # Categorize unknown columns
    lagged_refs = sorted(c for c in unknown_columns if re.match(r'\w+\[\d+\]', c))
    concept_refs = sorted(unknown_columns - set(lagged_refs))

    print(f"\nFIXABLE unknown column analysis:")
    print(f"  Lagged references (indicator[N] syntax): {len(lagged_refs)}")
    if lagged_refs[:10]:
        for c in lagged_refs[:10]:
            print(f"    {c}")
    print(f"  Other unknown refs: {len(concept_refs)}")
    if concept_refs[:10]:
        for c in concept_refs[:10]:
            print(f"    {c}")

    # ================================================================
    # STEP 2: CORRELATION FILTER (using alpha run results)
    # ================================================================
    print()
    print("=" * 70)
    print("STEP 2: NIFTY CORRELATION FILTER")
    print("=" * 70)

    # Load alpha run results — these have correctly computed Nifty correlation
    # from full-dataset backtest via run_generic_backtest()
    print(f"Loading alpha results from {ALPHA_RESULTS_PATH}...")
    with open(ALPHA_RESULTS_PATH) as f:
        alpha = json.load(f)

    # Build lookup: signal_id -> alpha run data
    alpha_lookup = {}
    for entry in alpha.get('tier_a', []) + alpha.get('tier_b', []) + alpha.get('dropped', []):
        sid = entry['signal_id']
        alpha_lookup[sid] = {
            'nifty_corr': entry.get('nifty_correlation', 0),
            'sharpe': entry.get('full_sharpe', 0),
            'trades': entry.get('full_trades', 0),
            'win_rate': entry.get('full_win_rate', 0),
            'max_dd': entry.get('full_max_dd', 0),
            'annual_return': entry.get('full_annual_return', 0),
            'tier': entry.get('tier', 'DROP'),
            'pass_rate': entry.get('pass_rate', 0),
            'aggregate_sharpe': entry.get('aggregate_sharpe', 0),
            'windows_passed': entry.get('windows_passed', 0),
            'total_windows': entry.get('total_windows', 0),
        }

    print(f"Alpha run: {len(alpha_lookup)} unique signal IDs")
    print(f"PASS signal IDs to cross-reference: {len(pass_ids)}")
    print()

    # Cross-reference PASS signals with alpha run
    survivors = []          # PASS + in alpha + |corr| <= threshold
    no_alpha_data = []      # PASS but not in alpha run (need backtest)
    high_corr = []          # PASS + |corr| > threshold

    for sid in pass_ids:
        a = alpha_lookup.get(sid)
        if a is None:
            no_alpha_data.append(sid)
            continue

        corr = a['nifty_corr']
        if abs(corr) > NIFTY_CORR_THRESHOLD:
            high_corr.append({'signal_id': sid, **a})
        else:
            survivors.append({'signal_id': sid, **a})

    # Sort survivors by full-dataset Sharpe
    survivors.sort(key=lambda x: x['sharpe'], reverse=True)

    print(f"Step 2 results (PASS signals only):")
    print(f"  In alpha run:            {len(pass_ids) - len(no_alpha_data)}")
    print(f"  Not in alpha run:        {len(no_alpha_data)}")
    print(f"  Rejected (|corr| > 0.5): {len(high_corr)}")
    print(f"  SURVIVORS:               {len(survivors)}")
    print()

    if no_alpha_data:
        print(f"  Signals not in alpha run (need fresh backtest):")
        for sid in no_alpha_data:
            print(f"    {sid}")
        print()

    # Survivor table
    if survivors:
        print("-" * 100)
        print("STEP 2 SURVIVORS — sorted by Sharpe (full dataset)")
        print("-" * 100)
        print(f"{'Signal ID':30s} {'Sharpe':>7s} {'Corr':>7s} {'WR':>6s} {'Trades':>6s} "
              f"{'MaxDD':>6s} {'WF Tier':>8s} {'WF Pass':>8s} {'WF Sharpe':>10s}")
        print("-" * 100)
        for s in survivors:
            wf_pass = f"{s['windows_passed']}/{s['total_windows']}" if s['total_windows'] else "n/a"
            print(f"{s['signal_id']:30s} {s['sharpe']:7.3f} {s['nifty_corr']:7.3f} "
                  f"{s['win_rate']:6.1%} {s['trades']:6d} {s['max_dd']:6.1%} "
                  f"{s['tier']:>8s} {wf_pass:>8s} {s['aggregate_sharpe']:10.3f}")
    print()

    # Also check FIXABLE signals in alpha
    fixable_in_alpha = []
    for sid in fixable_ids:
        a = alpha_lookup.get(sid)
        if a and abs(a['nifty_corr']) <= NIFTY_CORR_THRESHOLD:
            fixable_in_alpha.append({'signal_id': sid, **a})
    fixable_in_alpha.sort(key=lambda x: x['sharpe'], reverse=True)

    print(f"FIXABLE bucket alpha cross-reference:")
    print(f"  Total FIXABLE:            {len(fixable_ids)}")
    print(f"  In alpha + low corr:      {len(fixable_in_alpha)}")
    if fixable_in_alpha:
        print(f"\n  Top FIXABLE signals (worth adding lagged indicator support for):")
        for s in fixable_in_alpha[:10]:
            wf_pass = f"{s['windows_passed']}/{s['total_windows']}" if s['total_windows'] else "n/a"
            print(f"    {s['signal_id']:30s} sharpe={s['sharpe']:7.3f} corr={s['nifty_corr']:7.3f} "
                  f"trades={s['trades']:4d} tier={s['tier']} wf={wf_pass}")
    print()

    # ================================================================
    # SAVE RESULTS
    # ================================================================

    step2_output = {
        'nifty_corr_threshold': NIFTY_CORR_THRESHOLD,
        'pass_signal_ids': len(pass_ids),
        'in_alpha_run': len(pass_ids) - len(no_alpha_data),
        'not_in_alpha_run': no_alpha_data,
        'rejected_high_corr': len(high_corr),
        'survivor_count': len(survivors),
        'survivors': survivors,
        'rejected': high_corr,
        'fixable_in_alpha': fixable_in_alpha,
    }
    step2_path = os.path.join(OUTPUT_DIR, 'step2_survivors.json')
    with open(step2_path, 'w') as f:
        json.dump(step2_output, f, indent=2)
    print(f"Step 2 results saved to: {step2_path}")

    summary = {
        'total_backtestable': report['total_backtestable'],
        'step1_verdicts': dict(verdicts),
        'step1_unique_ids': {
            'pass': len(pass_ids),
            'fixable': len(fixable_ids),
            'warn': len(warn_ids),
            'review': len(review_ids),
            'fail': len(fail_ids),
        },
        'step2_survivors': len(survivors),
        'step2_rejected': len(high_corr),
        'step2_no_alpha_data': no_alpha_data,
        'issue_counts': report['issue_counts'],
        'severity_counts': report['severity_counts'],
        'buckets': {
            'pass_ids': pass_ids,
            'fixable_ids': fixable_ids,
            'warn_ids': warn_ids,
            'review_ids': review_ids,
            'fail_ids': fail_ids,
            'survivor_ids': [s['signal_id'] for s in survivors],
        },
        'fixable_unknown_columns': {
            'lagged_refs': lagged_refs,
            'concept_refs': concept_refs,
        },
    }
    summary_path = os.path.join(OUTPUT_DIR, 'validator_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")

    # ================================================================
    # PIPELINE STATUS
    # ================================================================
    print()
    print("=" * 70)
    print("PATH B+ PIPELINE STATUS")
    print("=" * 70)
    print()
    print(f"  615 backtestable signals")
    print(f"   ├─ PASS:    {len(pass_ids):3d} unique IDs  (valid translation)")
    print(f"   ├─ FIXABLE: {len(fixable_ids):3d} unique IDs  (syntax-only errors, salvageable)")
    print(f"   ├─ REVIEW:  {len(review_ids):3d} unique IDs  (medium-severity issues)")
    print(f"   ├─ WARN:    {len(warn_ids):3d} unique IDs  (high-severity issues)")
    print(f"   └─ FAIL:    {len(fail_ids):3d} unique IDs  (broken logic)")
    print()
    print(f"  Step 1 DONE — Translation validator")
    print(f"  Step 2 DONE — Correlation filter: {len(survivors)} survivors")
    print(f"  -------")
    print(f"  Step 3 TODO — Manual inspection of {len(survivors)} survivors")
    if no_alpha_data:
        print(f"               + backtest {len(no_alpha_data)} PASS signals missing alpha data")
    print(f"  Step 4 TODO — Hand-retranslate top 20-30 signals")
    print(f"               + FIXABLE bucket: {len(fixable_ids)} signals ({len(fixable_in_alpha)} already have alpha data)")
    print(f"               + Lagged indicator support unlocks {len(lagged_refs)} column references")
    print(f"  Step 5 TODO — Backtest hand-written rules")
    print("=" * 70)


if __name__ == '__main__':
    main()
