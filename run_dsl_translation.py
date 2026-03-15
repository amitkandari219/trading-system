"""
Run DSL-based translation on all signals.

Usage:
  python run_dsl_translation.py                       # full run on all 3,183 signals
  python run_dsl_translation.py --dry-run --limit 10  # test run on 10 signals
  python run_dsl_translation.py --resume              # skip already-translated signals
  python run_dsl_translation.py --force-retranslate   # ignore existing, retranslate all
  python run_dsl_translation.py --consolidate         # pick best variant per signal_id
  python run_dsl_translation.py --retranslate-fixable # retranslate FIXABLE signals only

Outputs to dsl_results/:
  PASS/{signal_id}_v{N}.json           — valid DSL translations
  UNTRANSLATABLE/{signal_id}_v{N}.json — signals Haiku can't express in DSL
  FAIL/{signal_id}_v{N}.json           — validation failures
  BEST/{signal_id}.json                — best variant per signal_id (after --consolidate)
  summary.json                         — aggregate stats
"""

import argparse
import glob
import json
import os
import time
from collections import Counter, defaultdict

from extraction.dsl_schema import DSLSignalRule
from extraction.dsl_translator import DSLTranslator, NON_TRADEABLE_CATEGORIES
from extraction.dsl_validator import DSLValidator
from extraction.dsl_to_backtest import DSLToBacktest

ALL_SIGNALS_PATH = 'extraction_results/approved/_ALL.json'
TRANSLATED_PATH = 'extraction_results/translated_signals.json'
VALIDATOR_REPORT_PATH = 'validation_results/validator_report.json'
OUTPUT_DIR = 'dsl_results'


def main():
    parser = argparse.ArgumentParser(description='Run DSL translation')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print output without saving')
    parser.add_argument('--limit', type=int, default=0,
                        help='Limit number of signals to process')
    parser.add_argument('--resume', action='store_true',
                        help='Skip signals that already have translation files')
    parser.add_argument('--force-retranslate', action='store_true',
                        help='Ignore existing files, retranslate everything')
    parser.add_argument('--consolidate', action='store_true',
                        help='Pick best variant per signal_id into BEST/')
    parser.add_argument('--retranslate-fixable', action='store_true',
                        help='Retranslate only FIXABLE signals from previous run')
    parser.add_argument('--workers', type=int, default=6,
                        help='Number of parallel workers')
    args = parser.parse_args()

    if args.consolidate:
        run_consolidate()
        return

    translator = DSLTranslator(num_workers=args.workers)
    validator = DSLValidator()
    compiler = DSLToBacktest()

    if args.retranslate_fixable:
        run_retranslate_fixable(translator, validator, compiler, args)
    else:
        run_full_translation(translator, validator, compiler, args)


def run_full_translation(translator, validator, compiler, args):
    """Translate all signals from the approved pool."""

    # Load all approved signals
    print("Loading signals...")
    with open(ALL_SIGNALS_PATH) as f:
        all_signals = json.load(f)
    print(f"Total approved signals: {len(all_signals)}")

    if args.limit > 0:
        all_signals = all_signals[:args.limit]
        print(f"Limited to: {args.limit}")

    # Create output directories
    if not args.dry_run:
        for subdir in ['PASS', 'UNTRANSLATABLE', 'FAIL']:
            os.makedirs(os.path.join(OUTPUT_DIR, subdir), exist_ok=True)

    # Process signals
    results = {'PASS': [], 'UNTRANSLATABLE': [], 'FAIL': []}
    book_stats = defaultdict(lambda: {'total': 0, 'pass': 0, 'untranslatable': 0, 'fail': 0})
    untranslatable_reasons = Counter()
    variant_counter = Counter()  # signal_id -> count, for unique filenames

    total = len(all_signals)
    start = time.time()
    skipped = 0

    # Build set of already-translated signal_ids for resume
    already_done = set()
    if args.resume and not args.force_retranslate:
        for subdir in ['PASS', 'UNTRANSLATABLE', 'FAIL']:
            for f in glob.glob(os.path.join(OUTPUT_DIR, subdir, '*_v*.json')):
                # Extract signal_id from filename like KAUFMAN_DRY_20_v1.json
                basename = os.path.basename(f).rsplit('_v', 1)[0]
                already_done.add(basename)
        print(f"  Resume mode: {len(already_done)} signal_ids already translated")

    for i, signal in enumerate(all_signals):
        sid = signal.get('signal_id', 'unknown')
        book_id = signal.get('book_id', 'unknown')
        book_stats[book_id]['total'] += 1

        # Resume: skip if already translated
        if args.resume and sid in already_done:
            skipped += 1
            continue

        # Pre-filter non-tradeable categories
        cat = signal.get('signal_category', '')
        primary = cat.split('|')[0].strip()
        if primary in NON_TRADEABLE_CATEGORIES:
            book_stats[book_id]['untranslatable'] += 1
            variant_counter[sid] += 1
            results['UNTRANSLATABLE'].append({
                'signal_id': sid,
                'reason': f'Category {cat} not tradeable',
            })
            untranslatable_reasons[f'NON_TRADEABLE:{primary}'] += 1
            continue

        if not signal.get('entry_conditions'):
            book_stats[book_id]['untranslatable'] += 1
            variant_counter[sid] += 1
            results['UNTRANSLATABLE'].append({
                'signal_id': sid,
                'reason': 'No entry conditions',
            })
            untranslatable_reasons['NO_ENTRY_CONDITIONS'] += 1
            continue

        # Translate
        source_chunk = signal.get('raw_chunk_text', '')
        rule = translator.translate(signal, source_chunk)

        if rule.untranslatable:
            book_stats[book_id]['untranslatable'] += 1
            variant_counter[sid] += 1
            vnum = variant_counter[sid]
            results['UNTRANSLATABLE'].append({
                'signal_id': sid,
                'reason': rule.untranslatable_reason,
            })
            untranslatable_reasons[rule.untranslatable_reason or 'unknown'] += 1

            if not args.dry_run:
                path = os.path.join(OUTPUT_DIR, 'UNTRANSLATABLE', f'{sid}_v{vnum}.json')
                with open(path, 'w') as f:
                    json.dump(rule.to_dict(), f, indent=2)
            continue

        # Validate
        vresult = validator.validate(rule)

        if not vresult.passed:
            book_stats[book_id]['fail'] += 1
            results['FAIL'].append({
                'signal_id': sid,
                'issues': vresult.issues,
            })

            variant_counter[sid] += 1
            vnum = variant_counter[sid]
            if not args.dry_run:
                path = os.path.join(OUTPUT_DIR, 'FAIL', f'{sid}_v{vnum}.json')
                with open(path, 'w') as f:
                    json.dump({**rule.to_dict(), 'validation_issues': vresult.issues}, f, indent=2)

            if args.dry_run:
                print(f"  FAIL  {sid}: {vresult.issues[:2]}")
            continue

        # Compile to backtest format
        try:
            compiled = compiler.compile(rule)
        except Exception as e:
            book_stats[book_id]['fail'] += 1
            results['FAIL'].append({
                'signal_id': sid,
                'issues': [f'Compilation error: {e}'],
            })
            continue

        # PASS
        book_stats[book_id]['pass'] += 1
        variant_counter[sid] += 1
        vnum = variant_counter[sid]
        results['PASS'].append({
            'signal_id': sid,
            'dsl_rule': rule.to_dict(),
            'backtest_rule': compiled,
        })

        if not args.dry_run:
            path = os.path.join(OUTPUT_DIR, 'PASS', f'{sid}_v{vnum}.json')
            with open(path, 'w') as f:
                json.dump({
                    'signal_id': sid,
                    'book_id': book_id,
                    'dsl_rule': rule.to_dict(),
                    'backtest_rule': compiled,
                    'source_rule_text': signal.get('rule_text', ''),
                }, f, indent=2)

        if args.dry_run:
            print(f"  PASS  {sid}: {json.dumps(compiled.get('entry_long', []))[:100]}")

        done = i + 1
        if done % 100 == 0 or done == total:
            elapsed = time.time() - start
            rate = done / elapsed if elapsed > 0 else 0
            print(f"  [{done}/{total}] ({rate:.1f}/s) "
                  f"PASS:{len(results['PASS'])} "
                  f"UNTRANS:{len(results['UNTRANSLATABLE'])} "
                  f"FAIL:{len(results['FAIL'])}", flush=True)

    # Summary
    elapsed = time.time() - start
    pass_count = len(results['PASS'])
    untrans_count = len(results['UNTRANSLATABLE'])
    fail_count = len(results['FAIL'])
    total_processed = pass_count + untrans_count + fail_count
    # For PASS rate, compare against tradeable signals only
    tradeable_processed = pass_count + fail_count
    pass_rate = pass_count / tradeable_processed * 100 if tradeable_processed > 0 else 0

    print()
    print("=" * 70)
    print("DSL TRANSLATION SUMMARY")
    print("=" * 70)
    print(f"Total signals:       {total_processed}")
    print(f"PASS:                {pass_count} ({pass_rate:.1f}% of tradeable)")
    print(f"UNTRANSLATABLE:      {untrans_count}")
    print(f"FAIL:                {fail_count}")
    print(f"Time:                {elapsed:.0f}s")
    print()

    # Per-book breakdown
    print(f"{'Book':<15s} {'Total':>6s} {'Pass':>6s} {'Untrans':>8s} {'Fail':>6s} {'Rate':>6s}")
    print("-" * 50)
    for book_id in sorted(book_stats.keys()):
        s = book_stats[book_id]
        tradeable = s['pass'] + s['fail']
        rate = s['pass'] / tradeable * 100 if tradeable > 0 else 0
        print(f"{book_id:<15s} {s['total']:>6d} {s['pass']:>6d} "
              f"{s['untranslatable']:>8d} {s['fail']:>6d} {rate:>5.0f}%")
    print()

    # Top untranslatable reasons
    print("Top untranslatable reasons:")
    for reason, count in untranslatable_reasons.most_common(15):
        print(f"  {count:4d}  {reason[:80]}")
    print()

    # Top validation failure reasons
    if results['FAIL']:
        fail_reasons = Counter()
        for f in results['FAIL']:
            for issue in f.get('issues', []):
                fail_reasons[issue[:60]] += 1
        print("Top validation failure reasons:")
        for reason, count in fail_reasons.most_common(10):
            print(f"  {count:4d}  {reason}")
        print()

    # Comparison with old translator
    print(f"Comparison with old free-form translator:")
    print(f"  Old PASS rate: 48.1% (296/615 of backtestable)")
    print(f"  New PASS rate: {pass_rate:.1f}% ({pass_count}/{tradeable_processed} of tradeable)")
    if pass_rate >= 75:
        print(f"  TARGET MET: {pass_rate:.1f}% >= 75%")
    elif pass_rate >= 60:
        print(f"  ACCEPTABLE: {pass_rate:.1f}% >= 60%")
    else:
        print(f"  BELOW TARGET: {pass_rate:.1f}% < 60% — review untranslatable reasons")
    print()

    # Save summary
    if not args.dry_run:
        summary = {
            'total_processed': total_processed,
            'pass_count': pass_count,
            'untranslatable_count': untrans_count,
            'fail_count': fail_count,
            'pass_rate_of_tradeable': round(pass_rate, 1),
            'book_stats': dict(book_stats),
            'top_untranslatable_reasons': dict(untranslatable_reasons.most_common(20)),
            'pass_signal_ids': [r['signal_id'] for r in results['PASS']],
        }
        summary_path = os.path.join(OUTPUT_DIR, 'summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to: {summary_path}")

        # Save compiled backtest rules for all PASS signals
        all_compiled = []
        for r in results['PASS']:
            all_compiled.append({
                'signal_id': r['signal_id'],
                'backtestable': True,
                'rules': r['backtest_rule'],
            })
        compiled_path = os.path.join(OUTPUT_DIR, 'dsl_translated_signals.json')
        with open(compiled_path, 'w') as f:
            json.dump(all_compiled, f, indent=2)
        print(f"Compiled rules saved to: {compiled_path}")


def run_retranslate_fixable(translator, validator, compiler, args):
    """Retranslate only FIXABLE signals from previous validator run."""

    print("Loading FIXABLE signals...")
    with open(VALIDATOR_REPORT_PATH) as f:
        report = json.load(f)

    fixable = [r for r in report['results'] if r['verdict'] == 'FIXABLE']
    print(f"FIXABLE signals: {len(fixable)}")

    # Load original translated signals for broken rules
    with open(TRANSLATED_PATH) as f:
        translated = json.load(f)
    trans_lookup = {}
    for t in translated:
        sid = t.get('signal_id')
        if sid and t.get('rules'):
            trans_lookup[sid] = t['rules']

    # Load approved signals for source chunks
    with open(ALL_SIGNALS_PATH) as f:
        approved = json.load(f)
    source_lookup = {}
    for s in approved:
        sid = s.get('signal_id')
        if sid and sid not in source_lookup:
            source_lookup[sid] = s.get('raw_chunk_text', '')

    if args.limit > 0:
        fixable = fixable[:args.limit]

    if not args.dry_run:
        os.makedirs(os.path.join(OUTPUT_DIR, 'RETRANSLATED'), exist_ok=True)

    results = {'pass': 0, 'fail': 0, 'still_untranslatable': 0}

    for i, f_entry in enumerate(fixable):
        sid = f_entry['signal_id']
        issues = [iss['detail'] for iss in f_entry.get('issues', [])]
        broken_rule = trans_lookup.get(sid, {})
        source_chunk = source_lookup.get(sid, '')

        rule = translator.retranslate_fixable(sid, source_chunk, broken_rule, issues)

        if rule.untranslatable:
            results['still_untranslatable'] += 1
            if args.dry_run:
                print(f"  UNTRANS {sid}: {rule.untranslatable_reason}")
            continue

        vresult = validator.validate(rule)
        if not vresult.passed:
            results['fail'] += 1
            if args.dry_run:
                print(f"  FAIL    {sid}: {vresult.issues[:2]}")
            continue

        results['pass'] += 1
        if args.dry_run:
            compiled = compiler.compile(rule)
            print(f"  PASS    {sid}: {json.dumps(compiled.get('entry_long', []))[:80]}")

        if not args.dry_run:
            compiled = compiler.compile(rule)
            path = os.path.join(OUTPUT_DIR, 'RETRANSLATED', f'{sid}.json')
            with open(path, 'w') as fp:
                json.dump({
                    'signal_id': sid,
                    'dsl_rule': rule.to_dict(),
                    'backtest_rule': compiled,
                    'retranslated_from': 'FIXABLE',
                }, fp, indent=2)

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(fixable)}] PASS:{results['pass']} "
                  f"FAIL:{results['fail']} UNTRANS:{results['still_untranslatable']}",
                  flush=True)

    print()
    print(f"RETRANSLATION RESULTS:")
    print(f"  Input FIXABLE:       {len(fixable)}")
    print(f"  Now PASS:            {results['pass']}")
    print(f"  Still FAIL:          {results['fail']}")
    print(f"  Still UNTRANSLATABLE: {results['still_untranslatable']}")
    salvage_rate = results['pass'] / len(fixable) * 100 if fixable else 0
    print(f"  Salvage rate:        {salvage_rate:.0f}%")


def run_consolidate():
    """Pick best variant per signal_id into BEST/ directory."""
    print("Consolidating variants...")

    best_dir = os.path.join(OUTPUT_DIR, 'BEST')
    os.makedirs(best_dir, exist_ok=True)

    # Collect all variants per signal_id across all directories
    all_variants = defaultdict(list)  # signal_id -> [(priority, path, data)]

    PRIORITY = {'PASS': 0, 'FAIL': 1, 'UNTRANSLATABLE': 2}

    for subdir in ['PASS', 'FAIL', 'UNTRANSLATABLE']:
        dir_path = os.path.join(OUTPUT_DIR, subdir)
        if not os.path.exists(dir_path):
            continue
        for fname in os.listdir(dir_path):
            if not fname.endswith('.json'):
                continue
            # Extract signal_id from filename like KAUFMAN_DRY_20_v1.json
            sid = fname.rsplit('_v', 1)[0]
            fpath = os.path.join(dir_path, fname)
            try:
                with open(fpath) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, IOError):
                continue

            # Score the variant for ranking
            priority = PRIORITY.get(subdir, 3)

            # For PASS variants, count conditions (more = more specific)
            n_conditions = 0
            indicators_used = set()
            if subdir == 'PASS':
                bt_rules = data.get('backtest_rule', data.get('dsl_rule', {}))
                for side in ['entry_long', 'entry_short', 'exit_long', 'exit_short']:
                    conds = bt_rules.get(side, [])
                    n_conditions += len(conds)
                    for c in conds:
                        ind = c.get('indicator', c.get('left', ''))
                        if ind:
                            indicators_used.add(ind)

            all_variants[sid].append({
                'subdir': subdir,
                'path': fpath,
                'data': data,
                'priority': priority,
                'n_conditions': n_conditions,
                'n_indicators': len(indicators_used),
            })

    # Select best variant per signal_id
    book_counts = defaultdict(lambda: {'pass': 0, 'untranslatable': 0, 'fail': 0})
    total_selected = 0
    multi_variant = 0

    for sid, variants in sorted(all_variants.items()):
        if len(variants) > 1:
            multi_variant += 1

        # Sort: lowest priority (PASS=0 first), then most conditions, then most diverse indicators
        variants.sort(key=lambda v: (v['priority'], -v['n_conditions'], -v['n_indicators']))
        best = variants[0]

        # Write to BEST/
        out_path = os.path.join(best_dir, f'{sid}.json')
        with open(out_path, 'w') as f:
            json.dump(best['data'], f, indent=2)

        total_selected += 1
        book_id = sid.split('_')[0]
        if best['subdir'] == 'PASS':
            book_counts[book_id]['pass'] += 1
        elif best['subdir'] == 'UNTRANSLATABLE':
            book_counts[book_id]['untranslatable'] += 1
        else:
            book_counts[book_id]['fail'] += 1

    # Report
    pass_count = sum(c['pass'] for c in book_counts.values())
    untrans_count = sum(c['untranslatable'] for c in book_counts.values())
    fail_count = sum(c['fail'] for c in book_counts.values())

    print(f"\nCONSOLIDATION RESULTS")
    print(f"{'='*60}")
    print(f"Total unique signal_ids:              {total_selected}")
    print(f"Signal_ids with multiple variants:    {multi_variant}")
    print(f"BEST selections:")
    print(f"  PASS:           {pass_count}")
    print(f"  UNTRANSLATABLE: {untrans_count}")
    print(f"  FAIL:           {fail_count}")
    print()

    print(f"{'Book':<15s} {'PASS':>6s} {'Untrans':>8s} {'Fail':>6s}")
    print("-" * 40)
    for book_id in sorted(book_counts.keys()):
        c = book_counts[book_id]
        print(f"{book_id:<15s} {c['pass']:>6d} {c['untranslatable']:>8d} {c['fail']:>6d}")

    print(f"\nBEST files saved to: {best_dir}/")
    print(f"Total PASS signal_ids: {pass_count}")
    if pass_count >= 150:
        print(f"✓ TARGET MET: {pass_count} >= 150")
    else:
        print(f"✗ BELOW TARGET: {pass_count} < 150")
        # Show top untranslatable reasons for BEST selections
        untrans_reasons = Counter()
        for sid, variants in all_variants.items():
            best = sorted(variants, key=lambda v: (v['priority'], -v['n_conditions']))[0]
            if best['subdir'] == 'UNTRANSLATABLE':
                reason = best['data'].get('untranslatable_reason', 'unknown')
                untrans_reasons[reason] += 1
        if untrans_reasons:
            print(f"\nTop untranslatable reasons blocking BEST PASS count:")
            for reason, count in untrans_reasons.most_common(10):
                print(f"  {count:4d}  {reason[:80]}")


if __name__ == '__main__':
    main()
