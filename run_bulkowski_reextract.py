"""
Re-extract BULKOWSKI signals with pattern-level signal IDs.

Problem: All 1,392 BULKOWSKI signals share signal_id BULKOWSKI_DRY_0.
Fix: Assign signal_id from pattern name (e.g., BULKOWSKI_DOUBLE_BOTTOM).
Then re-run DSL translation with pattern-specific prompt.

Usage:
    python run_bulkowski_reextract.py                 # re-assign IDs + translate
    python run_bulkowski_reextract.py --dry-run       # preview ID assignments
    python run_bulkowski_reextract.py --translate-only # skip re-assignment
"""

import argparse
import json
import os
import re
import time
from collections import Counter, defaultdict

from extraction.dsl_translator import DSLTranslator
from extraction.dsl_validator import DSLValidator
from extraction.dsl_to_backtest import DSLToBacktest

ALL_SIGNALS_PATH = 'extraction_results/approved/_ALL.json'
OUTPUT_DIR = 'dsl_results'

# Canonical pattern name mapping — normalizes 618 rule_text variants to ~35 patterns
PATTERN_CANONICALIZE = {
    # Broadening
    'broadening bottom': 'BROADENING_BOTTOM',
    'broadening top': 'BROADENING_TOP',
    'broadening formation': 'BROADENING_TOP',
    'broadening pattern': 'BROADENING_TOP',
    'broadening tops': 'BROADENING_TOP',
    'ascending broadening wedge': 'ASCENDING_BROADENING_WEDGE',
    'descending broadening wedge': 'DESCENDING_BROADENING_WEDGE',
    # Double
    'double bottom': 'DOUBLE_BOTTOM',
    'double top': 'DOUBLE_TOP',
    # Triple
    'triple bottom': 'TRIPLE_BOTTOM',
    'triple top': 'TRIPLE_TOP',
    # Head and Shoulders
    'head and shoulders': 'HEAD_SHOULDERS_TOP',
    'head-and-shoulders top': 'HEAD_SHOULDERS_TOP',
    'head and shoulders top': 'HEAD_SHOULDERS_TOP',
    'head-and-shoulders bottom': 'HEAD_SHOULDERS_BOTTOM',
    'head and shoulders bottom': 'HEAD_SHOULDERS_BOTTOM',
    # Diamond
    'diamond top': 'DIAMOND_TOP',
    'diamond bottom': 'DIAMOND_BOTTOM',
    'diamond pattern': 'DIAMOND_TOP',
    # Rectangle
    'rectangle': 'RECTANGLE',
    'rectangle top': 'RECTANGLE_TOP',
    'rectangle bottom': 'RECTANGLE_BOTTOM',
    'rectangle pattern': 'RECTANGLE',
    # Cup and Handle
    'cup with handle': 'CUP_HANDLE',
    'cup and handle': 'CUP_HANDLE',
    'cup-with-handle': 'CUP_HANDLE',
    'cup and handle pattern': 'CUP_HANDLE',
    'cup-with-handle pattern': 'CUP_HANDLE',
    # Flag / Pennant
    'flag': 'FLAG',
    'flag pattern': 'FLAG',
    'pennant': 'PENNANT',
    'high tight flag': 'HIGH_TIGHT_FLAG',
    # Wedge
    'wedge': 'WEDGE',
    'rising wedge': 'WEDGE_RISING',
    'falling wedge': 'WEDGE_FALLING',
    # Horn
    'horn bottom': 'HORN_BOTTOM',
    'horn top': 'HORN_TOP',
    'horn pattern': 'HORN_BOTTOM',
    # Pipe
    'pipe bottom': 'PIPE_BOTTOM',
    'pipe top': 'PIPE_TOP',
    'pipe pattern': 'PIPE_BOTTOM',
    # Scallop
    'scallop': 'SCALLOP',
    'scallop pattern': 'SCALLOP',
    'ascending scallop': 'ASCENDING_SCALLOP',
    'ascending scallops': 'ASCENDING_SCALLOP',
    'descending scallop': 'DESCENDING_SCALLOP',
    # Rounding
    'rounding bottom': 'ROUNDING_BOTTOM',
    'rounding top': 'ROUNDING_TOP',
    # Island
    'island reversal': 'ISLAND_REVERSAL',
    'island pattern': 'ISLAND_REVERSAL',
    # Measured Move
    'measured move down': 'MEASURED_MOVE_DOWN',
    'measured move up': 'MEASURED_MOVE_UP',
    # Bump and Run
    'bump and run': 'BUMP_AND_RUN',
    'bump and run reversal': 'BUMP_AND_RUN',
    # Symmetrical Triangle
    'symmetrical triangle': 'SYM_TRIANGLE',
    'ascending triangle': 'ASC_TRIANGLE',
    'descending triangle': 'DESC_TRIANGLE',
    # Generic
    'pattern name': 'GENERIC_PATTERN',
}


def canonicalize_pattern(rule_text: str) -> str:
    """Extract canonical pattern name from rule_text."""
    if not rule_text:
        return 'GENERIC_PATTERN'

    # Get text before colon
    if ':' in rule_text:
        prefix = rule_text.split(':')[0].strip()
    else:
        prefix = rule_text[:60].strip()

    prefix_lower = prefix.lower()

    # Try exact match
    if prefix_lower in PATTERN_CANONICALIZE:
        return PATTERN_CANONICALIZE[prefix_lower]

    # Try partial match (longest first)
    for key in sorted(PATTERN_CANONICALIZE.keys(), key=len, reverse=True):
        if key in prefix_lower:
            return PATTERN_CANONICALIZE[key]

    # Fallback: generate from prefix
    clean = re.sub(r'[^a-zA-Z\s]', '', prefix)
    clean = '_'.join(clean.upper().split()[:4])
    return clean or 'GENERIC_PATTERN'


def main():
    parser = argparse.ArgumentParser(description='Re-extract BULKOWSKI with pattern-level IDs')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--translate-only', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()

    # Load all signals
    print("Loading signals...")
    with open(ALL_SIGNALS_PATH) as f:
        all_signals = json.load(f)

    bulkowski = [s for s in all_signals if s.get('book_id') == 'BULKOWSKI']
    print(f"Total BULKOWSKI signals: {len(bulkowski)}")

    # Step 1: Assign pattern-level signal IDs
    pattern_counts = Counter()
    reassigned = []

    for s in bulkowski:
        rule_text = s.get('rule_text', '')
        canon = canonicalize_pattern(rule_text)
        pattern_counts[canon] += 1

        # New signal_id: BULKOWSKI_{PATTERN}
        new_sid = f"BULKOWSKI_{canon}"
        s_copy = dict(s)
        s_copy['signal_id'] = new_sid
        s_copy['parameters'] = dict(s.get('parameters', {}))
        s_copy['parameters']['_canonical_name'] = canon
        s_copy['parameters']['pattern_name'] = rule_text.split(':')[0].strip() if ':' in rule_text else rule_text[:50]
        reassigned.append(s_copy)

    print(f"\nPattern distribution ({len(pattern_counts)} unique patterns):")
    for p, c in pattern_counts.most_common():
        print(f"  {p:<35s} {c:>4d}")

    if args.dry_run:
        print("\n[DRY RUN] No files written.")
        return

    if args.limit > 0:
        reassigned = reassigned[:args.limit]
        print(f"\nLimited to {args.limit} signals")

    # Step 2: DSL translate each pattern signal
    translator = DSLTranslator(num_workers=args.workers)
    validator = DSLValidator()
    compiler = DSLToBacktest()

    # Create output dirs
    for subdir in ['PASS', 'UNTRANSLATABLE', 'FAIL']:
        os.makedirs(os.path.join(OUTPUT_DIR, subdir), exist_ok=True)

    results = {'pass': 0, 'untranslatable': 0, 'fail': 0}
    pattern_results = defaultdict(lambda: {'pass': 0, 'untranslatable': 0, 'fail': 0})
    variant_counter = Counter()
    start = time.time()

    for i, signal in enumerate(reassigned):
        sid = signal['signal_id']
        source_chunk = signal.get('raw_chunk_text', '')

        rule = translator.translate(signal, source_chunk)

        if rule.untranslatable:
            results['untranslatable'] += 1
            pattern_results[signal['parameters']['_canonical_name']]['untranslatable'] += 1
            variant_counter[sid] += 1
            vnum = variant_counter[sid]
            path = os.path.join(OUTPUT_DIR, 'UNTRANSLATABLE', f'{sid}_v{vnum}.json')
            with open(path, 'w') as f:
                json.dump(rule.to_dict(), f, indent=2)
            continue

        # Validate
        vresult = validator.validate(rule)
        if not vresult.passed:
            results['fail'] += 1
            pattern_results[signal['parameters']['_canonical_name']]['fail'] += 1
            variant_counter[sid] += 1
            vnum = variant_counter[sid]
            path = os.path.join(OUTPUT_DIR, 'FAIL', f'{sid}_v{vnum}.json')
            with open(path, 'w') as f:
                json.dump({**rule.to_dict(), 'validation_issues': vresult.issues}, f, indent=2)
            continue

        # Compile
        try:
            compiled = compiler.compile(rule)
        except Exception as e:
            results['fail'] += 1
            pattern_results[signal['parameters']['_canonical_name']]['fail'] += 1
            continue

        # PASS
        results['pass'] += 1
        pattern_results[signal['parameters']['_canonical_name']]['pass'] += 1
        variant_counter[sid] += 1
        vnum = variant_counter[sid]
        path = os.path.join(OUTPUT_DIR, 'PASS', f'{sid}_v{vnum}.json')
        with open(path, 'w') as f:
            json.dump({
                'signal_id': sid,
                'book_id': 'BULKOWSKI',
                'pattern_name': signal['parameters'].get('pattern_name', ''),
                'dsl_rule': rule.to_dict(),
                'backtest_rule': compiled,
                'source_rule_text': signal.get('rule_text', ''),
            }, f, indent=2)

        done = i + 1
        if done % 100 == 0 or done == len(reassigned):
            elapsed = time.time() - start
            print(f"  [{done}/{len(reassigned)}] ({elapsed:.0f}s) "
                  f"PASS:{results['pass']} UNTRANS:{results['untranslatable']} "
                  f"FAIL:{results['fail']}", flush=True)

    # Report
    elapsed = time.time() - start
    print(f"\n{'='*70}")
    print(f"BULKOWSKI RE-EXTRACTION RESULTS")
    print(f"{'='*70}")
    print(f"Total processed: {len(reassigned)}")
    print(f"PASS:            {results['pass']}")
    print(f"UNTRANSLATABLE:  {results['untranslatable']}")
    print(f"FAIL:            {results['fail']}")
    print(f"Time:            {elapsed:.0f}s")

    # Per-pattern breakdown
    print(f"\n{'Pattern':<35s} {'PASS':>6s} {'UNTRANS':>8s} {'FAIL':>6s}")
    print("-" * 60)
    for pattern in sorted(pattern_results.keys()):
        pr = pattern_results[pattern]
        print(f"{pattern:<35s} {pr['pass']:>6d} {pr['untranslatable']:>8d} {pr['fail']:>6d}")

    # Count unique signal_ids that got PASS
    pass_sids = set()
    for f_name in os.listdir(os.path.join(OUTPUT_DIR, 'PASS')):
        if f_name.startswith('BULKOWSKI_') and not f_name.startswith('BULKOWSKI_DRY_'):
            pass_sids.add(f_name.rsplit('_v', 1)[0])
    print(f"\nUnique BULKOWSKI pattern signal_ids with PASS: {len(pass_sids)}")
    for sid in sorted(pass_sids):
        print(f"  {sid}")


if __name__ == '__main__':
    main()
