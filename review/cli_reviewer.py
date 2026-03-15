"""
Human Review CLI for extracted trading signals.

Presents each extracted rule for human review.
Reviewer sees: original chunk, extracted rule, quality checks.
Reviewer approves, revises, rejects, or defers.

Usage:
    python -m review.cli_reviewer --tier 1 --book KAUFMAN
    python -m review.cli_reviewer --tier 2 --book NATENBERG
    python -m review.cli_reviewer --tier 1  # all books, tier 1
"""

import json
import os
from datetime import datetime
from typing import Optional

from ingestion.signal_candidate import SignalCandidate


def _signal_from_dict(d: dict) -> SignalCandidate:
    """Convert a JSON dict to a SignalCandidate for display."""
    return SignalCandidate(
        signal_id=d.get('signal_id', ''),
        book_id=d.get('book_id', ''),
        source_citation=d.get('source_citation', ''),
        raw_chunk_text=d.get('raw_chunk_text', ''),
        rule_text=d.get('rule_text', ''),
        signal_category=d.get('signal_category', ''),
        direction=d.get('direction', ''),
        entry_conditions=d.get('entry_conditions', []),
        parameters=d.get('parameters', {}),
        exit_conditions=d.get('exit_conditions', []),
        instrument=d.get('instrument', ''),
        timeframe=d.get('timeframe', ''),
        target_regimes=d.get('target_regimes', []),
        hallucination_verdict=d.get('hallucination_verdict', 'UNKNOWN'),
        hallucination_issues=d.get('hallucination_issues', []),
        completeness_warning=d.get('completeness_warning'),
        author_confidence=d.get('author_confidence', ''),
        variant_id=d.get('variant_id'),
    )


class CLIReviewer:
    """
    Simple CLI reviewer for signal candidates.
    No web framework needed at this stage.
    """

    def review_signal(self, signal_candidate) -> dict:
        """
        Shows reviewer:
        1. Original book text (source pages)
        2. Extracted rule (from Claude API)
        3. Hallucination check result
        4. Prompts for decision

        Returns dict with keys:
            decision: A|R|X|D
            notes: str
            revised_conditions: list[str] or None
            reviewer_timestamp: ISO timestamp
        """
        print("\n" + "=" * 60)
        print(f"SIGNAL: {signal_candidate.signal_id}")
        print(f"SOURCE: {signal_candidate.source_citation}")
        if signal_candidate.variant_id:
            print(f"VARIANT: {signal_candidate.variant_id}")
        print("=" * 60)

        print("\n--- ORIGINAL TEXT (source pages) ---")
        print(signal_candidate.raw_chunk_text[:1000])
        if len(signal_candidate.raw_chunk_text) > 1000:
            print(f"  ... ({len(signal_candidate.raw_chunk_text)} chars total)")

        print("\n--- EXTRACTED RULE ---")
        print(f"Rule: {signal_candidate.rule_text}")
        print(f"Category: {signal_candidate.signal_category}")
        print(f"Direction: {signal_candidate.direction}")
        print(f"Entry conditions:")
        for c in signal_candidate.entry_conditions:
            print(f"  - {c}")
        print(f"Parameters: {signal_candidate.parameters}")
        print(f"Exit conditions:")
        for c in signal_candidate.exit_conditions:
            print(f"  - {c}")
        print(f"Instrument: {signal_candidate.instrument}")
        print(f"Timeframe: {signal_candidate.timeframe}")
        print(f"Regime: {signal_candidate.target_regimes}")

        print("\n--- QUALITY CHECK ---")
        verdict = signal_candidate.hallucination_verdict
        verdict_display = {
            'PASS': 'PASS (faithful)',
            'WARN': 'WARNING (minor issues)',
            'FAIL': 'FAIL (significant issues)',
            'UNKNOWN': 'NOT CHECKED',
        }.get(verdict, verdict)
        print(f"Hallucination check: {verdict_display}")
        if signal_candidate.hallucination_issues:
            for issue in signal_candidate.hallucination_issues:
                print(f"  ! {issue}")

        if signal_candidate.completeness_warning:
            print(f"\n--- COMPLETENESS WARNING ---")
            print(signal_candidate.completeness_warning)

        if signal_candidate.author_confidence:
            print(f"Author confidence: {signal_candidate.author_confidence}")

        # Decision
        print("\n--- YOUR DECISION ---")
        print("A = Approve  |  R = Revise  |  X = Reject  |  D = Defer")

        while True:
            decision = input("Decision: ").strip().upper()
            if decision in ('A', 'R', 'X', 'D'):
                break
            print("Invalid. Enter A, R, X, or D.")

        notes = ""
        revised_conditions = None

        if decision == 'R':
            print("Enter revised entry conditions"
                  " (one per line, blank line to finish):")
            lines = []
            while True:
                line = input()
                if not line:
                    break
                lines.append(line)
            revised_conditions = lines
            notes = input("Revision notes: ")

        elif decision == 'X':
            notes = input("Rejection reason: ")

        elif decision == 'A':
            notes = input("Optional notes (press Enter to skip): ")

        return {
            'decision': decision,
            'notes': notes,
            'revised_conditions': revised_conditions,
            'reviewer_timestamp': datetime.now().isoformat(),
        }


def _load_tier_signals(tier: int, book: Optional[str] = None) -> list:
    """Load signals for review based on tier."""
    if tier == 1:
        # Sonnet-checked PASS + WARN signals
        source_dir = 'extraction_results/sonnet_checked'
    elif tier == 2:
        # Threshold-stripped REVISE signals
        source_dir = 'extraction_results/sonnet_checked'
        with open(os.path.join(source_dir, '_REVISE_STRIPPED.json')) as f:
            signals = json.load(f)
        if book:
            signals = [s for s in signals if s['book_id'] == book]
        return signals
    else:
        raise ValueError(f"Unknown tier: {tier}")

    # Tier 1: load per-book files (excluding _FAILED and _REVISE_STRIPPED)
    signals = []
    if book:
        path = os.path.join(source_dir, f'{book}.json')
        if os.path.exists(path):
            with open(path) as f:
                signals = json.load(f)
    else:
        for fn in sorted(os.listdir(source_dir)):
            if fn.startswith('_') or not fn.endswith('.json'):
                continue
            with open(os.path.join(source_dir, fn)) as f:
                signals.extend(json.load(f))

    return signals


def run_review(tier: int, book: Optional[str] = None):
    """Run interactive review session."""
    signals = _load_tier_signals(tier, book)

    if not signals:
        print(f"No signals found for tier={tier}" +
              (f" book={book}" if book else ""))
        return

    # Load existing progress
    progress_dir = 'extraction_results/review_progress'
    os.makedirs(progress_dir, exist_ok=True)
    progress_file = os.path.join(
        progress_dir,
        f'tier{tier}_{book or "ALL"}.json'
    )

    reviewed = {}
    if os.path.exists(progress_file):
        with open(progress_file) as f:
            reviewed = json.load(f)

    # Filter out already-reviewed signals
    pending = [s for s in signals if s.get('signal_id', '') not in reviewed]

    book_label = book or "ALL BOOKS"
    tier_label = "PASS+WARN" if tier == 1 else "STRIPPED (needs thresholds)"
    print(f"\n{'='*60}")
    print(f"REVIEW SESSION: Tier {tier} — {tier_label}")
    print(f"Book: {book_label}")
    print(f"Total: {len(signals)} | Already reviewed: {len(reviewed)} | "
          f"Pending: {len(pending)}")
    print(f"{'='*60}")

    if not pending:
        print("All signals already reviewed!")
        return

    reviewer = CLIReviewer()
    stats = {'A': 0, 'R': 0, 'X': 0, 'D': 0}

    try:
        for i, sig_dict in enumerate(pending):
            print(f"\n[{i+1}/{len(pending)}] "
                  f"({len(reviewed) + i + 1}/{len(signals)} overall)")

            # Show [THRESHOLD_NEEDED] markers for tier 2
            if tier == 2:
                removed = sig_dict.get('thresholds_removed', [])
                if removed:
                    print(f"  Thresholds stripped: {removed}")

            signal = _signal_from_dict(sig_dict)
            result = reviewer.review_signal(signal)

            # Save decision
            sig_id = sig_dict.get('signal_id', f'unknown_{i}')
            reviewed[sig_id] = {
                **result,
                'book_id': sig_dict.get('book_id', ''),
                'rule_text': sig_dict.get('rule_text', '')[:200],
            }
            stats[result['decision']] += 1

            # Save progress after each decision
            with open(progress_file, 'w') as f:
                json.dump(reviewed, f, indent=2)

            # Status line
            print(f"\n  Session: A:{stats['A']} R:{stats['R']} "
                  f"X:{stats['X']} D:{stats['D']}")

    except (KeyboardInterrupt, EOFError):
        print(f"\n\nSession interrupted. Progress saved to {progress_file}")

    print(f"\n{'='*60}")
    print(f"SESSION SUMMARY")
    print(f"  Approved:  {stats['A']}")
    print(f"  Revised:   {stats['R']}")
    print(f"  Rejected:  {stats['X']}")
    print(f"  Deferred:  {stats['D']}")
    print(f"  Progress:  {len(reviewed)}/{len(signals)}")
    print(f"  Saved to:  {progress_file}")
    print(f"{'='*60}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Review extracted signals')
    parser.add_argument('--tier', type=int, required=True, choices=[1, 2],
                        help='1=PASS+WARN signals, 2=stripped REVISE signals')
    parser.add_argument('--book', type=str, default=None,
                        help='Book ID (e.g. KAUFMAN). Omit for all books.')
    args = parser.parse_args()

    run_review(args.tier, args.book)
