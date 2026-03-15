"""Strip hallucinated numeric thresholds from REVISE signals using Haiku."""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from extraction.llm_client import LLMClient
from review.hallucination_checker import HallucinationChecker
from ingestion.vector_store import VectorStore

logger = logging.getLogger(__name__)

STRIP_PROMPT = """Given this source chunk from a trading book:
{chunk}

And this extracted signal:
{signal}

The signal contains numeric thresholds not found in the source.
Return the signal with ALL fabricated numeric thresholds removed,
replaced with [THRESHOLD_NEEDED].

Keep the rule structure intact. Only remove numbers not present
in the source text.

Return JSON only:
{{
  "rule_text": "...",
  "entry_conditions": ["...", "..."],
  "exit_conditions": ["...", "..."],
  "parameters": {{...}},
  "thresholds_removed": ["list of removed thresholds"]
}}
"""


def run_threshold_strip(num_workers=6):
    """Strip hallucinated thresholds from REVISE signals."""
    # Load FAIL signals
    with open('extraction_results/sonnet_checked/_FAILED.json') as f:
        failed = json.load(f)

    revise = [s for s in failed if s.get('hallucination_recommendation') == 'REVISE']
    print(f"REVISE signals to strip: {len(revise)}", flush=True)

    vs = VectorStore('./chromadb_data')
    checker = HallucinationChecker(vs, num_workers=num_workers)
    llm = LLMClient()

    total = len(revise)
    done = 0
    errors = 0
    start = time.time()

    def strip_one(signal):
        full_chunk = checker.get_full_chunk(signal)
        if not full_chunk:
            signal['strip_status'] = 'NO_CHUNK'
            return signal

        sig_summary = json.dumps({
            'rule_text': signal.get('rule_text', ''),
            'entry_conditions': signal.get('entry_conditions', []),
            'exit_conditions': signal.get('exit_conditions', []),
            'parameters': signal.get('parameters', {}),
        }, indent=2)

        prompt = STRIP_PROMPT.format(chunk=full_chunk, signal=sig_summary)
        try:
            result = llm._call_anthropic(prompt, 'claude-haiku-4-5-20251001')
            if result and isinstance(result, dict):
                signal['rule_text'] = result.get('rule_text', signal['rule_text'])
                signal['entry_conditions'] = result.get('entry_conditions', signal['entry_conditions'])
                signal['exit_conditions'] = result.get('exit_conditions', signal['exit_conditions'])
                signal['parameters'] = result.get('parameters', signal['parameters'])
                signal['thresholds_removed'] = result.get('thresholds_removed', [])
                signal['strip_status'] = 'STRIPPED'
            else:
                signal['strip_status'] = 'PARSE_ERROR'
        except Exception as e:
            logger.warning(f"Strip error: {e}")
            signal['strip_status'] = 'ERROR'
        return signal

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(strip_one, s): i for i, s in enumerate(revise)}
        for future in as_completed(futures):
            result = future.result()
            done += 1
            if result.get('strip_status') != 'STRIPPED':
                errors += 1
            if done % 100 == 0 or done == total:
                elapsed = time.time() - start
                rate = done / elapsed if elapsed > 0 else 0
                print(f"  Strip: {done}/{total} ({rate:.1f}/s) errors:{errors}",
                      flush=True)

    # Save stripped signals
    stripped = [s for s in revise if s.get('strip_status') == 'STRIPPED']
    with open('extraction_results/sonnet_checked/_REVISE_STRIPPED.json', 'w') as f:
        json.dump(stripped, f, indent=2)

    # Per-book breakdown
    from collections import Counter
    by_book = Counter(s['book_id'] for s in stripped)
    print(f"\n=== THRESHOLD STRIP COMPLETE ===", flush=True)
    print(f"Input: {len(revise)} -> Stripped: {len(stripped)} -> Errors: {errors}",
          flush=True)
    print(f"\nPer book:", flush=True)
    for b in sorted(by_book):
        print(f"  {b}: {by_book[b]}", flush=True)

    return stripped


if __name__ == '__main__':
    run_threshold_strip()
