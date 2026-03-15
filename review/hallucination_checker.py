"""
Multi-stage hallucination checking pipeline.

Stage 1 (Haiku): Cheap binary pre-screen — "Is this a testable rule?" ($0.001/call)
Stage 2 (Sonnet): Full grounded check against original source chunk ($0.01/call)

Both stages use ThreadPoolExecutor for parallel processing.
"""

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional

from extraction.llm_client import LLMClient
from extraction.prompts import HALLUCINATION_CHECK_PROMPT
from ingestion.vector_store import VectorStore

logger = logging.getLogger(__name__)

HAIKU_PRESCREEN_PROMPT = """You are evaluating an extracted trading signal for quality.

Signal category: {signal_category}
Rule text: {rule_text}
Entry conditions: {entry_conditions}
Exit conditions: {exit_conditions}

Answer with JSON only:
{{
  "is_testable": true/false,
  "reason": "one sentence"
}}

Criteria for is_testable=true:
- Has at least one specific, actionable entry condition (not vague advice)
- A programmer could write code to detect this condition from market data,
  portfolio state, or trader behavior rules
- Pure philosophy or motivation without any actionable trigger = false
- "Wait for X then do Y" patterns = true even if X is qualitative
"""


class HallucinationChecker:
    """Runs Haiku pre-screen and Sonnet hallucination checks."""

    def __init__(self, vector_store: VectorStore, num_workers: int = 4):
        self.vs = vector_store
        self.llm = LLMClient()
        self.num_workers = num_workers
        self._chunk_cache: Dict[str, str] = {}

    def _load_book_chunks(self, book_id: str):
        """Load all chunks for a book into cache. Called once per book."""
        cache_key = f"_book_{book_id}"
        if cache_key in self._chunk_cache:
            return
        chunks = self.vs.get_all_chunks(book_id)
        for c in chunks:
            text = c['text']
            # Index by first 40 chars (stripped) for lookup
            prefix = text[:40].strip()
            self._chunk_cache[f"{book_id}:{prefix}"] = text
        self._chunk_cache[cache_key] = True  # mark loaded
        logger.info(f"Cached {len(chunks)} chunks for {book_id}")

    def get_full_chunk(self, signal: dict) -> Optional[str]:
        """Retrieve full chunk text from ChromaDB using truncated raw_chunk_text."""
        book_id = signal.get('book_id', '')
        raw_prefix = signal.get('raw_chunk_text', '')[:40].strip()

        # Ensure book chunks are loaded
        self._load_book_chunks(book_id)

        cache_key = f"{book_id}:{raw_prefix}"
        if cache_key in self._chunk_cache:
            return self._chunk_cache[cache_key]

        # Fallback: substring search in cached chunks
        book_prefix = f"{book_id}:"
        for key, text in self._chunk_cache.items():
            if key.startswith(book_prefix) and key != f"_book_{book_id}":
                if raw_prefix[:25] in text[:100]:
                    self._chunk_cache[cache_key] = text  # cache for next time
                    return text

        logger.warning(f"Chunk not found for {book_id}: {raw_prefix[:30]}")
        return None

    # ------------------------------------------------------------------
    # Stage 1: Haiku pre-screen
    # ------------------------------------------------------------------

    def _haiku_prescreen_one(self, signal: dict) -> dict:
        """Run Haiku pre-screen on one signal. Returns signal with haiku_testable field."""
        prompt = HAIKU_PRESCREEN_PROMPT.format(
            signal_category=signal.get('signal_category', ''),
            rule_text=signal.get('rule_text', ''),
            entry_conditions=json.dumps(signal.get('entry_conditions', []))[:500],
            exit_conditions=json.dumps(signal.get('exit_conditions', []))[:300],
        )
        try:
            result = self.llm._call_anthropic(prompt, 'claude-haiku-4-5-20251001')
            if result and isinstance(result, dict):
                signal['haiku_testable'] = result.get('is_testable', False)
                signal['haiku_reason'] = result.get('reason', '')
            else:
                signal['haiku_testable'] = True  # keep on parse failure
        except Exception as e:
            logger.warning(f"Haiku prescreen error: {e}")
            signal['haiku_testable'] = True  # keep on error
        return signal

    def haiku_prescreen(self, signals: List[dict]) -> List[dict]:
        """Run Haiku pre-screen on all signals. Returns signals marked testable."""
        print(f"\n--- Haiku Pre-screen: {len(signals)} signals, "
              f"{self.num_workers} workers ---", flush=True)

        total = len(signals)
        done = 0
        start = time.time()

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {executor.submit(self._haiku_prescreen_one, s): i
                       for i, s in enumerate(signals)}
            for future in as_completed(futures):
                future.result()  # signal is modified in place
                done += 1
                if done % 100 == 0 or done == total:
                    elapsed = time.time() - start
                    rate = done / elapsed if elapsed > 0 else 0
                    print(f"  Haiku: {done}/{total} ({rate:.1f}/s)", flush=True)

        kept = [s for s in signals if s.get('haiku_testable', True)]
        dropped = total - len(kept)
        print(f"  Haiku result: {total} -> {len(kept)} "
              f"(dropped {dropped} non-testable)", flush=True)
        return kept

    # ------------------------------------------------------------------
    # Stage 2: Sonnet hallucination check (grounded against source chunk)
    # ------------------------------------------------------------------

    def _sonnet_check_one(self, signal: dict) -> dict:
        """Run Sonnet hallucination check on one signal with full source chunk."""
        full_chunk = self.get_full_chunk(signal)
        if not full_chunk:
            signal['hallucination_verdict'] = 'UNKNOWN'
            signal['hallucination_issues'] = ['Could not retrieve source chunk']
            return signal

        extracted = {
            'rule_text': signal.get('rule_text', ''),
            'signal_category': signal.get('signal_category', ''),
            'entry_conditions': signal.get('entry_conditions', []),
            'parameters': signal.get('parameters', {}),
            'exit_conditions': signal.get('exit_conditions', []),
            'instrument': signal.get('instrument', ''),
            'target_regimes': signal.get('target_regimes', []),
        }

        prompt = HALLUCINATION_CHECK_PROMPT.format(
            original_chunk=full_chunk,
            extracted_rule=json.dumps(extracted, indent=2),
        )

        try:
            result = self.llm.hallucination_check(prompt)
            if result and isinstance(result, dict):
                verdict = result.get('overall_verdict', 'UNKNOWN')
                verdict_map = {
                    'FAITHFUL': 'PASS',
                    'MINOR_ISSUES': 'WARN',
                    'SIGNIFICANT_ISSUES': 'FAIL',
                    'HALLUCINATED': 'FAIL',
                }
                signal['hallucination_verdict'] = verdict_map.get(verdict, 'UNKNOWN')
                signal['hallucination_issues'] = result.get('issues_found', [])
                signal['hallucination_recommendation'] = result.get('recommendation', '')
            else:
                signal['hallucination_verdict'] = 'UNKNOWN'
        except Exception as e:
            logger.warning(f"Sonnet check error: {e}")
            signal['hallucination_verdict'] = 'UNKNOWN'
        return signal

    def sonnet_check(self, signals: List[dict]) -> List[dict]:
        """Run Sonnet hallucination check on all signals. Updates in place."""
        print(f"\n--- Sonnet Hallucination Check: {len(signals)} signals, "
              f"{self.num_workers} workers ---", flush=True)

        total = len(signals)
        done = 0
        start = time.time()
        verdicts = {'PASS': 0, 'WARN': 0, 'FAIL': 0, 'UNKNOWN': 0}

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {executor.submit(self._sonnet_check_one, s): i
                       for i, s in enumerate(signals)}
            for future in as_completed(futures):
                s = future.result()
                v = s.get('hallucination_verdict', 'UNKNOWN')
                verdicts[v] = verdicts.get(v, 0) + 1
                done += 1
                if done % 50 == 0 or done == total:
                    elapsed = time.time() - start
                    rate = done / elapsed if elapsed > 0 else 0
                    print(f"  Sonnet: {done}/{total} ({rate:.1f}/s) | "
                          f"P:{verdicts['PASS']} W:{verdicts['WARN']} "
                          f"F:{verdicts['FAIL']} U:{verdicts['UNKNOWN']}",
                          flush=True)

        # Filter out FAIL
        kept = [s for s in signals if s.get('hallucination_verdict') != 'FAIL']
        print(f"  Sonnet result: {total} -> {len(kept)} "
              f"(dropped {total - len(kept)} FAIL)", flush=True)
        return kept


def run_full_pipeline(json_dir: str = 'extraction_results',
                      output_dir: str = 'extraction_results',
                      num_workers: int = 4):
    """Run complete filter + hallucination pipeline on all JSON files."""
    import glob
    from review.pre_filter import PreReviewFilter

    # Load all signals
    all_signals = []
    for f in sorted(glob.glob(f'{json_dir}/*.json')):
        signals = json.load(open(f))
        all_signals.extend(signals)
        book = os.path.basename(f).replace('.json', '')
        print(f"  Loaded {book}: {len(signals)} signals")
    print(f"\nTotal raw signals: {len(all_signals)}")

    # Step 1-2: Pre-filter (chunk-type + dedup)
    pf = PreReviewFilter()
    filtered = pf.filter(all_signals)

    # Step 3: Haiku pre-screen
    vs = VectorStore('./chromadb_data')
    checker = HallucinationChecker(vs, num_workers=num_workers)
    testable = checker.haiku_prescreen(filtered)

    # Step 4: Sonnet hallucination check on survivors
    checked = checker.sonnet_check(testable)

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    by_book = {}
    for s in checked:
        by_book.setdefault(s['book_id'], []).append(s)

    for book_id, signals in by_book.items():
        path = f'{output_dir}/{book_id}_checked.json'
        with open(path, 'w') as f:
            json.dump(signals, f, indent=2)
        print(f"  Saved {book_id}: {len(signals)} signals -> {path}")

    print(f"\n{'='*50}")
    print(f"PIPELINE COMPLETE")
    print(f"  Raw: {len(all_signals)} -> Pre-filter: {len(filtered)} "
          f"-> Haiku: {len(testable)} -> Sonnet: {len(checked)}")
    print(f"{'='*50}")

    return checked
