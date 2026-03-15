"""
Extraction Orchestrator.
Connects all RAG pipeline components:
  picks the right prompt -> calls LLM -> parses response ->
  runs hallucination check -> creates SignalCandidate(s) -> hands to CLIReviewer.

Supports Ollama (free, local) and Anthropic Claude API.
Set LLM_BACKEND=ollama (default) or LLM_BACKEND=anthropic.

Usage:
    python -m extraction.orchestrator --book GUJRAL
    LLM_BACKEND=anthropic python -m extraction.orchestrator --book GUJRAL
"""

import json
import logging
import os

from ingestion.signal_candidate import (
    SignalCandidate, store_approved_signal, next_signal_id
)
from ingestion.vector_store import VectorStore
from review.cli_reviewer import CLIReviewer
from review.pre_filter import PreReviewFilter
from extraction.llm_client import LLMClient
from extraction.prompts import (
    CONCRETE_BOOK_EXTRACTION_PROMPT,
    PRINCIPLE_BOOK_EXTRACTION_PROMPT,
    PSYCHOLOGY_BOOK_EXTRACTION_PROMPT,
    HALLUCINATION_CHECK_PROMPT,
    BOOK_SPECIFIC_PROMPTS,
)


BOOK_META = {
    'GUJRAL':    {'title': 'How to Make Money in Intraday Trading', 'author': 'Ashwani Gujral'},
    'KAUFMAN':   {'title': 'Trading Systems and Methods',           'author': 'Perry Kaufman'},
    'NATENBERG': {'title': 'Option Volatility and Pricing',         'author': 'Natenberg'},
    'SINCLAIR':  {'title': 'Options Trading',                       'author': 'Euan Sinclair'},
    'GRIMES':    {'title': 'The Art and Science of Technical Analysis', 'author': 'Adam Grimes'},
    'LOPEZ':     {'title': 'Advances in Financial Machine Learning', 'author': 'Lopez de Prado'},
    'HILPISCH':  {'title': 'Python for Finance',                    'author': 'Yves Hilpisch'},
    'DOUGLAS':   {'title': 'Trading in the Zone',                   'author': 'Mark Douglas'},
    'MCMILLAN':  {'title': 'Options as a Strategic Investment',     'author': 'Lawrence McMillan'},
    'AUGEN':     {'title': 'The Volatility Edge in Options Trading','author': 'Jeff Augen'},
    'HARRIS':    {'title': 'Trading and Exchanges',                 'author': 'Larry Harris'},
    'BULKOWSKI': {'title': 'Encyclopedia of Chart Patterns',        'author': 'Thomas Bulkowski'},
    'CANDLESTICK': {'title': 'Encyclopedia of Candlestick Charts',  'author': 'Thomas Bulkowski'},
    'CHAN_QT':   {'title': 'Quantitative Trading',                  'author': 'Ernest Chan'},
    'CHAN_AT':   {'title': 'Algorithmic Trading',                   'author': 'Ernest Chan'},
    'VINCE':     {'title': 'Mathematics of Money Management',       'author': 'Ralph Vince'},
    'INTERMARKET': {'title': 'Intermarket Analysis',                'author': 'John Murphy'},
    'MICROSTRUCTURE': {'title': 'Trading and Exchanges',            'author': 'Larry Harris'},
    'POSITIONAL': {'title': 'Positional Option Trading',            'author': 'Euan Sinclair'},
    'ARONSON':   {'title': 'Evidence-Based Technical Analysis',     'author': 'David Aronson'},
    'ALGO_SUCCESS': {'title': 'Successful Algorithmic Trading',     'author': 'Michael Halls-Moore'},
    'KAHNEMAN':  {'title': 'Thinking Fast and Slow',                'author': 'Daniel Kahneman'},
    'TALEB_BS':  {'title': 'The Black Swan',                        'author': 'Nassim Taleb'},
    'TALEB_AF':  {'title': 'Antifragile',                           'author': 'Nassim Taleb'},
    'THARP':     {'title': 'Trade Your Way to Financial Freedom',   'author': 'Van Tharp'},
    'SCHWAGER':  {'title': 'Market Wizards',                        'author': 'Jack Schwager'},
    'SOROS':     {'title': 'The Alchemy of Finance',                'author': 'George Soros'},
    'MAJOR_ASSETS': {'title': 'Expected Returns on Major Assets',   'author': 'Antti Ilmanen'},
    'GATHERAL':  {'title': 'The Volatility Surface',                'author': 'Jim Gatheral'},
    'KELLY':     {'title': 'Kelly Capital Growth Investment Criterion', 'author': 'Various'},
    'FLASH':     {'title': 'Flash Boys',                            'author': 'Michael Lewis'},
    'JOHNSON':   {'title': 'Algorithmic Trading and DMA',           'author': 'Barry Johnson'},
}


class ExtractionOrchestrator:
    """
    Drives the full extraction pipeline for one book:
      1. Query VectorStore for all chunks
      2. For each chunk: call LLM with the right prompt
      3. Run hallucination check on each extracted rule
      4. Create SignalCandidate(s)
      5. Present to CLIReviewer for human decision
      6. Store approved signals via store_approved_signal()

    CONCRETE books: one SignalCandidate per chunk (if rule_found).
    PRINCIPLE books: up to three SignalCandidates per chunk (variants).
    """

    def __init__(self, db, vector_store: VectorStore, logger=None,
                 llm_backend=None):
        self.db           = db
        self.vector_store = vector_store
        self.logger       = logger or logging.getLogger(__name__)
        self.llm          = LLMClient(backend=llm_backend)
        self.reviewer     = CLIReviewer()
        self.pre_filter   = PreReviewFilter()

    def run_book(self, book_id: str, chapters: list = None,
                 dry_run: bool = False):
        """Extract and review all signals from one book.
        chapters: optional list of chapter numbers to process.
        dry_run: if True, print extracted signals without DB/review.
        """
        abstraction = self.vector_store._get_abstraction(book_id)
        meta        = BOOK_META[book_id]

        # Get all chunks (optionally filtered by chapter)
        if chapters:
            all_chunks = []
            for ch in chapters:
                all_chunks.extend(
                    self.vector_store.get_all_chunks(book_id, chapter=ch)
                )
        else:
            all_chunks = self.vector_store.get_all_chunks(book_id)

        print(f"LLM backend: {self.llm.backend_name}")
        print(f"Book: {book_id} | Chunks: {len(all_chunks)} | "
              f"Abstraction: {abstraction} | Dry run: {dry_run}")

        stats = {'extracted': 0, 'approved': 0, 'rejected': 0,
                 'deferred': 0, 'skipped': 0}

        # Phase 1: Extract all candidates
        all_candidates = []
        for i, chunk_result in enumerate(all_chunks):
            chunk_text = chunk_result['text']
            chunk_meta = chunk_result['metadata']

            print(f"\n[{i+1}/{len(all_chunks)}] Ch.{chunk_meta.get('chapter_number', '?')} "
                  f"type={chunk_meta.get('chunk_type', '?')} "
                  f"({len(chunk_text)} chars)")

            candidates = self._extract_candidates(
                book_id, chunk_text, chunk_meta, meta, abstraction,
                dry_run=dry_run
            )

            for candidate in candidates:
                if candidate is None:
                    stats['skipped'] += 1
                    continue
                stats['extracted'] += 1
                all_candidates.append(candidate)

                if dry_run:
                    print(f"  -> {candidate.signal_category} | "
                          f"{candidate.direction} | "
                          f"{candidate.rule_text[:80]}...")
                    if candidate.entry_conditions:
                        for ec in candidate.entry_conditions[:3]:
                            ec_str = str(ec) if not isinstance(ec, str) else ec
                            print(f"     entry: {ec_str[:80]}")

        # Save extracted signals to JSON (always, for both dry-run and live)
        if all_candidates:
            self._save_signals_json(book_id, all_candidates)

        # Phase 2: Pre-filter + review (non-dry-run only)
        if not dry_run and all_candidates:
            filtered = self.pre_filter.filter(all_candidates)
            print(f"\nProceeding to review {len(filtered)} signals...")

            # Phase 3: Human review
            for candidate in filtered:
                decision = self.reviewer.review_signal(candidate)

                if decision['decision'] in ('A', 'R'):
                    store_approved_signal(self.db, candidate, decision)
                    self.db.connection.commit()
                    self.logger.info(f"Stored: {candidate.signal_id}")
                    stats['approved'] += 1
                elif decision['decision'] == 'X':
                    stats['rejected'] += 1
                else:
                    stats['deferred'] += 1

        print(f"\n{'='*50}")
        print(f"EXTRACTION COMPLETE: {book_id}")
        print(f"  Chunks processed: {len(all_chunks)}")
        print(f"  Signals extracted: {stats['extracted']}")
        print(f"  Skipped (no rule): {stats['skipped']}")
        if all_candidates:
            print(f"  Saved to: extraction_results/{book_id}.json")
        if not dry_run:
            print(f"  Passed pre-filter: {len(filtered) if all_candidates else 0}")
            print(f"  Approved: {stats['approved']}")
            print(f"  Rejected: {stats['rejected']}")
            print(f"  Deferred: {stats['deferred']}")
        print(f"{'='*50}")

    def _save_signals_json(self, book_id: str, candidates: list):
        """Save extracted signals to JSON for later review without re-extraction."""
        os.makedirs('extraction_results', exist_ok=True)
        records = []
        for c in candidates:
            records.append({
                'signal_id': c.signal_id,
                'book_id': c.book_id,
                'source_citation': c.source_citation,
                'raw_chunk_text': c.raw_chunk_text,
                'rule_text': c.rule_text,
                'signal_category': c.signal_category,
                'direction': c.direction,
                'entry_conditions': c.entry_conditions,
                'parameters': c.parameters,
                'exit_conditions': c.exit_conditions,
                'instrument': c.instrument,
                'timeframe': c.timeframe,
                'target_regimes': c.target_regimes,
                'author_confidence': c.author_confidence,
                'hallucination_verdict': c.hallucination_verdict,
                'hallucination_issues': c.hallucination_issues,
                'variant_id': getattr(c, 'variant_id', None),
                'chunk_type': getattr(c, 'chunk_type', None),
            })
        path = f'extraction_results/{book_id}.json'
        with open(path, 'w') as f:
            json.dump(records, f, indent=2)
        print(f"  Saved {len(records)} signals to {path}")

    @staticmethod
    def load_signals_json(book_id: str) -> list:
        """Load previously extracted signals from JSON."""
        path = f'extraction_results/{book_id}.json'
        with open(path) as f:
            records = json.load(f)
        candidates = []
        for r in records:
            c = SignalCandidate(
                signal_id=r['signal_id'],
                book_id=r['book_id'],
                source_citation=r['source_citation'],
                raw_chunk_text=r['raw_chunk_text'],
                rule_text=r['rule_text'],
                signal_category=r['signal_category'],
                direction=r['direction'],
                entry_conditions=r.get('entry_conditions', []),
                parameters=r.get('parameters', {}),
                exit_conditions=r.get('exit_conditions', []),
                instrument=r.get('instrument', 'ANY'),
                timeframe=r.get('timeframe', 'ANY'),
                target_regimes=r.get('target_regimes', ['ANY']),
                author_confidence=r.get('author_confidence', 'MEDIUM'),
            )
            c.hallucination_verdict = r.get('hallucination_verdict', 'UNKNOWN')
            c.hallucination_issues = r.get('hallucination_issues', [])
            c.variant_id = r.get('variant_id')
            c.chunk_type = r.get('chunk_type')
            candidates.append(c)
        return candidates

    def _extract_candidates(self, book_id, chunk_text, chunk_meta,
                             meta, abstraction, dry_run=False) -> list:
        """Returns list of SignalCandidates (0-3 depending on abstraction)."""
        # Book-specific prompts always use concrete extraction path
        if book_id in BOOK_SPECIFIC_PROMPTS:
            return [self._extract_concrete(book_id, chunk_text, chunk_meta, meta,
                                           dry_run=dry_run)]
        if abstraction == 'CONCRETE':
            return [self._extract_concrete(book_id, chunk_text, chunk_meta, meta,
                                           dry_run=dry_run)]
        elif abstraction == 'PSYCHOLOGY':
            return [self._extract_psychology(book_id, chunk_text, chunk_meta, meta,
                                              dry_run=dry_run)]
        elif abstraction in ('PRINCIPLE',):
            return self._extract_principle(book_id, chunk_text, chunk_meta, meta,
                                           dry_run=dry_run)
        else:
            # METHODOLOGY: use principle extraction
            return self._extract_principle(book_id, chunk_text, chunk_meta, meta,
                                           dry_run=dry_run)

    def _extract_concrete(self, book_id, chunk_text, chunk_meta, meta,
                          dry_run=False):
        """Extract one rule from a CONCRETE book chunk."""
        # Use book-specific prompt if available, else default
        prompt_template = BOOK_SPECIFIC_PROMPTS.get(book_id, CONCRETE_BOOK_EXTRACTION_PROMPT)
        prompt = prompt_template.format(
            book_title     = meta['title'],
            author         = meta['author'],
            chapter_title  = chunk_meta.get('chapter_title', ''),
            page_start     = chunk_meta.get('page_start', ''),
            page_end       = chunk_meta.get('page_end', ''),
            chapter_number = chunk_meta.get('chapter_number', ''),
            chunk_type     = chunk_meta.get('chunk_type', 'RULE'),
            chunk_text     = chunk_text,
        )
        response = self.llm.extract(prompt)
        if not isinstance(response, dict) or not response.get('rule_found'):
            return None

        if dry_run:
            candidate = SignalCandidate(
                signal_id=f"{book_id}_DRY_{chunk_meta.get('chapter_number', 0)}",
                book_id=book_id,
                source_citation=response.get('source_citation', ''),
                raw_chunk_text=chunk_text[:200],
                rule_text=response.get('rule_text', ''),
                signal_category=response.get('signal_category', ''),
                direction=response.get('direction', ''),
                entry_conditions=response.get('entry_conditions', []),
                parameters=response.get('parameters', {}),
                exit_conditions=response.get('exit_conditions', []),
                instrument=response.get('instrument', ''),
                timeframe=response.get('timeframe', ''),
                target_regimes=response.get('target_regimes', []),
                author_confidence=response.get('author_confidence', ''),
            )
            candidate.chunk_type = chunk_meta.get('chunk_type', 'RULE')
            return candidate

        signal_id = next_signal_id(self.db, book_id)

        class _Chunk:
            text = chunk_text

        candidate = SignalCandidate.from_claude_response(
            signal_id, book_id, _Chunk(), response
        )
        candidate.chunk_type = chunk_meta.get('chunk_type', 'RULE')
        candidate = self._run_hallucination_check(candidate, chunk_text, response)
        return candidate

    def _extract_psychology(self, book_id, chunk_text, chunk_meta, meta,
                            dry_run=False):
        """Extract one discipline rule from a PSYCHOLOGY book chunk."""
        prompt = PSYCHOLOGY_BOOK_EXTRACTION_PROMPT.format(
            book_title     = meta['title'],
            author         = meta['author'],
            chapter_title  = chunk_meta.get('chapter_title', ''),
            page_start     = chunk_meta.get('page_start', ''),
            page_end       = chunk_meta.get('page_end', ''),
            chapter_number = chunk_meta.get('chapter_number', ''),
            chunk_type     = chunk_meta.get('chunk_type', 'PSYCHOLOGY'),
            chunk_text     = chunk_text,
        )
        response = self.llm.extract(prompt)
        if not isinstance(response, dict) or not response.get('rule_found'):
            return None

        if dry_run:
            candidate = SignalCandidate(
                signal_id=f"{book_id}_DRY_{chunk_meta.get('chapter_number', 0)}",
                book_id=book_id,
                source_citation=response.get('source_citation', ''),
                raw_chunk_text=chunk_text[:200],
                rule_text=response.get('rule_text', ''),
                signal_category=response.get('signal_category', ''),
                direction=response.get('direction', 'NEUTRAL'),
                entry_conditions=response.get('entry_conditions', []),
                parameters=response.get('parameters', {}),
                exit_conditions=response.get('exit_conditions', []),
                instrument=response.get('instrument', 'ANY'),
                timeframe=response.get('timeframe', 'ANY'),
                target_regimes=response.get('target_regimes', ['ANY']),
                author_confidence=response.get('author_confidence', ''),
            )
            candidate.chunk_type = chunk_meta.get('chunk_type', 'PSYCHOLOGY')
            return candidate

        signal_id = next_signal_id(self.db, book_id)

        class _Chunk:
            text = chunk_text

        candidate = SignalCandidate.from_claude_response(
            signal_id, book_id, _Chunk(), response
        )
        candidate.chunk_type = chunk_meta.get('chunk_type', 'PSYCHOLOGY')
        candidate = self._run_hallucination_check(candidate, chunk_text, response)
        return candidate

    def _extract_principle(self, book_id, chunk_text, chunk_meta, meta,
                           dry_run=False) -> list:
        """Extract up to 3 variant SignalCandidates from a PRINCIPLE book chunk."""
        prompt = PRINCIPLE_BOOK_EXTRACTION_PROMPT.format(
            book_title     = meta['title'],
            author         = meta['author'],
            chapter_title  = chunk_meta.get('chapter_title', ''),
            page_start     = chunk_meta.get('page_start', ''),
            page_end       = chunk_meta.get('page_end', ''),
            chapter_number = chunk_meta.get('chapter_number', ''),
            chunk_type     = chunk_meta.get('chunk_type', 'RULE'),
            chunk_text     = chunk_text,
        )
        response = self.llm.extract(prompt)
        if response is None or not response.get('principle_found'):
            return []

        candidates = []
        for vi, variant in enumerate(response.get('variants', [])):
            flat = {
                'rule_text':         response.get('principle_text', ''),
                'signal_category':   response.get('signal_category', 'PATTERN'),
                'direction':         variant.get('direction', 'CONTEXT_DEPENDENT'),
                'entry_conditions':  variant.get('entry_conditions', []),
                'parameters':        variant.get('parameters', {}),
                'exit_conditions':   variant.get('exit_conditions', []),
                'instrument':        variant.get('instrument', 'ANY'),
                'timeframe':         'ANY',
                'target_regimes':    variant.get('target_regimes', ['ANY']),
                'source_citation':   response.get('source_citation', ''),
                'completeness_warning': response.get('what_author_leaves_unspecified'),
                'author_confidence': 'MEDIUM',
            }

            if dry_run:
                candidate = SignalCandidate(
                    signal_id=f"{book_id}_DRY_{chunk_meta.get('chapter_number', 0)}_{vi}",
                    book_id=book_id,
                    source_citation=flat['source_citation'],
                    raw_chunk_text=chunk_text[:200],
                    rule_text=flat['rule_text'],
                    signal_category=flat['signal_category'],
                    direction=flat['direction'],
                    entry_conditions=flat['entry_conditions'],
                    parameters=flat['parameters'],
                    exit_conditions=flat['exit_conditions'],
                    instrument=flat['instrument'],
                    timeframe=flat['timeframe'],
                    target_regimes=flat['target_regimes'],
                    author_confidence=flat['author_confidence'],
                )
                candidate.variant_id = variant.get('variant_id')
                candidate.chunk_type = chunk_meta.get('chunk_type', 'RULE')
                candidates.append(candidate)
                continue

            signal_id = next_signal_id(self.db, book_id)

            class _Chunk:
                text = chunk_text

            candidate = SignalCandidate.from_claude_response(
                signal_id, book_id, _Chunk(), flat
            )
            candidate.variant_id = variant.get('variant_id')
            candidate.chunk_type = chunk_meta.get('chunk_type', 'RULE')
            candidate = self._run_hallucination_check(candidate, chunk_text, flat)
            candidates.append(candidate)

        return candidates

    def _run_hallucination_check(self, candidate: SignalCandidate,
                                  original_chunk: str, extracted: dict):
        """Run hallucination check and populate candidate verdict."""
        check_prompt = HALLUCINATION_CHECK_PROMPT.format(
            original_chunk = original_chunk,
            extracted_rule = json.dumps(extracted, indent=2),
        )
        result = self.llm.hallucination_check(check_prompt)
        if result:
            verdict = result.get('overall_verdict', 'UNKNOWN')
            verdict_map = {
                'FAITHFUL':           'PASS',
                'MINOR_ISSUES':       'WARN',
                'SIGNIFICANT_ISSUES': 'FAIL',
                'HALLUCINATED':       'FAIL',
            }
            candidate.hallucination_verdict = verdict_map.get(verdict, 'UNKNOWN')
            candidate.hallucination_issues  = result.get('issues_found', [])
        return candidate


# ================================================================
# CLI ENTRY POINT
# ================================================================
if __name__ == '__main__':
    import argparse
    import psycopg2
    from config.settings import DATABASE_DSN

    parser = argparse.ArgumentParser(description='Extract signals from a book')
    parser.add_argument('--book', required=True,
                       choices=BOOK_META.keys(),
                       help='Book ID to extract from')
    parser.add_argument('--chapters', type=int, nargs='+', default=None,
                       help='Only process specific chapter numbers (e.g. --chapters 2 3)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Print extracted signals without DB or review')
    parser.add_argument('--chromadb-dir', default='./chromadb_data',
                       help='ChromaDB persistence directory')
    parser.add_argument('--backend', default=None,
                       choices=['ollama', 'anthropic'],
                       help='LLM backend (default: from LLM_BACKEND env, or ollama)')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('extraction')

    vector_store = VectorStore(args.chromadb_dir)

    if args.dry_run:
        # No DB needed for dry run
        orchestrator = ExtractionOrchestrator(
            None, vector_store, logger, llm_backend=args.backend
        )
        orchestrator.run_book(args.book, chapters=args.chapters,
                              dry_run=True)
    else:
        conn = psycopg2.connect(DATABASE_DSN)
        conn.autocommit = False
        cursor = conn.cursor()

        orchestrator = ExtractionOrchestrator(
            cursor, vector_store, logger, llm_backend=args.backend
        )

        try:
            orchestrator.run_book(args.book, chapters=args.chapters)
            conn.commit()
        except KeyboardInterrupt:
            print("\nInterrupted. Committing approved signals so far...")
            conn.commit()
        finally:
            conn.close()
