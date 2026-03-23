"""
Ingest 32 SSRN papers into ChromaDB and extract signals.

SSRN papers are short (10-30 pages each), so we:
1. Ingest all PDFs into ChromaDB (one collection per paper)
2. Run Haiku extraction on each paper's chunks
3. DSL translate the extracted signals
4. Save to dsl_results/PASS/ for walk-forward testing

Usage:
    python run_ssrn_ingest.py                  # full pipeline
    python run_ssrn_ingest.py --ingest-only    # just ChromaDB ingest
    python run_ssrn_ingest.py --extract-only   # just signal extraction (assumes ingested)
    python run_ssrn_ingest.py --dry-run        # preview without saving
"""

import argparse
import glob
import json
import os
import re
import time
from collections import Counter

from ingestion.pdf_ingester import PDFIngester, RawChunk
from ingestion.vector_store import VectorStore
from extraction.llm_client import LLMClient
from extraction.dsl_translator import DSLTranslator
from extraction.dsl_validator import DSLValidator
from extraction.dsl_to_backtest import DSLToBacktest

SSRN_DIR = 'data/papers/ssrn'
CHROMA_DIR = 'chroma_db'
OUTPUT_DIR = 'dsl_results'
ALL_SIGNALS_PATH = 'extraction_results/approved/_ALL.json'

# Map SSRN IDs to readable names and categories
SSRN_CATALOG = {
    '1150080': {'name': 'January Anomaly India', 'category': 'ANOMALY'},
    '1590784': {'name': 'Futures Expiration NSE', 'category': 'EVENT'},
    '1821643': {'name': 'Sharpe Ratio Frontier', 'category': 'METHODOLOGY'},
    '2049939': {'name': 'Betting Against Beta', 'category': 'FACTOR'},
    '2089463': {'name': 'Time Series Momentum', 'category': 'MOMENTUM'},
    '227016':  {'name': 'Contrarian Investment', 'category': 'REVERSAL'},
    '2284643': {'name': 'Low Volatility Portfolios', 'category': 'FACTOR'},
    '2308659': {'name': 'Pseudo-Mathematics Charlatanism', 'category': 'METHODOLOGY'},
    '2326253': {'name': 'Backtest Overfitting', 'category': 'METHODOLOGY'},
    '236939':  {'name': 'Jegadeesh Momentum', 'category': 'MOMENTUM'},
    '2435323': {'name': 'Momentum Investing AQR', 'category': 'MOMENTUM'},
    '2460551': {'name': 'Deflated Sharpe Ratio', 'category': 'METHODOLOGY'},
    '2659431': {'name': 'Volatility Managed Portfolios', 'category': 'VOLATILITY'},
    '2731886': {'name': 'Backtest Overfitting Markets', 'category': 'METHODOLOGY'},
    '2785541': {'name': 'Contrarian Momentum India', 'category': 'MOMENTUM'},
    '2785553': {'name': 'Momentum Contrarian India', 'category': 'MOMENTUM'},
    '2934020': {'name': 'Taming Factor Zoo', 'category': 'FACTOR'},
    '299107':  {'name': 'Jegadeesh Reversal', 'category': 'REVERSAL'},
    '3000963': {'name': 'Nifty Futures Rollover', 'category': 'EVENT'},
    '3065621': {'name': 'Weekly Momentum India', 'category': 'MOMENTUM'},
    '3221798': {'name': 'False Strategy Theorem', 'category': 'METHODOLOGY'},
    '3323746': {'name': 'Options CCI Nifty', 'category': 'OPTIONS'},
    '3365271': {'name': 'Ten Apps Financial ML', 'category': 'METHODOLOGY'},
    '3442749': {'name': 'Volatility Effect Revisited', 'category': 'VOLATILITY'},
    '3510433': {'name': 'Systematic Momentum India', 'category': 'MOMENTUM'},
    '368980':  {'name': 'Option Volume Information', 'category': 'OPTIONS'},
    '4766370': {'name': 'Options Selling ML', 'category': 'OPTIONS'},
    '5116091': {'name': 'Price Momentum India', 'category': 'MOMENTUM'},
    '5367997': {'name': 'Monthly High Timing NIFTY', 'category': 'ANOMALY'},
    '622869':  {'name': 'Option Volume Stock Prices', 'category': 'OPTIONS'},
    '641702':  {'name': 'Hedge Fund Loss Potential', 'category': 'RISK'},
    '980865':  {'name': 'Volatility Effect', 'category': 'VOLATILITY'},
}

# Papers that are methodology-only (no tradeable signals to extract)
METHODOLOGY_ONLY = {'1821643', '2308659', '2326253', '2460551', '2731886', '3221798', '3365271', '2934020'}

SSRN_EXTRACTION_PROMPT = """You are extracting trading signals from an academic research paper.

PAPER: {paper_name} (SSRN-{ssrn_id})
CATEGORY: {category}

TEXT:
{chunk_text}

Extract ANY concrete, backtestable trading rules. Look for:
- Entry conditions with specific thresholds (e.g., "buy when momentum > 0", "go long top decile")
- Exit conditions or holding periods
- Portfolio construction rules with specific parameters
- Anomaly descriptions with dates, frequencies, or thresholds

For each rule found, return a JSON object:
{{
  "signal_id": "SSRN_{ssrn_id}_{{N}}",
  "book_id": "SSRN_{ssrn_id}",
  "rule_text": "one-line description",
  "signal_category": "TREND|REVERSION|VOLATILITY|PATTERN|EVENT|MOMENTUM|FACTOR",
  "direction": "LONG|SHORT|BOTH",
  "entry_conditions": ["condition 1", "condition 2"],
  "exit_conditions": ["condition 1"],
  "parameters": {{"lookback": 20, "threshold": 0.5}},
  "confidence": "HIGH|MEDIUM|LOW",
  "backtestable": "YES|NO",
  "raw_chunk_text": "first 200 chars of source"
}}

If the text contains NO backtestable rules (just theory, proofs, lit review), return:
{{"no_rules": true, "reason": "description"}}

Return ONLY valid JSON. One object per rule found, or the no_rules object."""


def extract_ssrn_id(filename):
    """Extract SSRN ID from filename like ssrn-1234567.pdf"""
    m = re.search(r'ssrn-(\d+)', filename)
    return m.group(1) if m else None


def ingest_papers(dry_run=False):
    """Ingest all SSRN PDFs into ChromaDB."""
    print("=" * 60)
    print("  PHASE 1: INGEST SSRN PAPERS INTO CHROMADB")
    print("=" * 60)

    ingester = PDFIngester()

    # Default profile for academic papers
    ssrn_profile = {
        'abstraction_level': 'METHODOLOGY',
        'skip_pages': [1],  # title page only
        'chapter_pattern': r'^\d+\.\s+',  # section numbers
        'summary_marker': 'Conclusion',
        'table_heavy': True,
        'formula_heavy': True,
    }

    pdf_files = sorted(glob.glob(os.path.join(SSRN_DIR, 'ssrn-*.pdf')))
    print(f"Found {len(pdf_files)} SSRN PDFs")

    all_chunks = []
    for pdf_path in pdf_files:
        ssrn_id = extract_ssrn_id(pdf_path)
        if not ssrn_id:
            continue

        book_id = f"SSRN_{ssrn_id}"
        catalog = SSRN_CATALOG.get(ssrn_id, {})
        name = catalog.get('name', f'SSRN-{ssrn_id}')

        try:
            chunks = ingester.ingest_pdf(
                pdf_path, book_id,
                profile=ssrn_profile,
                chunk_size=500,
                overlap=100,
            )
            all_chunks.extend(chunks)
            print(f"  {book_id:<20s} {len(chunks):>4d} chunks  ({name})")
        except Exception as e:
            print(f"  {book_id:<20s} ERROR: {e}")

    print(f"\nTotal chunks: {len(all_chunks)}")

    if not dry_run and all_chunks:
        print("Storing in ChromaDB...", flush=True)
        vs = VectorStore(CHROMA_DIR)
        vs.store_chunks(all_chunks)
        print(f"  Stored {len(all_chunks)} chunks in ChromaDB")

    return all_chunks


def extract_signals(dry_run=False):
    """Extract signals from ingested SSRN papers via Haiku."""
    print("\n" + "=" * 60)
    print("  PHASE 2: EXTRACT SIGNALS FROM SSRN PAPERS")
    print("=" * 60)

    llm = LLMClient()
    all_signals = []

    pdf_files = sorted(glob.glob(os.path.join(SSRN_DIR, 'ssrn-*.pdf')))

    for pdf_path in pdf_files:
        ssrn_id = extract_ssrn_id(pdf_path)
        if not ssrn_id:
            continue

        # Skip methodology-only papers
        if ssrn_id in METHODOLOGY_ONLY:
            print(f"  SSRN-{ssrn_id}: SKIP (methodology only, no tradeable signals)")
            continue

        catalog = SSRN_CATALOG.get(ssrn_id, {})
        name = catalog.get('name', f'SSRN-{ssrn_id}')
        category = catalog.get('category', 'UNKNOWN')
        book_id = f"SSRN_{ssrn_id}"

        # Read PDF directly (short papers, no need for ChromaDB query)
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                # Extract text from all pages, chunk into ~500 token blocks
                full_text = ''
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        full_text += text + '\n'

            if len(full_text) < 200:
                print(f"  SSRN-{ssrn_id}: SKIP (too short, {len(full_text)} chars)")
                continue

            # Split into chunks (~2000 chars each)
            chunks = []
            for i in range(0, len(full_text), 1800):
                chunk = full_text[i:i+2000]
                if len(chunk.strip()) > 100:
                    chunks.append(chunk)

        except Exception as e:
            print(f"  SSRN-{ssrn_id}: ERROR reading PDF: {e}")
            continue

        print(f"  SSRN-{ssrn_id} ({name}): {len(chunks)} chunks, extracting...", end='', flush=True)

        paper_signals = []
        signal_counter = 0

        for chunk in chunks:
            prompt = SSRN_EXTRACTION_PROMPT.format(
                paper_name=name,
                ssrn_id=ssrn_id,
                category=category,
                chunk_text=chunk[:2000],
            )

            try:
                result = llm._call_anthropic(prompt, 'claude-haiku-4-5-20251001')
            except Exception as e:
                continue

            if not result or not isinstance(result, dict):
                continue

            if result.get('no_rules'):
                continue

            # Could be a single signal or we got one back
            signal_counter += 1
            result['signal_id'] = f"SSRN_{ssrn_id}_{signal_counter}"
            result['book_id'] = book_id
            result['raw_chunk_text'] = chunk[:300]
            paper_signals.append(result)

            time.sleep(0.5)  # rate limit

        print(f" {len(paper_signals)} signals")
        all_signals.extend(paper_signals)

    print(f"\nTotal SSRN signals extracted: {len(all_signals)}")

    # Save extracted signals
    if not dry_run and all_signals:
        os.makedirs('extraction_results', exist_ok=True)
        with open('extraction_results/ssrn_signals.json', 'w') as f:
            json.dump(all_signals, f, indent=2)
        print(f"  Saved to extraction_results/ssrn_signals.json")

        # Also append to _ALL.json
        if os.path.exists(ALL_SIGNALS_PATH):
            with open(ALL_SIGNALS_PATH) as f:
                existing = json.load(f)
            # Remove old SSRN signals
            existing = [s for s in existing if not s.get('book_id', '').startswith('SSRN_')]
            existing.extend(all_signals)
            with open(ALL_SIGNALS_PATH, 'w') as f:
                json.dump(existing, f, indent=2)
            print(f"  Appended {len(all_signals)} SSRN signals to {ALL_SIGNALS_PATH}")

    return all_signals


def translate_signals(signals, dry_run=False):
    """DSL translate extracted SSRN signals."""
    print("\n" + "=" * 60)
    print("  PHASE 3: DSL TRANSLATE SSRN SIGNALS")
    print("=" * 60)

    translator = DSLTranslator(num_workers=4)
    validator = DSLValidator()
    compiler = DSLToBacktest()

    tradeable = [s for s in signals if s.get('backtestable') == 'YES']
    print(f"Tradeable signals: {len(tradeable)} of {len(signals)}")

    if not tradeable:
        print("No tradeable signals to translate.")
        return

    for subdir in ['PASS', 'UNTRANSLATABLE', 'FAIL']:
        os.makedirs(os.path.join(OUTPUT_DIR, subdir), exist_ok=True)

    results = {'pass': 0, 'untranslatable': 0, 'fail': 0}
    start = time.time()

    for i, signal in enumerate(tradeable):
        sid = signal['signal_id']
        source_chunk = signal.get('raw_chunk_text', '')

        rule = translator.translate(signal, source_chunk)

        if rule.untranslatable:
            results['untranslatable'] += 1
            if not dry_run:
                path = os.path.join(OUTPUT_DIR, 'UNTRANSLATABLE', f'{sid}_v1.json')
                with open(path, 'w') as f:
                    json.dump(rule.to_dict(), f, indent=2)
            continue

        vresult = validator.validate(rule)
        if not vresult.passed:
            results['fail'] += 1
            continue

        try:
            compiled = compiler.compile(rule)
        except Exception:
            results['fail'] += 1
            continue

        results['pass'] += 1
        if not dry_run:
            path = os.path.join(OUTPUT_DIR, 'PASS', f'{sid}_v1.json')
            with open(path, 'w') as f:
                json.dump({
                    'signal_id': sid,
                    'book_id': signal.get('book_id', ''),
                    'dsl_rule': rule.to_dict(),
                    'backtest_rule': compiled,
                    'source_rule_text': signal.get('rule_text', ''),
                }, f, indent=2)

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(tradeable)}] PASS:{results['pass']} "
                  f"UT:{results['untranslatable']} FAIL:{results['fail']}", flush=True)

    elapsed = time.time() - start
    print(f"\nSSRN DSL TRANSLATION COMPLETE ({elapsed:.0f}s)")
    print(f"  PASS:           {results['pass']}")
    print(f"  UNTRANSLATABLE: {results['untranslatable']}")
    print(f"  FAIL:           {results['fail']}")


def main():
    parser = argparse.ArgumentParser(description='SSRN paper ingestion and extraction')
    parser.add_argument('--ingest-only', action='store_true')
    parser.add_argument('--extract-only', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    if args.ingest_only:
        ingest_papers(dry_run=args.dry_run)
    elif args.extract_only:
        signals = extract_signals(dry_run=args.dry_run)
        if signals and not args.dry_run:
            translate_signals(signals, dry_run=args.dry_run)
    else:
        # Full pipeline
        ingest_papers(dry_run=args.dry_run)
        signals = extract_signals(dry_run=args.dry_run)
        if signals and not args.dry_run:
            translate_signals(signals, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
