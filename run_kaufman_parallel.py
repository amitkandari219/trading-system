"""Run Kaufman extraction with 4 parallel workers on chunk batches."""
import logging
import time
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('chromadb').setLevel(logging.WARNING)
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
logging.basicConfig(level=logging.WARNING)

from extraction.orchestrator import ExtractionOrchestrator, BOOK_META
from ingestion.vector_store import VectorStore

BOOK_ID = 'KAUFMAN'
NUM_WORKERS = 6

vs = VectorStore('./chromadb_data')

# Get all chunks once
all_chunks = vs.get_all_chunks(BOOK_ID)
total = len(all_chunks)
print(f"KAUFMAN: {total} chunks, splitting into {NUM_WORKERS} batches", flush=True)

# Split into batches
batch_size = (total + NUM_WORKERS - 1) // NUM_WORKERS
batches = [all_chunks[i:i+batch_size] for i in range(0, total, batch_size)]

# Each worker processes a batch
all_candidates = [None] * NUM_WORKERS

def process_batch(batch_idx, chunks):
    orch = ExtractionOrchestrator(None, vs)
    meta = BOOK_META[BOOK_ID]
    abstraction = vs._get_abstraction(BOOK_ID)
    candidates = []
    for i, chunk_result in enumerate(chunks):
        chunk_text = chunk_result['text']
        chunk_meta = chunk_result['metadata']
        global_idx = batch_idx * batch_size + i + 1
        print(f"[{global_idx}/{total}] Ch.{chunk_meta.get('chapter_number', '?')} "
              f"type={chunk_meta.get('chunk_type', '?')} "
              f"({len(chunk_text)} chars)", flush=True)

        result = orch._extract_candidates(
            BOOK_ID, chunk_text, chunk_meta, meta, abstraction, dry_run=True
        )
        for c in result:
            if c is not None:
                candidates.append(c)
    return batch_idx, candidates

start = time.time()

with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
    futures = {executor.submit(process_batch, i, b): i for i, b in enumerate(batches)}
    for future in as_completed(futures):
        idx, candidates = future.result()
        all_candidates[idx] = candidates
        print(f"\n*** Batch {idx} done: {len(candidates)} signals ***", flush=True)

# Merge all candidates
merged = []
for batch in all_candidates:
    if batch:
        merged.extend(batch)

# Save JSON
os.makedirs('extraction_results', exist_ok=True)
records = []
for c in merged:
    records.append({
        'signal_id': c.signal_id, 'book_id': c.book_id,
        'source_citation': c.source_citation, 'raw_chunk_text': c.raw_chunk_text,
        'rule_text': c.rule_text, 'signal_category': c.signal_category,
        'direction': c.direction, 'entry_conditions': c.entry_conditions,
        'parameters': c.parameters, 'exit_conditions': c.exit_conditions,
        'instrument': c.instrument, 'timeframe': c.timeframe,
        'target_regimes': c.target_regimes, 'author_confidence': c.author_confidence,
        'hallucination_verdict': getattr(c, 'hallucination_verdict', None),
        'hallucination_issues': getattr(c, 'hallucination_issues', None),
        'variant_id': getattr(c, 'variant_id', None),
        'chunk_type': getattr(c, 'chunk_type', None),
    })

path = f'extraction_results/{BOOK_ID}.json'
with open(path, 'w') as f:
    json.dump(records, f, indent=2)

elapsed = time.time() - start
print(f"\nSaved {len(records)} signals to {path}")
print(f"*** KAUFMAN DONE in {elapsed:.0f}s ***", flush=True)
