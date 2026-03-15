"""Run dry-run extraction for remaining + re-run books with 4 parallel workers."""
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('chromadb').setLevel(logging.WARNING)
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
logging.basicConfig(level=logging.WARNING)

from extraction.orchestrator import ExtractionOrchestrator, BOOK_META
from ingestion.vector_store import VectorStore

# Remaining: never completed + re-runs (Douglas with new PSYCHOLOGY prompt, Hilpisch/Lopez credits died)
BOOKS = ['DOUGLAS', 'HILPISCH', 'LOPEZ', 'GRIMES', 'NATENBERG', 'KAUFMAN', 'MCMILLAN']

vs = VectorStore('./chromadb_data')

def run_book(book):
    start = time.time()
    orch = ExtractionOrchestrator(None, vs)
    orch.run_book(book, dry_run=True)
    elapsed = time.time() - start
    return book, elapsed

total_start = time.time()
results = {}

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = {executor.submit(run_book, book): book for book in BOOKS}
    for future in as_completed(futures):
        book, elapsed = future.result()
        results[book] = elapsed
        print(f"\n*** {book} DONE in {elapsed:.0f}s ***", flush=True)

total = time.time() - total_start
print(f"\n\n{'='*60}")
print(f"ALL BOOKS COMPLETE — {total:.0f}s total")
print(f"{'='*60}")
for book in BOOKS:
    if book in results:
        print(f"  {book}: {results[book]:.0f}s")
