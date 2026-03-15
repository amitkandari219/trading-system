"""
Ingest all available book PDFs into ChromaDB.
Step 1 of the RAG pipeline: PDF -> chunks -> embeddings -> ChromaDB.

Usage:
    python ingest_all_books.py [--book BOOK_ID]   # single book
    python ingest_all_books.py                      # all available books
"""

import os
import sys
import time

# Map book_id -> actual PDF filename in books/ directory
BOOK_PDF_MAP = {
    'GUJRAL':    'how-to-make-money-in-intraday-trading-1nbsped-9386268159-9789386268150.docx',
    'KAUFMAN':   'Trading Systems and Methods \u2014 Perry Kaufman.pdf',
    'NATENBERG': 'Options_Volatility_and_Pricing_Sheldon_N.pdf',
    'SINCLAIR':  'Volatility Trading, + Website-Wiley (2013).pdf',
    'GRIMES':    ' The Art and Science of Technical Analysis \u2014 Adam Grimes.pdf',
    'LOPEZ':     'Advances in Financial Machine Learning \u2014 Marcos Lopez de Prado.pdf',
    'HILPISCH':  'Python for Algorithmic Trading \u2014 Yves Hilpisch.pdf',
    'DOUGLAS':   'Trading_in_the_zone.pdf',
    'MCMILLAN':  'options-as-a-strategic-investment.pdf',
    # 'AUGEN':   not available yet
    'HARRIS':    'Trading-Exchanges-Market.pdf',
    # Batch 1
    'BULKOWSKI':  '042_Encyclopedia Of Chart Patterns, 2nd Edition.pdf',
    'CANDLESTICK': 'Encyclopedia_of_candlestick_chart_-_Thomas_NBulkowski.pdf',
    'CHAN_QT':    'Quantitative Trading_ How to Build Your Own Algorithmic Trading Business-Wiley (2008).pdf',
    'CHAN_AT':    'Algorithmic Trading Winning Strategies and their rationale ernest chan.pdf',
    'VINCE':     'Mathematics Of Money Management. Ralph Vince.pdf',
    # Batch 2
    'INTERMARKET': 'Intermarket Analysis.pdf',
    'MICROSTRUCTURE': 'Trading-Exchanges-Market-Microstructure-Practitioners Draft Copy.pdf',
    'POSITIONAL': 'positional-option-trading-an-advanced.pdf',
    'ARONSON':   'Evidence-Based_Technical_Analysis_-_David_Aronson.pdf',
    'ALGO_SUCCESS': 'Successful Algorithmic Trading.pdf',
    # Batch 3
    'KAHNEMAN':  'Daniel Kahneman-Thinking, Fast and Slow  .pdf',
    'TALEB_BS':  'the-black-swan_-the-impact-of-the-highly-improbable-second-edition-pdfdrive.com-.pdf',
    'TALEB_AF':  'Taleb_Antifragile__2012.pdf',
    'THARP':     'Trade_Your_Way_to_Financial_Freedom.pdf',
    'SCHWAGER':  'Market-Wizards.pdf',
    'SOROS':     'alchemy-of-finance-george.pdf',
    'MAJOR_ASSETS': 'Major Assets.PDF',
    # Batch 4
    'GATHERAL':  'Gatheral J. The volatility surface.pdf',
    'KELLY':     'kelly-capital-growth-investment-criterion.pdf',
    'FLASH':     'Flash.pdf',
    'JOHNSON':   'barry-johnson-algorithmic-trading-amp-dma.pdf',
    # NATENBERG already ingested — skip
}

BOOKS_DIR = 'books'
CHROMADB_DIR = 'chromadb_data'


def ingest_book(ingester, vector_store, book_id, pdf_filename):
    """Ingest a single book and store in ChromaDB."""
    pdf_path = os.path.join(BOOKS_DIR, pdf_filename)
    if not os.path.exists(pdf_path):
        print(f"  SKIP: {pdf_path} not found")
        return 0

    print(f"\n{'='*60}")
    print(f"INGESTING: {book_id}")
    print(f"  PDF: {pdf_filename}")
    print(f"{'='*60}")

    start = time.time()
    if pdf_path.endswith('.docx'):
        chunks = ingester.ingest_docx(pdf_path, book_id)
    else:
        chunks = ingester.ingest_book(pdf_path, book_id)
    ingest_time = time.time() - start
    print(f"  Chunks extracted: {len(chunks)} ({ingest_time:.1f}s)")

    if not chunks:
        print(f"  WARNING: No chunks extracted!")
        return 0

    # Show chunk type distribution
    type_counts = {}
    for c in chunks:
        type_counts[c.chunk_type] = type_counts.get(c.chunk_type, 0) + 1
    print(f"  Chunk types: {dict(sorted(type_counts.items()))}")

    # Store in ChromaDB
    start = time.time()
    vector_store.store_chunks(chunks)
    store_time = time.time() - start
    print(f"  Stored in ChromaDB ({store_time:.1f}s)")

    # Update books table chunk_count
    try:
        import psycopg2
        from config.settings import DATABASE_DSN
        conn = psycopg2.connect(DATABASE_DSN)
        cur = conn.cursor()
        cur.execute(
            "UPDATE books SET chunk_count = %s, ingested_at = NOW() WHERE book_id = %s",
            (len(chunks), book_id)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"  DB update skipped: {e}")

    return len(chunks)


def main():
    from ingestion.pdf_ingester import PDFIngester
    from ingestion.vector_store import VectorStore

    # Parse args
    target_book = None
    if len(sys.argv) > 1:
        if sys.argv[1] == '--book' and len(sys.argv) > 2:
            target_book = sys.argv[2].upper()
        else:
            target_book = sys.argv[1].upper()

    ingester = PDFIngester()

    print("Initializing VectorStore (loading embedding model)...")
    vector_store = VectorStore(CHROMADB_DIR)
    print("  Ready.\n")

    if target_book:
        if target_book not in BOOK_PDF_MAP:
            print(f"Unknown book: {target_book}")
            print(f"Available: {', '.join(BOOK_PDF_MAP.keys())}")
            sys.exit(1)
        books_to_process = {target_book: BOOK_PDF_MAP[target_book]}
    else:
        books_to_process = BOOK_PDF_MAP

    total_chunks = 0
    results = {}

    for book_id, pdf_filename in books_to_process.items():
        count = ingest_book(ingester, vector_store, book_id, pdf_filename)
        results[book_id] = count
        total_chunks += count

    print(f"\n{'='*60}")
    print(f"ALL DONE")
    print(f"{'='*60}")
    for book_id, count in results.items():
        status = f"{count} chunks" if count > 0 else "SKIPPED"
        print(f"  {book_id:12s} {status}")
    print(f"\n  Total: {total_chunks} chunks across {sum(1 for c in results.values() if c > 0)} books")


if __name__ == '__main__':
    main()
