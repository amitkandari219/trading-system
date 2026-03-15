"""
ChromaDB vector storage for book chunks.
One collection per book. Never query across all books simultaneously.
Cross-book synthesis happens at Claude API layer.

Embedding model: all-mpnet-base-v2 (768-dim, ~70ms/chunk on CPU).
"""

from typing import List, Optional


class VectorStore:
    """
    One ChromaDB collection per book.
    Never query across all books simultaneously.
    Cross-book synthesis happens at Claude API layer.
    """

    EMBEDDING_MODEL = 'all-mpnet-base-v2'

    ABSTRACTION_LEVELS = {
        'GUJRAL':      'CONCRETE',
        'KAUFMAN':     'CONCRETE',
        'MCMILLAN':    'CONCRETE',
        'AUGEN':       'CONCRETE',
        'BULKOWSKI':   'CONCRETE',    # Encyclopedia of Chart Patterns — concrete pattern rules
        'CANDLESTICK': 'CONCRETE',    # Encyclopedia of Candlestick Charts — concrete patterns
        'CHAN_QT':     'METHODOLOGY',  # Quantitative Trading — systematic strategies
        'CHAN_AT':     'METHODOLOGY',  # Algorithmic Trading — execution & mean reversion
        'VINCE':       'PRINCIPLE',    # Mathematics of Money Management — position sizing
        'NATENBERG':   'PRINCIPLE',
        'SINCLAIR':    'PRINCIPLE',
        'GRIMES':      'METHODOLOGY',
        'LOPEZ':       'METHODOLOGY',
        'HILPISCH':    'METHODOLOGY',
        'HARRIS':      'METHODOLOGY',
        'DOUGLAS':     'PSYCHOLOGY',
    }

    def __init__(self, persist_dir: str):
        import chromadb
        from sentence_transformers import SentenceTransformer

        self.client = chromadb.PersistentClient(path=persist_dir)
        self.encoder = SentenceTransformer(self.EMBEDDING_MODEL)
        self._collections = {}

    def get_collection(self, book_id: str):
        """Get or create collection for a book."""
        if book_id not in self._collections:
            self._collections[book_id] = \
                self.client.get_or_create_collection(
                    name=f"book_{book_id.lower()}",
                    metadata={"hnsw:space": "cosine"}
                )
        return self._collections[book_id]

    def store_chunks(self, chunks: list, batch_size: int = 50):
        """Store RawChunk list in their book's collection. Batched for memory efficiency."""
        by_book = {}
        for chunk in chunks:
            by_book.setdefault(chunk.book_id, []).append(chunk)

        for book_id, book_chunks in by_book.items():
            collection = self.get_collection(book_id)

            # Process in batches to avoid OOM on large books
            for start in range(0, len(book_chunks), batch_size):
                batch = book_chunks[start:start + batch_size]

                texts = [c.text for c in batch]
                embeddings = self.encoder.encode(
                    texts, batch_size=32, show_progress_bar=True
                ).tolist()

                ids = [
                    f"{c.book_id}_{c.chapter_number}_{start + i}"
                    for i, c in enumerate(batch)
                ]

                metadatas = [{
                    'book_id': c.book_id,
                    'chapter_number': c.chapter_number,
                    'chapter_title': c.chapter_title,
                    'page_start': c.page_start,
                    'page_end': c.page_end,
                    'chunk_type': c.chunk_type,
                    'has_table': c.has_table,
                    'has_formula': c.has_formula,
                    'abstraction_level': self._get_abstraction(c.book_id),
                } for c in batch]

                collection.upsert(
                    ids=ids,
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas
                )

            print(f"  Stored {len(book_chunks)} chunks for {book_id}")

    def get_all_chunks(self, book_id: str,
                      chapter: Optional[int] = None,
                      chunk_types: Optional[List[str]] = None) -> List[dict]:
        """Return ALL chunks for a book (no embedding query, no MMR).
        Optionally filter by chapter number and/or chunk types."""
        collection = self.get_collection(book_id)

        where_filter = {}
        conditions = []
        if chapter is not None:
            conditions.append({"chapter_number": {"$eq": chapter}})
        if chunk_types:
            conditions.append({"chunk_type": {"$in": chunk_types}})

        if len(conditions) == 1:
            where_filter = conditions[0]
        elif len(conditions) > 1:
            where_filter = {"$and": conditions}

        results = collection.get(
            where=where_filter if where_filter else None,
            include=['documents', 'metadatas']
        )

        out = []
        if results and results['ids']:
            for i, doc_id in enumerate(results['ids']):
                out.append({
                    'text': results['documents'][i],
                    'metadata': results['metadatas'][i],
                    'id': doc_id,
                })
        return out

    def query_book(self, book_id: str, query: str,
                   n_results: int = 8,
                   chunk_types: Optional[List[str]] = None) -> List[dict]:
        """
        Query a specific book's collection.
        Returns list of dicts with keys: text, metadata, relevance.
        """
        collection = self.get_collection(book_id)
        query_embedding = self.encoder.encode([query]).tolist()

        where_filter = {}
        if chunk_types:
            where_filter = {"chunk_type": {"$in": chunk_types}}

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            where=where_filter if where_filter else None,
            include=['documents', 'metadatas', 'distances']
        )

        return self._mmr_rerank(results, lambda_param=0.6)

    def _mmr_rerank(self, results, lambda_param=0.6, n_final=5):
        """
        Maximal Marginal Relevance reranking.
        Reduces redundant chunks from repeated concepts across chapters.
        lambda_param: 0=max diversity, 1=max relevance. 0.6 balances both.
        """
        if not results['ids'][0]:
            return []

        docs      = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]

        similarities = [1 - d for d in distances]

        if len(docs) <= 1:
            return [{'text': docs[0], 'metadata': metadatas[0],
                     'relevance': similarities[0]}]

        # Inter-document similarity via bigram overlap
        n = len(docs)
        inter_sim = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                sim = self._text_similarity(docs[i], docs[j])
                inter_sim[i][j] = sim
                inter_sim[j][i] = sim

        # MMR greedy selection
        selected_indices = []
        candidate_indices = list(range(n))

        best_idx = max(candidate_indices, key=lambda i: similarities[i])
        selected_indices.append(best_idx)
        candidate_indices.remove(best_idx)

        while (len(selected_indices) < min(n_final, n)
               and candidate_indices):
            mmr_scores = {}
            for idx in candidate_indices:
                relevance  = similarities[idx]
                redundancy = max(inter_sim[idx][s]
                                 for s in selected_indices)
                mmr_scores[idx] = (lambda_param * relevance
                                   - (1 - lambda_param) * redundancy)

            next_idx = max(mmr_scores, key=mmr_scores.get)
            selected_indices.append(next_idx)
            candidate_indices.remove(next_idx)

        return [
            {
                'text':      docs[i],
                'metadata':  metadatas[i],
                'relevance': similarities[i]
            }
            for i in selected_indices
        ]

    def _text_similarity(self, text_a: str, text_b: str) -> float:
        """Bigram overlap Jaccard similarity."""
        def bigrams(text):
            words = text.lower().split()
            return set(zip(words[:-1], words[1:]))
        bg_a = bigrams(text_a)
        bg_b = bigrams(text_b)
        if not bg_a or not bg_b:
            return 0.0
        return len(bg_a & bg_b) / len(bg_a | bg_b)

    def _get_abstraction(self, book_id: str) -> str:
        return self.ABSTRACTION_LEVELS.get(book_id, 'UNKNOWN')
