"""
Pre-review filter pipeline for extracted signals.

Filter pipeline (applied in order):
  1. Keep ALL FORMULA signals regardless of other criteria
  2. Drop signals with confidence_score < 0.6
  3. Chunk-type filter: drop PSYCHOLOGY/METHODOLOGY signals with price-based entries
  4. Drop near-duplicate signals within same book (bigram similarity > 0.5)
"""

import re
import logging
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)

# Books by abstraction level
METHODOLOGY_BOOKS = {'HARRIS', 'GRIMES', 'LOPEZ', 'HILPISCH'}
PSYCHOLOGY_BOOKS = {'DOUGLAS'}

# Pattern to detect price-based / indicator entries that shouldn't appear in
# PSYCHOLOGY or METHODOLOGY-sourced PSYCHOLOGY/EMPIRICAL/SUMMARY chunks
PRICE_INDICATOR_RE = re.compile(
    r'(MA\b|RSI|ATR|MACD|EMA\b|SMA\b|Bollinger|stochastic|'
    r'volume\s*>=|price\s+(?:above|below|crosses)|moving\s+average|'
    r'\d+[- ]?(?:day|period|bar))',
    re.IGNORECASE
)


def _get(signal, field, default=None):
    """Access signal field whether it's an object or dict."""
    if isinstance(signal, dict):
        return signal.get(field, default)
    return getattr(signal, field, default)


# Map author_confidence + hallucination_verdict to a 0-1 score
CONFIDENCE_MAP = {
    'HIGH': 0.9,
    'MEDIUM': 0.6,
    'LOW': 0.3,
}

HALLUCINATION_MODIFIER = {
    'PASS': 0.0,
    'WARN': -0.15,
    'FAIL': -0.4,
    'UNKNOWN': 0.0,   # no penalty when hallucination check hasn't run yet
}


def confidence_score(signal) -> float:
    """Compute composite confidence from author_confidence + hallucination_verdict."""
    base = CONFIDENCE_MAP.get(_get(signal, 'author_confidence'), 0.5)
    modifier = HALLUCINATION_MODIFIER.get(_get(signal, 'hallucination_verdict'), -0.1)
    return max(0.0, min(1.0, base + modifier))


def _has_price_based_entries(signal) -> bool:
    """Check if entry_conditions contain price/indicator references."""
    for ec in _get(signal, 'entry_conditions', []):
        if PRICE_INDICATOR_RE.search(str(ec)):
            return True
    return False


def _bigram_set(text: str) -> set:
    """Extract word bigrams for similarity comparison."""
    words = text.lower().split()
    return set(zip(words[:-1], words[1:]))


def _text_similarity(text_a: str, text_b: str) -> float:
    """Jaccard similarity on word bigrams."""
    bg_a = _bigram_set(text_a)
    bg_b = _bigram_set(text_b)
    if not bg_a or not bg_b:
        return 0.0
    return len(bg_a & bg_b) / len(bg_a | bg_b)


class PreReviewFilter:
    """
    Filters extracted signals before human review.

    Usage:
        pf = PreReviewFilter()
        filtered = pf.filter(all_signals)
    """

    def __init__(self, min_confidence=0.6, dedup_threshold=0.5):
        self.min_confidence = min_confidence
        self.dedup_threshold = dedup_threshold

    def filter(self, signals: list) -> list:
        """Apply all filters and return signals worth reviewing."""
        stats = {
            'input': len(signals),
            'kept_formula': 0,
            'dropped_confidence': 0,
            'dropped_chunk_type': 0,
            'dropped_duplicate': 0,
        }

        # Separate FORMULA signals (always keep)
        formula_signals = []
        other_signals = []
        for s in signals:
            chunk_type = _get(s, 'chunk_type')
            is_formula = (chunk_type == 'FORMULA' or
                          _get(s, 'signal_category', '') == 'FORMULA')
            if is_formula:
                formula_signals.append(s)
                stats['kept_formula'] += 1
            else:
                other_signals.append(s)

        # Filter 1: confidence threshold
        confident = []
        for s in other_signals:
            score = confidence_score(s)
            if score >= self.min_confidence:
                confident.append(s)
            else:
                stats['dropped_confidence'] += 1

        # Filter 2: chunk-type filter
        # - PSYCHOLOGY books: drop signals with price-based entry conditions
        # - METHODOLOGY books: drop PSYCHOLOGY/EMPIRICAL/SUMMARY chunk signals
        #   that have price-based entries (hallucinated indicator rules)
        chunk_filtered = []
        for s in confident:
            book_id = _get(s, 'book_id', '')
            chunk_type = _get(s, 'chunk_type', '')

            if book_id in PSYCHOLOGY_BOOKS and _has_price_based_entries(s):
                stats['dropped_chunk_type'] += 1
                continue

            if (book_id in METHODOLOGY_BOOKS and
                    chunk_type in ('PSYCHOLOGY', 'EMPIRICAL', 'SUMMARY') and
                    _has_price_based_entries(s)):
                stats['dropped_chunk_type'] += 1
                continue

            chunk_filtered.append(s)

        # Filter 3: dedup within same book (threshold=0.5)
        deduped = self._dedup_by_book(chunk_filtered, stats)

        # Also dedup formula signals among themselves
        deduped_formula = self._dedup_by_book(formula_signals, stats)

        result = deduped_formula + deduped
        stats['output'] = len(result)

        self._log_stats(stats)
        return result

    def _dedup_by_book(self, signals: list, stats: dict) -> list:
        """Remove near-duplicates within the same book using bigram similarity."""
        by_book: Dict[str, list] = {}
        for s in signals:
            by_book.setdefault(_get(s, 'book_id'), []).append(s)

        kept = []
        for book_id, book_signals in by_book.items():
            book_kept = []
            for s in book_signals:
                is_dup = False
                for existing in book_kept:
                    sim = _text_similarity(_get(s, 'rule_text', ''), _get(existing, 'rule_text', ''))
                    if sim > self.dedup_threshold:
                        is_dup = True
                        break
                if is_dup:
                    stats['dropped_duplicate'] += 1
                else:
                    book_kept.append(s)
            kept.extend(book_kept)

        return kept

    def _log_stats(self, stats: dict):
        """Print filter statistics."""
        print(f"\n{'='*50}")
        print(f"PRE-REVIEW FILTER RESULTS")
        print(f"{'='*50}")
        print(f"  Input signals:           {stats['input']}")
        print(f"  Kept (FORMULA bypass):   {stats['kept_formula']}")
        print(f"  Dropped (low confidence):{stats['dropped_confidence']}")
        print(f"  Dropped (chunk-type):    {stats['dropped_chunk_type']}")
        print(f"  Dropped (duplicates):    {stats['dropped_duplicate']}")
        print(f"  OUTPUT for review:       {stats['output']}")
        print(f"  Reduction:               {stats['input']} → {stats['output']} "
              f"({100 - stats['output'] * 100 // max(stats['input'], 1)}% filtered)")
        print(f"{'='*50}")
