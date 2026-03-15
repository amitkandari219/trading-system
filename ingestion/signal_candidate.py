"""
SignalCandidate dataclass — in-memory extraction result from Claude API.
Created by ExtractionOrchestrator, reviewed by CLIReviewer,
then written to the signals table via store_approved_signal().
"""

import json
import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from psycopg2.extras import Json as PgJson


@dataclass
class SignalCandidate:
    """
    In-memory extraction result from Claude API.
    Produced by ExtractionOrchestrator.
    Passed to CLIReviewer.review_signal() for human approval.
    If approved, stored to signals table via store_approved_signal().
    """
    # Identity
    signal_id:             str
    book_id:               str
    source_citation:       str

    # Original source (shown to reviewer)
    raw_chunk_text:        str

    # Claude extraction output
    rule_text:             str
    signal_category:       str          # TREND|REVERSION|VOL|PATTERN|EVENT
    direction:             str          # LONG|SHORT|NEUTRAL|CONTEXT_DEPENDENT
    entry_conditions:      List[str]    = field(default_factory=list)
    parameters:            Dict[str, Any] = field(default_factory=dict)
    exit_conditions:       List[str]    = field(default_factory=list)
    instrument:            str          = 'AUTHOR_SILENT'
    timeframe:             str          = 'AUTHOR_SILENT'
    target_regimes:        List[str]    = field(default_factory=lambda: ['ANY'])

    # Quality checks
    hallucination_verdict: str          = 'UNKNOWN'  # PASS|WARN|FAIL
    hallucination_issues:  List[str]    = field(default_factory=list)
    completeness_warning:  Optional[str] = None
    author_confidence:     str          = 'MEDIUM'

    # For PRINCIPLE-level books: which variant is this?
    variant_id:            Optional[str] = None  # CONSERVATIVE|MODERATE|AGGRESSIVE

    @classmethod
    def from_claude_response(cls, signal_id: str, book_id: str,
                              chunk, response: dict) -> 'SignalCandidate':
        """
        Parse Claude API JSON response into SignalCandidate.
        chunk: RawChunk (or any object with .text attribute).
        response: parsed JSON dict from Claude.
        """
        return cls(
            signal_id             = signal_id,
            book_id               = book_id,
            source_citation       = response.get('source_citation', ''),
            raw_chunk_text        = chunk.text,
            rule_text             = response.get('rule_text', ''),
            signal_category       = response.get('signal_category', 'PATTERN'),
            direction             = response.get('direction', 'CONTEXT_DEPENDENT'),
            entry_conditions      = response.get('entry_conditions', []),
            parameters            = response.get('parameters', {}),
            exit_conditions       = response.get('exit_conditions', []),
            instrument            = response.get('instrument', 'AUTHOR_SILENT'),
            timeframe             = response.get('timeframe', 'AUTHOR_SILENT'),
            target_regimes        = response.get('target_regimes', ['ANY']),
            completeness_warning  = response.get('completeness_warning'),
            author_confidence     = response.get('author_confidence', 'MEDIUM'),
        )


# ================================================================
# SIGNAL ID GENERATION
# ================================================================

SIGNAL_ID_PREFIXES = {
    'GUJRAL':      'GUJ',
    'KAUFMAN':     'KAU',
    'MCMILLAN':    'MCM',
    'AUGEN':       'AUG',
    'NATENBERG':   'NAT',
    'SINCLAIR':    'SIN',
    'GRIMES':      'GRI',
    'LOPEZ':       'LOP',
    'HILPISCH':    'HIL',
    'HARRIS':      'HAR',
    'DOUGLAS':     'DOU',
    'BULKOWSKI':   'BUL',
    'CANDLESTICK': 'CAN',
    'CHAN_QT':     'CQT',
    'CHAN_AT':     'CAT',
    'VINCE':       'VIN',
}


def make_signal_id(book_id: str, seq_number: int) -> str:
    """Generate signal ID like 'GUJ_001' from book_id and sequence number."""
    prefix = SIGNAL_ID_PREFIXES.get(book_id, book_id[:3].upper())
    return f"{prefix}_{seq_number:03d}"


def next_signal_id(db, book_id: str) -> str:
    """Query DB for next available signal ID for this book."""
    cursor = db.execute(
        "SELECT COUNT(*) AS n FROM signals WHERE book_id = %s",
        (book_id,)
    )
    row = cursor.fetchone()
    # Handle both dict-style and tuple-style cursor results
    if isinstance(row, dict):
        count = row['n']
    else:
        count = row[0] if row else 0
    return make_signal_id(book_id, count + 1)


# ================================================================
# STORE APPROVED SIGNAL TO DB
# ================================================================

def store_approved_signal(db, candidate: SignalCandidate,
                          decision: dict) -> str:
    """
    Bridge between CLIReviewer's decision and the signals table INSERT.
    Called after review_signal() returns decision 'A' or 'R'.
    Returns: signal_id of the stored record.
    """
    # Apply reviewer revisions if any
    entry_conditions = candidate.entry_conditions
    if decision['decision'] == 'R' and decision.get('revised_conditions'):
        entry_conditions = decision['revised_conditions']

    # Parse source_citation for chapter and page
    source_chapter = f"Ch. from {candidate.book_id}"
    source_page_start = None
    m = re.search(r'p\.(\d+)', candidate.source_citation)
    if m:
        source_page_start = int(m.group(1))
    m2 = re.search(r'Ch\.(\d+)', candidate.source_citation)
    if m2:
        source_chapter = f"Chapter {m2.group(1)}"

    # Build review_notes
    review_notes_parts = []
    if candidate.completeness_warning:
        review_notes_parts.append(f"COMPLETENESS: {candidate.completeness_warning}")
    if candidate.author_confidence:
        review_notes_parts.append(f"AUTHOR_CONFIDENCE: {candidate.author_confidence}")
    if candidate.variant_id:
        review_notes_parts.append(f"VARIANT: {candidate.variant_id}")
    if decision.get('notes'):
        review_notes_parts.append(f"REVIEWER: {decision['notes']}")
    review_notes = " | ".join(review_notes_parts) or None

    signal_name = f"{candidate.book_id}: {candidate.rule_text[:60]}"

    # Map AUTHOR_SILENT -> 'ANY' for DB CHECK constraints
    instrument = 'ANY' if candidate.instrument == 'AUTHOR_SILENT' else candidate.instrument
    timeframe  = 'ANY' if candidate.timeframe  == 'AUTHOR_SILENT' else candidate.timeframe

    db.execute("""
        INSERT INTO signals (
            signal_id, name, book_id,
            source_chapter, source_page_start, raw_chunk_text,
            signal_category, direction,
            entry_conditions, parameters, exit_conditions,
            instrument, timeframe, target_regimes,
            status, classification,
            avoid_rbi_day,
            review_notes, human_reviewer,
            created_at, updated_at,
            pending_change_by, pending_change_reason
        ) VALUES (
            %s, %s, %s,
            %s, %s, %s,
            %s, %s,
            %s, %s, %s,
            %s, %s, %s,
            'CANDIDATE', NULL,
            FALSE,
            %s, %s,
            NOW(), NOW(),
            %s, %s
        )
    """, (
        candidate.signal_id,
        signal_name,
        candidate.book_id,
        source_chapter,
        source_page_start,
        candidate.raw_chunk_text,
        candidate.signal_category,
        candidate.direction,
        PgJson(entry_conditions),
        PgJson(candidate.parameters),
        PgJson(candidate.exit_conditions),
        instrument,
        timeframe,
        candidate.target_regimes,
        review_notes,
        'HUMAN_REVIEWER',
        decision.get('notes', ''),
        f"APPROVED_{decision['decision']}_by_human_review",
    ))

    return candidate.signal_id
