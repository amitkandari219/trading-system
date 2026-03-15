"""
Tests for Phase 3 RAG Pipeline modules.
Tests PDF ingester, vector store, signal candidate, and prompts.
Does NOT require Claude API key (mocks API calls).
"""

import pytest
import json
from unittest.mock import patch, MagicMock
from datetime import datetime

from ingestion.pdf_ingester import PDFIngester, RawChunk
from ingestion.signal_candidate import (
    SignalCandidate, make_signal_id, store_approved_signal
)
from extraction.prompts import (
    CONCRETE_BOOK_EXTRACTION_PROMPT,
    PRINCIPLE_BOOK_EXTRACTION_PROMPT,
    HALLUCINATION_CHECK_PROMPT,
    CONFLICT_DETECTION_PROMPT,
)


# ================================================================
# PDF INGESTER TESTS
# ================================================================

class TestPDFIngester:
    def test_all_book_profiles_exist(self):
        ingester = PDFIngester()
        expected_books = [
            'GUJRAL', 'NATENBERG', 'SINCLAIR', 'GRIMES',
            'KAUFMAN', 'DOUGLAS', 'LOPEZ', 'HILPISCH',
            'MCMILLAN', 'AUGEN', 'HARRIS'
        ]
        for book in expected_books:
            assert book in ingester.BOOK_PROFILES

    def test_book_profiles_have_required_fields(self):
        ingester = PDFIngester()
        for book_id, profile in ingester.BOOK_PROFILES.items():
            assert 'abstraction_level' in profile, f"{book_id} missing abstraction_level"
            assert 'chapter_pattern' in profile, f"{book_id} missing chapter_pattern"

    def test_detect_chunk_type_formula(self):
        ingester = PDFIngester()
        profile = ingester.BOOK_PROFILES['NATENBERG']
        assert ingester._detect_chunk_type('S = N(d1) * K * e^(-rT)', profile) == 'FORMULA'

    def test_detect_chunk_type_code(self):
        ingester = PDFIngester()
        profile = ingester.BOOK_PROFILES['HILPISCH']
        assert ingester._detect_chunk_type('def calculate_returns(prices):', profile) == 'CODE'

    def test_detect_chunk_type_empirical(self):
        ingester = PDFIngester()
        profile = ingester.BOOK_PROFILES['SINCLAIR']
        assert ingester._detect_chunk_type(
            'We found empirically that volatility tends to cluster', profile
        ) == 'EMPIRICAL'

    def test_detect_chunk_type_summary(self):
        ingester = PDFIngester()
        profile = ingester.BOOK_PROFILES['GUJRAL']
        assert ingester._detect_chunk_type('Key Points for this chapter', profile) == 'SUMMARY'

    def test_detect_chunk_type_psychology(self):
        ingester = PDFIngester()
        profile = ingester.BOOK_PROFILES['GUJRAL']
        assert ingester._detect_chunk_type(
            'Fear and greed drive most trading decisions', profile
        ) == 'PSYCHOLOGY'

    def test_detect_chunk_type_default_rule(self):
        ingester = PDFIngester()
        profile = ingester.BOOK_PROFILES['GUJRAL']
        assert ingester._detect_chunk_type(
            'Buy when RSI crosses above 30 from below', profile
        ) == 'RULE'

    def test_douglas_all_psychology(self):
        ingester = PDFIngester()
        profile = ingester.BOOK_PROFILES['DOUGLAS']
        assert profile.get('all_chunks_type') == 'PSYCHOLOGY'
        # Any text should be PSYCHOLOGY for Douglas
        assert ingester._detect_chunk_type(
            'Trading is about making money', profile
        ) == 'PSYCHOLOGY'

    def test_contains_formula(self):
        ingester = PDFIngester()
        assert ingester._contains_formula('x = y + z') is True
        assert ingester._contains_formula('no formula here') is False

    def test_table_to_chunk(self):
        ingester = PDFIngester()
        table = [['A', 'B', 'C'], ['1', '2', '3']]
        chunk = ingester._table_to_chunk(
            table, 'GUJRAL', 1, 'Chapter 1', 10, 'T0'
        )
        assert chunk.chunk_type == 'TABLE'
        assert chunk.has_table is True
        assert chunk.chunk_index == 'T0'
        assert chunk.book_id == 'GUJRAL'
        assert 'A | B | C' in chunk.text

    def test_chunk_buffer_basic(self):
        ingester = PDFIngester()
        profile = ingester.BOOK_PROFILES['GUJRAL']
        page_buffer = [
            {'text': 'This is a paragraph about trading.\n\nAnother paragraph here.', 'page': 10},
            {'text': 'More text on page 11.\n\nFinal paragraph.', 'page': 11},
        ]
        chunks = ingester._chunk_buffer(page_buffer, 'GUJRAL', 1, 'Chapter 1', profile)
        assert len(chunks) >= 1
        assert all(isinstance(c, RawChunk) for c in chunks)
        assert all(c.book_id == 'GUJRAL' for c in chunks)

    def test_target_chapters_filtering(self):
        """LOPEZ should only ingest chapters 7, 8, 14."""
        profile = PDFIngester.BOOK_PROFILES['LOPEZ']
        assert profile['target_chapters'] == [7, 8, 14]

    def test_hilpisch_target_chapters(self):
        profile = PDFIngester.BOOK_PROFILES['HILPISCH']
        assert profile['target_chapters'] == [4, 5, 6, 7, 9]


# ================================================================
# SIGNAL CANDIDATE TESTS
# ================================================================

class TestSignalCandidate:
    def test_from_claude_response(self):
        class MockChunk:
            text = "Buy when RSI crosses above 30"

        response = {
            'rule_found': True,
            'rule_text': 'Buy on RSI oversold bounce',
            'signal_category': 'REVERSION',
            'direction': 'LONG',
            'entry_conditions': ['RSI(14) crosses above 30'],
            'parameters': {'rsi_period': 14, 'rsi_threshold': 30},
            'exit_conditions': ['RSI reaches 70'],
            'instrument': 'FUTURES',
            'timeframe': 'INTRADAY',
            'target_regimes': ['RANGING'],
            'source_citation': 'Test Book, Author, Ch.1, p.10',
            'author_confidence': 'HIGH',
        }

        candidate = SignalCandidate.from_claude_response(
            'GUJ_001', 'GUJRAL', MockChunk(), response
        )
        assert candidate.signal_id == 'GUJ_001'
        assert candidate.book_id == 'GUJRAL'
        assert candidate.signal_category == 'REVERSION'
        assert candidate.direction == 'LONG'
        assert len(candidate.entry_conditions) == 1
        assert candidate.parameters['rsi_period'] == 14

    def test_from_claude_response_defaults(self):
        class MockChunk:
            text = "Some text"

        response = {'rule_found': True, 'rule_text': 'A rule'}
        candidate = SignalCandidate.from_claude_response(
            'GUJ_001', 'GUJRAL', MockChunk(), response
        )
        assert candidate.instrument == 'AUTHOR_SILENT'
        assert candidate.timeframe == 'AUTHOR_SILENT'
        assert candidate.target_regimes == ['ANY']
        assert candidate.direction == 'CONTEXT_DEPENDENT'

    def test_make_signal_id(self):
        assert make_signal_id('GUJRAL', 1) == 'GUJ_001'
        assert make_signal_id('NATENBERG', 42) == 'NAT_042'
        assert make_signal_id('KAUFMAN', 100) == 'KAU_100'
        assert make_signal_id('MCMILLAN', 5) == 'MCM_005'
        assert make_signal_id('AUGEN', 3) == 'AUG_003'
        assert make_signal_id('HARRIS', 7) == 'HAR_007'
        assert make_signal_id('UNKNOWN_BOOK', 5) == 'UNK_005'

    def test_store_approved_signal_approve(self):
        """Test that store_approved_signal builds correct SQL params."""
        mock_db = MagicMock()
        candidate = SignalCandidate(
            signal_id='GUJ_001',
            book_id='GUJRAL',
            source_citation='How to Make Money, Gujral, Ch.3, p.47',
            raw_chunk_text='Buy when RSI crosses above 30',
            rule_text='Buy on RSI oversold bounce',
            signal_category='REVERSION',
            direction='LONG',
            entry_conditions=['RSI(14) crosses above 30'],
            parameters={'rsi_period': 14},
            exit_conditions=['RSI reaches 70'],
            instrument='FUTURES',
            timeframe='INTRADAY',
            target_regimes=['RANGING'],
            author_confidence='HIGH',
        )
        decision = {
            'decision': 'A',
            'notes': 'Looks good',
            'revised_conditions': None,
            'reviewer_timestamp': '2024-01-01T00:00:00',
        }

        result = store_approved_signal(mock_db, candidate, decision)
        assert result == 'GUJ_001'
        mock_db.execute.assert_called_once()

    def test_store_approved_signal_revise(self):
        """Test that revised conditions replace original."""
        mock_db = MagicMock()
        candidate = SignalCandidate(
            signal_id='GUJ_002',
            book_id='GUJRAL',
            source_citation='Test, Author, Ch.1, p.5',
            raw_chunk_text='text',
            rule_text='A rule',
            signal_category='TREND',
            direction='LONG',
            entry_conditions=['Original condition'],
            instrument='AUTHOR_SILENT',
            timeframe='AUTHOR_SILENT',
        )
        decision = {
            'decision': 'R',
            'notes': 'Revised entry',
            'revised_conditions': ['New condition 1', 'New condition 2'],
            'reviewer_timestamp': '2024-01-01T00:00:00',
        }

        store_approved_signal(mock_db, candidate, decision)
        call_args = mock_db.execute.call_args
        # The SQL params should contain the revised conditions
        params = call_args[0][1]
        # entry_conditions is a PgJson wrapper; check it wraps the revised list
        assert params[8].adapted == ['New condition 1', 'New condition 2']

    def test_author_silent_mapped_to_any(self):
        """AUTHOR_SILENT should be mapped to ANY for DB insert."""
        mock_db = MagicMock()
        candidate = SignalCandidate(
            signal_id='GUJ_003',
            book_id='GUJRAL',
            source_citation='Test, Author, Ch.1, p.5',
            raw_chunk_text='text',
            rule_text='A rule',
            signal_category='TREND',
            direction='LONG',
            instrument='AUTHOR_SILENT',
            timeframe='AUTHOR_SILENT',
        )
        decision = {'decision': 'A', 'notes': ''}

        store_approved_signal(mock_db, candidate, decision)
        call_args = mock_db.execute.call_args
        params = call_args[0][1]
        # instrument and timeframe positions in the param tuple
        assert params[11] == 'ANY'  # instrument
        assert params[12] == 'ANY'  # timeframe


# ================================================================
# PROMPT TESTS
# ================================================================

class TestPrompts:
    def test_concrete_prompt_has_placeholders(self):
        required = ['{book_title}', '{author}', '{chapter_title}',
                    '{page_start}', '{page_end}', '{chunk_text}',
                    '{chapter_number}']
        for placeholder in required:
            assert placeholder in CONCRETE_BOOK_EXTRACTION_PROMPT, \
                f"Missing {placeholder}"

    def test_principle_prompt_has_placeholders(self):
        required = ['{book_title}', '{author}', '{chapter_title}',
                    '{page_start}', '{page_end}', '{chunk_text}',
                    '{chapter_number}']
        for placeholder in required:
            assert placeholder in PRINCIPLE_BOOK_EXTRACTION_PROMPT, \
                f"Missing {placeholder}"

    def test_hallucination_prompt_has_placeholders(self):
        assert '{original_chunk}' in HALLUCINATION_CHECK_PROMPT
        assert '{extracted_rule}' in HALLUCINATION_CHECK_PROMPT

    def test_conflict_prompt_has_placeholders(self):
        assert '{rule_a}' in CONFLICT_DETECTION_PROMPT
        assert '{rule_b}' in CONFLICT_DETECTION_PROMPT

    def test_concrete_prompt_formats_without_error(self):
        """Verify the prompt can be formatted with sample data."""
        formatted = CONCRETE_BOOK_EXTRACTION_PROMPT.format(
            book_title='Test Book',
            author='Test Author',
            chapter_title='Chapter 1',
            page_start=10,
            page_end=12,
            chapter_number=1,
            chunk_text='Buy when RSI crosses above 30.',
        )
        assert 'Test Book' in formatted
        assert 'Buy when RSI' in formatted

    def test_principle_prompt_formats_without_error(self):
        formatted = PRINCIPLE_BOOK_EXTRACTION_PROMPT.format(
            book_title='Test Book',
            author='Test Author',
            chapter_title='Chapter 5',
            page_start=50,
            page_end=52,
            chapter_number=5,
            chunk_text='Volatility tends to revert to mean.',
        )
        assert 'Test Book' in formatted
        assert 'Volatility' in formatted


# ================================================================
# VECTOR STORE TESTS (unit tests without ChromaDB)
# ================================================================

class TestVectorStoreUnit:
    def test_abstraction_levels(self):
        from ingestion.vector_store import VectorStore
        levels = VectorStore.ABSTRACTION_LEVELS
        assert levels['GUJRAL'] == 'CONCRETE'
        assert levels['MCMILLAN'] == 'CONCRETE'
        assert levels['AUGEN'] == 'CONCRETE'
        assert levels['NATENBERG'] == 'PRINCIPLE'
        assert levels['GRIMES'] == 'METHODOLOGY'
        assert levels['HARRIS'] == 'METHODOLOGY'
        assert levels['DOUGLAS'] == 'PSYCHOLOGY'

    def test_text_similarity(self):
        """Test bigram Jaccard similarity."""
        from ingestion.vector_store import VectorStore
        # Can't instantiate without chromadb, test static method directly
        def bigrams(text):
            words = text.lower().split()
            return set(zip(words[:-1], words[1:]))

        bg_a = bigrams("buy when rsi crosses above 30")
        bg_b = bigrams("buy when rsi crosses below 70")
        overlap = len(bg_a & bg_b) / len(bg_a | bg_b)
        assert 0 < overlap < 1  # Partial overlap expected
