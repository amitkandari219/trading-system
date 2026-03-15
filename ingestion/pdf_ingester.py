"""
Book-aware PDF ingester.
Each book has a profile that handles its specific layout.

Usage:
    python -m ingestion.pdf_ingester --book GUJRAL --pdf books/gujral.pdf
"""

import re
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional

# OCR imports are deferred to avoid loading when not needed
_ocr_available = None  # lazy-checked


@dataclass
class RawChunk:
    text: str
    book_id: str
    chapter_number: int
    chapter_title: str
    page_start: int
    page_end: int
    chunk_index: object   # int for text chunks (0,1,2...), str for tables ('T0','T1',...)
                          # ChromaDB ID: f"{book_id}_{chapter_number}_{chunk_index}"
    chunk_type: str       # RULE|SUMMARY|EMPIRICAL|FORMULA|TABLE|CODE|PSYCHOLOGY|HEADING
    has_table: bool
    has_formula: bool


class PDFIngester:
    """
    Book-aware PDF ingester.
    Each book has a profile that handles its specific layout.
    """

    BOOK_PROFILES = {
        'GUJRAL': {
            'abstraction_level': 'CONCRETE',
            'skip_pages': [1, 2, 3],
            'chapter_pattern': r'^Chapter\s+\d+',
            'summary_marker': 'Key Points',
            'table_heavy': False,
            'formula_heavy': False,
        },
        'NATENBERG': {
            'abstraction_level': 'PRINCIPLE',
            'skip_pages': [1, 2, 3, 4, 5],
            'chapter_pattern': r'^\d+\s+[A-Z]',
            'summary_marker': None,
            'table_heavy': True,
            'formula_heavy': True,
            'skip_sections': ['Appendix'],
        },
        'SINCLAIR': {
            'abstraction_level': 'PRINCIPLE',
            'skip_pages': [1, 2, 3],
            'chapter_pattern': r'^CHAPTER\s+\d+',
            'summary_marker': 'Summary',
            'table_heavy': False,
            'formula_heavy': True,
            'empirical_markers': [
                'empirically', 'we found', 'data shows',
                'historically', 'on average', 'tends to'
            ],
        },
        'GRIMES': {
            'abstraction_level': 'METHODOLOGY',
            'skip_pages': [1, 2, 3, 4],  # title pages only
            'chapter_pattern': r'JWBT634-c(\d+)',
            'chapter_detect_mode': 'watermark',  # detect from watermark, not heading
            'fix_joined_words': True,  # PDF has broken word spacing
            'summary_marker': 'Summary',
            'table_heavy': True,
            'empirical_markers': [
                'statistically', 'edge', 'no edge',
                'does not', 'fails', 'works'
            ],
            'edge_confirmed_marker': 'has edge',
            'edge_denied_marker': 'no statistical edge',
        },
        'KAUFMAN': {
            'abstraction_level': 'CONCRETE',
            'skip_pages': list(range(1, 21)),    # TOC, preface (~20 pages)
            'chapter_pattern': r'^CHAPTER\s+\d+',
            'summary_marker': None,
            'table_heavy': True,
            'formula_heavy': False,
            'pseudocode_markers': ['Step 1', 'If price', 'When'],
        },
        'DOUGLAS': {
            'abstraction_level': 'PSYCHOLOGY',
            'skip_pages': [1, 2, 3],
            'chapter_pattern': r'^CHAPTER\s+\d+',
            'all_chunks_type': 'PSYCHOLOGY',
        },
        'LOPEZ': {
            'abstraction_level': 'METHODOLOGY',
            'skip_pages': list(range(1, 10)),
            'chapter_pattern': r'^(?:CHAPTER|Chapter)\s+\d+',
            'formula_heavy': True,
            'code_heavy': True,
        },
        'HILPISCH': {
            'abstraction_level': 'METHODOLOGY',
            'skip_pages': list(range(1, 8)),
            'chapter_pattern': r'^CHAPTER\s+\d+',
            'code_heavy': True,
        },
        'MCMILLAN': {
            'abstraction_level': 'CONCRETE',
            'skip_pages': [1, 2, 3, 4, 5],
            'chapter_pattern': r'^Chapter\s+\d+',
            'summary_marker': 'Summary',
            'table_heavy': True,
            'formula_heavy': False,
        },
        'AUGEN': {
            'abstraction_level': 'CONCRETE',
            'skip_pages': [1, 2, 3],
            'chapter_pattern': r'^Chapter\s+\d+',
            'summary_marker': None,
            'table_heavy': True,
            'formula_heavy': False,
            'empirical_markers': [
                'historically', 'on average', 'tends to',
                'IV crush', 'expiration', 'weekly'
            ],
        },
        'BULKOWSKI': {
            'abstraction_level': 'CONCRETE',
            'skip_pages': list(range(1, 10)),
            'chapter_pattern': r'^(?:CHAPTER|Chapter)\s+\d+',
            'summary_marker': None,
            'table_heavy': True,
            'formula_heavy': False,
            'empirical_markers': [
                'performance', 'failure rate', 'average rise',
                'average decline', 'breakout', 'confirmation',
            ],
        },
        'CANDLESTICK': {
            'abstraction_level': 'CONCRETE',
            'skip_pages': list(range(1, 10)),
            'chapter_pattern': r'^(?:CHAPTER|Chapter)\s+\d+',
            'summary_marker': None,
            'table_heavy': True,
            'formula_heavy': False,
            'empirical_markers': [
                'performance', 'frequency', 'reversal',
                'continuation', 'bull', 'bear',
            ],
        },
        'CHAN_QT': {
            'abstraction_level': 'METHODOLOGY',
            'skip_pages': list(range(1, 8)),
            'chapter_pattern': r'^(?:CHAPTER|Chapter)\s+\d+',
            'summary_marker': None,
            'formula_heavy': True,
            'code_heavy': True,
        },
        'CHAN_AT': {
            'abstraction_level': 'METHODOLOGY',
            'skip_pages': list(range(1, 8)),
            'chapter_pattern': r'^(?:CHAPTER|Chapter)\s+\d+',
            'summary_marker': None,
            'formula_heavy': True,
            'code_heavy': True,
        },
        'VINCE': {
            'abstraction_level': 'PRINCIPLE',
            'skip_pages': list(range(1, 6)),
            'chapter_pattern': r'^(?:CHAPTER|Chapter)\s+\d+',
            'summary_marker': None,
            'formula_heavy': True,
            'table_heavy': True,
        },
        # Batch 2
        'INTERMARKET': {
            'abstraction_level': 'METHODOLOGY',
            'skip_pages': list(range(1, 8)),
            'chapter_pattern': r'^(?:CHAPTER|Chapter)\s+\d+',
            'summary_marker': None,
            'table_heavy': True,
        },
        'MICROSTRUCTURE': {
            'abstraction_level': 'METHODOLOGY',
            'skip_pages': list(range(1, 8)),
            'chapter_pattern': r'^(?:CHAPTER|Chapter)\s+\d+',
            'summary_marker': None,
            'table_heavy': True,
        },
        'POSITIONAL': {
            'abstraction_level': 'CONCRETE',
            'skip_pages': list(range(1, 6)),
            'chapter_pattern': r'^(?:CHAPTER|Chapter)\s+\d+',
            'summary_marker': None,
            'table_heavy': True,
            'formula_heavy': True,
        },
        'ARONSON': {
            'abstraction_level': 'METHODOLOGY',
            'skip_pages': list(range(1, 10)),
            'chapter_pattern': r'^(?:CHAPTER|Chapter)\s+\d+',
            'summary_marker': None,
            'table_heavy': True,
            'formula_heavy': True,
            'empirical_markers': ['statistically', 'p-value', 'significant', 'data-mined'],
        },
        'ALGO_SUCCESS': {
            'abstraction_level': 'METHODOLOGY',
            'skip_pages': list(range(1, 6)),
            'chapter_pattern': r'^(?:CHAPTER|Chapter)\s+\d+',
            'code_heavy': True,
        },
        # Batch 3
        'KAHNEMAN': {
            'abstraction_level': 'PSYCHOLOGY',
            'skip_pages': list(range(1, 8)),
            'chapter_pattern': r'^(?:CHAPTER|Chapter)\s+\d+',
            'all_chunks_type': 'PSYCHOLOGY',
        },
        'TALEB_BS': {
            'abstraction_level': 'PSYCHOLOGY',
            'skip_pages': list(range(1, 8)),
            'chapter_pattern': r'^(?:CHAPTER|Chapter)\s+\d+',
            'all_chunks_type': 'PSYCHOLOGY',
        },
        'TALEB_AF': {
            'abstraction_level': 'PSYCHOLOGY',
            'skip_pages': list(range(1, 8)),
            'chapter_pattern': r'^(?:CHAPTER|Chapter)\s+\d+',
            'all_chunks_type': 'PSYCHOLOGY',
        },
        'THARP': {
            'abstraction_level': 'METHODOLOGY',
            'skip_pages': list(range(1, 6)),
            'chapter_pattern': r'^(?:CHAPTER|Chapter)\s+\d+',
            'summary_marker': None,
        },
        'SCHWAGER': {
            'abstraction_level': 'PSYCHOLOGY',
            'skip_pages': list(range(1, 6)),
            'chapter_pattern': r'^(?:CHAPTER|Chapter)\s+\d+',
            'all_chunks_type': 'PSYCHOLOGY',
        },
        'SOROS': {
            'abstraction_level': 'PSYCHOLOGY',
            'skip_pages': list(range(1, 8)),
            'chapter_pattern': r'^(?:CHAPTER|Chapter)\s+\d+',
        },
        'MAJOR_ASSETS': {
            'abstraction_level': 'METHODOLOGY',
            'skip_pages': list(range(1, 6)),
            'chapter_pattern': r'^(?:CHAPTER|Chapter)\s+\d+',
            'table_heavy': True,
        },
        # Batch 4
        'GATHERAL': {
            'abstraction_level': 'PRINCIPLE',
            'skip_pages': list(range(1, 6)),
            'chapter_pattern': r'^(?:CHAPTER|Chapter)\s+\d+',
            'formula_heavy': True,
        },
        'KELLY': {
            'abstraction_level': 'PRINCIPLE',
            'skip_pages': list(range(1, 8)),
            'chapter_pattern': r'^(?:CHAPTER|Chapter)\s+\d+',
            'formula_heavy': True,
            'table_heavy': True,
        },
        'FLASH': {
            'abstraction_level': 'METHODOLOGY',
            'skip_pages': list(range(1, 6)),
            'chapter_pattern': r'^(?:CHAPTER|Chapter)\s+\d+',
        },
        'JOHNSON': {
            'abstraction_level': 'METHODOLOGY',
            'skip_pages': list(range(1, 8)),
            'chapter_pattern': r'^(?:CHAPTER|Chapter)\s+\d+',
            'table_heavy': True,
            'formula_heavy': True,
        },
        'HARRIS': {
            'abstraction_level': 'METHODOLOGY',
            'skip_pages': [1, 2, 3, 4],
            'chapter_pattern': r'^(?:CHAPTER|Chapter)\s+\d+',
            'summary_marker': 'Summary',
            'table_heavy': True,
            'empirical_markers': [
                'statistically', 'liquidity', 'bid-ask',
                'informed', 'market impact', 'spread'
            ],
        },
    }

    # PDFs above this page count are processed in page-range batches
    # to avoid OOM from pdfplumber loading all pages at once.
    LARGE_PDF_THRESHOLD = 500
    LARGE_PDF_BATCH_SIZE = 100

    # OCR settings
    OCR_DPI = 300  # resolution for page-to-image conversion

    @staticmethod
    def _check_ocr_available():
        global _ocr_available
        if _ocr_available is None:
            try:
                import pytesseract
                from PIL import Image
                pytesseract.get_tesseract_version()
                _ocr_available = True
            except Exception:
                _ocr_available = False
        return _ocr_available

    def _needs_ocr(self, pdf_path: str, profile: dict) -> bool:
        """Check if PDF is scanned by sampling first few non-skip pages."""
        import pdfplumber

        skip_pages = set(profile.get('skip_pages', []))
        with pdfplumber.open(pdf_path) as pdf:
            sampled = 0
            empty = 0
            for i, page in enumerate(pdf.pages):
                if (i + 1) in skip_pages:
                    continue
                text = page.extract_text() or ''
                sampled += 1
                if len(text.strip()) < 50:
                    empty += 1
                if sampled >= 5:
                    break
        # If 80%+ of sampled pages have no text, it's scanned
        return sampled > 0 and (empty / sampled) >= 0.8

    def _ocr_page(self, page) -> str:
        """Extract text from a scanned page using OCR."""
        import pytesseract
        img = page.to_image(resolution=self.OCR_DPI).original
        text = pytesseract.image_to_string(img)
        return text

    # DOCX chunking settings
    DOCX_CHUNK_TARGET = 800    # target chars per chunk
    DOCX_CHUNK_MIN = 150       # minimum chars to emit a chunk

    def ingest_docx(self, docx_path: str, book_id: str) -> List[RawChunk]:
        """Ingest a DOCX book and return list of RawChunks.
        Uses Heading styles for chapter detection and streams paragraphs
        directly into fixed-size chunks (no page simulation)."""
        from docx import Document
        from docx.table import Table
        from docx.oxml.ns import qn

        profile = self.BOOK_PROFILES[book_id]
        doc = Document(docx_path)

        all_chunks = []
        current_chapter = 0
        current_chapter_title = ''
        chunk_index = 0
        accumulator = []       # list of paragraph strings
        accum_chars = 0

        # Walk document body elements in order (paragraphs + tables inline)
        for element in doc.element.body:
            tag = element.tag.split('}')[-1] if '}' in element.tag else element.tag

            if tag == 'p':
                # Paragraph element
                from docx.text.paragraph import Paragraph
                para = Paragraph(element, doc)
                text = para.text.strip()
                if not text:
                    continue

                # Detect chapter via Heading 1 style
                style_name = para.style.name if para.style else ''
                is_heading1 = style_name.startswith('Heading 1')

                if is_heading1:
                    # Flush accumulator as chunk before starting new chapter
                    if accumulator:
                        emitted = self._emit_docx_chunk(
                            accumulator, book_id, current_chapter,
                            current_chapter_title, chunk_index, profile
                        )
                        all_chunks.extend(emitted)
                        chunk_index += len(emitted)
                        accumulator = []
                        accum_chars = 0

                    current_chapter += 1
                    current_chapter_title = text
                    continue

                accumulator.append(text)
                accum_chars += len(text)

                # Emit chunk when target reached
                if accum_chars >= self.DOCX_CHUNK_TARGET:
                    emitted = self._emit_docx_chunk(
                        accumulator, book_id, current_chapter,
                        current_chapter_title, chunk_index, profile
                    )
                    all_chunks.extend(emitted)
                    chunk_index += len(emitted)
                    # Keep last paragraph as overlap
                    last = accumulator[-1]
                    accumulator = [last]
                    accum_chars = len(last)

            elif tag == 'tbl':
                # Table element — render inline as text
                table = Table(element, doc)
                rows = []
                for row in table.rows:
                    cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                    if cells:
                        rows.append(' | '.join(cells))
                if rows:
                    table_text = '\n'.join(rows)
                    accumulator.append(table_text)
                    accum_chars += len(table_text)

                    if accum_chars >= self.DOCX_CHUNK_TARGET:
                        emitted = self._emit_docx_chunk(
                            accumulator, book_id, current_chapter,
                            current_chapter_title, chunk_index, profile
                        )
                        all_chunks.extend(emitted)
                        chunk_index += len(emitted)
                        accumulator = []
                        accum_chars = 0

        # Flush remaining
        if accumulator:
            emitted = self._emit_docx_chunk(
                accumulator, book_id, current_chapter,
                current_chapter_title, chunk_index, profile
            )
            all_chunks.extend(emitted)

        return all_chunks

    def _emit_docx_chunk(self, paragraphs, book_id, chapter,
                         chapter_title, chunk_index, profile):
        """Create RawChunk(s) from accumulated paragraphs.
        Returns list of chunks (splits oversized ones)."""
        text = '\n\n'.join(paragraphs)
        if len(text) < self.DOCX_CHUNK_MIN:
            return []

        chunk_type = self._dominant_type(paragraphs, profile)
        has_table = any('|' in p for p in paragraphs)
        has_formula = self._contains_formula(text)

        if len(text) <= self.CHUNK_HARD_MAX:
            return [RawChunk(
                text=text, book_id=book_id,
                chapter_number=chapter, chapter_title=chapter_title,
                page_start=0, page_end=0, chunk_index=chunk_index,
                chunk_type=chunk_type, has_table=has_table,
                has_formula=has_formula,
            )]

        # Split oversized on line boundaries
        result = []
        lines = text.split('\n')
        buf = []
        buf_len = 0
        for line in lines:
            if buf_len + len(line) > self.DOCX_CHUNK_TARGET and buf:
                result.append(RawChunk(
                    text='\n'.join(buf), book_id=book_id,
                    chapter_number=chapter, chapter_title=chapter_title,
                    page_start=0, page_end=0,
                    chunk_index=chunk_index + len(result),
                    chunk_type=chunk_type, has_table=has_table,
                    has_formula=has_formula,
                ))
                buf = [buf[-1]]
                buf_len = len(buf[0])
            buf.append(line)
            buf_len += len(line)
        if buf and buf_len >= self.DOCX_CHUNK_MIN:
            result.append(RawChunk(
                text='\n'.join(buf), book_id=book_id,
                chapter_number=chapter, chapter_title=chapter_title,
                page_start=0, page_end=0,
                chunk_index=chunk_index + len(result),
                chunk_type=chunk_type, has_table=has_table,
                has_formula=has_formula,
            ))
        return result

    def ingest_book(self, pdf_path: str, book_id: str) -> List[RawChunk]:
        """Ingest a PDF book and return list of RawChunks."""
        import pdfplumber
        import gc

        profile = self.BOOK_PROFILES[book_id]

        # Detect scanned PDFs and enable OCR if available
        use_ocr = False
        if self._needs_ocr(pdf_path, profile):
            if self._check_ocr_available():
                print(f"  Scanned PDF detected — using OCR (tesseract)")
                use_ocr = True
            else:
                print(f"  WARNING: Scanned PDF detected but tesseract not installed. "
                      f"Install with: apt-get install tesseract-ocr && pip install pytesseract Pillow")
                return []

        # Check total page count
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)

        if total_pages > self.LARGE_PDF_THRESHOLD:
            return self._ingest_book_batched(pdf_path, book_id, profile, total_pages, use_ocr=use_ocr)
        return self._ingest_book_single(pdf_path, book_id, profile, use_ocr=use_ocr)

    def _ingest_book_single(self, pdf_path, book_id, profile, use_ocr=False):
        """Standard ingestion for small/medium PDFs."""
        import pdfplumber
        import gc

        skip_pages = set(profile.get('skip_pages', []))
        all_chunks = []
        state = {
            'current_chapter': 0,
            'current_chapter_title': '',
            'page_buffer': [],
            'last_chapter_page': 0,
            'table_counter': 0,
        }

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_1indexed = page_num + 1
                if page_1indexed in skip_pages:
                    continue

                text = page.extract_text() or ''
                if not text.strip() and use_ocr:
                    text = self._ocr_page(page)
                if not text.strip():
                    continue

                self._process_single_page(
                    text, page, page_1indexed, book_id,
                    profile, state, all_chunks
                )

        # Flush final buffer
        if state['page_buffer']:
            all_chunks.extend(self._chunk_buffer(
                state['page_buffer'], book_id,
                state['current_chapter'],
                state['current_chapter_title'],
                profile
            ))

        return all_chunks

    def _ingest_book_batched(self, pdf_path, book_id, profile, total_pages, use_ocr=False):
        """Memory-efficient ingestion for large PDFs (500+ pages).
        Opens the PDF in page-range batches to avoid OOM."""
        import pdfplumber
        import gc

        all_chunks = []
        state = {
            'current_chapter': 0,
            'current_chapter_title': '',
            'page_buffer': [],
            'last_chapter_page': 0,
            'table_counter': 0,
        }

        skip_pages = set(profile.get('skip_pages', []))

        for batch_start in range(0, total_pages, self.LARGE_PDF_BATCH_SIZE):
            batch_end = min(batch_start + self.LARGE_PDF_BATCH_SIZE, total_pages)
            print(f"    Pages {batch_start+1}-{batch_end}...")

            with pdfplumber.open(pdf_path) as pdf:
                for page_num in range(batch_start, batch_end):
                    page_1indexed = page_num + 1

                    if page_1indexed in skip_pages:
                        continue

                    page = pdf.pages[page_num]
                    text = page.extract_text() or ''
                    if not text.strip() and use_ocr:
                        text = self._ocr_page(page)
                    if not text.strip():
                        continue

                    self._process_single_page(
                        text, page, page_1indexed, book_id,
                        profile, state, all_chunks
                    )

            gc.collect()

        # Flush final buffer
        if state['page_buffer']:
            all_chunks.extend(self._chunk_buffer(
                state['page_buffer'], book_id,
                state['current_chapter'],
                state['current_chapter_title'],
                profile
            ))

        return all_chunks

    @staticmethod
    def _fix_joined_words(text: str) -> str:
        """Split camelCase-joined words from broken PDF word spacing.
        E.g. 'alsoworthconsideringthat' → 'also worth considering that'"""
        # Insert space before uppercase letters preceded by lowercase
        # (handles: 'theMarket' → 'the Market')
        text = re.sub(r'([a-z])([A-Z][a-z])', r'\1 \2', text)
        # Insert space between lowercase and opening paren/digit
        text = re.sub(r'([a-z])(\d)', r'\1 \2', text)
        # Strip printer watermark lines
        text = re.sub(r'^P\d:OTA.*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^JWBT\d+-\S+.*Printer:\w+\s*$', '', text, flags=re.MULTILINE)
        return text

    def _process_single_page(self, text, page, page_1indexed,
                              book_id, profile, state, chunks):
        """Process one page, updating state and appending chunks."""
        import re as _re

        # Fix joined words if needed (e.g. Grimes PDF)
        if profile.get('fix_joined_words'):
            text = self._fix_joined_words(text)

        # Detect chapter boundary
        detect_mode = profile.get('chapter_detect_mode', 'heading')
        if detect_mode == 'watermark':
            # Search entire text for watermark pattern (e.g. JWBT634-c01)
            chapter_match = _re.search(
                profile.get('chapter_pattern', r'^Chapter'),
                text
            )
        else:
            first_lines = '\n'.join(text.split('\n')[:3])
            chapter_match = _re.search(
                profile.get('chapter_pattern', r'^Chapter'),
                first_lines,
                _re.MULTILINE | _re.IGNORECASE
            )
        pages_since = page_1indexed - state['last_chapter_page']

        # For watermark mode, only trigger chapter break when chapter number changes
        if chapter_match and detect_mode == 'watermark':
            watermark_num = int(chapter_match.group(1))
            if watermark_num == state.get('_watermark_chapter', -1):
                chapter_match = None  # same chapter, not a boundary
            else:
                state['_watermark_chapter'] = watermark_num

        if chapter_match and pages_since >= 2:
            if state['page_buffer']:
                new_chunks = self._chunk_buffer(
                    state['page_buffer'], book_id,
                    state['current_chapter'],
                    state['current_chapter_title'],
                    profile
                )
                chunks.extend(new_chunks)
                state['page_buffer'] = []
            state['current_chapter'] += 1
            state['current_chapter_title'] = text.split('\n')[0].strip()
            state['last_chapter_page'] = page_1indexed

        if 'target_chapters' in profile:
            if state['current_chapter'] not in profile['target_chapters']:
                return

        tables = page.extract_tables()
        if tables and profile.get('table_heavy'):
            for table in tables:
                chunks.append(self._table_to_chunk(
                    table, book_id, state['current_chapter'],
                    state['current_chapter_title'], page_1indexed,
                    table_id=f"T{state['table_counter']}"
                ))
                state['table_counter'] += 1

        state['page_buffer'].append({
            'text': text,
            'page': page_1indexed
        })

    # Chunk size targets (chars)
    CHUNK_TARGET = 800
    CHUNK_MIN = 150
    CHUNK_HARD_MAX = 1200  # absolute max — force split even mid-line

    def _chunk_buffer(self, page_buffer, book_id,
                      chapter, chapter_title, profile):
        """
        Chunk a buffer of pages into ~800-char segments.
        Splits on line boundaries to keep sentences intact.
        """
        first_page = page_buffer[0]['page']
        last_page = page_buffer[-1]['page']

        # Collect all lines from all pages
        lines = []
        for p in page_buffer:
            for line in p['text'].split('\n'):
                stripped = line.strip()
                if stripped:
                    lines.append(stripped)

        chunks = []
        current_lines = []
        current_len = 0

        for line in lines:
            line_len = len(line)

            # Flush when target reached
            if current_len + line_len > self.CHUNK_TARGET and current_lines:
                chunk_text = '\n'.join(current_lines)
                if len(chunk_text) >= self.CHUNK_MIN:
                    chunks.append(RawChunk(
                        text=chunk_text,
                        book_id=book_id,
                        chapter_number=chapter,
                        chapter_title=chapter_title,
                        page_start=first_page,
                        page_end=last_page,
                        chunk_index=len(chunks),
                        chunk_type=self._dominant_type(current_lines, profile),
                        has_table=False,
                        has_formula=self._contains_formula(chunk_text),
                    ))
                # Keep last line as overlap for context continuity
                last = current_lines[-1]
                current_lines = [last]
                current_len = len(last)

            current_lines.append(line)
            current_len += line_len

        # Flush remaining
        if current_lines:
            chunk_text = '\n'.join(current_lines)
            if len(chunk_text) >= self.CHUNK_MIN:
                chunks.append(RawChunk(
                    text=chunk_text,
                    book_id=book_id,
                    chapter_number=chapter,
                    chapter_title=chapter_title,
                    page_start=first_page,
                    page_end=last_page,
                    chunk_index=len(chunks),
                    chunk_type=self._dominant_type(current_lines, profile),
                    has_table=False,
                    has_formula=self._contains_formula(chunk_text),
                ))

        # Safety: split any chunk exceeding HARD_MAX
        final_chunks = []
        for chunk in chunks:
            if len(chunk.text) <= self.CHUNK_HARD_MAX:
                final_chunks.append(chunk)
            else:
                # Re-split oversized chunk on line boundaries
                sub_lines = chunk.text.split('\n')
                buf = []
                buf_len = 0
                for sl in sub_lines:
                    if buf_len + len(sl) > self.CHUNK_TARGET and buf:
                        final_chunks.append(RawChunk(
                            text='\n'.join(buf),
                            book_id=chunk.book_id,
                            chapter_number=chunk.chapter_number,
                            chapter_title=chunk.chapter_title,
                            page_start=chunk.page_start,
                            page_end=chunk.page_end,
                            chunk_index=len(final_chunks),
                            chunk_type=chunk.chunk_type,
                            has_table=chunk.has_table,
                            has_formula=chunk.has_formula,
                        ))
                        buf = [buf[-1]]  # overlap
                        buf_len = len(buf[0])
                    buf.append(sl)
                    buf_len += len(sl)
                if buf and buf_len >= self.CHUNK_MIN:
                    final_chunks.append(RawChunk(
                        text='\n'.join(buf),
                        book_id=chunk.book_id,
                        chapter_number=chunk.chapter_number,
                        chapter_title=chunk.chapter_title,
                        page_start=chunk.page_start,
                        page_end=chunk.page_end,
                        chunk_index=len(final_chunks),
                        chunk_type=chunk.chunk_type,
                        has_table=chunk.has_table,
                        has_formula=chunk.has_formula,
                    ))

        return final_chunks

    def _detect_chunk_type(self, para: str, profile: dict) -> str:
        """Classify a paragraph's type."""
        para_lower = para.lower()

        # Formula detection
        formula_patterns = [r'=\s*[A-Za-z]', r'\^', r'\u221a', r'\u2211',
                           r'\u03c3', r'\u03b4', r'\u0394', r'\u2202']
        if any(re.search(p, para) for p in formula_patterns):
            return 'FORMULA'

        # Code detection
        code_patterns = [r'def ', r'\bif\s+\w.*:', r'for\s+\w+\s+in\s',
                        r'^import ', r'print\(', r'return ']
        if any(re.search(p, para) for p in code_patterns):
            return 'CODE'

        # Empirical finding detection
        empirical_markers = profile.get('empirical_markers', [])
        if any(m in para_lower for m in empirical_markers):
            return 'EMPIRICAL'

        # Summary detection
        summary_marker = profile.get('summary_marker')
        if summary_marker and summary_marker.lower() in para_lower:
            return 'SUMMARY'

        # Psychology detection
        psych_markers = ['fear', 'greed', 'discipline', 'emotion',
                        'belief', 'confidence', 'anxiety', 'control']
        if any(m in para_lower for m in psych_markers):
            return 'PSYCHOLOGY'

        # Force type if configured (e.g. Douglas → all PSYCHOLOGY)
        if profile.get('all_chunks_type'):
            return profile['all_chunks_type']

        return 'RULE'

    def _contains_formula(self, text: str) -> bool:
        return bool(re.search(r'[=\^\u221a\u2211\u03c3\u03b4\u0394\u2202]', text))

    def _dominant_type(self, paras: list, profile: dict) -> str:
        if profile.get('all_chunks_type'):
            return profile['all_chunks_type']
        # Classify the full joined text rather than voting on individual lines
        full_text = '\n'.join(paras)
        return self._detect_chunk_type(full_text, profile)

    def _table_to_chunk(self, table, book_id, chapter,
                         chapter_title, page, table_id='T0'):
        """Convert extracted table to a RawChunk."""
        rows = [' | '.join(str(c) for c in row if c)
                for row in table if any(row)]
        table_text = '\n'.join(rows)
        return RawChunk(
            text=table_text,
            book_id=book_id,
            chapter_number=chapter,
            chapter_title=chapter_title,
            page_start=page,
            page_end=page,
            chunk_index=table_id,
            chunk_type='TABLE',
            has_table=True,
            has_formula=False,
        )


# ================================================================
# CLI ENTRY POINT
# ================================================================
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Ingest a trading book PDF')
    parser.add_argument('--book', required=True,
                       choices=PDFIngester.BOOK_PROFILES.keys(),
                       help='Book ID')
    parser.add_argument('--pdf', required=True, help='Path to PDF file')
    args = parser.parse_args()

    ingester = PDFIngester()
    chunks = ingester.ingest_book(args.pdf, args.book)

    print(f"Ingested {len(chunks)} chunks from {args.book}")
    for i, chunk in enumerate(chunks[:5]):
        print(f"\n--- Chunk {i} (Ch.{chunk.chapter_number}, "
              f"p.{chunk.page_start}-{chunk.page_end}, "
              f"type={chunk.chunk_type}) ---")
        print(chunk.text[:200] + "...")
