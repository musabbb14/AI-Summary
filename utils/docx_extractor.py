"""
utils/docx_extractor.py
Extracts text from .docx files using python-docx, reading paragraphs
and table cells sequentially so structured Word documents (reports,
contracts) aren't silently skipped.

IMPROVEMENT (2026-07-28):
Added deduplication of repeated paragraphs (headers, footers, running
titles, watermark paragraphs) that appear verbatim 2+ times across the
document. This mirrors the positional header/footer detection done in
pdf_extractor.py for PDF files, so DOCX documents get the same clean
body-only text handed to the summarizer.
"""

import re
from collections import Counter
from docx import Document


def _normalize(text):
    """Lower-case, whitespace-collapsed version of a string for comparison."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def extract_text_from_docx(filepath):
    document = Document(filepath)

    # ── Pass 1: collect all paragraphs with their style names ──────────────
    # python-docx exposes the paragraph style name (e.g. "Header", "Footer",
    # "Normal", "Heading 1").  We use this to detect explicit header/footer
    # paragraphs AND to count repeated lines regardless of style.
    raw_paragraphs = []
    for para in document.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        style_name = (para.style.name or "").lower()
        raw_paragraphs.append((text, style_name))

    # ── Pass 2: count occurrences of each normalized line ──────────────────
    norm_counts = Counter(_normalize(t) for t, _ in raw_paragraphs if len(t) >= 4)

    # Lines appearing 2+ times are candidate repeated decorations.
    # We also auto-suppress paragraphs whose style is "header" or "footer".
    def is_suppressed(text, style_name):
        # Suppress by explicit header/footer style
        if "header" in style_name or "footer" in style_name:
            return True
        # Suppress by repetition (2+ identical lines in the whole document)
        norm = _normalize(text)
        if len(norm) >= 4 and norm_counts[norm] >= 2:
            return True
        return False

    # ── Pass 3: build clean body text ─────────────────────────────────────
    # Keep the first occurrence of any repeated line (so section headings
    # that happen to repeat aren't silently dropped) but skip subsequent ones.
    seen_repeated = set()
    parts = []

    for text, style_name in raw_paragraphs:
        norm = _normalize(text)

        if is_suppressed(text, style_name):
            if norm not in seen_repeated:
                # Keep the very first occurrence (it may be the title or a
                # genuine heading that happens to repeat)
                seen_repeated.add(norm)
                parts.append(text)
            # Skip all subsequent occurrences
            continue

        parts.append(text)

    # ── Tables ─────────────────────────────────────────────────────────────
    # Tables are common in contracts/reports and are easy to miss.
    for table in document.tables:
        for row in table.rows:
            row_text = " | ".join(
                cell.text.strip() for cell in row.cells if cell.text.strip()
            )
            if row_text:
                parts.append(row_text)

    full_text = "\n".join(parts)

    if not full_text.strip():
        raise ValueError("No readable text was found in this DOCX file.")

    return full_text
