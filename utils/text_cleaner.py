"""
utils/text_cleaner.py

Normalizes raw extracted text before it hits the summarizer, and
extracts document metadata (title, course, author, lecturer, date)
for structured display at the top of the summary.

Specially optimised for presentation slides converted to PDF,
which exhibit three specific problems:

  1. REPEATED SLIDE DECORATIONS
     Every slide repeats the course code, title, lecturer name,
     department, date, and slide number in a header/footer strip.
     These carry zero summarization value and must be stripped
     completely after their first occurrence in the metadata block.

  2. FRAGMENTED BULLET POINTS
     Slide bullet points are short fragments (3-8 words) spread
     across many slides on the same topic. The merger pass groups
     consecutive short lines on a related theme into a single
     coherent paragraph so the summarizer sees continuous prose
     rather than disconnected fragments.

  3. SLIDE NUMBERS & NAVIGATION ARTEFACTS
     "Slide 3", "3/12", "3 of 12", or standalone numerals that
     appear between content blocks are stripped completely.

Public API:
  clean_text(raw)                  — ligature / encoding cleanup only
  preprocess_document(raw)         — full pipeline → (cleaned_text, metadata)
  format_metadata_block(metadata)  — renders metadata dict as markdown header
"""

import re
import unicodedata
from collections import Counter

# ---------------------------------------------------------------------------
# Ligature replacements
# ---------------------------------------------------------------------------

LIGATURE_REPLACEMENTS = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\ufb05": "st",
    "\ufb06": "st",
}


# ---------------------------------------------------------------------------
# Basic text cleaning
# ---------------------------------------------------------------------------

def clean_text(raw_text, preserve_newlines=True):
    """
    Normalises raw extracted text before it hits the summarizer.

    `preserve_newlines=True` (default): keeps single newlines so that
    heading detection in the summarizer can identify section titles.
    Set to False only for contexts that need flat prose (e.g. legacy
    single-paragraph output).
    """
    text = raw_text

    text = unicodedata.normalize("NFKC", text)

    # Fix ligature characters before anything else touches the text.
    for ligature, plain in LIGATURE_REPLACEMENTS.items():
        text = text.replace(ligature, plain)

    # If a floating accent mark had no standard combined character to
    # merge into (rare), it's still separate at this point. Since a
    # mark with nothing to attach to has no real meaning on its own,
    # it's safe to drop rather than show as a stray symbol.
    text = "".join(ch for ch in text if not unicodedata.combining(ch))

    # Drop private-use-area glyphs (show as boxes in most fonts)
    text = re.sub(r"[\uE000-\uF8FF]", "", text)

    # Strip Table of Contents page number attachments (e.g. "■201", "▪202", "I201")
    text = re.sub(r"[■□▪▫◾◽◼◻⬛⬜▮▯▬•●]\s*\d{1,4}\b", " ", text)
    text = re.sub(r"(?<=[a-zA-Z])I\d{1,4}\b", " ", text)

    # Convert black/white square bullets at start of line to standard markdown bullets
    text = re.sub(r"^[ \t]*[■□▪▫◾◽◼◻⬛⬜▮▯▬▲►▼◄◊○●◦\u25A0-\u25FF\u2B1B\u2B1C]\s*", "- ", text, flags=re.MULTILINE)
    # Replace all other square/geometric box glyphs with space to prevent black boxes
    text = re.sub(r"[■□▪▫◾◽◼◻⬛⬜▮▯▬▲►▼◄◊○●◦\u25A0-\u25FF\u2B1B\u2B1C]", " ", text)

    # Drop Unicode replacement character
    text = text.replace("\ufffd", "")

    # Rejoin hyphenated line-breaks: "sum-\nmarize" → "summarize"
    # Only join when the next line starts with a lowercase letter
    # (a heading starts with uppercase, so those are preserved).
    text = re.sub(r"-\n([a-z])", r"\1", text)

    if preserve_newlines:
        # Collapse runs of 3+ blank lines into exactly two (paragraph break)
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Collapse multiple spaces/tabs within a line
        text = re.sub(r"[ \t]{2,}", " ", text)
    else:
        # Collapse all newlines into spaces (legacy flat mode)
        text = re.sub(r"\s*\n\s*", " ", text)
        text = re.sub(r"[ \t]{2,}", " ", text)

    # Drop standalone page-number fragments like " 12 " before a capital letter
    text = re.sub(r"\s\d{1,4}\s(?=[A-Z])", " ", text)

    return text.strip()


# ---------------------------------------------------------------------------
# Post-summarization output cleaner
# ---------------------------------------------------------------------------

def clean_output_text(summary_text, metadata=None):
    """
    Enforces strict output quality rules on the final summary string.
    Called AFTER extraction so it applies to the exact text the user sees.

    Rules applied (in order):

    1.  SYMBOL STRIPPING — Remove LaTeX / dollar-sign math markers that
        appear around words in textbook-sourced PDFs:
          $UNIVAC$  →  UNIVAC
          $x_i$     →  x_i
          \\(term\\) →  term
        Also strips stray Unicode bullet characters and non-ASCII symbols
        that appear as boxes or question marks in the rendered output.

    2.  BULLET NORMALIZATION — Standardize every bullet character to a
        clean Markdown dash "- " so the frontend renders consistently:
          • UNIVAC   →  - UNIVAC
          ● item     →  - item
          ▪ item     →  - item
          * item     →  - item  (only at line start, preserving **bold**)

    3.  ONE ITEM PER LINE — Detect lines where multiple bullet characters
        appear inline (collapsed list bug) and split each item onto its
        own line:
          "- UNIVAC - EDVAC - ENIAC"  →  "- UNIVAC\\n- EDVAC\\n- ENIAC"
          "• UNIVAC • EDVAC"          →  "- UNIVAC\\n- EDVAC"

    4.  METADATA ISOLATION — If metadata is supplied, strip any extracted
        sentence that is <= 12 words AND whose normalized form contains
        the lecturer name, course code, or date. This catches fragments
        that slipped through the pre-summarization deduplication pass.

    Args:
        summary_text: the assembled summary string from _format_structured_summary()
        metadata: the metadata dict from preprocess_document() (optional)

    Returns:
        Cleaned summary string ready for display.
    """
    if not summary_text:
        return summary_text

    # ── Rule 1: LaTeX / dollar-sign math marker stripping ───────────────────
    # Pattern: $word$ or $WORD$ — remove the dollar signs, keep the word
    text = re.sub(r"\$([A-Za-z0-9_\-\+\^{}|\\]+)\$", r"\1", summary_text)
    # Inline LaTeX: \(term\) and \[term\]
    text = re.sub(r"\\\((.+?)\\\)", r"\1", text)
    text = re.sub(r"\\\[(.+?)\\\]", r"\1", text)
    # Stray lone dollar signs not part of math (e.g. "$ 500" price context)
    # only strip if the $ is directly attached to an alpha character
    text = re.sub(r"\$([A-Za-z])", r"\1", text)
    # Backslash-prefixed LaTeX commands like \textbf{}, \emph{}, \cite{}
    text = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\[a-zA-Z]+\b", "", text)
    # Strip Table of Contents page number attachments and black box dingbats (e.g. "■201", "▪202", "I201")
    text = re.sub(r"[■□▪▫◾◽◼◻⬛⬜▮▯▬•●]\s*\d{1,4}\b", " ", text)
    text = re.sub(r"(?<=[a-zA-Z])I\d{1,4}\b", " ", text)
    text = re.sub(r"References\s*[■▪•●I\s]\s*\d{1,4}\b", "", text, flags=re.IGNORECASE)

    # Strip stray Unicode box characters anywhere and non-line-start bullets
    text = re.sub(r"[■□▪▫◾◽◼◻⬛⬜▮▯▬▲►▼◄◊○●◦\u25A0-\u25FF\u2B1B\u2B1C\uE000-\uF8FF]", " ", text)
    text = re.sub(r"(?<!^)(?<![\n\r])[•●▸►→✓✔\uf0b7\uf0a7\uf0de\uf0e0\uf06e\uf076]", "", text, flags=re.MULTILINE)
    # Registered-sign runs in textbook PDFs are often decorative bullets
    # emitted before headings or list items, not meaningful prose symbols.
    text = re.sub(r"(?<![A-Za-z0-9])®+(?=\s|[A-Za-z]|$)", "", text)

    # ── Rule 2: Bullet normalization ────────────────────────────────────────
    # Normalize Unicode bullet variants (including black square bullets) at the start of a line
    text = re.sub(r"^[ \t]*[•●▪■□▫◾◽◼◻⬛⬜▮▯▬▸►→✓✔\uf0b7\uf0a7\uf0de\uf0e0\uf06e\uf076]\s*", "- ", text, flags=re.MULTILINE)
    # Normalize asterisk bullets at line start (but NOT **bold** markers)
    text = re.sub(r"^[ \t]*\*(?!\*)\s+", "- ", text, flags=re.MULTILINE)
    # Normalize em-dash / en-dash bullets at line start
    text = re.sub(r"^[ \t]*[–—]\s+", "- ", text, flags=re.MULTILINE)

    # ── Rule 3: One item per line enforcement ────────────────────────────────
    # Split lines that contain multiple inline bullet markers into separate lines.
    # These appear when the PDF extractor or merger joined multiple items:
    #   "- UNIVAC - EDVAC - ENIAC" → three separate lines
    #
    # Pattern: detect a line that starts with a bullet and contains 2+ more
    # inline bullets separated by text.
    def _split_collapsed_bullets(line):
        stripped = line.strip()
        # Match: starts with "- " or "* " and contains another "- " or "* " inline
        if re.match(r"^[-*•●▪]\s+\S", stripped):
            # Split on internal bullet markers (dash or asterisk preceded by space)
            parts = re.split(r"\s+[-•●▪]\s+", stripped)
            if len(parts) > 1:
                # Re-prefix all parts with "- "
                return "\n".join(
                    "- " + p.lstrip("-•●▪* \t") for p in parts if p.strip()
                )
        return line

    lines = text.splitlines()
    expanded = []
    for line in lines:
        expanded.append(_split_collapsed_bullets(line))
    text = "\n".join(expanded)

    # Collapse 3+ consecutive blank lines to 2 (paragraph break)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # ── Rule 4: Metadata isolation — strip residual footer fragments ─────────
    if metadata:
        # Build fingerprints from metadata values
        meta_values = []
        for key in ("lecturer", "course_code", "date", "title", "university", "department"):
            val = metadata.get(key, "")
            if val and len(val) >= 4:
                meta_values.append(val.lower().strip())

        if meta_values:
            clean_lines = []
            for line in text.splitlines():
                stripped = line.strip()
                word_count = len(stripped.split())
                # Always keep markdown headings (## / ### / ####) — these
                # are intentional structural lines, not residual fragments.
                if stripped.startswith("#"):
                    clean_lines.append(line)
                    continue
                # Always keep the intro line
                if stripped.startswith("Here's a summary"):
                    clean_lines.append(line)
                    continue
                if word_count > 12:
                    # Long lines are genuine content — always keep
                    clean_lines.append(line)
                    continue
                # Short line: check if it's just a metadata fragment
                lower = stripped.lower()
                is_meta = any(mv in lower for mv in meta_values if mv)
                if not is_meta:
                    clean_lines.append(line)
            text = "\n".join(clean_lines)
            # Re-collapse blank lines after stripping
            text = re.sub(r"\n{3,}", "\n\n", text)

    # ── Rule 5: Slide / course header pattern removal ───────────────────────
    # Final safety net: strip any output line that matches a lecture-slide
    # header pattern (date + course code) even if it slipped through all
    # preprocessing stages.  _SLIDE_HEADER_RE is defined later in this module
    # but is available at call time since the module is fully loaded.
    filtered_lines = []
    for line in text.splitlines():
        bare = line.strip().lstrip("- ").strip()
        if _SLIDE_HEADER_RE.match(bare):
            continue   # discard from summary
    # ── Rule 6: Table of Contents residual index & heading sanitization ─────
    # Strip any leftover TOC page number tails and square dingbats from headings and prose
    final_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            # Clean heading: strip trailing TOC page numbers (e.g. "Heading 201 References 202")
            cleaned_h = re.sub(r"[■□▪▫◾◽◼◻⬛⬜▮▯▬▲►▼◄◊○●◦\u25A0-\u25FF\u2B1B\u2B1C]", "", stripped)
            cleaned_h = re.sub(r"References\s*\d{1,4}\b", "", cleaned_h, flags=re.IGNORECASE)
            cleaned_h = re.sub(r"\s+\d{1,4}(?:\s+\d{1,4})*$", "", cleaned_h)
            cleaned_h = re.sub(r"\s{2,}", " ", cleaned_h).strip()
            # If the heading still has meaningful title text after '#'
            hash_prefix = re.match(r"^(#+)\s*", cleaned_h)
            if hash_prefix:
                title_part = cleaned_h[hash_prefix.end():].strip()
                if title_part and not re.match(r"^\d+(?:\.\d+)*$", title_part):
                    final_lines.append(cleaned_h)
            continue

        # For non-heading lines: strip any residual dot-leaders with page numbers
        cleaned_line = re.sub(r"\.{3,}\s*(?:[ivxlc\d]+|\d{1,4})\b", "", line, flags=re.IGNORECASE)
        cleaned_line = re.sub(r"[■□▪▫◾◽◼◻⬛⬜▮▯▬▲►▼◄◊○●◦\u25A0-\u25FF\u2B1B\u2B1C]", " ", cleaned_line)
        cleaned_line = re.sub(r"\s{2,}", " ", cleaned_line)
        if cleaned_line.strip():
            final_lines.append(cleaned_line)

    text = "\n".join(final_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()



# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------

# Patterns for metadata fields found on course/lecture slide covers.
_META_PATTERNS = {
    "course_code": [
        # "COSC 414", "CS 301", "CSC 201", "CMP 402", "EEE 302"
        re.compile(r"\b([A-Z]{2,5}\s*\d{3,4})\b"),
    ],
    "course_name": [
        # Lines that look like a course title after a colon or standalone
        re.compile(r"(?:course[:\s]+|title[:\s]+|subject[:\s]+)(.+)", re.IGNORECASE),
        # All-caps / Title Case lines that contain "Design", "Theory", etc.
        re.compile(r"^([A-Z][A-Za-z\s\-&]{8,60})$"),
    ],
    "author": [
        re.compile(r"(?:prepared\s+by|presented\s+by|compiled\s+by|author)[:\s]+(.+)", re.IGNORECASE),
        re.compile(r"(?:student[:\s]+|name[:\s]+)(.+)", re.IGNORECASE),
    ],
    "lecturer": [
        re.compile(r"(?:lecturer|instructor|facilitator|supervisor|by\s+dr\.?|by\s+prof\.?)[:\s]+(.+)", re.IGNORECASE),
        re.compile(r"\b(Dr\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b"),
        re.compile(r"\b(Prof\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b"),
    ],
    "date": [
        re.compile(r"(?:date[:\s]+|submitted[:\s]+|semester[:\s]+)(.+)", re.IGNORECASE),
        # Month Year or just Year
        re.compile(r"\b((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})\b"),
        re.compile(r"\b(20\d{2}/20\d{2})\b"),   # academic year "2025/2026"
        re.compile(r"\b(20\d{2})\b"),             # plain year
    ],
    "university": [
        re.compile(r"((?:university|polytechnic|college|institute)\s+of\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", re.IGNORECASE),
        re.compile(r"([A-Z][a-z]+\s+State\s+University)", re.IGNORECASE),
    ],
    "department": [
        re.compile(r"(?:department\s+of|dept\.?\s+of)[:\s]+(.+)", re.IGNORECASE),
        re.compile(r"\b(Department\s+of\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b"),
    ],
}

# Candidate title: a short, prominently-placed line near the top of the doc
_TITLE_LINE_RE = re.compile(r"^[A-Z][A-Za-z0-9\s\-:,&/']{5,80}$")


def _extract_metadata(lines):
    """
    Scans the first 40 lines of the document for metadata fields.
    Returns a dict with whatever fields were found.
    """
    meta = {}
    scan = lines[:40]

    # The first non-empty line that is reasonably short and title-case / ALL-CAPS
    # is the document title.
    for line in scan:
        stripped = line.strip()
        if not stripped or len(stripped) < 5:
            continue
        # Likely a title: ≤ 12 words, starts with uppercase
        if len(stripped.split()) <= 12 and stripped[0].isupper():
            if _TITLE_LINE_RE.match(stripped) and "title" not in meta:
                meta["title"] = stripped
                break

    # Scan all lines for structured fields
    for line in scan:
        stripped = line.strip()
        if not stripped:
            continue

        for field, patterns in _META_PATTERNS.items():
            if field in meta:
                continue
            for pat in patterns:
                m = pat.search(stripped)
                if m:
                    value = m.group(1).strip() if m.lastindex else m.group(0).strip()
                    # Sanity: value should be at least 2 chars and not just numbers
                    if len(value) >= 2 and not value.isdigit():
                        meta[field] = value
                    break

    return meta


# ---------------------------------------------------------------------------
# Repetitive line deduplication
# ---------------------------------------------------------------------------

def _normalize_line(line):
    """Lowercased, punctuation-stripped version of a line for comparison."""
    return re.sub(r"[^a-z0-9\s]", "", line.lower()).strip()


def _deduplicate_lines(lines, min_occurrences=2, min_length=8, full_remove_threshold=4):
    """
    Removes repetitive lines across the whole document.

    Two tiers:
    - Lines appearing >= full_remove_threshold times (default 4) are COMPLETELY
      removed (all occurrences). At 4+ repetitions they are definitively
      decorative slide headers / running footers with zero content value.
    - Lines appearing 2-3 times: keep the FIRST occurrence only, in case
      the line is a legitimate repeated structural element (e.g. a chapter
      title that also appears in the TOC).

    Lines shorter than `min_length` characters are skipped to avoid
    accidentally removing short but legitimate lines.
    """
    normalized = [_normalize_line(l) for l in lines]
    counts = Counter(n for n in normalized if len(n) >= min_length)

    fully_remove = {norm for norm, cnt in counts.items() if cnt >= full_remove_threshold}
    keep_once    = {norm for norm, cnt in counts.items()
                    if min_occurrences <= cnt < full_remove_threshold}

    seen_keep_once = set()
    result = []
    for line, norm in zip(lines, normalized):
        if norm in fully_remove:
            continue          # drop every occurrence
        if norm in keep_once:
            if norm in seen_keep_once:
                continue      # drop subsequent occurrences
            seen_keep_once.add(norm)
        result.append(line)
    return result


# Regex that matches a typical lecture-slide running header line:
# e.g. "10-Sep-25 COSC 414: DISTRIBUTED OPERATING SYSTEM BY DR. YAHAYA LAMIDO"
# Pattern: (optional date)(optional course code)(course title)(BY name)
_SLIDE_HEADER_RE = re.compile(
    r"^(?:\d{1,2}[\-/]\w+[\-/]\d{2,4}\s+)?"
    r"[A-Z]{2,6}\s*\d{3,4}[:\s]",
    re.IGNORECASE,
)


def _strip_lecture_slide_headers(lines):
    """
    Removes lines that match a lecture/course slide header pattern,
    regardless of how many times they appear.

    Targets lines like:
      "10-Sep-25 COSC 414: DISTRIBUTED OPERATING SYSTEM BY DR. YAHAYA LAMIDO"
      "CSC 301: Introduction to Programming - Dr. Musa"
      "EEE 402 POWER ELECTRONICS Semester 2, 2025"

    These are never part of the document body content.
    """
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if _SLIDE_HEADER_RE.match(stripped):
            continue   # drop slide course-header line
        cleaned.append(line)
    return cleaned



# Slide number patterns — standalone whole-line (anchored)
_SLIDE_NUM_RE = re.compile(
    r"^\s*"
    r"(?:"
    r"slide\s*\d+"
    r"|s\d{1,3}\b"
    r"|\d{1,3}\s*/\s*\d{1,3}"
    r"|\d{1,3}\s+of\s+\d{1,3}"
    r"|page\s*\d+"
    r"|\d{1,3}"
    r")\s*$",
    re.IGNORECASE,
)

# Slide number patterns — inline (not anchored, used as substitution)
_SLIDE_NUM_INLINE_RE = re.compile(
    r"\b(?:slide\s*\d+|page\s*\d+|\d{1,3}\s*/\s*\d{1,3}|\d{1,3}\s+of\s+\d{1,3})\b",
    re.IGNORECASE,
)


def _strip_slide_numbers(lines):
    """
    Removes slide-navigation artefacts from a list of text lines.

    Two passes:
      1. Whole-line removal — lines that consist entirely of a slide
         number pattern (e.g. "Slide 3", "3/12", "Page 4", "3") are
         dropped completely.
      2. Inline substitution — slide/page/fraction patterns that appear
         WITHIN an otherwise meaningful line are stripped in place
         (e.g. "COSC Slide 5" → "COSC", "Intro Page 3" → "Intro").
         Bare standalone integers are NOT stripped inline to avoid
         accidentally removing years, counts, or numbered references.
    """
    cleaned = []
    for line in lines:
        # Pass 1: drop whole-line slide number artefacts
        if _SLIDE_NUM_RE.match(line):
            continue
        # Pass 2: strip inline patterns (slide N, page N, N/M, N of M)
        stripped = _SLIDE_NUM_INLINE_RE.sub("", line).strip()
        # Drop the line if stripping left it empty or only whitespace
        if stripped:
            cleaned.append(stripped)
    return cleaned


# ---------------------------------------------------------------------------
# Page-boundary-aware running header / footer detector
# ---------------------------------------------------------------------------

def _split_into_pages(lines):
    """
    Splits a list of lines into pages.

    Strategy:
    1. If any line contains a form-feed character (\f) — the standard
       PDF page separator — split there. PDFs extracted with PyMuPDF
       insert \f between pages, giving an exact page count.
    2. Otherwise fall back to splitting on groups of 2+ consecutive
       blank lines, treating each gap as a rough page boundary.
    """
    # Flatten lines back to text for form-feed detection
    text = "\n".join(lines)
    if "\f" in text:
        pages = text.split("\f")
        return [p.splitlines() for p in pages if p.strip()]

    # Fallback: split on double blank lines
    pages, current = [], []
    blank_count = 0
    for line in lines:
        if not line.strip():
            blank_count += 1
            current.append(line)
        else:
            if blank_count >= 2 and current:
                pages.append(current)
                current = []
            blank_count = 0
            current.append(line)
    if current:
        pages.append(current)
    return pages if pages else [lines]


def _strip_running_headers_footers(lines, threshold_pct=0.25, max_words=8, min_length=4):
    """
    Detects and removes running page headers and footers.

    A running header/footer is a short line (≤ max_words words) that
    appears on many pages throughout the document.

    Algorithm:
    1. Split document into pages using form-feed characters or blank-line
       groups (see _split_into_pages).
    2. For each unique normalised short line, count how many distinct
       pages it appears on.
    3. If it appears on ≥ threshold_pct of all pages (default 25%),
       mark it as a running decoration and REMOVE ALL occurrences.
       (Unlike _deduplicate_lines which keeps the first occurrence,
       running headers carry no content value at all.)
    4. Return the cleaned line list plus the set of detected patterns
       (so callers can also use it for output-stage filtering).

    Typical examples caught:
      - Book title printed at top of every page ("The Alchemist")
      - Author name printed at every page ("Paulo Coelho")
      - Course code / lecturer on every slide ("COSC 302 | Dr. Smith")
      - Running chapter label repeated across all pages of a chapter
    """
    pages = _split_into_pages(lines)
    n_pages = len(pages)

    if n_pages < 3:
        # Not enough pages to meaningfully detect headers/footers
        return lines, set()

    # Count per-page occurrence for each short normalised line
    page_count: Counter = Counter()
    for page in pages:
        seen_on_page: set = set()
        # Check first and last few lines of each page (headers/footers appear there)
        candidate_lines = page[:4] + page[-4:] if len(page) > 6 else page
        for line in candidate_lines:
            stripped = line.strip()
            norm = _normalize_line(stripped)
            if (
                norm
                and len(norm) >= min_length
                and len(stripped.split()) <= max_words
                and norm not in seen_on_page
            ):
                seen_on_page.add(norm)
                page_count[norm] += 1

    min_pages = max(2, round(n_pages * threshold_pct))
    running = {norm for norm, cnt in page_count.items() if cnt >= min_pages}

    if not running:
        return lines, set()

    # Remove ALL occurrences of detected running headers/footers
    cleaned = []
    for line in lines:
        norm = _normalize_line(line.strip())
        if norm in running:
            continue  # drop completely — it's a page decoration
        cleaned.append(line)

    return cleaned, running


def _merge_slide_bullets(lines, min_words=3, max_words=12, gap_threshold=2):
    """
    Merge fragmented slide bullet points into continuous thematic paragraphs.

    Slide PDFs produce many very short lines (3-8 words) that are
    conceptually related but split across multiple slides. When the
    summarizer sees these as isolated fragments it treats each one as a
    separate sentence, which degrades summary quality.

    This pass groups consecutive short lines into a single sentence
    separated by "; " so the summarizer reads them as one flowing thought.
    A group is broken when:
      - A blank line appears (paragraph / slide boundary with a gap)
      - A long line appears (proper prose sentence — do not merge into it)
      - A line that looks like a heading appears (starts uppercase, no verb)

    Only lines of min_words..max_words are eligible for merging.
    Lines outside that range are kept as-is.
    """
    result = []
    buffer = []   # accumulates short bullet fragments

    def flush():
        if not buffer:
            return
        if len(buffer) == 1:
            result.append(buffer[0])
        else:
            # Join with "; " to make a single readable sentence
            merged = "; ".join(b.rstrip(".").strip() for b in buffer) + "."
            result.append(merged)
        buffer.clear()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            flush()
            result.append(line)   # preserve blank line (paragraph gap)
            continue

        word_count = len(stripped.split())

        if min_words <= word_count <= max_words:
            # Eligible bullet fragment — accumulate
            buffer.append(stripped)
        else:
            # Long prose line or heading — flush buffer first, then add
            flush()
            result.append(line)

    flush()   # end of document
    return result


# ---------------------------------------------------------------------------
# Full preprocessing pipeline
# ---------------------------------------------------------------------------

def preprocess_document(raw_text):
    """
    Full preprocessing pipeline for a document (especially slide PDFs):

    1. Basic cleaning (ligatures, encodings, whitespace)
    2. Extract metadata from the first ~40 lines (course code, lecturer, date…)
    3. Deduplicate repetitive slide headers/footers (≥2 occurrences)
    4. Strip body lines that contain metadata values (lecturer, course code, date)
    5. Strip standalone slide-number lines ("Slide 3", "3/12", "3 of 12")
    6. [NEW] Detect and remove page-level running headers/footers using
       page-boundary-aware frequency analysis (catches book titles, author
       names, course codes printed on every page/slide)
    7. Merge fragmented slide bullet points into coherent prose paragraphs
    8. Return:
       - cleaned_text: str  — body text ready for summarizer
       - metadata: dict     — extracted fields for the metadata block

    The metadata dict may contain any of:
        title, course_code, course_name, author, lecturer,
        date, university, department
    """
    # Step 1: basic cleaning
    text = clean_text(raw_text, preserve_newlines=True)

    # Step 2: split into lines for analysis
    lines = text.splitlines()

    # Step 3: extract metadata from top of document
    metadata = _extract_metadata(lines)

    # Step 4: deduplicate repetitive lines.
    # Lines appearing 4+ times are fully removed (all occurrences).
    # Lines appearing 2-3 times keep only the first occurrence.
    deduped_lines = _deduplicate_lines(lines, min_occurrences=2)

    # Step 4b: remove lecture slide course-header lines by pattern match
    # (catches "10-Sep-25 COSC 414: DISTRIBUTED OPERATING SYSTEM BY DR. …")
    deduped_lines = _strip_lecture_slide_headers(deduped_lines)

    # Step 4c: strip Table of Contents index lines
    def _is_toc_entry(line_str):
        s = line_str.strip()
        if not s:
            return False
        if re.search(r'\.{3,}\s*(?:[ivxlc\d]+|\d{1,4})\b', s, re.IGNORECASE):
            return True
        if re.search(r'^\d+\.\d+.*?[■▪•●]\s*\d{1,4}\b', s):
            return True
        if re.search(r'References\s*[■▪•●I\s]\s*\d{1,4}\b', s, re.IGNORECASE):
            return True
        return False

    deduped_lines = [line for line in deduped_lines if not _is_toc_entry(line)]

    # Step 5: build a set of metadata "fingerprints" — normalized snippets of
    # each extracted value that we can match against body lines.
    meta_fingerprints = set()
    for key in ("lecturer", "course_code", "date", "title"):
        val = metadata.get(key, "")
        if val and len(val) >= 4:
            meta_fingerprints.add(_normalize_line(val))

    # Also add individual name parts from the lecturer field so that
    # "Dr. Yahaya Lamido" catches lines containing just "Yahaya" or "Lamido"
    # (slide headers often rearrange the name or drop the title).
    lecturer_name = metadata.get("lecturer", "")
    if lecturer_name:
        # Extract surname-length words (3+ chars, not Dr/Prof/Mr/Mrs)
        name_words = [
            w.lower().strip(".,;:()")
            for w in lecturer_name.split()
            if len(w) >= 3 and w.lower().strip(".") not in
            {"dr", "prof", "mr", "mrs", "ms", "engr", "sir", "madam"}
        ]
        for nw in name_words:
            if len(nw) >= 4:  # only meaningful name parts
                meta_fingerprints.add(nw)

    # Step 6: scan ALL lines and remove any line whose normalized form
    # contains a metadata fingerprint. No line is exempt — lecturer names,
    # course codes, and dates must be stripped from EVERY page/slide,
    # not just after the cover area.
    after_meta_lines = []
    for line in deduped_lines:
        stripped = line.strip()
        if not stripped:
            after_meta_lines.append(line)
            continue
        norm = _normalize_line(stripped)
        word_count = len(stripped.split())
        is_meta_repetition = False

        # Short lines (≤20 words): check against metadata fingerprints
        if word_count <= 20:
            for fp in meta_fingerprints:
                if fp and fp in norm:
                    is_meta_repetition = True
                    break

        # Also catch standalone name lines like "Dr. Yahaya Lamido" or
        # "BY DR. YAHAYA LAMIDO" that appear as slide decorations
        if not is_meta_repetition and word_count <= 8:
            if re.match(
                r'^(?:by\s+)?(?:dr\.?|prof\.?|engr\.?|mr\.?|mrs\.?)\s+[A-Z]',
                stripped, re.IGNORECASE
            ):
                # Check if any lecturer name word is in this line
                norm_words = set(norm.split())
                for nw in name_words if lecturer_name else []:
                    if nw in norm_words:
                        is_meta_repetition = True
                        break

        if not is_meta_repetition:
            after_meta_lines.append(line)

    # Step 7: strip standalone slide-number lines (e.g. "Slide 3", "3/12")
    after_slide_strip = _strip_slide_numbers(after_meta_lines)

    # Step 8 [NEW]: page-boundary-aware running header/footer removal.
    # This catches decorative lines (book title, author name, course code)
    # that repeat across many pages but weren't caught by per-line dedup
    # because they appeared slightly differently or were near page boundaries.
    final_lines, running_patterns = _strip_running_headers_footers(
        after_slide_strip,
        threshold_pct=0.20,  # remove if appears on >=20% of all pages
        max_words=15,        # catch longer lecture/course header lines too
    )
    # Merge detected running patterns into metadata fingerprints so that
    # clean_output_text() can catch any fragments that still slip through
    meta_fingerprints |= running_patterns

    # Rejoin and re-clean spacing
    cleaned_text = "\n".join(final_lines)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
    cleaned_text = cleaned_text.strip()

    return cleaned_text, metadata



def format_metadata_block(metadata, filename=""):
    """
    Formats the extracted metadata dict as a structured markdown block
    to be prepended to the summary output.

    Returns a string with the metadata header + a page-break divider,
    or an empty string if no metadata was found.
    """
    if not metadata:
        return ""

    lines = ["## Document Information", ""]

    field_labels = [
        ("title",       "**Title:**"),
        ("course_code", "**Course Code:**"),
        ("course_name", "**Course / Subject:**"),
        ("author",      "**Author / Student:**"),
        ("lecturer",    "**Lecturer / Instructor:**"),
        ("department",  "**Department:**"),
        ("university",  "**Institution:**"),
        ("date",        "**Date / Semester:**"),
    ]

    for key, label in field_labels:
        if key in metadata:
            lines.append(f"{label} {metadata[key]}")

    if filename:
        lines.append(f"**Source File:** {filename}")

    lines.append("")
    lines.append("---")   # page-break divider
    lines.append("")

    return "\n".join(lines)
