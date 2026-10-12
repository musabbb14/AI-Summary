"""
utils/pdf_extractor.py
Extracts content from PDF files using PyMuPDF (imported as `fitz`).

MAJOR IMPROVEMENT (2026-07-28):
The previous version joined all text blocks per-page with a single space,
which passed repeated slide headers, footers, course codes, lecturer names,
page numbers, and figure captions raw into the summarizer on every page.

This version uses positional header/footer detection BEFORE text joining:
- Blocks in the top ~12% of the page height are candidate headers
- Blocks in the bottom ~10% of the page height are candidate footers
- Any block appearing verbatim (or near-verbatim) on 3+ pages in a
  header/footer zone is stripped entirely from the extracted text
- Page numbers (short numeric-only tokens) are always removed
- Figure captions (Fig. N / Table N) are deduplicated

This ensures only genuine document body content reaches the summarizer.
"""

import re
import pymupdf as fitz  # PyMuPDF (modern import)
from collections import Counter

# PyMuPDF reports a "flags" number for each span of text describing
# how it's styled. Bit value 8 means "this font is monospaced".
MONOSPACE_FLAG = 1 << 3

MIN_CODE_BLOCK_LENGTH = 40
MIN_IMAGE_DIMENSION = 120
MAX_IMAGES = 12

# A block is considered a "header" if its top edge is within the top
# HEADER_ZONE_FRACTION of the page height.
HEADER_ZONE_FRACTION = 0.12

# A block is considered a "footer" if its bottom edge is within the
# bottom FOOTER_ZONE_FRACTION of the page height.
FOOTER_ZONE_FRACTION = 0.10

# A block appearing on this many pages or more in the header/footer zone
# is treated as a repeated decoration and stripped.
REPEATED_BLOCK_THRESHOLD = 2

# Regex: page numbers in various formats
_PAGE_NUMBER_RE = re.compile(
    r"^\s*(?:page\s*)?\d{1,4}(?:\s*of\s*\d{1,4})?\s*$",
    re.IGNORECASE,
)

# Figure/Table caption patterns that should not be repeated
_FIGURE_CAPTION_RE = re.compile(
    r"^\s*(?:fig(?:ure)?|table|diagram|chart|image|exhibit|appendix)\s*\d",
    re.IGNORECASE,
)

_STRUCTURAL_PREFIX_RE = re.compile(
    r"^(?:lesson|chapter|unit|module|section|part|appendix)\b|"
    r"^\d+(?:\.\d+)*[.)]?\s+",
    re.IGNORECASE,
)

# Table of Contents patterns
_TOC_HEADING_RE = re.compile(
    r"^\s*(?:[ivxlc\d]+\s+)?(?:table\s+of\s+contents|brief\s+contents|contents)\b",
    re.IGNORECASE,
)
_TOC_LINE_RE = re.compile(
    r"(?:[■▪▫◾◽◼◻⬛⬜▮▯▬•●]\s*\d{1,4}\b)|"
    r"(?:[.\-–—_\t\s]{2,}\s*(?:[ivxlc\d]+|\d{1,4})\s*$)|"
    r"(?:References\s*[■▪•●I\s]*\s*\d{1,4}\b)|"
    r"(?:\b\d+\.\d+.*?I\d{1,4}\b)|"
    r"(?:\.{3,}\s*(?:[ivxlc\d]+|\d{1,4})\b)",
    re.IGNORECASE,
)


def _is_toc_line(text):
    """Detect lines that are pure Table of Contents index entries."""
    stripped = text.strip()
    if not stripped:
        return False
    if _TOC_LINE_RE.search(stripped):
        return True
    if re.search(r"\.{4,}", stripped):
        return True
    return False


BULLET_PREFIX_RE = re.compile(
    r"^[•\u2022\uf0b7\ufffd\u25aa\u25ab\u25cf\u25cb\u25a0\u25a1\x00-\x1f\x7f-\x9f*–—-]\s*"
)


def _is_context_heading(block, suppressed=None, suppressed_headers=None):
    """Identify a likely section label for associating later code/images."""
    if block.get("is_footer"):
        return False
    norm = block.get("normalized", "")
    hnorm = block.get("header_norm", norm)
    if suppressed and norm in suppressed:
        return False
    if suppressed_headers and hnorm in suppressed_headers:
        return False

    text = block["text"].strip()
    if _is_toc_line(text) or _is_figure_caption(text):
        return False

    # Skip person names / presentation metadata
    if re.match(r'^(?:by\s+)?(?:dr\.?|prof\.?|engr\.?|mr\.?|mrs\.?|ms\.?)\s+', text, re.IGNORECASE):
        return False
    if re.match(r'^(?:by|prepared\s+by|presented\s+by|compiled\s+by)\s+', text, re.IGNORECASE):
        return False
    if re.match(r'^(?:\d{1,2}[-/]\w+[-/]\d{2,4}|20\d{2}[/\-]20\d{2})\s*$', text):
        return False

    words = text.split()
    if not words or len(words) > 12:
        return False

    # Allow question headings: "What is Distributed Computing System?"
    if text.endswith("?"):
        if words[0].lower() in ("what", "why", "how", "who", "when", "where", "can", "is", "are", "which") and len(words) <= 10:
            return True
        return False

    if re.search(r"[.!;,]$", text):
        return False

    if re.search(r"[{}();=<>]|\b(?:class|public|private|protected|static|void|return|new|javac)\b", text, re.IGNORECASE):
        return False

    numbered = re.match(r"^(\d+\.)+\d*\s+(.+)$", text)
    if numbered:
        remainder = numbered.group(2).strip()
        rem_words = [w.lower().strip('"\'().[],:') for w in remainder.split()]
        sentence_verbs = {"is", "are", "was", "were", "has", "have", "had", "they", "we", "can", "will"}
        if any(w in sentence_verbs for w in rem_words):
            return False
        if len(remainder.split()) <= 8 and (remainder[0].isupper() or not remainder[0].isalpha()):
            return True
        return False

    if _STRUCTURAL_PREFIX_RE.match(text):
        return True

    if len(text) >= 4 and text.upper() == text and len(words) <= 8 and not re.search(r"\d{4}", text):
        return True

    # Comparison and difference headings
    if re.match(r"^(?:differences?|comparison|contrast)\s+(?:between|of)\s+.+", text, re.IGNORECASE):
        return True

    if len(words) <= 10:
        lower_words = [w.lower().strip('"\'().[],:') for w in words]
        sentence_verbs = {"is", "are", "was", "were", "has", "have", "had", "can", "could", "should", "will", "would", "that", "which", "who", "they", "we", "it", "you", "been", "being"}
        if any(w in sentence_verbs for w in lower_words):
            return False
        edge_connectors = {"the", "a", "an", "and", "or", "but", "for", "with", "from", "in", "on", "at", "to", "by", "between"}
        if lower_words[0] in edge_connectors or lower_words[-1] in edge_connectors:
            return False
        if words[0][0].isupper():
            return True

    return False

def _is_monospace_span(span):
    return bool(span.get("flags", 0) & MONOSPACE_FLAG)


def _normalize_block_text(text):
    """Collapse whitespace for comparison."""
    return re.sub(r"\s+", " ", text).strip().lower()


def _normalize_header_text(text):
    """Strip leading/trailing page numbers, roman numerals, and punctuation to match alternating running headers."""
    t = re.sub(r"\s+", " ", text).strip().lower()
    t = re.sub(r"^(?:page\s*)?(?:\d+|[ivxlcdm]+)\s*[:|\-.]?\s*", "", t)
    t = re.sub(r"\s*[:|\-.]?\s*(?:page\s*)?(?:\d+|[ivxlcdm]+)$", "", t)
    return t.strip()


def _join_extracted_lines(lines):
    """Join PDF lines while preserving bullet points and distinct subheadings."""
    result = []
    current_para = []
    for line in lines:
        line_s = line.strip()
        if not line_s or _is_toc_line(line_s):
            continue
        is_bullet = bool(BULLET_PREFIX_RE.match(line_s))
        cleaned_line = BULLET_PREFIX_RE.sub("", line_s).strip()
        if not cleaned_line:
            continue
        is_short_title = len(cleaned_line.split()) <= 6 and cleaned_line[0].isupper() and not cleaned_line.endswith((".", ",", ";"))
        if is_bullet or is_short_title or (current_para and current_para[-1].endswith(":")):
            if current_para:
                result.append(" ".join(current_para))
                current_para = []
        current_para.append(cleaned_line)
    if current_para:
        result.append(" ".join(current_para))
    return "\n\n".join(result)


def _clean_prose_artifacts(text):
    """Remove standalone decorative glyphs, dingbats, and TOC index remnants."""
    # Strip unprintable control characters that render as black boxes
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", " ", text)
    # Strip TOC page number artifacts: e.g. "■201" or "References■202" or "I201"
    text = re.sub(r"[■▪▫◾◽◼◻⬛⬜▮▯▬•●]\s*\d{1,4}\b", " ", text)
    text = re.sub(r"(?<=[a-zA-Z])I\d{1,4}\b", " ", text)
    # Strip dingbat square and geometric glyphs that cause black boxes
    text = re.sub(r"[■□▪▫◾◽◼◻⬛⬜▮▯▬▲►▼◄◊○●◦]", " ", text)
    text = re.sub(r"[\u25A0-\u25FF\u2B1B\u2B1C\uE000-\uF8FF]", " ", text)
    text = re.sub(r"(?<![A-Za-z0-9])®+(?=\s|[A-Za-z]|$)", "", text)
    text = text.replace("\ufffd", "")
    text = re.sub(r"[^\S\r\n]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def _split_inline_code(text):
    """Split flattened C-like code runs from surrounding explanatory prose."""
    start_pattern = re.compile(
        r"\b(?:import\s+[\w.]+;|(?:public\s+|private\s+|protected\s+|abstract\s+)?"
        r"(?:class|interface|struct)\s+[A-Za-z_]\w*)"
    )
    segments = []
    cursor = 0
    while True:
        match = start_pattern.search(text, cursor)
        if not match:
            if text[cursor:].strip():
                segments.append(("text", text[cursor:].strip()))
            break
        brace_start = text.find("{", match.end())
        if brace_start < 0:
            cursor = match.end()
            continue
        depth = 0
        end = None
        for index in range(brace_start, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    end = index + 1
                    break
        if end is None:
            cursor = match.end()
            continue
        if text[cursor:match.start()].strip():
            segments.append(("text", text[cursor:match.start()].strip()))
        code = text[match.start():end]
        segments.append(("code", code))
        cursor = end
    return segments or [("text", text)]


def _format_inline_code(code):
    """Restore readable line structure to flattened brace-based code."""
    lines = []
    current = []
    indent = 0
    for char in code.strip():
        if char == "{":
            current.append(char)
            lines.append("    " * indent + "".join(current).strip())
            current = []
            indent += 1
        elif char == "}":
            if "".join(current).strip():
                lines.append("    " * indent + "".join(current).strip())
            current = []
            indent = max(0, indent - 1)
            lines.append("    " * indent + "}")
        elif char == ";":
            current.append(char)
            lines.append("    " * indent + "".join(current).strip())
            current = []
        else:
            current.append(char)
    if "".join(current).strip():
        lines.append("    " * indent + "".join(current).strip())
    return "\n".join(lines)


def _collect_page_blocks(filepath):
    """
    First pass: read all pages and collect every text block with its
    positional bounding box. Returns a list of per-page block lists.
    """
    all_pages = []
    with fitz.open(filepath) as doc:
        for page in doc:
            page_rect = page.rect
            page_height = page_rect.height or 1
            page_width  = page_rect.width  or 1
            page_blocks = []
            page_dict = page.get_text("dict")
            for block in page_dict.get("blocks", []):
                if block.get("type") != 0:
                    continue
                bbox = block.get("bbox", (0, 0, 0, 0))  # x0,y0,x1,y1
                y_top = bbox[1]
                y_bot = bbox[3]

                # Collect spans for font classification
                block_lines = []
                monospace_chars = 0
                total_chars = 0
                for line in block.get("lines", []):
                    line_text = "".join(
                        span.get("text", "") for span in line.get("spans", [])
                    )
                    if not line_text.strip():
                        continue
                    block_lines.append(line_text)
                    for span in line.get("spans", []):
                        st = span.get("text", "")
                        total_chars += len(st)
                        if _is_monospace_span(span):
                            monospace_chars += len(st)

                if not block_lines:
                    continue

                block_text = _join_extracted_lines(block_lines)
                normalized = _normalize_block_text(block_text)
                header_norm = _normalize_header_text(block_text)

                is_header_zone = (y_top / page_height) < HEADER_ZONE_FRACTION
                is_footer_zone = (y_bot / page_height) > (1 - FOOTER_ZONE_FRACTION)
                monospace_ratio = monospace_chars / total_chars if total_chars else 0
                code_syntax = re.search(
                    r"(?:/\*|\*/|\b(?:class|public|private|protected|static|void|return)\b|[{};])",
                    block_text,
                )
                is_code = (
                    monospace_ratio > 0.6
                    and (len(block_text.strip()) >= MIN_CODE_BLOCK_LENGTH or code_syntax)
                )

                page_blocks.append({
                    "text":          block_text,
                    "normalized":    normalized,
                    "header_norm":   header_norm,
                    "raw_lines":     block_lines,
                    "is_header":     is_header_zone,
                    "is_footer":     is_footer_zone,
                    "is_code":       is_code,
                    "bbox":          bbox,
                    "page_height":   page_height,
                    "y_top":         y_top,
                    "y_bot":         y_bot,
                })
            # Some PDFs return section headings out of object order. Use the
            # page coordinates so headings and their body content follow the
            # document's visual reading order.
            page_blocks.sort(key=lambda block: (block["y_top"], block["bbox"][0]))
            all_pages.append(page_blocks)
    return all_pages


def _detect_repeated_decorations(all_pages):
    """
    Second pass: find repeated header/footer text across pages.
    Returns:
        suppressed: set of exact normalized strings that should be suppressed.
        suppressed_headers: set of normalized header strings (without page numbers) to suppress.
    """
    zone_counts = Counter()
    zone_header_counts = Counter()
    for page_blocks in all_pages:
        for block in page_blocks:
            if block["is_header"] or block["is_footer"]:
                norm = block["normalized"]
                hnorm = block.get("header_norm", norm)
                if len(norm) >= 3:
                    zone_counts[norm] += 1
                if len(hnorm) >= 3:
                    zone_header_counts[hnorm] += 1

    all_counts = Counter()
    all_header_counts = Counter()
    for page_blocks in all_pages:
        seen_this_page_norm = set()
        seen_this_page_hnorm = set()
        for block in page_blocks:
            norm = block["normalized"]
            hnorm = block.get("header_norm", norm)
            if norm not in seen_this_page_norm and len(norm) >= 3:
                all_counts[norm] += 1
                seen_this_page_norm.add(norm)
            if hnorm not in seen_this_page_hnorm and len(hnorm) >= 3:
                all_header_counts[hnorm] += 1
                seen_this_page_hnorm.add(hnorm)

    suppressed = set()
    suppressed_headers = set()

    for norm, cnt in zone_counts.items():
        if cnt >= REPEATED_BLOCK_THRESHOLD:
            suppressed.add(norm)

    for hnorm, cnt in zone_header_counts.items():
        if cnt >= REPEATED_BLOCK_THRESHOLD:
            suppressed_headers.add(hnorm)

    total_pages = max(len(all_pages), 1)
    # Alternating running headers appear on ~20%-50% of pages
    header_threshold = max(2, int(total_pages * 0.15))
    for hnorm, cnt in all_header_counts.items():
        if cnt >= header_threshold:
            suppressed_headers.add(hnorm)

    for norm, cnt in all_counts.items():
        if cnt >= max(2, int(total_pages * 0.50)):
            suppressed.add(norm)

    return suppressed, suppressed_headers


def _detect_toc_pages(all_pages):
    """
    Detect Table of Contents pages in the document's front matter.
    Returns a set of 1-based page indices that are TOC pages.
    """
    total_pages = len(all_pages)
    if total_pages <= 1:
        return set()

    max_front_matter_pages = min(40, max(5, int(total_pages * 0.25)))
    toc_pages = set()
    prev_was_toc = False

    for page_num in range(1, min(total_pages + 1, max_front_matter_pages + 1)):
        page_blocks = all_pages[page_num - 1]
        all_lines = []
        for b in page_blocks:
            all_lines.extend(b.get("raw_lines", []))

        non_empty = [l.strip() for l in all_lines if l.strip()]
        if not non_empty:
            if prev_was_toc:
                toc_pages.add(page_num)
            continue

        has_heading = any(_TOC_HEADING_RE.search(l) for l in non_empty[:5])
        toc_lines = sum(1 for l in non_empty if _is_toc_line(l))
        ratio = toc_lines / len(non_empty) if non_empty else 0

        if has_heading or ratio >= 0.20 or (prev_was_toc and (ratio >= 0.12 or len(non_empty) < 6)):
            toc_pages.add(page_num)
            prev_was_toc = True
        else:
            prev_was_toc = False

    # Guard: If all pages are marked as TOC, do not discard the entire document
    if len(toc_pages) >= total_pages:
        return set()

    return toc_pages


_FRONT_MATTER_HEADINGS = {
    "preface", "foreword", "acknowledgment", "acknowledgments",
    "acknowledgement", "acknowledgements", "about our author", "about our authors",
    "about the author", "about the authors", "about the editor", "about the editors",
    "contributors", "brief contents", "contents", "table of contents",
    "list of figures", "list of tables", "copyright", "dedication",
    "author biography", "author biographies", "biography", "biographies",
}

_LESSON_START_PATTERNS = [
    re.compile(r"(?i)^\s*(?:chapter|lesson|unit|module|lecture|part)\s*(?:1|one|i\b)"),
    re.compile(r"(?i)^\s*1(?:\.0|\.1)?\s*[:.\-]?\s+(?:introduction|background|overview|fundamentals|basics|getting started)"),
    re.compile(r"(?i)^\s*1\.1\s+[A-Z]"),
    re.compile(r"(?i)^\s*(?:abstract|executive\s+summary)\s*$"),
]


def _detect_lesson_start_page(all_pages, toc_pages):
    """
    Detect the page where primary lesson/chapter content begins,
    allowing us to skip front-matter (copyright notices, publisher info,
    preface, author biographies) so the summary starts directly with the lesson.
    """
    total_pages = len(all_pages)
    if total_pages <= 1:
        return 1

    search_limit = min(40, max(5, int(total_pages * 0.25)))
    start_search_from = (max(toc_pages) + 1) if toc_pages else 1

    for page_num in range(start_search_from, min(total_pages + 1, search_limit + 1)):
        page_blocks = all_pages[page_num - 1]
        is_front_matter = False
        for b in page_blocks[:4]:
            t = b["text"].strip().lower()
            clean_h = re.sub(r"^(?:[ivxlcdm\d\s\-.:]+)", "", t).strip()
            clean_h = re.sub(r"[\d\s\-.:]+$", "", clean_h).strip()
            if clean_h in _FRONT_MATTER_HEADINGS:
                is_front_matter = True
                break
        if is_front_matter:
            continue

        for b in page_blocks:
            clean_text = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", b["text"])
            for line in clean_text.splitlines():
                line_s = re.sub(r"\s+", " ", line).strip()
                if len(line_s.split()) <= 12:
                    for pat in _LESSON_START_PATTERNS:
                        if pat.search(line_s):
                            return page_num
                if re.match(r"^\s*1\s*$", line_s):
                    return page_num

    if toc_pages and max(toc_pages) + 1 <= total_pages:
        return max(toc_pages) + 1

    return 1


def _is_page_number(text):
    """True if this block is just a page number."""
    return bool(_PAGE_NUMBER_RE.match(text.strip()))


def _is_figure_caption(text):
    """True if this block is a figure/table caption label."""
    return bool(_FIGURE_CAPTION_RE.match(text.strip()))


def extract_pdf_content(filepath, progress_callback=None):
    """
    Extracts everything needed from a PDF in two passes:

    Pass 1: Collect all text blocks with positional data.
    Pass 2: Detect repeated header/footer/watermark patterns and TOC pages.
    Pass 3: Join only genuine body content, excluding decorations and TOC.

    Returns a dict:
        {
            "text": "...",            # clean body prose, decorations removed
            "code_blocks": [str, ...] # verbatim code listings, in order
            "images": [
                {"page": int, "bytes": bytes, "ext": "png", "width": int, "height": int},
                ...
            ]
        }
    """
    # Pass 1: collect all blocks with positions
    all_pages = _collect_page_blocks(filepath)

    if not all_pages:
        raise ValueError(
            "No extractable text was found in this PDF. It may be a scanned "
            "image without OCR — try a text-based PDF instead."
        )

    # Pass 2: find repeated decorations and Table of Contents pages to suppress
    suppressed, suppressed_headers = _detect_repeated_decorations(all_pages)
    toc_pages = _detect_toc_pages(all_pages)
    lesson_start_page = _detect_lesson_start_page(all_pages, toc_pages)

    # Pass 3: extract images and build clean text
    text_parts = []
    code_blocks = []
    images = []
    ordered_elements = []
    seen_image_xrefs = set()
    seen_figure_captions = set()

    total_pages = len(all_pages)
    current_heading = ""

    with fitz.open(filepath) as doc:
        for page_idx, (page, page_blocks) in enumerate(zip(doc, all_pages), start=1):
            is_toc = page_idx in toc_pages
            is_front_matter = (page_idx < lesson_start_page)

            for block in page_blocks:
                # If this page is part of the Table of Contents or front-matter, skip its prose
                if is_toc or is_front_matter:
                    continue

                norm = block["normalized"]
                hnorm = block.get("header_norm", norm)
                text = block["text"]

                # Always skip suppressed (repeated header/footer/watermark) blocks
                if norm in suppressed:
                    continue

                # Always skip running headers (repeated textbook title / chapter running heads)
                if hnorm in suppressed_headers:
                    if block["is_header"] or block["is_footer"] or (block["y_top"] / block["page_height"] < 0.18 and len(block["raw_lines"]) <= 2):
                        continue

                # Skip standalone publisher DOI stamps
                if re.match(r"(?i)^\s*DOI:\s*10\.\d{4,9}/", text):
                    continue

                # Always skip pure page numbers or Table of Contents index lines
                if _is_page_number(text) or _is_toc_line(text):
                    continue

                # Skip standalone figure captions from body prose so they are not
                # mistakenly turned into markdown headings or mixed into text
                if _is_figure_caption(text):
                    continue

                # Route to code blocks vs prose
                if block["is_code"]:
                    code_lines = [
                        re.sub(r"^(\s*)®+\s*", r"\1", line.rstrip())
                        for line in block["raw_lines"]
                    ]
                    code_text = "\n".join(code_lines).strip()
                    code_blocks.append(code_text)
                    ordered_elements.append({
                        "type": "code", "text": code_text,
                        "page": page_idx, "y": block["y_top"],
                        "section_heading": current_heading,
                    })
                else:
                    # Join the raw lines with spaces (clean single-line prose)
                    prose = _clean_prose_artifacts(_join_extracted_lines(block["raw_lines"]))
                    inline_segments = _split_inline_code(prose)
                    for segment_index, (segment_type, segment_text) in enumerate(inline_segments):
                        if not segment_text:
                            continue
                        if segment_type == "code":
                            code_text = _format_inline_code(segment_text)
                            code_blocks.append(code_text)
                            ordered_elements.append({
                                "type": "code", "text": code_text,
                                "page": page_idx, "y": block["y_top"] + segment_index / 1000,
                                "section_heading": current_heading,
                            })
                        else:
                            text_parts.append(segment_text)
                            ordered_elements.append({
                                "type": "text", "text": segment_text,
                                "page": page_idx, "y": block["y_top"] + segment_index / 1000,
                            })
                    if _is_context_heading({**block, "text": prose}, suppressed, suppressed_headers):
                        current_heading = prose

            # Extract images from this page (skip front-matter pages like cover art and logos)
            if len(images) < MAX_IMAGES and not is_front_matter:
                _extract_page_images(doc, page, page_idx, images, seen_image_xrefs)

            for image in images:
                if image["page"] == page_idx and "_ordered" not in image:
                    image["_ordered"] = True
                    image_heading = image.get("caption_title") or current_heading
                    ordered_elements.append({
                        "type": "image", "image": image,
                        "page": page_idx, "y": image.get("y", page.rect.height),
                        "section_heading": image_heading,
                    })

            if progress_callback:
                progress_callback(page_idx, total_pages)

    if not text_parts and not code_blocks:
        raise ValueError(
            "No extractable text was found in this PDF. It may be a scanned "
            "image without OCR — try a text-based PDF instead."
        )

    full_text = "\n".join(text_parts)

    ordered_elements.sort(key=lambda item: (item["page"], item["y"]))
    previous_text = ""
    for element in ordered_elements:
        if element["type"] == "text":
            previous_text = element["text"]
        elif element["type"] in ("image", "code"):
            element["anchor"] = previous_text

    return {
        "text": full_text,
        "code_blocks": code_blocks,
        "images": images,
        "ordered_elements": ordered_elements,
    }


def _detect_figure_info_for_image(page, img_rect=None):
    """
    Detects the figure caption number (e.g. 'Figure 3.1', 'Figure 1.6', 'Fig. 2')
    and caption title associated with an image on the page based on text proximity.
    Returns (fig_num, caption_title).
    """
    try:
        blocks = page.get_text("blocks")
    except Exception:
        return None, None

    captions = []
    for b in blocks:
        # b: (x0, y0, x1, y1, text, block_no, block_type)
        if len(b) < 5:
            continue
        text = str(b[4]).strip()
        m = re.search(r"(?i)\b(?:figure|fig\.?)\s*(\d+(?:[.\-_]\d+)*)(?:\s*[:\-–—.]\s*([^\n\r]+))?", text)
        if m:
            fig_num = f"Figure {m.group(1)}"
            cap_title = m.group(2).strip() if m.group(2) else ""
            by0, by1 = b[1], b[3]
            if img_rect is not None:
                # Vertical distance to image boundary
                if by0 >= img_rect[3] - 15:       # below image (most common)
                    dist = by0 - img_rect[3]
                elif by1 <= img_rect[1] + 15:     # above image
                    dist = img_rect[1] - by1
                else:
                    dist = abs(by0 - img_rect[1])
            else:
                dist = 0
            starts_with_fig = bool(re.match(r"(?i)^\s*(?:figure|fig\.?)\s*\d", text))
            captions.append((0 if starts_with_fig else 1, dist, fig_num, cap_title))

    if captions:
        captions.sort(key=lambda x: (x[0], x[1]))
        return captions[0][2], captions[0][3]
    return None, None


def _detect_figure_number_for_image(page, img_rect=None):
    """
    Detects the figure caption number (e.g. 'Figure 3.1', 'Figure 1.6', 'Fig. 2')
    associated with an image on the page based on text proximity.
    """
    fig_num, _ = _detect_figure_info_for_image(page, img_rect)
    return fig_num


def _extract_page_images(doc, page, page_number, images, seen_image_xrefs):
    """Pulls out images on one page that are large enough to be a real
    diagram or figure, skipping ones already seen (a logo repeated on
    every page, for example) and tiny decorative graphics."""
    for image_info in page.get_images(full=True):
        xref = image_info[0]
        if xref in seen_image_xrefs:
            continue
        seen_image_xrefs.add(xref)

        try:
            extracted = doc.extract_image(xref)
        except Exception:
            continue

        width  = extracted.get("width",  0)
        height = extracted.get("height", 0)
        if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
            continue

        rects = page.get_image_rects(xref)
        img_rect = rects[0] if rects else None
        fig_num, cap_title = _detect_figure_info_for_image(page, img_rect)
        y_pos = min((rect[1] for rect in rects), default=page.rect.height) if rects else page.rect.height

        images.append({
            "page":          page_number,
            "bytes":         extracted["image"],
            "ext":           extracted.get("ext", "png"),
            "width":         width,
            "height":        height,
            "y":             y_pos,
            "figure_number": fig_num,
            "caption_title": cap_title,
        })

        if len(images) >= MAX_IMAGES:
            return


def extract_text_from_pdf(filepath, progress_callback=None):
    """
    Kept for backward compatibility with anything that only wants the
    plain prose text (no code/image extraction).
    """
    return extract_pdf_content(filepath, progress_callback)["text"]
