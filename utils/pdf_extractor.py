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
import fitz  # PyMuPDF
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


def _is_context_heading(block):
    """Identify a likely section label for associating later code/images."""
    if block["is_header"] or block["is_footer"]:
        return False
    text = block["text"].strip()
    if _is_toc_line(text):
        return False
    words = text.split()
    if not words or len(words) > 8 or re.search(r"[.!?;,]$", text):
        return False
    if re.search(r"[{}();=<>]|\b(?:class|public|private|protected|static|void|return|new|javac)\b", text, re.IGNORECASE):
        return False
    numbered = re.match(r"^\d+(?:\.\d+)*[.)]?\s+(.+)$", text)
    if numbered:
        remainder = numbered.group(1).strip()
        if not remainder or remainder[0].islower() or re.match(r"^[\d.]+(?:\s|$)", remainder):
            return False
    if _STRUCTURAL_PREFIX_RE.match(text):
        return True
    if len(text) >= 6 and text.upper() == text:
        return True
    lower_words = {word.strip("\"'()[],:;").lower() for word in words}
    prose_words = {
        "a", "an", "and", "are", "as", "be", "because", "can", "for",
        "from", "has", "have", "in", "is", "it", "may", "of", "on",
        "or", "that", "the", "this", "to", "was", "were", "with",
    }
    return len(words) <= 6 and not (lower_words & prose_words) and text[0].isupper()

def _is_monospace_span(span):
    return bool(span.get("flags", 0) & MONOSPACE_FLAG)


def _normalize_block_text(text):
    """Collapse whitespace for comparison."""
    return re.sub(r"\s+", " ", text).strip().lower()


def _join_extracted_lines(lines):
    """Join PDF lines without preserving artificial word-break hyphens."""
    joined = ""
    for line in lines:
        line = line.strip()
        if not line or _is_toc_line(line):
            continue
        if joined.endswith("-") and line[:1].islower():
            joined = joined[:-1] + line
        else:
            joined = f"{joined} {line}".strip()
    return joined


def _clean_prose_artifacts(text):
    """Remove standalone decorative glyphs, dingbats, and TOC index remnants."""
    # Strip TOC page number artifacts: e.g. "■201" or "References■202" or "I201"
    text = re.sub(r"[■▪▫◾◽◼◻⬛⬜▮▯▬•●]\s*\d{1,4}\b", " ", text)
    text = re.sub(r"(?<=[a-zA-Z])I\d{1,4}\b", " ", text)
    # Strip dingbat square and geometric glyphs that cause black boxes
    text = re.sub(r"[■□▪▫◾◽◼◻⬛⬜▮▯▬▲►▼◄◊○●◦]", " ", text)
    text = re.sub(r"[\u25A0-\u25FF\u2B1B\u2B1C\uF000-\uF8FF]", " ", text)
    text = re.sub(r"(?<![A-Za-z0-9])®+(?=\s|[A-Za-z]|$)", "", text)
    text = text.replace("\ufffd", "")
    return re.sub(r"\s{2,}", " ", text).strip()


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
    Returns a set of normalized strings that should be suppressed.
    """
    # Count how often each normalized string appears in a header/footer zone
    zone_counts = Counter()
    for page_blocks in all_pages:
        for block in page_blocks:
            if block["is_header"] or block["is_footer"]:
                norm = block["normalized"]
                if len(norm) >= 3:
                    zone_counts[norm] += 1

    # Also count blocks that appear identically on many pages even outside
    # the strict header/footer zone (e.g. watermarks, running titles in
    # the middle of the page margin area)
    all_counts = Counter()
    for page_blocks in all_pages:
        seen_this_page = set()
        for block in page_blocks:
            norm = block["normalized"]
            if norm not in seen_this_page and len(norm) >= 3:
                all_counts[norm] += 1
                seen_this_page.add(norm)

    suppressed = set()

    # Repeated in header/footer zone
    for norm, cnt in zone_counts.items():
        if cnt >= REPEATED_BLOCK_THRESHOLD:
            suppressed.add(norm)

    # Appearing on every page (or almost every page) — running titles / watermarks
    total_pages = max(len(all_pages), 1)
    for norm, cnt in all_counts.items():
        if cnt >= max(2, total_pages * 0.6):  # appears on 60%+ of pages
            suppressed.add(norm)

    return suppressed


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
    suppressed = _detect_repeated_decorations(all_pages)
    toc_pages = _detect_toc_pages(all_pages)

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
            for block in page_blocks:
                # If this page is part of the Table of Contents, skip its prose
                if is_toc:
                    continue

                norm = block["normalized"]
                text = block["text"]

                # Always skip suppressed (repeated header/footer/watermark) blocks
                if norm in suppressed:
                    continue

                # Always skip pure page numbers or Table of Contents index lines
                if _is_page_number(text) or _is_toc_line(text):
                    continue

                # De-duplicate figure captions (keep first occurrence only)
                if _is_figure_caption(text):
                    caption_key = re.sub(r"\s+", " ", text.strip().lower()[:60])
                    if caption_key in seen_figure_captions:
                        continue
                    seen_figure_captions.add(caption_key)

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
                    if _is_context_heading({**block, "text": prose}):
                        current_heading = prose

            # Extract images from this page
            if len(images) < MAX_IMAGES:
                _extract_page_images(doc, page, page_idx, images, seen_image_xrefs)

            for image in images:
                if image["page"] == page_idx and "_ordered" not in image:
                    image["_ordered"] = True
                    ordered_elements.append({
                        "type": "image", "image": image,
                        "page": page_idx, "y": image.get("y", page.rect.height),
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

        images.append({
            "page":   page_number,
            "bytes":  extracted["image"],
            "ext":    extracted.get("ext", "png"),
            "width":  width,
            "height": height,
            "y": min((rect[1] for rect in page.get_image_rects(xref)), default=page.rect.height),
        })

        if len(images) >= MAX_IMAGES:
            return


def extract_text_from_pdf(filepath, progress_callback=None):
    """
    Kept for backward compatibility with anything that only wants the
    plain prose text (no code/image extraction).
    """
    return extract_pdf_content(filepath, progress_callback)["text"]
