"""
utils/download_generator.py

Builds the downloadable summary file in whichever format the user
picks on the results page: plain text, Word (.docx), or PDF.

Beyond the summary text and key highlights, this also includes:
- Code listings pulled from the source document, shown exactly as
  written (in a monospaced font) rather than run through the AI
  summarizer, which would risk breaking the code.
- Diagrams/figures pulled from the source document (PDF only), so a
  textbook's key drawings aren't lost from the summary.

Each function returns the finished file as bytes, ready to send
straight to the browser - nothing is saved permanently on the server.

FIX (2026-07-27):
  _split_sections previously only matched "## Section N" headings
  (literal text "Section" followed by a number). The extractive
  summarizer produces arbitrary headings like "## Introduction" or
  "## Related Work". The regex now handles any ## heading so
  structured summaries are preserved in PDF and DOCX downloads.
"""

import io
import os
import re
import unicodedata
from xml.sax.saxutils import escape

from docx import Document
from docx.shared import Inches, Pt
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Preformatted, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from datetime import datetime


_PDF_FONT_FILES = {
    "arial": ("arial.ttf", "arialbd.ttf"),
    "calibri": ("calibri.ttf", "calibrib.ttf"),
    "cambria": ("cambria.ttf", "cambriab.ttf"),
    "consolas": ("consola.ttf", "consolab.ttf"),
    "courier": (None, None),
    "georgia": ("georgia.ttf", "georgiab.ttf"),
    "helvetica": (None, None),
    "tahoma": ("tahoma.ttf", "tahomabd.ttf"),
    "times": (None, None),
    "verdana": ("verdana.ttf", "verdanab.ttf"),
}


def _resolve_pdf_font(value):
    key = str(value or "helvetica").strip().lower()
    if key not in _PDF_FONT_FILES:
        key = "helvetica"
    if key == "courier":
        return "Courier"
    if key == "helvetica":
        return "Helvetica"
    if key == "times":
        return "Times-Roman"

    regular_file, bold_file = _PDF_FONT_FILES[key]
    font_dir = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
    regular_path = os.path.join(font_dir, regular_file)
    bold_path = os.path.join(font_dir, bold_file)
    regular_name = f"Export-{key}"
    bold_name = f"{regular_name}-Bold"
    if not os.path.exists(regular_path):
        return "Helvetica"
    if regular_name not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont(regular_name, regular_path))
    if os.path.exists(bold_path) and bold_name not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont(bold_name, bold_path))
    return regular_name


def _clean_heading_text(heading_str):
    """Strip Table of Contents page number tails and symbols from headings."""
    h = re.sub(r"[■□▪▫◾◽◼◻⬛⬜▮▯▬▲►▼◄◊○●◦\u25A0-\u25FF\u2B1B\u2B1C]", "", str(heading_str))
    h = re.sub(r"References\s*\d{1,4}\b", "", h, flags=re.IGNORECASE)
    h = re.sub(r"\s+\d{1,4}(?:\s+\d{1,4})*$", "", h)
    return re.sub(r"\s{2,}", " ", h).strip()


def _pdf_safe_text(text):
    """
    PDF downloads use reportlab's built-in fonts, which only support a
    limited set of characters (English plus common Western European
    letters). Any character outside that set - certain accented
    letters, for example - has no picture in that font and shows up
    as a solid black box instead of the correct letter.

    This converts accented letters to their closest plain-letter
    version, converts all square bullets and symbols to standard dashes,
    drops unprintable characters, and escapes markup for ReportLab.
    """
    s = str(text)
    # Strip References with attached TOC page numbers or dingbats first
    s = re.sub(r"References\s*[■▪•●I\s]*\s*\d{1,4}\b", "", s, flags=re.IGNORECASE)
    # Strip Table of Contents page number attachments and black box dingbats
    s = re.sub(r"[■□▪▫◾◽◼◻⬛⬜▮▯▬•●]\s*\d{1,4}\b", " ", s)
    s = re.sub(r"(?<=[a-zA-Z])I\d{1,4}\b", " ", s)

    replacements = s.translate(str.maketrans({
        "\uf0b7": "- ", "•": "- ", "‣": "- ",
        "■": "- ", "□": "- ", "▪": "- ", "▫": "- ",
        "◾": "- ", "◽": "- ", "◼": "- ", "◻": "- ",
        "⬛": "- ", "⬜": "- ", "▮": "- ", "▯": "- ", "▬": "- ",
        "–": "-", "—": "-", "−": "-", "‑": "-",
        "→": "", "←": "", "⇒": "", "⇢": "", "➜": "", "➔": "",
        "➞": "", "↔": "", "↑": "", "↓": "", "⟶": "", "➤": "",
        "“": '"', "”": '"', "‘": "'", "’": "'",
        "\u00a0": " ", "\ufffd": "",
    }))
    # Eradicate all geometric shapes, square dingbats, and PUA characters
    # that cause ReportLab to switch to ZapfDingbats font
    replacements = re.sub(r"[■□▪▫◾◽◼◻⬛⬜▮▯▬▲►▼◄◊○●◦\u25A0-\u25FF\u2B1B\u2B1C\uE000-\uF8FF]", " ", replacements)
    replacements = re.sub(r"(?<![A-Za-z0-9])®+(?=\s|[A-Za-z]|$)", "", replacements)
    # Strip residual page numbers preceded by dots
    replacements = re.sub(r"\.{3,}\s*(?:[ivxlc\d]+|\d{1,4})\b", "", replacements, flags=re.IGNORECASE)

    decomposed = unicodedata.normalize("NFKD", replacements)
    plain = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    plain = plain.encode("latin-1", errors="ignore").decode("latin-1")
    plain = re.sub(r"[■▪]", "", plain)
    return escape(plain)


def _split_sections(summary_text):
    """
    Splits markdown-formatted summary text into (intro_text, sections).

    sections is a list of (heading_str, level, content_str) tuples where
    level is 2 for ##, 3 for ###, 4 for ####.

    Handles all heading levels: ##, ###, ####.
    If there are no headings, returns (summary_text, []).
    """
    if "#" not in summary_text:
        return summary_text, []

    lines = summary_text.split("\n")
    intro_lines = []
    sections = []
    current_heading = None
    current_level = 2
    current_content = []

    for line in lines:
        stripped = line.strip()
        # Check for heading: ####, ###, ## (check longer first)
        if stripped.startswith("#### "):
            # Flush previous section
            if current_heading is not None:
                sections.append((current_heading, current_level, "\n".join(current_content).strip()))
            current_heading = _clean_heading_text(stripped[5:].strip())
            current_level = 4
            current_content = []
        elif stripped.startswith("### "):
            if current_heading is not None:
                sections.append((current_heading, current_level, "\n".join(current_content).strip()))
            current_heading = _clean_heading_text(stripped[4:].strip())
            current_level = 3
            current_content = []
        elif stripped.startswith("## "):
            if current_heading is not None:
                sections.append((current_heading, current_level, "\n".join(current_content).strip()))
            current_heading = _clean_heading_text(stripped[3:].strip())
            current_level = 2
            current_content = []
        else:
            if current_heading is None:
                intro_lines.append(line)
            else:
                current_content.append(line)

    # Flush last section
    if current_heading is not None:
        sections.append((current_heading, current_level, "\n".join(current_content).strip()))

    intro = "\n".join(intro_lines).strip()
    return intro, sections


def _strip_markdown_bold(text):
    """Remove **bold** markers for contexts that don't support markdown."""
    return text.replace("**", "")


# ---------------------------------------------------------------------------
# TXT
# ---------------------------------------------------------------------------

def build_txt_file(original_filename, summary_text, highlights=None, code_blocks=None, images=None, export_options=None, ordered_elements=None):
    """Returns the summary as a plain .txt file (bytes)."""
    if export_options is None:
        export_options = {}

    lines = []
    include_header = str(export_options.get("include_header", "")).lower() == "true" or export_options.get("include_header") is True
    if include_header:
        lines.append(f"Document: {original_filename}")
        lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d')}")
        lines.append("-" * 40)
        lines.append("")

    # No 'Summary of:' title or highlights — the summary starts directly
    # with the intro line and document structure.

    # Strip markdown bold markers for plain text output
    plain_summary = _strip_markdown_bold(summary_text)
    # Convert ##/###/#### headings to plain text underlined headings
    def _heading_to_txt(m):
        heading = m.group(1).strip()
        return f"\n{heading}\n{'-' * len(heading)}"
    plain_summary = re.sub(r"(?m)^#{2,4}\s+(.+)$", _heading_to_txt, plain_summary)
    lines.append(plain_summary)

    if images:
        lines.append("")
        lines.append("Figures")
        lines.append("-" * 7)
        for i, image in enumerate(images, start=1):
            lines.append(f"[Figure {i}, page {image['page']} - see the PDF or DOCX download to view it]")

    if code_blocks and not ordered_elements:
        lines.append("")
        lines.append("Code Listings")
        lines.append("-" * 13)
        for i, code in enumerate(code_blocks, start=1):
            lines.append(f"\n-- Listing {i} --")
            lines.append(code)

    return "\n".join(lines).encode("utf-8")


# ---------------------------------------------------------------------------
# DOCX
# ---------------------------------------------------------------------------

def build_docx_file(original_filename, summary_text, highlights=None, code_blocks=None, images=None, export_options=None):
    """Returns the summary as a Word (.docx) file (bytes)."""
    if export_options is None:
        export_options = {}

    document = Document()

    margin_inches_val = float(export_options.get("margin_inches", 1.0))
    font_size_val = float(export_options.get("font_size", 12))
    line_spacing_val = float(export_options.get("line_spacing", 1.15))
    font_family_val = {
        "arial": "Arial", "calibri": "Calibri", "cambria": "Cambria",
        "consolas": "Consolas", "courier": "Courier New", "georgia": "Georgia",
        "helvetica": "Helvetica", "tahoma": "Tahoma", "times": "Times New Roman",
        "verdana": "Verdana",
    }.get(str(export_options.get("font_family", "helvetica")).lower(), "Helvetica")

    include_header = str(export_options.get("include_header", "")).lower() == "true" or export_options.get("include_header") is True
    include_footer = str(export_options.get("include_footer", "")).lower() == "true" or export_options.get("include_footer") is True

    section = document.sections[0]
    section.left_margin = Inches(margin_inches_val)
    section.right_margin = Inches(margin_inches_val)
    section.top_margin = Inches(margin_inches_val)
    section.bottom_margin = Inches(margin_inches_val)

    if include_header:
        header = section.header
        header_para = header.paragraphs[0]
        header_para.text = original_filename

    if include_footer:
        footer = section.footer
        footer_para = footer.paragraphs[0]
        run = footer_para.add_run()
        fldChar1 = OxmlElement('w:fldChar')
        fldChar1.set(qn('w:fldCharType'), 'begin')
        instrText = OxmlElement('w:instrText')
        instrText.set(qn('xml:space'), 'preserve')
        instrText.text = "PAGE"
        fldChar2 = OxmlElement('w:fldChar')
        fldChar2.set(qn('w:fldCharType'), 'separate')
        fldChar3 = OxmlElement('w:fldChar')
        fldChar3.set(qn('w:fldCharType'), 'end')
        run._r.append(fldChar1)
        run._r.append(instrText)
        run._r.append(fldChar2)
        run._r.append(fldChar3)

    style = document.styles['Normal']
    style.font.size = Pt(font_size_val)
    style.font.name = font_family_val
    style.paragraph_format.line_spacing = line_spacing_val
    for style_name in ("Title", "Heading 1", "Heading 2", "Heading 3"):
        if style_name in document.styles:
            document.styles[style_name].font.name = font_family_val
            document.styles[style_name].paragraph_format.line_spacing = line_spacing_val

    # No 'Summary of:' title or highlights — the summary starts directly
    # with the intro line and document structure.

    # Use the fixed _split_sections that handles any ## heading
    intro, sections = _split_sections(summary_text)
    if intro:
        # Render plain intro paragraphs (strip bold markers)
        for para in intro.split("\n\n"):
            if para.strip():
                document.add_paragraph(_strip_markdown_bold(para.strip()))

    for heading, level, content in sections:
        # Map markdown levels: ## → h1, ### → h2, #### → h3
        docx_level = max(1, level - 1)
        document.add_heading(heading, level=docx_level)
        for para in content.split("\n\n"):
            if para.strip():
                document.add_paragraph(_strip_markdown_bold(para.strip()))

    if images:
        document.add_heading("Figures", level=2)
        for i, image in enumerate(images, start=1):
            try:
                document.add_picture(io.BytesIO(image["bytes"]), width=Inches(5.5))
            except Exception:
                continue
            caption = document.add_paragraph(f"Figure {i} (page {image['page']})")
            caption.style = document.styles["Caption"] if "Caption" in [s.name for s in document.styles] else caption.style

    if code_blocks:
        document.add_heading("Code Listings", level=2)
        for i, code in enumerate(code_blocks, start=1):
            document.add_paragraph(f"Listing {i}", style="Intense Quote")
            code_paragraph = document.add_paragraph()
            run = code_paragraph.add_run(code)
            run.font.name = font_family_val
            run.font.size = document.styles["Normal"].font.size

    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

def build_pdf_file(original_filename, summary_text, highlights=None, code_blocks=None, images=None, export_options=None, ordered_elements=None):
    """Returns the summary as a PDF file (bytes)."""
    if export_options is None:
        export_options = {}

    margin_inches_val = float(export_options.get("margin_inches", 0.85)) * inch
    page_orientation = export_options.get("page_orientation", "portrait")
    pagesize = landscape(letter) if page_orientation == "landscape" else letter
    font_family = _resolve_pdf_font(export_options.get("font_family", "helvetica"))
    font_size = float(export_options.get("font_size", 11.5))
    line_spacing = float(export_options.get("line_spacing", 1.4))

    buffer = io.BytesIO()

    class PageNumCanvas(canvas.Canvas):
        def __init__(self, *args, **kwargs):
            canvas.Canvas.__init__(self, *args, **kwargs)
            self.pages = []

        def showPage(self):
            self.pages.append(dict(self.__dict__))
            self._startPage()

        def save(self):
            page_count = len(self.pages)
            for page in self.pages:
                self.__dict__.update(page)
                self.draw_page_number(page_count)
                canvas.Canvas.showPage(self)
            canvas.Canvas.save(self)

        def draw_page_number(self, page_count):
            include_header = str(export_options.get("include_header", "")).lower() == "true" or export_options.get("include_header") is True
            include_footer = str(export_options.get("include_footer", "")).lower() == "true" or export_options.get("include_footer") is True
            include_page_numbers = str(export_options.get("include_page_numbers", "")).lower() == "true" or export_options.get("include_page_numbers") is True

            self.saveState()
            self.setFont("Helvetica", 9)

            if include_header:
                header_text = f"{original_filename} - {datetime.now().strftime('%Y-%m-%d')}"
                self.drawString(margin_inches_val, self._pagesize[1] - (margin_inches_val / 2), header_text)

            if include_footer:
                self.setStrokeColorRGB(0.7, 0.7, 0.7)
                self.setLineWidth(0.5)
                self.line(margin_inches_val, margin_inches_val - 10, self._pagesize[0] - margin_inches_val, margin_inches_val - 10)

            if include_page_numbers:
                text = f"Page {self._pageNumber} of {page_count}"
                self.drawRightString(self._pagesize[0] - margin_inches_val, margin_inches_val - 25, text)

            self.restoreState()

    doc = BaseDocTemplate(buffer, pagesize=pagesize,
                          leftMargin=margin_inches_val, rightMargin=margin_inches_val,
                          topMargin=margin_inches_val, bottomMargin=margin_inches_val)

    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id='normal')
    template = PageTemplate(id='test', frames=frame)
    doc.addPageTemplates([template])

    styles = getSampleStyleSheet()

    # Determine bold font: built-in Type 1 names in ReportLab are "Helvetica-Bold", "Times-Bold", etc.
    def _get_bold_font_name(base_font):
        if base_font == "Helvetica":
            return "Helvetica-Bold"
        if base_font == "Times-Roman":
            return "Times-Bold"
        if base_font == "Courier":
            return "Courier-Bold"
        candidate = f"{base_font}-Bold"
        if candidate in pdfmetrics.getRegisteredFontNames():
            return candidate
        return "Helvetica-Bold"

    bold_font = _get_bold_font_name(font_family)

    # Base typography styles with guaranteed bold headings and clean spacing
    styles["Normal"].fontName = font_family
    styles["Normal"].fontSize = font_size
    styles["Normal"].leading = font_size * line_spacing

    styles["Title"].fontName = bold_font
    styles["Title"].fontSize = 18
    styles["Title"].leading = 22
    styles["Title"].spaceAfter = 12

    styles["Heading1"].fontName = bold_font
    styles["Heading1"].fontSize = 14
    styles["Heading1"].leading = 18
    styles["Heading1"].spaceBefore = 14
    styles["Heading1"].spaceAfter = 6
    styles["Heading1"].keepWithNext = True

    styles["Heading2"].fontName = bold_font
    styles["Heading2"].fontSize = 12
    styles["Heading2"].leading = 16
    styles["Heading2"].spaceBefore = 12
    styles["Heading2"].spaceAfter = 4
    styles["Heading2"].keepWithNext = True

    from reportlab.lib.styles import ParagraphStyle
    if "Heading3" not in styles.byName:
        h3_style = ParagraphStyle(
            "Heading3",
            parent=styles["Heading2"],
            fontName=bold_font,
            fontSize=10.5,
            leading=14,
            spaceBefore=10,
            spaceAfter=4,
            keepWithNext=True,
        )
        styles.add(h3_style)
    else:
        styles["Heading3"].fontName = bold_font
        styles["Heading3"].keepWithNext = True

    styles["Code"].fontName = "Courier"
    styles["Code"].fontSize = 9.5
    styles["Code"].leading = 13

    story = [Spacer(1, 8)]

    caption_style = ParagraphStyle(
        "_FigCaption", parent=styles["Normal"],
        fontSize=styles["Normal"].fontSize * 0.9,
        italic=True,
        spaceAfter=6,
    )

    max_width = frame._width - frame.leftPadding - frame.rightPadding
    max_height = frame._height - frame.topPadding - frame.bottomPadding

    preserved = []
    if ordered_elements:
        preserved = [element for element in ordered_elements if element["type"] in ("image", "code")]
        summary_lines = summary_text.splitlines()
        placements = []
        ignored_anchor_words = {
            "a", "an", "and", "as", "at", "be", "by", "for", "from",
            "in", "is", "it", "of", "on", "or", "that", "the", "to",
            "this", "was", "with",
        }
        for element in preserved:
            anchor_words = set(re.findall(r"[a-z0-9_]+", element.get("anchor", element.get("text", "")).lower()))
            anchor_words -= ignored_anchor_words
            section_words = set(re.findall(r"[a-z0-9_]+", element.get("section_heading", "").lower()))
            section_words -= ignored_anchor_words
            best_index = len(summary_lines)
            best_score = 0
            section_start = 0
            section_end = len(summary_lines)
            if section_words:
                heading_candidates = []
                for index, line in enumerate(summary_lines):
                    if line.lstrip().startswith("#"):
                        heading_words = set(re.findall(r"[a-z0-9_]+", line.lower())) - ignored_anchor_words
                        score = len(section_words & heading_words)
                        if score:
                            heading_candidates.append((score, index))
                if heading_candidates:
                    _, section_start = max(heading_candidates)
                    following = [
                        index for index in range(section_start + 1, len(summary_lines))
                        if summary_lines[index].lstrip().startswith("#")
                    ]
                    section_end = following[0] if following else len(summary_lines)
            for index, line in enumerate(summary_lines):
                if index < section_start or index >= section_end:
                    continue
                if line.lstrip().startswith("#") or line.lstrip().startswith("[[PRESERVED_"):
                    continue
                line_words = set(re.findall(r"[a-z0-9_]+", line.lower())) - ignored_anchor_words
                score = len(anchor_words & line_words)
                if score >= 2 and score > best_score:
                    best_index, best_score = index + 1, score
            placements.append((best_index, element))
        ordered_placements = sorted(placements, key=lambda item: item[0])
        for marker_index in range(len(ordered_placements) - 1, -1, -1):
            line_index = min(len(summary_lines), ordered_placements[marker_index][0])
            summary_lines.insert(line_index, f"[[PRESERVED_{marker_index}]]")
        summary_text = "\n".join(summary_lines)

    def append_preserved(marker):
        index = int(marker[12:-2])
        element = ordered_placements[index][1]
        if element["type"] == "code":
            story.append(Preformatted(element["text"], styles["Code"]))
        else:
            image = element["image"]
            from PIL import Image as PILImage
            PILImage.open(io.BytesIO(image["bytes"])).load()
            width, height = image["width"], image["height"]
            scale = min(1.0, max_width / width, max_height / height)
            story.append(Image(io.BytesIO(image["bytes"]), width=width * scale, height=height * scale))
            story.append(Paragraph(_pdf_safe_text(f"Figure (page {image['page']})"), caption_style))
        story.append(Spacer(1, 14))

    def append_text(text, style):
        lines = text.splitlines()
        for line in lines:
            marker = re.fullmatch(r"\[\[PRESERVED_(\d+)\]\]", line.strip())
            if marker:
                append_preserved(marker.group(0))
            elif line.strip():
                cleaned = _pdf_safe_text(_strip_markdown_bold(line.strip()))
                if cleaned:
                    story.append(Paragraph(cleaned, style))
                    story.append(Spacer(1, 8))

    # Use the fixed _split_sections that handles ##/###/#### headings
    intro, sections = _split_sections(summary_text)
    if intro:
        for para in intro.split("\n\n"):
            if para.strip():
                append_text(para, styles["Normal"])

    for heading, level, content in sections:
        # Map markdown levels to PDF styles: ## → Heading1, ### → Heading2, #### → Heading3
        style_map = {2: "Heading1", 3: "Heading2", 4: "Heading3"}
        heading_style = style_map.get(level, "Heading2")
        story.append(Paragraph(_pdf_safe_text(heading), styles[heading_style]))
        story.append(Spacer(1, 8))
        for para in content.split("\n\n"):
            if para.strip():
                append_text(para, styles["Normal"])

    if not ordered_elements and images:
        story.append(Paragraph("Figures", styles["Heading2"]))
        story.append(Spacer(1, 8))
        for i, image in enumerate(images, start=1):
            try:
                # Pre-validate the image by loading it with PIL before
                # handing it to ReportLab.  If the bytes are corrupt or
                # in an unsupported format, this raises immediately
                # (caught below) instead of crashing later inside
                # doc.build() where there is no try/except.
                from PIL import Image as PILImage
                pil_img = PILImage.open(io.BytesIO(image["bytes"]))
                pil_img.load()  # force full decode

                width, height = image["width"], image["height"]
                scale = min(1.0, max_width / width, max_height / height)
                story.append(Image(io.BytesIO(image["bytes"]), width=width * scale, height=height * scale))
                story.append(Paragraph(_pdf_safe_text(f"Figure {i} (page {image['page']})"), caption_style))
                story.append(Spacer(1, 14))
            except Exception:
                # Skip images that can't be decoded — better than
                # crashing the entire PDF download.
                continue

    if code_blocks and not ordered_elements:
        story.append(Paragraph("Code Listings", styles["Heading2"]))
        story.append(Spacer(1, 8))
        code_style = styles["Code"]
        for i, code in enumerate(code_blocks, start=1):
            story.append(Paragraph(_pdf_safe_text(f"Listing {i}"), styles["Heading3"]))
            # Preformatted keeps whitespace and line breaks exactly as
            # written, unlike Paragraph - important so the code's
            # indentation isn't lost.
            safe_code = code.encode("latin-1", errors="ignore").decode("latin-1")
            story.append(Preformatted(safe_code, code_style))
            story.append(Spacer(1, 14))

    doc.build(story, canvasmaker=PageNumCanvas)
    return buffer.getvalue()


def build_download_file(file_format, original_filename, summary_text, highlights=None, code_blocks=None, images=None, export_options=None, ordered_elements=None):
    """
    Builds the download file in the requested format.
    file_format must be "txt", "docx", or "pdf".
    Returns (file_bytes, mime_type).
    """
    if export_options is None:
        export_options = {}

    if file_format == "docx":
        return (
            build_docx_file(original_filename, summary_text, highlights, code_blocks, images, export_options),
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
    if file_format == "pdf":
        return (
            build_pdf_file(original_filename, summary_text, highlights, code_blocks, images, export_options, ordered_elements),
            "application/pdf",
        )

    return (
        build_txt_file(original_filename, summary_text, highlights, code_blocks, images, export_options, ordered_elements),
        "text/plain",
    )
