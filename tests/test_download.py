import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.download_generator import build_pdf_file, build_download_file, _pdf_safe_text


def test_pdf_safe_text_removes_black_boxes():
    dirty_text = '■ Topic 1\n▪ Subtopic A\n\uf0b7 Point with \u25fe square'
    cleaned = _pdf_safe_text(dirty_text)
    assert '■' not in cleaned
    assert '▪' not in cleaned
    assert '\u25fe' not in cleaned
    assert '- ' in cleaned


def test_build_pdf_file_generates_valid_pdf_with_margins():
    summary = '## Introduction\n\nThis is a test summary.\n\n### Methodology\n\n- Step 1\n- Step 2'
    pdf_bytes = build_pdf_file('test.pdf', summary, export_options={'margin_inches': '0.85'})
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 500
    assert pdf_bytes.startswith(b'%PDF')


def test_build_download_file_formats():
    summary = '## Section 1\n\nSome content.'
    for fmt in ['txt', 'pdf', 'docx']:
        file_bytes, mime_type = build_download_file(fmt, 'sample.txt', summary)
        assert len(file_bytes) > 0
        assert isinstance(mime_type, str)


def test_pdf_safe_text_removes_toc_and_black_boxes():
    toc_text = "4.12 Challenges In Iot With Big Data Analytics■201 References■202"
    cleaned = _pdf_safe_text(toc_text)
    assert "■" not in cleaned
    assert "201" not in cleaned
    assert "References" not in cleaned


def test_build_pdf_has_no_zapf_dingbats():
    import pymupdf as fitz
    summary = "## 4.12 Challenges In Iot With Big Data Analytics■201 References■202\n\nCloud computing details."
    pdf_bytes = build_pdf_file("test.pdf", summary)
    doc = fitz.open("pdf", pdf_bytes)
    fonts = [f[3] for p in doc for f in p.get_fonts()]
    assert "ZapfDingbats" not in fonts
