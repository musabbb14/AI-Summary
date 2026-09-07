"""Smoke test for the PDF extractor's error handling on invalid input."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.pdf_extractor import extract_text_from_pdf, _is_toc_line, _clean_prose_artifacts


def test_invalid_pdf_raises():
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(b"not a real pdf")
        path = f.name
    try:
        try:
            extract_text_from_pdf(path)
            assert False, "Expected an exception for a corrupt PDF"
        except Exception:
            pass
    finally:
        os.remove(path)


def test_toc_line_detection():
    assert _is_toc_line("4.12 Challenges In Iot With Big Data Analytics■201 References■202")
    assert _is_toc_line("5.1 Cloud Analytics for IoT Applications■207")
    assert _is_toc_line("Chapter Roles of Cloud Computing, Fog Computing, and Edge Computing■207")
    assert _is_toc_line("Index........................................................ xi")
    assert not _is_toc_line("This is an ordinary sentence about Internet of Things devices and fog computing.")


def test_clean_prose_artifacts_removes_black_boxes():
    dirty = "IoT systems often face security risks■201 and communication bottlenecks."
    cleaned = _clean_prose_artifacts(dirty)
    assert "■" not in cleaned
    assert "201" not in cleaned
    assert "security risks" in cleaned
