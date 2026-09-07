"""Tests for plain-text extraction."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.txt_extractor import extract_text_from_txt


def test_extract_utf8_text():
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("Hello, this is a test document.")
        path = f.name
    try:
        assert extract_text_from_txt(path) == "Hello, this is a test document."
    finally:
        os.remove(path)


def test_empty_file_raises():
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("   ")
        path = f.name
    try:
        try:
            extract_text_from_txt(path)
            assert False, "Expected ValueError for empty file"
        except ValueError:
            pass
    finally:
        os.remove(path)
