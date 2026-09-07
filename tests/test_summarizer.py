"""
Tests for the map-reduce summarization logic.
NOTE: these tests load the real BART model and will download weights on
first run (requires internet + several hundred MB). Skip in offline CI
with: pytest -m "not model"
"""
import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.summarizer import summarize_long_text


@pytest.mark.model
def test_short_text_summarizes_in_one_pass():
    text = (
        "Artificial intelligence has transformed how organisations process "
        "documents. Automatic summarization tools save professionals hours "
        "of reading time by condensing lengthy reports into their key points, "
        "while preserving the essential meaning of the original text."
    )
    result = summarize_long_text(text)
    assert result["chunk_count"] == 1
    assert len(result["summary"]) > 0
