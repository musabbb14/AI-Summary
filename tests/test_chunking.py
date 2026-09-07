"""Tests for token-aware chunking."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.chunking import split_into_sentences


def test_sentence_split_basic():
    text = "This is one sentence. This is another! Is this a third?"
    sentences = split_into_sentences(text)
    assert len(sentences) == 3
    assert sentences[0].startswith("This is one")


def test_sentence_split_handles_empty_string():
    assert split_into_sentences("") == []
