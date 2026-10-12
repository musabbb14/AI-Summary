"""
utils/chunking.py

Why this file exists:
The AI model (BART) can only read about 1,024 "tokens" (roughly words
and word-pieces) at once. A textbook chapter is way longer than that.
So before we summarize a long document, we cut it into smaller pieces
("chunks") that each fit inside that limit.

We also keep a small overlap between chunks (a few sentences repeated
at the start of the next chunk), so the AI doesn't lose context right
at the boundary between two chunks.

Performance note (2026-07-27):
The original implementation called tokenizer.encode() once per sentence
inside the chunking loop. For a 50-page document with 3,000+ sentences
that meant 3,000 individual tokenizer calls, each with significant Python
overhead, adding 1-3 minutes before summarization even started.

This version uses a single batch tokenizer call to get all sentence
lengths up front, then does the chunking with a plain integer lookup.
The chunking logic is identical; only the length-counting is faster.

FIX (2026-07-27): The previous sentence splitter used a naive
(?<=[.!?])\\s+ lookbehind that broke on:
  - Common abbreviations: "Dr. Smith", "Fig. 3", "e.g. a method"
  - Decimal numbers: "Section 3.14 covers..."
  - Initialisms: "U.S. Department"
This caused hundreds of junk fragments under 5 words that were silently
filtered out, gutting the summary. The new splitter uses a two-pass
approach: first protect known non-sentence-ending patterns with
placeholders, then split on sentence boundaries.
"""

import re
from config import Config


# ---------------------------------------------------------------------------
# Abbreviation protection for sentence splitting
# ---------------------------------------------------------------------------

# Common abbreviations that end in a period but are NOT sentence endings.
# The splitter replaces "." in these with a private placeholder so the
# sentence-boundary regex doesn't split on them.
_ABBREVS = (
    # Titles / honorifics
    r"Mr", r"Mrs", r"Ms", r"Dr", r"Prof", r"Rev", r"Sr", r"Jr",
    r"Gen", r"Sgt", r"Cpl", r"Pvt", r"Lt", r"Col", r"Maj", r"Cdr",
    # Academic / scientific
    r"Fig", r"Eq", r"Vol", r"No", r"pp", r"ed", r"eds", r"et\sal",
    r"etc", r"vs", r"i\.e", r"e\.g", r"cf", r"ca", r"approx",
    r"dept", r"univ", r"assoc", r"corp", r"inc", r"ltd",
    # US States / geography initials (single uppercase letter)
    r"[A-Z]",
    # Numbers before a period (list items: "1. Two items" → keep together)
    # handled separately below
)

# Pre-compile a single pattern that matches any abbreviation + period
# NOT followed by a sentence-ending uppercase or end-of-string.
_ABBREV_RE = re.compile(
    r"(?<!\w)(?:" + r"|".join(_ABBREVS) + r")\.",
    re.IGNORECASE,
)

# Decimal numbers: "3.14", "version 2.0" — protect the dot
_DECIMAL_RE = re.compile(r"(\d+)\.(\d)")

# Ellipses — never a sentence boundary on their own
_ELLIPSIS_RE = re.compile(r"\.\.\.")

_PLACEHOLDER = "\x00"  # null byte — never appears in real text


def split_into_sentences(text):
    """
    Splits a block of text into a list of sentences.

    This implementation protects common abbreviations, decimal numbers,
    and ellipses from being treated as sentence boundaries, fixing the
    major bug where "Dr. Smith" or "Fig. 3" caused false splits that
    produced hundreds of junk fragments under 5 words.
    """
    if not text:
        return []

    # 1. Protect ellipses (three dots are never a lone sentence end)
    protected = _ELLIPSIS_RE.sub(_PLACEHOLDER * 3, text)

    # 2. Protect decimal numbers
    protected = _DECIMAL_RE.sub(r"\1" + _PLACEHOLDER + r"\2", protected)

    # 3. Protect known abbreviations
    protected = _ABBREV_RE.sub(lambda m: m.group(0).replace(".", _PLACEHOLDER), protected)

    # 3b. Protect list item numbering (e.g. "1. ", "2. ", "10. ") so the number isn't severed from the sentence
    protected = re.sub(r"(?<!\w)(\d{1,3})\.(?=\s+)", r"\1" + _PLACEHOLDER, protected)

    # 4. Split on sentence-ending punctuation followed by whitespace
    #    (now that we've hidden the false-positive dots).
    #    We also handle: "end." or "end!" at end-of-string.
    pieces = re.split(r"(?<=[.!?])\s+", protected)

    # 5. Restore placeholders and clean up
    results = []
    for piece in pieces:
        restored = piece.replace(_PLACEHOLDER, ".")
        stripped = restored.strip()
        if stripped:
            results.append(stripped)

    return results


def count_tokens(text, tokenizer):
    """Returns how many tokens a piece of text takes up."""
    return len(tokenizer.encode(text, add_special_tokens=False))


def _batch_count_tokens(sentences, tokenizer):
    """
    Returns a list of token counts for each sentence, computed in ONE
    batch tokenizer call instead of N individual calls.

    This replaces the original per-sentence tokenizer.encode() loop that
    was adding 1-3 minutes of overhead on large documents (3000+ sentences
    each triggering a separate Python→tokenizer round trip).

    The tokenizer's batch mode handles padding internally and is dramatically
    faster than repeated individual calls for large sentence lists.
    """
    if not sentences:
        return []
    # encode_batch processes all sentences at once with no padding
    # (padding=False means each is encoded to its natural length)
    batch = tokenizer(
        sentences,
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    return [len(ids) for ids in batch["input_ids"]]


def chunk_text(text, tokenizer, chunk_size=None, overlap=None):
    """
    Splits `text` into a list of chunks.

    Each chunk:
      - stays under `chunk_size` tokens
      - never cuts a sentence in half
      - starts with a small overlap of sentences from the end of the
        previous chunk, so context carries over

    Performance: sentence token counts are computed in a single batch
    call rather than one tokenizer call per sentence.
    """
    chunk_size = chunk_size or Config.CHUNK_TOKEN_SIZE
    overlap = overlap or Config.CHUNK_OVERLAP_TOKENS

    sentences = split_into_sentences(text)
    if not sentences:
        return []

    # Compute all sentence token lengths in one batch call (fast path).
    token_counts = _batch_count_tokens(sentences, tokenizer)

    chunks = []
    current_chunk_sentences = []
    current_chunk_tokens = 0

    for sentence, sentence_tokens in zip(sentences, token_counts):
        # If adding this sentence would go over the limit, close the
        # current chunk and start a new one.
        if current_chunk_tokens + sentence_tokens > chunk_size and current_chunk_sentences:
            chunks.append(" ".join(current_chunk_sentences))

            # Start the new chunk with the last few sentences of the
            # old chunk, so there's some shared context (the overlap).
            overlap_sentences = []
            overlap_tokens = 0
            # Walk backwards through current_chunk_sentences, collecting
            # their precomputed lengths from token_counts where possible.
            # For overlap sentences we re-use individual counts from the
            # already-computed list (no extra tokenizer calls).
            for s in reversed(current_chunk_sentences):
                # Find the token count for this sentence from our precomputed list.
                # Since sentences are unique within a chunk this lookup is safe.
                try:
                    s_idx = sentences.index(s)
                    s_tokens = token_counts[s_idx]
                except ValueError:
                    s_tokens = count_tokens(s, tokenizer)  # fallback (rare)
                if overlap_tokens + s_tokens > overlap:
                    break
                overlap_sentences.insert(0, s)
                overlap_tokens += s_tokens

            current_chunk_sentences = overlap_sentences
            current_chunk_tokens = overlap_tokens

        current_chunk_sentences.append(sentence)
        current_chunk_tokens += sentence_tokens

    if current_chunk_sentences:
        chunks.append(" ".join(current_chunk_sentences))

    return chunks
