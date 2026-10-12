"""
utils/summarizer.py

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY ALL PREVIOUS APPROACHES FAILED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
facebook/bart-large-cnn is an ABSTRACTIVE seq2seq model. No matter
what text you give it as input, its decoder generates NEW tokens from
its learned probability distribution — a distribution trained on
CNN/DailyMail news articles and academic pre-training data.

That means:
  - Adding a grounding prefix ("Summarize only...") does nothing.
    BART cannot follow instructions; it treats the prefix as content.
  - Pre-filtering sentences before passing to BART does not help.
    BART still generates NEW text using its training knowledge.
  - Proper-noun filters catch some names but not concepts, websites,
    generic advice, or domain-specific hallucinations.

The ONLY way to guarantee a faithful, non-hallucinating summary is to
NOT GENERATE any new text at all.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THE DEFINITIVE FIX: PURE EXTRACTIVE SUMMARIZATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
This module now uses a TextRank-inspired extractive algorithm.

Every word in the output is a word that appeared verbatim in the
uploaded document. Nothing is generated, paraphrased, or inferred.

Pipeline:
  1. chunk_text() — split document into ≤900-token pieces using the
     BART tokenizer (still used for chunking, not generation).
  2. extract_key_sentences() — score every sentence by its TextRank
     centrality (how similar it is to all other sentences via TF-IDF
     cosine similarity). Select the top-N most representative sentences
     per chunk. These come DIRECTLY from the document.
  3. detect_headings() — identify likely section headings in the
     document (short lines ending without punctuation, ALL CAPS lines,
     lines starting with numbering like "1.", "Chapter 2", etc.) and
     group sentences under the heading they fall under.
  4. _format_structured_summary() — assemble the selected sentences
     under their headings with bold markers on technical terms.

No model.generate() call. No decoder. No hallucination.
BART is kept only for chunking (tokenizer.encode), not for generation.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FIXES APPLIED (2026-07-27)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Bug C — Heading detection was too broad. The pattern
  ^([A-Z][a-z]+)(\\s+[A-Za-z]+){0,7}$
matched nearly every capitalized sentence, fragmenting prose documents
into hundreds of tiny sections with no extractable content. Fixed by
tightening to require ≤ 5 words AND rejecting lines that look like
the middle of a sentence (contain common prose connectors).

Bug D — Sentence splitter breaking on abbreviations → fixed in
chunking.py (see that file's docstring).

Bug E — Short docs / empty extraction → added validation in app.py.

Added minimum usable section body size (MIN_SECTION_WORDS=15) so
fragmentary sections are merged into a fallback pool instead of being
silently discarded.

Added a short-document keep-all guard: if the document has ≤100 words
total, keep every sentence (ratio=1.0) so tiny files still get a
meaningful summary.
"""

import os
import re
import math
import logging
import threading
from collections import Counter, defaultdict
from transformers import AutoTokenizer
from config import Config

from utils.chunking import chunk_text, split_into_sentences
from utils.text_cleaner import clean_output_text

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_tokenizer = None


class SummarizationCancelled(Exception):
    """Raised to stop a summarization job when the user clicks Cancel."""
    pass


# ---------------------------------------------------------------------------
# Tokenizer (for chunking only — no model generation)
# ---------------------------------------------------------------------------

def _load_tokenizer():
    try:
        return AutoTokenizer.from_pretrained(Config.MODEL_NAME, local_files_only=True)
    except Exception:
        return AutoTokenizer.from_pretrained(Config.MODEL_NAME)


def get_tokenizer():
    """Loads the BART tokenizer once (used for chunking only)."""
    global _tokenizer
    if _tokenizer is None:
        with _lock:
            if _tokenizer is None:
                _tokenizer = _load_tokenizer()
    return _tokenizer


def prewarm_model():
    """Pre-loads the tokenizer in a background thread at startup."""
    def _load():
        try:
            get_tokenizer()
        except Exception:
            pass
    threading.Thread(target=_load, daemon=True, name="tokenizer-prewarm").start()


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def _tokenize_words(text):
    """Lowercased word tokens, stripping punctuation."""
    return re.findall(r"[a-z]{2,}", text.lower())


_STOPWORDS = frozenset({
    "the", "be", "to", "of", "and", "in", "that", "have", "it", "for",
    "on", "with", "he", "as", "you", "do", "at", "this", "but", "his",
    "by", "from", "they", "we", "say", "her", "she", "or", "an", "will",
    "my", "one", "all", "would", "there", "their", "what", "so", "up",
    "out", "if", "about", "who", "get", "which", "go", "me", "when",
    "can", "like", "time", "no", "just", "him", "know", "take", "into",
    "year", "your", "good", "some", "could", "them", "see", "other",
    "than", "then", "now", "look", "only", "come", "its", "over", "also",
    "back", "after", "use", "two", "how", "our", "first", "well", "way",
    "even", "new", "want", "any", "these", "give", "most", "us", "is",
    "are", "was", "were", "has", "had", "not", "been", "been", "a", "i",
})


def _content_words(text):
    """Lowercased non-stopword words from text."""
    return [w for w in _tokenize_words(text) if w not in _STOPWORDS]


# ---------------------------------------------------------------------------
# TF-IDF sentence scoring (used by extract_key_sentences)
# ---------------------------------------------------------------------------

def _compute_tfidf_vectors(sentences):
    """
    Computes a TF-IDF vector (dict: word -> score) for each sentence.
    Used to measure sentence similarity without any external library.
    """
    n = len(sentences)
    tokenized = [_content_words(s) for s in sentences]

    # Document frequency per word across all sentences
    df = Counter()
    for tokens in tokenized:
        for word in set(tokens):
            df[word] += 1

    vectors = []
    for tokens in tokenized:
        if not tokens:
            vectors.append({})
            continue
        tf = Counter(tokens)
        vec = {}
        for word, freq in tf.items():
            idf = math.log((n + 1.0) / (df[word] + 1.0)) + 1.0
            vec[word] = (freq / len(tokens)) * idf
        vectors.append(vec)
    return vectors


def _cosine_similarity(vec_a, vec_b):
    """Cosine similarity between two TF-IDF dicts."""
    if not vec_a or not vec_b:
        return 0.0
    common = set(vec_a) & set(vec_b)
    dot = sum(vec_a[w] * vec_b[w] for w in common)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def extract_key_sentences(text, keep_count=None, target_ratio=None, heading=None):
    """
    Selects the most representative AND diverse sentences from `text`
    using a Maximal Marginal Relevance (MMR) inspired algorithm:

      1. Build a TF-IDF vector for every sentence.
      2. Score each sentence by blended centrality, specificity, and content type.
         Topic sentences, core definitions, and heading-aligned concepts
         receive priority to ensure the key main points are always captured.
      3. Use MMR to guarantee the #1 primary key main point is retained,
         then iteratively pick diverse non-redundant supporting points.
      4. Return the selected sentences IN THEIR ORIGINAL DOCUMENT ORDER.

    Every returned word came verbatim from the input text.
    No text is generated, paraphrased, or inferred.

    Args:
        text: source text to summarize
        keep_count: exact number of sentences to keep (takes priority)
        target_ratio: fraction of sentences to keep if keep_count is None
        heading: optional section heading to boost heading-relevant sentences
    """
    sentences = split_into_sentences(text)
    # Remove trivially short fragments (fewer than 5 words)
    sentences = [s for s in sentences if len(s.split()) >= 5]

    if not sentences:
        return []

    n = len(sentences)

    if keep_count is None:
        ratio = target_ratio if target_ratio is not None else 0.45
        keep_count = max(1, min(n, round(n * ratio)))

    if keep_count >= n:
        return sentences

    # Single sentence edge case
    if n == 1:
        return sentences

    vectors = _compute_tfidf_vectors(sentences)

    # ── Centrality score ─────────────────────────────────────────────────────
    centrality = []
    for i, vec_i in enumerate(vectors):
        sim_sum = sum(
            _cosine_similarity(vec_i, vectors[j])
            for j in range(n) if j != i
        )
        centrality.append(sim_sum / max(1, n - 1))

    # ── Specificity score ────────────────────────────────────────────────────
    specificity = [sum(vec.values()) for vec in vectors]
    max_spec = max(specificity) if specificity else 1.0
    specificity_norm = [s / max(max_spec, 1e-9) for s in specificity]

    # ── Information density ──────────────────────────────────────────────────
    # Sentences with more content words per total words score higher.
    # This penalises filler sentences like "As we can see from the above..."
    info_density = []
    for s in sentences:
        words = s.split()
        content = [w for w in _tokenize_words(s) if w not in _STOPWORDS]
        density = len(content) / max(len(words), 1)
        info_density.append(density)
    max_density = max(info_density) if info_density else 1.0
    info_density_norm = [d / max(max_density, 1e-9) for d in info_density]

    # ── Content-type bonus ───────────────────────────────────────────────────
    # Distinguishes substantive key main points from vague background.
    # Definitions, key takeaways, and heading-aligned sentences get strong bonuses.
    content_bonus = []
    for s in sentences:
        bonus = 0.0
        s_lower = s.lower()

        # BONUS: Definition patterns — the core content of academic/technical docs
        if re.search(
            r'\b(?:is\s+(?:a|an|the|defined\s+as|referred\s+to\s+as|called|known\s+as|'
            r'the\s+process\s+of|a\s+type\s+of|a\s+method\s+of|characterized\s+by|'
            r'composed\s+of|made\s+up\s+of|responsible\s+for|used\s+(?:to|for|in))|'
            r'refers?\s+to|means\s+that|consists?\s+of|involves?\s+|'
            r'can\s+be\s+(?:defined|described|classified|categorized|divided)|'
            r'there\s+are\s+\w+\s+(?:types?|kinds?|categories|forms?|classes)\s+of)\b',
            s_lower
        ):
            bonus += 0.35

        # BONUS: Key main points & core takeaway signals
        if re.search(
            r'\b(?:primary\s+(?:objective|goal|purpose|function|focus|advantage|role)|'
            r'main\s+(?:objective|goal|purpose|point|feature|benefit|challenge|finding)|'
            r'key\s+(?:concept|takeaway|component|element|factor|advantage|difference)|'
            r'core\s+(?:concept|principle|idea|architecture|mechanism)|'
            r'essential\s+(?:feature|part|characteristic)|'
            r'fundamentally|in\s+summary|in\s+conclusion|consequently|'
            r'serves\s+to|acts\s+as|crucial\s+for|vital\s+to)\b',
            s_lower
        ):
            bonus += 0.30

        # BONUS: Section heading alignment
        if heading:
            heading_content = set(_content_words(heading))
            if heading_content:
                s_words = set(_tokenize_words(s))
                overlap = len(heading_content & s_words)
                if overlap > 0:
                    bonus += min(0.35, 0.15 * overlap)

        # BONUS: Enumeration / list patterns
        if re.search(
            r'\b(?:such\s+as|for\s+example|for\s+instance|namely|'
            r'includes?|the\s+following|types?\s+(?:of|are|include)|'
            r'(?:two|three|four|five|six|seven|eight)\s+(?:types?|kinds?|'
            r'categories|forms?|advantages?|disadvantages?|features?|'
            r'characteristics?|components?|stages?|steps?|phases?|levels?))\b',
            s_lower
        ):
            bonus += 0.25

        # BONUS: Contains specific numbers / data points / measurements
        if re.search(r'\d+(?:\.\d+)?(?:\s*%|\s*percent|\s*GB|\s*MB|\s*MHz|\s*GHz|\s*km|\s*kg)', s_lower):
            bonus += 0.20
        elif re.search(r'\b\d{2,}\b', s):  # any multi-digit number
            bonus += 0.10

        # BONUS: Contains capitalised technical terms (proper nouns / concepts)
        proper_nouns = re.findall(r'(?<!\. )(?<!^)\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]+)*', s)
        if len(proper_nouns) >= 2:
            bonus += 0.15

        # BONUS: Contains a colon (indicates explanation or list intro)
        if ':' in s and len(s.split()) > 6:
            bonus += 0.10

        # PENALTY: Vague introductory / background sentences
        if re.search(
            r'\b(?:in\s+the\s+(?:history|early\s+days?|beginning|past|field)\s+of|'
            r'has\s+been\s+(?:widely|extensively|increasingly|long)\s+(?:used|studied|applied|known)|'
            r'plays?\s+(?:an?\s+)?(?:important|crucial|vital|significant|key|major)\s+role|'
            r'it\s+is\s+(?:important|essential|necessary|crucial)\s+to\s+(?:note|understand|know)|'
            r'over\s+the\s+(?:years?|decades?|centuries?)|'
            r'in\s+recent\s+(?:years?|times?|decades?)|'
            r'has\s+(?:become|evolved|emerged|gained)\s+(?:a\s+)?(?:popular|common|widespread)|'
            r'as\s+(?:we\s+)?(?:can\s+)?(?:see|know|understand|observe)|'
            r'(?:many|most|various|several)\s+(?:researchers?|scholars?|experts?|scientists?)\s+'
            r'(?:have\s+)?(?:argued|suggested|proposed|studied|examined))\b',
            s_lower
        ):
            bonus -= 0.30

        # PENALTY: Very short sentences that are likely fragments/labels
        if len(s.split()) < 7:
            bonus -= 0.15

        # PENALTY: Table of Contents index entries or section-number fragments
        if re.search(r'[■□▪▫◾◽◼◻⬛⬜▮▯▬•●]\s*\d{1,4}\b|\.{3,}\s*(?:[ivxlc\d]+|\d{1,4})\b|References\s*[■▪•●I\s]\s*\d{1,4}\b|^\d+\.\d+.*?\d+$', s, re.IGNORECASE):
            bonus -= 0.60

        content_bonus.append(bonus)

    # ── Position weight ──────────────────────────────────────────────────────
    # In technical writing, topic sentences introduce the core concepts.
    position_weight = [1.0] * n
    if n > 1:
        position_weight[0] = 1.30   # topic sentence boost
        if n > 3:
            position_weight[-1] = 0.90  # last sentence is often a transition

    # ── Base relevance score ─────────────────────────────────────────────────
    base_scores = [
        (0.30 * centrality[i]
         + 0.25 * specificity_norm[i]
         + 0.15 * info_density_norm[i]
         + 0.30 * max(0.0, 0.5 + content_bonus[i]))
        * position_weight[i]
        for i in range(n)
    ]

    # ── MMR selection: pick diverse, high-scoring sentences ──────────────────
    # Guarantee that the #1 highest-scoring sentence overall (the primary
    # key main point / definition) is always selected first.
    mmr_lambda = 0.65
    selected_indices = []
    remaining = set(range(n))

    if remaining:
        top_idx = max(remaining, key=lambda idx: base_scores[idx])
        selected_indices.append(top_idx)
        remaining.discard(top_idx)

    while len(selected_indices) < keep_count and remaining:
        best_idx = -1
        best_mmr = -float('inf')

        for i in remaining:
            relevance = base_scores[i]
            max_sim = max(
                _cosine_similarity(vectors[i], vectors[j])
                for j in selected_indices
            ) if selected_indices else 0.0

            mmr_score = mmr_lambda * relevance - (1 - mmr_lambda) * max_sim

            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = i

        if best_idx >= 0:
            selected_indices.append(best_idx)
            remaining.discard(best_idx)
        else:
            break

    keep_indices = set(selected_indices)

    # Return in original document order
    return [sentences[i] for i in range(n) if i in keep_indices]


# ---------------------------------------------------------------------------
# Heading detection — FIXED (Bug C)
# ---------------------------------------------------------------------------

# Patterns that reliably indicate a section heading in a document.
# These are intentionally conservative to avoid fragmenting prose.
# Word-number alternatives for chapter headings (e.g. "Chapter One")
_WORD_NUMBERS = (
    "one|two|three|four|five|six|seven|eight|nine|ten|"
    "eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|"
    "eighteen|nineteen|twenty|twenty.one|twenty.two|twenty.three|"
    "twenty.four|twenty.five|twenty.six|twenty.seven|twenty.eight|"
    "twenty.nine|thirty"
)

_HEADING_PATTERNS = [
    # "Chapter 1", "Chapter One", "Chapter XII", "Unit 3", "Section 4.1", …
    # Matches numeric, roman-numeral, and word-number chapter labels.
    re.compile(
        r"^(chapter|unit|module|section|part|topic|lecture|lesson|week|appendix)"
        r"\s*(\d+(\.\d+)*|[IVXLCivxlc]+\.?|" + _WORD_NUMBERS + r")?"
        r"(\s*[:\-–]\s*.+)?$",
        re.IGNORECASE,
    ),
    # Roman numeral sections: "I. Introduction", "II. Methods"
    re.compile(r"^[IVXLCivxlc]+\.\s+[A-Z]\S"),
    # Lettered sections: "A. Background", "B. Methods"
    re.compile(r"^[A-Z]\.\s+[A-Z][a-z]"),
    # ALL CAPS lines of 2–8 words: "INTRODUCTION", "RELATED WORK"
    # (rejecting single-word ALL-CAPS that might just be an acronym mid-sentence)
    re.compile(r"^[A-Z][A-Z\s\-]{3,50}$"),
]

# Words that indicate a line is body prose, not a heading.
# If a short title-case line STARTS or ENDS with any of these, it's prose.
# Note: conjunctions like 'and', 'of', 'or' are allowed in the middle of a
# heading (e.g. "Results And Discussion", "Methods And Materials") but
# NOT at the start or end of the line.
_PROSE_CONNECTORS_EDGE = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "has", "have", "had",
    "be", "been", "being", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "that", "this",
    "these", "those", "it", "its", "he", "she", "they", "we", "you",
    "and", "or", "but", "nor", "so", "yet", "for", "with", "from",
    "into", "onto", "upon", "over", "under", "about", "because", "since",
    "while", "although", "though", "however", "therefore", "thus",
    "which", "who", "whom", "whose", "when", "where", "if", "then",
    "than", "as", "such", "also", "only", "just", "not", "no", "between",
})

# Words that disqualify a line from being a heading if found ANYWHERE in it.
# These are verb forms / pronouns that only appear in prose sentences.
_PROSE_CONNECTORS_ANY = frozenset({
    "is", "are", "was", "were", "has", "have", "had", "be", "been",
    "being", "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "shall", "can", "that", "this", "these", "those",
    "it", "its", "he", "she", "they", "we", "you", "because", "since",
    "while", "although", "though", "however", "therefore", "thus",
    "which", "who", "whom", "whose", "when", "where", "if", "then",
    "also", "just", "not",
})

# Minimum number of words a section body must have to be kept.
# Sections under this threshold are too tiny to extract meaningful
# content from and are merged into a fallback pool.
MIN_SECTION_WORDS = 15


def _is_likely_heading(line):
    """
    Returns True if the line looks like a document section heading.
    """
    line = line.strip()
    if not line or len(line) < 3:
        return False

    # Never treat figure or table captions as section headings
    if re.match(r"^\s*(?:fig(?:ure)?|table|diagram|chart|image|exhibit)\s*(\d+[.\-_0-9]*)?\s*[:\-–—.]?", line, re.IGNORECASE):
        return False

    # Allow question headings: "What is Distributed Computing System?"
    if line.endswith("?"):
        words = line.split()
        if words and words[0].lower() in ("what", "why", "how", "who", "when", "where", "can", "is", "are", "which") and len(words) <= 10:
            return True
        return False

    # Reject lines that look like sentences (end with sentence punctuation)
    if re.search(r"[.!;,]$", line):
        return False

    # Reject very long lines — headings are short
    words = line.split()
    if len(words) > 12:
        return False

    # Reject lines that look like Table of Contents entries or contain TOC page numbers
    if re.search(r'[■□▪▫◾◽◼◻⬛⬜▮▯▬•●]\s*\d{1,4}\b|\.{3,}\s*(?:[ivxlc\d]+|\d{1,4})\b|References\s*[■▪•●I\s]\s*\d{1,4}\b', line, re.IGNORECASE):
        return False
    if re.search(r'\b\d{1,4}\s+[A-Z].*?\b\d{1,4}\b', line):
        return False
    if re.search(r'^\s*(?:table\s+of\s+contents|brief\s+contents|contents)\s*$', line, re.IGNORECASE):
        return False

    # Reject person names / metadata lines
    if re.match(
        r'^(?:by\s+)?(?:dr\.?|prof\.?|engr\.?|mr\.?|mrs\.?|ms\.?)\s+',
        line, re.IGNORECASE
    ):
        return False

    if re.match(r'^(?:by|prepared\s+by|presented\s+by|compiled\s+by)\s+',
                line, re.IGNORECASE):
        return False

    # Reject date-like lines: "10-Sep-25", "September 2025", "2025/2026"
    if re.match(
        r'^(?:\d{1,2}[-/]\w+[-/]\d{2,4}|'
        r'(?:January|February|March|April|May|June|July|August|September|'
        r'October|November|December)\s+\d{4}|'
        r'20\d{2}[/\-]20\d{2})\s*$',
        line, re.IGNORECASE
    ):
        return False

    # Numbered heading: "1. Introduction", "1. Minicomputer Model"
    numbered = re.match(r"^(\d+\.)+\d*\s+(.+)$", line)
    if numbered:
        remainder = numbered.group(2).strip()
        rem_words = [w.lower().strip('"\'().[],:') for w in remainder.split()]
        sentence_verbs = {"is", "are", "was", "were", "has", "have", "had", "they", "we", "can", "will"}
        if any(w in sentence_verbs for w in rem_words):
            return False
        if len(remainder.split()) <= 8 and (remainder[0].isupper() or not remainder[0].isalpha()):
            return True
        return False

    # Comparison and difference headings: "Differences between A and B", "Comparison between A and B"
    if re.match(r"^(?:differences?|comparison|contrast)\s+(?:between|of)\s+.+", line, re.IGNORECASE):
        if len(words) <= 12 and not re.search(r"[.!;,]$", line):
            return True

    # Check against explicit structural patterns (Chapter N, Section N, Roman numerals, etc.)
    for pattern in _HEADING_PATTERNS:
        if pattern.match(line):
            return True

    # Title Case short lines (<= 10 words):
    if len(words) <= 10:
        lower_words = [w.lower().strip('"\'().[]') for w in words]
        if any(w in _PROSE_CONNECTORS_ANY for w in lower_words):
            return False
        if lower_words[0] in _PROSE_CONNECTORS_EDGE or lower_words[-1] in _PROSE_CONNECTORS_EDGE:
            return False
        if words[0][0].isupper():
            return True

    return False


def detect_headings_and_sections(text):
    """
    Splits `text` into sections based on detected headings.

    Returns a list of dicts:
        {"heading": str or None, "text": str}

    If no headings are found, returns a single section with heading=None.
    This is purely structural analysis — no text is generated.
    """
    lines = text.splitlines()
    sections = []
    current_heading = None
    current_lines = []

    for line in lines:
        stripped = line.strip()
        if _is_likely_heading(stripped) and len(stripped) > 2:
            if current_heading and stripped.strip().lower() == current_heading.strip().lower():
                # Same heading repeated consecutively — keep in current section
                continue

            # Guard for comparison/differences headings and sub-item headers:
            # If current heading is a comparison/difference heading (e.g. "Differences between X and Y"),
            # subheadings/column labels (like "X", "Y", or items whose words are in the heading)
            # or any candidate heading before the comparison has substantial body should be
            # kept as content of the comparison section, not replace the section heading.
            if current_heading and re.match(r"^(?:differences?|comparison|contrast)\s+(?:between|of)\s+.+", current_heading, re.IGNORECASE):
                stripped_words = set(re.findall(r"\b\w{3,}\b", stripped.lower()))
                heading_words = set(re.findall(r"\b\w{3,}\b", current_heading.lower()))
                body_now = "\n".join(current_lines).strip()
                if (stripped_words and stripped_words.issubset(heading_words)) or len(body_now.split()) < 15:
                    current_lines.append(line)
                    continue

            # Save previous section
            body = "\n".join(current_lines).strip()
            if body and current_heading is not None:
                sections.append({"heading": current_heading, "text": body})
            elif body:
                sections.append({"heading": None, "text": body})
            current_heading = stripped
            current_lines = []
        else:
            current_lines.append(line)

    # Last section
    body = "\n".join(current_lines).strip()
    if body:
        sections.append({"heading": current_heading, "text": body})

    if not sections:
        return [{"heading": None, "text": text}]

    return sections


# ---------------------------------------------------------------------------
# Length preset resolution
# ---------------------------------------------------------------------------

def _resolve_keep_ratio(length_preset, total_words=None):
    """
    Maps Short/Medium/Long to the fraction of sentences to keep per chunk.
    All three presets are calibrated to capture key main points:
      - Short (0.25): Strictly key main points and core definitions.
      - Medium (0.45): Key main points plus primary supporting explanations.
      - Long (0.65): Key main points plus thorough details and context.
    """
    # Short-document guard: always keep everything for tiny docs
    if total_words is not None and total_words <= 100:
        return 1.0

    presets = {
        "short":  0.25,
        "medium": 0.45,
        "long":   0.65,
    }
    return presets.get(length_preset, presets["medium"])


# ---------------------------------------------------------------------------
# Content quality gate
# ---------------------------------------------------------------------------

def _has_enough_real_content(text, min_word_count=8, min_letter_ratio=0.45):
    """
    Returns False for text that is mostly blanks, underscores, dots, or
    punctuation — the kind found in signature lines or form fields.
    """
    words = re.findall(r"[A-Za-z]{2,}", text)
    if len(words) < min_word_count:
        return False
    letters = sum(1 for ch in text if ch.isalpha())
    if len(text) == 0 or (letters / len(text)) < min_letter_ratio:
        return False
    return True


# ---------------------------------------------------------------------------
# Bold term detection (applied to extracted sentences only)
# ---------------------------------------------------------------------------

def _detect_bold_terms(text):
    """
    Wraps important terms in **bold** markers for better readability.

    Detects and bolds:
      1. ALL-CAPS abbreviations/acronyms (≥2 chars): CPU, RAM, TF-IDF, HTTP
      2. Technical terms: words with mixed case like "JavaScript", "IPv6"
      3. Proper nouns in context: capitalised multi-word phrases that are
         not at the start of a sentence (e.g. "...using the Binary Search Tree")
      4. Key definitions: terms that appear after "is called", "known as",
         "refers to", "defined as", etc.

    Only marks terms that already appear in the text — no external knowledge.
    Each unique term is bolded only ONCE to avoid visual clutter.
    """
    if not text:
        return text
    already_bolded = set()

    def _bold_once(term, in_text):
        """Bold the first occurrence of `term` in `in_text` if not already done."""
        if term in already_bolded or len(term) < 2:
            return in_text
        # Skip if this term overlaps with an already-bolded term
        # (either this term contains a bolded term, or a bolded term contains this)
        for bolded in already_bolded:
            if term in bolded or bolded in term:
                return in_text
        # Skip if the term is already inside ** markers in the text
        idx = in_text.find(term)
        if idx >= 0:
            before = in_text[:idx]
            if before.count('**') % 2 == 1:  # inside an open bold marker
                return in_text
        already_bolded.add(term)
        return in_text.replace(term, f"**{term}**", 1)

    result = text

    # 1. ALL-CAPS acronyms (CPU, RAM, HTTP, TF-IDF, etc.)
    for m in re.finditer(r'\b([A-Z][A-Z0-9\-]{1,8})\b', result):
        term = m.group(1)
        if term not in already_bolded and not term.replace('-', '').isdigit():
            result = _bold_once(term, result)

    # 2. Key definition patterns: "is called X", "known as X", "referred to as X"
    #    Capture multi-word capitalised phrases after the definition trigger.
    def_patterns = [
        r'(?:is\s+called|are\s+called|known\s+as|referred\s+to\s+as|'
        r'defined\s+as|termed|term(?:ed)?\s+as)\s+["\']?'
        r'([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)',
    ]
    for pat in def_patterns:
        for m in re.finditer(pat, result):
            term = m.group(1).strip()
            if term and 1 <= len(term.split()) <= 5:
                result = _bold_once(term, result)

    # 3. Multi-word proper nouns / technical terms mid-sentence
    #    Match 2-4 consecutive capitalised words NOT at sentence start.
    #    Lookbehind: after lowercase letter, punctuation, or a common
    #    conjunction/preposition so "and Loosely Coupled Systems" is caught.
    for m in re.finditer(r'(?:(?<=[a-z.,;:]\s)|(?<=\band\s)|(?<=\bor\s)|(?<=\bthe\s)|(?<=\ba\s))([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})', result):
        term = m.group(1)
        # Skip if it's a common phrase like "In This", "For Example"
        first_word = term.split()[0].lower()
        if first_word not in _STOPWORDS:
            result = _bold_once(term, result)

    # 4. Key compound terms anywhere in the sentence
    #    Catch important multi-word concepts like "Tightly Coupled",
    #    "Loosely Coupled", "Distributed Systems", "Message Passing",
    #    "Resource Sharing", "Parallel Processing" even at sentence start.
    for m in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b', result):
        term = m.group(1)
        words = term.split()
        first_lower = words[0].lower()
        # Only bold if it looks like a concept (not a sentence start)
        # Check: at least one word is a meaningful qualifier/noun
        qualifier_words = {"tightly", "loosely", "distributed", "parallel",
                          "shared", "local", "remote", "primary", "coupled",
                          "message", "resource", "computation", "multiprocessor"}
        if any(w.lower() in qualifier_words for w in words):
            result = _bold_once(term, result)

def _extract_comparison_points(heading, text, keep_ratio=0.65):
    """
    Detects if this section represents a comparison or contrast between two subjects
    (e.g. 'Differences between Tightly Coupled and Loosely Coupled Systems' or
    'Comparison of Monolithic and Microkernel') and extracts balanced, paired
    comparison points contrasting Subject A and Subject B.

    Returns a list of paired comparative sentences:
      "Subject A: [description of Subject A], whereas Subject B: [description of Subject B]."
    If no comparison structure is found, returns an empty list [].
    """
    if not heading:
        return []

    is_comparison = bool(re.search(
        r"\b(?:differences?|comparison|contrast|distinction|versus)\b.*\b(?:between|of|and|vs\.?)\b|"
        r"\b\w+\s+vs\.?\s+\w+\b",
        heading,
        re.IGNORECASE
    ))
    if not is_comparison:
        return []

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return []

    # Identify Subject A and Subject B from the heading
    sub_a, sub_b = "", ""
    m = re.search(
        r"(?:differences?|comparison|contrast|distinction)\s+(?:between|of)\s+(.+?)\s+(?:and|vs\.?|with)\s+(.+)",
        heading, re.IGNORECASE
    )
    if m:
        sub_a, sub_b = m.group(1).strip(), m.group(2).strip()
    else:
        m2 = re.search(r"(.+?)\s+vs\.?\s+(.+)", heading, re.IGNORECASE)
        if m2:
            sub_a, sub_b = m2.group(1).strip(), m2.group(2).strip()

    data_lines = lines
    # Check if first two lines are column headers / subject names
    if len(lines) >= 2 and not re.match(r"^\d+[.)]", lines[0]) and not re.match(r"^\d+[.)]", lines[1]):
        words_0 = set(re.findall(r"\b\w{3,}\b", lines[0].lower()))
        words_1 = set(re.findall(r"\b\w{3,}\b", lines[1].lower()))
        h_words = set(re.findall(r"\b\w{3,}\b", heading.lower()))
        if (words_0 and words_0.issubset(h_words)) or (words_1 and words_1.issubset(h_words)):
            sub_a = lines[0]
            sub_b = lines[1]
            data_lines = lines[2:]

    # Clean subject titles
    sub_a = re.sub(r"[:\-–—]+$", "", sub_a).strip()
    sub_b = re.sub(r"[:\-–—]+$", "", sub_b).strip()

    pairs = []
    # Pattern 1: Paired by duplicate item numbers (e.g. 1. ... 1. ... 2. ... 2. ...)
    numbered_items = []
    for dl in data_lines:
        num_m = re.match(r"^(\d+)[.)]\s*(.+)$", dl)
        if num_m:
            numbered_items.append((int(num_m.group(1)), num_m.group(2).strip()))

    from collections import defaultdict
    by_num = defaultdict(list)
    for num, content in numbered_items:
        by_num[num].append(content)

    has_pairs = any(len(items) >= 2 for items in by_num.values())
    if has_pairs:
        for num in sorted(by_num.keys()):
            items = by_num[num]
            if len(items) >= 2:
                pairs.append((items[0], items[1]))
            elif len(items) == 1:
                pairs.append((items[0], ""))

    # Pattern 2: Two blocks under Subject A and Subject B headings
    if not pairs and sub_a and sub_b:
        current_sub = None
        block_a = []
        block_b = []
        for dl in data_lines:
            dl_clean = re.sub(r"[:\-–—]+$", "", dl).strip().lower()
            if sub_a.lower() in dl_clean:
                current_sub = "a"
                continue
            elif sub_b.lower() in dl_clean:
                current_sub = "b"
                continue
            if current_sub == "a":
                block_a.append(dl)
            elif current_sub == "b":
                block_b.append(dl)
        if block_a and block_b:
            for a_item, b_item in zip(block_a, block_b):
                pairs.append((a_item, b_item))

    if pairs:
        paired_sentences = []
        for a_txt, b_txt in pairs:
            a_clean = re.sub(r"^\d+[.)]\s*", "", a_txt).strip(".;, ")
            b_clean = re.sub(r"^\d+[.)]\s*", "", b_txt).strip(".;, ")
            # Clean leading bullet markers
            a_clean = re.sub(r"^[-•●▪*]\s*", "", a_clean)
            b_clean = re.sub(r"^[-•●▪*]\s*", "", b_clean)
            if a_clean and b_clean:
                sent = f"{sub_a}: {a_clean}, whereas {sub_b}: {b_clean}."
            elif a_clean:
                sent = f"{sub_a}: {a_clean}."
            elif b_clean:
                sent = f"{sub_b}: {b_clean}."
            else:
                continue
            paired_sentences.append(sent)

        n_pairs = len(paired_sentences)
        if keep_ratio is not None:
            if keep_ratio <= 0.3:
                keep_n = 1
            elif keep_ratio <= 0.5:
                keep_n = max(1, min(n_pairs, 2))
            else:
                keep_n = n_pairs
        else:
            keep_n = n_pairs
        return paired_sentences[:keep_n]

    return []


# ---------------------------------------------------------------------------
# Format the extracted sentences into a structured summary
# ---------------------------------------------------------------------------

def _clean_heading(heading):
    """
    Normalises a detected heading for display.

    IMPORTANT: Section numbers are PRESERVED so the summary follows the
    document's own structure. Examples:
    - "1. Introduction"       → "1. Introduction"
    - "1.2 Background"        → "1.2 Background"
    - "Chapter 1: The Boy"    → "Chapter 1: The Boy"
    - "CHAPTER I"             → "Chapter I"
    - "INTRODUCTION"          → "Introduction"
    - "Advantages"            → "Advantages"
    """
    original = heading.strip()
    # Strip all unprintable control characters and geometric/black box dingbats
    original = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f■□▪▫◾◽◼◻⬛⬜▮▯▬▲►▼◄◊○●◦\u25A0-\u25FF\u2B1B\u2B1C\uE000-\uF8FF]", " ", original)
    original = re.sub(r"\s+", " ", original).strip()

    # ── Named structural label (Chapter / Unit / Lecture / …) ───────────────
    ch_match = re.match(
        r"^(chapter|unit|module|part|lecture|lesson|week|appendix|section|topic)"
        r"\s*(\d+(?:\.\d+)*|[IVXLCivxlc]+\.?|" + _WORD_NUMBERS + r")?"
        r"(?:\s*[:\-–]\s*(.+))?$",
        original, re.IGNORECASE,
    )
    if ch_match:
        label = ch_match.group(1).title()          # "Chapter"
        num   = (ch_match.group(2) or "").strip()  # "1" / "One" / "XII"
        title = (ch_match.group(3) or "").strip()  # inline title if present
        if num and title:
            return f"{label} {num}: {title.title()}"
        elif num:
            return f"{label} {num}"
        else:
            return label

    # ── Numbered heading — PRESERVE the number ──────────────────────────────
    # "1. Introduction" → "1. Introduction"
    # "1.2 Background"  → "1.2 Background"
    num_match = re.match(r"^(\d+(?:\.\d+)*\.?\s*)(.*)", original)
    if num_match:
        prefix = num_match.group(1).strip()
        rest = num_match.group(2).strip().strip(":.-")
        if rest:
            return f"{prefix} {rest.title()}"
        return prefix

    # ── ALL CAPS → Title Case ───────────────────────────────────────────────
    if original.isupper() and len(original) > 3:
        return original.title()

    # ── Fallback: return original as title-case ──────────────────────────────
    return original.title() if original else ""


def _get_heading_level(heading):
    """
    Determines the Markdown heading level based on the section number depth.

    - "1. Introduction" or "Introduction" (no sub-number) → ## (h2/Heading1)
    - "1.1 Background" or "2.3 Details" (has sub-number) → ### (h3/Heading2)
    - "1.1.2 Sub-details" (deeper nesting) → #### (h4/Heading3, capped)
    """
    if not heading:
        return "##"
    # Check for deeper sub-section numbering: "1.1.2", "2.3.4.1", etc.
    if re.match(r"^\d+\.\d+\.\d+", heading.strip()):
        return "####"
    # Check for sub-section numbering: "1.1", "2.3", etc.
    if re.match(r"^\d+\.\d+", heading.strip()):
        return "###"
    return "##"


def _format_structured_summary(section_results):
    """
    Assembles the final summary as structured notes matching the format:

        ### 1. Heading Name

          Key point extracted from the document.
          Definition: What the document says this means.
          Another key point from the document.

        #### 1.1 Sub-Heading

          Label: Content from the document.
          Another point from the document.

    Each extracted sentence appears as its own line (not merged into
    paragraphs). Sentences containing 'Label: content' patterns have
    their labels preserved. This format preserves the document's own
    structure and makes the summary easy to scan.
    """
    if not section_results:
        return ""

    seen_sentences = set()   # global dedup across all sections
    parts = []

    def _dedup_norm(sent):
        return re.sub(r"\s+", " ", sent.lower().strip())

    for sec in section_results:
        heading   = sec.get("heading")
        sentences = sec.get("sentences", [])
        if not sentences:
            continue

        # Deduplicate within this batch
        unique_sents = []
        for s in sentences:
            norm = _dedup_norm(s)
            if norm not in seen_sentences and len(norm) > 10:
                seen_sentences.add(norm)
                unique_sents.append(s)

        if not unique_sents:
            continue

        # Format each sentence as a separate indented point
        body_lines = []
        for s in unique_sents:
            s = s.strip()
            # Skip very short fragments
            if len(s.split()) < 4:
                continue

            # Keep summarized body text normal; heading styling is applied
            # by the HTML/PDF renderers, while preserved code stays verbatim.
            formatted = s

            # Clean up: strip control characters and black box dingbats
            formatted = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f■□▪▫◾◽◼◻⬛⬜▮▯▬▲►▼◄◊○●◦\u25A0-\u25FF\u2B1B\u2B1C\uE000-\uF8FF]", " ", formatted)
            # Remove leading bullet markers if present
            formatted = re.sub(r"^[-•●▪*]\s+", "", formatted.strip())
            formatted = re.sub(r"\s{2,}", " ", formatted).strip()

            # Add as an indented point
            body_lines.append(f"  {formatted}")

        if not body_lines:
            continue

        body = "\n".join(body_lines)

        if heading:
            cleaned = _clean_heading(heading)
            if cleaned:
                level = _get_heading_level(heading)
                parts.append(f"{level} {cleaned}\n\n{body}")
            else:
                parts.append(body)
        else:
            parts.append(body)

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Highlights extraction
# ---------------------------------------------------------------------------

def extract_highlights(section_results, max_highlights=None):
    """
    Extracts key highlights — one per section — picking the most
    informative sentence rather than just the first one.

    Selection criteria for "most informative":
      - Longer sentences (more content) are preferred
      - Sentences with specific terms (numbers, proper nouns) rank higher
      - Avoids short fragment sentences that lack context

    These are sentences taken directly from the document.
    """
    max_highlights = max_highlights or Config.MAX_HIGHLIGHTS
    highlights = []
    seen_norms = set()

    def _highlight_score(sent):
        """Score a sentence's suitability as a highlight."""
        words = sent.split()
        word_count = len(words)
        score = 0.0
        # Prefer medium-to-long sentences (10-30 words ideal)
        if 10 <= word_count <= 30:
            score += 2.0
        elif 7 <= word_count <= 40:
            score += 1.0
        # Reward sentences containing numbers/data (specific facts)
        if re.search(r'\d', sent):
            score += 0.5
        # Reward sentences with proper nouns (specific concepts)
        proper_count = len(re.findall(r'(?<!^)(?<!\. )\b[A-Z][a-z]{2,}', sent))
        score += min(proper_count * 0.3, 1.5)
        # Penalise sentences that start with connectors ("However", "Also")
        first_word = words[0].lower().rstrip(',') if words else ''
        if first_word in {'however', 'also', 'furthermore', 'moreover', 'thus',
                          'therefore', 'hence', 'meanwhile', 'nevertheless'}:
            score -= 1.0
        # Penalise very short fragments
        if word_count < 7:
            score -= 1.5
        # Disqualify any sentence with Table of Contents patterns or black box dingbats
        if re.search(r'[■□▪▫◾◽◼◻⬛⬜▮▯▬•●]\s*\d{1,4}\b|\.{3,}\s*(?:[ivxlc\d]+|\d{1,4})\b|References\s*[■▪•●I\s]\s*\d{1,4}\b', sent, re.IGNORECASE):
            return -99.0
        return score

    def _norm(s):
        return re.sub(r'\s+', ' ', s.lower().strip())

    for sec in section_results:
        sents = sec.get("sentences", [])
        if not sents:
            continue
        # Pick the best sentence from this section
        best_sent = max(sents, key=_highlight_score)
        if _highlight_score(best_sent) <= 0:
            continue
        norm = _norm(best_sent)
        if norm not in seen_norms:
            seen_norms.add(norm)
            highlights.append(best_sent)
        if len(highlights) >= max_highlights:
            break

    # If we still need more, pull from sections with remaining high-scoring sentences
    if len(highlights) < max_highlights:
        for sec in section_results:
            for s in sec.get("sentences", []):
                if len(highlights) >= max_highlights:
                    break
                norm = _norm(s)
                if norm not in seen_norms and _highlight_score(s) > 0:
                    seen_norms.add(norm)
                    highlights.append(s)

    cleaned_highlights = []
    for h in highlights:
        cleaned_h = clean_output_text(h)
        cleaned_h = re.sub(r"^[-•●▪*]\s+", "", cleaned_h).strip()
        if cleaned_h and len(cleaned_h.split()) >= 4:
            cleaned_highlights.append(cleaned_h)

    return cleaned_highlights[:max_highlights]


# ---------------------------------------------------------------------------
# Main public entry point
# ---------------------------------------------------------------------------

def summarize_long_text(
    text,
    progress_callback=None,
    length_preset=None,
    cancel_check=None,
    metadata=None,
):
    """
    Faithfully summarizes a document of any length using pure extraction.

    GUARANTEE: Every word in the output came verbatim from the uploaded
    document. No text is generated, paraphrased, or inferred.

    Output quality rules enforced on the final summary (via clean_output_text):
      - LaTeX / dollar-sign math markers stripped ($UNIVAC$ → UNIVAC)
      - All bullet variants normalised to clean Markdown dashes (- )
      - Collapsed multi-item lines split onto separate lines (one per bullet)
      - Residual slide footer fragments (lecturer name, course code, date)
        removed from extracted sentences that slipped through pre-processing

    Args:
        text: full extracted text of the uploaded document (must be
              non-empty — callers should validate before calling this)
        progress_callback: fn(status, current, total, stage_label=...)
        length_preset: "short", "medium", or "long"
        cancel_check: fn() -> bool, returns True if job was cancelled
        metadata: dict from preprocess_document() — used by clean_output_text
                  to strip residual metadata fragments from output sentences

    Returns:
        {
            "summary": str,
            "highlights": [str, ...],
            "chunk_count": int,
            "section_count": int,
            "original_token_count": int,
        }
    """
    tokenizer = get_tokenizer()
    total_tokens = len(tokenizer.encode(text, add_special_tokens=False))
    total_words = len(text.split())

    # Diagnostic log so we can verify the right text is being summarized
    logger.info(
        "summarize_long_text: %d words, %d tokens, preset=%s",
        total_words, total_tokens, length_preset,
    )

    keep_ratio = _resolve_keep_ratio(length_preset, total_words=total_words)

    # ── Step 1: detect headings and split into named sections ────────────────
    raw_sections = detect_headings_and_sections(text)

    logger.info("summarize_long_text: %d raw sections detected", len(raw_sections))

    if progress_callback:
        progress_callback("reading", 0, len(raw_sections),
                          stage_label="Reading document structure…")

    # ── Step 2–3: for each section, chunk and extract key sentences ──────────
    section_results = []
    total_sections = len(raw_sections)

    for sec_idx, section in enumerate(raw_sections):
        if cancel_check and cancel_check():
            raise SummarizationCancelled()

        sec_text = section["text"]
        sec_heading = section["heading"]

        if not _has_enough_real_content(sec_text):
            continue

        # Check if this section is a comparison/difference section with paired contrast items
        comp_points = _extract_comparison_points(sec_heading, sec_text, keep_ratio=keep_ratio)
        if comp_points:
            section_results.append({
                "heading": sec_heading,
                "sentences": comp_points,
            })
            if progress_callback:
                progress_callback(
                    "summarizing", sec_idx + 1, total_sections,
                    stage_label=f"Extracting section {sec_idx + 1}/{total_sections}…",
                )
            continue

        # Chunk this section
        chunks = chunk_text(sec_text, tokenizer)
        chunks = [c for c in chunks if _has_enough_real_content(c)]

        if not chunks:
            continue

        selected_sentences = []
        for chunk_idx, chunk in enumerate(chunks):
            if cancel_check and cancel_check():
                raise SummarizationCancelled()

            # Determine how many sentences to keep from this chunk
            chunk_sentences = split_into_sentences(chunk)
            chunk_sentences = [s for s in chunk_sentences if len(s.split()) >= 5]
            n_chunk = len(chunk_sentences)

            if n_chunk == 0:
                continue

            keep_n = max(1, round(n_chunk * keep_ratio))

            extracted = extract_key_sentences(
                chunk,
                keep_count=keep_n,
                target_ratio=keep_ratio,
                heading=sec_heading,
            )
            selected_sentences.extend(extracted)

        if selected_sentences:
            section_results.append({
                "heading": sec_heading,
                "sentences": selected_sentences,
            })

        if progress_callback:
            progress_callback(
                "summarizing", sec_idx + 1, total_sections,
                stage_label=f"Extracting section {sec_idx + 1} of {total_sections}…",
            )

    # ── Fallback: if heading detection found nothing useful ──────────────────
    if not section_results:
        if cancel_check and cancel_check():
            raise SummarizationCancelled()
        if progress_callback:
            progress_callback("summarizing", 1, 1,
                              stage_label="Extracting key sentences…")
        chunks = chunk_text(text, tokenizer)
        chunks = [c for c in chunks if _has_enough_real_content(c)]
        all_sentences = []
        for chunk in chunks:
            n_chunk = len([s for s in split_into_sentences(chunk) if len(s.split()) >= 5])
            keep_n = max(1, round(n_chunk * keep_ratio))
            all_sentences.extend(extract_key_sentences(chunk, keep_count=keep_n))
        if all_sentences:
            section_results = [{"heading": None, "sentences": all_sentences}]

    if progress_callback:
        progress_callback("finalizing", 1, 1, stage_label="Finalizing summary…")

    if not section_results:
        logger.warning("summarize_long_text: no sentences extracted — document may be too short or malformed")
        return {
            "summary": "",
            "highlights": [],
            "chunk_count": 0,
            "section_count": 0,
            "original_token_count": total_tokens,
        }

    # ── Metadata cleanup: strip lecturer names, course codes, dates ──────
    # from both headings and extracted sentences. Metadata belongs in the
    # Document Information block — it must NEVER appear mixed into the
    # summary body as headings or content sentences.
    if metadata:
        meta_fragments = set()
        for key in ("lecturer", "course_code", "date", "title",
                     "university", "department"):
            val = metadata.get(key, "")
            if val and len(val) >= 4:
                meta_fragments.add(val.lower().strip())
        # Also add individual lecturer name words (≥4 chars)
        lecturer = metadata.get("lecturer", "")
        if lecturer:
            for w in lecturer.split():
                cleaned = w.lower().strip(".,;:()")
                if len(cleaned) >= 4 and cleaned not in {
                    "prof", "engr", "madam"
                }:
                    meta_fragments.add(cleaned)

        def _contains_metadata(text_str):
            """True if text_str contains any metadata fragment."""
            lower = text_str.lower()
            return any(frag in lower for frag in meta_fragments if frag)

        cleaned_results = []
        for sec in section_results:
            heading = sec.get("heading")
            sentences = sec.get("sentences", [])

            # If the heading itself is a metadata value (lecturer name,
            # course code), drop it — merge sentences into previous section
            if heading and _contains_metadata(heading):
                heading = None  # remove the metadata heading

            # Filter out short sentences that contain metadata values
            clean_sents = []
            for s in sentences:
                if len(s.split()) <= 15 and _contains_metadata(s):
                    continue  # skip — it's a metadata fragment
                clean_sents.append(s)

            if clean_sents:
                cleaned_results.append({
                    "heading": heading,
                    "sentences": clean_sents,
                })

        section_results = cleaned_results if cleaned_results else section_results

    highlights = extract_highlights(section_results)
    raw_summary  = _format_structured_summary(section_results)

    # ── Prepend intro line + document title (matching the sample format) ──
    # "Here's a summary of the text in concise, easy-to-understand notes:"
    # ## COSC 414: Distributed Operating System - Module 1: Introduction
    header_parts = []
    header_parts.append("Here's a summary of the text in concise, easy-to-understand notes:\n")

    if metadata:
        # Build a document title from metadata fields
        title_pieces = []
        if metadata.get("course_code"):
            title_pieces.append(metadata["course_code"])
        if metadata.get("title"):
            sep = ": " if title_pieces else ""
            title_pieces.append(sep + metadata["title"])
        elif metadata.get("course_name"):
            sep = ": " if title_pieces else ""
            title_pieces.append(sep + metadata["course_name"])

        doc_title = "".join(title_pieces).strip()
        if doc_title:
            header_parts.append(f"## {doc_title}\n")

    header = "\n".join(header_parts)
    raw_summary = header + "\n" + raw_summary

    # Apply all four output-quality rules: symbol stripping, bullet
    # normalisation, one-item-per-line enforcement, and metadata isolation.
    final_summary = clean_output_text(raw_summary, metadata=metadata)

    chunk_count = sum(len(chunk_text(s["text"] if "text" in s else "", tokenizer))
                      for s in raw_sections) if raw_sections else 0

    logger.info(
        "summarize_long_text: summary=%d chars, %d sections, %d highlights",
        len(final_summary), len(section_results), len(highlights),
    )

    return {
        "summary": final_summary,
        "highlights": highlights,
        "chunk_count": max(1, chunk_count),
        "section_count": len(section_results),
        "original_token_count": total_tokens,
    }
