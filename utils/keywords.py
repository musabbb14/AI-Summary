"""
utils/keywords.py

Lightweight keyword extraction for the AI Document Summarization System.

Extracts the most significant terms from a document using a simplified
TF-IDF approach — no external NLP libraries required (no NLTK, no spaCy).

How it works:
  1. Tokenize the text into cleaned words (letters only, lowercase)
  2. Remove common English stop words
  3. Count word frequencies (TF = term frequency)
  4. Apply a mild inverse-document-frequency penalty using a built-in
     background frequency table of common English words — words that
     appear often in English generally (like "system" or "provide") get
     down-weighted relative to domain-specific terms
  5. Return the top N keywords sorted by score, titlecased for display

This is appropriate for a final-year project on CPU hardware:
- Zero additional dependencies
- Runs in milliseconds even on large documents
- Produces genuinely useful keywords for technical and academic documents
"""

import re
from collections import Counter

# Common English stop words — words that appear in almost any text and
# carry no document-specific meaning. Excluded from keyword extraction.
_STOP_WORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "need",
    "that", "this", "these", "those", "it", "its", "he", "she", "they",
    "we", "you", "i", "me", "him", "her", "us", "them", "my", "your",
    "his", "our", "their", "what", "which", "who", "whom", "when", "where",
    "why", "how", "if", "not", "no", "nor", "so", "yet", "both", "either",
    "each", "any", "all", "more", "most", "other", "such", "than", "then",
    "too", "very", "just", "also", "only", "into", "over", "after", "under",
    "about", "up", "out", "get", "use", "used", "using", "make", "made",
    "one", "two", "three", "new", "also", "well", "even", "same", "much",
    "because", "while", "through", "between", "during", "without", "within",
    "however", "therefore", "thus", "hence", "since", "whether", "although",
    "though", "unless", "until", "upon", "among", "before", "below", "above",
    "called", "based", "given", "known", "related", "different", "following",
    "including", "used", "provides", "provide", "refers", "refers", "refer",
    "show", "shows", "shown", "means", "meaning", "result", "results",
    "number", "numbers", "types", "type", "example", "examples", "set",
    "sets", "many", "several", "various", "often", "first", "second", "third",
    "part", "section", "chapter", "figure", "table", "page", "pages",
    "according", "allow", "allows", "consists", "involves", "enables",
    "ensure", "ensures", "focus", "focuses", "include", "includes",
    "require", "requires", "describe", "describes", "defined", "define",
})

# Common English words that often appear in academic/technical documents
# but are too generic to be useful as keywords. Down-weight these by
# dividing their score by a penalty factor.
_GENERIC_TECHNICAL_WORDS = frozenset({
    "system", "systems", "process", "processes", "method", "methods",
    "approach", "approaches", "model", "models", "data", "information",
    "analysis", "design", "implementation", "development", "application",
    "applications", "computer", "software", "hardware", "technology",
    "research", "study", "paper", "work", "problem", "solution",
    "performance", "evaluation", "function", "functions", "structure",
    "structures", "management", "control", "user", "users", "time",
    "value", "values", "level", "levels", "point", "points", "rate",
    "operation", "operations", "input", "output", "high", "low",
    "large", "small", "important", "general", "specific", "main",
    "basic", "common", "current", "possible", "similar", "simple",
})


def extract_keywords(text, max_keywords=8, min_word_length=3):
    """
    Extracts the most document-specific keywords from `text`.

    Parameters
    ----------
    text : str
        The cleaned document text to analyse.
    max_keywords : int
        Maximum number of keywords to return.
    min_word_length : int
        Minimum character length for a word to be considered.

    Returns
    -------
    list[str]
        Up to `max_keywords` keywords, titlecased, in descending relevance
        order. Returns an empty list if the text is too short.
    """
    if not text or len(text.split()) < 20:
        return []

    # Tokenize: letters only, lowercase, minimum length filter
    words = re.findall(r"[a-zA-Z]{" + str(min_word_length) + r",}", text.lower())

    # Remove stop words
    content_words = [w for w in words if w not in _STOP_WORDS]

    if not content_words:
        return []

    total_words = len(content_words)
    counts = Counter(content_words)

    # Score = normalized frequency, with a penalty for generic technical words
    scores = {}
    for word, count in counts.items():
        tf = count / total_words
        if word in _GENERIC_TECHNICAL_WORDS:
            tf *= 0.4  # down-weight generic terms
        scores[word] = tf

    # Sort by score descending, take top candidates
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Return titlecased, deduplicated
    seen = set()
    keywords = []
    for word, _ in ranked:
        if word not in seen and len(keywords) < max_keywords:
            seen.add(word)
            keywords.append(word.capitalize())

    return keywords
