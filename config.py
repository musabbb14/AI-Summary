"""
config.py
Central configuration for the AI Document Summarization System.
Keeping every tunable value here means nobody has to go hunting
through app.py or utils/ to change a limit or a model name.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Config:
    # --- Folders ---
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
    OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs")
    LOG_FOLDER = os.path.join(BASE_DIR, "logs")

    # --- Upload rules ---
    ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}
    # No document-length limit exists anywhere in this system — a full
    # textbook is summarized start to finish via chunking + reduce passes.
    # This is purely a file-size safety ceiling (memory/disk), set high
    # enough that it never becomes a practical restriction.
    MAX_CONTENT_LENGTH = 20 * 1024 * 1024   # 20 MB

    # --- Model settings ---
    MODEL_NAME = "facebook/bart-large-cnn"
    MODEL_MAX_TOKENS = 1024        # BART's hard input limit
    CHUNK_TOKEN_SIZE = 900         # leave headroom below the 1024 limit
    CHUNK_OVERLAP_TOKENS = 60      # keeps context continuous between chunks
    SUMMARY_MAX_LENGTH = 260       # tokens, per chunk summary (longer, more detailed)
    SUMMARY_MIN_LENGTH = 120       # tokens - forces a fuller summary, not a one-liner

    # Summary length presets — the user chooses Short / Medium / Long
    # before clicking Summarize. Each preset maps to (min_tokens, max_tokens)
    # per chunk summary, plus a base reduce threshold that scales with doc size.
    # How long a summary should be, as a fraction of the original text's
    # length - NOT a fixed number of words. Using a ratio (rather than a
    # fixed token count) is what makes Short/Medium/Long/Custom actually
    # produce different results regardless of how long the input is;
    # a fixed count was getting overridden for shorter documents,
    # making every option collapse to roughly the same output.
    SUMMARY_PRESETS = {
        "short":  {"ratio": 0.25},
        "medium": {"ratio": 0.40},
        "long":   {"ratio": 0.60},
    }
    DEFAULT_SUMMARY_LENGTH = "medium"

    # A summary's minimum length, as a fraction of its own target
    # maximum - keeps the model from stopping too abruptly.
    MIN_LENGTH_RATIO_OF_MAX = 0.6

    # Hard floor and ceiling per chunk, regardless of ratio, so an
    # extremely short or long chunk still gets a sensible result.
    # Raised alongside the ratios above so "Long" (60% of a full-size
    # ~900-token chunk) isn't clipped by this ceiling.
    ABSOLUTE_MIN_SUMMARY_TOKENS = 16
    # 400 tokens max per chunk: BART generates tokens one-by-one, so a lower
    # ceiling directly reduces generation time. 700 was over-long for CPU.
    ABSOLUTE_MAX_SUMMARY_TOKENS = 400

    # Custom length: instead of picking Short/Medium/Long, the user can
    # type an exact target percentage (how long the summary should be
    # relative to the original document). Kept within a sane range so
    # a typo can't ask for an unreasonably tiny or huge summary.
    CUSTOM_PERCENT_MIN = 10
    CUSTOM_PERCENT_MAX = 70
    CUSTOM_PERCENT_DEFAULT = 25

    # --- Key highlights ---
    # A short bullet list shown alongside the full summary, similar to
    # the "highlights" feature in tools like Scholarcy.
    MAX_HIGHLIGHTS = 5

    # Reading time estimation: average words per minute for an adult reader
    READING_WPM = 238

    # --- Section grouping (replaces the old iterative "reduce" approach) ---
    # A long document is split into chunks (~900 tokens each) and each
    # chunk gets its own mini-summary. For a genuinely long document
    # (a textbook chapter), that can be 40-60+ mini-summaries - too many
    # to show as separate sections. Instead of repeatedly re-summarizing
    # everything down to a fixed small size (which was the old, lossy
    # approach), a few original chunks are grouped together and
    # summarized into ONE section - a single extra pass, not a repeated
    # shrinking loop. More original content in a group means a longer
    # target length for that section, so the final summary's total
    # length scales with the document's length instead of being capped
    # at roughly the same size no matter how long the source document is.
    CHUNKS_PER_SECTION = 4   # roughly how many original chunks feed one section
    MAX_SECTIONS = 12        # hard cap; 12 readable sections is plenty for any document

    # Beam search width for the AI model. Higher = slightly more fluent
    # wording but roughly proportionally slower. Set to 1 (no beam
    # search, just the single most likely next word each step) since
    # this is running on ordinary CPU hardware, not a GPU server - this
    # is the single biggest speed lever available without changing the
    # underlying model, and summarization quality is only mildly
    # affected compared to the large jump in speed.
    # Beam search width for the AI model. This was previously set to 1
    # (greedy decoding - the fastest option) for speed. However, greedy
    # decoding is a well-documented cause of hallucinated or off-topic
    # output in this kind of model, especially on documents very
    # different from what it was trained on (news articles) - things
    # like structured guidelines, forms, or technical documents. Since
    # factual accuracy matters more than shaving off processing time,
    # this is back up to a real beam search width, which meaningfully
    # improves how well the output stays grounded in the actual input.
    # 2 beams: ~2x faster than 4 beams on CPU with negligible quality difference.
    # 4 beams on CPU (no GPU parallelism) was the single biggest cause of
    # 13-15 minute summarizations. 2 beams is the best CPU trade-off.
    NUM_BEAMS = 2

    # How many chunks to summarize in a single model call. Processing
    # several chunks together is noticeably faster than one at a time,
    # since a lot of the model's overhead is per-call, not per-word.
    # 3 chunks per batch on CPU: reduces peak memory pressure and padding waste
    # compared to 6. Smaller batches are faster in practice on CPU hardware.
    BATCH_SIZE = 3

    # --- Jobs ---
    # Long documents can take minutes. Rather than block an HTTP request
    # (and risk browser/proxy timeouts), summarization runs as a background
    # job and the frontend polls /status/<job_id> for progress.
    JOB_CLEANUP_AFTER_SECONDS = 60 * 60  # 1 hour

    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-key-change-in-production")
    
    # --- Export customization defaults ---
    EXPORT_FONT_FAMILY = "Helvetica"
    EXPORT_FONT_SIZE = 12
    EXPORT_MARGIN_INCHES = 1.0
    EXPORT_LINE_SPACING = 1.15
    EXPORT_ORIENTATION = "portrait"

    # --- Rate limiting config ---
    RATE_LIMIT_REQUESTS = 10
    RATE_LIMIT_WINDOW_SECONDS = 60
