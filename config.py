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

    # --- Model & Tokenizer settings ---
    MODEL_NAME = "facebook/bart-large-cnn"
    MODEL_MAX_TOKENS = 1024        # BART tokenizer input limit
    CHUNK_TOKEN_SIZE = 900         # token headroom per chunk
    CHUNK_OVERLAP_TOKENS = 60      # keeps context continuous between chunks

    # Summary length presets — the user chooses Short / Medium / Long
    # before clicking Summarize. Each preset maps to a sentence keep ratio.
    SUMMARY_PRESETS = {
        "short":  {"ratio": 0.25},
        "medium": {"ratio": 0.40},
        "long":   {"ratio": 0.60},
    }
    DEFAULT_SUMMARY_LENGTH = "medium"

    # --- Key highlights ---
    MAX_HIGHLIGHTS = 5

    # Reading time estimation: average words per minute for an adult reader
    READING_WPM = 238

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
