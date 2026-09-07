"""
app.py
Flask backend for the AI Document Summarization System.

Design note on HTTP + long-running AI jobs:
Summarizing a whole textbook chapter can take minutes. Blocking an
HTTP request for that long risks browser and reverse-proxy timeouts,
and it stops the frontend from showing meaningful progress. So
/summarize kicks off a background thread and immediately returns a
job_id; the frontend polls /status/<job_id> for progress and the
final result. This is a standard pattern for AI-in-the-browser UX.
"""

import os
import re
import io
import time
import threading
import traceback
import logging
from collections import defaultdict

from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename

from config import Config
from utils.file_validator import validate_upload, get_extension, FileValidationError
from utils.pdf_extractor import extract_pdf_content
from utils.docx_extractor import extract_text_from_docx
from utils.txt_extractor import extract_text_from_txt
from utils.text_cleaner import clean_text, preprocess_document, format_metadata_block, clean_output_text
from utils.summarizer import summarize_long_text, SummarizationCancelled, prewarm_model

from utils.download_generator import build_download_file
from utils.helpers import new_job_id, human_word_count
from utils.keywords import extract_keywords

# Minimum word count for the extracted text to be considered summarisable.
# Below this threshold the document is either empty, a scanned image with
# no OCR layer, or so short that there is nothing useful to summarise.
MIN_EXTRACTABLE_WORDS = 30

app = Flask(__name__)
app.config.from_object(Config)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.jinja_env.auto_reload = True

# Configure logging so the summarizer's diagnostic messages are visible.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

if app.config.get("SECRET_KEY") == "dev-key-change-in-production":
    app.logger.warning("SECRET_KEY is still using the dev fallback. Please set a secure key in production.")

os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
os.makedirs(Config.OUTPUT_FOLDER, exist_ok=True)
os.makedirs(Config.LOG_FOLDER, exist_ok=True)

START_TIME = time.time()

# ---------------------------------------------------------------------------
# In-memory job store.
# For an undergraduate / single-instance deployment this is sufficient.
# (A production multi-worker deployment would move this to Redis so all
# worker processes share job state — noted in docs/architecture.md.)
# ---------------------------------------------------------------------------
_jobs = {}
_jobs_lock = threading.Lock()

# Extracted diagrams/figures are kept separately from the main job
# dict. The job dict gets sent to the browser as JSON on every status
# poll, and raw image bytes don't belong in a JSON response - instead,
# each image is served on its own through a dedicated route (see
# /image/<job_id>/<index> below), and the job dict only carries a
# small amount of metadata (how many images, which page each is from).
_job_images = {}
_job_ordered_elements = {}

# Each running job gets a threading.Event here. The background thread
# checks it periodically while summarizing; clicking Cancel just sets
# the flag rather than trying to forcibly kill the thread, which is
# the safe way to stop a Python thread partway through its work.
_cancel_flags = {}

_rate_limits = defaultdict(list)
_rate_limit_lock = threading.Lock()


def _get_client_ip():
    """
    Returns the real client IP, respecting X-Forwarded-For when the app
    runs behind a reverse proxy (e.g. Nginx, Gunicorn, Hugging Face Spaces).

    FIX Bug N: The previous implementation always used request.remote_addr,
    which behind a proxy is the proxy's IP, making rate limiting ineffective.
    """
    forwarded_for = request.headers.get("X-Forwarded-For", "")
    if forwarded_for:
        # X-Forwarded-For can be a comma-separated list; take the first (real) IP
        return forwarded_for.split(",")[0].strip()
    return request.remote_addr or "unknown"


def check_rate_limit(ip):
    now = time.time()
    with _rate_limit_lock:
        timestamps = _rate_limits[ip]
        # filter out old timestamps
        timestamps = [t for t in timestamps if now - t < Config.RATE_LIMIT_WINDOW_SECONDS]
        _rate_limits[ip] = timestamps
        if len(timestamps) >= Config.RATE_LIMIT_REQUESTS:
            return False
        timestamps.append(now)
        return True


def cleanup_stale_jobs():
    while True:
        time.sleep(600)  # 10 minutes
        now = time.time()
        cutoff = now - Config.JOB_CLEANUP_AFTER_SECONDS
        with _jobs_lock:
            stale_jobs = [jid for jid, j in _jobs.items() if j.get("updated_at", j.get("created_at", now)) < cutoff]
            for jid in stale_jobs:
                del _jobs[jid]
                _job_images.pop(jid, None)
                _job_ordered_elements.pop(jid, None)
                _cancel_flags.pop(jid, None)

threading.Thread(target=cleanup_stale_jobs, daemon=True).start()

# Pre-warm the BART tokenizer in the background at startup so the first
# summarization request isn't slowed by model loading.
prewarm_model()


def _set_job(job_id, **fields):
    with _jobs_lock:
        _jobs[job_id].update(fields)
        _jobs[job_id]["updated_at"] = time.time()


def _get_job(job_id):
    with _jobs_lock:
        return dict(_jobs.get(job_id, {}))


def _run_summarization_job(job_id, filepath, filename, ext, length_preset):
    try:
        _set_job(job_id, status="extracting", stage_detail="Reading document…")

        code_blocks = []
        images = []
        ordered_elements = []

        if ext == "pdf":
            def pdf_progress(current, total):
                _set_job(job_id, stage_detail=f"Reading page {current} of {total}…")
            pdf_content = extract_pdf_content(filepath, progress_callback=pdf_progress)
            raw_text = pdf_content["text"]
            code_blocks = pdf_content["code_blocks"]
            images = pdf_content["images"]
            ordered_elements = pdf_content["ordered_elements"]
        elif ext == "docx":
            raw_text = extract_text_from_docx(filepath)
        else:
            raw_text = extract_text_from_txt(filepath)

        # ── Text preprocessing + metadata extraction ────────────────────────
        # preprocess_document() does everything clean_text() does PLUS:
        #   1. Extracts metadata (title, course, author, lecturer, date) from
        #      the first ~40 lines for the structured header block.
        #   2. Deduplicates repetitive slide headers / footers / running titles
        #      that appear 3+ times (e.g. "COSC 414 BY DR. YAHAYA LAMIDO").
        text, doc_metadata = preprocess_document(raw_text)
        word_count_after_clean = len(text.split())

        app.logger.info(
            "Job %s: %d words after preprocessing (raw=%d chars, ext=%s, meta=%s)",
            job_id, word_count_after_clean, len(raw_text), ext,
            list(doc_metadata.keys()),
        )

        if word_count_after_clean < MIN_EXTRACTABLE_WORDS:
            if not raw_text.strip():
                raise ValueError(
                    "No text could be extracted from this document. "
                    "If this is a scanned PDF, it may not have a text layer — "
                    "try a text-based PDF, DOCX, or TXT file instead."
                )
            else:
                raise ValueError(
                    f"The document contains fewer than {MIN_EXTRACTABLE_WORDS} words of readable text "
                    f"(found {word_count_after_clean}). Please upload a document with more content."
                )

        original_words = human_word_count(text)

        _set_job(job_id, status="summarizing", stage_detail="Preparing the summarization model…")

        def progress_callback(stage, current, total, stage_label=None):
            if stage_label:
                detail = stage_label
            elif stage == "summarizing":
                detail = f"Summarizing chunk {current} of {total}…"
            else:
                detail = f"Condensing section {current} of {total}…"
            _set_job(job_id, status=stage, stage_detail=detail,
                     progress_current=current, progress_total=total)

        cancel_event = _cancel_flags.get(job_id)
        result = summarize_long_text(
            text,
            progress_callback=progress_callback,
            length_preset=length_preset,
            cancel_check=(cancel_event.is_set if cancel_event else None),
            metadata=doc_metadata,
        )
        summary = result["summary"]

        # Validate the summary is non-empty before declaring success
        if not summary or not summary.strip():
            raise ValueError(
                "Summarization produced an empty result. The document may not contain "
                "enough structured text to extract key sentences from. "
                "Please try a different document or summary length."
            )

        # ── Summary is returned clean — no metadata prepended ─────────────────
        # doc_metadata is sent as a separate JSON field (see _set_job below).
        # The frontend renders Document Information first, then highlights,
        # then the clean summary — keeping them visually separate.
        full_summary = clean_output_text(summary, metadata=doc_metadata)

        summary_words = human_word_count(full_summary)

        # Reading times estimation
        original_reading_time = max(1, int(round(original_words / Config.READING_WPM)))
        summary_reading_time = max(1, int(round(summary_words / Config.READING_WPM)))

        # Extract keywords from the cleaned text (fast, no model call needed)
        keywords = extract_keywords(text, max_keywords=8)

        # A sensible default name for the download box, e.g. "chapter1"
        # from "chapter1.pdf" - the user can change this before downloading.
        suggested_name = os.path.splitext(filename)[0]

        _job_images[job_id] = images
        _job_ordered_elements[job_id] = ordered_elements

        _set_job(
            job_id,
            status="done",
            stage_detail="Done.",
            summary=full_summary,
            highlights=result.get("highlights", []),
            keywords=keywords,
            code_blocks=code_blocks,
            image_count=len(images),
            images_meta=[{"page": img["page"]} for img in images],
            original_word_count=original_words,
            summary_word_count=summary_words,
            original_reading_time=original_reading_time,
            summary_reading_time=summary_reading_time,
            reduction_percent=round(100 * (1 - summary_words / max(original_words, 1)), 1),
            chunk_count=result["chunk_count"],
            section_count=result["section_count"],
            suggested_filename=suggested_name,
            doc_metadata=doc_metadata,
        )

    except SummarizationCancelled:
        _set_job(job_id, status="cancelled", stage_detail="Cancelled.")
    except Exception as exc:  # noqa: BLE001 — surface any failure to the user
        app.logger.error("Job %s failed: %s\n%s", job_id, exc, traceback.format_exc())
        _set_job(job_id, status="error", error=str(exc))
    finally:
        # Clean up the uploaded file — we only need it for the duration
        # of processing, not indefinitely.
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except OSError:
            pass
        _cancel_flags.pop(job_id, None)


@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    # FIX Bug P: Added Google Fonts domains so fonts load correctly.
    # Without this, the CSP blocked fonts.googleapis.com and fonts.gstatic.com,
    # causing Inter to fall back to system fonts silently.
    response.headers['Content-Security-Policy'] = (
        "default-src 'self'; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com; "
        "script-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: blob:; "
        "connect-src 'self';"
    )
    return response


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/help")
def help_page():
    return render_template("help.html")


@app.route("/accessibility")
def accessibility():
    return render_template("accessibility.html")


@app.route("/evaluation")
def evaluation():
    return render_template("evaluation.html")


@app.route("/api/health")
def health_check():
    # FIX Bug M: _model was removed from summarizer.py when the extractive
    # rewrite landed, causing an ImportError here. We now import only _tokenizer
    # (which still exists) and use its presence as the readiness signal.
    from utils.summarizer import _model, _tokenizer
    return jsonify({
        "status": "ok",
        "model_loaded": _tokenizer is not None,
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "active_jobs": sum(1 for j in _jobs.values() if j.get("status") not in ("done", "error", "cancelled")),
    })


@app.route("/robots.txt")
def robots():
    return "User-agent: *\nDisallow: /\n", 200, {'Content-Type': 'text/plain'}


@app.route("/summarize", methods=["POST"])
def summarize():
    client_ip = _get_client_ip()
    if not check_rate_limit(client_ip):
        return jsonify({"error": "Too many requests. Please try again later."}), 429

    try:
        filename = validate_upload(request.files.get("document"))
    except FileValidationError as exc:
        return jsonify({"error": str(exc)}), 400

    file_storage = request.files["document"]
    ext = get_extension(filename)

    length_preset = request.form.get("summary_length", Config.DEFAULT_SUMMARY_LENGTH).lower()
    if length_preset not in Config.SUMMARY_PRESETS:
        length_preset = Config.DEFAULT_SUMMARY_LENGTH

    job_id = new_job_id()
    stored_name = f"{job_id}_{filename}"
    filepath = os.path.join(Config.UPLOAD_FOLDER, stored_name)

    # FIX Bug O: Ensure the job is only registered once the file is safely
    # saved. If saving fails (e.g. disk full) we return an error immediately
    # rather than leaving a zombie job in the _jobs dict.
    try:
        file_storage.save(filepath)
    except Exception as exc:
        app.logger.error("Failed to save uploaded file: %s", exc)
        return jsonify({"error": "Failed to save the uploaded file. Please try again."}), 500

    with _jobs_lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "stage_detail": "Queued…",
            "filename": filename,
            "created_at": time.time(),
        }
    _cancel_flags[job_id] = threading.Event()

    thread = threading.Thread(
        target=_run_summarization_job,
        args=(job_id, filepath, filename, ext, length_preset),
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/cancel/<job_id>", methods=["POST"])
def cancel(job_id):
    job = _get_job(job_id)
    if not job:
        return jsonify({"error": "Unknown job id."}), 404

    if job.get("status") in ("done", "error", "cancelled"):
        return jsonify({"error": "This job has already finished."}), 400

    cancel_event = _cancel_flags.get(job_id)
    if cancel_event:
        cancel_event.set()

    _set_job(job_id, status="cancelling", stage_detail="Cancelling…")
    return jsonify({"status": "cancelling"})


@app.route("/status/<job_id>")
def status(job_id):
    job = _get_job(job_id)
    if not job:
        return jsonify({"error": "Unknown job id."}), 404
    return jsonify(job)


@app.route("/image/<job_id>/<int:index>")
def get_image(job_id, index):
    images = _job_images.get(job_id)
    if not images or index < 0 or index >= len(images):
        return jsonify({"error": "Image not found."}), 404

    image = images[index]
    mime_map = {"png": "image/png", "jpeg": "image/jpeg", "jpg": "image/jpeg"}
    mimetype = mime_map.get(image["ext"].lower(), "image/png")

    return send_file(io.BytesIO(image["bytes"]), mimetype=mimetype)


@app.route("/download/<job_id>")
def download(job_id):
    job = _get_job(job_id)
    if not job or job.get("status") != "done":
        return jsonify({"error": "That summary isn't ready to download."}), 404

    # Which format did the user pick? Default to plain text.
    file_format = request.args.get("format", "txt").lower()
    if file_format not in ("txt", "docx", "pdf"):
        file_format = "txt"

    # What name did the user type? Fall back to a name based on the
    # original document if they left it blank.
    requested_name = request.args.get("filename", "").strip()
    if not requested_name:
        requested_name = job.get("suggested_filename", "summary")
    safe_name = secure_filename(requested_name) or "summary"

    export_options = {
        "font_family": request.args.get("font_family", Config.EXPORT_FONT_FAMILY),
        "font_size": request.args.get("font_size", Config.EXPORT_FONT_SIZE),
        "margin_inches": request.args.get("margin_inches", Config.EXPORT_MARGIN_INCHES),
        "line_spacing": request.args.get("line_spacing", Config.EXPORT_LINE_SPACING),
        "page_orientation": request.args.get("page_orientation", Config.EXPORT_ORIENTATION),
        "include_header": request.args.get("include_header", "false"),
        "include_footer": request.args.get("include_footer", "false"),
        "include_page_numbers": request.args.get("include_page_numbers", "false")
    }

    try:
        file_bytes, mime_type = build_download_file(
            file_format,
            job.get("filename", "document"),
            job.get("summary", ""),
            job.get("highlights", []),
            job.get("code_blocks", []),
            _job_images.get(job_id, []),
            export_options,
            _job_ordered_elements.get(job_id, [])
        )
    except Exception as exc:
        app.logger.error("Download generation failed for job %s: %s", job_id, exc, exc_info=True)
        return jsonify({"error": "Could not generate the download file. Please try a different format or re-upload your document."}), 500

    return send_file(
        io.BytesIO(file_bytes),
        mimetype=mime_type,
        as_attachment=True,
        download_name=f"{safe_name}.{file_format}",
    )


@app.errorhandler(413)
def too_large(_exc):
    return jsonify({"error": "That file is larger than the 20 MB limit. Please upload a smaller file."}), 413


@app.errorhandler(404)
def not_found(_exc):
    return render_template("error.html", message="Page not found."), 404


@app.errorhandler(500)
def server_error(_exc):
    return render_template("error.html", message="Something went wrong on our end."), 500


if __name__ == "__main__":
    # Hugging Face Spaces (Docker SDK) expects the app to listen on 7860.
    port = int(os.environ.get("PORT", 7860))
    # use_reloader=True: Flask watches all .py files and automatically
    # restarts the server whenever any file is saved.
    # This means code changes apply immediately — no manual restart needed.
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        use_reloader=True,
    )
