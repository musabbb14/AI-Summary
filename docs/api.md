# API Reference

## `POST /summarize`
Uploads a document and starts an asynchronous background summarization job.

**Request:** `multipart/form-data`
- `document`: File object (`.pdf`, `.docx`, or `.txt`, max 20 MB).
- `summary_length`: String preset (`short`, `medium`, or `long`).

**Response `200`:**
```json
{ "job_id": "9f3a1b2c-..." }
```

**Response `400` / `413` / `429`:**
```json
{ "error": "Error description message." }
```

---

## `GET /status/<job_id>`
Poll this endpoint to track extraction and summarization progress, and to retrieve the final result.

**Response while running:**
```json
{
  "job_id": "9f3a1b2c-...",
  "status": "summarizing",
  "stage_detail": "Extracting section 3 of 8…",
  "progress_current": 3,
  "progress_total": 8
}
```

**Response when done:**
```json
{
  "status": "done",
  "stage_detail": "Done.",
  "summary": "### 1. Section Title\n\n  Verbatim extracted sentence.",
  "highlights": ["Key highlight 1", "Key highlight 2"],
  "keywords": ["distributed", "operating", "multiprocessor", "memory"],
  "code_blocks": ["// Monospaced code listing"],
  "image_count": 2,
  "images_meta": [{"page": 4, "figure_number": "Figure 1.1"}],
  "original_word_count": 1220,
  "summary_word_count": 531,
  "reduction_percent": 56.5,
  "original_reading_time": 5,
  "summary_reading_time": 2,
  "chunk_count": 2,
  "section_count": 8,
  "suggested_filename": "COSC_414_M1-2026",
  "doc_metadata": {
    "course_code": "COSC 414",
    "title": "Distributed Operating System",
    "lecturer": "Dr. Yahaya Lamido"
  }
}
```

---

## `POST /cancel/<job_id>`
Safely signals the running background thread to abort summarization.

**Response `200`:**
```json
{ "status": "cancelling" }
```

---

## `GET /image/<job_id>/<int:index>`
Streams extracted raster figure bytes (`image/png` or `image/jpeg`) for display in the web UI.

---

## `GET /download/<job_id>`
Generates and downloads the summary file on demand.

**Query Parameters:**
- `format`: `pdf`, `docx`, or `txt` (default: `txt`)
- `filename`: Desired output filename
- `margin_inches`, `font_family`, `font_size`, `line_spacing`, `page_orientation`
- `include_header`, `include_footer`, `include_page_numbers`

---

## `GET /api/health`
Health check endpoint reporting system uptime, active jobs, and model pre-warming status.

