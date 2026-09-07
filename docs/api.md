# API Reference

## `POST /summarize`
Uploads a document and starts a background summarization job.

**Request:** `multipart/form-data` with a field named `document`.

**Response `200`:**
```json
{ "job_id": "9f3a1b2c..." }
```

**Response `400`:** invalid or missing file
```json
{ "error": "'.png' files aren't supported yet. Please upload a PDF, DOCX, or TXT file." }
```

## `GET /status/<job_id>`
Poll this to track progress and retrieve the final result.

**Response while running:**
```json
{
  "status": "summarizing",
  "stage_detail": "Summarizing section 4 of 12…",
  "progress_current": 4,
  "progress_total": 12
}
```

**Response when done:**
```json
{
  "status": "done",
  "summary": "…",
  "original_word_count": 8213,
  "summary_word_count": 412,
  "reduction_percent": 95.0,
  "chunk_count": 12,
  "reduce_passes": 1,
  "download_filename": "chapter1_summary.txt"
}
```

**Response on failure:**
```json
{ "status": "error", "error": "No extractable text was found in this PDF..." }
```

**Response `404`:** unknown job id.

## `GET /download/<filename>`
Downloads a previously generated `.txt` summary file.
