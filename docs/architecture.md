# System Architecture

## Overview

```
Browser (HTML/CSS/JS)
      │  1. POST /summarize (file)
      ▼
Flask app (app.py)
      │  2. validate → save to /uploads
      │  3. spawn background thread → returns job_id immediately
      ▼
Background job (utils/*)
      │  extract text  → pdf_extractor / docx_extractor / txt_extractor
      │  clean text    → text_cleaner
      │  chunk text    → chunking (token-accurate, overlapping)
      │  summarize     → summarizer (BART, map-reduce for long docs)
      │  save result   → download_generator
      ▼
In-memory job store (dict, thread-locked)
      ▲
      │  4. Browser polls GET /status/<job_id> every ~1.2s
      ▼
Browser renders progress → final summary
```

## Why a background job instead of a blocking request?

Summarizing a single-page memo takes a couple of seconds. Summarizing a
300-page textbook chapter, split into 60+ overlapping chunks, can take
several minutes. A blocking HTTP request that long risks:

- Browser default timeouts
- Reverse proxy / load balancer timeouts (common default: 30–60s)
- No way to show real progress to the user

So `/summarize` only validates the upload, saves the file, and starts a
daemon thread — then returns a `job_id` right away. The frontend polls
`/status/<job_id>`, which reads from a thread-safe in-memory dict updated
by the job as it works through extraction → chunk 1 → chunk 2 → ... →
reduce passes → done.

## Why map-reduce summarization?

`facebook/bart-large-cnn` has a hard 1,024-token input limit. A textbook
chapter is easily 10–50x that. The system:

1. **Map** — splits the cleaned text into overlapping ~900-token chunks
   (sentence-aligned, so no chunk starts or ends mid-sentence) and
   summarizes each one independently.
2. **Reduce** — concatenates the chunk summaries. If that's still too
   long to read comfortably (or to feed into another summarization pass
   meaningfully), it re-chunks and re-summarizes the summaries themselves,
   repeating up to `MAX_REDUCE_PASSES` times (config.py) — which
   guarantees the process always terminates even on enormous input.

## Scaling beyond an undergraduate deployment

The in-memory job store is intentionally simple (a Python dict). It works
correctly for a single-process deployment (which is what `run.py` and the
Docker image both are). If this were extended to run behind multiple
worker processes or multiple machines, job state should move to a shared
store like Redis, so any worker can answer a `/status/<job_id>` poll
regardless of which worker handled the original `/summarize` request.
