# System Architecture

## Overview

```
Browser (HTML5/CSS3/JavaScript)
      │  1. POST /summarize (file, length_preset)
      ▼
Flask app (app.py)
      │  2. Validate (MIME, size) → save to /uploads
      │  3. Spawn background thread → returns job_id immediately
      ▼
Background Worker Thread (utils/*)
      │  • Extract text, images, code → pdf_extractor / docx_extractor / txt_extractor
      │  • Clean text & metadata       → text_cleaner (slide dedup, 5-layer black box fix)
      │  • Detect sections & headings  → summarizer (detect_headings_and_sections)
      │  • Token-bounded chunking      → chunking (BART tokenizer, 900 tokens)
      │  • Extractive summarization    → summarizer (TF-IDF + MMR, zero-hallucination)
      │  • Multi-format export builder → download_generator (PDF, DOCX, TXT)
      ▼
In-Memory Thread-Safe Job Store (_jobs, _jobs_lock)
      ▲
      │  4. Browser polls GET /status/<job_id> every ~1.2s
      ▼
Browser renders progress, metrics, highlights, diagrams, and structured notes
```

## Why a background job instead of a blocking request?

Summarizing a multi-page document or textbook chapter involves structural parsing, bounding-box coordinate analysis, and TF-IDF matrix computations. A blocking HTTP request risks:

- Browser connection timeouts.
- Reverse proxy or load-balancer timeouts (often defaulted to 30–60 seconds).
- Inability to provide live visual progress or allow user cancellation.

Therefore, `/summarize` validates the upload, stores the file temporarily, starts a background daemon thread, and returns a unique `job_id` immediately. The client polls `/status/<job_id>` to display dynamic progress updates (`extracting`, `summarizing section X of Y`, `done`). If the user clicks Cancel, a `threading.Event` safely terminates processing.

## Why Pure Extractive Summarization (TF-IDF + MMR)?

Early prototypes explored abstractive sequence-to-sequence models (such as `facebook/bart-large-cnn`). However, deep neural decoders introduced critical flaws for technical literature:
1. **Hallucination:** Neural decoders generate new tokens based on probabilistic training data, inventing or distorting technical facts and numbers.
2. **Context Window Limitations:** BART's 1,024-token window truncated long documents or required recursive summarization passes.
3. **Destruction of Non-Text Assets:** Generative models stripped embedded diagrams and corrupted source code syntax.
4. **Hardware Bottlenecks:** Required dedicated GPUs or multi-gigabyte RAM allocations, making CPU execution painfully slow.

The current architecture employs **pure extractive summarization**:
1. **Sub-linear TF-IDF Vectorization:** Calculates term frequency and inverse document frequency across sentences in each chunk.
2. **Centrality & Specificity Scoring:** Ranks sentences based on cosine similarity to the section's core content.
3. **Content-Type Bonuses:** Heuristically prioritizes definitions (+0.35), core takeaways (+0.30), topic sentences (1.30x position boost), and section-heading alignment.
4. **Maximal Marginal Relevance (MMR, $\lambda = 0.65$):** Eliminates redundancy by penalizing sentences highly similar to already-selected points.
5. **Structural Integrity:** Preserves the author's section hierarchy (`##`, `###`, `####`), pairs balanced comparative points, and maintains original document chronological order.
6. **Token-Accurate Chunking:** Uses the Hugging Face BART tokenizer (`AutoTokenizer`) solely for fast, sentence-aligned token counting—with zero model inference or generation.

## Multi-Modal Asset Preservation

- **Diagrams & Figures:** Extracted from PDFs using PyMuPDF (`fitz`), assigned sequential figure numbers, and anchored contextually alongside related sections in both on-screen previews and exports.
- **Code Listings:** Detected via monospace font metrics (`Courier`, `Consolas`, `Monaco`) and preserved verbatim with syntax formatting without undergoing text summarization.

## Scaling and Deployment

The in-memory job store (`_jobs`) is thread-safe and optimized for single-process instances (e.g. `run.py` or Docker deployments on port 7860). Completed jobs have an automated 1-hour Time-To-Live (TTL) after which volatile memory is cleared. For multi-server horizontal scaling behind load balancers, the job store can be transitioned to an external Redis cache and Celery worker queue.
