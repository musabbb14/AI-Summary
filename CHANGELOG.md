# Changelog

Tracks the iterative development of this project (Design Science
Research methodology: build, evaluate, refine).

## v2.0 - Extractive Architecture, Table of Contents & Black Box Fixes, Minimalist UI Redesign
- **Detailed Reference:** See [`CHANGES.md`](CHANGES.md) for the complete architectural and line-level agent manual.
- **Architectural Pivot:** Migrated from abstractive BART to high-performance extractive summarization (TF-IDF + Maximal Marginal Relevance) for 100% factual accuracy, zero hallucinations, and sub-second CPU processing.
- **Table of Contents & Black Box (`■`) Fix:** Diagnosed publisher TOC dot-leader glyphs using `ZapfDingbats` that polluted summaries; deployed a 5-layer detection and sanitization pipeline across PDF extraction, text cleaning, TF-IDF scoring, ReportLab export, and frontend rendering.
- **Summary Presets Overhaul:** Replaced first-sentence penalties with topic-sentence boosting and heading keyword alignment. Calibrated Short (0.25), Medium (0.45), and Long (0.65) presets to strictly capture key main points.
- **Minimalist UI/UX Overhaul:** Cleaned dropdown options (strictly Short, Medium, Long without subtitles or hints); removed upload formatting accordion; designed a sleek floating pill bottom action bar with "Summarize Another" and format dropdown + Download button; modernized footer.
- **Reliability:** Enabled Flask template auto-reloading (`TEMPLATES_AUTO_RELOAD = True`), hardened CSP headers for fonts, and resolved frontend button event bindings.

## v1.0 - Initial build
- Flask backend, BART summarization via the `pipeline()` shortcut
- Basic HTML/CSS/JS frontend
- Background job + polling for long documents

## v1.1 - Bug fix
- Replaced the `pipeline("summarization", ...)` shortcut with a direct
  `AutoModelForSeq2SeqLM` + `.generate()` call, after the shortcut broke
  on some installed library versions ("Unknown task summarization")

## v1.2 - Simplification pass
- Rewrote chunking, summarization, and frontend code in plainer,
  more explainable style (no jargon, no unnecessary abstractions)
- Loosened pinned dependency versions to avoid install failures on
  newer Python versions
- Removed placeholder branding, renamed to plain "AI Document Summarizer"

## v1.3 - Output quality + UX
- Increased summary length so output is a proper summary, not a
  one-line blurb
- Added a "Clear" button for restarting before or after summarizing
- Added color, shadows, and a styled scrollbar to the interface

## v1.4 - Multi-format download
- Added on-demand PDF, DOCX, and TXT export with a user-chosen file name

## v1.5 - Accessibility pass
- Added a skip-to-content link, visible keyboard focus outlines,
  reduced-motion support, and improved color contrast
- Added labels for all form controls
- Added SUS questionnaire materials and a requirements traceability
  document for Objective 4 and defence preparation

## v1.7 - Diagrams, code preservation, and real speed fixes
- Fixed a major bug where long documents were repeatedly re-summarized
  down to roughly the same small size no matter how long the original
  was. Summary length now scales with document size (a simulated
  79-page document went from ~2 pages of output to ~7-8 pages).
- Fixed a bug where floating accent marks in extracted text could
  render as a stray symbol next to a letter.
- Diagrams and figures are now extracted from PDFs and included in the
  summary (on-screen and in every download format).
- Code listings are detected by font (monospaced text) and preserved
  exactly as written - never passed through the AI, since summarizing
  code risks corrupting it.
- Sped up processing by batching several chunks into a single AI model
  call instead of one at a time, and by lowering the model's beam
  search width - together, roughly halved the number of AI calls
  needed for a long document in testing.
- Fixed a bug where "Recent Summaries" could only ever save one entry
  because job IDs weren't being included in status responses.

## v1.8 - Further speed, Cancel button, and a responsive design audit
- Pushed the remaining safe speed levers: switched from beam search to
  greedy decoding (num_beams=1), increased batch size, and explicitly
  set PyTorch to use all available CPU cores (a free speed gain most
  default setups miss). In testing, this cut a 79-page-equivalent
  document from 22 AI calls down to 15.
- Being upfront: these are real, measured improvements, but a laptop
  CPU will still be considerably slower than a GPU-backed server for
  a model this size - that's a hosting decision, not something pure
  code changes can fully close.
- Added a Cancel button, visible while a summary is being generated.
  Clicking it stops the job cleanly within a few seconds (it finishes
  its current step, then stops - Python can't safely kill a thread
  instantly mid-calculation).
- Responsive design audit: found and fixed a completely unstyled
  custom-percentage slider, and added proper mobile stacking for the
  header, length selector, and result action buttons on narrow screens.
