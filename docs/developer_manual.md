# Developer Manual

## Project layout
See the top-level `README.md` for setup. Module responsibilities:

| File | Responsibility |
|---|---|
| `app.py` | Flask routes, job orchestration, rate limiting, security headers |
| `config.py` | All tunable constants (presets, upload sizes, token boundaries) |
| `run.py` | Single-file local launcher (starts server + opens browser) |
| `utils/pdf_extractor.py` | PDF → text, raster diagrams, code blocks, TOC detection (PyMuPDF) |
| `utils/docx_extractor.py` | DOCX → text extraction (python-docx) |
| `utils/txt_extractor.py` | TXT → UTF-8 text extraction |
| `utils/text_cleaner.py` | Preprocessing, slide header dedup, metadata extraction, 5-layer black box fix |
| `utils/chunking.py` | Sentence splitting & token-bounded chunking (BART tokenizer) |
| `utils/summarizer.py` | Extractive engine: TF-IDF, MMR, headings, balanced differences, highlights |
| `utils/keywords.py` | Sub-linear TF-IDF keyword extraction |
| `utils/file_validator.py` | Upload validation (MIME, magic bytes, 20 MB limit) |
| `utils/download_generator.py` | Multi-format export builder: PDF (ReportLab), Word (python-docx), TXT |
| `utils/helpers.py` | Shared utilities (UUID generator, human word counter) |

## Adding a new file format
1. Add a `<format>_extractor.py` in `utils/` exposing `extract_text_from_<format>(filepath) -> str`.
2. Register the extension in `Config.ALLOWED_EXTENSIONS` (`config.py`).
3. Wire it into the `if/elif` in `app.py`'s `_run_summarization_job`.

## Adjusting summarization behavior
All core configuration knobs — chunk size, overlap, summary length presets (Short 25%, Medium 40%, Long 60%) — live in `config.py`. Tunable algorithmic weights (content bonuses, MMR $\lambda = 0.65$) are maintained in `utils/summarizer.py`.

## Running tests
```bash
pytest -v
```
