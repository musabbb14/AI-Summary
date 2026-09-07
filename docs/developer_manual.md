# Developer Manual

## Project layout
See the top-level `README.md` for setup. Module responsibilities:

| File | Responsibility |
|---|---|
| `app.py` | Flask routes, job orchestration |
| `config.py` | All tunable constants |
| `run.py` | Single-file local launcher (opens browser) |
| `utils/pdf_extractor.py` | PDF → text (PyMuPDF) |
| `utils/docx_extractor.py` | DOCX → text (python-docx) |
| `utils/txt_extractor.py` | TXT → text |
| `utils/text_cleaner.py` | Normalize extracted text |
| `utils/chunking.py` | Token-accurate, overlapping chunking |
| `utils/summarizer.py` | BART pipeline + map-reduce summarization |
| `utils/file_validator.py` | Upload validation |
| `utils/download_generator.py` | Writes `.txt` summary output |
| `utils/helpers.py` | Small shared helpers |

## Adding a new file format
1. Add a `<format>_extractor.py` in `utils/` exposing `extract_text_from_<format>(filepath) -> str`.
2. Register the extension in `Config.ALLOWED_EXTENSIONS` (`config.py`).
3. Wire it into the `if/elif` in `app.py`'s `_run_summarization_job`.

## Adjusting summarization behavior
All the knobs — chunk size, overlap, summary length, reduce-pass limits —
live in `config.py`. No need to touch `utils/summarizer.py` for tuning.

## Running tests
```
pip install pytest
pytest -m "not model"      # fast tests, no model download
pytest                       # full suite, downloads BART on first run
```
