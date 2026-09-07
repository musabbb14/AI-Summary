# Requirements Traceability

Maps each objective in the project proposal to where it's implemented,
for quick reference during development or defence.

| Proposal Objective | Implemented In |
|---|---|
| Obj. 1 - Model selection (facebook/bart-large-cnn) | `config.py` (MODEL_NAME), `utils/summarizer.py` |
| Obj. 2 - Flask backend, file extraction | `app.py`, `utils/pdf_extractor.py`, `utils/docx_extractor.py`, `utils/txt_extractor.py` |
| Obj. 2 - No document-length limit (chunk + reduce) | `utils/chunking.py`, `utils/summarizer.py` |
| Obj. 2 - Background job (avoids HTTP timeouts on long documents) | `app.py` (`_run_summarization_job`, `/status/<job_id>`) |
| Obj. 3 - Plain HTML/CSS/JS frontend, no reload | `templates/index.html`, `static/css/style.css`, `static/js/script.js` |
| Obj. 3 - Multiple download formats (PDF/DOCX/TXT), custom file name | `utils/download_generator.py`, `/download/<job_id>` route |
| Obj. 4 - Quantitative evaluation (ROUGE) | `evaluation/rouge_evaluation.py` |
| Obj. 4 - Qualitative evaluation (SUS, 10+ participants) | `evaluation/sus_questionnaire.md`, `evaluation/sus_results_template.csv` |
| Accessibility (HCI requirement) | Skip link, keyboard support, screen-reader labels, contrast, reduced-motion - see `templates/accessibility.html` and `static/css/style.css` |
| Deployment | `Dockerfile`, `README.md` (Hugging Face Spaces steps) |
