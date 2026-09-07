---
title: DocSummarize - AI Document Summarizer
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---

# DocSummarize &mdash; AI Document Summarization System

**DocSummarize** is a modern, web-based document summarization platform designed for technical and academic documents (PDF, DOCX, and TXT). It utilizes an advanced **extractive summarization engine** combining TF-IDF centrality, specificity scoring, and **Maximal Marginal Relevance (MMR)** with automatic **diagram, table, and code block preservation**.

Developed for the final year undergraduate project in the Department of Computer Science, Gombe State University.

---

## Key Features

- **Zero-Hallucination Extractive AI**: Every summarized sentence is extracted verbatim from the original source text. It never invents citations, facts, or numbers.
- **Key Main-Point Focus Across All Presets**:
  - **Short**: Strictly key main points, core definitions, and primary takeaways.
  - **Medium**: Balanced summary capturing key main points alongside primary explanations.
  - **Long**: Comprehensive coverage retaining key points, in-depth technical context, and examples.
- **Visual Diagram & Code Preservation**:
  - Preserves diagrams, charts, and architectural figures extracted from PDFs.
  - Retains code listings and pseudo-code blocks verbatim in syntax-highlighted blocks.
- **Table of Contents (TOC) & Artifact Filtering**:
  - Automatically identifies and filters front-matter Table of Contents pages and indices.
  - Strips dingbat bullet glyphs (e.g. `■`), dot leaders, and residual slide metadata.
- **Minimalist & Accessible UI**:
  - Distraction-free, single-font typography (**Inter**).
  - Native Dark Mode, Light Mode, and High Contrast accessibility themes.
  - Quick in-browser actions: **Copy to Clipboard**, **Full-Screen Preview**, and **Print**.
- **100% Local & Private Processing**:
  - Uploaded files are processed strictly in memory and deleted immediately after extraction. No document data is stored or transmitted to third-party APIs.

---

## System Requirements

- **Python 3.10, 3.11, or 3.12**
- **Git**
- Modern web browser (Chrome, Edge, Firefox, Safari)

---

## Quick Start (Local Setup)

### 1. Clone or Open the Repository
```bash
git clone https://github.com/YOUR_USERNAME/AI-Document-Summarization-System.git
cd AI-Document-Summarization-System
```

### 2. Create and Activate Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
python run.py
```
This automatically launches the server on `http://127.0.0.1:7860` and opens it in your default browser.

To run without automatically popping the browser (e.g., in headless environments):
```bash
python app.py
```

---

## Project Structure

```
├── app.py                     # Flask application routes & background job orchestration
├── config.py                  # Central configuration parameters
├── run.py                     # Single-command launcher (server + browser launch)
├── Dockerfile                 # Container specification for Hugging Face Spaces / Docker
├── requirements.txt           # Python dependencies
├── pytest.ini                 # Pytest test configuration
│
├── templates/                 # Jinja2 HTML templates
│   ├── base.html              # Base layout (header, minimalist footer, accessibility controls)
│   ├── index.html             # Main summarizer interface (dropzone, length selector, results)
│   ├── about.html             # Project background, methodology, and design architecture
│   ├── help.html              # User guide & FAQ
│   ├── evaluation.html        # Performance benchmarks & evaluation metrics
│   └── accessibility.html     # Accessibility statement (WCAG 2.1 AA compliance)
│
├── static/
│   ├── css/
│   │   └── style.css          # Design tokens, CSS variables, dark/light themes, typography
│   └── js/
│       └── script.js          # Client-side validation, asynchronous polling, results handling
│
├── utils/
│   ├── summarizer.py          # Extractive NLP engine (TF-IDF, MMR, key-point scoring)
│   ├── text_cleaner.py        # Normalization, TOC line stripping, and output sanitization
│   ├── pdf_extractor.py       # PyMuPDF parser (TOC detection, image & code extraction)
│   ├── docx_extractor.py      # python-docx parser for Word documents
│   ├── txt_extractor.py       # Text file parser with encoding auto-detection
│   ├── chunking.py            # Sentence boundary detection and chunking
│   ├── file_validator.py      # File size and MIME type security validation
│   └── download_generator.py  # Structured text & PDF export utilities
│
├── evaluation/
│   └── rouge_evaluation.py   # ROUGE metric benchmarking against reference summaries
└── tests/                     # Comprehensive automated pytest test suite (17 unit tests)
```

---

## Running Automated Tests

Run the full automated test suite using `pytest`:
```bash
pytest
```

To run a specific test module:
```bash
pytest tests/test_summarizer.py
pytest tests/test_pdf.py
```

---

## Evaluation & Benchmarks

Evaluate extractive summarization performance using standard ROUGE metrics:
```bash
python -m evaluation.rouge_evaluation --data datasets/sample_documents/samples.csv
```
This computes ROUGE-1, ROUGE-2, and ROUGE-L precision, recall, and F1-scores, exporting the results to `evaluation/rouge_results.csv`.

---

## Deployment to Hugging Face Spaces

This repository includes a `Dockerfile` pre-configured for Docker-based Spaces:

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space).
2. Choose **Docker** as the Space SDK.
3. Push your repository to the Hugging Face remote:
   ```bash
   git remote add space https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
   git push space main
   ```
4. The container builds and starts on port `7860`.

---

## License & Credits

- **Institution**: Gombe State University &middot; Faculty of Science &middot; Department of Computer Science
- **Project**: Development of a Web-Based AI Document Summarization System
- **Year**: 2026
