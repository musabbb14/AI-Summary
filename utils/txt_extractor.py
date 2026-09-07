"""
utils/txt_extractor.py
Reads plain-text files. Tries UTF-8 first, falls back to latin-1 so
older or oddly-encoded files don't crash the pipeline.
"""


def extract_text_from_txt(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
    except UnicodeDecodeError:
        with open(filepath, "r", encoding="latin-1") as f:
            text = f.read()

    if not text.strip():
        raise ValueError("This text file appears to be empty.")

    return text
