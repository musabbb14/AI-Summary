"""
utils/file_validator.py
Guards the front door: makes sure only the file types we can actually
process get passed further down the pipeline, and gives a clear,
human-readable reason when a file is rejected.
"""

import os
from werkzeug.utils import secure_filename
from config import Config


class FileValidationError(Exception):
    """Raised when an uploaded file fails validation."""
    pass


def get_extension(filename):
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


def validate_upload(file_storage):
    """
    Validates a Werkzeug FileStorage object.
    Returns a safe filename if valid, otherwise raises FileValidationError
    with a message meant to be shown directly to the user.
    """
    if file_storage is None or file_storage.filename == "":
        raise FileValidationError("No file was selected. Please choose a document to summarize.")

    filename = secure_filename(file_storage.filename)
    ext = get_extension(filename)

    if ext not in Config.ALLOWED_EXTENSIONS:
        raise FileValidationError(
            f"'.{ext}' files aren't supported yet. Please upload a PDF, DOCX, or TXT file."
        )

    return filename
