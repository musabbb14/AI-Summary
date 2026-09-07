"""Basic route tests that don't require the AI model to be loaded."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import app


def test_index_page_loads():
    client = app.test_client()
    response = client.get("/")
    assert response.status_code == 200


def test_summarize_without_file_returns_400():
    client = app.test_client()
    response = client.post("/summarize", data={})
    assert response.status_code == 400


def test_unknown_job_status_returns_404():
    client = app.test_client()
    response = client.get("/status/does-not-exist")
    assert response.status_code == 404
