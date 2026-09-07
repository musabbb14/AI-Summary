"""
run.py
The ONE file to run locally. Double-click it (or `python run.py`) and it:
  1. Starts the Flask backend.
  2. Waits for it to be ready.
  3. Opens your default browser straight to the app.

This exists so you don't have to think about ports or URLs — just run
this file and the summarizer opens itself.
"""

import os
import sys
import time
import socket
import threading
import webbrowser

os.environ.setdefault("PORT", "7860")

def _port_is_open(host, port, timeout=0.5):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _open_browser_when_ready(host, port, max_wait_seconds=60):
    waited = 0
    while waited < max_wait_seconds:
        if _port_is_open(host, port):
            webbrowser.open(f"http://{host}:{port}/")
            return
        time.sleep(0.5)
        waited += 0.5
    print(f"Server didn't come up within {max_wait_seconds}s — "
          f"open http://{host}:{port}/ manually once it does.")


if __name__ == "__main__":
    host = "127.0.0.1"
    port = int(os.environ.get("PORT", 7860))

    print("=" * 60)
    print(" AI Document Summarization System")
    print(" Starting up... the model loads on first request, so the")
    print(" very first summary may take a little longer than the rest.")
    print("=" * 60)

    threading.Thread(target=_open_browser_when_ready, args=(host, port), daemon=True).start()

    # Import after setting env vars / before browser thread races too far ahead.
    from app import app  # noqa: E402

    try:
        app.run(host=host, port=port, debug=False)
    except KeyboardInterrupt:
        sys.exit(0)
