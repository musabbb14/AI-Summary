# Dockerfile — used by Hugging Face Spaces (Docker SDK).
# Plain Flask apps aren't one of Spaces' built-in SDKs (Gradio/Streamlit),
# so we ship our own container. Spaces builds this automatically whenever
# you push to the Space's git repo.

FROM python:3.11-slim

# System deps: PyMuPDF needs a couple of shared libs to build/run cleanly.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Spaces (Docker SDK) routes traffic to port 7860 by convention.
ENV PORT=7860
EXPOSE 7860

# Pre-create writable folders (Spaces containers run as a non-root user).
RUN mkdir -p uploads outputs logs && chmod -R 777 uploads outputs logs

CMD ["python", "app.py"]
