# syntax=docker/dockerfile:1
# Spendly: Flask + SQLite app, served by gunicorn. No LLM is inside this image: the app uses
# Ollama on the host (OLLAMA_URL) or Groq (LLM_PROVIDER=groq, GROQ_API_KEY), both set in .env
# and passed in by docker-compose.yml at run time, never baked into the image.

FROM python:3.11-slim

# No .pyc files, unbuffered logs (so `docker compose logs` shows [AI] lines immediately),
# slow-network-friendly pip (long timeout + retries). HF_HOME = where faster-whisper keeps the downloaded model
# (a named volume in docker-compose.yml, so it's downloaded only once).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models/huggingface \
    SPENDLY_DB_PATH=/app/data/spendly.db

# System libraries: none needed. opencv-python-headless, PyMuPDF and faster-whisper (PyAV,
# which bundles its own ffmpeg) all ship self-contained wheels for Linux x86-64.

# Non-root user that will run the app.
RUN useradd --create-home --uid 1000 spendly

WORKDIR /app

# 1) Dependencies first: this layer is cached and only rebuilt when requirements.txt changes,
#    not on every code edit.
#    The cache mount keeps downloaded wheels on the build machine between builds (NOT in the
#    image), so a failed or changed build doesn't download ~1 GB again.
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

# 2) Then the code (changes often, cheap layer).
COPY --chown=spendly:spendly . .

# Folders the volumes mount onto. Created here, owned by the app user, so a NEW named volume
# starts with the right owner and the non-root user can write the database / model cache.
RUN mkdir -p /app/data /models/huggingface && chown -R spendly:spendly /app/data /models

USER spendly

EXPOSE 5001

# gunicorn instead of the Flask dev server. ONE worker: chat memory (InMemorySaver) and the
# AI rate limit live in this process's memory, so several worker processes would each have
# their own copy (lost chats, rate limit x workers). Threads give concurrency inside it.
# --timeout 300: an AI answer can take minutes on a laptop GPU.
CMD ["gunicorn", "--workers", "1", "--threads", "8", "--bind", "0.0.0.0:5001", \
     "--timeout", "300", "--access-logfile", "-", "app:app"]
