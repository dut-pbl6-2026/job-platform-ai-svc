# job-platform-ai-svc: local-model resume scoring. No third-party AI API; the ONNX model is baked in at build time.
FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng tesseract-ocr-vie curl \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_THREAD_LIMIT=1 HF_HUB_DISABLE_XET=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv PATH=/opt/venv/bin:$PATH PYTHONPATH=/app \
    EMBEDDING_CACHE_DIR=/models PORT=6000
WORKDIR /app

RUN pip install --no-cache-dir uv
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project

COPY app ./app
COPY data ./data
COPY scripts ./scripts
# Model: reuse ./models when present (mise run models), else download now. (Dockerfile is copied only so the glob always matches.)
COPY Dockerfile models* /models/
RUN python scripts/download_models.py

ENV HF_HUB_OFFLINE=1
RUN useradd --create-home --uid 10001 app && chown -R app:app /app /models
USER app
EXPOSE 6000
HEALTHCHECK --interval=15s --timeout=5s --start-period=40s --retries=5 CMD curl -fsS http://localhost:6000/health || exit 1
CMD ["sh", "-c", "uvicorn app.main:create_app --factory --host 0.0.0.0 --port ${PORT}"]
