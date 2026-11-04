# job-platform-ai-svc

Python FastAPI — part of **Vietnam Job Platform** (`pbl6`) under [`dut-pbl6-2026`](https://github.com/dut-pbl6-2026).

**Resume scoring (SCORE-01)**: `POST /api/ai/score-resume` scores a CV (PDF — text *or scanned* — or DOCX) against a job posting and returns a 0–100 match with an explainable breakdown and suggestions.
**No third-party AI/LLM API is called**: everything runs locally (ONNX embedding model + Tesseract OCR + rules), per the supervisor's constraint.

- Spec: `job-platform-docs/docs/ai/resume-scoring.md` (EN) / `.vi.md` · Evaluation: [`eval/REPORT.md`](eval/REPORT.md)
- Branch flow: `feature/* → main` (see job-platform-docs/.github/git-strategy.md) · Jira PBL6 · TM2

## Run locally

```bash
# prerequisites: Python 3.12 + uv (mise install), tesseract-ocr with eng+vie  (apt install tesseract-ocr tesseract-ocr-eng tesseract-ocr-vie)
mise run install      # uv sync
mise run models       # one-off: downloads the ONNX embedding model (235 MB) into ./models
mise run sync-env     # copies the shared .env (JWT secret etc.)
JOB_SERVICE_URL=http://localhost:5002 mise run dev   # http://localhost:6000

curl -s localhost:6000/health
curl -s -H "Authorization: Bearer $TOKEN" -F cv_file=@cv.pdf -F job_id=<uuid> localhost:6000/api/ai/score-resume
```

Auth: `Authorization: Bearer <JWT from auth-svc>`; with no `JWT_SECRET` configured the service accepts `X-User-Id` (dev only). Through the gateway: `POST /api/ai/score-resume` (requires login).
In Docker: `docker compose up -d --build ai gateway` from `job-platform-infra` (the image bakes the model in; the container is offline at runtime).

## Configuration (env)

`PORT` (6000) · `JOB_SERVICE_URL` · `REDIS_URL` (optional; falls back to in-process LRU) · `JWT_SECRET/ISSUER/AUDIENCE` · `OCR_CONCURRENCY` (2) · `OCR_DPI` (200) · `MAX_UPLOAD_BYTES` (5 MB) · `MAX_PAGES` (5) · `RATE_LIMIT_PER_MIN` (10 per user) · `LOG_LEVEL` (accepts `Information` or `INFO`) · `EMBEDDING_CACHE_DIR` (models)

## Layout

```
app/parsing    PDF (text layer + hidden-text filter, per-page OCR fallback), DOCX, content-based type detection
app/nlp        normalise (NFC, fold), sections, skill taxonomy + sentence-scoped negation + "A or B" groups, years, education
app/scoring    specs (CV/job views), scorer (explainable components), template suggestions
app/embed.py   ONNX embeddings (fastembed) + deterministic hashing fallback for tests
data/          skills.yaml (taxonomy), model_params.json (fitted weights + calibration)
eval/          jobs, pairs, blind labels (A/B), run.py (fit + evaluate), REPORT.md
tests/         66+ unit/API/robustness tests (no model download needed)
```

## Develop

`mise run verify` (ruff + pytest). `mise run eval` re-fits weights on the train jobs and rewrites `eval/results_*.json` and `data/model_params.json`.
