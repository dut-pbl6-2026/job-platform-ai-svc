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
mise run models       # one-off: downloads the ONNX embedding model (paraphrase-multilingual-MiniLM-L12-v2, ~235 MB) into ./models
mise run sync-env     # copies the shared .env (JWT secret etc.)
JOB_SERVICE_URL=http://localhost:5002 mise run dev   # http://localhost:6000

curl -s localhost:6000/health
curl -s -H "Authorization: Bearer $TOKEN" -F cv_file=@cv.pdf -F job_id=<uuid> localhost:6000/api/ai/score-resume
```

Auth: `Authorization: Bearer <JWT from auth-svc>`; with no `JWT_SECRET` configured the service accepts `X-User-Id` (dev only). Through the gateway: `POST /api/ai/score-resume` (requires login).
In Docker: `docker compose up -d --build ai gateway` from `job-platform-infra` (the image bakes the model in; the container is offline at runtime).
Runtime footprint (measured, MiniLM + Tesseract): ~830 MB resident, ~900 MB peak → fits the `mem_limit: 1g` container; minimum host = 2 vCPU / 2 GB RAM / 5 GB disk, no GPU (see `eval/REPORT.md` §6).

## Configuration (env)

`PORT` (6000) · `JOB_SERVICE_URL` · `REDIS_URL` (optional, either `redis://[:password@]host:6379/0` / `rediss://…` or the StackExchange form the .NET services share, `host:6379,password=…[,ssl=true][,defaultDatabase=1]`; unset or unreachable → in-process LRU, Redis is retried every 30 s) · `CACHE_TTL_SECONDS` (86400) · `JWT_SECRET/ISSUER/AUDIENCE` · `OCR_CONCURRENCY` (2) · `OCR_DPI` (200) · `MAX_UPLOAD_BYTES` (5 MB) · `MAX_PAGES` (5) · `RATE_LIMIT_PER_MIN` (10 per user) · `LOG_LEVEL` (accepts `Information` or `INFO`) · `EMBEDDING_CACHE_DIR` (models)

## Score cache

A scored response is cached for 24 h under `ai:score:{ruleset}:{params version}:{embedder}:{sha256(file)}:{job id}:{job version}:{lang}`, so a different embedder (for example the hashing fallback when the model fails to load), a refit of `data/model_params.json` or an edited job never reuses an old score. Only the response is stored, never the CV file or its text. `/health` reports `"cache": "redis"` or `"memory"`.

## Layout

```
app/parsing    PDF (text layer + hidden-text filter, per-page OCR fallback), DOCX, content-based type detection
app/nlp        normalise (NFC, fold), sections, skill taxonomy + sentence-scoped negation + "A or B" groups, years, education
app/scoring    specs (CV/job views), scorer (explainable components), template suggestions
app/embed.py   ONNX embeddings (fastembed, multilingual MiniLM-L12-v2) + deterministic hashing fallback for tests
data/          skills.yaml (taxonomy), model_params.json (fitted weights + calibration + score map)
eval/          jobs, 5 760-pair label grid, blind labels (A/B + Haiku + Sonnet), run.py / fit_grid.py, REPORT.md
tests/         79+ unit/API/robustness tests (no model download needed)
```

## Develop

`mise run verify` (ruff + pytest). `mise run eval` fits the weights on the train jobs and rewrites `eval/results_*.json`.
`mise run eval-grid` fits on the full 5 760-pair label grid (`eval/labels/grid`) and writes `data/model_params.json`; see `eval/REPORT.md`.
