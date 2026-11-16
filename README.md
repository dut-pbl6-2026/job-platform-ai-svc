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
mise run sync-env     # copies the shared .env (JWT metadata, internal tokens, etc.)
JOB_SERVICE_URL=http://localhost:5002 mise run dev   # http://localhost:6000

curl -s localhost:6000/health
curl -s -H "Authorization: Bearer $TOKEN" -F cv_file=@cv.pdf -F job_id=<uuid> localhost:6000/api/ai/score-resume
```

Auth: `Authorization: Bearer <JWT from auth-svc>`. The service refuses to start unless `JWT_METADATA_ADDRESS` (auth-svc discovery URL, for example `http://auth-svc:5001/.well-known/openid-configuration`) is set. Tokens must be RS256 with a `kid` from that JWKS; HS256 is refused. `X-User-Id` is trusted only in local development: `DEV_AUTH_ENABLED=true` AND `DOTNET_ENVIRONMENT`/`ASPNETCORE_ENVIRONMENT` = `Development` AND `JWT_METADATA_ADDRESS` unset. Never set `DEV_AUTH_ENABLED` outside a developer machine. Through the gateway: `POST /api/ai/score-resume` requires login **and role `Recruiter` or `Admin`** — a match score is recruiter/admin-only. A candidate (role `User`, or a token with no role) gets `403` with the stable code `forbidden`; in dev auth the role comes from `X-User-Role`. The JWT role is read from the `role` claim or the ASP.NET `ClaimTypes.Role` URI.

**Embedding caller (search-svc).** search-svc calls `POST /api/internal/embed` and `GET /api/internal/embed/info` with its own bearer, `AI_INTERNAL_TOKEN_FOR_SEARCH` (the name follows `<CALLEE>_INTERNAL_TOKEN_FOR_<CALLER>`; the same name is set on search-svc). Only that key is accepted on those routes; the shared `INTERNAL_SERVICE_TOKEN`, the scoring token and user JWTs are refused with 401. It is compared in constant time on SHA-256 digests. Outside Development startup fails if it is unset or shorter than 32 characters, or equal to `AI_SCORING_SERVICE_TOKEN`. In Development an unset key makes those routes refuse every call.

**Scoring service (app-svc).** app-svc scores applications in the background with its own bearer token, `AI_SCORING_SERVICE_TOKEN` (environment only, not in the shared `.env`), not with the platform JWT secret. The token is accepted only by `POST /api/ai/score-resume`, where it identifies the caller as subject `app-svc-scoring` (role service). On every other route it is refused with `401`, and it is never a user identity. A user JWT is also accepted there only with role `Recruiter` or `Admin`; a candidate gets `403 forbidden`. The token is compared in constant time on SHA-256 digests; an empty or unset token matches nothing. In Production, startup fails if the token is set and shorter than 32 characters. The scoring caller is exempt from the per-user cap (`RATE_LIMIT_PER_MIN`) but not from the global scoring concurrency limit (`OCR_CONCURRENCY`, then `503 busy`). Known limitation: its scores are stored under the subject `app-svc-scoring`, so only that caller can reopen them; app-svc keeps the response body instead. Every response carries a `score_id`; the owner can reopen the result with `GET /api/ai/score-resume/{scoreId}` and delete it with `DELETE` (204). A score that is missing, expired or belongs to someone else is `404 score_not_found`.
In Docker: `docker compose up -d --build ai gateway` from `job-platform-infra` (the image bakes the model in; the container is offline at runtime).
Runtime footprint (measured, MiniLM + Tesseract): ~830 MB resident, ~900 MB peak → fits the `mem_limit: 1g` container; minimum host = 2 vCPU / 2 GB RAM / 5 GB disk, no GPU (see `eval/REPORT.md` §6).

## Configuration (env)

`PORT` (6000) · `JOB_SERVICE_URL` · `JWT_METADATA_ADDRESS` (auth-svc discovery URL; required unless explicit dev auth) · `JWT_ISSUER/AUDIENCE` · `REDIS_URL` (optional, either `redis://[:password@]host:6379/0` / `rediss://…` or the StackExchange form the .NET services share, `host:6379,password=…[,ssl=true][,defaultDatabase=1]`; unset or unreachable → in-process LRU, Redis is retried every 30 s) · `CACHE_TTL_SECONDS` (86400) · `SCORE_RETENTION_SECONDS` (2592000 = 30 days) · `SEARCH_SERVICE_URL` (chat job search; empty = unavailable) · `SEARCH_HTTP_TIMEOUT_SECONDS` (5) · `CHAT_RATE_PER_HOUR` (20) · `CHAT_SESSION_TTL_SECONDS` (86400) · `DEV_AUTH_ENABLED` (dev only, see Auth) · `EMBEDDER_FALLBACK` (unset = fail startup if the model does not load; `hashing` = run degraded, shown as `degraded: true` in `/health`) · `OCR_TIMEOUT_SECONDS` (20) · `OCR_CONCURRENCY` (2) · `OCR_DPI` (200) · `MAX_UPLOAD_BYTES` (5 MB) · `MAX_PAGES` (5) · `RATE_LIMIT_PER_MIN` (10 per user) · `AI_SCORING_SERVICE_TOKEN` (optional, >= 32 chars in Production; app-svc scoring bearer, see Auth) · `AI_INTERNAL_TOKEN_FOR_SEARCH` (search-svc's bearer for `/api/internal/*`; required, >= 32 chars, outside Development; see Auth) · `LOG_LEVEL` (accepts `Information` or `INFO`) · `EMBEDDING_CACHE_DIR` (models)

## Score cache

A scored response is cached for 24 h under `ai:score:{ruleset}:{params version}:{embedder}:{sha256(file)}:{job id}:{job version}:{lang}`, so a different embedder (for example the hashing fallback when the model fails to load), a refit of `data/model_params.json` or an edited job never reuses an old score. Only the response is stored, never the CV file or its text. `/health` reports `"cache": "redis"` or `"memory"`.

## Score records

Each response is also stored for its owner under `ai:scorerec:{score_id}` for 30 days (`SCORE_RETENTION_SECONDS`), together with the user id, job id and creation time. `score_id` is a UUIDv5 of the user id and the cache key, so scoring the same file for the same job again returns the same id. The shared cache never holds the id or the owner. `GET` returns the stored response (no re-scoring, no job-svc call); `DELETE` removes the record and the shared 24 h entry for that file. Both are authenticated and not rate limited.

What is kept: the response only. It includes short excerpts of the CV (`skills.evidence`, `top_matches[].cv_excerpt`) with e-mail addresses, links and phone numbers masked; **names and addresses are not masked**. The CV file and its full text are never stored or logged. Records are best-effort: with Redis down they live in memory and are lost on restart.

## Chat (PBL6-57)

`POST /api/ai/chat` streams an answer as server-sent events (`token`, `jobs`, `error`, `done`); `stream: false` returns JSON. Replies come from **rules and templates**: intent by patterns (job search, CV, interview, greeting, help, off-topic), parameters (role, skills, city) by dictionaries, job hits from search-svc `GET /api/search/jobs/semantic` (top 3 hits, query = the criteria text). No language model is called. Sessions: `POST /api/ai/session`, `GET /api/ai/session/{id}`, `GET /api/ai/session/{id}/history`, `DELETE /api/ai/session/{id}`; they are owned by the user, kept 24 h after the last message, and capped at 20 messages. Limit: 20 chat messages per user per hour (token bucket). Set `SEARCH_SERVICE_URL` (for example `http://search:5003`) or job search answers fail with an `error` event. The web and mobile clients no longer surface the assistant (owner decision, 2026-11-16); the endpoints remain available for API evaluation.

## Layout

```
app/parsing    PDF (text layer + hidden-text filter, per-page OCR fallback), DOCX, content-based type detection
app/nlp        normalise (NFC, fold), sections, skill taxonomy + sentence-scoped negation + "A or B" groups, years, education
app/scoring    specs (CV/job views), scorer (explainable components), template suggestions
app/chat       chat intents, rule-based replies over search-svc hits, sessions, SSE framing
app/embed.py   ONNX embeddings (fastembed, multilingual MiniLM-L12-v2) + deterministic hashing fallback for tests
data/          skills.yaml (taxonomy), model_params.json (fitted weights + calibration + score map)
eval/          jobs, 5 760-pair label grid, blind labels (A/B + Haiku + Sonnet), run.py / fit_grid.py, REPORT.md
tests/         370+ unit/API/robustness tests (no model download needed)
```

## Develop

`mise run verify` (ruff + pytest). `mise run eval` fits the weights on the train jobs and rewrites `eval/results_*.json`.
`mise run eval-grid` fits on the full 5 760-pair label grid (`eval/labels/grid`) and writes `data/model_params.json`; see `eval/REPORT.md`.
