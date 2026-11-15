# AGENTS — job-platform-ai-svc

> Resume scoring (SCORE-01). Spec: `job-platform-docs/docs/ai/resume-scoring.md`. Evaluation: `eval/REPORT.md`. Git: `job-platform-docs/.github/git-strategy.md` (`feature/* → main`, scope `ai`).

## Hard constraints
- **No third-party AI/LLM API** (supervisor requirement). Only local models/rules. Never add OpenAI/Gemini/Anthropic clients or keys.
- CV text is PII: never log it, never persist it; only the score result is stored, in two places: a cache shared by file (Redis/LRU, 24 h, key = ruleset + params version + embedder name + sha256(cv) + job id + job version + lang) and a per-user score record (`ai:scorerec:{score_id}`, 30 days, `SCORE_RETENTION_SECONDS`, holds the owner and the same response). Both are best-effort: memory when Redis is down, lost on restart then. The response carries short CV excerpts (`skills.evidence`, `top_matches[].cv_excerpt`) with e-mail, links and phone masked, but names and addresses are not masked. `DELETE` removes the record and the shared entry. `REDIS_URL` accepts `redis://…` and the StackExchange form `host:port,password=…` shared with the .NET services; on a Redis error the cache uses memory and retries after 30 s. Never log the connection string (redis-py errors can contain the password).
- No literals for URLs/secrets/ports in code — read from env (`app/config.py`).

## Commands (mise)
`mise run install` · `models` (one-off download) · `dev` (:6000) · `test` · `lint` · `verify` · `eval` (240-pair ablation) · `eval-grid` (production fit)
Agent: use `mise exec -- uv run ...` in non-interactive shells. Tests run without the model (hashing embedder); `eval`/`eval-grid` need `models/`.

## Architecture
`parse (text|OCR|DOCX) → sanitize hidden text → normalise/sections → extract (skills, years, degree) → embed (chunks) → score → suggestions`. The scorer is a pure function of `(CvSpec, JobSpec, embedder, params)`. Weights/calibration/score map live in `data/model_params.json` and are **fitted** by `eval/fit_grid.py` (5 760-pair label grid, train jobs only), not hand-set. `eval/run.py` is the older 240-pair ablation and does **not** write the shipped parameters.
- Add skills in `data/skills.yaml` (canonical names must not be common English words; broad skills go in group `generic`).
- Any scoring change must be re-evaluated (`mise run eval-grid` + `mise run eval`) and reported in `eval/REPORT.md`; do not tune on the held-out jobs.
- Third-party embedding APIs are **not** allowed in the service; they may only be used in `eval/compare_embedders.py` to measure a provider (never ship their vectors or keys).

## API (via gateway `/api/ai/*`, port 6000)
`POST /api/ai/score-resume` (multipart `cv_file`, `job_id`; response includes `score_id`) · `GET` / `DELETE /api/ai/score-resume/{scoreId}` (owner only, 204 on delete, `404 score_not_found` for missing, expired or someone else's, `400 bad_request` for a non-UUID) · `GET /api/ai/health` · `GET /health`. Errors are RFC 7807 with a stable `code` (`cv_no_text`, `cv_encrypted`, `cv_unreadable`, `unsupported_media_type`, `file_too_large`, `job_not_found`, `rate_limited`, `score_not_found`, `busy`, `job_service_unavailable`).
