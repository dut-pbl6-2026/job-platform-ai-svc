# AGENTS — job-platform-ai-svc

> Resume scoring (SCORE-01). Spec: `job-platform-docs/docs/ai/resume-scoring.md`. Evaluation: `eval/REPORT.md`. Git: `job-platform-docs/.github/git-strategy.md` (`feature/* → main`, scope `ai`).

## Hard constraints
- **No third-party AI/LLM API** (supervisor requirement). Only local models/rules. Never add OpenAI/Gemini/Anthropic clients or keys.
- CV text is PII: never log it, never persist it; only the score result is cached (Redis/LRU, 24 h, key = ruleset + sha256(cv) + job id + job version).
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
`POST /api/ai/score-resume` (multipart `cv_file`, `job_id`) · `GET /api/ai/health` · `GET /health`. Errors are RFC 7807 with a stable `code` (`cv_no_text`, `cv_encrypted`, `cv_unreadable`, `unsupported_media_type`, `file_too_large`, `job_not_found`, `rate_limited`, `busy`, `job_service_unavailable`).
