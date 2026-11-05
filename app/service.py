"""Orchestration: bytes + job dict -> score response. Synchronous and CPU-bound; run in a worker thread."""
import hashlib
import time
from datetime import date

from app.config import Settings
from app.embed import Embedder
from app.parsing.parse import parse_cv
from app.scoring import scorer
from app.scoring.specs import build_cv_spec, build_job_spec
from app.scoring.suggest import hidden_text_suggestion


def cache_key(cv_bytes: bytes, job_id: str, job_version: str, lang: str = "") -> str:
    h = hashlib.sha256(cv_bytes).hexdigest()
    return f"{scorer.RULESET}:{h}:{job_id}:{job_version}:{lang}"


def score_cv_bytes(cv_bytes: bytes, job: dict, settings: Settings, emb: Embedder, today: date | None = None, lang: str | None = None) -> dict:
    t0 = time.perf_counter()
    parsed = parse_cv(cv_bytes, settings)
    cv = build_cv_spec(parsed.text, today)
    result = scorer.score(cv, build_job_spec(job), emb, lang=lang)
    warnings = list(parsed.warnings)
    if parsed.hidden_removed:
        result["suggestions"] = ([hidden_text_suggestion(lang or cv.language)] + result["suggestions"])[:6]
    if "not_cv_like" in result["notes"]:
        warnings.append("not_cv_like")
    result["warnings"] = warnings
    result["meta"] = {
        "parser": parsed.parser, "pages": parsed.pages, "pages_processed": parsed.pages_processed, "language": cv.language,
        "ruleset": scorer.RULESET, "params": scorer.load_params().version, "model": emb.name, "cached": False,
        "elapsed_ms": round((time.perf_counter() - t0) * 1000),
    }
    return result
