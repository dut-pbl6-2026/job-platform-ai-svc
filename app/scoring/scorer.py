"""Scoring (SPEC section 6). Pure function of (CvSpec, JobSpec, embedder, params): deterministic and explainable."""
import json
from dataclasses import dataclass
from functools import lru_cache
from itertools import pairwise
from pathlib import Path

import numpy as np

from app.embed import Embedder
from app.nlp.education import rank
from app.nlp.normalize import redact
from app.nlp.skills import is_generic
from app.scoring.specs import CvSpec, JobSpec
from app.scoring.suggest import make_suggestions

RULESET = "1.0.0"
PARAMS_PATH = Path(__file__).resolve().parents[2] / "data" / "model_params.json"
NICE_WEIGHT = 0.4
GENERIC_FACTOR = 0.3  # broad skills (communication, leadership, MS Office...) count for less
CAP_NOT_CV = 40.0
CAP_NO_REQUIRED = 60.0
TOP_MATCHES = 3


@dataclass
class Params:
    weights: dict[str, float]
    calibration: dict[str, tuple[float, float]]  # {"same": (lo, hi), "cross": (lo, hi)} by CV/JD language pair
    title_weight: float
    version: str
    label_map: dict | None = None  # optional monotone map raw 0-100 score -> calibrated 0-100 score


@lru_cache
def load_params(path: str = str(PARAMS_PATH)) -> Params:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    cal = {k: (v["lo"], v["hi"]) for k, v in d["calibration"].items()}
    return Params(d["weights"], cal, d.get("title_weight", 0.3), d["version"], d.get("label_map"))


def apply_label_map(score: float, p: Params) -> float:
    """Map the raw weighted score onto the calibrated 0-100 scale (monotone, so the ranking is unchanged).

    The map is fitted on labelled pairs (eval/fit_grid.py): it lines the score up with the rubric
    (0-4) instead of leaving it compressed. Linear maps are `a * score + b`; isotonic maps are
    piecewise linear through `knots` = [[raw_score, calibrated_score], ...]."""
    lm = p.label_map
    if not lm:
        return score
    if lm.get("type") == "isotonic" and lm.get("knots"):
        knots = lm["knots"]
        xs = [k[0] for k in knots]
        ys = [k[1] for k in knots]
        if score <= xs[0]:
            return float(ys[0])
        if score >= xs[-1]:
            return float(ys[-1])
        for (x0, y0), (x1, y1) in pairwise(knots):
            if x0 <= score <= x1:
                if x1 == x0:
                    return float(y1)
                t = (score - x0) / (x1 - x0)
                return float(y0 + t * (y1 - y0))
        return float(ys[-1])
    a, b = lm.get("a", 1.0), lm.get("b", 0.0)
    return float(a * score + b)


def band(score: float) -> str:
    return "Strong" if score >= 75 else "Good" if score >= 55 else "Partial" if score >= 35 else "Weak"


def lang_pair(cv_lang: str, job_lang: str) -> str:
    """Cosines are systematically lower across languages (VI job vs EN CV), so each pair type has its own calibration."""
    return "cross" if {cv_lang, job_lang} == {"en", "vi"} else "same"


def _calibrate(raw: float, p: Params, pair: str) -> float:
    lo, hi = p.calibration[pair]
    return float(np.clip((raw - lo) / max(hi - lo, 1e-6), 0.0, 1.0))


def semantic_signal(cv: CvSpec, job: JobSpec, emb: Embedder, p: Params) -> tuple[float, float, list[dict]]:
    """Returns (raw, title_fit, top_matches). raw = mixture of requirement coverage and title fit (uncalibrated cosine).

    Embedders may expose asymmetric sides (embed_query / embed_passage); symmetric ones just use embed()."""
    if not cv.chunks:
        return 0.0, 0.0, []
    embed_q = getattr(emb, "embed_query", emb.embed)
    embed_p = getattr(emb, "embed_passage", emb.embed)
    chunk_vecs = embed_p(cv.chunks)
    lines = job.lines or ([job.title] if job.title else [])
    cover, matches = 0.0, []
    if lines:
        sims = embed_q(lines) @ chunk_vecs.T  # (lines, chunks)
        best = sims.max(axis=1)
        cover = float(best.mean())
        for i in np.argsort(-best)[:TOP_MATCHES]:
            matches.append({"requirement": lines[i][:160], "cv_excerpt": redact(cv.chunks[int(sims[i].argmax())])[:160], "similarity": round(float(best[i]), 3)})
    title_fit = 0.0
    if job.title and cv.title_candidates:
        title_fit = float((embed_q([job.title]) @ embed_p(cv.title_candidates).T).max())
    raw = (1 - p.title_weight) * cover + p.title_weight * title_fit if lines else title_fit
    return raw, title_fit, matches


def score(cv: CvSpec, job: JobSpec, emb: Embedder, p: Params | None = None, lang: str | None = None) -> dict:
    p = p or load_params()
    present, negated = cv.skills.present, cv.skills.negated

    # --- skills: each requirement is an item; alternatives ("Django or Flask") form one item satisfied by any member
    def w_of(members: list[str], base: float) -> float:
        return base * (GENERIC_FACTOR if all(is_generic(m) for m in members) else 1.0)

    grouped = {m for g in job.groups for m in g}
    items: list[tuple[list[str], float, bool]] = [(g, w_of(g, 1.0), True) for g in job.groups]
    items += [([s], w_of([s], 1.0), True) for s in job.required if s not in grouped]
    items += [([s], w_of([s], NICE_WEIGHT), False) for s in job.nice if s not in grouped]
    label = lambda members: " / ".join(members)
    matching: list[str] = []
    missing_req: list[str] = []
    negated_req: list[str] = []
    got = total = 0.0
    for members, w, is_required in items:
        total += w
        hit = [m for m in members if m in present]
        if hit:
            got += w
            matching += [m for m in hit if m not in matching]
        elif is_required:
            (negated_req if any(m in negated for m in members) else missing_req).append(label(members))
    skills_score = got / total if total else None
    wanted = {m: 1.0 for members, _, _ in items for m in members}
    extra = sorted((s for s in present if s not in wanted), key=lambda s: -cv.skills.counts.get(s, 0))[:10]

    # --- semantic
    raw, title_fit, top = semantic_signal(cv, job, emb, p)
    pair = lang_pair(cv.language, job.language)
    semantic = _calibrate(raw, p, pair)

    # --- qualification fit (only meaningful inside the job's domain, so it is gated by relevance)
    relevance = max(semantic, skills_score or 0.0)
    experience = raw_experience = None
    if job.required_years:
        raw_experience = min(cv.years / job.required_years, 1.0)
        experience = raw_experience * relevance
    education = raw_education = None
    if job.required_level != "none":
        gap = rank(cv.level) - rank(job.required_level)
        raw_education = 1.0 if gap >= 0 else 0.5 if gap == -1 else 0.0
        education = raw_education * relevance

    comps = {"skills": skills_score, "semantic": semantic, "experience": experience, "education": education}
    avail = {k: v for k, v in comps.items() if v is not None}
    wsum = sum(p.weights[k] for k in avail)
    overall = 100.0 * sum(p.weights[k] * v for k, v in avail.items()) / wsum if wsum else 0.0
    overall = apply_label_map(overall, p)

    notes, caps = [], []
    core = [m for members, _, req in items if req for m in members if not is_generic(m)]
    if core and not any(m in present for m in core):
        overall = min(overall, CAP_NO_REQUIRED)
        caps.append("no_required_skill_found")
    if not cv.cv_like:
        overall = min(overall, CAP_NOT_CV)
        caps.append("not_cv_like")
    if skills_score is None:
        notes.append("job_has_no_recognised_skills")
    overall = round(float(np.clip(overall, 0, 100)))

    suggestions = make_suggestions(cv, job, missing_req, negated_req, comps, caps, lang)
    return {
        "overall_score": overall,
        "band": band(overall),
        "breakdown": {
            "skills": {"score": None if skills_score is None else round(100 * skills_score), "matching": matching, "missing": missing_req, "negated": negated_req, "extra": extra,
                       "evidence": {s: present[s] for s in matching[:6]}},
            "semantic": {"score": round(100 * semantic), "title_fit": round(title_fit, 3), "top_matches": top},
            "experience": {"score": None if raw_experience is None else round(100 * raw_experience), "cv_years": cv.years, "required_years": job.required_years},
            "education": {"score": None if raw_education is None else round(100 * raw_education), "cv_level": cv.level, "required_level": job.required_level},
        },
        "suggestions": suggestions,
        "notes": notes + caps,
        "components": {k: (None if v is None else round(v, 4)) for k, v in comps.items()},
        "raw_semantic": round(raw, 4),
        "lang_pair": pair,
        "relevance": round(relevance, 3),
    }
