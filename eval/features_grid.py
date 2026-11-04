"""Feature exploration on the full label grid (24 jobs x 240 CVs = 5 760 pairs).

Computes candidate signals for every pair (current scorer components + extra candidate
features), then reports each feature's rank correlation with the labels and the result of
fitting a blend per fold (cross-validated by job). Used to decide which signals are worth
putting into the production scorer - no held-out job is used for any decision here.

Usage:
  python eval/features_grid.py [--labels PATH] [--model NAME] [--hashing] [--out FILE]
"""
import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import Settings
from app.embed import HashingEmbedder, OnnxEmbedder
from app.nlp.normalize import fold
from app.nlp.skills import load_index
from app.scoring import scorer
from app.scoring.specs import build_cv_spec, build_job_spec

HERE = Path(__file__).parent
DEFAULT_LABELS = os.path.expanduser("~/enfue-eval/dataset/pairs_labels.jsonl")
TOKEN = re.compile(r"[a-z0-9+#]+")
STOP = {
    "a", "an", "and", "or", "of", "in", "with", "to", "for", "the", "at", "on", "as", "by", "is", "are", "be",
    "have", "has", "had", "will", "we", "you", "your", "our", "their", "that", "this", "it", "from", "including",
    "such", "other", "more", "least", "years", "year", "experience", "experienced", "work", "working", "ability",
    "able", "strong", "solid", "good", "excellent", "knowledge", "familiarity", "skills", "skill", "plus", "preferred",
    "advantage", "required", "requirements", "role", "team", "teams", "candidate", "candidates", "job", "position",
    "than", "using", "use", "used", "well", "also", "etc", "may", "must", "should", "not", "no",
}


def content_tokens(text):
    return [t for t in TOKEN.findall(fold(text)) if t not in STOP and len(t) > 2]


def token_coverage(req_text, cv_text):
    """Fraction of the requirement's content tokens that appear in the CV (bag-of-words)."""
    req = content_tokens(req_text)
    if not req:
        return None
    cv = set(TOKEN.findall(fold(cv_text)))
    return sum(1 for t in req if t in cv) / len(req)


def alias_coverage(job_lines, cv_aliases, cv_canon):
    """Strict alias-level coverage: a JD alias only counts when the CV names the same alias."""
    if not job_lines:
        return None
    idx = load_index()
    hit, total = 0, 0
    for line in job_lines:
        aliases = set()
        for m in idx.pattern.finditer(fold(line)):
            aliases.add(m.group(0))
        for a in aliases:
            total += 1
            if a in cv_aliases:
                hit += 1
    return hit / total if total else None


def cv_alias_set(text):
    idx = load_index()
    return {m.group(0) for m in idx.pattern.finditer(fold(text))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=DEFAULT_LABELS)
    ap.add_argument("--model", default=Settings().embedding_model)
    ap.add_argument("--hashing", action="store_true")
    ap.add_argument("--out", default=str(HERE / "results_features_grid.json"))
    a = ap.parse_args()

    jobs = json.loads((HERE / "jobs.json").read_text(encoding="utf-8"))
    grid = {}
    if str(a.labels).endswith(".json"):
        blob = json.loads(Path(a.labels).read_text(encoding="utf-8"))
        labels = blob.get("labels", blob)
        ind_file = HERE / "labels" / "grid" / "in_domain_768.json"
        in_domain = set(json.loads(ind_file.read_text(encoding="utf-8"))) if ind_file.exists() else set()
        for pair, score in labels.items():
            job, cv = pair.split("__", 1)
            grid[pair] = {"pair": pair, "job": job, "cv": cv, "haiku_label": int(score), "in_domain": pair in in_domain}
    else:
        with open(a.labels, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                grid[r["pair"]] = r
    emb = HashingEmbedder() if a.hashing else OnnxEmbedder(a.model, Settings().embedding_cache_dir, cache_size=60000)

    t0 = time.time()
    cvs = {}
    for cid in sorted({r["cv"] for r in grid.values()}):
        cvs[cid] = build_cv_spec((HERE / "cache" / "cvs" / f"{cid}.txt").read_text(encoding="utf-8"))
    jb = {j["slug"]: build_job_spec(j) for j in jobs}
    print(f"specs in {time.time() - t0:.0f}s")

    params = scorer.load_params()
    rows = []
    t0 = time.time()
    for i, (pair, r) in enumerate(grid.items()):
        job, cv_id = r["job"], r["cv"]
        cv, js = cvs[cv_id], jb[job]
        sc = scorer.score(cv, js, emb, params)
        raw, title_fit, _ = scorer.semantic_signal(cv, js, emb, params)
        cover = None
        if js.lines and cv.chunks:
            sims = emb.embed(js.lines) @ emb.embed(cv.chunks).T
            cover = float(sims.max(axis=1).mean())
        aliases = cv_alias_set(cv.text)
        rows.append({
            "pair": pair, "job": job, "cv": cv_id, "label": int(r["haiku_label"]), "in_domain": bool(r["in_domain"]),
            "skills": sc["components"]["skills"], "semantic": sc["components"]["semantic"],
            "experience": sc["components"]["experience"], "education": sc["components"]["education"],
            "raw_cos": raw, "cover": cover, "title_fit": title_fit,
            "token_cov": token_coverage(js.lines[0] if False else " ".join(js.lines), cv.text),
            "alias_cov": alias_coverage(js.lines, aliases, set(cv.skills.present)),
            "exp_raw": (min(cv.years / js.required_years, 1.0) if js.required_years else None),
            "edu_raw": (1.0 if js.required_level == "none" else None),
            "cv_years": cv.years, "req_years": js.required_years,
            "caps": sc["notes"], "relevance": sc["relevance"],
            "n_req": len(js.required), "n_present": len(cv.skills.present),
        })
        if i and i % 1000 == 0:
            print(f"  {i}/{len(grid)} ({time.time() - t0:.0f}s)")
    print(f"features in {time.time() - t0:.0f}s")

    y = np.asarray([r["label"] for r in rows], float)
    feats = ["skills", "semantic", "experience", "education", "raw_cos", "cover", "title_fit",
             "token_cov", "alias_cov", "exp_raw"]
    print("\n=== single-feature Spearman (all 5760 / in-domain 768) ===")
    ind = np.asarray([r["in_domain"] for r in rows])
    for f in feats:
        v = np.asarray([r[f] if r[f] is not None else np.nan for r in rows], float)
        ok = ~np.isnan(v)
        rho_all = spearmanr(v[ok], y[ok])[0]
        ok2 = ok & ind
        rho_ind = spearmanr(v[ok2], y[ok2])[0] if ok2.sum() > 10 else float("nan")
        print(f"  {f:12s} n={ok.sum():5d} rho_all={rho_all:+.3f} rho_in_domain={rho_ind:+.3f}")

    # ---- blend the candidate features, cross-validated by job
    usable = [f for f in feats if f not in ("experience", "education", "exp_raw")]
    X = np.column_stack([np.asarray([r[f] if r[f] is not None else 0.0 for r in rows], float) for f in usable])

    def fit_blend(idx):
        Xt, yt = X[idx], y[idx]
        mu, sd = Xt.mean(axis=0), Xt.std(axis=0) + 1e-9
        Z = (Xt - mu) / sd

        def obj(w):
            w = np.abs(w)
            if w.sum() == 0:
                return 1.0
            w = w / w.sum()
            s = Z @ w
            if np.std(s) == 0:
                return 1.0
            return -spearmanr(s, yt)[0]

        best = None
        starts = [np.ones(len(usable)) / len(usable)]
        for cand in ([0.4, 0.3, 0.1, 0.1, 0.1], [0.5, 0.2, 0.1, 0.1, 0.1], [0.3, 0.2, 0.1, 0.2, 0.2]):
            starts.append(np.asarray((cand + [0.0] * len(usable))[:len(usable)], float))
        for st in starts:
            res = minimize(obj, np.asarray(st, float), method="Nelder-Mead", options={"maxiter": 1200, "xatol": 1e-3})
            if best is None or res.fun < best.fun:
                best = res
        w = np.abs(best.x)
        w = w / w.sum()
        return w, mu, sd

    by_job = defaultdict(list)
    for i, r in enumerate(rows):
        by_job[r["job"]].append(i)
    job_slugs = sorted(by_job)
    folds = []
    for f in range(4):
        tr = [i for j, idxs in enumerate(job_slugs) if j % 4 == f for i in by_job[idxs]]
        te = [i for j, idxs in enumerate(job_slugs) if j % 4 != f for i in by_job[idxs]]
        w, mu, sd = fit_blend(tr)
        s_te = ((X[te] - mu) / sd) @ w
        folds.append({
            "rho": float(spearmanr(s_te, y[te])[0]),
            "weights": {f: round(float(x), 3) for f, x in zip(usable, w)},
        })
    print("\n=== 4-fold blend (cross-validated by job) ===")
    for f in folds:
        print(f"  rho={f['rho']:.3f} weights={f['weights']}")
    print(f"  mean rho = {np.mean([f['rho'] for f in folds]):.3f}")

    Path(a.out).write_text(json.dumps({"rows": rows, "folds": folds, "usable": usable}, indent=1))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
