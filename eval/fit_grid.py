"""Fit the scorer on the full 5 760-pair label grid and write data/model_params.json.

Inputs (all local, nothing is committed except the resulting parameter file):
- eval/jobs.json          24 postings (16 train / 8 held-out by job)
- eval/cache/cvs/*.txt    240 OCR CV texts (git-ignored, never committed)
- grid labels             0-4 rubric labels for every (job, CV) pair - default source is the
                          local `enfue-eval` dataset (Haiku labels for all 5 760 pairs, Sonnet
                          cross-check on 292 same-domain pairs)
- eval/labels/A|B         240 pairs annotated by two blind annotators (the repo's own labels)

The script fits on the *train* jobs only and reports every number on the held-out jobs.
It also fits a monotone (isotonic) map from the raw 0-100 score to the expected rubric label,
so the score that recruiters see is calibrated to the 0-4 scale instead of being compressed.

Usage:
  python eval/fit_grid.py                       # fit + write data/model_params.json
  python eval/fit_grid.py --no-write-params    # report only
  python eval/fit_grid.py --model ... --rescore # force a re-score (the score cache is reused otherwise)
"""
import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import Settings
from app.embed import HashingEmbedder, OnnxEmbedder
from app.nlp import education
from app.scoring import scorer
from app.scoring.specs import build_cv_spec, build_job_spec

HERE = Path(__file__).parent
DEFAULT_LABELS = os.path.expanduser("~/enfue-eval/dataset/pairs_labels.jsonl")
DEFAULT_SONNET = os.path.expanduser("~/enfue-eval/dataset/sonnet-labels.json")
CACHE = HERE / "results_scored_grid.json"
KEYS = ["skills", "semantic", "experience", "education"]
# reference scale: rubric label -> expected 0-100 score (the scale used by the enfue worker,
# fitted on an independent label set - used here only to report calibration error)
REFERENCE = {0: 5.0, 1: 25.0, 2: 45.0, 3: 70.0, 4: 85.0}
# bump when the cached row schema changes (the score cache stores the rows of score_grid)
ROW_SCHEMA = 2


def code_fingerprint():
    h = hashlib.sha256()
    for p in sorted((ROOT / "app").rglob("*.py")) + [ROOT / "data" / "skills.yaml"]:
        h.update(p.read_bytes())
    h.update(f"schema{ROW_SCHEMA}".encode())
    return h.hexdigest()[:16]


def load_labels(path):
    """Accepts the local dataset (JSONL) or the copy exported into eval/labels/grid (JSON)."""
    out = {}
    if str(path).endswith(".json"):
        blob = json.loads(Path(path).read_text(encoding="utf-8"))
        labels = blob.get("labels", blob)
        ind_file = HERE / "labels" / "grid" / "in_domain_768.json"
        in_domain = set(json.loads(ind_file.read_text(encoding="utf-8"))) if ind_file.exists() else set()
        for pair, score in labels.items():
            job, cv = pair.split("__", 1)
            out[pair] = {"label": int(score), "job": job, "cv": cv, "in_domain": pair in in_domain}
        return out
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            out[r["pair"]] = {"label": int(r["haiku_label"]), "job": r["job"], "cv": r["cv"], "in_domain": bool(r["in_domain"])}
    return out


def load_sonnet(path):
    if not path or not Path(path).exists():
        return {}
    return {k: int(v["score"]) for k, v in json.loads(Path(path).read_text(encoding="utf-8")).items()}


def load_ab_labels():
    """The repo's own labels use pair ids like 'python-backend__00'; the grid uses 'python-backend__<cv id>'.

    eval/pairs.json maps the former to (job, cv), so the two label sets can be joined."""
    out = {}
    for who in "AB":
        d = {}
        for f in sorted((HERE / "labels" / who).glob("batch*.json")):
            d.update({k: int(v["score"]) for k, v in json.loads(f.read_text(encoding="utf-8")).items()})
        out[who] = d
    pairs = json.loads((HERE / "pairs.json").read_text(encoding="utf-8"))
    by_id = {p["pair_id"]: p for p in pairs}
    joined = {"A": {}, "B": {}}
    for who in "AB":
        for pid, score in out[who].items():
            p = by_id.get(pid)
            if p:
                joined[who][f"{p['job_slug']}__{p['cv_id']}"] = score
    return {"raw": out, "joined": joined}


def score_grid(labels, model, hashing, rescore=False):
    jobs = json.loads((HERE / "jobs.json").read_text(encoding="utf-8"))
    fp = code_fingerprint()
    if CACHE.exists() and not rescore:
        cached = json.loads(CACHE.read_text())
        if cached.get("fingerprint") == fp and cached.get("model") == model and cached.get("hashing") == hashing:
            print(f"reusing score cache ({CACHE.name})")
            return cached["rows"]
    emb = HashingEmbedder() if hashing else OnnxEmbedder(model, Settings().embedding_cache_dir, cache_size=60000)
    t0 = time.time()
    cvs = {}
    for cid in sorted({v["cv"] for v in labels.values()}):
        cvs[cid] = build_cv_spec((HERE / "cache" / "cvs" / f"{cid}.txt").read_text(encoding="utf-8"))
    jb = {j["slug"]: build_job_spec(j) for j in jobs}
    print(f"specs built in {time.time() - t0:.0f}s")
    params = scorer.load_params()
    rows = []
    t0 = time.time()
    for i, (pair, v) in enumerate(labels.items()):
        r = scorer.score(cvs[v["cv"]], jb[v["job"]], emb, params)
        cv, js = cvs[v["cv"]], jb[v["job"]]
        # the qualification signals are multiplied by "relevance", which depends on the embedding
        # model, so they are stored ungated and re-gated whenever the cosine window is refitted
        exp_raw = min(cv.years / js.required_years, 1.0) if js.required_years else None
        edu_raw = None
        if js.required_level != "none":
            gap = education.rank(cv.level) - education.rank(js.required_level)
            edu_raw = 1.0 if gap >= 0 else 0.5 if gap == -1 else 0.0
        rows.append({"pair": pair, "job": v["job"], "cv": v["cv"], "in_domain": v["in_domain"],
                     "skills": r["components"]["skills"], "raw": r["raw_semantic"], "lang": r["lang_pair"],
                     "exp_raw": exp_raw, "edu_raw": edu_raw, "caps": r["notes"],
                     "score_shipped": float(r["overall_score"])})
        if i and i % 2000 == 0:
            print(f"  scored {i}/{len(labels)} ({time.time() - t0:.0f}s)")
    print(f"scored {len(rows)} pairs in {time.time() - t0:.0f}s")
    CACHE.write_text(json.dumps({"fingerprint": fp, "model": model, "hashing": hashing, "rows": rows}, indent=1))
    return rows


def fit_cosine_calibration(rows, labels):
    """Per language pair: lo = median raw cosine of weak/unrelated pairs, hi = p90 of good/strong pairs."""
    out = {}
    for lang, fb in (("same", (0.25, 0.65)), ("cross", (0.2, 0.55))):
        idx = [i for i, r in enumerate(rows) if r["lang"] == lang]
        raw = np.asarray([rows[i]["raw"] for i in idx], float)
        y = np.asarray([labels[rows[i]["pair"]] for i in idx], float)
        if (y <= 1).sum() < 8 or (y >= 3).sum() < 5:
            out[lang] = fb
            continue
        lo = float(np.percentile(raw[y <= 1], 50))
        hi = float(np.percentile(raw[y >= 3], 90))
        out[lang] = (round(lo, 3), round(max(hi, lo + 0.05), 3))
    return out


def components_for(rows, cal):
    """Recompute the components for a given cosine window (the window is model specific)."""
    out = []
    for r in rows:
        lo, hi = cal[r["lang"]]
        sem = float(np.clip((r["raw"] - lo) / max(hi - lo, 1e-6), 0.0, 1.0))
        relevance = max(sem, r["skills"] or 0.0)
        comps = {"skills": r["skills"], "semantic": sem,
                 "experience": (r["exp_raw"] * relevance) if r["exp_raw"] is not None else None,
                 "education": (r["edu_raw"] * relevance) if r["edu_raw"] is not None else None}
        out.append({"pair": r["pair"], "job": r["job"], "components": comps, "relevance": relevance})
    return out


def blended(rows, w):
    out = []
    for r in rows:
        avail = {k: v for k, v in r["components"].items() if v is not None}
        ws = sum(w[k] for k in avail)
        out.append(100 * sum(w[k] * avail[k] for k in avail) / ws if ws else 0.0)
    return np.asarray(out)


def fit_weights(rows, y, keys=KEYS, seed=None):
    """Maximise Spearman of the weighted blend against the labels (ranking is what the score is used for)."""
    m = np.full((len(rows), len(keys)), np.nan)
    for i, r in enumerate(rows):
        for j, k in enumerate(keys):
            v = r["components"].get(k)
            if v is not None:
                m[i, j] = v
    ok = ~np.isnan(m)
    y = np.asarray([y[r["pair"]] for r in rows] if isinstance(y, dict) else list(y), float)
    prior = np.array([0.35, 0.40, 0.15, 0.10])

    def obj(w):
        w = np.abs(w)
        if w.sum() == 0:
            return 1.0
        w = w / w.sum()
        wm = np.where(ok, w, 0.0)
        den = wm.sum(axis=1)
        s = np.where(den > 0, 100 * np.nansum(np.nan_to_num(m) * wm, axis=1) / np.maximum(den, 1e-9), 0.0)
        if np.std(s) == 0:
            return 1.0
        return -spearmanr(s, y)[0] + 0.02 * float(np.sum((w - prior) ** 2))

    starts = [prior, np.ones(len(keys)) / len(keys), np.array([0.5, 0.4, 0.05, 0.05]), np.array([0.3, 0.6, 0.05, 0.05])]
    if seed is not None:
        starts.insert(0, np.asarray(seed, float))
    best = None
    for st in starts:
        res = minimize(obj, np.asarray(st, float), method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-5, "maxiter": 800})
        if best is None or res.fun < best.fun:
            best = res
    w = np.abs(best.x)
    w = w / w.sum()
    return dict(zip(keys, [round(float(v), 3) for v in w]))


def eval_scores(rows, scores, labels, subset=None):
    idx = [i for i, r in enumerate(rows) if (subset is None or subset(r)) and r["pair"] in labels]
    if not idx:
        return {}
    s = np.asarray([scores[i] for i in idx], float)
    y = np.asarray([labels[r["pair"]] for i, r in enumerate(rows) if i in set(idx)], float)
    by_label = defaultdict(list)
    for si, yi in zip(s, y):
        by_label[yi].append(si)
    ref = np.asarray([REFERENCE[int(v)] for v in y], float)
    out = {"n": len(idx), "spearman": round(float(spearmanr(s, y)[0]), 4),
           "mae_vs_reference_scale": round(float(np.mean(np.abs(s - ref))), 2),
           "mean_score_by_label": {str(int(k)): round(float(np.mean(v)), 1) for k, v in sorted(by_label.items())}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=DEFAULT_LABELS)
    ap.add_argument("--sonnet", default=DEFAULT_SONNET)
    ap.add_argument("--model", default=Settings().embedding_model)
    ap.add_argument("--hashing", action="store_true")
    ap.add_argument("--rescore", action="store_true")
    ap.add_argument("--no-write-params", action="store_true")
    ap.add_argument("--keep-cosine-calibration", action="store_true",
                    help="use the cosine window already in data/model_params.json (e.g. fitted by eval/run.py)")
    ap.add_argument("--cosine-source", default="grid", choices=["grid", "ab"],
                    help="which labels fit the cosine window (grid = 5760 Haiku labels, ab = the repo's A/B labels)")
    ap.add_argument("--weights", default="auto", choices=["auto", "shipped", "grid", "ab", "grid+ab"],
                    help="which weight set to ship (auto = the best of the candidates on the held-out jobs)")
    a = ap.parse_args()

    grid = load_labels(a.labels)
    sonnet = load_sonnet(a.sonnet)
    ab_all = load_ab_labels()
    ab = ab_all["joined"]
    ab_cons = {p: (ab["A"][p] + ab["B"][p]) / 2 for p in ab["A"] if p in ab["B"]}
    jobs = json.loads((HERE / "jobs.json").read_text(encoding="utf-8"))
    train_jobs = {j["slug"] for j in jobs if j.get("_split") == "train"}
    test_jobs = {j["slug"] for j in jobs if j.get("_split") != "train"}
    print(f"grid {len(grid)} pairs | train jobs {len(train_jobs)} | held-out jobs {len(test_jobs)} | A/B labels {len(ab_cons)} | sonnet {len(sonnet)}")

    rows = score_grid(grid, a.model, a.hashing, a.rescore)
    hy = {p: v["label"] for p, v in grid.items()}
    shipped = scorer.load_params()
    # the cosine window is model specific: refit it on the train jobs before anything else
    raw_tr = [r for r in rows if r["job"] in train_jobs]
    if a.keep_cosine_calibration:
        cal = dict(shipped.calibration)
    elif a.cosine_source == "ab":
        ab_tr_raw = [r for r in raw_tr if r["pair"] in ab_cons]
        cal = fit_cosine_calibration(ab_tr_raw, ab_cons)
    else:
        cal = fit_cosine_calibration(raw_tr, hy)
    comp_rows = components_for(rows, cal)
    tr = [r for r in comp_rows if r["job"] in train_jobs]
    te = [r for r in comp_rows if r["job"] in test_jobs]

    rep = {"model": a.model, "shipped_weights": shipped.weights,
           "cosine_calibration": {k: list(v) for k, v in cal.items()}}

    # --- weight fits: grid labels (train jobs) vs the repo's own A/B labels (train jobs)
    w_grid = fit_weights(tr, hy, seed=list(shipped.weights.values()))
    ab_train = [r for r in tr if r["pair"] in ab_cons]
    w_ab = fit_weights(ab_train, ab_cons) if len(ab_train) > 30 else None
    w_both = None
    if w_ab:
        merged = dict(hy)
        merged.update({p: v for p, v in ab_cons.items()})
        w_both = fit_weights(tr, merged, seed=list(shipped.weights.values()))
    rep["weights"] = {"grid": w_grid, "ab": w_ab, "grid+ab": w_both}

    candidates = {"shipped": shipped.weights, "grid": w_grid}
    if w_ab:
        candidates["ab"] = w_ab
        candidates["grid+ab"] = w_both
    # --- every candidate is evaluated on the held-out jobs, against three label sets
    rep["heldout"] = {}
    for name, w in candidates.items():
        s_tr = blended(tr, w)
        s_te = blended(te, w)
        rep["heldout"][name] = {
            "vs_grid_labels": eval_scores(te, s_te, hy),
            "vs_AB_consensus": eval_scores(te, s_te, ab_cons),
            "vs_sonnet": eval_scores(te, s_te, sonnet),
            "train_vs_grid": eval_scores(tr, s_tr, hy),
        }

    # --- calibration: isotonic map raw score -> expected rubric label.
    # Three training sources are compared; the map must be calibrated for the score a recruiter
    # sees (a pool of applicants, not a grid that is 87 % out-of-domain), so the A/B pairs
    # (stratified relevant/adjacent/unrelated) are the reference set.
    # `--weights` pins the weight set: "auto" picks the best candidate on the held-out jobs
    # (which is a mild optimistic bias), "grid" uses the fit on the 5 760 Haiku labels only.
    if a.weights == "auto":
        best_name = max(rep["heldout"], key=lambda k: rep["heldout"][k]["vs_grid_labels"]["spearman"])
    else:
        best_name = a.weights
    w = candidates[best_name]
    s_tr = blended(tr, w)
    s_te = blended(te, w)
    iso_grid = IsotonicRegression(y_min=0.0, y_max=4.0, out_of_bounds="clip")
    iso_grid.fit(s_tr, np.asarray([hy[r["pair"]] for r in tr], float))

    ab_tr = [r for r in tr if r["pair"] in ab_cons]
    ab_te = [r for r in te if r["pair"] in ab_cons]
    iso_ab = IsotonicRegression(y_min=0.0, y_max=4.0, out_of_bounds="clip")
    iso_ab.fit(blended(ab_tr, w), np.asarray([ab_cons[r["pair"]] for r in ab_tr], float))

    merged_tr = dict(hy)
    merged_tr.update(ab_cons)
    iso_both = IsotonicRegression(y_min=0.0, y_max=4.0, out_of_bounds="clip")
    iso_both.fit(s_tr, np.asarray([merged_tr[r["pair"]] for r in tr], float))

    def fit_linear(rows_, scores_, labels_):
        """Least-squares linear map from the raw score to the reference 0-100 scale."""
        y = np.asarray([REFERENCE[int(labels_[r["pair"]])] for r in rows_], float)
        a, b = np.polyfit(np.asarray(scores_, float), y, 1)
        return (float(a), float(b))

    lin_grid = fit_linear(tr, s_tr, hy)
    lin_ab = fit_linear(ab_tr, blended(ab_tr, w), ab_cons) if ab_tr else lin_grid

    def apply(rows_, scores_, iso_):
        return np.asarray([100 * float(iso_.predict([x])[0]) / 4.0 for x in scores_])

    def apply_linear(scores_, lin_):
        a, b = lin_
        return np.clip(a * np.asarray(scores_, float) + b, 0, 100)

    cal_candidates = {"isotonic_grid": iso_grid, "isotonic_ab": iso_ab, "isotonic_both": iso_both}
    linear_candidates = {"linear_grid": lin_grid, "linear_ab": lin_ab}
    cal_rep = {}
    for name, iso_ in cal_candidates.items():
        cal_rep[name] = {
            "grid_test": eval_scores(te, apply(te, s_te, iso_), hy),
            "AB_test": eval_scores(ab_te, apply(ab_te, blended(ab_te, w), iso_), ab_cons),
            "sonnet": eval_scores([r for r in te if r["pair"] in sonnet],
                                  apply([r for r in te if r["pair"] in sonnet],
                                        blended([r for r in te if r["pair"] in sonnet], w), iso_), sonnet),
        }
    for name, lin_ in linear_candidates.items():
        cal_rep[name] = {
            "grid_test": eval_scores(te, apply_linear(s_te, lin_), hy),
            "AB_test": eval_scores(ab_te, apply_linear(blended(ab_te, w), lin_), ab_cons),
            "sonnet": eval_scores([r for r in te if r["pair"] in sonnet],
                                  apply_linear(blended([r for r in te if r["pair"] in sonnet], w), lin_), sonnet),
            "map": [round(x, 4) for x in lin_],
        }
    rep["calibration_candidates"] = cal_rep
    rep["calibration"] = {
        "source": best_name,
        "uncalibrated": {"grid_test": eval_scores(te, s_te, hy), "AB_test": eval_scores(ab_te, blended(ab_te, w), ab_cons)},
    }
    cal_source = min(cal_rep, key=lambda k: cal_rep[k]["grid_test"]["mae_vs_reference_scale"] + cal_rep[k]["AB_test"]["mae_vs_reference_scale"])
    lin_ = linear_candidates.get(cal_source)
    iso = cal_candidates.get(cal_source)
    rep["calibration"]["chosen"] = cal_source
    if cal_source.startswith("linear"):
        lin_ = linear_candidates[cal_source]
        rep["calibration"]["map"] = [round(x, 4) for x in lin_]
    else:
        iso = cal_candidates[cal_source]
        rep["calibration"]["points"] = [[round(float(x), 1), round(float(y), 3)] for x, y in zip(iso.X_thresholds_, iso.y_thresholds_)]
    rep["calibration"]["grid_test"] = cal_rep[cal_source]["grid_test"]
    rep["calibration"]["AB_test"] = cal_rep[cal_source]["AB_test"]

    # --- write the parameters
    if not a.no_write_params:
        if cal_source.startswith("linear"):
            label_map = {"type": "linear", "a": round(lin_[0], 5), "b": round(lin_[1], 4)}
        else:
            label_map = {"type": "isotonic",
                         "knots": [[round(float(x), 1), round(float(y) * 25.0, 2)]
                                   for x, y in zip(iso.X_thresholds_, iso.y_thresholds_)]}
        out = {
            "version": f"grid-{a.model.split('/')[-1][:24]}-{len(grid)}pairs",
            "weights": w,
            "calibration": {k: {"lo": v[0], "hi": v[1]} for k, v in cal.items()},
            "label_map": label_map,
            "title_weight": shipped.title_weight,
            "fitted_on": {"model": a.model, "grid_pairs": len(grid), "train_jobs": sorted(train_jobs),
                          "calibration_source": cal_source,
                          "heldout_spearman_vs_grid": rep["heldout"][best_name]["vs_grid_labels"]["spearman"],
                          "heldout_spearman_vs_AB": rep["heldout"][best_name]["vs_AB_consensus"]["spearman"],
                          "heldout_spearman_vs_sonnet": rep["heldout"][best_name]["vs_sonnet"]["spearman"],
                          "heldout_mae_vs_reference_scale": rep["calibration"]["grid_test"]["mae_vs_reference_scale"]},
        }
        (ROOT / "data" / "model_params.json").write_text(json.dumps(out, indent=1))
        print("\nwrote data/model_params.json")
        print(json.dumps(out["fitted_on"], indent=1))

    (HERE / "results_fit_grid.json").write_text(json.dumps(rep, indent=1, default=float))
    print("\n=== held-out jobs (8) ===")
    print(json.dumps(rep["heldout"], indent=1, default=float))
    print("\n=== calibration ===")
    print(json.dumps(rep["calibration"], indent=1, default=float))


if __name__ == "__main__":
    main()
