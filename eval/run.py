"""Fit weights on the train JDs, evaluate on held-out JDs, run ablations, write eval/REPORT.md (+ optionally data/model_params.json).

Usage: python eval/run.py [--model NAME] [--no-write-params] [--hashing]
Ground truth = mean of the two blind annotators (eval/labels/A, eval/labels/B). See SPEC section 8.
"""
import argparse
import glob
import json
import time
from datetime import date
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import cohen_kappa_score, ndcg_score, roc_auc_score

from app.config import Settings
from app.embed import HashingEmbedder, OnnxEmbedder
from app.nlp import skills as skills_mod
from app.scoring import scorer
from app.scoring.scorer import Params
from app.scoring.specs import build_cv_spec, build_job_spec

HERE = Path(__file__).parent
TODAY = date(2026, 11, 2)
KEYS = ["skills", "semantic", "experience", "education"]
PRIOR = np.array([0.35, 0.40, 0.15, 0.10])


def load_labels():
    out = {}
    for who in "AB":
        d = {}
        for f in sorted(glob.glob(str(HERE / "labels" / who / "batch*.json"))):
            d.update({k: v["score"] for k, v in json.loads(Path(f).read_text()).items()})
        out[who] = d
    return out


def build_specs(pairs, jobs, negation=True):
    skills_mod.NEGATION_ENABLED = negation
    cvs = {}
    for p in pairs:
        if p["cv_id"] not in cvs:
            cvs[p["cv_id"]] = build_cv_spec((HERE / "cache" / "cvs" / f"{p['cv_id']}.txt").read_text(encoding="utf-8"), TODAY)
    jb = {j["slug"]: build_job_spec(j) for j in jobs}
    skills_mod.NEGATION_ENABLED = True
    return cvs, jb


def components(pairs, cvs, jb, emb, params):
    rows = []
    for p in pairs:
        r = scorer.score(cvs[p["cv_id"]], jb[p["job_slug"]], emb, params)
        rows.append({"comp": r["components"], "raw": r["raw_semantic"], "overall": r["overall_score"], "notes": r["notes"], "pair": r["lang_pair"]})
    return rows


def overall_from(comp, w):
    avail = {k: v for k, v in comp.items() if v is not None}
    ws = sum(w[k] for k in avail)
    return 100 * sum(w[k] * v for k, v in avail.items()) / ws if ws else 0.0


def metrics(scores, y, jobs_of, thr=3.0):
    s, y = np.asarray(scores, float), np.asarray(y, float)
    rho = spearmanr(s, y)[0]
    out = {"n": len(y), "spearman": float(rho), "kendall": float(kendalltau(s, y)[0]), "pearson": float(pearsonr(s, y)[0])}
    out["auc_relevant"] = float(roc_auc_score(y >= thr, s)) if 0 < (y >= thr).sum() < len(y) else float("nan")
    nd = []
    for j in set(jobs_of):
        idx = [i for i, jj in enumerate(jobs_of) if jj == j]
        if len(idx) > 1 and y[idx].max() > 0:
            nd.append(ndcg_score([y[idx]], [s[idx]], k=5))
    out["ndcg5"] = float(np.mean(nd))
    return out


def bootstrap_ci(scores, y, jobs_of, n=2000, seed=7):
    rng = np.random.default_rng(seed)
    js = sorted(set(jobs_of))
    idx_by = {j: [i for i, jj in enumerate(jobs_of) if jj == j] for j in js}
    s, y = np.asarray(scores, float), np.asarray(y, float)
    vals = []
    for _ in range(n):
        pick = rng.choice(js, len(js))
        idx = [i for j in pick for i in idx_by[j]]
        if np.std(s[idx]) > 0 and np.std(y[idx]) > 0:
            vals.append(spearmanr(s[idx], y[idx])[0])
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def fit_weights(train_rows, y, lam=0.05):
    comps = [r["comp"] for r in train_rows]
    y = np.asarray(y, float)

    def obj(w):
        w = np.abs(w) / np.abs(w).sum()
        s = np.array([overall_from(c, dict(zip(KEYS, w))) for c in comps])
        if np.std(s) == 0:
            return 1.0
        return (1 - pearsonr(s, y)[0]) + lam * np.sum((w - PRIOR) ** 2)

    best = None
    for start in (PRIOR, np.array([0.25, 0.25, 0.25, 0.25]), np.array([0.5, 0.4, 0.05, 0.05]), np.array([0.3, 0.6, 0.05, 0.05])):
        res = minimize(obj, start, method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-5, "maxiter": 600})
        if best is None or res.fun < best.fun:
            best = res
    w = np.abs(best.x) / np.abs(best.x).sum()
    return dict(zip(KEYS, [round(float(v), 3) for v in w]))


def fit_calibration(raw, y, fallback):
    """lo = median raw cosine of weak/unrelated pairs (label <= 1); hi = 90th percentile of good/strong pairs (label >= 3)."""
    raw, y = np.asarray(raw), np.asarray(y)
    if (y <= 1).sum() < 8 or (y >= 3).sum() < 5:
        return fallback
    lo = float(np.percentile(raw[y <= 1], 50))
    hi = float(np.percentile(raw[y >= 3], 90))
    return round(lo, 3), round(max(hi, lo + 0.1), 3)


def fit_calibrations(rows, y):
    out = {}
    for pair, fb in (("same", (0.25, 0.65)), ("cross", (0.2, 0.55))):
        idx = [i for i, r in enumerate(rows) if r["pair"] == pair]
        out[pair] = fit_calibration([rows[i]["raw"] for i in idx], [y[i] for i in idx], fb)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=Settings().embedding_model)
    ap.add_argument("--hashing", action="store_true")
    ap.add_argument("--no-write-params", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    jobs = json.loads((HERE / "jobs.json").read_text())
    pairs = json.loads((HERE / "pairs.json").read_text())
    L = load_labels()
    pairs = [p for p in pairs if p["pair_id"] in L["A"] and p["pair_id"] in L["B"]]
    y = {p["pair_id"]: (L["A"][p["pair_id"]] + L["B"][p["pair_id"]]) / 2 for p in pairs}
    tr = [p for p in pairs if p["split"] == "train"]
    te = [p for p in pairs if p["split"] == "test"]
    emb = HashingEmbedder() if a.hashing else OnnxEmbedder(a.model, Settings().embedding_cache_dir, cache_size=40000)
    cvs, jb = build_specs(pairs, jobs)

    base = Params(dict(zip(KEYS, PRIOR)), {"same": (0.25, 0.65), "cross": (0.2, 0.55)}, 0.3, "eval")
    t0 = time.time()
    rows_tr0 = components(tr, cvs, jb, emb, base)
    print(f"features computed in {time.time() - t0:.0f}s")
    calib = fit_calibrations(rows_tr0, [y[p["pair_id"]] for p in tr])
    cal = Params(dict(zip(KEYS, PRIOR)), calib, 0.3, "eval")
    rows_tr = components(tr, cvs, jb, emb, cal)
    w = fit_weights(rows_tr, [y[p["pair_id"]] for p in tr])
    fitted = Params(w, calib, 0.3, f"fitted-{date.today().isoformat()}")

    def evaluate(split, params, label_fn=lambda p: y[p["pair_id"]], cv_specs=cvs):
        rows = components(split, cv_specs, jb, emb, params)
        return rows, [r["overall"] for r in rows]

    rep = {"model": emb.name, "n_pairs": len(pairs), "n_train": len(tr), "n_test": len(te), "weights": w, "calibration": {k: list(v) for k, v in calib.items()}}
    # inter-annotator agreement (all pairs)
    ya = [L["A"][p["pair_id"]] for p in pairs]
    yb = [L["B"][p["pair_id"]] for p in pairs]
    rep["agreement"] = {"spearman": float(spearmanr(ya, yb)[0]), "weighted_kappa": float(cohen_kappa_score(ya, yb, weights="quadratic")),
                        "exact": float(np.mean(np.array(ya) == np.array(yb))), "within1": float(np.mean(np.abs(np.array(ya) - np.array(yb)) <= 1))}
    yte = [y[p["pair_id"]] for p in te]
    jte = [p["job_slug"] for p in te]
    ytr = [y[p["pair_id"]] for p in tr]

    variants = {}
    rows_te, s_full = evaluate(te, fitted)
    _, s_full_tr = evaluate(tr, fitted)
    variants["hybrid (fitted)"] = s_full
    variants["hybrid (prior weights, uncalibrated defaults)"] = [r["overall"] for r in components(te, cvs, jb, emb, base)]
    variants["skills only"] = [100 * (r["comp"]["skills"] or 0.0) for r in rows_te]
    variants["embedding only"] = [100 * r["comp"]["semantic"] for r in rows_te]
    variants["embedding only (raw cosine)"] = [r["raw"] for r in rows_te]
    # TF-IDF baseline
    texts = {p["cv_id"]: (HERE / "cache" / "cvs" / f"{p['cv_id']}.txt").read_text() for p in pairs}
    jtext = {j["slug"]: f"{j['title']} {j['requirements']} {j['description']}" for j in jobs}
    vec = TfidfVectorizer(sublinear_tf=True, stop_words="english", min_df=2).fit(list(texts.values()) + list(jtext.values()))
    variants["TF-IDF cosine"] = [float((vec.transform([jtext[p["job_slug"]]]) @ vec.transform([texts[p["cv_id"]]]).T).toarray()[0, 0]) for p in te]
    # ablation: no negation handling
    cvs_nn, jb_nn = build_specs(pairs, jobs, negation=False)
    jb_saved = dict(jb)
    jb.update(jb_nn)
    variants["hybrid without negation handling"] = evaluate(te, fitted, cv_specs=cvs_nn)[1]
    jb.update(jb_saved)
    rep["results"] = {}
    for name, sc in variants.items():
        m = metrics(sc, yte, jte)
        m["spearman_ci95"] = bootstrap_ci(sc, yte, jte)
        rep["results"][name] = m
    rep["train_spearman_fitted"] = float(spearmanr(s_full_tr, ytr)[0])
    # per-annotator view of the headline model
    rep["vs_annotator"] = {w_: float(spearmanr(s_full, [L[w_][p["pair_id"]] for p in te])[0]) for w_ in "AB"}
    # score distribution by consensus label (test)
    by = {}
    for sc, yy in zip(s_full, yte):
        by.setdefault(round(yy), []).append(sc)
    rep["mean_score_by_label"] = {k: round(float(np.mean(v)), 1) for k, v in sorted(by.items())}
    # per-JD held-out
    rep["per_job_spearman"] = {}
    for j in sorted(set(jte)):
        idx = [i for i, jj in enumerate(jte) if jj == j]
        rep["per_job_spearman"][j] = round(float(spearmanr(np.array(s_full)[idx], np.array(yte)[idx])[0]), 2)

    tag = a.tag or emb.name.split("/")[-1]
    (HERE / f"results_{tag}.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ("model", "weights", "calibration", "agreement", "train_spearman_fitted", "vs_annotator", "mean_score_by_label")}, indent=1))
    for k, v in rep["results"].items():
        print(f"{k:50s} rho={v['spearman']:.3f} CI={v['spearman_ci95'][0]:.2f}..{v['spearman_ci95'][1]:.2f} auc={v['auc_relevant']:.3f} ndcg5={v['ndcg5']:.3f}")
    if not a.no_write_params:
        params_path = Path("data/model_params.json")
        label_map = {}
        if params_path.exists():
            # keep the score map of the shipped (grid) fit: it is fitted by eval/fit_grid.py
            label_map = json.loads(params_path.read_text(encoding="utf-8")).get("label_map") or {}
        params_path.write_text(json.dumps({
            "version": fitted.version, "weights": w, "calibration": {k: {"lo": v[0], "hi": v[1]} for k, v in calib.items()}, "title_weight": 0.3,
            **({"label_map": label_map} if label_map else {}),
            "fitted_on": {"model": emb.name, "train_pairs": len(tr), "train_jobs": len({p["job_slug"] for p in tr}), "test_spearman": rep["results"]["hybrid (fitted)"]["spearman"]},
        }, indent=1))
        print("wrote data/model_params.json (240-pair fit - run eval/fit_grid.py for the shipped grid fit)")


if __name__ == "__main__":
    main()
