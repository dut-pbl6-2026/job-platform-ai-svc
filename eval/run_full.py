"""Benchmark the local scorer on the full label grid: 24 jobs x 240 CVs = 5 760 pairs.

The grid labels come from the local `enfue-eval` dataset (a different project, same CVs /
jobs / rubric): Haiku labels for every pair, plus an independent Sonnet cross-check on 292
same-domain pairs. Labels are rubric judgements (0-4); CV text stays local and is never
committed (it is read from eval/cache/cvs, which is git-ignored).

Usage:
  python eval/run_full.py                       # score the grid with the shipped params
  python eval/run_full.py --cv-folds 4          # cross-validate by job (fit on train jobs)
  python eval/run_full.py --fit                 # fit on train jobs and write data/model_params.json
  python eval/run_full.py --errors              # per-job worst pairs + taxonomy coverage

Never tune on the held-out jobs: `--fit` fits on the train jobs of eval/jobs.json only.
"""
import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import Settings
from app.embed import HashingEmbedder, OnnxEmbedder
from app.scoring import scorer
from app.scoring.specs import build_cv_spec, build_job_spec

HERE = Path(__file__).parent
TODAY = None  # let the specs use "today" = now
DEFAULT_LABELS = os.path.expanduser("~/enfue-eval/dataset/pairs_labels.jsonl")
DEFAULT_SONNET = os.path.expanduser("~/enfue-eval/dataset/sonnet-labels.json")
KEYS = ["skills", "semantic", "experience", "education"]
PRIOR = np.array([0.35, 0.40, 0.15, 0.10])


# ---------------------------------------------------------------- data loading
def load_grid_labels(path):
    """{pair_id: {"label": int, "in_domain": bool, "job": str, "cv": str}}

    Accepts the local dataset (JSONL) or the copy exported into eval/labels/grid (JSON)."""
    out = {}
    if str(path).endswith(".json"):
        blob = json.loads(Path(path).read_text(encoding="utf-8"))
        labels = blob.get("labels", blob)
        ind_file = HERE / "labels" / "grid" / "in_domain_768.json"
        in_domain = set(json.loads(ind_file.read_text(encoding="utf-8"))) if ind_file.exists() else set()
        for pair, score in labels.items():
            job, cv = pair.split("__", 1)
            out[pair] = {"label": int(score), "in_domain": pair in in_domain, "job": job, "cv": cv, "original": None}
        return out
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            out[r["pair"]] = {"label": int(r["haiku_label"]), "in_domain": bool(r["in_domain"]),
                              "job": r["job"], "cv": r["cv"], "original": r.get("original")}
    return out


def load_sonnet(path):
    if not path or not Path(path).exists():
        return {}
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    labels = d.get("labels", d)
    return {k: int(v["score"] if isinstance(v, dict) else v) for k, v in labels.items()}


def load_original_labels():
    """The 240 pairs annotated by two blind annotators (A/B) in eval/labels/."""
    out = {}
    for who in "AB":
        d = {}
        for f in sorted((HERE / "labels" / who).glob("batch*.json")):
            d.update({k: int(v["score"]) for k, v in json.loads(f.read_text(encoding="utf-8")).items()})
        out[who] = d
    return out


def build_specs(jobs, cv_ids):
    cvs = {}
    for cid in cv_ids:
        p = HERE / "cache" / "cvs" / f"{cid}.txt"
        if p.exists():
            cvs[cid] = build_cv_spec(p.read_text(encoding="utf-8"), TODAY)
    return cvs, {j["slug"]: build_job_spec(j) for j in jobs}


# ---------------------------------------------------------------- scoring
def score_grid(pairs, cvs, jb, emb, params):
    rows = []
    for pair, job, cv in pairs:
        r = scorer.score(cvs[cv], jb[job], emb, params)
        rows.append({
            "pair": pair, "job": job, "cv": cv, "score": float(r["overall_score"]),
            "components": r["components"], "raw": r["raw_semantic"], "lang": r["lang_pair"],
            "caps": r["notes"], "relevance": r["relevance"],
        })
    return rows


# ---------------------------------------------------------------- metrics
def ndcg_at_k(rows, labels, k=5):
    by_job = defaultdict(list)
    for r in rows:
        by_job[r["job"]].append(r)
    vals = []
    for rs in by_job.values():
        ranked = sorted(rs, key=lambda r: -r["score"])[:k]
        ideal = sorted(rs, key=lambda r: -labels[r["pair"]])[:k]
        dcg = sum(labels[r["pair"]] / np.log2(i + 2) for i, r in enumerate(ranked))
        idcg = sum(labels[r["pair"]] / np.log2(i + 2) for i, r in enumerate(ideal))
        if idcg > 0:
            vals.append(dcg / idcg)
    return float(np.mean(vals)) if vals else float("nan")


def auc(scores, labels, min_label):
    s, y = np.asarray(scores, float), np.asarray(labels, float)
    pos, neg = s[y >= min_label], s[y < min_label]
    if not len(pos) or not len(neg):
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def metrics(rows, labels):
    if not rows:
        return {}
    s = [r["score"] for r in rows]
    y = [labels[r["pair"]] for r in rows]
    out = {"n": len(rows), "spearman": float(spearmanr(s, y)[0]),
           "auc_ge2": auc(s, y, 2), "auc_ge3": auc(s, y, 3), "ndcg5": ndcg_at_k(rows, labels)}
    by_label = defaultdict(list)
    for r, yy in zip(rows, y):
        by_label[yy].append(r["score"])
    out["mean_score_by_label"] = {str(k): round(float(np.mean(v)), 1) for k, v in sorted(by_label.items())}
    return out


def bootstrap_ci(rows, labels, n=1000, seed=7):
    rng = np.random.default_rng(seed)
    by_job = defaultdict(list)
    for r in rows:
        by_job[r["job"]].append(r)
    jobs = sorted(by_job)
    vals = []
    for _ in range(n):
        idx = [r for j in rng.choice(jobs, len(jobs), replace=True) for r in by_job[j]]
        s = [r["score"] for r in idx]
        y = [labels[r["pair"]] for r in idx]
        if np.std(s) > 0 and np.std(y) > 0:
            vals.append(spearmanr(s, y)[0])
    return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]


def component_corr(rows, labels):
    out = {}
    for k in KEYS:
        s = [r["components"].get(k) for r in rows]
        idx = [i for i, v in enumerate(s) if v is not None]
        if len(idx) < 10:
            continue
        out[k] = {
            "n": len(idx),
            "spearman": round(float(spearmanr([s[i] for i in idx], [labels[rows[i]["pair"]] for i in idx])[0]), 3),
        }
    return out


def per_job(rows, labels):
    by_job = defaultdict(list)
    for r in rows:
        by_job[r["job"]].append(r)
    return {j: round(float(spearmanr([r["score"] for r in rs], [labels[r["pair"]] for r in rs])[0]), 3)
            for j, rs in sorted(by_job.items())}


# ---------------------------------------------------------------- fitting
def overall_from(comp, w):
    avail = {k: v for k, v in comp.items() if v is not None}
    ws = sum(w[k] for k in avail)
    return 100 * sum(w[k] * v for k, v in avail.items()) / ws if ws else 0.0


def _component_matrix(rows, keys):
    """(n, len(keys)) matrix of components; missing components are NaN."""
    m = np.full((len(rows), len(keys)), np.nan)
    for i, r in enumerate(rows):
        for j, k in enumerate(keys):
            v = r["components"].get(k)
            if v is not None:
                m[i, j] = v
    return m


def fit_weights(rows, labels, keys=KEYS, prior=None, seed_weights=None):
    """Maximise Spearman (the ranking metric) of the weighted component blend against the labels."""
    prior = np.asarray(PRIOR if prior is None else prior, float)
    m = _component_matrix(rows, keys)
    y = np.asarray([labels[r["pair"]] for r in rows], float)
    ok = ~np.isnan(m)
    starts = [prior] + ([np.asarray(seed_weights, float)] if seed_weights is not None else []) + [
        np.ones(len(keys)) / len(keys), np.array([0.5, 0.4, 0.05, 0.05]), np.array([0.3, 0.6, 0.05, 0.05])]

    def obj(w):
        w = np.abs(w)
        if w.sum() == 0:
            return 1.0
        w = w / w.sum()
        wm = np.where(ok, w, 0.0)
        denom = wm.sum(axis=1)
        s = np.where(denom > 0, 100 * np.nansum(np.nan_to_num(m) * wm, axis=1) / np.maximum(denom, 1e-9), 0.0)
        if np.std(s) == 0:
            return 1.0
        return -spearmanr(s, y)[0] + 0.02 * float(np.sum((w - prior) ** 2))

    best = None
    for st in starts:
        res = minimize(obj, np.asarray(st, float), method="Nelder-Mead",
                       options={"xatol": 1e-3, "fatol": 1e-5, "maxiter": 800})
        if best is None or res.fun < best.fun:
            best = res
    w = np.abs(best.x)
    w = w / w.sum()
    return dict(zip(keys, [round(float(v), 3) for v in w]))


def fit_isotonic(rows, labels):
    """Monotone map score -> expected label (fit on the given rows)."""
    s = np.asarray([r["score"] for r in rows], float)
    y = np.asarray([labels[r["pair"]] for r in rows], float)
    iso = IsotonicRegression(y_min=0.0, y_max=4.0, out_of_bounds="clip")
    iso.fit(s, y)
    return iso


def apply_calibration(rows, iso):
    for r in rows:
        r["cal_score"] = float(iso.predict([r["score"]])[0])
        r["score_cal"] = round(100 * r["cal_score"] / 4.0, 1)
    return rows


# ---------------------------------------------------------------- diagnostics
def taxonomy_coverage(rows, jobs, cvs, jb):
    """Which required skills of a job are never matched by any CV, and which CV skills are unknown."""
    req_seen = defaultdict(Counter)
    cv_unknown = Counter()
    for r in rows:
        js = jb[r["job"]]
        spec = cvs[r["cv"]]
        present = set(spec.skills.present)
        for s in js.required:
            req_seen[r["job"]][s] += 1 if s in present else 0
    for spec in cvs.values():
        for s in spec.skills.present:
            cv_unknown[s] += 1
    never = {job: sorted([s for s, n in c.items() if n == 0]) for job, c in req_seen.items()}
    return {"required_never_matched": {k: v for k, v in never.items() if v},
            "most_common_cv_skills": cv_unknown.most_common(25)}


def print_errors(rows, labels, top=6):
    by_job = defaultdict(list)
    for r in rows:
        by_job[r["job"]].append(r)
    for job, rs in sorted(by_job.items()):
        rs = sorted(rs, key=lambda r: -(labels[r["pair"]] - r["score"] / 20.0))
        print(f"\n-- {job}")
        for r in rs[:top]:
            print(f"   label={labels[r['pair']]} score={r['score']:5.1f} caps={','.join(r['caps']) or '-':22s} "
                  f"comp={ {k: v for k, v in r['components'].items()} } pair={r['pair']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=DEFAULT_LABELS)
    ap.add_argument("--sonnet", default=DEFAULT_SONNET)
    ap.add_argument("--model", default=Settings().embedding_model)
    ap.add_argument("--hashing", action="store_true")
    ap.add_argument("--fit", action="store_true", help="fit weights on the train jobs and write data/model_params.json")
    ap.add_argument("--cv-folds", type=int, default=0, help="cross-validate by job (0 = off)")
    ap.add_argument("--no-write-params", action="store_true")
    ap.add_argument("--errors", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    jobs = json.loads((HERE / "jobs.json").read_text(encoding="utf-8"))
    grid = load_grid_labels(a.labels)
    sonnet = load_sonnet(a.sonnet)
    pairs = [(p, r["job"], r["cv"]) for p, r in grid.items()]
    print(f"grid: {len(pairs)} pairs, {len(grid)} labelled, {len(sonnet)} sonnet cross-check labels")

    emb = HashingEmbedder() if a.hashing else OnnxEmbedder(a.model, Settings().embedding_cache_dir, cache_size=60000)
    cv_ids = sorted({cv for _, _, cv in pairs})
    t0 = time.time()
    cvs, jb = build_specs(jobs, cv_ids)
    print(f"specs built in {time.time() - t0:.0f}s ({len(cvs)} CVs, {len(jb)} jobs)")

    params = scorer.load_params()
    t0 = time.time()
    rows = score_grid(pairs, cvs, jb, emb, params)
    print(f"scored {len(rows)} pairs in {time.time() - t0:.0f}s")

    labels = {p: r["label"] for p, r in grid.items()}
    rep = {"model": emb.name, "n_pairs": len(rows), "params": {"weights": params.weights, "calibration": params.calibration,
                                                             "title_weight": params.title_weight, "version": params.version}}

    rep["overall"] = metrics(rows, labels)
    rep["overall"]["spearman_ci95"] = bootstrap_ci(rows, labels)
    rep["in_domain"] = metrics([r for r in rows if grid[r["pair"]]["in_domain"]], labels)
    rep["out_of_domain"] = metrics([r for r in rows if not grid[r["pair"]]["in_domain"]], labels)
    rep["component_corr"] = component_corr(rows, labels)
    rep["per_job_spearman"] = per_job(rows, labels)

    if sonnet:
        srows = [r for r in rows if r["pair"] in sonnet]
        sl = {p: v for p, v in sonnet.items()}
        rep["sonnet_crosscheck"] = metrics(srows, sl)
        rep["sonnet_crosscheck"]["spearman_ci95"] = bootstrap_ci(srows, sl)
    ab = load_original_labels()
    if ab:
        both = {p: (ab["A"][p] + ab["B"][p]) / 2 for p in ab["A"] if p in ab["B"] and p in labels}
        rep["vs_AB_consensus_240"] = metrics([r for r in rows if r["pair"] in both], both)
        rep["haiku_vs_AB_spearman_240"] = round(float(spearmanr([labels[p] for p in both], [both[p] for p in both])[0]), 3)

    # ---- cross-validation by job (fit on some jobs, evaluate on the rest)
    if a.cv_folds:
        by_job = defaultdict(list)
        for r in rows:
            by_job[r["job"]].append(r)
        job_slugs = sorted(by_job)
        folds = []
        for f in range(a.cv_folds):
            tr_jobs = [j for i, j in enumerate(job_slugs) if i % a.cv_folds == f]
            te_jobs = [j for i, j in enumerate(job_slugs) if i % a.cv_folds != f]
            folds.append((tr_jobs, te_jobs))
        cvrep = []
        for tr_jobs, te_jobs in folds:
            tr_rows = [r for j in tr_jobs for r in by_job[j]]
            te_rows = [r for j in te_jobs for r in by_job[j]]
            w = fit_weights(tr_rows, labels, seed_weights=list(params.weights.values()))
            iso = fit_isotonic(tr_rows, labels)
            for r in te_rows:
                r["cv_score"] = overall_from(r["components"], w)
            for r in te_rows:
                r["cv_score_cal"] = float(iso.predict([r["cv_score"]])[0])
            m = metrics(te_rows, labels)
            m["weights"] = w
            m["train_jobs"], m["test_jobs"] = len(tr_jobs), len(te_jobs)
            m["spearman_ci95"] = bootstrap_ci(te_rows, labels)
            cvrep.append(m)
        rep["cross_validation"] = {"folds": cvrep,
                                   "mean_spearman": round(float(np.mean([m["spearman"] for m in cvrep])), 3),
                                   "mean_auc_ge2": round(float(np.mean([m["auc_ge2"] for m in cvrep])), 3),
                                   "mean_ndcg5": round(float(np.mean([m["ndcg5"] for m in cvrep])), 3)}

    # ---- fit on the train jobs of eval/jobs.json and (optionally) write the params
    if a.fit:
        train_jobs = {j["slug"] for j in jobs if j.get("_split") == "train"}
        tr_rows = [r for r in rows if r["job"] in train_jobs]
        te_rows = [r for r in rows if r["job"] not in train_jobs]
        w = fit_weights(tr_rows, labels, seed_weights=list(params.weights.values()))
        iso = fit_isotonic(tr_rows, labels)
        for r in rows:
            r["fit_score"] = overall_from(r["components"], w)
            r["fit_cal"] = float(iso.predict([r["fit_score"]])[0])
        rep["fit"] = {"weights": w, "train_jobs": sorted(train_jobs),
                      "train": metrics(tr_rows, labels), "test": metrics(te_rows, labels)}
        rep["fit"]["test"]["spearman_ci95"] = bootstrap_ci(te_rows, labels)
        # calibrate the 0-100 scale of the test split with the isotonic map
        for r in te_rows:
            r["score_cal"] = round(100 * r["fit_cal"] / 4.0, 1)
        rep["fit"]["test_calibrated"] = metrics(te_rows, labels)
        rep["fit"]["isotonic_points"] = [[round(float(x), 1), round(float(y), 3)]
                                         for x, y in zip(iso.X_thresholds_, iso.y_thresholds_)]
        if not a.no_write_params:
            rep["fit"]["wrote_params"] = False

    if a.errors:
        print_errors(rows, labels)
        print("\n=== taxonomy coverage ===")
        print(json.dumps(taxonomy_coverage(rows, jobs, cvs, jb), indent=1)[:4000])

    tag = a.tag or emb.name.split("/")[-1]
    (HERE / f"results_full_{tag}.json").write_text(json.dumps(rep, indent=1, default=float))
    print(f"\nwrote {HERE / f'results_full_{tag}.json'}")
    print(json.dumps({k: rep[k] for k in ("overall", "in_domain", "out_of_domain", "component_corr")}, indent=1, default=float))
    if "sonnet_crosscheck" in rep:
        print("sonnet cross-check:", json.dumps(rep["sonnet_crosscheck"], default=float))
    if "cross_validation" in rep:
        print("cross-validation:", json.dumps(rep["cross_validation"]["mean_spearman"]), "mean rho over folds")
    if "fit" in rep:
        print("fit:", json.dumps(rep["fit"]["weights"]), "test rho", rep["fit"]["test"]["spearman"])


if __name__ == "__main__":
    main()
