"""Merge the provider comparison runs and test the differences (paired bootstrap over jobs).

Reads the per-pair score dumps written by `eval/compare_embedders.py` (git-ignored) and writes
`eval/results_compare_summary.json` (small, committable): the metrics of every provider plus
the paired bootstrap of their differences on the held-out jobs.

Usage:
  python eval/compare_stats.py --local eval/results_compare_local.json \
                                --nvidia eval/results_compare_embedders.json \
                                --out eval/results_compare_summary.json
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).parent
DEFAULT_GRID = HERE / "labels" / "grid" / "haiku_5760.json"


def load_grid(path):
    blob = json.loads(Path(path).read_text(encoding="utf-8"))
    return blob.get("labels", blob)


def paired_bootstrap(a, b, y, by_job, n=4000, seed=11):
    """Distribution of rho(a) - rho(b) when whole jobs are resampled."""
    rng = np.random.default_rng(seed)
    jobs = sorted(by_job)
    deltas = []
    for _ in range(n):
        pick = rng.choice(jobs, len(jobs))
        idx = [i for j in pick for i in by_job[j]]
        deltas.append(spearmanr(a[idx], y[idx])[0] - spearmanr(b[idx], y[idx])[0])
    d = np.asarray(deltas)
    return {"delta_rho_median": round(float(np.median(d)), 4),
            "ci95": [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)],
            "p_first_better": round(float(np.mean(d > 0)), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", default=str(HERE / "results_compare_local.json"))
    ap.add_argument("--nvidia", default=str(HERE / "results_compare_embedders.json"))
    ap.add_argument("--grid", default=str(DEFAULT_GRID))
    ap.add_argument("--out", default=str(HERE / "results_compare_summary.json"))
    a = ap.parse_args()

    grid = load_grid(a.grid)
    jobs = json.loads((HERE / "jobs.json").read_text(encoding="utf-8"))
    test_jobs = {j["slug"] for j in jobs if j.get("_split") != "train"}

    providers = {}
    for f in (a.local, a.nvidia):
        if not Path(f).exists():
            continue
        blob = json.loads(Path(f).read_text(encoding="utf-8"))
        for name, p in blob["providers"].items():
            providers[name] = {k: v for k, v in p.items() if k != "scores"}

    scores = {}
    for f in (a.local, a.nvidia):
        if not Path(f).exists():
            continue
        blob = json.loads(Path(f).read_text(encoding="utf-8"))
        for name, p in blob["providers"].items():
            scores[name] = p["scores"]

    pairs = sorted(set.intersection(*(set(s) for s in scores.values())))
    te = [p for p in pairs if p.split("__", 1)[0] in test_jobs]
    y = np.asarray([grid[p] for p in te], float)
    by_job = defaultdict(list)
    for i, p in enumerate(te):
        by_job[p.split("__", 1)[0]].append(i)

    rho = {}
    for name, s in scores.items():
        v = np.asarray([s[p] for p in te], float)
        rho[name] = float(spearmanr(v, y)[0])
        providers[name]["heldout_rho_recomputed"] = round(rho[name], 4)

    diffs = {}
    names = list(rho)
    for i, n1 in enumerate(names):
        for n2 in names[i + 1:]:
            a1 = np.asarray([scores[n1][p] for p in te], float)
            a2 = np.asarray([scores[n2][p] for p in te], float)
            diffs[f"{n1}_minus_{n2}"] = paired_bootstrap(a1, a2, y, by_job)

    rep = {"protocol": "same PII-scrubbed CV text, provider-specific cosine window and weights fitted on the 16 train jobs; "
                       "held-out = the 8 unseen jobs (1 920 pairs)",
           "n_heldout_pairs": len(te),
           "providers": providers,
           "heldout_rho_vs_grid": {k: round(v, 4) for k, v in rho.items()},
           "paired_bootstrap": diffs}
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps({"heldout_rho_vs_grid": rep["heldout_rho_vs_grid"], "paired_bootstrap": diffs}, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
