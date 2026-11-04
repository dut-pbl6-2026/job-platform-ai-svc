"""Validate label files and report inter-annotator agreement."""
import collections
import glob
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

H = Path(__file__).parent
pairs = {p["pair_id"]: p for p in json.loads((H / "pairs.json").read_text())}
L = {}
for who in "AB":
    d = {}
    for f in sorted(glob.glob(str(H / "labels" / who / "batch*.json"))):
        d.update({k: v["score"] for k, v in json.loads(Path(f).read_text()).items()})
    L[who] = d
    print(who, len(d), "missing", len(set(pairs) - set(d)), "invalid", [k for k, v in d.items() if v not in (0, 1, 2, 3, 4)], sorted(collections.Counter(d.values()).items()))
ids = sorted(pairs)
a, b = np.array([L["A"][i] for i in ids]), np.array([L["B"][i] for i in ids])
print("spearman", round(spearmanr(a, b)[0], 3), "weighted kappa", round(cohen_kappa_score(a, b, weights="quadratic"), 3), "exact", round(float(np.mean(a == b)), 3), "within1", round(float(np.mean(abs(a - b) <= 1)), 3))
for who in "AB":
    by = collections.defaultdict(list)
    for i in ids:
        by[pairs[i]["stratum"]].append(L[who][i])
    print(who, "mean label by sampling stratum (hidden from annotators):", {s: round(float(np.mean(v)), 2) for s, v in by.items()})
