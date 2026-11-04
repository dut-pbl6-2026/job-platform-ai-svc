"""Latency benchmark against a running service. Usage: python eval/latency.py --url http://localhost:6000/api/ai/score-resume --token T --job JOBID [-n 30]"""
import argparse
import json
import random
import statistics as st
import time
from pathlib import Path

import httpx

ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True)
ap.add_argument("--token", required=True)
ap.add_argument("--job", required=True)
ap.add_argument("-n", type=int, default=30)
a = ap.parse_args()
root = Path.home() / "cv-local-sample" / "archive" / "Resumes PDF"
pairs = json.loads((Path(__file__).parent / "pairs.json").read_text())
files = sorted({p["cv_path"] for p in pairs})
random.Random(3).shuffle(files)
h = {"Authorization": f"Bearer {a.token}"}
cold, hit = [], []
with httpx.Client(timeout=120) as c:
    for f in files[: a.n]:
        data = (root / f).read_bytes()
        t = time.perf_counter()
        r = c.post(a.url, headers=h, files={"cv_file": ("cv.pdf", data)}, data={"job_id": a.job})
        dt = time.perf_counter() - t
        if r.status_code != 200:
            print("ERR", r.status_code, r.text[:120])
            continue
        cold.append(dt)
        t = time.perf_counter()
        c.post(a.url, headers=h, files={"cv_file": ("cv.pdf", data)}, data={"job_id": a.job})
        hit.append(time.perf_counter() - t)
q = lambda v, p: sorted(v)[min(len(v) - 1, int(p * len(v)))]
print(json.dumps({"n": len(cold), "cold_p50_s": round(st.median(cold), 2), "cold_p95_s": round(q(cold, 0.95), 2), "cold_max_s": round(max(cold), 2),
                  "cache_hit_p50_s": round(st.median(hit), 3), "cache_hit_p95_s": round(q(hit, 0.95), 3)}))
