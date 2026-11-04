"""Compare embedding providers on the full label grid (5 760 pairs), same text for every provider.

Providers: local paraphrase-multilingual-MiniLM-L12-v2, local paraphrase-multilingual-mpnet-base-v2
and the NVIDIA NIM API (nvidia/nemotron-3.5-embed-8b).

PII: CV texts are personal data. Before anything leaves the machine they are scrubbed (contact
lines, e-mails, phones, links) and the *same* scrubbed text is used for the local providers, so
the comparison is apples to apples. Nothing is written to the repo except the metrics file.

Usage:
  NVIDIA_API_KEY=... python eval/compare_embedders.py
"""
import argparse
import base64
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import fit_grid as fg

from app.config import Settings
from app.embed import OnnxEmbedder
from app.nlp.normalize import redact
from app.scoring import scorer
from app.scoring.specs import build_cv_spec, build_job_spec

NVIDIA_URL = "https://integrate.api.nvidia.com/v1/embeddings"
NVIDIA_MODEL = "nvidia/nemotron-3.5-embed-8b"
CACHE_DIR = HERE / "cache" / "nvidia"


def scrub_cv(text: str) -> str:
    """De-identify a CV before it leaves the machine: drop the name/contact header, mask contacts."""
    lines = text.split("\n")
    body = "\n".join(lines[3:]) if len(lines) > 3 else text
    return redact(body)


class NvidiaEmbedder:
    """Embedder for the NVIDIA NIM embeddings endpoint (query/passage input types).

    Measured on this endpoint (Nov 2026): a request with **more than one input** fails with
    HTTP 500 in ~15-20 % of the cases, while single-input requests are reliable, so the client
    sends one text per request and parallelises instead. `encoding_format=base64` returns the
    same float32 vector with 2.4x less traffic.
    """

    def __init__(self, model=NVIDIA_MODEL, cache_dir=CACHE_DIR, batch_size=1, workers=6):
        self.name = model
        self.model = model
        self.batch_size = batch_size
        self.workers = workers
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[tuple[str, str], np.ndarray] = {}
        self._api_calls = 0
        self._failed = 0

    # -- cache -------------------------------------------------------------
    def _cache_path(self, input_type):
        return self.cache_dir / f"{self.model.replace('/', '_')}__{input_type}.npz"

    def load_cache(self):
        for input_type in ("query", "passage"):
            p = self._cache_path(input_type)
            if not p.exists():
                continue
            d = np.load(p, allow_pickle=False)
            for h, v in zip(d["hashes"], d["vectors"]):
                self._cache[(h, input_type)] = v
        print(f"cache: {len(self._cache)} vectors", flush=True)

    def save_cache(self):
        for input_type in ("query", "passage"):
            items = [(h, v) for (h, t), v in self._cache.items() if t == input_type]
            if not items:
                continue
            hashes = np.array([h for h, _ in items])
            vectors = np.stack([v for _, v in items])
            np.savez_compressed(self._cache_path(input_type), hashes=hashes, vectors=vectors)

    # -- api ---------------------------------------------------------------
    def _embed_batch(self, texts, input_type, client, key):
        payload = {"model": self.model, "input": texts, "encoding_format": "base64",
                   "modality": ["text"], "input_type": input_type, "truncate": "END"}
        last = ""
        for attempt in range(6):
            try:
                r = client.post(NVIDIA_URL, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                                json=payload, timeout=300)
            except httpx.HTTPError as exc:  # network hiccup
                last = str(exc)[:120]
                time.sleep(min(1.5 * (attempt + 1), 10))
                continue
            if r.status_code == 200:
                self._api_calls += 1
                out = []
                for d in r.json()["data"]:
                    emb = d["embedding"]
                    if isinstance(emb, str):  # base64 float32
                        emb = np.frombuffer(base64.b64decode(emb), dtype="<f4")
                    out.append(np.asarray(emb, dtype=np.float32))
                return np.stack(out)
            last = f"HTTP {r.status_code}: {r.text[:160]}"
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(min(1.5 * (attempt + 1), 10))
                continue
            raise RuntimeError(f"NVIDIA embeddings {last}")
        self._failed += 1
        raise RuntimeError(f"NVIDIA embeddings: retries exhausted ({last})")

    def _embed(self, texts, input_type, client, key):
        missing = [t for t in dict.fromkeys(texts) if (hashlib.sha1(t.encode()).hexdigest(), input_type) not in self._cache]
        if missing:
            batches = [missing[i:i + self.batch_size] for i in range(0, len(missing), self.batch_size)]
            t_start = time.time()
            done = 0
            with ThreadPoolExecutor(self.workers) as ex:
                futures = [ex.submit(self._embed_batch, b, input_type, client, key) for b in batches]
                for f, b in zip(futures, batches):
                    vecs = f.result()
                    for t, v in zip(b, vecs):
                        self._cache[(hashlib.sha1(t.encode()).hexdigest(), input_type)] = v
                    done += len(b)
                    if done % 200 < self.batch_size or done == len(missing):
                        self.save_cache()  # durable progress: the run is long and the endpoint is flaky
                        rate = done / max(time.time() - t_start, 1e-6)
                        eta = (len(missing) - done) / max(rate, 1e-6)
                        print(f"  {input_type}: {done}/{len(missing)} texts ({rate:.1f}/s, eta {eta / 60:.0f} min)", flush=True)
        out = np.stack([self._cache[(hashlib.sha1(t.encode()).hexdigest(), input_type)] for t in texts])
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)

    # -- Embedder protocol -------------------------------------------------
    def embed(self, texts):
        return self._embed(texts, "passage", self._client, self._key)

    def embed_query(self, texts):
        return self._embed(texts, "query", self._client, self._key)

    def embed_passage(self, texts):
        return self._embed(texts, "passage", self._client, self._key)


def score_with(emb, cvs, jb, pairs):
    rows = []
    params = scorer.load_params()
    for pair, job, cv_id in pairs:
        r = scorer.score(cvs[cv_id], jb[job], emb, params)
        rows.append({"pair": pair, "job": job, "cv": cv_id, "components": r["components"], "raw": r["raw_semantic"],
                     "lang": r["lang_pair"], "relevance": r["relevance"], "score_shipped": float(r["overall_score"])})
    return rows


def fit_calibration(rows, labels, by_lang):
    """Per language-pair cosine window, fitted on the given rows (median of weak pairs, p90 of good pairs)."""
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


def semantic_with_calibration(rows, cal):
    """Recompute the semantic component with a provider-specific cosine window."""
    out = []
    for r in rows:
        lo, hi = cal[r["lang"]]
        out.append(float(np.clip((r["raw"] - lo) / max(hi - lo, 1e-6), 0.0, 1.0)))
    return np.asarray(out)


def blend(rows, weights, semantic):
    """Blend components with a provider-specific semantic value (experience/education are gated by
    relevance, which depends on the semantic component, so they are recomputed here)."""
    out = []
    for r, sem in zip(rows, semantic):
        comps = dict(r["components"])
        comps["semantic"] = float(sem)
        relevance = max(sem, comps.get("skills") or 0.0)
        if comps.get("experience") is not None:
            comps["experience"] = comps["experience"] / max(r["relevance"], 1e-9) * relevance
        if comps.get("education") is not None:
            comps["education"] = comps["education"] / max(r["relevance"], 1e-9) * relevance
        avail = {k: v for k, v in comps.items() if v is not None}
        ws = sum(weights[k] for k in avail)
        out.append(100 * sum(weights[k] * avail[k] for k in avail) / ws if ws else 0.0)
    return np.asarray(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=fg.DEFAULT_LABELS)
    ap.add_argument("--sonnet", default=fg.DEFAULT_SONNET)
    ap.add_argument("--providers", default="minilm,mpnet,nvidia")
    ap.add_argument("--out", default=str(HERE / "results_compare_embedders.json"))
    a = ap.parse_args()

    grid = fg.load_labels(a.labels)
    sonnet = fg.load_sonnet(a.sonnet)
    ab_all = fg.load_ab_labels()
    ab_cons = {p: (ab_all["joined"]["A"][p] + ab_all["joined"]["B"][p]) / 2
               for p in ab_all["joined"]["A"] if p in ab_all["joined"]["B"]}
    hy = {p: v["label"] for p, v in grid.items()}
    jobs = json.loads((HERE / "jobs.json").read_text(encoding="utf-8"))
    train_jobs = {j["slug"] for j in jobs if j.get("_split") == "train"}

    t0 = time.time()
    cvs = {}
    for cid in sorted({v["cv"] for v in grid.values()}):
        raw = (HERE / "cache" / "cvs" / f"{cid}.txt").read_text(encoding="utf-8")
        cvs[cid] = build_cv_spec(scrub_cv(raw))
    jb = {j["slug"]: build_job_spec(j) for j in jobs}
    pairs = [(p, v["job"], v["cv"]) for p, v in grid.items()]
    print(f"specs (PII-scrubbed) built in {time.time() - t0:.0f}s")

    providers = {}
    if "minilm" in a.providers:
        providers["minilm"] = OnnxEmbedder(Settings().embedding_model, Settings().embedding_cache_dir, cache_size=60000)
    if "mpnet" in a.providers:
        providers["mpnet"] = OnnxEmbedder("sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                                          Settings().embedding_cache_dir, cache_size=60000)
    if "nvidia" in a.providers:
        key = os.environ.get("NVIDIA_API_KEY")
        if not key:
            print("NVIDIA_API_KEY not set - skipping the nvidia provider")
        else:
            nv = NvidiaEmbedder()
            nv.load_cache()
            nv._key, nv._client = key, httpx.Client()
            providers["nvidia"] = nv

    rep = {"note": "all providers score the same PII-scrubbed CV text", "providers": {}}
    for name, emb in providers.items():
        if name == "nvidia":
            # pre-embed every unique text in one pass: the endpoint only accepts 2 texts per
            # request, and per-pair calls would issue thousands of tiny requests.
            passages = {c for cv in cvs.values() for c in cv.chunks} | {c for cv in cvs.values() for c in cv.title_candidates}
            queries = {l for j in jb.values() for l in (j.lines or [])} | {j.title for j in jb.values() if j.title}
            t0 = time.time()
            emb._embed(sorted(passages), "passage", emb._client, emb._key)
            emb._embed(sorted(queries), "query", emb._client, emb._key)
            emb.save_cache()
            print(f"nvidia: pre-embedded {len(passages)} passages + {len(queries)} queries in {time.time() - t0:.0f}s")
        t0 = time.time()
        rows = score_with(emb, cvs, jb, pairs)
        dt = time.time() - t0
        tr = [r for r in rows if r["job"] in train_jobs]
        te = [r for r in rows if r["job"] not in train_jobs]
        cal = fit_calibration(tr, hy, None)
        sem_tr = semantic_with_calibration(tr, cal)
        sem_te = semantic_with_calibration(te, cal)
        # weights fitted on the train jobs with the provider-specific semantic component
        tmp_rows = [dict(r, components=dict(r["components"], semantic=float(s)), relevance=max(float(s), r["components"].get("skills") or 0.0))
                    for r, s in zip(tr, sem_tr)]
        w = fg.fit_weights(tmp_rows, hy)
        s_te = blend(te, w, sem_te)
        ab_idx = [i for i, r in enumerate(te) if r["pair"] in ab_cons]
        so_idx = [i for i, r in enumerate(te) if r["pair"] in sonnet]
        out = {
            "weights": w,
            "calibration": {k: list(v) for k, v in cal.items()},
            "seconds_for_5760_pairs": round(dt, 1),
            "heldout_vs_grid": fg.eval_scores(te, s_te, hy),
            "heldout_vs_AB": fg.eval_scores([te[i] for i in ab_idx], s_te[ab_idx], ab_cons),
            "heldout_vs_sonnet": fg.eval_scores([te[i] for i in so_idx], s_te[so_idx], sonnet),
            "semantic_rho_heldout": round(float(spearmanr(sem_te, [hy[r["pair"]] for r in te])[0]), 4),
            "skills_rho_heldout": round(float(spearmanr([r["components"]["skills"] for r in te], [hy[r["pair"]] for r in te])[0]), 4),
        }
        if hasattr(emb, "_api_calls"):
            out["api_calls"] = emb._api_calls
        out["scores"] = {r["pair"]: round(float(s), 4) for r, s in zip(rows, blend(rows, w, semantic_with_calibration(rows, cal)))}
        rep["providers"][name] = out
        print(f"\n=== {name} ({dt:.0f}s for 5760 pairs) ===")
        print(json.dumps({k: v for k, v in out.items() if k != "weights"}, indent=1, default=float))

    if hasattr(providers.get("nvidia"), "save_cache"):
        providers["nvidia"].save_cache()
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float))
    print(f"\nwrote {a.out}")
    # a small, committable summary (the per-pair scores above are git-ignored)
    summary = {"note": rep["note"],
               "providers": {k: {kk: vv for kk, vv in v.items() if kk != "scores"} for k, v in rep["providers"].items()}}
    summary_path = Path(a.out).with_name(Path(a.out).stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=1, default=float))
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
