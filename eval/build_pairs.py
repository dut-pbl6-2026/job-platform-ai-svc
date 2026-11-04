"""Sample CV/JD pairs (stratified, blind to the scorer) and OCR the CVs once into eval/cache/cvs/.

Per JD: 4 CVs from likely-relevant occupation folders, 3 adjacent, 3 unrelated. Folder names are used ONLY for sampling;
annotators and the scorer never see them (and they are known to be noisy). Usage: python eval/build_pairs.py
"""
import hashlib
import json
import os
import random
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_THREAD_LIMIT", "1")
ROOT = Path.home() / "cv-local-sample" / "archive" / "Resumes PDF"
OUT = Path(__file__).parent
CACHE = OUT / "cache" / "cvs"
SEED = 20261102


def norm(s: str) -> str:
    return re.sub(r"(resumes|[\s_\-]+)", "", s.lower())


def match(folder: str, key: str) -> bool:
    f, k = norm(folder), norm(key)
    return f == k or (len(k) >= 3 and f.startswith(k))


def ocr_one(args):
    cv_id, path = args
    from app.config import Settings
    from app.parsing.parse import parse_cv

    out = CACHE / f"{cv_id}.txt"
    if out.exists():
        return cv_id, "cached"
    try:
        p = parse_cv(Path(path).read_bytes(), Settings(max_pages=3))
        out.write_text(p.text, encoding="utf-8")
        return cv_id, p.parser
    except Exception as e:  # noqa: BLE001
        return cv_id, f"ERR {type(e).__name__}"


def main():
    rnd = random.Random(SEED)
    jobs = json.loads((OUT / "jobs.json").read_text())
    folders = sorted(p.name for p in ROOT.iterdir() if p.is_dir())
    files = {f: sorted((ROOT / f).glob("*.pdf")) for f in folders}
    used: set[str] = set()
    pairs = []

    def pick(candidate_folders, k):
        pool = [p for f in candidate_folders for p in files[f] if str(p) not in used]
        rnd.shuffle(pool)
        got = pool[:k]
        used.update(str(p) for p in got)
        return got

    for j in jobs:
        rel = [f for f in folders if any(match(f, k) for k in j["_pool"]["relevant"])]
        adj = [f for f in folders if any(match(f, k) for k in j["_pool"]["adjacent"]) and f not in rel]
        other = [f for f in folders if f not in rel and f not in adj]
        chosen = [(p, "relevant") for p in pick(rel, 4)] + [(p, "adjacent") for p in pick(adj, 3)]
        chosen += [(p, "unrelated") for p in pick(other, 10 - len(chosen))]
        rnd.shuffle(chosen)
        for i, (p, stratum) in enumerate(chosen):
            cv_id = hashlib.sha1(str(p.relative_to(ROOT)).encode()).hexdigest()[:10]
            pairs.append({"pair_id": f"{j['slug']}__{i:02d}", "job_slug": j["slug"], "cv_id": cv_id, "cv_path": str(p.relative_to(ROOT)), "stratum": stratum, "split": j["_split"]})
    CACHE.mkdir(parents=True, exist_ok=True)
    todo = sorted({(p["cv_id"], str(ROOT / p["cv_path"])) for p in pairs})
    with ProcessPoolExecutor(6) as ex:
        res = list(ex.map(ocr_one, todo))
    bad = [r for r in res if r[1].startswith("ERR")]
    print("pairs", len(pairs), "unique cvs", len(todo), "errors", bad)
    (OUT / "pairs.json").write_text(json.dumps(pairs, indent=1))


if __name__ == "__main__":
    main()
