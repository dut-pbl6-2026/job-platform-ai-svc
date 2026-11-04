"""Render annotation batches (JD + 10 anonymised CV excerpts each) for the blind annotators."""
import json
from pathlib import Path

OUT = Path(__file__).parent
jobs = json.loads((OUT / "jobs.json").read_text())
pairs = json.loads((OUT / "pairs.json").read_text())
(OUT / "batches").mkdir(exist_ok=True)
MAXC = 3200
by_job = {}
for p in pairs:
    by_job.setdefault(p["job_slug"], []).append(p)
for b in range(4):
    parts = [f"# Annotation batch {b + 1}\n\nGrade every pair 0-4 per RUBRIC. Pair ids are opaque.\n"]
    for j in jobs[b * 6 : (b + 1) * 6]:
        parts.append(f"\n\n========== JOB `{j['slug']}` ==========\nTitle: {j['title']}\nExperience level: {j['experienceLevel']}\n\nDescription:\n{j['description']}\n\nRequirements:\n{j['requirements']}\n")
        for p in by_job[j["slug"]]:
            cv = (OUT / "cache" / "cvs" / f"{p['cv_id']}.txt").read_text(encoding="utf-8")
            cut = cv[:MAXC] + ("\n[...truncated...]" if len(cv) > MAXC else "")
            parts.append(f"\n---------- PAIR `{p['pair_id']}` ----------\n{cut}\n")
    (OUT / "batches" / f"batch{b + 1}.md").write_text("".join(parts), encoding="utf-8")
    print(b + 1, sum(len(x) for x in parts) // 4, "approx tokens")
