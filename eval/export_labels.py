"""Export the grid labels into the repo so the 5 760-pair benchmark is reproducible.

The labels are rubric judgements (0-4) per (job, CV) pair - no CV text, no profile, no PII.
Source: the local `enfue-eval` dataset (another project of the team, same CVs / jobs / rubric):
  - Haiku labels for every pair (5 760)
  - an independent Sonnet cross-check on 292 same-domain pairs
  - the 240 pairs that already carry the repo's own two-annotator labels (A/B)

Usage: python eval/export_labels.py [--labels PATH] [--sonnet PATH] [--out-dir eval/labels/grid]
"""
import argparse
import json
import os
from pathlib import Path

HERE = Path(__file__).parent
DEFAULT_LABELS = os.path.expanduser("~/enfue-eval/dataset/pairs_labels.jsonl")
DEFAULT_SONNET = os.path.expanduser("~/enfue-eval/dataset/sonnet-labels.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=DEFAULT_LABELS)
    ap.add_argument("--sonnet", default=DEFAULT_SONNET)
    ap.add_argument("--out-dir", default=str(HERE / "labels" / "grid"))
    a = ap.parse_args()

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = {}
    in_domain = {}
    with open(a.labels, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            grid[r["pair"]] = int(r["haiku_label"])
            in_domain[r["pair"]] = bool(r["in_domain"])
    (out_dir / "haiku_5760.json").write_text(json.dumps({
        "source": "enfue-eval dataset (local) - Claude Haiku 5.5 rubric labelling, identical rubric file (eval/RUBRIC.md)",
        "note": "0-4 label per '<job>__<cv id>' pair; CV text stays local (eval/cache/cvs, git-ignored)",
        "labels": grid,
    }, indent=0, sort_keys=True))
    (out_dir / "in_domain_768.json").write_text(json.dumps(sorted(k for k, v in in_domain.items() if v), indent=1))

    if Path(a.sonnet).exists():
        sonnet = {k: int(v["score"]) for k, v in json.loads(Path(a.sonnet).read_text(encoding="utf-8")).items()}
        (out_dir / "sonnet_292.json").write_text(json.dumps({
            "source": "enfue-eval dataset (local) - Claude Sonnet cross-check on 292 same-domain pairs",
            "labels": sonnet,
        }, indent=0, sort_keys=True))

    print(f"exported {len(grid)} grid labels and {len(sonnet) if Path(a.sonnet).exists() else 0} sonnet labels to {out_dir}")


if __name__ == "__main__":
    main()
