# Resume scoring — evaluation report (SCORE-01)

Generated from `eval/run.py` (numbers: `eval/results_paraphrase-multilingual-MiniLM-L12-v2.json`). Spec: `job-platform-docs/docs/ai/resume-scoring.md`.

## 1. Setup (what was actually measured)

| Item | Value |
|---|---|
| CVs | 240 distinct CVs sampled from `~/cv-local-sample` (8 905 scanned PDFs, English), OCR'd by the service's own parser |
| Jobs | 24 hand-written postings (16 EN, 8 VI; IT and non-IT) in `eval/jobs.json` |
| Pairs | 240 (10 CVs per job: 4 likely-relevant, 3 adjacent, 3 unrelated occupation folders; folder names used only for sampling) |
| Split | by job: 16 train jobs (160 pairs) fit weights and calibration; **8 held-out jobs (80 pairs)** are reported below (5 EN + 3 VI) |
| Labels | 0–4 by two independent blind annotators (A = Claude Opus, B = Claude Sonnet) following `eval/RUBRIC.md`; ground truth = mean of A and B |
| Model | `paraphrase-multilingual-MiniLM-L12-v2` (ONNX, int8, 235 MB), local CPU, no network at runtime |

### 1.1 Annotator agreement (all 240 pairs)

Spearman 0.947 · quadratic-weighted κ 0.947 · exact match 82.1 % · within ±1: **100 %**. Mean label by sampling stratum (hidden from annotators): A — relevant 2.46 / adjacent 1.12 / unrelated 0.33; B — 2.32 / 1.00 / 0.21. The labels therefore track relevance, but **both annotators are LLMs using the same rubric, so their errors are probably correlated**; agreement is an upper bound on label reliability, not proof of it (see §5).

## 2. Headline result (held-out jobs, n = 80 pairs)

| Metric | Value | Target (SPEC §1.3) |
|---|---|---|
| Spearman ρ vs. consensus | **0.787** (95 % CI 0.66 – 0.86, job-level bootstrap) | ≥ 0.60 ✔ |
| ρ vs. annotator A / B alone | 0.766 / 0.795 | |
| Kendall τ | 0.640 | |
| AUC, relevant = consensus ≥ 3 | **0.953** | ≥ 0.85 ✔ |
| NDCG@5 (mean over jobs) | 0.871 | |
| Mean score by consensus label 0 / 1 / 2 / 3 / 4 | 4.8 / 14.9 / 33.6 / 65.3 / 64.5 | |

The score separates *unrelated → weak → partial → good* clearly. It does **not** separate *good* (3) from *strong* (4): both average ≈ 65.

## 3. Ablation / baselines (same held-out pairs)

| Variant | ρ (95 % CI) | AUC | NDCG@5 |
|---|---|---|---|
| **Hybrid, fitted (shipped)** | **0.787** (0.66–0.86) | **0.953** | **0.871** |
| Hybrid, prior weights, default calibration | 0.686 (0.58–0.79) | 0.930 | 0.865 |
| Skills only | 0.765 (0.67–0.83) | 0.907 | 0.854 |
| Embedding only (calibrated) | 0.594 (0.44–0.71) | 0.904 | 0.819 |
| Embedding only (raw cosine) | 0.593 (0.43–0.70) | 0.932 | 0.835 |
| TF-IDF cosine | 0.368 (0.09–0.75) | 0.605 | 0.797 |
| Hybrid without negation handling | 0.793 (0.67–0.86) | 0.951 | 0.869 |

Findings, stated plainly:
* The fitted hybrid is best on all three metrics, but its ρ advantage over **skills-only is within the confidence intervals** (0.787 vs 0.765). The clearer gain is AUC (0.953 vs 0.907): the embedding signal fixes relevance for CVs whose skills are outside the taxonomy and for Vietnamese jobs.
* Fitting on labelled data matters: the same components with prior weights score ρ = 0.686.
* **Negation handling shows no measurable effect** on this dataset (0.793 without vs 0.787 with): the sample CVs almost never contain negated skills. Its correctness is covered by unit tests (`tests/test_nlp.py`, 9 positive + 7 false-positive cases), not by this benchmark.
* TF-IDF is a weak baseline, as expected for cross-lingual and paraphrased text.

Fitted parameters (`data/model_params.json`): weights skills 0.481 / semantic 0.433 / experience 0.087 / education 0.000; calibration same-language [0.401, 0.671], cross-language (VI job vs EN CV) [0.457, 0.570]. Education received weight 0 (no predictive value on this data); it is still shown in the breakdown for explanation.

Per held-out job ρ: java-backend 0.92 · vi-python-backend 0.93 · vi-ky-su-dien 0.85 · accountant 0.79 · bi-analyst 0.74 · vi-ke-toan 0.70 · graphic-designer 0.49 · **digital-marketing 0.27**. Weakest on non-IT creative/marketing roles, where the skill taxonomy is thin and CV vocabulary is varied.

## 4. Development history (no tuning on the held-out set)

| Run | Change | ρ | AUC |
|---|---|---|---|
| 1 | first implementation, fitted weights | 0.727 | 0.944 |
| 2 | down-weight generic soft skills; wider, per-language-pair calibration; `crm` alias fix (found by error analysis on the **train** split only) | 0.781 | 0.951 |
| 3 | "A or B" requirement groups (found by running the service on real input: "Django or Flask" had been scored as requiring both) | 0.787 | 0.953 |

The held-out numbers were computed after every run and are shown for transparency; decisions were taken from train-split errors and from manual inspection of service output, not by searching for held-out gains. With only 8 held-out jobs the estimate is still noisy (CI width ≈ 0.2).

## 5. Limitations (read before quoting numbers)

1. **Labels are LLM-generated, not by recruiters.** Results measure agreement with LLM judgement. A human spot-check of ≥ 20 pairs by a team member is recommended before the defence.
2. English CVs only (the available sample). Vietnamese CVs are covered by unit/API tests and by cross-language *job* evaluation (3 of 8 held-out jobs are Vietnamese), not by a Vietnamese-CV benchmark.
3. 24 jobs / 240 pairs is small; the CI is wide. Weights were fitted on only 16 jobs.
4. Only one embedding model was evaluated (`paraphrase-multilingual-MiniLM-L12-v2`); the larger multilingual models are future work (their download size is several times larger).
5. The skill taxonomy (≈ 150 skills, ≈ 700 aliases) is hand-built and IT/business-weighted.
6. Hidden-text defence is tested on generated PDFs/DOCX (`tests/test_parsing.py`), not on the scanned benchmark (scans contain no hidden text layer).
7. Scores are advisory. No automated hiring decision should rely on them.

## 6. Service performance (dev machine, 12 cores, 30 distinct scanned 1-page CVs, sequential)

| | p50 | p95 | max |
|---|---|---|---|
| Cold request (parse + OCR + score) | 2.04 s | 4.29 s | 6.56 s |
| Repeat of same CV + job (Redis cache) | 6 ms | 9 ms | |

Target p95 < 10 s for scanned 1-page CVs: met on this machine. OCR is single-threaded per request (`OMP_THREAD_LIMIT=1`), so per-request time on a 2-vCPU VM should be similar, but throughput is limited to `OCR_CONCURRENCY` parallel requests (default 2). To be re-measured on the deployment VM.

## 7. Reproduce

```
mise run models        # one-off, downloads the ONNX model
mise run test          # 66 unit/API/robustness tests, no model needed
mise run eval          # fits on train jobs, evaluates held-out, rewrites results + data/model_params.json
python eval/agreement.py
python eval/latency.py --url ... --token ... --job ...
```
