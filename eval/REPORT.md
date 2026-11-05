# Resume scoring — evaluation report (SCORE-01)

Generated from `eval/fit_grid.py` (parameters in `data/model_params.json`, raw numbers in
`eval/results_fit_grid.json`, `eval/results_full_*.json`). Spec: `job-platform-docs/docs/ai/resume-scoring.md`.

## 1. Setup

### 1.1 Two benchmarks, one dataset

| | Benchmark A (before) | Benchmark B (now) |
|---|---|---|
| CVs | 240 OCR texts (git-ignored, `eval/cache/cvs`) | same 240 texts |
| Jobs | 24 postings (16 train / 8 held-out) | same 24 postings |
| Pairs | 240 (10 CVs per job) | **5 760** (every CV × every job) |
| Labels | 2 blind annotators (A: Opus, B: Sonnet), consensus | + **Haiku labels for all 5 760 pairs** and an **independent Sonnet cross-check on 292 same-domain pairs** |
| Source | `eval/labels/A`, `eval/labels/B` | `eval/labels/grid/` (exported from the local `enfue-eval` dataset — same CVs, same jobs, byte-identical rubric) |

Benchmark B is *the same data* as A, just labelled far more densely: the 240 A/B pairs are a subset
of it and their labels match exactly. It is 24× more labelled pairs, which is what makes the
held-out numbers below meaningful (per-job n = 240 instead of 10) and makes it possible to fit on
one label set and verify on two others.

Labels are LLM rubric judgements (0–4), not recruiter decisions. On the 240 pairs that carry both
annotations, the Haiku labels agree with the A/B consensus at **ρ = 0.90** — the label sets are
related but not identical, so every table below reports all three.

The CV texts stay on the machine (`eval/cache/cvs` is git-ignored). Only the labels (integers)
are committed. When the NVIDIA embedding API was tested, the CV text was scrubbed of the
name/contact header, e-mails, phones and links before it left the machine, and the local models
scored the *same* scrubbed text, so the comparison is apples-to-apples.

### 1.2 Protocol

* Split by job: **16 train jobs** fit the weights, the cosine window and the score map; the
  **8 held-out jobs** (1 920 grid pairs) carry every reported number.
* Metrics: Spearman ρ (job-level bootstrap CI), AUC (≥2, ≥3), NDCG@5 per job, and MAE against a
  fixed reference scale (0→5, 1→25, 2→45, 3→70, 4→85).

## 2. Headline result

### 2.1 Held-out jobs (8 jobs, 1 920 grid pairs)

| Metric | Before (MiniLM, 240-pair fit, no score map) | **Now (MiniLM, 5 760-pair fit + score map)** |
|---|---|---|
| ρ vs. Haiku labels | 0.5938 | 0.5947 |
| ρ vs. A/B consensus (80 pairs) | 0.7888 | **0.7890** |
| ρ vs. Sonnet cross-check (110 pairs) | 0.6473 | **0.6583** |
| MAE vs. reference scale | 10.78 | 10.74 → **8.22** with the score map |
| Mean score by label 0…4 | 2.5 / 12.1 / 32.6 / 54.9 / 78.9 | **8.9 / 16.1 / 29.2 / 46.8 / 69.7** (reference 5 / 25 / 45 / 70 / 85) |

### 2.2 Full grid (24 jobs, 5 760 pairs)

| Metric | Before | **Now** |
|---|---|---|
| ρ vs. Haiku labels | 0.5911 (CI 0.538–0.636) | **0.5924** (CI 0.541–0.635) |
| AUC ≥ 2 | 0.9162 | 0.9153 |
| AUC ≥ 3 | 0.9597 | **0.9601** |
| NDCG@5 | 0.8520 | 0.8515 |
| ρ, in-domain pairs only (768) | 0.6751 | **0.6812** |
| ρ vs. Sonnet cross-check (292) | 0.6386 | **0.6485** |
| Mean score by label 0…4 | 4.0 / 14.9 / 36.1 / 55.7 / 73.6 | **9.5 / 17.7 / 32.6 / 48.8 / 67.6** |

**Honest reading:** the *ranking* is flat (ρ, AUC and NDCG@5 move by ≤ 0.001 on the full grid);
what the new data buys is the **absolute scale** — a "strong" CV now averages 68/100 instead of
74/100 with the wrong shape (before, label 1 and label 4 were only 10 points apart; now the five
levels are spread over 9…68 instead of 4…74), and the MAE against the rubric drops by 24 %.
In-domain ρ (+0.006) and the independent Sonnet cross-check (+0.010) move up slightly.

### 2.3 Why the embedding model was *not* changed

`paraphrase-multilingual-mpnet-base-v2` was fitted and measured on the same data (§5): it buys
**+0.028 ρ** on the held-out jobs (paired bootstrap CI [+0.005, +0.050], P = 0.998) but nothing on
AUC/NDCG@5 (±0.01, inside the noise). It costs **+780 MB of RAM** (1.60 GB vs 0.83 GB resident) and
**3× the embedding latency** (693 ms vs 221 ms per request on 2 vCPU), which breaks the service's
1 GB container budget (`mem_limit: 1g` in `job-platform-infra/docker-compose.yml`). A 0.03 ρ gain
is not worth doubling the memory and the latency of the whole service, so the shipped model stays
**MiniLM** — the measured headroom is in the features, not in the encoder (§4).

## 3. What changed, and what it bought

| Change | Effect (held-out jobs) |
|---|---|
| **Fit on the 5 760-pair label grid** instead of 240 pairs (weights, cosine window, score map) | ranking flat, but the score map now exists at all: MAE vs. rubric **10.74 → 8.22** |
| **Isotonic score map** (`data/model_params.json`, fitted on the train jobs only) | MAE vs. reference scale −23 %; monotone, so the ranking is unchanged (the flat parts of the curve create the ±0.001 ρ wobble) |
| **Plural alias matching** (`firewalls` → `firewall`, `reconciliations` → `reconciliation`) | correctness (the JD/CV side was silently missing them); metric-neutral |
| **Taxonomy fixes**: `risk management` / `budgeting` no longer map to *auditing* / *financial analysis* (they were false requirements for every project-manager CV); added IDS/IPS, incident response, SDLC, test plan, Redux, month-end closing, social media campaigns, full-cycle recruitment | project-manager JD no longer asks for accounting skills it never wanted; semantic component ρ 0.485 |
| **Weights refitted on the grid** (skills 0.472 · semantic 0.490 · experience 0.038 · education 0.001) | weights are *not* the bottleneck: four different fits land within ±0.002 of each other (see §4) |

## 4. What did **not** work (measured, then dropped)

The extra data made it cheap to test a lot of ideas. These did not survive:

| Idea | Result |
|---|---|
| Re-fitting the weights (grid labels / A/B labels / both) | ρ 0.5931 – 0.5940 — **all identical**. The formula was already at its optimum; the shipped weights were fine. |
| Alternative score formulas (skills-only, geometric mean, relevance gating, skill×semantic product, different weight mixes) | every variant lands in ρ 0.53 – 0.596; the current linear blend is at the plateau |
| New feature families: alias-level requirement coverage (0.471), token coverage (0.271), title fit (0.367), raw years of experience (0.016) | a 4-fold blend of **all** candidate features scored 0.586 — *worse* than the shipped 4-component blend (0.591) |
| Education as a signal | weight 0.001 — a CV from another field still has a degree (the enfue worker measured a −0.14 correlation for the same component) |
| Fitting the cosine window on the grid labels | ρ 0.5756 vs. 0.5933 with the window fitted on the A/B labels: the Haiku labels are harsh, so a window fitted on them is too strict |
| A *linear* score map instead of isotonic | MAE 9.56 / 9.67 vs. 8.22; it compresses the top end (a "strong" CV would land at 56 instead of 68) |

**Conclusion: with a rule-based scorer the remaining headroom is in the *features*, not in the
formula.** The components are ~0.49–0.56 ρ each and the blend adds ~0.05 over the best single
one; the measured ceiling of this approach on 5 760 pairs is ρ ≈ 0.62.

## 5. Embedding providers

All providers scored the same 5 760 pairs, the same PII-scrubbed CV text, a provider-specific
cosine window (fitted on the train jobs) and provider-specific weights (refitted on the train
jobs) — so the table compares the models, not their calibration. Raw numbers:
`eval/results_compare_summary.json`.

| Provider | ρ vs. Haiku (held-out) | ρ vs. A/B | ρ vs. Sonnet | Semantic ρ | RAM / latency | Constraint |
|---|---|---|---|---|---|---|
| **`paraphrase-multilingual-MiniLM-L12-v2` (local ONNX, 224 MB)** | 0.5706 | 0.7633 | 0.6462 | 0.4544 | **830 MB, 221 ms/request** | shipped; fits the 1 GB container |
| `paraphrase-multilingual-mpnet-base-v2` (local ONNX, 1 059 MB) | **0.5980** | 0.7777 | **0.6631** | **0.5158** | 1 600 MB, 693 ms/request | +0.028 ρ, but 2× RAM and 3× latency → breaks the 1 GB budget |
| `nvidia/nemotron-3.5-embed-8b` (NVIDIA NIM API, 4 096-d) | 0.5954 | **0.7997** | **0.6871** | 0.5145 | 1 call per CV chunk | third-party API: violates the "no third-party AI/LLM API" constraint |

Paired bootstrap over the 8 held-out jobs (4 000 resamples of whole jobs):

| Comparison | Δρ | 95 % CI | P(first is better) |
|---|---|---|---|
| mpnet − MiniLM | **+0.0263** | **+0.0054 … +0.0496** | 0.998 |
| NVIDIA − MiniLM | +0.0208 | −0.0489 … +0.0977 | 0.699 |
| NVIDIA − mpnet | −0.0045 | −0.0691 … +0.0594 | 0.564 |

**Recommendation: stay local, stay on MiniLM.** Two independent reasons:

1. *The API is not allowed and does not help.* The frontier NVIDIA encoder is statistically
   indistinguishable from the local mpnet (Δρ = −0.005, CI straddles 0) and even against MiniLM
   the difference is not significant. The reason is visible in the components: the rule-based
   skills component (ρ 0.5842 for every provider — it does not use embeddings) dominates the
   blend, and the *semantic* component saturates at ρ ≈ 0.515 for both mpnet and Nemotron. A
   stronger encoder cannot lift a saturated feature. On top of that, an API breaks the "no
   third-party AI/LLM API" constraint, adds a runtime network dependency, sends CV text to a third
   party and costs money per request.
2. *The local upgrade is not worth its price.* mpnet buys +0.028 ρ and nothing on AUC/NDCG@5 for
   +780 MB RAM and 3× the embedding latency (§2.3).

NVIDIA endpoint notes (measured while running the comparison): a request with **more than one
input** fails with HTTP 500 in 15–20 % of the cases (single-input requests are reliable), the
response is 4 096 float32 dimensions (52 KB as JSON, 22 KB as base64), and a free key sustained
~12–15 texts/s with 6 parallel connections for the 11 128 unique texts (16 min for the whole grid).

## 6. Service performance and hardware requirement

Measured on this service (MiniLM + Tesseract OCR, `OMP_THREAD_LIMIT=1` as in the Docker image):

| | Resident memory | Embedding per request (2 vCPU) | Notes |
|---|---|---|---|
| Python + imports (FastAPI, onnxruntime, pymupdf) | 161 MB | — | fixed overhead |
| + ONNX model loaded | 667 MB | — | model file 224 MB |
| Service, steady state | **≈ 830 MB** | **221 ms median / 1 155 ms p95** | 12 cores: 253 / 1 171 ms |
| + OCR (2 concurrent `tesseract` children) | +58 MB | — | 5-page scanned PDF: 2.04 s |
| Peak (model load + request) | **≈ 900 MB** | — | 2 concurrent 5-page scanned PDFs: 2.69 s/request |

**Minimum requirement: 2 vCPU, 2 GB RAM, 5 GB disk, no GPU.** The 1 GB container limit
(`mem_limit: 1g`) fits with ~100 MB of headroom. The Azure VM `Standard_B2ms` (2 vCPU / 8 GB)
is comfortable. Cold request (parse + OCR + score) is p50 ≈ 2 s / p95 ≈ 4 s, inside the < 10 s
target. Because ONNX runs single-threaded here, extra cores do not help — 2 vCPU is enough.

With mpnet the same service needs 1.60 GB steady / 2.0 GB peak, i.e. the limit would have to move
to 2–3 GB.

## 7. Limitations

1. **Labels are LLM judgements.** Three label sets agree at ρ 0.85–0.90 with each other; a human
   spot-check (≥ 20 pairs) is still recommended before the defence.
2. The grid is 87 % out-of-domain (every CV against every job), which is realistic for ranking a
   pool but not for a single CV's absolute score; the absolute score map is fitted on the A/B
   pairs (stratified relevant/adjacent/unrelated), the ranking metrics on the grid.
3. CVs are OCR text, not scanned PDFs; English CVs only, Vietnamese jobs (8 of 24).
4. The skill taxonomy (≈ 160 skills, ≈ 750 aliases) is hand-built and still IT-weighted; the
   weakest jobs are the non-IT ones (civil-engineer, teaching, project management).
5. Scores are advisory. No automated hiring decision should rely on them.
6. The shipped parameters pick the weight set and the score-map source with a rule that is fixed in
   `eval/fit_grid.py` (`--weights grid`, map = grid+A/B), not by peeking at the held-out jobs;
   the candidates differ by ≤ 0.002 ρ anyway.

## 8. Reproduce

```bash
mise run models        # one-off, downloads the ONNX model (~235 MB for MiniLM)
mise run test          # 79 unit/API/robustness tests, no model needed
mise run eval          # 240-pair ablation (does not write data/model_params.json)
mise run eval-grid     # the 5 760-pair fit -> data/model_params.json (shipped params)
python eval/run_full.py --labels eval/labels/grid/haiku_5760.json \
                        --sonnet eval/labels/grid/sonnet_292.json --tag shipped   # full-grid report
python eval/features_grid.py                         # feature exploration (single-feature ρ, blends)
python eval/compare_embedders.py                     # provider comparison (NVIDIA_API_KEY optional)
python eval/compare_stats.py                         # merge the provider runs + paired bootstrap
python eval/agreement.py
python eval/latency.py --url ... --token ... --job ...
```
