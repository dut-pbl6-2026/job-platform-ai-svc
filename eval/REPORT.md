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

| Metric | Before (MiniLM, 240-pair fit) | **Now (mpnet, 5 760-pair fit + score map)** |
|---|---|---|
| ρ vs. Haiku labels | 0.5938 | **0.6216** (0.6188 after the score map) |
| ρ vs. A/B consensus (80 pairs) | 0.7888 | 0.7894 (**0.8000** after the score map) |
| ρ vs. Sonnet cross-check (110 pairs) | 0.6473 | **0.6701** (0.6662 after the score map) |
| MAE vs. reference scale | 10.78 | 10.61 raw → **8.14** with the score map |
| Mean score by label 0…4 (reference 5 / 25 / 45 / 70 / 85) | 4.0 / 14.2 / 33.6 / 53.5 / 75.2 | 2.5 / 12.1 / 32.6 / 54.9 / 78.9 raw → **8.4 / 19.8 / 37.7 / 57.2 / 78.5** with the map |

("Before" = MiniLM with its own fitted window, the protocol of §5; "now" = mpnet with the shipped
window, before and after the score map.)

The score map is monotone, so it cannot change the ranking; it only lines the score up with the
rubric (the small ρ differences come from the flat parts of the isotonic curve, which create ties).

### 2.2 Full grid (24 jobs, 5 760 pairs)

| Metric | Before | **Now** |
|---|---|---|
| ρ vs. Haiku labels | 0.5911 (CI 0.538–0.636) | **0.5989** (CI 0.549–0.642) |
| AUC ≥ 2 | 0.9162 | 0.9075 |
| AUC ≥ 3 | 0.9597 | 0.9586 |
| NDCG@5 | **0.8520** | 0.8333 |
| ρ, in-domain pairs only (768) | 0.6751 | **0.6855** |
| ρ vs. Sonnet cross-check (292) | 0.6386 | **0.6526** |
| Mean score by label 0…4 | 4.0 / 14.9 / 36.1 / 55.7 / 73.6 | **10.2 / 22.5 / 40.8 / 58.1 / 76.6** |

Honest reading: the *ranking* improves a little (ρ, in-domain ρ, Sonnet ρ) and the *absolute
scale* improves a lot (the label means move towards the rubric: a "strong" CV used to average
74/100, now 77, and a "weak" one 4 → 10), but NDCG@5 and AUC dip slightly. The grid is 87 %
out-of-domain, which rewards a score that separates the whole scale; NDCG@5 only looks at the
top-5 of each job, where the improvement is not visible.

## 3. What changed, and what it bought

| Change | Effect (held-out jobs) |
|---|---|
| **Embedding model** `paraphrase-multilingual-mpnet-base-v2` instead of `…-MiniLM-L12-v2` | ρ +0.0275 (paired bootstrap, job clusters: 95 % CI **+0.0065 … +0.0509**, P(better) = 0.998); semantic component ρ 0.485 → 0.520. Cost: model 241 MB → 860 MB, embedding 253 ms → 772 ms per request (2 vCPU: 221 ms → 693 ms median) |
| **Score map** (isotonic, fitted on the train jobs) | MAE vs. reference scale 10.61 → **8.14** (−23 %); ranking unchanged (monotone) |
| **Plural alias matching** (`firewalls` → `firewall`, `reconciliations` → `reconciliation`) | correctness (the JD/CV side was silently missing them); metric-neutral |
| **Taxonomy fixes**: `risk management` / `budgeting` no longer map to *auditing* / *financial analysis* (they were false requirements for every project-manager CV); added IDS/IPS, incident response, SDLC, test plan, Redux, month-end closing, social media campaigns, full-cycle recruitment | project-manager JD no longer asks for accounting skills it never wanted |
| **Weights refitted on 5 760 pairs** (skills 0.494 · semantic 0.490 · experience 0.010 · education 0.006) | weights are *not* the bottleneck: four different fits land within ±0.002 of each other (see §4) |

## 4. What did **not** work (measured, then dropped)

The extra data made it cheap to test a lot of ideas. These did not survive:

| Idea | Result |
|---|---|
| Re-fitting the weights (grid labels / A/B labels / both) | ρ 0.6207 – 0.6216 — **all identical**. The formula was already at its optimum; the shipped weights were fine. |
| Alternative score formulas (skills-only, geometric mean, relevance gating, skill×semantic product, different weight mixes) | every variant lands in ρ 0.53 – 0.596; the current linear blend is at the plateau |
| New feature families: alias-level requirement coverage (0.471), token coverage (0.271), title fit (0.367), raw years of experience (0.016) | a 4-fold blend of **all** candidate features scored 0.586 — *worse* than the shipped 4-component blend (0.591) |
| Education as a signal | weight 0.006 — a CV from another field still has a degree (the enfue worker measured a −0.14 correlation for the same component) |
| Fitting the cosine window on the grid labels | ρ 0.6143 vs. 0.6216 with the window fitted on the A/B labels: the Haiku labels are harsh, so a window fitted on them is too strict |
| A *linear* score map instead of isotonic | MAE 9.27 (grid-fitted) / 9.40 (A/B-fitted) vs. 8.14; it compresses the top end (a "strong" CV would land at 66 instead of 79) |

**Conclusion: with a rule-based scorer the remaining headroom is in the *features*, not in the
formula.** The components are ~0.52–0.56 ρ each and the blend adds ~0.05 over the best single
one; the measured ceiling of this approach on 5 760 pairs is ρ ≈ 0.62.

## 5. Embedding providers

All providers scored the same 5 760 pairs, the same PII-scrubbed CV text, a provider-specific
cosine window (fitted on the train jobs) and provider-specific weights (refitted on the train
jobs) — so the table compares the models, not their calibration. Raw numbers:
`eval/results_compare_summary.json`.

| Provider | ρ vs. Haiku (held-out) | ρ vs. A/B | ρ vs. Sonnet | Semantic ρ | Cost / constraint |
|---|---|---|---|---|---|
| `paraphrase-multilingual-MiniLM-L12-v2` (local ONNX, 241 MB) | 0.5706 | 0.7633 | 0.6462 | 0.4544 | shipped before; fastest (253 ms/request) |
| **`paraphrase-multilingual-mpnet-base-v2` (local ONNX, 860 MB)** | **0.5980** | 0.7777 | 0.6631 | 0.5158 | local, no network at runtime; 3× slower embedding, +620 MB image |
| `nvidia/nemotron-3.5-embed-8b` (NVIDIA NIM API, 4 096-d) | 0.5954 | **0.7997** | **0.6871** | 0.5145 | third-party API: violates the "no third-party AI/LLM API" constraint, needs the network at runtime, sends CV text to a third party |

Paired bootstrap over the 8 held-out jobs (4 000 resamples of whole jobs):

| Comparison | Δρ | 95 % CI | P(first is better) |
|---|---|---|---|
| mpnet − MiniLM | **+0.0263** | **+0.0054 … +0.0496** | 0.998 |
| NVIDIA − MiniLM | +0.0208 | −0.0489 … +0.0977 | 0.699 |
| NVIDIA − mpnet | −0.0045 | −0.0691 … +0.0594 | 0.564 |

**Recommendation: stay local, move to MiniLM → mpnet; do not use NVIDIA.** The frontier API buys
nothing here: its ranking is statistically indistinguishable from the local mpnet (Δρ = −0.005,
CI straddles 0), and even against the old MiniLM the difference is not significant. The reason is
visible in the components: the rule-based skills component (ρ 0.5842, identical for every provider
because it does not use embeddings) dominates the blend, and the *semantic* component saturates at
ρ ≈ 0.515 for both mpnet and Nemotron — a stronger encoder cannot lift a saturated feature.

Switching to NVIDIA would also add a third-party dependency (constraint), a runtime network
dependency, PII leaving the machine, and a per-request cost of one API call per CV chunk.

NVIDIA endpoint notes (measured while running the comparison): a request with **more than one
input** fails with HTTP 500 in 15–20 % of the cases (single-input requests are reliable), the
response is 4 096 float32 dimensions (52 KB as JSON, 22 KB as base64), and a free key sustained
~12–15 texts/s with 6 parallel connections for the 11 128 unique texts (16 min for the whole grid).


## 6. Service performance (dev machine, 12 cores; and 2 vCPU = deployment VM)

| | MiniLM | mpnet |
|---|---|---|
| Embedding per request (median / p95, 12 cores) | 253 ms / 1 171 ms | 772 ms / 3 090 ms |
| Embedding per request (median / p95, 2 vCPU) | 221 ms / 1 155 ms | 693 ms / 3 264 ms |
| Model size on disk | 241 MB | 860 MB |

Cold request (parse + OCR + score) was p50 2.04 s / p95 4.29 s with MiniLM; mpnet adds ≈ 0.5 s
(median) to ≈ 2 s (p95), so the p95 stays inside the < 10 s target but has less headroom on a
2 vCPU VM. Re-measure on the deployment VM before switching the default.

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

## 8. Reproduce

```bash
mise run models        # one-off, downloads the ONNX model (~860 MB for mpnet)
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
