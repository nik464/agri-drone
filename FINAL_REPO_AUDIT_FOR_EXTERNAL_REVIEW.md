# FINAL REPO AUDIT FOR EXTERNAL REVIEW

**Auditor role:** adversarial third-party research auditor.
**Evidence basis:** file system only (no execution). All claims cite exact paths.
**Date of audit:** 2026-04-20.

---

## § 1 Repository identity

| Field | Value |
|---|---|
| Repo root | `d:\Projects\agri-drone` |
| Declared name | `agri-drone` |
| Declared upstream | `github.com/Ashut0sh-mishra/agri-drone` (per [PAPER1_NEGATIVE_RESULT.md](agri-drone/PAPER1_NEGATIVE_RESULT.md)) |
| Licence | MIT ([LICENSE](agri-drone/LICENSE)) |
| Citation file | [CITATION.cff](agri-drone/CITATION.cff) present |
| Declared status | "Research Prototype" ([README.md](agri-drone/README.md#L16)) |
| Primary languages | Python (src package + evaluate + scripts + notebooks), React/Vite (separate folder `agri-drone-frontend/`) |
| Git SHA recorded in artefacts | `7d34d28` (in [evaluate/results/v2/matrix/full-matrix-v4/run_metadata.json](agri-drone/evaluate/results/v2/matrix/full-matrix-v4/run_metadata.json)) |

**Auditor's note.** The repo is a personal research prototype by a single author. Nothing in the tree suggests CI-enforced reviews, external contributors, or institutional affiliation.

---

## § 2 High-level project map

Top-level structure of [d:\Projects\agri-drone](agri-drone/):

```
agri-drone/
├── src/agridrone/                 # Python package (17 subdirs incl. vision, knowledge, api)
├── evaluate/                      # 20+ evaluation scripts + results/
│   ├── ablation_study.py          # Experiment 1 — configs A/B/C/D
│   ├── statistical_tests.py       # Bootstrap CIs + McNemar
│   ├── eml_analysis.py            # Expected monetary loss
│   ├── cross_dataset_eval.py / pdt_cross_eval.py
│   ├── sensitivity_analysis.py    # 125-config weight sweep
│   ├── matrix/                    # run_matrix.py, train.py, audit_baseline.py
│   └── results/                   # v1 artefacts + v2/{matrix,eml,pdt,statistics,baseline_audit}
├── configs/                       # base, model, data, sim, prescription, actuation, inference
│   ├── matrix/                    # smoke / quick / large / full / paper2 YAMLs
│   └── economics/india_2025.yaml
├── notebooks/                     # colab/01_run_matrix.ipynb, kaggle/, Train_YOLOv8_Colab_GPU.ipynb
├── scripts/                       # ~30 scripts incl. make_splits.py, train_model.py
├── tests/                         # unit/ (3 files), integration/ (empty except __init__)
├── docs/                          # training_recipe.md, data_availability.md, results_schema.md
├── data/                          # raw, processed, interim, wheat_annotated, wheat_raw, yolo_raw
├── models/                        # 6 .pt files (yolov8n-cls, efficientnet_b0 21-class, …)
├── frontend/ + dashboard/         # React/Vite (also duplicated in sibling ../agri-drone-frontend/)
├── Dockerfile + docker-compose.yml
└── 9 × RESEARCH_PAPER*.md + PAPER1_NEGATIVE_RESULT.md + ~20 other status markdowns
```

**Counts.**
- 29 markdown files at repo root (9 of which are paper variants; the rest are status / handoff / checklist files).
- 51 files under [evaluate/results/](agri-drone/evaluate/results/) (JSON / CSV / PNG / LaTeX).
- 9 stray `test_*.py` at repo root AND at workspace root (`d:\Projects\test_wheat_model.py`, `test_models.py`, etc.) → these are scratch scripts outside [tests/](agri-drone/tests/).

---

## § 3 What research question the repo is actually answering

Reading the code and artefacts (not the marketing), the repo answers **one narrow question very well and one broad question poorly**:

**Well-answered (v1, executed):**
> On a single 934-image / 21-class curated leaf-photo test set with a YOLOv8n-cls backbone trained once at seed 42, does a hand-authored HSV + texture + spatial rule re-ranker add measurable value over softmax-argmax?

The pipeline for this question is fully reproducible from CSVs: [predictions_A_yolo_only.csv](agri-drone/evaluate/results/predictions_A_yolo_only.csv), [predictions_B_yolo_rules.csv](agri-drone/evaluate/results/predictions_B_yolo_rules.csv), [predictions_C_rules_only.csv](agri-drone/evaluate/results/predictions_C_rules_only.csv) (each contains exactly 934 rows + header).

**Poorly answered (v2/v4, unexecuted):**
> Does the answer generalise across backbones (YOLO/EfficientNet/ConvNeXt/MobileNet/ViT), datasets (PlantVillage / PDT / PlantDoc / Indian-21), and train fractions?

The entire v2/v4 matrix is a **dry-run stub** (see § 6). The repo *defines* this experiment; it has not *executed* it.

**Claimed but not supported by code:** "field-ready drone deployment." No drone-altitude imagery is used anywhere ([docs/data_availability.md#L14-L19](agri-drone/docs/data_availability.md)).

---

## § 4 Manuscript inventory

| File | Bytes | Last modified | Role | Verdict |
|---|---:|---|---|---|
| [RESEARCH_PAPER_CURRENT.md](agri-drone/RESEARCH_PAPER_CURRENT.md) | 34 606 | 2026-04-20 | Active working draft (negative-result reframe, §5.5 McNemar + §5.6 per-class + §5.7 threats) | **Current** |
| [PAPER1_NEGATIVE_RESULT.md](agri-drone/PAPER1_NEGATIVE_RESULT.md) | 11 509 | 2026-04-17 | Short workshop-length version of the same negative result (EML crossover at r=18.7) | **Parallel draft — conflicting numbers** |
| [RESEARCH_PAPER_v4.md](agri-drone/RESEARCH_PAPER_v4.md) | 59 920 | 2026-04-17 | Prior "multi-architecture matrix" narrative | **Superseded, still in tree** |
| [RESEARCH_PAPER_FINAL_v3.md](agri-drone/RESEARCH_PAPER_FINAL_v3.md) | 54 104 | 2026-04-13 | Earlier "final" — cited from README badge | **Superseded** |
| [RESEARCH_PAPER_FINAL_v2.md](agri-drone/RESEARCH_PAPER_FINAL_v2.md) | 62 595 | 2026-04-11 | — | **Stale** |
| [RESEARCH_PAPER_FINAL.md](agri-drone/RESEARCH_PAPER_FINAL.md) | 94 576 | 2026-04-11 | Largest variant | **Stale** |
| [RESEARCH_PAPER_FINAL_old.md](agri-drone/RESEARCH_PAPER_FINAL_old.md) | 43 068 | 2026-04-11 | — | **Stale** |
| [RESEARCH_PAPER_COMPLETE.md](agri-drone/RESEARCH_PAPER_COMPLETE.md) | 45 823 | 2026-04-11 | — | **Stale** |
| [RESEARCH_PAPER.md](agri-drone/RESEARCH_PAPER.md) | 60 741 | 2026-04-09 | Oldest | **Stale** |
| [docs/AgriDrone_paper_final.pdf](agri-drone/docs/AgriDrone_paper_final.pdf) | — | — | A PDF, version unknown | **Opaque — unclear which .md was compiled** |

**Major inconsistencies across drafts:**
- [README.md#L12](agri-drone/README.md) badge links to `RESEARCH_PAPER_FINAL_v3.md`, **not** the current active draft `RESEARCH_PAPER_CURRENT.md`.
- [PAPER1_NEGATIVE_RESULT.md](agri-drone/PAPER1_NEGATIVE_RESULT.md) reports McNemar *p ≈ 0.40*; [RESEARCH_PAPER_CURRENT.md](agri-drone/RESEARCH_PAPER_CURRENT.md) reports **p = 0.134** (matching [evaluate/results/statistical_tests.json#L476](agri-drone/evaluate/results/statistical_tests.json)). One of the two papers cites the wrong number.
- PAPER1 says "test n = 935"; CURRENT says 935; CSVs contain **934** rows; [eml_summary.json](agri-drone/evaluate/results/eml_summary.json) says `n_test_samples: 934`. **Off-by-one error is in the papers, not the code.**

**Verdict:** nine competing paper files with non-identical numerical claims is an immediate red flag for any reviewer. Only `RESEARCH_PAPER_CURRENT.md` should remain; the others belong in `docs/archive/` or deleted.

---

## § 5 Experiment inventory

| Experiment | Script | Config artefact | Output artefact | Executed? |
|---|---|---|---|---|
| E1: Ablation A/B/C (± D stub) | [evaluate/ablation_study.py](agri-drone/evaluate/ablation_study.py) | `severity_tiers` hard-coded inline | [predictions_A/B/C_*.csv](agri-drone/evaluate/results/), [ablation_summary.json](agri-drone/evaluate/results/ablation_summary.json), [confusion_matrix_*.png](agri-drone/evaluate/results/) | **YES — real numbers** |
| E2: Bootstrap CI + McNemar | [evaluate/statistical_tests.py](agri-drone/evaluate/statistical_tests.py) | n_boot = 10 000 (CLI default) | [statistical_tests.json](agri-drone/evaluate/results/statistical_tests.json) | **YES** |
| E3: Cross-dataset on PDT | [evaluate/pdt_cross_eval.py](agri-drone/evaluate/pdt_cross_eval.py) | — | [cross_dataset_PDT.json](agri-drone/evaluate/results/cross_dataset_PDT.json) (acc 0.8438) | **YES** |
| E4: 125-config weight sensitivity | [evaluate/sensitivity_analysis.py](agri-drone/evaluate/sensitivity_analysis.py) | — | [sensitivity_results.csv](agri-drone/evaluate/results/) (referenced by paper_tables.py) | **Likely yes** (referenced but not spot-checked) |
| E5: EML headline | [evaluate/eml_analysis.py](agri-drone/evaluate/eml_analysis.py) | [configs/economics/india_2025.yaml](agri-drone/configs/economics/india_2025.yaml) | [eml_summary.json](agri-drone/evaluate/results/eml_summary.json) — 294.33 vs 2769.06 INR/ha | **YES (v1)** |
| E5b: EML headline (v4, 7-disease primary-source subset) | [evaluate/eml_sensitivity.py](agri-drone/evaluate/eml_sensitivity.py) | same | [v2/eml/headline_v4.json](agri-drone/evaluate/results/v2/eml/headline_v4.json) — **317.89** INR/ha | **YES** |
| E6: Robustness (noise pipeline) | [evaluate/noise_pipeline.py](agri-drone/evaluate/noise_pipeline.py), [evaluate/robustness_eval.py](agri-drone/evaluate/robustness_eval.py) | — | [robustness_summary.json](agri-drone/evaluate/results/robustness_summary.json), [predictions_noisy.csv](agri-drone/evaluate/results/), [confusion_matrix_noisy.png](agri-drone/evaluate/results/) | **YES** |
| E7: Holm-Bonferroni per-class McNemar (v2) | — (output-only) | — | [v2/statistics/holm_bonferroni_mcnemar.json](agri-drone/evaluate/results/v2/statistics/holm_bonferroni_mcnemar.json) | **SUSPECT — see § 11** |
| E8: Friedman-Nemenyi across cells (v2) | — | requires populated matrix | [v2/statistics/friedman_nemenyi.json](agri-drone/evaluate/results/v2/statistics/friedman_nemenyi.json) | **Unverified** |
| E9: Dietterich 5×2cv (v2) | — | — | [v2/statistics/dietterich_5x2cv.json](agri-drone/evaluate/results/v2/statistics/dietterich_5x2cv.json) | **Stub — `p_value: null`** |
| E10: Per-class bootstrap CI (A, B) | — | — | [v2/statistics/per_class_bootstrap_ci_{A,B}.json](agri-drone/evaluate/results/v2/statistics/) | **YES** |
| E11: PDT threshold sweep + few-shot (v2) | [evaluate/pdt_v2.py](agri-drone/evaluate/pdt_v2.py) | — | [v2/pdt/threshold_sweep.json](agri-drone/evaluate/results/v2/pdt/threshold_sweep.json), [calibration.json](agri-drone/evaluate/results/v2/pdt/calibration.json), [few_shot_10.json](agri-drone/evaluate/results/v2/pdt/few_shot_10.json) | **YES — but damning** (see § 6) |
| E12: Full matrix v4 (6×4×5×4×5 = 2400 cells) | [evaluate/matrix/run_matrix.py](agri-drone/evaluate/matrix/run_matrix.py) | [configs/matrix/full.yaml](agri-drone/configs/matrix/full.yaml) | [v2/matrix/full-matrix-v4/](agri-drone/evaluate/results/v2/matrix/full-matrix-v4/) | **DRY-RUN ONLY** |
| E13: Shared-recipe EfficientNet-B0 fair audit | [evaluate/matrix/audit_baseline.py](agri-drone/evaluate/matrix/audit_baseline.py) | docs/training_recipe.md@v1 | [v2/baseline_audit/efficientnet_b0_fair_v1/baseline_audit.json](agri-drone/evaluate/results/v2/baseline_audit/efficientnet_b0_fair_v1/baseline_audit.json) | **DRY-RUN ONLY (`status: dry-run`, `metrics: null`)** |
| E14: Paper-2 matrix (paper2.yaml, 45-cell) | — (Colab runner) | [configs/matrix/paper2.yaml](agri-drone/configs/matrix/paper2.yaml), [notebooks/colab/01_run_matrix.ipynb](agri-drone/notebooks/colab/01_run_matrix.ipynb) | none | **Not executed** |
| E15: LLaVA latency / validator | [test_llava_latency.py](agri-drone/test_llava_latency.py), [evaluate/llava_eval.py](agri-drone/evaluate/llava_eval.py) | — | [llava_analysis.csv](agri-drone/evaluate/results/llava_analysis.csv) | **Partially executed** |

**Scorecard.** ~7 of 15 experiments have real, populated outputs. The "reproducibility matrix" that the paper leans on is unexecuted.

---

## § 6 Results artifact audit

| Artefact | Path | Status | Matches paper? | Evidence |
|---|---|---|---|---|
| Ablation predictions A/B/C | [evaluate/results/predictions_*.csv](agri-drone/evaluate/results/) | **Populated** | **Mostly — n=934 in CSVs, paper claims 935** | `wc -l` = 935 incl. header → 934 rows |
| Ablation summary | [ablation_summary.json](agri-drone/evaluate/results/ablation_summary.json) | Populated | **Yes** (0.9615 / 0.9572 / 0.1338 all match) | direct read |
| Bootstrap CI + McNemar (paired A/B) | [statistical_tests.json#L476](agri-drone/evaluate/results/statistical_tests.json) | Populated | **Yes** (p=0.133614) | direct read |
| Stand-alone McNemar | [mcnemar.json](agri-drone/evaluate/results/mcnemar.json) | **Broken stub** — only 21 observations (`both_right=12, n_discordant=7, both_wrong=2`), p=1.0 | **Contradicts paper** | direct read |
| Per-class F1 delta | inside [ablation_summary.json#B_over_A_F1_delta](agri-drone/evaluate/results/ablation_summary.json) | Populated | **Yes** (tan_spot −0.0383 etc. match §5.6) | direct read |
| EML v1 | [eml_summary.json](agri-drone/evaluate/results/eml_summary.json) | Populated | **Mostly** — n=934 (paper 935); `default_miss_cost=5000` but per-disease critical classes do carry `cost_miss=12000`, so paper's §4.3 cost-table claim is partially supported. | direct read |
| EML v4 headline | [v2/eml/headline_v4.json](agri-drone/evaluate/results/v2/eml/headline_v4.json) | Populated: **317.89** INR/ha (7 diseases, primary-source subset) | **Not cited by current paper**; paper uses v1's 294.33 | direct read |
| EML sensitivity tornado | [v2/eml/sensitivity_tornado.json](agri-drone/evaluate/results/v2/eml/sensitivity_tornado.json) | Populated | Not yet cited | — |
| Cross-dataset PDT (v1) | [cross_dataset_PDT.json](agri-drone/evaluate/results/cross_dataset_PDT.json) | Populated, acc 0.8438 | **Yes** (README 84.4%) | direct read |
| PDT v2 threshold sweep | [v2/pdt/threshold_sweep.json](agri-drone/evaluate/results/v2/pdt/threshold_sweep.json) | Populated: ROC-AUC 0.7086; **honest note: "At τ=argmax the model collapses to constant 'unhealthy'"** | **Embarrassing for any 'domain transfer' claim** — if the paper leans on PDT as cross-dataset evidence without this caveat it is overselling. | direct read |
| Robustness / noise | [robustness_summary.json](agri-drone/evaluate/results/robustness_summary.json) | Populated | Presumably yes | partial read |
| EfficientNet v3 baseline | [efficientnet_results.json](agri-drone/evaluate/results/efficientnet_results.json) | Populated, acc 0.7615 under **unequal** recipe | Paper v3 cites 76.15%; the "shared-recipe fair audit" that would redeem it is **unexecuted** | direct read |
| EfficientNet v4 fair audit | [v2/baseline_audit/efficientnet_b0_fair_v1/baseline_audit.json](agri-drone/evaluate/results/v2/baseline_audit/efficientnet_b0_fair_v1/baseline_audit.json) | **`status: "dry-run", metrics: null`** | Paper v4 implies this audit exists; it does not. | direct read |
| Full matrix v4 aggregate | [v2/matrix/full-matrix-v4/aggregate.json](agri-drone/evaluate/results/v2/matrix/full-matrix-v4/aggregate.json) | **`"status": "dry-run", "n_cells": 2400`**, note "Real aggregates populated after GPU runs finish" | **Flatly contradicts any claim that the 2400-cell matrix has been run** | direct read |
| Full matrix v4 per-run | [v2/matrix/full-matrix-v4/per_run.jsonl](agri-drone/evaluate/results/v2/matrix/full-matrix-v4/per_run.jsonl) | Every row: `"status": "smoke", "metrics": null, "notes": "dry-run: no training executed"` | **Every single row is a stub.** | direct read |
| Holm-Bonferroni (v2) | [v2/statistics/holm_bonferroni_mcnemar.json](agri-drone/evaluate/results/v2/statistics/holm_bonferroni_mcnemar.json) | All 21 classes show `"discordant_pairs": 0, "chi2": 0.0, "p_value": 1.0` | **Contradicts** E1 ablation where A ≠ B. The v2 test was evidently run against identical prediction vectors (A vs A, not A vs B). **Broken.** | direct read |
| Dietterich 5×2cv | [v2/statistics/dietterich_5x2cv.json](agri-drone/evaluate/results/v2/statistics/dietterich_5x2cv.json) | `p_value: null` | Stub | direct read |
| Per-class bootstrap CIs A & B | [v2/statistics/per_class_bootstrap_ci_A.json](agri-drone/evaluate/results/v2/statistics/per_class_bootstrap_ci_A.json), `…_B.json` | Populated | Paper's §5.6 numbers are derivable from these | direct read |
| Paper tables (LaTeX) | [table2_ablation.tex](agri-drone/evaluate/results/table2_ablation.tex), `table3_sensitivity.tex`, `table4_eml.tex` | Populated | — | direct read |

**Worst findings (§ 6):**
1. **E12 full matrix: 100 % dry-run.** Paper 2 / "v4 reproducibility matrix" is vapour.
2. **E13 fair baseline audit: dry-run.** Paper's remedy for the suspiciously low 76.15 % EfficientNet is a placeholder file.
3. **E7 Holm-Bonferroni v2 output is numerically impossible** (zero discordant pairs across all 21 classes while A and B disagree at the aggregate level). This artefact should be regenerated or deleted.
4. **E11 PDT threshold sweep** ships an honest note that the model collapses to constant "unhealthy" at argmax. If any paper cites PDT transfer without this caveat, it is dishonest.

---

## § 7 Code-to-claim traceability

| Manuscript claim (from [RESEARCH_PAPER_CURRENT.md](agri-drone/RESEARCH_PAPER_CURRENT.md)) | Code / artefact | Supported? |
|---|---|---|
| "935-image 21-class test set" | [predictions_A_yolo_only.csv](agri-drone/evaluate/results/predictions_A_yolo_only.csv) (934 rows) | **Off-by-one — actual n = 934** |
| "YOLOv8n-cls baseline 96.15 %" | [ablation_summary.json](agri-drone/evaluate/results/ablation_summary.json) | ✅ exact |
| "McNemar p ≈ 0.134 (A vs B)" | [statistical_tests.json#L476](agri-drone/evaluate/results/statistical_tests.json) (0.133614) | ✅ exact |
| "Bootstrap 10 000 replicates" | CLI default in [statistical_tests.py](agri-drone/evaluate/statistical_tests.py) | ✅ |
| "Config A EML ₹294.33, Config B ₹2 769.06, +840 %" | [eml_summary.json](agri-drone/evaluate/results/eml_summary.json) | ✅ |
| "Critical-disease miss cost ₹12 000 (ICAR-style)" | per-disease `cost_miss=12000` exists in `eml_summary.json`; **top-level `default_miss_cost=5000` is contradictory** | ⚠️ partial — cost-table provenance is inconsistent |
| "Cost crossover ratio r = 18.7" | No file named `eml_comparison.csv` was found on the tree; PAPER1 claims it lives at `evaluate/results/eml_comparison.csv` | ❓ **unverified — artefact missing** |
| "Hand-authored rule engine: HSV gate + lesion shape + crop context (3 stages)" | [src/agridrone/vision/rule_engine.py](agri-drone/src/agridrone/vision/rule_engine.py) implements color + texture (bleaching, spots) + spatial (stripe-vs-spot, stripe-vs-head-disease) — **>3 rule families** | ⚠️ paper under-describes the actual engine; engine is more elaborate than stated |
| "Override-decomposition: 0 rescues, 4 corruptions out of 97 overrides" (§5.3 of CURRENT) | No `override_analysis.json` / `override_decomposition.csv` was found. The CURRENT paper's own addition is not backed by a saved artefact. | ❌ **numbers appear asserted, not derived from a committed file** |
| "54-cell matrix" (Paper 2 / [configs/matrix/large.yaml](agri-drone/configs/matrix/large.yaml)) | Config enumerates 3 × 1 × 2 × 3 × 3 = 54; [configs/matrix/full.yaml](agri-drone/configs/matrix/full.yaml) enumerates 2400. **No v2 run exists for either.** | ❌ defined, not executed |
| "Shared training recipe v1" | [docs/training_recipe.md](agri-drone/docs/training_recipe.md) exists and is internally consistent | ✅ for documentation; ❌ for execution evidence |
| "Pre-registered hypothesis H1" | No ledger, no timestamped hypothesis document, no OSF link | ❌ label-only "pre-registration" |
| "84.4 % PDT cross-dataset accuracy" | [cross_dataset_PDT.json](agri-drone/evaluate/results/cross_dataset_PDT.json) | ✅ but see E11 caveat |
| "FastAPI endpoints `/analyse`, `/api/ml/matrix`, `/api/training/artifacts`" | [src/agridrone/api/](agri-drone/src/agridrone/api/) folder exists (not spot-checked line-by-line here) | ⚠️ claimed — sampling only verifies folder presence |
| "pytest unit & integration tests" | [tests/unit/](agri-drone/tests/unit/) has 3 files; [tests/integration/](agri-drone/tests/integration/) has only `__init__.py` | ❌ integration tests are empty |

**Traceability verdict:** core ablation numbers are faithfully traced to CSVs. Everything above the basic ablation (matrix, fair baseline, override-decomposition, cost crossover, "pre-registration") is **rhetorically claimed without a committed, populated evidence file**.

---

## § 8 Reproducibility audit

| Dimension | Evidence | Score |
|---|---|---|
| Seeds fixed & recorded | `--seed 42` in [scripts/make_splits.py](agri-drone/scripts/make_splits.py); seeds [42,43,44] in [configs/matrix/large.yaml](agri-drone/configs/matrix/large.yaml) | 8/10 |
| Split manifest with hash | [scripts/make_splits.py](agri-drone/scripts/make_splits.py) writes `splits_manifest.json` with SHA-256 of sorted file list | 9/10 |
| Dataset provenance | [docs/data_availability.md](agri-drone/docs/data_availability.md) lists 4 sources; [configs/matrix/large.yaml](agri-drone/configs/matrix/large.yaml) lists different Kaggle slugs + fallbacks. **Two inconsistent provenance stories.** | 4/10 |
| Deterministic data download | "Automatic download is intentionally not implemented" ([data_availability.md#L40](agri-drone/docs/data_availability.md)) | 5/10 |
| Pinned dependencies | [requirements.lock.txt](agri-drone/requirements.lock.txt) exists alongside `requirements.txt`, [pyproject.toml](agri-drone/pyproject.toml) | 8/10 |
| Container | [Dockerfile](agri-drone/Dockerfile), [docker-compose.yml](agri-drone/docker-compose.yml) | 8/10 |
| Hardware spec | T4 targeted in quick.yaml; A100 in large.yaml; [docs/training_recipe.md](agri-drone/docs/training_recipe.md) has full recipe | 7/10 |
| Result schema | [docs/results_schema.md](agri-drone/docs/results_schema.md) referenced; versioned recipe tag `docs/training_recipe.md@v1` written into `per_run.jsonl` | 9/10 (design) / 3/10 (populated) |
| Crash-resumable runner | [evaluate/matrix/run_matrix.py](agri-drone/evaluate/matrix/run_matrix.py) has `execution.resume: true`, `fail_fast: false` | 9/10 |
| **End-to-end reproducible for headline numbers?** | YES for Configs A/B/C ablation (rerun [evaluate/ablation_study.py](agri-drone/evaluate/ablation_study.py) from CSVs). NO for matrix, NO for fair baseline. | mixed |

**Composite reproducibility score: 6 / 10.** The scaffolding is genuinely good (hashing, resumable runner, recipe versioning, containerisation). The *populated* reproducibility is limited to the v1 ablation. The flagship v2 matrix reproducibility is theatre.

---

## § 9 Dataset and data-pipeline audit

**Source reality.** Two inconsistent provenance documents:
- [docs/data_availability.md](agri-drone/docs/data_availability.md) → PlantVillage + PDT + UCI Rice Leaf + Kaggle Rice Pest. Four datasets.
- [configs/matrix/large.yaml](agri-drone/configs/matrix/large.yaml) → PlantVillage + Bangladesh Rice Leaf (loki4514) + Bangladesh Wheat Leaf (rajkumar9999) + 15 fallback Kaggle mirrors. Three datasets + fallbacks.

These disagree on which Kaggle slugs are canonical. A reviewer cannot, from the repo alone, determine which corpus produced the 934 ablation test images.

**Class taxonomy.** 21 classes in [ablation_summary.json](agri-drone/evaluate/results/ablation_summary.json) and [README.md#L70-L74](agri-drone/README.md): 14 wheat diseases/pests + 5 rice diseases + 2 healthy. This is consistent across artefacts.

**Split hygiene — potential leakage.**
- [scripts/make_splits.py](agri-drone/scripts/make_splits.py) shuffles *all files* in each class directory and partitions 70/15/15. **It makes no attempt to group augmented variants of the same base image.**
- The test CSVs contain files like `aug_0_101.jpg`, `aug_0_1149.jpg` ([predictions_A_yolo_only.csv#L2-L3](agri-drone/evaluate/results/predictions_A_yolo_only.csv)).
- The train manifest contains files like `aug_0_1039.jpg`, `aug_0_1082.jpg` ([evaluate/data_split_manifest.json](agri-drone/evaluate/data_split_manifest.json)).
- If these `aug_*` files were generated by off-line augmentation from a shared set of base images **before** the split, then copies of a single source image can land in both train and test. The repo contains no evidence of "group-aware" splitting (e.g., split base IDs first, then augment train only).
- **Risk level: HIGH.** Classic PlantVillage-style leakage is exactly this pattern, and the Mohanty-2016 literature is explicit about it. The paper never addresses it.

**Imbalance.** 934 images across 21 classes → ~44 per class average. [data_split_manifest.json](agri-drone/evaluate/data_split_manifest.json) confirms ~45 test images per class for healthy_rice / bacterial_blight / blast / brown_spot. This is a small, curated evaluation set — reviewers will correctly flag it as inadequate for a 21-class claim.

**Domain validity.** [data_availability.md#L14-L19](agri-drone/docs/data_availability.md) explicitly concedes that no dataset is drone-altitude. The paper's title word "drone" is therefore aspirational. The reframed `RESEARCH_PAPER_CURRENT.md` owns this, which is correct.

---

## § 10 Rule-engine audit

**Structure.** [src/agridrone/vision/rule_engine.py](agri-drone/src/agridrone/vision/rule_engine.py) + [src/agridrone/vision/rule_engine_base.py](agri-drone/src/agridrone/vision/rule_engine_base.py) + [src/agridrone/vision/rules_learned.py](agri-drone/src/agridrone/vision/rules_learned.py) + [src/agridrone/vision/rules_llm.py](agri-drone/src/agridrone/vision/rules_llm.py) + [src/agridrone/vision/feature_extractor.py](agri-drone/src/agridrone/vision/feature_extractor.py).

**Substance.** The engine has proper dataclasses (`RuleMatch`, `Rejection`, `ConflictReport`, `CandidateScore`, `RuleEngineResult`) and distinct rule families evaluated against a per-disease profile loaded from [src/agridrone/knowledge/diseases.json](agri-drone/src/agridrone/knowledge/diseases.json):
- **Color rules** — HSV-signature matching against `profile.color_signatures`.
- **Texture rules** — bleaching-ratio rule, spot/pustule rule conditioned on disease symptoms.
- **Spatial rules** — stripe-vs-spot penalisation using Hough-line directionality; stripe-contradicts-head-disease penalty.
- **Conflict resolution** — `CandidateScore` + `ConflictReport` when YOLO top-1 ≠ rule top-1.

**Verdict: this is NOT a straw-man rule engine.** The paper's negative-result framing ("rules do not help") is therefore a non-trivial result: the engine is substantive enough that its failure carries information. This is the single strongest asset of the repo.

**But note two inaccuracies in the paper's description:**
1. Paper says *3 rule stages*; code has ~5 rule families plus conflict resolution.
2. Paper says thresholds are "tuned on validation"; `diseases.json` appears hand-authored rather than empirically tuned. Reviewer will ask for a validation-set optimisation log that does not exist in the tree.

**Alternative rule families** ([rules_learned.py](agri-drone/src/agridrone/vision/rules_learned.py), [rules_llm.py](agri-drone/src/agridrone/vision/rules_llm.py)) are in the code but **not evaluated** against the ablation's test set — these are the "learned tree" and "llm generated" rule engines referenced in `configs/matrix/full.yaml`, and their absence from E1 is itself a missed opportunity.

---

## § 11 Statistical rigor audit

**What's done well:**
- 10 000-replicate bootstrap percentile CIs for accuracy / macro-F1 / MCC ([statistical_tests.py](agri-drone/evaluate/statistical_tests.py)), producing three CIs per config in [statistical_tests.json](agri-drone/evaluate/results/statistical_tests.json).
- Per-class bootstrap CIs split across ([v2/statistics/per_class_bootstrap_ci_A.json](agri-drone/evaluate/results/v2/statistics/per_class_bootstrap_ci_A.json), `..._B.json`).
- Continuity-corrected McNemar on paired predictions (A vs B p=0.1336) is correctly computed.

**What's broken or misleading:**
1. **Two McNemar files disagree.** [mcnemar.json](agri-drone/evaluate/results/mcnemar.json) reports n=21, p=1.0 — this is a stub (probably an early development artefact that was never deleted). [statistical_tests.json](agri-drone/evaluate/results/statistical_tests.json) reports the real n=934, p=0.1336. Keeping both in the tree is confusing and will destroy reviewer trust.
2. **Holm-Bonferroni v2 is numerically impossible.** All 21 classes with zero discordant pairs while the aggregate A-vs-B disagrees at the image level → the script was run on identical inputs (probably A vs A).
3. **Dietterich 5×2cv result is a placeholder** (`p_value: null`). 5×2cv requires 10 training runs; only 1 training run's predictions exist for Configs A/B.
4. **McNemar is underpowered on n=934 for a −0.43 pp delta.** Paper acknowledges this in §5.5. Correct.
5. **No multiple-comparison correction is applied to the per-class F1 deltas** in §5.6 of the paper, despite 21 simultaneous comparisons. Holm-Bonferroni v2 *was the correction* and it is broken.
6. **No prediction-interval or effect-size decomposition for EML** (paper reports a point ratio, not a bootstrapped distribution of the ratio under cost uncertainty).

**Blunt judgment:** statistical work on the *ablation* is adequate. Statistical work *advertised in v2* is mostly stubs, broken scripts, or unexecuted. The paper should not cite Holm-Bonferroni / Dietterich / Friedman-Nemenyi in their current state.

---

## § 12 Novelty / research-gap audit

**Claimed novelty (per CURRENT paper):** a cost-weighted falsification of the "rules-help-CNN" folk recipe, with a concrete EML crossover ratio.

**Actual novelty in the crop-disease-AI literature:**
- Negative results on neuro-symbolic add-ons to image classifiers: **~25 % novel** (negative results are chronically under-published in applied AI — scarcity alone gives the paper publishable air).
- Cost-weighted evaluation via EML with published ICAR cost tables: **~40 % novel** for agri-CV specifically; mature in medical-imaging literature.
- "r = 18.7 crossover ratio" as a deployment decision rule: **~55 % novel** *if* the number is derivable from a committed artefact — but § 7 shows [eml_comparison.csv](agri-drone/evaluate/results/eml_comparison.csv) is not in the tree, so this is unverified.
- The 21-class Indian wheat+rice benchmark: **~10 % novel** (PlantVillage-derived class lists are abundant; the specific 21-class Indian framing is useful but not methodologically new).
- Full v4 / paper-2 architecture matrix: **0 % novel** because it is not executed.

**Composite research-gap fulfilment: ~25 %.** The repo is one validated negative-result datapoint plus aspirational scaffolding for a larger contribution that was not executed.

---

## § 13 Publication readiness audit

| Sub-dimension | Score /10 | Comment |
|---|---:|---|
| Narrative coherence | 6 | CURRENT draft is tight; 8 other drafts on disk undermine it. |
| Evidence quality | 5 | v1 ablation solid; v2/v4 evidence is stubs. |
| Statistical rigor | 5 | Adequate bootstrap; broken Holm-Bonferroni; null Dietterich. |
| Reproducibility | 6 | Ablation yes, matrix no. |
| Novelty | 4 | Incremental negative result; no new method; no new dataset. |
| Honesty of scope | 6 | §5.7 threats-to-validity is present; drone-framing still leaks into README. |
| Prose quality | 7 | Recent reframe reads well. |
| Fit to target venue | 5 | CV4A / AI4Ag workshop — plausible short paper. Not a main-conference full paper. |

**Plausible venues (in descending probability of acceptance):**
1. **AI4Ag @ NeurIPS workshop (short paper)** — negative-result framing is a natural fit.
2. **CV4A @ CVPR workshop** — same.
3. **Smart Agricultural Technology** (Elsevier, open review) — plausible but reviewers will demand the v4 matrix actually run.
4. **Computers and Electronics in Agriculture (CEA)** — would reject at review for the n=934 + leakage concern.
5. **ICLR / CVPR main conference** — no chance.

**Top 5 reasons a reviewer would reject today:**
1. **v2 matrix is a dry-run** ([per_run.jsonl](agri-drone/evaluate/results/v2/matrix/full-matrix-v4/per_run.jsonl)) while the paper discusses it as if executed.
2. **Potential train/test leakage** from `aug_*.jpg` files not group-split in [scripts/make_splits.py](agri-drone/scripts/make_splits.py).
3. **Broken / inconsistent statistical artefacts** ([mcnemar.json](agri-drone/evaluate/results/mcnemar.json), [holm_bonferroni_mcnemar.json](agri-drone/evaluate/results/v2/statistics/holm_bonferroni_mcnemar.json)).
4. **Single backbone, single training run, single seed** for the headline experiment.
5. **n = 934 (not 935), 21 classes ⇒ ~44 images/class**: test set is too small to support claims about "21-class rice and wheat triage."

**Top 5 improvements that would most change the verdict:**
1. **Execute the 45-cell [configs/matrix/paper2.yaml](agri-drone/configs/matrix/paper2.yaml)** on Colab/Kaggle. A single real EfficientNet-B0 and MobileNetV3-S number kills reject-reason #4.
2. **Fix the split script** ([scripts/make_splits.py](agri-drone/scripts/make_splits.py)) to group-split by base-image ID, re-run the ablation, and publish the new 934-row CSVs. Kills reject-reason #2.
3. **Delete 8 of the 9 paper variants and the broken [mcnemar.json](agri-drone/evaluate/results/mcnemar.json)**. Repo hygiene alone shifts reviewer trust one full band.
4. **Regenerate [holm_bonferroni_mcnemar.json](agri-drone/evaluate/results/v2/statistics/holm_bonferroni_mcnemar.json)** from the A and B prediction CSVs so the per-class pairs are non-trivial.
5. **Commit [override_decomposition.csv](agri-drone/evaluate/results/override_decomposition.csv) and [eml_comparison.csv](agri-drone/evaluate/results/eml_comparison.csv)** so the §5.3 and §5.5 / crossover claims are derivable from files. Currently they are prose-only.

---

## § 14 Harsh truth

The repo contains **one real, well-instrumented experiment** — a single-seed, single-backbone ablation on 934 curated leaf photographs — and a large amount of *scaffolding pretending to be experiments*. The 2400-cell "full matrix", the 54-cell "large" matrix, the "fair EfficientNet-B0 audit", the Holm-Bonferroni per-class test, the Dietterich 5×2cv test, and the "cost crossover ratio r = 18.7" either ship as dry-run stubs, produce numerically impossible outputs, or have no committed artefact backing them. The rule engine is substantive and the negative result is defensible at workshop length. It is not defensible at full-paper length. The nine competing manuscript files on disk — each citing slightly different numbers for the same experiment — will, by themselves, cost the author a reviewer round. The author has written a good short paper and surrounded it with artefacts that look like more work than has actually been done. An external evaluator who reads the repo before the paper will trust the paper less, not more.

---

## § 15 Final verdict

**As-is publication potential:** **4 / 10** — workshop short paper viable after one hygiene pass; main-conference / journal full paper not viable without executing at least `configs/matrix/paper2.yaml` and fixing the split-leakage risk.

**Research-gap fulfilment:** **~25 %** — one validated negative datapoint in a literature that badly needs negative datapoints, surrounded by unexecuted scaffolding that was promised as contribution.

**Confidence in repo-to-paper alignment:** **5 / 10** — the ablation numbers in the paper are honestly derivable from the CSVs. Almost everything else the paper references (matrix, fair baseline, Holm-Bonferroni, crossover ratio, override decomposition) is either stubbed, broken, or missing on disk.

**Single sentence for the external reviewer:** *A competent short negative-result paper is hiding inside this repository, bracketed by unexecuted scaffolding and stale drafts that the author must delete before anyone reads the paper alongside the code.*
