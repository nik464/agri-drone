# Supported Claims Only

Every claim retained in `MANUSCRIPT_SUBMISSION_VERSION.md` is listed here with its supporting artefact(s) and verification status.

**Generated:** 2026-04-20
**Validation script:** `tools/research_validation/validate_ablation.py`

---

## Retained Claims

| # | Claim | Supporting File(s) | Verification |
|---|---|---|---|
| 1 | Test set n = 934 images, 21 classes | `evaluate/data_split_manifest.json` | Counted: 934 unique test images across 21 class keys |
| 2 | Config A accuracy = 96.15% | `evaluate/results/predictions_A_yolo_only.csv`, `evaluate/results/ablation_summary.json` | Recomputed from CSV: 0.9615. Matches `ablation_summary.json` |
| 3 | Config B accuracy = 95.72% | `evaluate/results/predictions_B_yolo_rules.csv`, `evaluate/results/ablation_summary.json` | Recomputed from CSV: 0.9572. Matches |
| 4 | Config C accuracy = 13.38% | `evaluate/results/predictions_C_rules_only.csv`, `evaluate/results/ablation_summary.json` | Recomputed from CSV: 0.1338. Matches |
| 5 | Config A macro-F1 = 0.9618 | `evaluate/results/predictions_A_yolo_only.csv` | Recomputed: 0.9618 |
| 6 | Config B macro-F1 = 0.9574 | `evaluate/results/predictions_B_yolo_rules.csv` | Recomputed: 0.9574 |
| 7 | Bootstrap 95% CI A accuracy: [0.9486, 0.9732] | `evaluate/results/statistical_tests.json` | Committed artefact, n_boot=10000 |
| 8 | McNemar A vs B: p = 0.134, χ² = 2.25, 4 discordant | `evaluate/results/statistical_tests.json`, CSV recompute | Recomputed: χ²=2.25, p=0.133614. Matches |
| 9 | McNemar A vs C: p < 10⁻³⁰⁰, 777 discordant | `evaluate/results/statistical_tests.json` | Committed artefact |
| 10 | 4 of 934 predictions change A→B, all wrong-direction | `evaluate/results/override_decomposition.json` (regenerated) | Recomputed: 4 corruptions, 0 rescues out of 7 overrides |
| 11 | Override decomposition: 0 rescues, 4 corruptions, 3 neutral | `evaluate/results/override_decomposition.json` | Recomputed from paired CSVs |
| 12 | EML A = ₹294.33/ha | `evaluate/results/eml_summary.json` | Committed artefact |
| 13 | EML B = ₹2,769.06/ha | `evaluate/results/eml_summary.json` | Committed artefact |
| 14 | EML delta = +840.8% | `evaluate/results/eml_summary.json` | Committed: `delta_pct: 840.8` |
| 15 | Cost matrix: miss_cost 12,000 for critical diseases | `configs/economics/india_2025.yaml`, `evaluate/results/eml_summary.json` | Committed |
| 16 | Config A latency 15.4 ms on T4 | `evaluate/results/ablation_summary.json` | Committed artefact |
| 17 | Config B latency 444.4 ms | `evaluate/results/ablation_summary.json` | Committed artefact |
| 18 | YOLOv8n-cls 1.4M params | Model architecture (public Ultralytics spec) | Verifiable |
| 19 | Seed 42, 70/15/15 split | `scripts/make_splits.py`, `evaluate/data_split_manifest.json` | Committed code + manifest |
| 20 | No train/test leakage from aug files | `LEAKAGE_INVESTIGATION_REPORT.md`, `tools/research_validation/deep_leakage_analysis.py` | Verified: 0 overlapping base IDs |
| 21 | Rule engine has ≥5 rule families (color, texture, spatial, conflict) | `src/agridrone/vision/rule_engine.py` | Code inspection |
| 22 | Per-class F1 deltas: tan_spot −0.038, yellow_rust −0.022, black_rust −0.019 | `evaluate/results/ablation_summary.json` | Committed artefact |
| 23 | Cross-dataset PDT accuracy 84.4% | `evaluate/results/cross_dataset_PDT.json` (acc: 0.8438) | Committed artefact |
| 24 | Config C = 10× random chance on 21 classes | 1/21 = 4.76%, Config C = 13.4% | Arithmetic: 13.4/4.76 ≈ 2.8×. **CORRECTED:** C is ~2.8× random, not 10× | 

## Removed Claims

| # | Claim (from previous drafts) | Reason removed |
|---|---|---|
| R1 | n = 935 | Actual n = 934 (manifest), 933 (predictions). Off-by-one. |
| R2 | "Pre-registered" experiment | No timestamped external registration exists (no OSF, AsPredicted, or registry URL) |
| R3 | 97 overrides (10.4% of test set) | Recomputation shows only 7 overrides. The 97 figure was wrong. |
| R4 | "93 of 97 (95.9%) confirmed" | Based on the wrong 97-override count |
| R5 | 54-cell matrix as an executed contribution | Matrix infrastructure exists but ALL 2400 cells are dry-run stubs with null metrics |
| R6 | r = 18.7 cost crossover ratio | No committed artefact (`eml_comparison.csv` does not exist) |
| R7 | "Shared-recipe fair EfficientNet-B0 audit" | `v2/baseline_audit/` is a dry-run placeholder |
| R8 | Holm-Bonferroni per-class McNemar correction | Committed file had 0 discordant pairs (numerically impossible); quarantined |
| R9 | Dietterich 5×2cv test | Placeholder with p_value: null; quarantined |
| R10 | McNemar p ≈ 0.40 (from PAPER1_NEGATIVE_RESULT.md) | Wrong number; correct value is 0.134 |
| R11 | Config C = 10× random chance | Arithmetic: 13.4% / 4.76% ≈ 2.8×, not 10× |

## Corrected Claims

| Original | Corrected | Reason |
|---|---|---|
| n = 935 | n = 934 | Manifest count; 1 image failed inference |
| 97 overrides | 7 overrides | Recomputed from A/B prediction CSVs |
| "Pre-registered" | Removed | No external pre-registration evidence |
| 10× random chance (Config C) | ~2.8× random chance | 13.4% / (100%/21) = 2.81 |
