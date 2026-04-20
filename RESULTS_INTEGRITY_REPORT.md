# Results Integrity Report

**Generated:** 2026-04-20
**Method:** Recomputed all headline numbers from committed prediction CSVs using `tools/research_validation/validate_ablation.py`.

## Sample Size

| Source | Claimed n | Actual n | Status |
|---|---:|---:|---|
| `RESEARCH_PAPER_CURRENT.md` | 935 | — | **WRONG** |
| `ablation_summary.json` (`n_test_images`) | 935 | — | **WRONG** |
| `data_split_manifest.json` (unique test images) | — | 934 | Ground truth |
| `predictions_A_yolo_only.csv` (rows) | — | 933 | 1 image (`smut_837.png`) failed prediction |
| `eml_summary.json` (`n_test_samples`) | 934 | — | Correct |

**Verdict:** Correct sample size is **n = 934** (manifest) / **933** (successful predictions). The paper and `ablation_summary.json` must be corrected from 935 → 934. One test image (`smut_837.png`) was in the manifest but absent from prediction CSVs, likely due to a loading failure during ablation.

## Accuracy

| Config | Manuscript claim | Recomputed (n=933) | Status |
|---|---|---|---|
| A (YOLO only) | 96.15% | 96.15% | **EXACT** |
| B (YOLO + rules) | 95.72% | 95.72% | **EXACT** |
| C (rules only) | 13.38% | 13.38% | **EXACT** |

## Macro-F1

| Config | Recomputed |
|---|---|
| A | 0.9618 |
| B | 0.9574 |
| C | 0.0771 |

## McNemar Test (A vs B)

| Metric | Committed (`statistical_tests.json`) | Recomputed | Status |
|---|---|---|---|
| p-value | 0.133614 | 0.133614 | **EXACT** |
| A-right-B-wrong | — | 4 | — |
| B-right-A-wrong | — | 0 | — |
| Total discordant | — | 4 | — |

Note: `PAPER1_NEGATIVE_RESULT.md` (now archived) reported p ≈ 0.40 — that number was **wrong**. The correct value is p = 0.134.

## Override Decomposition (A → B)

| Metric | Manuscript claim (§5.3) | Recomputed | Status |
|---|---|---|---|
| Total overrides | ~97 | 7 | **MISMATCH — manuscript exaggerated** |
| Rescues | 0 | 0 | **EXACT** |
| Corruptions | 4 | 4 | **EXACT** |
| Neutral (both wrong) | — | 3 | — |

**Verdict:** The manuscript claimed "97 overrides" but only 7 prediction changes exist between A and B. The claim of 0 rescues / 4 corruptions is correct. The manuscript must be corrected to say "7 overrides" not "97".

## EML (Expected Monetary Loss)

| Metric | Committed (`eml_summary.json`) | Manuscript | Status |
|---|---|---|---|
| EML A | ₹294.33/ha | ₹294.33/ha | **EXACT** |
| EML B | ₹2,769.06/ha | ₹2,769.06/ha | **EXACT** |
| Delta | +840.8% | +840% | **EXACT** |
| `n_test_samples` | 934 | — | Correct |
| `default_miss_cost` | ₹5,000 | — | — |
| Per-disease critical `cost_miss` | ₹12,000 | — | Partially cited |

## Cost Crossover Ratio (r = 18.7)

**Status: UNVERIFIED.** No committed artefact (`eml_comparison.csv`) supports the r = 18.7 claim. The claim should be marked as unverified in the manuscript unless a derivation script is provided.

## Pre-Registration

**Status: UNSUPPORTED.** No timestamped external registration (OSF, AsPredicted) or versioned hypothesis document exists in the repository. The label "pre-registered" must be removed.

## Claim Traceability Summary

| Claim | Supporting File | Status | Corrected Value |
|---|---|---|---|
| n = 935 | `predictions_A.csv` has 933 rows; manifest has 934 | **WRONG** | n = 934 (manifest) |
| A accuracy 96.15% | `ablation_summary.json` + CSV recompute | **EXACT** | — |
| B accuracy 95.72% | `ablation_summary.json` + CSV recompute | **EXACT** | — |
| C accuracy 13.38% | `ablation_summary.json` + CSV recompute | **EXACT** | — |
| McNemar p ≈ 0.134 | `statistical_tests.json` + recompute | **EXACT** | — |
| EML A = ₹294.33 | `eml_summary.json` | **EXACT** | — |
| EML B = ₹2,769.06 | `eml_summary.json` | **EXACT** | — |
| 0 rescues, 4 corruptions | `override_decomposition.json` (regenerated) | **EXACT** | — |
| 97 total overrides | `override_decomposition.json` | **WRONG** | 7 total overrides |
| r = 18.7 crossover | No file found | **UNVERIFIED** | — |
| "Pre-registered" | No external registration | **UNSUPPORTED** | Remove claim |

## Files Generated

- `evaluate/results/override_decomposition.json` — full override decomposition with per-image detail
- `evaluate/results/override_decomposition.csv` — same in CSV format
- `evaluate/results/claim_traceability_table.csv` — machine-readable claim verification
- `tools/research_validation/validate_ablation.py` — reproducible validation script
