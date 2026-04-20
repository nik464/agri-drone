# Paper Package Checklist

**Status:** Post-cleanup, 2026-04-20

---

## Files to Include for Workshop Submission

### Manuscript
- [ ] `MANUSCRIPT_SUBMISSION_VERSION.md` — canonical submission draft

### Supporting Evidence (safe to cite)
- [ ] `evaluate/results/predictions_A_yolo_only.csv` — 933 rows, real
- [ ] `evaluate/results/predictions_B_yolo_rules.csv` — 933 rows, real
- [ ] `evaluate/results/predictions_C_rules_only.csv` — 933 rows, real
- [ ] `evaluate/results/ablation_summary.json` — real (note: n_test_images says 935, actual is 934)
- [ ] `evaluate/results/statistical_tests.json` — real, McNemar p=0.133614
- [ ] `evaluate/results/eml_summary.json` — real, EML A=294.33, B=2769.06
- [ ] `evaluate/results/override_decomposition.json` — regenerated from CSVs
- [ ] `evaluate/results/override_decomposition.csv` — same in CSV
- [ ] `evaluate/results/claim_traceability_table.csv` — claim verification
- [ ] `evaluate/results/cross_dataset_PDT.json` — real, acc=0.8438
- [ ] `evaluate/results/robustness_summary.json` — real
- [ ] `evaluate/results/sensitivity_results.csv` — real
- [ ] `configs/economics/india_2025.yaml` — cost parameters
- [ ] `evaluate/data_split_manifest.json` — split provenance

### Code (for supplemental material)
- [ ] `evaluate/ablation_study.py` — main experiment script
- [ ] `evaluate/statistical_tests.py` — bootstrap + McNemar
- [ ] `evaluate/eml_analysis.py` — EML computation
- [ ] `src/agridrone/vision/rule_engine.py` — substantive rule engine
- [ ] `src/agridrone/knowledge/diseases.json` — disease profiles
- [ ] `scripts/make_splits.py` — deterministic splitter
- [ ] `tools/research_validation/validate_ablation.py` — claim validation

### Reports
- [ ] `RESULTS_INTEGRITY_REPORT.md`
- [ ] `LEAKAGE_INVESTIGATION_REPORT.md`
- [ ] `SUPPORTED_CLAIMS_ONLY.md`

---

## Files to EXCLUDE from Submission / Supplemental

### Quarantined (broken/misleading)
- ❌ `evaluate/results/_quarantined/mcnemar.json` — stale stub
- ❌ `evaluate/results/_quarantined/v2_holm_bonferroni/` — numerically impossible
- ❌ `evaluate/results/_quarantined/v2_dietterich/` — null placeholder
- ❌ `evaluate/results/v2/matrix/full-matrix-v4/` — dry-run stubs
- ❌ `evaluate/results/v2/baseline_audit/efficientnet_b0_fair_v1/` — dry-run placeholder

### Archived manuscripts
- ❌ `docs/archive_manuscripts/` — all 8 stale drafts

### Infrastructure (not evidence)
- ❌ `evaluate/matrix/run_matrix.py` — matrix runner (infrastructure, not results)
- ❌ `configs/matrix/*.yaml` — experimental designs (not executed)
- ❌ `notebooks/colab/01_run_matrix.ipynb` — runner notebook

---

## Which Results Are Safe to Cite

| Result | File | Safe? |
|---|---|---|
| Ablation A/B/C accuracy | `ablation_summary.json` + CSVs | ✅ Yes |
| McNemar p=0.134 | `statistical_tests.json` | ✅ Yes |
| Bootstrap CIs | `statistical_tests.json` | ✅ Yes |
| EML A=294.33, B=2769.06 | `eml_summary.json` | ✅ Yes |
| Override decomposition | `override_decomposition.json` | ✅ Yes (regenerated) |
| Per-class F1 deltas | `ablation_summary.json` | ✅ Yes |
| Cross-dataset PDT 84.4% | `cross_dataset_PDT.json` | ✅ Yes (with caveats from threshold sweep) |
| 54-cell or 2400-cell matrix results | — | ❌ No (dry-run only) |
| Holm-Bonferroni correction | — | ❌ No (quarantined, broken) |
| Dietterich 5×2cv | — | ❌ No (null placeholder) |
| Fair EfficientNet baseline | — | ❌ No (dry-run) |
| r=18.7 crossover ratio | — | ❌ No (no artefact) |

---

## What Still Blocks Journal/Full-Paper Submission

1. **Multi-backbone ablation not executed.** Need to run `configs/matrix/paper2.yaml` (45 cells) with at least EfficientNet-B0 and MobileNetV3 on GPU. This would address the "single backbone" limitation.

2. **Single seed, single training run.** Need ≥3 seeds for error bars on training variance.

3. **Small test set.** n=934 is adequate for workshop paper. Journal reviewers will demand n ≥ 2,000–5,000.

4. **No multiple-comparison correction executed.** Need to regenerate Holm-Bonferroni from A/B CSVs (the quarantined version was broken).

5. **No in-field evaluation.** All evaluation is on lab-quality smartphone images.

---

## What Is Enough for a Workshop Short Paper

The current evidence package supports a **4–6 page workshop paper** at venues like:
- AI4Ag @ NeurIPS
- CV4A @ CVPR
- AIAI workshop tracks

**The paper is workshop-ready** with the following conditions:
- [x] Canonical manuscript written (`MANUSCRIPT_SUBMISSION_VERSION.md`)
- [x] All claims verified against committed artefacts
- [x] No leakage found
- [x] Broken artefacts quarantined
- [x] Override decomposition regenerated
- [x] Sample size corrected (934, not 935)
- [ ] Convert manuscript to LaTeX (venue-specific template)
- [ ] Generate camera-ready figures from committed data
- [ ] Final proofread pass

---

## Recommended Next Steps (priority order)

1. Convert `MANUSCRIPT_SUBMISSION_VERSION.md` to LaTeX in venue template
2. Run `configs/matrix/paper2.yaml` on Colab/Kaggle (adds multi-backbone evidence)
3. Regenerate Holm-Bonferroni from A/B CSVs (fixes per-class correction gap)
4. Collect 1,000+ additional test images for stronger statistical power
5. Submit to AI4Ag workshop at NeurIPS 2026 or CV4A at CVPR 2027
