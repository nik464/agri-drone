# Quarantined Results

Files in this folder were moved here during the 2026-04-20 repo cleanup because
they are **broken, numerically impossible, or dry-run placeholders** that must
not be cited as evidence in any manuscript.

| File / subfolder | Reason | Original location |
|---|---|---|
| `mcnemar.json` | Stale stub (n=21, p=1.0) — contradicts the real McNemar in `statistical_tests.json` | `evaluate/results/mcnemar.json` |
| `v2_holm_bonferroni/` | All 21 classes show 0 discordant pairs — numerically impossible given A≠B; likely ran A-vs-A | `evaluate/results/v2/statistics/holm_bonferroni_mcnemar.json` |
| `v2_dietterich/` | Placeholder with `p_value: null`; requires 10 training runs that were never executed | `evaluate/results/v2/statistics/dietterich_5x2cv.json` |
| `v2_matrix_dryrun/` | Every row is `status: smoke, metrics: null` — 2400-cell matrix was never trained | `evaluate/results/v2/matrix/full-matrix-v4/` |
| `v2_baseline_dryrun/` | `status: dry-run, metrics: null` — fair EfficientNet audit was never executed | `evaluate/results/v2/baseline_audit/efficientnet_b0_fair_v1/` |

**Action required before any of these files can return to active results:**
regenerate from real GPU runs, or delete permanently.
