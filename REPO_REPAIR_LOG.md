# Repo Repair Log

**Date:** 2026-04-20
**Scope:** Full scientific cleanup per external audit (`FINAL_REPO_AUDIT_FOR_EXTERNAL_REVIEW.md`)

---

## STEP 1 — Establish Single Source of Truth

### Action: Archived 8 stale manuscript drafts

Moved to `docs/archive_manuscripts/`:
1. `RESEARCH_PAPER.md`
2. `RESEARCH_PAPER_COMPLETE.md`
3. `RESEARCH_PAPER_FINAL_old.md`
4. `RESEARCH_PAPER_FINAL.md`
5. `RESEARCH_PAPER_FINAL_v2.md`
6. `RESEARCH_PAPER_FINAL_v3.md`
7. `RESEARCH_PAPER_v4.md`
8. `PAPER1_NEGATIVE_RESULT.md`

### Canonical manuscript files:
- **`MANUSCRIPT_SUBMISSION_VERSION.md`** — clean short-paper-ready submission draft (new)
- **`RESEARCH_PAPER_CURRENT.md`** — retained as historical working draft (not for submission)

### README updated:
- Links point to `MANUSCRIPT_SUBMISSION_VERSION.md`
- "Repository status and evidence scope" section added
- "Current evidence-backed contribution" section added

---

## STEP 2 — Validate Core Ablation

### Script created: `tools/research_validation/validate_ablation.py`

Recomputed all headline numbers from committed CSVs:

| Claim | Committed | Recomputed | Status |
|---|---|---|---|
| n = 935 | ablation_summary.json | 934 (manifest), 933 (predictions) | **WRONG — corrected to 934** |
| A accuracy 96.15% | ablation_summary.json | 0.9615 | EXACT |
| B accuracy 95.72% | ablation_summary.json | 0.9572 | EXACT |
| C accuracy 13.38% | ablation_summary.json | 0.1338 | EXACT |
| McNemar p=0.134 | statistical_tests.json | 0.133614 | EXACT |
| EML A=₹294.33 | eml_summary.json | 294.33 | EXACT |
| EML B=₹2769.06 | eml_summary.json | 2769.06 | EXACT |
| 97 overrides | RESEARCH_PAPER_CURRENT.md §5.3 | **7 overrides** | **WRONG — corrected** |
| 0 rescues, 4 corruptions | RESEARCH_PAPER_CURRENT.md §5.3 | 0 rescues, 4 corruptions | EXACT |

### Output: `RESULTS_INTEGRITY_REPORT.md`

---

## STEP 3 — Investigate Leakage

### Scripts created:
- `tools/research_validation/investigate_leakage.py`
- `tools/research_validation/deep_leakage_analysis.py`

### Finding: **NO LEAKAGE**

- Each `aug_*` file has a unique base ID (no sibling augmentations)
- 0 overlapping base IDs between train and test
- `scripts/make_splits.py` is safe for this dataset
- The split script lacks group-aware splitting as a defensive measure, but it doesn't matter here

### Output: `LEAKAGE_INVESTIGATION_REPORT.md`

---

## STEP 4 — Quarantine Broken Artefacts

### Created: `evaluate/results/_quarantined/`

Moved/copied with README explaining each:

| Artefact | Original path | Reason |
|---|---|---|
| `mcnemar.json` | `evaluate/results/mcnemar.json` | Stale stub: n=21, p=1.0 — contradicts real McNemar (n=934, p=0.134) |
| `holm_bonferroni_mcnemar.json` | `evaluate/results/v2/statistics/` | 0 discordant pairs across all 21 classes — numerically impossible |
| `dietterich_5x2cv.json` | `evaluate/results/v2/statistics/` | Placeholder: p_value=null |

### Added WARNING_DRY_RUN.md to:
- `evaluate/results/v2/matrix/full-matrix-v4/` — all 2400 cells are smoke/dry-run
- `evaluate/results/v2/baseline_audit/efficientnet_b0_fair_v1/` — status: dry-run, metrics: null

---

## STEP 5 — Regenerate Missing Artefacts

### Generated from committed prediction CSVs:
- `evaluate/results/override_decomposition.json` — full per-image override analysis
- `evaluate/results/override_decomposition.csv` — same in CSV format
- `evaluate/results/claim_traceability_table.csv` — machine-readable claim verification

---

## STEP 6 — Write Honest Manuscript

### Created: `MANUSCRIPT_SUBMISSION_VERSION.md`

Key changes from `RESEARCH_PAPER_CURRENT.md`:
- Removed "Pre-Registered" from title (no registry exists)
- Corrected n=935 → n=934 throughout
- Corrected "97 overrides" → "7 overrides"
- Removed 54-cell matrix as an executed contribution
- Removed r=18.7 crossover ratio (no artefact)
- Removed "pre-registered" claims
- Corrected "10× random chance" → "~2.8× random chance" for Config C
- Narrowed scope to match actual evidence
- Added sharper threats-to-validity section
- Added leakage investigation finding

### Created: `SUPPORTED_CLAIMS_ONLY.md`

Lists every retained claim with its supporting file and verification status, plus every removed/corrected claim with rationale.

---

## STEP 7 — Rewrite README

Updated `README.md` with:
- "Repository Status and Evidence Scope" section
- "Current Evidence-Backed Contribution" section
- Links to `MANUSCRIPT_SUBMISSION_VERSION.md` (not stale drafts)
- Clear distinction between executed and planned experiments
- Honest framing of what the repo actually supports

---

## STEP 8 — Submission Checklist

### Created: `PAPER_PACKAGE_CHECKLIST.md`

---

## Summary of Changes

| Metric | Count |
|---|---|
| Stale manuscripts archived | 8 |
| Broken artefacts quarantined | 3 |
| Dry-run folders labelled | 2 |
| Artefacts regenerated from CSVs | 3 |
| Validation scripts created | 3 |
| Report files created | 6 |
| Numerical errors corrected | 4 (n=935→934, 97→7 overrides, 10×→2.8×, "pre-registered"→removed) |
| Leakage risk | **NONE confirmed** |
| Workshop paper ready | **Yes** (after addressing remaining items in checklist) |
| Journal/full paper ready | **No** (requires multi-backbone runs + larger test set) |
