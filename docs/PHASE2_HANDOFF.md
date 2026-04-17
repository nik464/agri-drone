# Phase 2 handoff — agri-drone research-upgrade

This document is the **only** thing you need to read to hand the
`research-upgrade` branch off to the Colab runs and the eventual
Smart Agricultural Technology submission. Everything else is linked from
here.

---

## 1. Phase 2 commit summary (6 commits, all on `research-upgrade`)

| # | Hash | Commit |
|---|------|--------|
| 1 | `7547da3` | fix(tests): make hotspot detector test device-agnostic |
| 2 | `763bb23` | chore(audit): add system environment audit scripts |
| 3 | `21b6309` | feat(colab): end-to-end experimental matrix notebook |
| 4 | `ecdfc59` | feat(colab): PDT calibration and few-shot notebook |
| 5 | `ab16b98` | feat(colab): fair multi-backbone baseline notebook |
| 6 | `a6e98fc` | docs(colab): non-ML user walkthrough for all three notebooks |

All 35 tests pass, 1 GPU test auto-skipped (no local CUDA), `ruff check .`
clean. See the raw output of the audit in Section 2 below.

---

## 2. System audit (agent sandbox)

Copied verbatim from `python scripts/audit_system.py` at the time of
commit `a6e98fc`:

```
============================================================
  agri-drone system audit  -  2026-04-17T09:41:35Z
============================================================
  OS         : Windows-10-10.0.26200-SP0
  Python     : 3.11.7   venv=False
  Git        : research-upgrade@7547da3   ahead-of-main=11   dirty=7
  GPU        : nvidia-smi not on PATH
  torch      : 2.4.1+cpu   cuda_available=False   cuda=None
  disk free  : 97.14 GB
  RAM        : total=16.2 GB available=1.95 GB
  ports      : 8000=in-use   9000=in-use
============================================================
```

Full machine-readable snapshot lives in
[`docs/system_audit_report.json`](system_audit_report.json). Rerun on any
host via `python scripts/audit_system.py` or `pwsh scripts/audit_system.ps1`.

---

## 3. Your action checklist (ordered)

### 3.1 Open the Phase 1 + Phase 2 PR

<https://github.com/Ashut0sh-mishra/agri-drone/pull/new/research-upgrade>

Paste the body from [`PR_DESCRIPTION.md`](../PR_DESCRIPTION.md).

### 3.2 Fill author / DOI placeholders

Files containing `{{AUTHOR_NAME}}`:
- [`RESEARCH_PAPER_v4.md`](../RESEARCH_PAPER_v4.md) (line 3)
- [`CHANGELOG_RESEARCH_UPGRADE.md`](../CHANGELOG_RESEARCH_UPGRADE.md) (line 24)

Files containing `{{AUTHOR_FAMILY_NAME}}` / `{{AUTHOR_GIVEN_NAME}}`:
- [`CITATION.cff`](../CITATION.cff) (lines 17–18 and 32–33)

Files containing `{{AUTHOR_AFFILIATION}}`:
- [`RESEARCH_PAPER_v4.md`](../RESEARCH_PAPER_v4.md) (line 4)
- [`CITATION.cff`](../CITATION.cff) (line 19)

Files containing `{{AUTHOR_EMAIL}}`:
- [`RESEARCH_PAPER_v4.md`](../RESEARCH_PAPER_v4.md) (line 5)

Files containing `{{AUTHOR_ORCID}}` / `{{ORCID}}`:
- [`CITATION.cff`](../CITATION.cff) (line 20)

Files containing `{{ZENODO_DOI}}`:
- none yet — add once you archive the dataset to Zenodo. Expected
  locations: `docs/data_availability.md`, `CITATION.cff`.

Files containing `{{GDRIVE_URL}}`:
- none yet — add once you upload the dataset snapshot to Google Drive.
  Expected location: `docs/data_availability.md`.

Suggested sweep:

```powershell
# PowerShell find/replace — run from repo root
(Get-ChildItem -Recurse -Include *.md,*.cff,*.yaml,*.yml |
    Select-String -Pattern '\{\{(AUTHOR_|ZENODO_|GDRIVE_|ORCID)' -List).Path
```

### 3.3 Upload the dataset to Google Drive

Create this folder tree in your Drive:

```
MyDrive/agri-drone/
    data/
        plantvillage/
        PDT_datasets/
        riceleaf/        (optional)
        ricepest/        (optional)
    models_v2/           (leave empty — notebooks write here)
    results_v2/          (leave empty — notebooks write here)
```

URLs + licences: [`docs/data_availability.md`](data_availability.md).

### 3.4 Run the Colab notebooks (in order)

1. Open [`notebooks/colab/01_run_matrix.ipynb`](../notebooks/colab/01_run_matrix.ipynb)
   → Runtime → GPU (T4) → Run all. ~2 h. Writes
   `MyDrive/agri-drone/results_v2/matrix/` and
   `results_v2_<timestamp>.zip`.
2. Open [`notebooks/colab/02_pdt_calibration.ipynb`](../notebooks/colab/02_pdt_calibration.ipynb)
   → Run all. ~45 min. Writes `results_v2/pdt/PDT_SECTION.md`.
3. Open [`notebooks/colab/03_baseline_reaudit.ipynb`](../notebooks/colab/03_baseline_reaudit.ipynb)
   → Run all. ~2.5 h. Writes `results_v2/baselines/BASELINES_TABLE.md`.

### 3.5 Paste generated fragments into the v4 paper

- `results_v2/pdt/PDT_SECTION.md` → replaces the `[TO BE RE-RUN]` block in
  `RESEARCH_PAPER_v4.md` §5.4.
- `results_v2/baselines/BASELINES_TABLE.md` → drop in under §6.7.
- `results_v2/RESULTS_SUMMARY.md` → quote into the abstract numbers.

### 3.6 Commit results back to the branch

```powershell
cd D:\Projects\agri-drone
git checkout research-upgrade
Expand-Archive -Path results_v2_<timestamp>.zip `
    -DestinationPath evaluate/results/v2/ -Force
git add evaluate/results/v2/
git commit -m "results(v2): add Colab-generated matrix/PDT/baseline artifacts"
git push origin research-upgrade
```

---

## 4. Expected cost

- **Colab free tier** — enough for quick mode of notebook 1 + all of
  notebook 2. Cost: ₹0.
- **Colab Pro (~₹400/month)** — recommended if you run full mode of
  notebook 1 (2400 cells on A100) or repeat notebook 3 multiple times.
- **GitHub** — public repo, no cost.
- **Storage** — ~5 GB Google Drive. Free tier is 15 GB.

---

## 5. Known limitations still unresolved

These are documented in the v4 paper and the README; they are **not** fixed
by this PR and would need additional work / data collection:

- **Drone imagery.** We still have zero true drone-altitude farm-canopy
  images. The v4 paper is explicit about this in §3.1. Real drone data
  would require a field campaign (expert pilot + pathologist + multispectral
  camera + monsoon-season timing) that is out of scope here.
- **Pathologist-review.** Rule-engine scores have never been validated
  against an agricultural pathologist's clinical opinion. Until someone
  does, the "rule engine is safer" claim cannot be tested. §7 of the v4
  paper flags this as future work.
- **Farmer UX study.** The dashboard is a research demo, not a deployed
  product. No usability testing with Indian farmers has been done.
- **Temporal generalisation.** All images are single time-points; no
  dataset covers multi-season or multi-year disease cycles.
- **Full-matrix numbers are unverified.** Every v2 artefact tagged
  `[TO BE RE-RUN]` depends on section 3.4 above. Until the Colab notebooks
  actually run on real GPUs, the scaffolding is correct but the numbers
  are stubs.

---

## 6. If Phase 3 is ever scoped, likely contents

- Run the Colab notebooks, materialise the `[TO BE RE-RUN]` numbers, and
  do a final paper polish.
- Collect a modest drone-altitude validation set (~200 images, 3 diseases)
  even without a full field campaign, as an honest external test.
- Get pathologist labels on 100 rule-engine-disagreement cases.
- Submit to Smart Agricultural Technology.
