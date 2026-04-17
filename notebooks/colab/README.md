# Running agri-drone experiments on Google Colab

You do **not** need a GPU on your own computer. Colab gives you a free Tesla
T4 for ~2 hours at a time, which is enough for the "quick" mode of the
experimental matrix. Colab Pro (₹400/month as of April 2026) gives you an
A100 and is needed only if you want to re-run the full 2400-cell grid.

Three notebooks live in this folder; click the badges to open them in Colab.

| # | Notebook | Badge | Estimated runtime |
|---|----------|-------|-------------------|
| 1 | Experimental matrix | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ashut0sh-mishra/agri-drone/blob/research-upgrade/notebooks/colab/01_run_matrix.ipynb) | T4 ≈ 2 h (quick) · A100 ≈ 12 h (full) |
| 2 | PDT calibration + few-shot | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ashut0sh-mishra/agri-drone/blob/research-upgrade/notebooks/colab/02_pdt_calibration.ipynb) | T4 ≈ 45 min |
| 3 | Fair multi-backbone baselines | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ashut0sh-mishra/agri-drone/blob/research-upgrade/notebooks/colab/03_baseline_reaudit.ipynb) | T4 ≈ 2.5 h |

## Prerequisites (10 minutes)

1. You need a **Google account** and **~5 GB free on Google Drive**.
2. In your Drive, pre-create this folder structure:
   ```
   MyDrive/agri-drone/
       data/          ← upload the datasets here
       models_v2/     ← notebooks write trained weights here
       results_v2/    ← notebooks write JSON / CSV results here
   ```
3. Upload the datasets listed in [docs/data_availability.md](../../docs/data_availability.md)
   into `MyDrive/agri-drone/data/`. Typical structure:
   ```
   MyDrive/agri-drone/data/
       plantvillage/
       PDT_datasets/
       riceleaf/
       ricepest/
   ```

## What to click (for a non-ML user)

1. Open **Notebook 1** (badge above).
2. On the top menu: **Runtime → Change runtime type → GPU (T4)** → Save.
3. Click **Runtime → Run all**. Authorise Google Drive when prompted.
4. Wait ~2 hours. The notebook prints progress and will pause only if a cell
   fails. If it does, read the error, fix the input, and press the
   ▶ button on that cell to resume.
5. When done, the last cell prints a path like
   `/content/drive/MyDrive/agri-drone/results_v2_<timestamp>.zip` — that is
   your deliverable.
6. Repeat for **Notebook 2** and **Notebook 3** in that order.
7. Download the generated `RESULTS_SUMMARY.md`, `PDT_SECTION.md`, and
   `BASELINES_TABLE.md` from Drive and paste them into the relevant
   sections of [`RESEARCH_PAPER_v4.md`](../../RESEARCH_PAPER_v4.md) (§5.4
   and §6.7).

## Running order (important)

- Notebook 1 **first** — it primes the matrix results that 2 and 3 cross-reference.
- Notebook 2 **next** — uses the PDT predictions CSV produced by notebook 1.
- Notebook 3 **last** — produces the baselines table for v4 §6.7.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `No GPU` assertion fails | Runtime is CPU | Runtime → Change runtime type → GPU |
| `CUDA out of memory` | Batch too large | Edit recipe cell: lower `batch_size` (e.g. 32 → 16) |
| Session disconnects mid-run | Colab idle timeout (~90 min) | Re-open notebook; cells are idempotent — rerun from the last completed marker |
| `Missing: /content/drive/MyDrive/agri-drone/data` | Drive folder not created | Pre-create the folder tree described above |
| "Permission denied" for Drive | Mount not authorised | Re-run the Drive-mount cell and complete the auth flow |

## What **not** to do

- Do **not** commit trained weights back to the repo. They belong in
  `MyDrive/agri-drone/models_v2/` only.
- Do **not** modify `RESEARCH_PAPER_FINAL_v3.md` or any file under
  `evaluate/results/*.json` (both are frozen by regression tests).
- Do **not** force-push or rewrite history on `research-upgrade`.

## Expected cost

- Free tier is enough for **quick mode** of notebook 1 + all of notebook 2.
- **Colab Pro (~₹400/month)** is recommended if you want to run full mode
  of notebook 1 or multiple re-runs of notebook 3.

## After the runs: committing results back

The results (JSON + Markdown) are small and safe to commit:

```powershell
# on your local machine, after downloading the zip from Drive
cd D:\Projects\agri-drone
git checkout research-upgrade
# unzip the Colab output into evaluate/results/v2/
Expand-Archive -Path results_v2_<timestamp>.zip -DestinationPath evaluate/results/v2/ -Force
git add evaluate/results/v2/
git commit -m "results(v2): add Colab-generated matrix/PDT/baseline artifacts"
git push origin research-upgrade
```

Open the PR at
<https://github.com/Ashut0sh-mishra/agri-drone/pull/new/research-upgrade>.
