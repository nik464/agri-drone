# CHANGELOG — research-upgrade branch

This document summarises the changes introduced by the ``research-upgrade``
branch. Every change is additive or in-place non-breaking; no existing
public function signatures were altered, no result files under
``evaluate/results/*.json`` were modified, and ``RESEARCH_PAPER_FINAL_v3.md``
is preserved byte-identical.

## New files

### Step 1 — Safety net
- ``ruff.toml`` — permissive lint config (all pre-existing style noise muted).
- ``tests/regression/test_frozen_metrics.py`` — 16 assertions locking every
  headline v3 number (A=96.15%, B=95.72%, C=13.38%, EML(A)=₹294.33,
  EML(B)=₹2,769.06, PDT spec=0, 672 images, EffNet=76.15%, 125 sensitivity
  configs).
- ``scripts/smoke_test.py`` — 0.5 s CPU smoke test (budget 60 s).
- ``.github/workflows/ci.yml`` — ruff + regression tests + smoke on push/PR.

### Step 2 — Paper reframing
- ``RESEARCH_PAPER_v4.md`` — v3 cloned, title/abstract reframed around the
  negative rule-engine result, §3.1 Data Provenance added, §5.4 PDT section
  rewritten honestly with three rescue variants tagged ``[TO BE RE-RUN]``,
  author templated as ``{{AUTHOR_NAME}}``.

### Step 3 — Experiment matrix
- ``docs/results_schema.md`` — v2 JSONL + aggregate schema.
- ``configs/matrix/{full,smoke}.yaml`` — 6×4×5×4×5 grid (2,400 cells) and a
  tiny CI grid.
- ``evaluate/matrix/run_matrix.py`` — matrix runner with ``--dry-run``,
  ``--smoke-test``, ``--tracker {none,mlflow,wandb}``.
- ``evaluate/matrix/train.py`` — backbone registry + training dispatch stub.
- ``evaluate/results/v2/README.md`` — v2 artifact layout.

### Step 4 — Fair baseline re-audit
- ``docs/training_recipe.md`` — shared AdamW/cosine/50ep/label-smooth recipe.
- ``evaluate/matrix/audit_baseline.py`` — EfficientNet-B0 re-audit CLI.

### Step 5 — Statistical protocol v2
- ``evaluate/statistical_tests.py`` — added ``--v2`` flag (non-breaking
  shim); v1 behaviour preserved when flag absent.
- ``evaluate/statistical_tests_v2.py`` — per-class bootstrap CI,
  Holm-Bonferroni, Dietterich 5×2cv, Friedman-Nemenyi placeholder.

### Step 6 — PDT rescue
- ``evaluate/pdt_v2.py`` — threshold sweep (ROC-AUC, PR-AUC, spec@90%sens,
  sens@90%spec), few-shot fine-tune stub, calibration stub. Writes to
  ``evaluate/results/v2/pdt/``.

### Step 7 — Learned / neuro-symbolic rule baselines
- ``src/agridrone/vision/rule_engine_base.py`` — ``RuleEngine`` Protocol +
  registry.
- ``src/agridrone/vision/rules_learned.py`` — decision-tree rule engine.
- ``src/agridrone/vision/rules_llm.py`` — LLM-generated rule engine
  (offline fixture by default; online mode guarded by
  ``ENABLE_LLM_RULES=1``).
- ``src/agridrone/vision/rules_llm_fixtures/cached_rules.json`` — seed
  fixture so tests and dry runs are fully offline.
- Default remains the existing handcrafted engine in ``rule_engine.py``
  (unchanged).

### Step 8 — EML defensibility
- ``configs/economics/india_2025.yaml`` — per-entry ``source:`` citations;
  entries marked ``ESTIMATE`` are excluded from the headline.
- ``evaluate/eml_sensitivity.py`` — headline + ±25/50% tornado analysis.
  Output: ``evaluate/results/v2/eml/{headline_v4,sensitivity_tornado}.json``.

### Step 9 — Reproducibility
- ``scripts/download_data.py`` — authoritative dataset locations (URLs,
  licences, local paths); no automatic download.
- ``scripts/make_splits.py`` — deterministic split writer with
  ``splits_manifest.json`` (seed + SHA-256).
- ``CITATION.cff`` — citation metadata (author templated).
- ``requirements.lock.txt`` — exact-version lock of the v3 interpreter.
- ``Dockerfile`` + ``docker-compose.yml`` — CPU image that runs the full CI
  gate.
- ``docs/data_availability.md`` — dataset table, domain caveat, split
  reproduction commands.

### Step 10 — Repo hygiene
- ``README.md`` — "Research Prototype" badge, "Known Limitations" and
  "Research Roadmap" sections; "precision agriculture system" softened to
  "research codebase".
- ``dashboard/README.md`` — "premium, production-grade" softened to
  "research demo dashboard".
- ``CHANGELOG_RESEARCH_UPGRADE.md`` — this file.
- ``PR_DESCRIPTION.md`` — ready-to-paste PR description.

## Modified (non-breakingly)

- ``evaluate/statistical_tests.py`` — added ``--v2`` argparse flag.
- ``README.md``, ``dashboard/README.md`` — tone corrections and links to
  new research artefacts.

## Untouched (by explicit design)

- ``RESEARCH_PAPER_FINAL_v3.md`` — byte-identical (54,104 bytes, 638 lines).
- All files under ``evaluate/results/*.json`` — frozen for regression tests.
- All existing public function signatures in ``src/agridrone/``.
- All model weights in ``models/``.

## What still needs a GPU host to fully materialise

1. Matrix runs (``run_matrix.py`` without ``--dry-run``).
2. EfficientNet-B0 fair re-audit real numbers.
3. PDT few-shot and calibration variants.
4. Real LLM-generated rule file to replace the offline fixture.

All four paths have CLIs that run end-to-end on CPU and emit
``status: skipped`` records with ``TODO-GPU`` notes; nothing is hidden
behind TODO comments in source.
