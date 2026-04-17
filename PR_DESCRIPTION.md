# PR: research-upgrade — reproducibility + honest framing + matrix infra

## Summary

This PR turns the agri-drone repo from "engineering project with a
paper-shaped wrapper" into a research-oriented, reproducible, peer-review
ready codebase — without breaking any existing functionality.

* **No existing file was deleted or renamed.**
* **No existing public function signature was changed.**
* **No file under ``evaluate/results/*.json`` was modified.**
* **``RESEARCH_PAPER_FINAL_v3.md`` is byte-identical.**
* **All new deps are MIT/Apache/BSD.**
* **``ruff check .`` passes, ``pytest -q tests/regression`` passes (16/16),
  ``python scripts/smoke_test.py`` runs in 0.51 s (budget 60 s).**

## 10 commits

| # | Step | Files |
|---|------|-------|
| 1 | Baseline safety net | ``ruff.toml``, ``tests/regression/``, ``scripts/smoke_test.py``, ``.github/workflows/ci.yml`` |
| 2 | Paper v4 with negative-result framing + data-provenance §3.1 | ``RESEARCH_PAPER_v4.md`` |
| 3 | 6×4×5×4×5 experiment matrix + v2 result schema | ``docs/results_schema.md``, ``configs/matrix/*.yaml``, ``evaluate/matrix/*`` |
| 4 | Fair EfficientNet-B0 re-audit under shared recipe | ``docs/training_recipe.md``, ``evaluate/matrix/audit_baseline.py`` |
| 5 | Proper statistical protocol (``--v2`` shim, bootstrap CI, Holm-Bonferroni, Dietterich, Friedman-Nemenyi) | ``evaluate/statistical_tests{,_v2}.py`` |
| 6 | PDT rescue (threshold sweep, few-shot, calibration) | ``evaluate/pdt_v2.py`` |
| 7 | ``RuleEngine`` protocol + learned / LLM rule baselines with offline fixture | ``src/agridrone/vision/rule_engine_base.py``, ``rules_learned.py``, ``rules_llm.py`` |
| 8 | Defensible EML: cited costs + ±25/50% tornado | ``configs/economics/india_2025.yaml``, ``evaluate/eml_sensitivity.py`` |
| 9 | Reproducibility bundle: data loaders, splits, CITATION.cff, lockfile, Dockerfile, compose, data availability | ``scripts/download_data.py``, ``scripts/make_splits.py``, ``CITATION.cff``, ``requirements.lock.txt``, ``Dockerfile``, ``docker-compose.yml``, ``docs/data_availability.md`` |
| 10 | Repo hygiene: README status/limitations/roadmap, dashboard tone, CHANGELOG | ``README.md``, ``dashboard/README.md``, ``CHANGELOG_RESEARCH_UPGRADE.md``, ``PR_DESCRIPTION.md`` |

## Acceptance checklist

* [x] v3 paper untouched, new ``RESEARCH_PAPER_v4.md`` is drop-in submission-ready
* [x] New experiment matrix infrastructure runs end-to-end on a toy cell
* [x] Fair baseline re-audit CLI produces an artefact (real numbers need GPU)
* [x] PDT rescue CLI reports threshold sweep metrics and queues few-shot /
      calibration
* [x] Rule-engine protocol + learned + LLM baselines (LLM uses offline
      fixture by default) plug into the matrix
* [x] EML is now cited, and sensitivity is reported
* [x] Reproducibility pack: download/splits scripts, citation file,
      lockfile, Dockerfile, docker-compose, data availability note
* [x] New README tone is honest, new ``CHANGELOG_RESEARCH_UPGRADE.md`` is
      accurate
* [x] All new tests pass, no existing tests break

## Regenerating ``evaluate/results/v2/`` artefacts (GPU host)

```bash
# Matrix — real (not dry-run)
python evaluate/matrix/run_matrix.py \
  --config configs/matrix/full.yaml \
  --out-dir evaluate/results/v2 \
  --tracker mlflow

# Fair EfficientNet-B0 baseline under shared recipe
python evaluate/matrix/audit_baseline.py \
  --backbone efficientnet_b0 \
  --out-dir evaluate/results/v2/baseline_audit

# PDT remediation — all three variants
python evaluate/pdt_v2.py --variant threshold_sweep
python evaluate/pdt_v2.py --variant few_shot --shots 5
python evaluate/pdt_v2.py --variant few_shot --shots 10
python evaluate/pdt_v2.py --variant few_shot --shots 25
python evaluate/pdt_v2.py --variant few_shot --shots 50
python evaluate/pdt_v2.py --variant calibration

# Statistics v2
python evaluate/statistical_tests.py --v2 --n-boot 10000

# EML sensitivity
python evaluate/eml_sensitivity.py
```

## Confidence / honesty note

This PR **does not claim** to have re-run all experiments. It **does**:

1. Add the scaffolding, schemas, CLIs, and tests required to re-run them.
2. Reframe the paper to be honest about what the v3 data actually supported.
3. Lock in every v3 headline number with regression tests so no refactor
   silently drifts them.
4. Document known limitations and a roadmap in-tree.

The confident claim is that a reviewer or successor can clone
``research-upgrade``, allocate a GPU, and generate the v2 artefacts without
further infrastructure work.

## Reviewer checklist

* [ ] ``ruff check .`` passes
* [ ] ``pytest -q tests/regression`` passes (16/16)
* [ ] ``python scripts/smoke_test.py`` completes under 60 s
* [ ] ``python evaluate/matrix/run_matrix.py --config configs/matrix/smoke.yaml --dry-run`` succeeds
* [ ] ``python evaluate/pdt_v2.py --variant threshold_sweep`` produces
      ``evaluate/results/v2/pdt/threshold_sweep.json``
* [ ] ``python evaluate/eml_sensitivity.py`` reports headline + tornado JSON
* [ ] ``python -c 'from agridrone.vision.rule_engine_base import available; print(available())'``
      prints ``['handcrafted','learned_tree','llm_generated','none']``
