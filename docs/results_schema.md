# AgriDrone v2 results schema

All artifacts written by the `evaluate/matrix/` runner and the v2 statistical-
protocol scripts land under `evaluate/results/v2/`. The v1 artifacts in
`evaluate/results/` are **frozen** — they are locked by
`tests/regression/test_frozen_metrics.py` and must not be edited.

Directory layout
----------------

```
evaluate/results/v2/
├── matrix/
│   └── <run_id>/                 # one per matrix invocation
│       ├── config.yaml           # resolved config snapshot
│       ├── run_metadata.json     # host, git sha, timestamp, seed
│       ├── per_run.jsonl         # one JSON line per (backbone, rules, frac, dataset, fold)
│       ├── aggregate.json        # mean±std across folds
│       └── logs/
├── baseline_audit/
│   └── <run_id>/
│       └── baseline_audit.json   # fair-recipe EfficientNet re-audit (Step 4)
├── statistics/
│   ├── per_class_bootstrap_ci.json
│   ├── holm_bonferroni_mcnemar.json
│   ├── dietterich_5x2cv.json
│   └── friedman_nemenyi.json
├── pdt/
│   ├── threshold_sweep.json      # Step 6a: ROC/PR, spec@90sens, etc.
│   ├── few_shot_<k>.json         # Step 6b: k ∈ {5,10,25,50}
│   └── calibration.json          # Step 6c: temperature + Platt
└── eml/
    ├── india_2025_headline.json
    ├── sensitivity_tornado.json  # ±25%, ±50% per cost
    └── tornado_<disease>.png
```

`per_run.jsonl` schema (one object per line)
-------------------------------------------

```json
{
  "run_id": "string",
  "backbone": "yolov8n-cls | yolov8s-cls | efficientnet_b0 | convnext_tiny | mobilenetv3_small | vit_b_16",
  "rule_engine": "none | handcrafted | learned_tree | llm_generated",
  "train_fraction": 1.0,
  "dataset": "indian21 | plantvillage_subset | plantdoc_subset | pdt",
  "seed": 42,
  "fold": 0,
  "n_train": 4364,
  "n_val":   935,
  "n_test":  935,
  "metrics": {
    "accuracy": 0.9615,
    "macro_f1": 0.9618,
    "mcc": 0.9596,
    "per_class_f1": {"wheat_yellow_rust": 0.99, "...": 0.0}
  },
  "latency_ms": {"mean": 15.4, "p50": 14.0, "p95": 18.2},
  "confusion_matrix_path": "relative/path/to/cm.npy",
  "predictions_csv_path":  "relative/path/to/preds.csv",
  "trained_at": "2026-04-17T12:00:00+05:30",
  "git_sha": "abcdef1",
  "training_recipe": "docs/training_recipe.md@v1",
  "status": "ok | smoke | skipped | failed",
  "notes": "free-form string"
}
```

`aggregate.json` schema
-----------------------

```json
{
  "run_id": "string",
  "axes": ["backbone", "rule_engine", "train_fraction", "dataset"],
  "aggregates": [
    {
      "cell": {"backbone": "yolov8n-cls", "rule_engine": "handcrafted",
               "train_fraction": 1.0, "dataset": "indian21"},
      "n_folds": 5,
      "accuracy":  {"mean": 0.9572, "std": 0.004, "ci95": [0.9490, 0.9650]},
      "macro_f1":  {"mean": 0.9574, "std": 0.004, "ci95": [0.9490, 0.9650]}
    }
  ]
}
```

Versioning
----------

Any breaking change to this schema must:
1. Bump the `schema_version` field (to be added when first breaking change lands).
2. Update this document in the same commit.
3. Add a migration note in `CHANGELOG_RESEARCH_UPGRADE.md`.
