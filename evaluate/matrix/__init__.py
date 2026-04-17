"""Experimental matrix runner (Step 3 of research-upgrade).

Config-driven, supports --dry-run (< 5s) and --tracker {none,mlflow,wandb}.
Writes per-run JSONL + aggregate JSON to evaluate/results/v2/matrix/<run_id>/.

See docs/results_schema.md for the output schema.
See configs/matrix/full.yaml and configs/matrix/smoke.yaml for examples.
"""
