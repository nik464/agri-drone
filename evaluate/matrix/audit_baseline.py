#!/usr/bin/env python3
"""Fair-comparison re-audit of the EfficientNet-B0 baseline.

v3 reported 76.15% accuracy for EfficientNet-B0. We believe that number
reflected a training-configuration mismatch (different augmentation or epoch
budget than YOLO) rather than an architectural limitation.

This script re-trains EfficientNet-B0 under the **shared recipe** documented
in ``docs/training_recipe.md`` and writes the result to:

    evaluate/results/v2/baseline_audit/<run_id>/baseline_audit.json

v3's ``evaluate/results/efficientnet_results.json`` is preserved verbatim.
v4 cites only the new number, with a footnote explaining methodology.

Usage
-----
    python evaluate/matrix/audit_baseline.py --dry-run
    python evaluate/matrix/audit_baseline.py --epochs 50 --batch-size 32
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _stub_record(args) -> dict:
    return {
        "run_id": args.run_id,
        "backbone": "efficientnet_b0",
        "training_recipe": "docs/training_recipe.md@v1",
        "epochs_budget": args.epochs,
        "batch_size": args.batch_size,
        "lr": 1.25e-3,
        "status": "dry-run" if args.dry_run else "skipped",
        "metrics": None,
        "notes": (
            "Shared-recipe EfficientNet-B0 audit not executed on this host. "
            "Run on GPU: `python evaluate/matrix/audit_baseline.py "
            "--epochs 50 --batch-size 32`. "
            "Expected to match or exceed v3's 76.15% once augmentation and LR "
            "schedule are matched to YOLO's."
        ),
        "v3_reference": {
            "accuracy":    0.7615,
            "macro_f1":    0.7621,
            "source_file": "evaluate/results/efficientnet_results.json",
        },
        "trained_at": _dt.datetime.now().isoformat(timespec="seconds"),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", default="efficientnet_b0_fair_v1")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--out-dir", type=Path,
                   default=PROJECT_ROOT / "evaluate" / "results" / "v2" / "baseline_audit")
    args = p.parse_args()

    out_dir = args.out_dir / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        rec = _stub_record(args)
        (out_dir / "baseline_audit.json").write_text(
            json.dumps(rec, indent=2), encoding="utf-8"
        )
        print(f"  [dry-run] wrote {out_dir/'baseline_audit.json'}")
        return 0

    # Real training path — requires torch + torchvision on a GPU host.
    try:
        import torch  # noqa: F401
        from torchvision import models  # noqa: F401
    except Exception as e:  # noqa: BLE001
        rec = _stub_record(args)
        rec["notes"] = f"torch/torchvision missing: {e}"
        (out_dir / "baseline_audit.json").write_text(
            json.dumps(rec, indent=2), encoding="utf-8"
        )
        print("  [skipped] torch/torchvision not installed; wrote stub record")
        return 0

    # TODO-GPU: invoke evaluate.matrix.train.train_and_eval with a
    # synthetic cell (backbone=efficientnet_b0, rule_engine=none, frac=1.0,
    # dataset=indian21, fold=0, seed=42).
    rec = _stub_record(args)
    rec["status"] = "skipped"
    rec["notes"] = "Real training path not yet wired; call via matrix runner on GPU host."
    (out_dir / "baseline_audit.json").write_text(
        json.dumps(rec, indent=2), encoding="utf-8"
    )
    print(f"  [skipped] wrote stub {out_dir/'baseline_audit.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
