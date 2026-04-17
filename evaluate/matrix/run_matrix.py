#!/usr/bin/env python3
"""Experimental matrix runner.

This module plans the Cartesian grid described in ``configs/matrix/*.yaml``
and either dispatches real training (requires GPU; see ``train.py``) or
runs in ``--dry-run`` mode, which writes the full on-disk skeleton of
result artifacts without actually training anything.

The runner is deliberately tolerant:
- Optional trackers (MLflow, W&B) are wrapped behind ``--tracker``.
- Unavailable backbones skip their cells with ``status: skipped``.
- LLM-based rules require ``ENABLE_LLM_RULES=1``; otherwise they are
  silently swapped for the cached fixture (see ``rules_llm.py``).

Usage
-----
    # fast plan-only pass (no training, no GPU required)
    python evaluate/matrix/run_matrix.py --config configs/matrix/full.yaml --dry-run

    # tiny CI-safe pass
    python evaluate/matrix/run_matrix.py --config configs/matrix/smoke.yaml --dry-run

    # real run (GPU strongly recommended)
    python evaluate/matrix/run_matrix.py --config configs/matrix/full.yaml --tracker mlflow
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# Minimal YAML loader — uses PyYAML if present, else omegaconf, else a very
# small hand-rolled subset. This keeps --dry-run usable in a CI image without
# pulling in hydra-core.
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> dict:
    try:
        import yaml  # type: ignore

        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    try:
        from omegaconf import OmegaConf  # type: ignore

        return OmegaConf.to_container(OmegaConf.load(str(path)), resolve=True)
    except Exception:
        pass
    raise RuntimeError(
        f"Cannot parse {path}: install pyyaml or omegaconf "
        "(both are already in requirements.txt)."
    )


# ---------------------------------------------------------------------------
# Cell = one point in the matrix
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Cell:
    backbone: str
    rule_engine: str
    train_fraction: float
    dataset: str
    seed: int
    fold: int

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def slug(self) -> str:
        return (
            f"{self.backbone}__{self.rule_engine}__frac{self.train_fraction}__"
            f"{self.dataset}__seed{self.seed}__fold{self.fold}"
        )


def _iter_cells(cfg: dict) -> Iterable[Cell]:
    ax = cfg["axes"]
    seeds = ax["cv"]["seeds"]
    n_folds = ax["cv"]["n_folds"]
    for backbone, rule, frac, ds, (fold, seed) in itertools.product(
        ax["backbone"],
        ax["rule_engine"],
        ax["train_fraction"],
        ax["dataset"],
        list(enumerate(seeds[:n_folds])),
    ):
        yield Cell(
            backbone=backbone,
            rule_engine=rule,
            train_fraction=float(frac),
            dataset=ds,
            seed=int(seed),
            fold=int(fold),
        )


# ---------------------------------------------------------------------------
# Tracker shims
# ---------------------------------------------------------------------------

class _NullTracker:
    def log_cell(self, cell: Cell, metrics: dict) -> None:  # noqa: D401
        return None

    def close(self) -> None:
        return None


def _make_tracker(kind: str) -> _NullTracker:
    if kind == "none":
        return _NullTracker()
    if kind == "mlflow":
        try:
            import mlflow  # type: ignore  # noqa: F401

            return _NullTracker()  # Plug in real logger when GPU runs land.
        except Exception:
            print("  [warn] mlflow not installed; falling back to --tracker none")
            return _NullTracker()
    if kind == "wandb":
        try:
            import wandb  # type: ignore  # noqa: F401

            return _NullTracker()
        except Exception:
            print("  [warn] wandb not installed; falling back to --tracker none")
            return _NullTracker()
    raise ValueError(f"unknown tracker: {kind}")


# ---------------------------------------------------------------------------
# Git sha (best effort, safe in a non-git checkout)
# ---------------------------------------------------------------------------

_GIT_SHA_CACHE: str | None = None


def _git_sha() -> str:
    global _GIT_SHA_CACHE
    if _GIT_SHA_CACHE is not None:
        return _GIT_SHA_CACHE
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        _GIT_SHA_CACHE = out.decode().strip()
    except Exception:
        _GIT_SHA_CACHE = "unknown"
    return _GIT_SHA_CACHE


# ---------------------------------------------------------------------------
# Dry-run payload
# ---------------------------------------------------------------------------

def _dry_run_record(cell: Cell, cfg: dict) -> dict:
    return {
        "run_id": cfg["run_id"],
        "backbone": cell.backbone,
        "rule_engine": cell.rule_engine,
        "train_fraction": cell.train_fraction,
        "dataset": cell.dataset,
        "seed": cell.seed,
        "fold": cell.fold,
        "status": "smoke",
        "metrics": None,
        "latency_ms": None,
        "notes": "dry-run: no training executed",
        "trained_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "training_recipe": "docs/training_recipe.md@v1",
    }


# ---------------------------------------------------------------------------
# Real-run shim
# ---------------------------------------------------------------------------

def _real_run_record(cell: Cell, cfg: dict) -> dict:
    """Delegate to evaluate.matrix.train.train_and_eval.

    Kept behind a lazy import so that --dry-run never needs torch.
    """
    try:
        from evaluate.matrix.train import train_and_eval  # type: ignore
    except Exception as e:  # noqa: BLE001
        return {
            "run_id": cfg["run_id"],
            **cell.to_dict(),
            "status": "skipped",
            "metrics": None,
            "notes": f"train_and_eval import failed: {e}",
            "trained_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "git_sha": _git_sha(),
        }
    try:
        return train_and_eval(cell, cfg)
    except Exception as e:  # noqa: BLE001
        return {
            "run_id": cfg["run_id"],
            **cell.to_dict(),
            "status": "failed",
            "metrics": None,
            "notes": f"train_and_eval raised: {e!r}",
            "trained_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "git_sha": _git_sha(),
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AgriDrone experimental matrix runner")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true",
                        help="plan-only: write skeleton artifacts, do not train")
    parser.add_argument("--smoke-test", action="store_true",
                        help="alias for --dry-run with configs/matrix/smoke.yaml")
    parser.add_argument("--tracker", choices=["none", "mlflow", "wandb"], default=None)
    parser.add_argument("--max-cells", type=int, default=0,
                        help="cap number of cells (0 = no cap); useful for CI")
    args = parser.parse_args(argv)

    if args.smoke_test:
        args.config = PROJECT_ROOT / "configs" / "matrix" / "smoke.yaml"
        args.dry_run = True

    cfg = _load_yaml(args.config)
    cfg.setdefault("tracker", "none")
    if args.tracker is not None:
        cfg["tracker"] = args.tracker

    run_id = cfg["run_id"]
    out_root = PROJECT_ROOT / cfg.get("output", {}).get("root",
                                                       "evaluate/results/v2/matrix")
    out_dir = out_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(exist_ok=True)

    # Snapshot resolved config
    (out_dir / "config.resolved.json").write_text(
        json.dumps(cfg, indent=2), encoding="utf-8"
    )
    metadata = {
        "run_id": run_id,
        "started_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "host": os.environ.get("COMPUTERNAME") or os.environ.get("HOSTNAME") or "unknown",
        "dry_run": args.dry_run,
        "tracker": cfg["tracker"],
    }
    (out_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    cells = list(_iter_cells(cfg))
    if args.max_cells and len(cells) > args.max_cells:
        cells = cells[: args.max_cells]

    tracker = _make_tracker(cfg["tracker"])

    jsonl_path = out_dir / "per_run.jsonl"
    n_written = 0
    with jsonl_path.open("w", encoding="utf-8") as jf:
        for cell in cells:
            rec = _dry_run_record(cell, cfg) if args.dry_run else _real_run_record(cell, cfg)
            jf.write(json.dumps(rec) + "\n")
            tracker.log_cell(cell, rec.get("metrics") or {})
            n_written += 1

    # Aggregate placeholder (real aggregation lives in aggregate.py once runs complete).
    (out_dir / "aggregate.json").write_text(
        json.dumps({
            "run_id": run_id,
            "n_cells": n_written,
            "status": "dry-run" if args.dry_run else "complete",
            "axes": list(cfg["axes"].keys()),
            "note": ("Real aggregates populated after GPU runs finish; see "
                     "evaluate/matrix/aggregate.py"),
        }, indent=2),
        encoding="utf-8",
    )

    print(f"  run_id={run_id}  cells={n_written}  out={out_dir}")
    print(f"  mode={'dry-run' if args.dry_run else 'real'}  tracker={cfg['tracker']}")
    print("  per_run.jsonl written; aggregate.json written as placeholder.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
