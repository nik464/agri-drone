"""Shared training entrypoint used by the matrix runner.

Design goals
------------
* **One recipe, all backbones.** Identical augmentation, LR schedule, optimizer,
  epoch budget across {YOLOv8n-cls, YOLOv8s-cls, EfficientNet-B0, ConvNeXt-Tiny,
  MobileNetV3-Small, ViT-B/16}. Documented in ``docs/training_recipe.md``.
* **Non-breaking.** Does not import from or mutate the existing
  ``src/agridrone`` training paths. Lives entirely under ``evaluate/matrix/``.
* **CPU-safe imports.** Torch/torchvision imports happen inside ``train_and_eval``
  so that ``--dry-run`` on a CPU-only CI image never loads them.
* **GPU-required for real runs.** The body of ``train_and_eval`` is a thin
  dispatch table; concrete training code can be filled in on a GPU host via
  the hook points below.

The returned record always conforms to ``docs/results_schema.md``.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any

# Public API
__all__ = ["train_and_eval", "BACKBONE_REGISTRY"]


BACKBONE_REGISTRY = {
    "yolov8n-cls":         {"family": "yolo",        "params": 1_440_000},
    "yolov8s-cls":         {"family": "yolo",        "params": 6_360_000},
    "efficientnet_b0":     {"family": "torchvision", "params": 4_034_449},
    "convnext_tiny":       {"family": "torchvision", "params": 28_589_128},
    "mobilenetv3_small":   {"family": "torchvision", "params": 2_542_856},
    "vit_b_16":            {"family": "torchvision", "params": 86_567_656},
}


def train_and_eval(cell, cfg: dict) -> dict:
    """Train one matrix cell, return a record that matches the v2 schema.

    This function is intentionally a stub for the research-upgrade branch:
    it validates the cell, selects the correct backbone family, and returns
    a ``status: "skipped"`` record with ``notes`` explaining the dependency
    that must be available on the target machine.

    To enable real training on a GPU host, fill in the branches below (clearly
    marked ``TODO-GPU``) using the shared recipe specified in ``cfg[
    "training_recipe"]``.
    """
    backbone = cell.backbone
    rule = cell.rule_engine

    if backbone not in BACKBONE_REGISTRY:
        return _record(cell, cfg, status="failed",
                       notes=f"unknown backbone: {backbone}")

    family = BACKBONE_REGISTRY[backbone]["family"]

    try:
        import torch  # noqa: F401
    except Exception as e:  # noqa: BLE001
        return _record(cell, cfg, status="skipped",
                       notes=f"torch not installed on this host: {e}")

    # ---- TODO-GPU: actual training dispatch -------------------------------
    # if family == "yolo":
    #     from ultralytics import YOLO
    #     ...
    # elif family == "torchvision":
    #     from torchvision.models import efficientnet_b0, convnext_tiny, ...
    #     ...
    # Evaluate, then apply the rule engine if rule != "none".
    # ----------------------------------------------------------------------

    return _record(
        cell, cfg,
        status="skipped",
        notes=(f"real training for backbone={backbone} family={family} "
               f"rule={rule} not executed on this host; run on a GPU machine "
               f"with CUDA-enabled torch + ultralytics + torchvision."),
    )


def _record(cell, cfg: dict, *, status: str, notes: str,
            metrics: dict | None = None, latency: dict | None = None) -> dict:
    return {
        "run_id": cfg["run_id"],
        "backbone": cell.backbone,
        "rule_engine": cell.rule_engine,
        "train_fraction": cell.train_fraction,
        "dataset": cell.dataset,
        "seed": cell.seed,
        "fold": cell.fold,
        "status": status,
        "metrics": metrics,
        "latency_ms": latency,
        "notes": notes,
        "trained_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "training_recipe": "docs/training_recipe.md@v1",
    }
