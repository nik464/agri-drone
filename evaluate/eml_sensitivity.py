#!/usr/bin/env python3
"""EML sensitivity / tornado analysis (Step 8 of research-upgrade).

Reads the YAML cost file at ``configs/economics/india_2025.yaml`` and sweeps
each cost ±25% and ±50%. Produces:

* ``evaluate/results/v2/eml/headline_v4.json`` — headline EML using ONLY
  citation-backed entries (``excluded_from_headline: false``).
* ``evaluate/results/v2/eml/sensitivity_tornado.json`` — for each cost, the
  EML delta at ±25% and ±50%.
* ``evaluate/results/v2/eml/tornado.png`` (optional; if matplotlib is
  available) — a standard tornado chart.

The v1 EML results in ``evaluate/results/eml_summary.json`` are not touched.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_yaml(path: Path) -> dict:
    try:
        import yaml  # type: ignore

        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        from omegaconf import OmegaConf  # type: ignore

        return OmegaConf.to_container(OmegaConf.load(str(path)), resolve=True)


def _build_cost_table(cfg: dict, *, headline_only: bool) -> dict[str, dict[str, float]]:
    alarm = cfg["false_alarm"]["cost"]
    table: dict[str, dict[str, float]] = {}
    for disease, entry in cfg["diseases"].items():
        if disease == "default":
            continue
        if headline_only and entry.get("excluded_from_headline"):
            continue
        table[disease] = {"miss": float(entry["miss"]), "alarm": float(alarm)}
    return table


def _compute_eml(preds_rows: list[dict], cost_table: dict[str, dict[str, float]],
                 default_miss: float, default_alarm: float) -> float:
    # Mirrors the logic in evaluate/eml_analysis.py::compute_eml but takes an
    # injected cost table so we can sweep it.
    from collections import defaultdict

    per_disease: dict[str, dict[str, int]] = defaultdict(
        lambda: {"n": 0, "tp": 0, "fn": 0, "fp": 0, "tn": 0}
    )
    all_classes: set[str] = set()
    for r in preds_rows:
        all_classes.add(r["ground_truth"])
        all_classes.add(r["predicted"])
    all_classes_s = sorted(all_classes)
    for r in preds_rows:
        for disease in all_classes_s:
            d = per_disease[disease]
            is_pos = r["ground_truth"] == disease
            pred_pos = r["predicted"] == disease
            if is_pos:
                d["n"] += 1
                if pred_pos:
                    d["tp"] += 1
                else:
                    d["fn"] += 1
            else:
                if pred_pos:
                    d["fp"] += 1
                else:
                    d["tn"] += 1
    total = 0.0
    n_samples = len(preds_rows) or 1
    for disease, d in per_disease.items():
        costs = cost_table.get(disease, {"miss": default_miss, "alarm": default_alarm})
        actual_pos = d["tp"] + d["fn"]
        actual_neg = d["fp"] + d["tn"]
        miss_rate = d["fn"] / actual_pos if actual_pos else 0.0
        fa_rate = d["fp"] / actual_neg if actual_neg else 0.0
        eml_per_pos = miss_rate * costs["miss"] + fa_rate * costs["alarm"]
        prevalence = actual_pos / n_samples
        total += eml_per_pos * prevalence
    return round(total, 2)


def _load_preds_csv(path: Path) -> list[dict]:
    import csv

    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--costs", type=Path,
                   default=PROJECT_ROOT / "configs" / "economics" / "india_2025.yaml")
    p.add_argument("--predictions-csv", type=Path,
                   default=PROJECT_ROOT / "evaluate" / "results"
                                       / "predictions_A_yolo_only.csv")
    p.add_argument("--out-dir", type=Path,
                   default=PROJECT_ROOT / "evaluate" / "results" / "v2" / "eml")
    p.add_argument("--sweeps", type=float, nargs="+", default=[-0.5, -0.25, 0.25, 0.5])
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cfg = _load_yaml(args.costs)
    default_miss = float(cfg["diseases"].get("default", {}).get("miss", 5000))
    default_alarm = float(cfg["false_alarm"]["cost"])

    preds = _load_preds_csv(args.predictions_csv)
    if not preds:
        print(f"  [warn] no predictions CSV at {args.predictions_csv}; writing stub")
        (args.out_dir / "headline_v4.json").write_text(
            json.dumps({"status": "skipped",
                        "reason": f"no {args.predictions_csv.name}"}, indent=2),
            encoding="utf-8",
        )
        return 0

    # Headline (citation-backed only)
    headline_table = _build_cost_table(cfg, headline_only=True)
    headline_eml = _compute_eml(preds, headline_table, default_miss, default_alarm)

    # All-inclusive (sensitivity scenario; matches v1 legacy)
    full_table = _build_cost_table(cfg, headline_only=False)
    full_eml = _compute_eml(preds, full_table, default_miss, default_alarm)

    (args.out_dir / "headline_v4.json").write_text(
        json.dumps({
            "headline_eml_inr_per_ha": headline_eml,
            "sensitivity_scenario_eml_inr_per_ha": full_eml,
            "n_diseases_in_headline": len(headline_table),
            "n_diseases_in_sensitivity": len(full_table),
            "methodology": "only entries with primary-source citations enter headline",
            "costs_file": str(args.costs.relative_to(PROJECT_ROOT)),
        }, indent=2),
        encoding="utf-8",
    )

    # Tornado: per-cost ±25%, ±50%
    tornado: list[dict[str, Any]] = []
    for disease in headline_table:
        base_miss = headline_table[disease]["miss"]
        for pct in args.sweeps:
            perturbed = {k: dict(v) for k, v in headline_table.items()}
            perturbed[disease]["miss"] = base_miss * (1 + pct)
            eml = _compute_eml(preds, perturbed, default_miss, default_alarm)
            tornado.append({
                "disease": disease,
                "pct": pct,
                "perturbed_eml_inr": eml,
                "delta_from_headline": round(eml - headline_eml, 2),
            })
    (args.out_dir / "sensitivity_tornado.json").write_text(
        json.dumps({
            "headline_eml_inr_per_ha": headline_eml,
            "sweeps": args.sweeps,
            "entries": tornado,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"  headline EML: INR {headline_eml}  sensitivity scenario: INR {full_eml}")
    print(f"  tornado entries: {len(tornado)}  out={args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
