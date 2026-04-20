#!/usr/bin/env python3
"""
fix_statistics.py — Regenerate broken statistical artefacts.

Fixes:
  1. mcnemar.json   — was a 21-obs stub (B vs C on dry-run data, p=1.0).
                       Now runs A vs B on the real 934-image test set.
  2. Holm-Bonferroni — per-class McNemar with Holm-Bonferroni correction
                       for 21 simultaneous tests.
  3. n=935 → 934    — corrects off-by-one in ablation_summary.json.

Reads:
  evaluate/results/predictions_A_yolo_only.csv   (Config A: YOLO only)
  evaluate/results/predictions_B_yolo_rules.csv   (Config B: YOLO + rules)

Writes:
  evaluate/results/mcnemar_A_vs_B.json            (corrected McNemar)
  evaluate/results/v2/statistics/holm_bonferroni_mcnemar.json
  evaluate/results/ablation_summary.json          (n_test_images 935→934)

Usage:
    python evaluate/fix_statistics.py
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "evaluate" / "results"
V2_STATS_DIR = RESULTS_DIR / "v2" / "statistics"


def load_preds(csv_path: Path) -> list[dict]:
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "image": row["image"],
                "ground_truth": row["ground_truth"],
                "predicted": row["predicted"],
                "correct": int(row["correct"]),
            })
    return rows


def mcnemar_chi2(n01: int, n10: int) -> tuple[float, float]:
    """McNemar chi² with continuity correction. Returns (chi2, p_value)."""
    disc = n01 + n10
    if disc == 0:
        return 0.0, 1.0
    chi2 = (abs(n01 - n10) - 1) ** 2 / disc
    # p from chi2(1) survival function using error function
    z = math.sqrt(chi2) if chi2 > 0 else 0
    p = 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))
    return chi2, max(p, 0.0)


def main() -> int:
    # ── Load predictions ──
    pa_path = RESULTS_DIR / "predictions_A_yolo_only.csv"
    pb_path = RESULTS_DIR / "predictions_B_yolo_rules.csv"

    if not pa_path.exists() or not pb_path.exists():
        print(f"ERROR: Need both {pa_path.name} and {pb_path.name}")
        return 1

    preds_a = load_preds(pa_path)
    preds_b = load_preds(pb_path)
    assert len(preds_a) == len(preds_b), "Prediction files must have same length"
    n = len(preds_a)
    print(f"Loaded {n} paired predictions (A vs B)")

    # ── 1. Global McNemar (A vs B) ──
    n11, n10, n01, n00 = 0, 0, 0, 0
    for a, b in zip(preds_a, preds_b):
        a_ok, b_ok = a["correct"], b["correct"]
        if a_ok and b_ok:
            n11 += 1
        elif a_ok and not b_ok:
            n10 += 1
        elif not a_ok and b_ok:
            n01 += 1
        else:
            n00 += 1

    chi2, p_val = mcnemar_chi2(n01, n10)
    mcnemar_result = {
        "comparison": "A_yolo_only vs B_yolo_rules",
        "n_test_images": n,
        "n11_both_correct": n11,
        "n10_A_correct_B_wrong": n10,
        "n01_A_wrong_B_correct": n01,
        "n00_both_wrong": n00,
        "discordant_pairs": n01 + n10,
        "chi2_statistic": round(chi2, 4),
        "p_value": round(p_val, 6),
        "significant_005": p_val < 0.05,
        "significant_001": p_val < 0.01,
        "method": "chi2_continuity_corrected",
    }

    out = RESULTS_DIR / "mcnemar_A_vs_B.json"
    out.write_text(json.dumps(mcnemar_result, indent=2), encoding="utf-8")
    sig = "***" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns")
    print(f"\nGlobal McNemar A vs B: chi2={chi2:.4f}, p={p_val:.6f} {sig}")
    print(f"  Discordant: n01={n01}, n10={n10}, total={n01+n10}")
    print(f"  → {out}")

    # ── 2. Per-class McNemar with Holm-Bonferroni ──
    classes = sorted(set(a["ground_truth"] for a in preds_a))
    k = len(classes)
    print(f"\nPer-class McNemar ({k} classes) with Holm-Bonferroni correction:")

    per_class: dict[str, dict] = {}
    raw_p_values: list[tuple[str, float]] = []

    for cls in classes:
        cls_n01, cls_n10 = 0, 0
        cls_n11, cls_n00 = 0, 0
        for a, b in zip(preds_a, preds_b):
            if a["ground_truth"] != cls:
                continue
            a_ok = a["correct"]
            b_ok = b["correct"]
            if a_ok and b_ok:
                cls_n11 += 1
            elif a_ok and not b_ok:
                cls_n10 += 1
            elif not a_ok and b_ok:
                cls_n01 += 1
            else:
                cls_n00 += 1

        cls_chi2, cls_p = mcnemar_chi2(cls_n01, cls_n10)
        per_class[cls] = {
            "n_samples": cls_n11 + cls_n10 + cls_n01 + cls_n00,
            "n01": cls_n01,
            "n10": cls_n10,
            "discordant": cls_n01 + cls_n10,
            "chi2": round(cls_chi2, 4),
            "p_raw": round(cls_p, 6),
        }
        raw_p_values.append((cls, cls_p))

    # Holm-Bonferroni correction
    sorted_p = sorted(raw_p_values, key=lambda x: x[1])
    for rank, (cls, p_raw) in enumerate(sorted_p, 1):
        adjusted = min(1.0, p_raw * (k - rank + 1))
        per_class[cls]["p_adjusted"] = round(adjusted, 6)
        per_class[cls]["significant_005_corrected"] = adjusted < 0.05
        per_class[cls]["holm_rank"] = rank

    # Summary
    n_sig = sum(1 for v in per_class.values() if v["significant_005_corrected"])
    holm_result = {
        "method": "Holm-Bonferroni corrected per-class McNemar",
        "comparison": "A_yolo_only vs B_yolo_rules",
        "n_classes": k,
        "n_significant_corrected": n_sig,
        "n_test_images": n,
        "per_class": per_class,
    }

    V2_STATS_DIR.mkdir(parents=True, exist_ok=True)
    holm_out = V2_STATS_DIR / "holm_bonferroni_mcnemar.json"
    holm_out.write_text(json.dumps(holm_result, indent=2), encoding="utf-8")
    print(f"  {n_sig}/{k} classes significant after Holm-Bonferroni correction")
    print(f"  → {holm_out}")

    for cls in classes:
        d = per_class[cls]
        sig_mark = "*" if d["significant_005_corrected"] else ""
        if d["discordant"] > 0:
            print(f"  {cls:40s}: disc={d['discordant']:3d}  p_raw={d['p_raw']:.4f}  p_adj={d['p_adjusted']:.4f} {sig_mark}")

    # ── 3. Fix n=935→934 in ablation_summary.json ──
    ablation_path = RESULTS_DIR / "ablation_summary.json"
    if ablation_path.exists():
        ablation = json.loads(ablation_path.read_text(encoding="utf-8"))
        if ablation.get("n_test_images") == 935:
            ablation["n_test_images"] = 934
            ablation["_fix_note"] = "Corrected from 935 to 934 (off-by-one fix)"
            ablation_path.write_text(json.dumps(ablation, indent=2), encoding="utf-8")
            print(f"\n  Fixed n_test_images: 935 → 934 in {ablation_path.name}")
        elif ablation.get("n_test_images") == 934:
            print(f"\n  n_test_images already correct (934) in {ablation_path.name}")
        else:
            actual_n = ablation.get("n_test_images")
            print(f"\n  n_test_images = {actual_n} (unexpected, not modifying)")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
