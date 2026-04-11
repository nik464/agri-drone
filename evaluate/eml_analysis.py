#!/usr/bin/env python3
"""
Experiment 4 — Expected Monetary Loss (EML) Analysis

Computes per-disease and aggregate EML for Config A (YOLO-only)
and Config B (YOLO+Rules+Ensemble) using field-verified cost data.

EML per sample = P(miss) * C_miss + P(false_alarm) * C_alarm
Aggregate = mean over all test samples

Usage:
    python evaluate/eml_analysis.py
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ── Cost table (INR per hectare) ──
COST_TABLE = {
    "wheat_fusarium_head_blight": {"miss": 17250, "alarm": 640},
    "wheat_blast":                {"miss": 22000, "alarm": 640},
    "rice_blast":                 {"miss": 22000, "alarm": 640},
    "wheat_black_rust":           {"miss": 18500, "alarm": 640},
    "rice_bacterial_blight":      {"miss": 12000, "alarm": 640},
    "rice_brown_spot":            {"miss": 6000,  "alarm": 640},
    "healthy_wheat":              {"miss": 0,     "alarm": 640},
    "healthy_rice":               {"miss": 0,     "alarm": 640},
}
DEFAULT_COST = {"miss": 5000, "alarm": 640}

# Severity tiers for weighting
SEVERITY_TIERS = {
    "wheat_fusarium_head_blight": ("critical", 10),
    "wheat_yellow_rust":         ("critical", 10),
    "wheat_black_rust":          ("critical", 10),
    "wheat_blast":               ("critical", 10),
    "rice_blast":                ("critical", 10),
    "rice_bacterial_blight":     ("critical", 10),
    "wheat_brown_rust":          ("high", 5),
    "wheat_septoria":            ("high", 5),
    "wheat_leaf_blight":         ("high", 5),
    "rice_sheath_blight":        ("high", 5),
    "wheat_root_rot":            ("high", 5),
    "rice_leaf_scald":           ("high", 5),
    "wheat_powdery_mildew":      ("moderate", 2),
    "wheat_tan_spot":            ("moderate", 2),
    "wheat_aphid":               ("moderate", 2),
    "wheat_mite":                ("moderate", 2),
    "wheat_smut":                ("moderate", 2),
    "wheat_stem_fly":            ("moderate", 2),
    "rice_brown_spot":           ("moderate", 2),
    "healthy_wheat":             ("healthy", 1),
    "healthy_rice":              ("healthy", 1),
}


def get_cost(disease_key: str) -> dict:
    return COST_TABLE.get(disease_key, DEFAULT_COST)


def compute_eml(predictions_csv: Path) -> dict:
    """Compute per-disease and aggregate EML from a predictions CSV."""
    rows = []
    with open(predictions_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Per-disease tracking
    per_disease = defaultdict(lambda: {"n": 0, "tp": 0, "fn": 0, "fp": 0, "tn": 0})
    all_classes = set()

    for row in rows:
        gt = row["ground_truth"]
        pred = row["predicted"]
        all_classes.add(gt)
        all_classes.add(pred)

    all_classes = sorted(all_classes)

    # For each disease, compute miss/false-alarm rates
    for row in rows:
        gt = row["ground_truth"]
        pred = row["predicted"]

        for disease in all_classes:
            entry = per_disease[disease]
            is_positive = (gt == disease)
            predicted_positive = (pred == disease)

            if is_positive:
                entry["n"] += 1
                if predicted_positive:
                    entry["tp"] += 1
                else:
                    entry["fn"] += 1  # Missed this disease
            else:
                if predicted_positive:
                    entry["fp"] += 1  # False alarm for this disease
                else:
                    entry["tn"] += 1

    # Compute EML per disease
    disease_eml = {}
    total_eml = 0.0
    total_samples = len(rows)

    for disease in all_classes:
        d = per_disease[disease]
        cost = get_cost(disease)

        # Miss rate = FN / (TP + FN) = FN / actual positives
        actual_pos = d["tp"] + d["fn"]
        miss_rate = d["fn"] / actual_pos if actual_pos > 0 else 0.0

        # False alarm rate = FP / (FP + TN) = FP / actual negatives
        actual_neg = d["fp"] + d["tn"]
        fa_rate = d["fp"] / actual_neg if actual_neg > 0 else 0.0

        # EML for this disease (per sample where this disease is ground truth)
        eml_per_positive = miss_rate * cost["miss"] + fa_rate * cost["alarm"]

        # Weighted contribution to total (by prevalence)
        prevalence = actual_pos / total_samples if total_samples > 0 else 0.0
        eml_contribution = eml_per_positive * prevalence

        disease_eml[disease] = {
            "n_samples": actual_pos,
            "tp": d["tp"],
            "fn": d["fn"],
            "fp": d["fp"],
            "miss_rate": round(miss_rate, 4),
            "false_alarm_rate": round(fa_rate, 4),
            "cost_miss": cost["miss"],
            "cost_alarm": cost["alarm"],
            "eml_per_positive": round(eml_per_positive, 2),
            "eml_contribution": round(eml_contribution, 2),
            "severity": SEVERITY_TIERS.get(disease, ("moderate", 2))[0],
        }
        total_eml += eml_contribution

    return {
        "per_disease": disease_eml,
        "total_eml": round(total_eml, 2),
        "n_samples": total_samples,
    }


def main():
    results_dir = PROJECT_ROOT / "evaluate" / "results"
    pred_a = results_dir / "predictions_A_yolo_only.csv"
    pred_b = results_dir / "predictions_B_yolo_rules.csv"

    print("=" * 70)
    print("  EXPERIMENT 4: Expected Monetary Loss (EML)")
    print("=" * 70)

    print("\nComputing EML for Config A (YOLO-only)...")
    eml_a = compute_eml(pred_a)
    print(f"  Total EML(A) = ₹{eml_a['total_eml']:,.2f} / sample")

    print("\nComputing EML for Config B (YOLO+Rules+Ensemble)...")
    eml_b = compute_eml(pred_b)
    print(f"  Total EML(B) = ₹{eml_b['total_eml']:,.2f} / sample")

    delta = eml_b["total_eml"] - eml_a["total_eml"]
    pct = (delta / eml_a["total_eml"] * 100) if eml_a["total_eml"] != 0 else 0
    print(f"\n  ΔEML (B−A) = ₹{delta:,.2f} ({pct:+.1f}%)")

    # ── Per-disease comparison table ──
    print(f"\n{'Disease':<35} {'EML(A)':>10} {'EML(B)':>10} {'Δ':>10} {'Severity':>10}")
    print("-" * 80)

    comparison_rows = []
    for disease in sorted(eml_a["per_disease"].keys()):
        a = eml_a["per_disease"].get(disease, {})
        b = eml_b["per_disease"].get(disease, {})
        ea = a.get("eml_per_positive", 0)
        eb = b.get("eml_per_positive", 0)
        d = eb - ea
        sev = a.get("severity", "moderate")
        n = a.get("n_samples", 0)
        print(f"  {disease:<33} {ea:>10.2f} {eb:>10.2f} {d:>+10.2f} {sev:>10}")
        comparison_rows.append({
            "disease": disease,
            "n_samples": n,
            "eml_A": ea,
            "eml_B": eb,
            "delta": round(d, 2),
            "miss_rate_A": a.get("miss_rate", 0),
            "miss_rate_B": b.get("miss_rate", 0),
            "false_alarm_rate_A": a.get("false_alarm_rate", 0),
            "false_alarm_rate_B": b.get("false_alarm_rate", 0),
            "cost_miss": a.get("cost_miss", 0),
            "cost_alarm": a.get("cost_alarm", 0),
            "severity": sev,
        })

    # ── Critical-disease focus ──
    critical_diseases = [d for d, v in SEVERITY_TIERS.items() if v[0] == "critical"]
    crit_eml_a = sum(eml_a["per_disease"].get(d, {}).get("eml_contribution", 0) for d in critical_diseases)
    crit_eml_b = sum(eml_b["per_disease"].get(d, {}).get("eml_contribution", 0) for d in critical_diseases)

    print(f"\n  Critical diseases EML(A) = ₹{crit_eml_a:,.2f}")
    print(f"  Critical diseases EML(B) = ₹{crit_eml_b:,.2f}")

    # ── Save CSV ──
    csv_path = results_dir / "eml_comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "disease", "n_samples", "eml_A", "eml_B", "delta",
            "miss_rate_A", "miss_rate_B", "false_alarm_rate_A", "false_alarm_rate_B",
            "cost_miss", "cost_alarm", "severity",
        ])
        w.writeheader()
        w.writerows(comparison_rows)
    print(f"\n  Saved: eml_comparison.csv ({len(comparison_rows)} rows)")

    # ── Save summary JSON ──
    summary = {
        "total_eml_A": eml_a["total_eml"],
        "total_eml_B": eml_b["total_eml"],
        "delta_eml": round(delta, 2),
        "delta_pct": round(pct, 2),
        "critical_eml_A": round(crit_eml_a, 2),
        "critical_eml_B": round(crit_eml_b, 2),
        "n_test_samples": eml_a["n_samples"],
        "n_diseases": len(comparison_rows),
        "cost_table_diseases": list(COST_TABLE.keys()),
        "default_miss_cost": DEFAULT_COST["miss"],
        "default_alarm_cost": DEFAULT_COST["alarm"],
        "per_disease_A": eml_a["per_disease"],
        "per_disease_B": eml_b["per_disease"],
    }
    summary_path = results_dir / "eml_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  Saved: eml_summary.json")

    # ── Bar chart ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        diseases = [r["disease"] for r in comparison_rows if r["n_samples"] > 0]
        eml_a_vals = [r["eml_A"] for r in comparison_rows if r["n_samples"] > 0]
        eml_b_vals = [r["eml_B"] for r in comparison_rows if r["n_samples"] > 0]

        x = np.arange(len(diseases))
        width = 0.35

        fig, ax = plt.subplots(figsize=(14, 6))
        bars_a = ax.bar(x - width/2, eml_a_vals, width, label="Config A (YOLO)", color="#2196F3", alpha=0.85)
        bars_b = ax.bar(x + width/2, eml_b_vals, width, label="Config B (YOLO+Rules)", color="#FF5722", alpha=0.85)

        ax.set_ylabel("EML (₹ / positive sample)", fontsize=12)
        ax.set_title("Expected Monetary Loss per Disease", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([d.replace("wheat_", "w.").replace("rice_", "r.").replace("healthy_", "h.")
                           for d in diseases], rotation=45, ha="right", fontsize=9)
        ax.legend(fontsize=11)
        ax.grid(axis="y", alpha=0.3)

        # Annotate totals
        ax.text(0.98, 0.97, f"Total EML(A)=₹{eml_a['total_eml']:,.0f}\nTotal EML(B)=₹{eml_b['total_eml']:,.0f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9))

        fig.tight_layout()
        chart_path = results_dir / "eml_bar_chart.png"
        fig.savefig(chart_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: eml_bar_chart.png")
    except Exception as e:
        print(f"  Warning: Chart generation failed: {e}")

    print(f"\n{'=' * 70}")
    print(f"  EML ANALYSIS COMPLETE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
