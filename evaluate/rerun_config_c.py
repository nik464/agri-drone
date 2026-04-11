#!/usr/bin/env python3
"""Quick re-run of Config C only after bugfix (primary_disease → disease_key)."""
import sys, csv, json, warnings
from pathlib import Path
import numpy as np
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "evaluate"))

# Force reimport (clear any cached module)
if "ablation_study" in sys.modules:
    del sys.modules["ablation_study"]

from ablation_study import (
    load_test_set, run_config_c, compute_metrics, write_predictions_csv,
    SEVERITY_TIERS, get_tier_weight,
)

# Verify the fix is loaded
import inspect
src = inspect.getsource(run_config_c)
assert "disease_key" in src, f"BUG: run_config_c still uses primary_disease!\n{src[:500]}"
print("VERIFIED: run_config_c uses disease_key")

def main():
    test_dir = PROJECT_ROOT / "data" / "training" / "test"
    output_dir = PROJECT_ROOT / "evaluate" / "results"
    model_path = PROJECT_ROOT / "models" / "india_agri_cls_21class_backup.pt"

    # Load classes from model (needed for compute_metrics)
    from ultralytics import YOLO
    model = YOLO(str(model_path), task="classify")
    classes = sorted(model.names.values())
    print(f"Classes: {len(classes)}")

    test_images = load_test_set(test_dir)
    print(f"Test images: {len(test_images)}")

    # Run Config C
    preds_c = []
    for i, img_info in enumerate(test_images):
        image_bgr = cv2.imread(img_info["path"])
        if image_bgr is None:
            continue
        result = run_config_c(image_bgr, img_info["crop_type"])
        preds_c.append({
            "path": img_info["path"],
            "ground_truth": img_info["ground_truth"],
            "predicted": result["predicted"],
            "confidence": result["confidence"],
            "latency_ms": result["latency_ms"],
        })
        if (i + 1) % 100 == 0:
            acc = sum(1 for p in preds_c if p["ground_truth"] == p["predicted"]) / len(preds_c)
            print(f"  [{i+1}/{len(test_images)}] running acc: {acc:.1%}")
        if i < 5:
            print(f"  DEBUG [{i}]: gt={preds_c[-1]['ground_truth']}, pred={preds_c[-1]['predicted']}, conf={preds_c[-1]['confidence']}")

    metrics_c = compute_metrics(preds_c, classes)
    mean_lat_c = np.mean([p["latency_ms"] for p in preds_c])
    print(f"\nConfig C — Accuracy: {metrics_c['accuracy']:.1%}, "
          f"RWA: {metrics_c['rwa']:.1%}, "
          f"Macro-F1: {metrics_c['macro_f1']:.3f}, "
          f"MCC: {metrics_c['mcc']:.3f}, "
          f"Mean latency: {mean_lat_c:.0f}ms")

    write_predictions_csv(preds_c, "C", output_dir / "predictions_C_rules_only.csv")

    # Update ablation_summary.json with new Config C numbers
    summary_path = output_dir / "ablation_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["config_C_accuracy"] = round(metrics_c["accuracy"], 4)
    summary["config_C_RWA"] = round(metrics_c["rwa"], 4)
    summary["config_C_macro_f1"] = round(metrics_c["macro_f1"], 4)
    summary["config_C_mcc"] = round(metrics_c["mcc"], 4)
    summary["config_C_mean_latency_ms"] = round(mean_lat_c, 1)
    summary["safety_gap_C"] = round(metrics_c["rwa"] - metrics_c["accuracy"], 4) if metrics_c["accuracy"] > 0 else 0.0
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nUpdated {summary_path}")
    print(json.dumps({k: v for k, v in summary.items() if "C" in str(k)}, indent=2))

if __name__ == "__main__":
    main()
