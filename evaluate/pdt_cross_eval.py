#!/usr/bin/env python3
"""
Cross-Dataset Evaluation on PDT (Plant Disease from drone imagery) Dataset.

PDT is a detection dataset with:
  - LH/images/  → whole drone images (healthy wheat fields)
  - LL/YOLO_txt/test/images/ → cropped tiles of unhealthy wheat

We evaluate our YOLOv8n-cls model as a binary healthy/unhealthy classifier:
  - LH images: correct if model predicts healthy_wheat
  - LL images: correct if model predicts ANY disease (not healthy_wheat)

Usage:
    python evaluate/pdt_cross_eval.py \
        --dataset-dir "datasets/externals/PDT_datasets/PDT dataset/PDT dataset" \
        --model-path models/india_agri_cls.pt
"""

import argparse
import csv
import json
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def count_images(directory: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return sorted([p for p in directory.iterdir() if p.suffix.lower() in exts])


def evaluate_pdt(model, healthy_imgs: list[Path], unhealthy_imgs: list[Path],
                 healthy_class: str = "healthy_wheat") -> dict:
    """Run binary healthy/unhealthy evaluation on PDT images."""
    predictions = []
    tp = fp = tn = fn = 0
    latencies = []

    all_images = [(p, "healthy") for p in healthy_imgs] + \
                 [(p, "unhealthy") for p in unhealthy_imgs]

    for i, (img_path, gt_label) in enumerate(all_images):
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            warnings.warn(f"Cannot read {img_path}")
            continue

        t0 = time.perf_counter()
        results = model(image_bgr, verbose=False)
        latency = (time.perf_counter() - t0) * 1000
        latencies.append(latency)

        if not results or results[0].probs is None:
            predicted_class = "unknown"
            confidence = 0.0
        else:
            probs = results[0].probs
            names = model.names
            predicted_class = names[probs.top1]
            confidence = round(probs.top1conf.item(), 4)

        pred_healthy = predicted_class == healthy_class
        gt_healthy = gt_label == "healthy"

        if gt_healthy and pred_healthy:
            tn += 1   # true negative (correctly identified healthy)
            correct = True
        elif gt_healthy and not pred_healthy:
            fp += 1   # false positive (healthy predicted as diseased)
            correct = False
        elif not gt_healthy and not pred_healthy:
            tp += 1   # true positive (correctly identified diseased)
            correct = True
        else:
            fn += 1   # false negative (diseased predicted as healthy)
            correct = False

        predictions.append({
            "path": str(img_path.name),
            "ground_truth": gt_label,
            "predicted_class": predicted_class,
            "predicted_binary": "healthy" if pred_healthy else "unhealthy",
            "confidence": confidence,
            "correct": correct,
            "latency_ms": round(latency, 1),
        })

        if (i + 1) % 50 == 0:
            acc = sum(1 for p in predictions if p["correct"]) / len(predictions)
            print(f"  [{i+1}/{len(all_images)}] running acc: {acc:.1%}")

    n = len(predictions)
    accuracy = (tp + tn) / n if n > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Confidence analysis
    correct_confs = [p["confidence"] for p in predictions if p["correct"]]
    wrong_confs = [p["confidence"] for p in predictions if not p["correct"]]

    # Per-class prediction distribution
    pred_dist = defaultdict(int)
    for p in predictions:
        pred_dist[p["predicted_class"]] += 1

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall_sensitivity": round(recall, 4),
        "specificity": round(specificity, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": {
            "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        },
        "n_healthy": len(healthy_imgs),
        "n_unhealthy": len(unhealthy_imgs),
        "n_total": n,
        "mean_latency_ms": round(float(np.mean(latencies)), 1) if latencies else 0,
        "mean_confidence_correct": round(float(np.mean(correct_confs)), 4) if correct_confs else 0,
        "mean_confidence_wrong": round(float(np.mean(wrong_confs)), 4) if wrong_confs else 0,
        "confidence_gap": round(
            float(np.mean(correct_confs)) - float(np.mean(wrong_confs)), 4
        ) if correct_confs and wrong_confs else 0,
        "prediction_distribution": dict(pred_dist),
        "predictions": predictions,
    }


def main():
    parser = argparse.ArgumentParser(description="PDT Cross-Dataset Evaluation (Binary)")
    parser.add_argument("--dataset-dir", required=True,
                        help="Root of PDT dataset (containing LH/ and LL/)")
    parser.add_argument("--model-path",
                        default=str(PROJECT_ROOT / "models" / "india_agri_cls.pt"))
    parser.add_argument("--output-dir",
                        default=str(PROJECT_ROOT / "evaluate" / "results"))
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Locate LH and LL directories
    lh_dir = dataset_dir / "LH" / "images"
    ll_test_dir = dataset_dir / "LL" / "YOLO_txt" / "test" / "images"

    if not lh_dir.exists():
        print(f"ERROR: LH images not found: {lh_dir}")
        sys.exit(1)
    if not ll_test_dir.exists():
        print(f"ERROR: LL test images not found: {ll_test_dir}")
        sys.exit(1)

    healthy_imgs = count_images(lh_dir)
    unhealthy_imgs = count_images(ll_test_dir)

    print(f"PDT Dataset:")
    print(f"  Healthy (LH):   {len(healthy_imgs)} images")
    print(f"  Unhealthy (LL): {len(unhealthy_imgs)} images (test split)")
    print(f"  Total:          {len(healthy_imgs) + len(unhealthy_imgs)}")

    # Load model
    print(f"\nLoading YOLO classifier: {args.model_path}")
    from ultralytics import YOLO
    model = YOLO(args.model_path, task="classify")
    print(f"  Model classes: {list(model.names.values())}")

    # Determine healthy class
    healthy_class = "healthy_wheat"
    if healthy_class not in model.names.values():
        # Fallback: look for any healthy class
        for name in model.names.values():
            if "healthy" in name.lower():
                healthy_class = name
                break
    print(f"  Healthy class: {healthy_class}")

    # Evaluate
    print(f"\n{'='*70}")
    print(f"  PDT CROSS-DATASET EVALUATION (Binary: healthy vs unhealthy)")
    print(f"{'='*70}")

    results = evaluate_pdt(model, healthy_imgs, unhealthy_imgs, healthy_class)

    # Extract predictions for CSV
    predictions = results.pop("predictions")

    # Print results
    print(f"\n  Results:")
    print(f"    Accuracy:       {results['accuracy']:.1%}")
    print(f"    Precision:      {results['precision']:.3f}")
    print(f"    Recall/Sens:    {results['recall_sensitivity']:.3f}")
    print(f"    Specificity:    {results['specificity']:.3f}")
    print(f"    F1 Score:       {results['f1_score']:.3f}")
    print(f"    Mean latency:   {results['mean_latency_ms']:.1f} ms")
    cm = results["confusion_matrix"]
    print(f"\n    Confusion Matrix:")
    print(f"                  Pred Diseased  Pred Healthy")
    print(f"      Diseased:      {cm['TP']:>5d}         {cm['FN']:>5d}")
    print(f"      Healthy:       {cm['FP']:>5d}         {cm['TN']:>5d}")
    print(f"\n    Confidence Analysis:")
    print(f"      Correct predictions: {results['mean_confidence_correct']:.3f}")
    print(f"      Wrong predictions:   {results['mean_confidence_wrong']:.3f}")
    print(f"      Confidence gap:      {results['confidence_gap']:.3f}")
    print(f"\n    Prediction Distribution:")
    for cls, count in sorted(results["prediction_distribution"].items(),
                              key=lambda x: -x[1]):
        print(f"      {cls:30s}: {count}")

    # Save JSON
    results_path = output_dir / "cross_dataset_PDT.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n  Metrics → {results_path}")

    # Save CSV
    csv_path = output_dir / "cross_dataset_PDT_predictions.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "ground_truth", "predicted_class", "predicted_binary",
                     "confidence", "correct", "latency_ms"])
        for p in predictions:
            w.writerow([p["path"], p["ground_truth"], p["predicted_class"],
                        p["predicted_binary"], p["confidence"],
                        1 if p["correct"] else 0, p["latency_ms"]])
    print(f"  Predictions CSV → {csv_path}")


if __name__ == "__main__":
    main()
