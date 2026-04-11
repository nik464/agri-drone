#!/usr/bin/env python3
"""
Robustness Evaluation — Clean vs Noisy Test Set Comparison

Evaluates YOLOv8n-cls on both the original (clean) test set and the
noisy test set produced by noise_pipeline.py.  Generates:

  evaluate/results/noisy_eval.csv       — per-class metrics for both sets
  evaluate/results/robustness_summary.json  — headline numbers & deltas
  evaluate/results/robustness_report.txt    — human-readable summary

Usage:
    python evaluate/robustness_eval.py
    python evaluate/robustness_eval.py \
        --model-path models/india_agri_cls.pt \
        --clean-dir data/training/test \
        --noisy-dir evaluate/noisy_dataset \
        --output-dir evaluate/results
"""

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ════════════════════════════════════════════════════════════════
# Data loading
# ════════════════════════════════════════════════════════════════

def load_images(data_dir: Path) -> list[dict]:
    """Load image paths with ground-truth labels from folder structure."""
    images = []
    classes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    for cls in classes:
        cls_dir = data_dir / cls
        for img_path in sorted(cls_dir.glob("*")):
            if img_path.suffix.lower() in IMAGE_EXTENSIONS:
                images.append({"path": str(img_path), "ground_truth": cls})
    return images


# ════════════════════════════════════════════════════════════════
# YOLO-only inference
# ════════════════════════════════════════════════════════════════

def predict_yolo(model, image_bgr: np.ndarray) -> dict:
    """Run YOLO classifier and return top-1 prediction."""
    t0 = time.perf_counter()
    results = model(image_bgr, verbose=False)
    latency = (time.perf_counter() - t0) * 1000

    if not results or results[0].probs is None:
        return {"predicted": "unknown", "confidence": 0.0, "latency_ms": latency}

    probs = results[0].probs
    names = model.names
    return {
        "predicted": names[probs.top1],
        "confidence": round(probs.top1conf.item(), 4),
        "latency_ms": round(latency, 1),
    }


# ════════════════════════════════════════════════════════════════
# Metrics computation
# ════════════════════════════════════════════════════════════════

def compute_metrics(predictions: list[dict], classes: list[str]) -> dict:
    """Per-class precision, recall, F1 + macro aggregates."""
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    total_correct = 0

    for pred in predictions:
        gt = pred["ground_truth"]
        pr = pred["predicted"]
        if gt == pr:
            total_correct += 1
            tp[gt] += 1
        else:
            fn[gt] += 1
            fp[pr] += 1

    total = len(predictions)
    accuracy = total_correct / total if total > 0 else 0

    per_class = {}
    macro_p, macro_r, macro_f1 = 0.0, 0.0, 0.0
    for cls in classes:
        p = tp[cls] / (tp[cls] + fp[cls]) if (tp[cls] + fp[cls]) > 0 else 0
        r = tp[cls] / (tp[cls] + fn[cls]) if (tp[cls] + fn[cls]) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        support = tp[cls] + fn[cls]
        per_class[cls] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "support": support,
        }
        macro_p += p
        macro_r += r
        macro_f1 += f1

    n = len(classes)
    return {
        "accuracy": round(accuracy, 4),
        "macro_precision": round(macro_p / n, 4),
        "macro_recall": round(macro_r / n, 4),
        "macro_f1": round(macro_f1 / n, 4),
        "per_class": per_class,
        "n_samples": total,
        "n_correct": total_correct,
    }


# ════════════════════════════════════════════════════════════════
# Evaluation runner
# ════════════════════════════════════════════════════════════════

def evaluate_dataset(model, data_dir: Path, label: str) -> tuple[dict, list[dict]]:
    """Run YOLO on every image in data_dir and compute metrics."""
    images = load_images(data_dir)
    classes = sorted(set(img["ground_truth"] for img in images))

    print(f"\n{'='*60}")
    print(f"  Evaluating: {label}")
    print(f"  Directory:  {data_dir}")
    print(f"  Images:     {len(images)}")
    print(f"  Classes:    {len(classes)}")
    print(f"{'='*60}")

    predictions = []
    latencies = []
    for i, img_info in enumerate(images):
        image_bgr = cv2.imread(img_info["path"])
        if image_bgr is None:
            continue

        result = predict_yolo(model, image_bgr)
        result["ground_truth"] = img_info["ground_truth"]
        result["image_path"] = img_info["path"]
        predictions.append(result)
        latencies.append(result["latency_ms"])

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(images)}] ...")

    metrics = compute_metrics(predictions, classes)
    metrics["mean_latency_ms"] = round(np.mean(latencies), 1) if latencies else 0
    metrics["label"] = label

    print(f"\n  Results ({label}):")
    print(f"    Accuracy:   {metrics['accuracy']*100:.2f}%")
    print(f"    Macro-F1:   {metrics['macro_f1']:.4f}")
    print(f"    Macro-R:    {metrics['macro_recall']:.4f}")
    print(f"    Latency:    {metrics['mean_latency_ms']:.1f} ms/image")

    return metrics, predictions


# ════════════════════════════════════════════════════════════════
# Comparison & output
# ════════════════════════════════════════════════════════════════

def generate_comparison(
    clean_metrics: dict,
    noisy_metrics: dict,
    output_dir: Path,
):
    """Generate CSV, JSON summary, and text report comparing clean vs noisy."""
    output_dir.mkdir(parents=True, exist_ok=True)

    classes = sorted(clean_metrics["per_class"].keys())

    # ── noisy_eval.csv ──
    csv_path = output_dir / "noisy_eval.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "class", "split",
            "precision", "recall", "f1", "support",
        ])
        for cls in classes:
            c = clean_metrics["per_class"].get(cls, {})
            n = noisy_metrics["per_class"].get(cls, {})
            writer.writerow([
                cls, "clean",
                c.get("precision", 0), c.get("recall", 0),
                c.get("f1", 0), c.get("support", 0),
            ])
            writer.writerow([
                cls, "noisy",
                n.get("precision", 0), n.get("recall", 0),
                n.get("f1", 0), n.get("support", 0),
            ])
        # Macro row
        writer.writerow([
            "MACRO_AVG", "clean",
            clean_metrics["macro_precision"], clean_metrics["macro_recall"],
            clean_metrics["macro_f1"], clean_metrics["n_samples"],
        ])
        writer.writerow([
            "MACRO_AVG", "noisy",
            noisy_metrics["macro_precision"], noisy_metrics["macro_recall"],
            noisy_metrics["macro_f1"], noisy_metrics["n_samples"],
        ])
    print(f"\n  CSV saved: {csv_path}")

    # ── robustness_summary.json ──
    delta_accuracy = noisy_metrics["accuracy"] - clean_metrics["accuracy"]
    delta_f1 = noisy_metrics["macro_f1"] - clean_metrics["macro_f1"]
    delta_recall = noisy_metrics["macro_recall"] - clean_metrics["macro_recall"]

    per_class_delta = {}
    worst_drops = []
    for cls in classes:
        c_f1 = clean_metrics["per_class"].get(cls, {}).get("f1", 0)
        n_f1 = noisy_metrics["per_class"].get(cls, {}).get("f1", 0)
        c_r = clean_metrics["per_class"].get(cls, {}).get("recall", 0)
        n_r = noisy_metrics["per_class"].get(cls, {}).get("recall", 0)
        d_f1 = round(n_f1 - c_f1, 4)
        d_r = round(n_r - c_r, 4)
        per_class_delta[cls] = {"delta_f1": d_f1, "delta_recall": d_r}
        worst_drops.append((cls, d_f1, d_r))

    worst_drops.sort(key=lambda x: x[1])  # most negative first

    summary = {
        "clean": {
            "accuracy": clean_metrics["accuracy"],
            "macro_f1": clean_metrics["macro_f1"],
            "macro_recall": clean_metrics["macro_recall"],
            "macro_precision": clean_metrics["macro_precision"],
            "n_samples": clean_metrics["n_samples"],
            "mean_latency_ms": clean_metrics["mean_latency_ms"],
        },
        "noisy": {
            "accuracy": noisy_metrics["accuracy"],
            "macro_f1": noisy_metrics["macro_f1"],
            "macro_recall": noisy_metrics["macro_recall"],
            "macro_precision": noisy_metrics["macro_precision"],
            "n_samples": noisy_metrics["n_samples"],
            "mean_latency_ms": noisy_metrics["mean_latency_ms"],
        },
        "delta": {
            "accuracy_pp": round(delta_accuracy * 100, 2),
            "macro_f1": round(delta_f1, 4),
            "macro_recall": round(delta_recall, 4),
        },
        "per_class_delta": per_class_delta,
        "worst_5_f1_drops": [
            {"class": cls, "delta_f1": d, "delta_recall": dr}
            for cls, d, dr in worst_drops[:5]
        ],
        "n_classes_degraded": sum(
            1 for v in per_class_delta.values() if v["delta_f1"] < 0
        ),
        "n_classes_total": len(classes),
    }

    json_path = output_dir / "robustness_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  JSON saved: {json_path}")

    # ── robustness_report.txt ──
    report_lines = [
        "=" * 70,
        "  ROBUSTNESS EVALUATION REPORT",
        "  YOLOv8n-cls: Clean vs Noisy Test Set",
        "=" * 70,
        "",
        "AGGREGATE METRICS",
        "-" * 50,
        f"{'Metric':<25} {'Clean':>10} {'Noisy':>10} {'Delta':>10}",
        "-" * 50,
        f"{'Accuracy':<25} {clean_metrics['accuracy']*100:>9.2f}% {noisy_metrics['accuracy']*100:>9.2f}% {delta_accuracy*100:>+9.2f}pp",
        f"{'Macro-F1':<25} {clean_metrics['macro_f1']:>10.4f} {noisy_metrics['macro_f1']:>10.4f} {delta_f1:>+10.4f}",
        f"{'Macro-Recall':<25} {clean_metrics['macro_recall']:>10.4f} {noisy_metrics['macro_recall']:>10.4f} {delta_recall:>+10.4f}",
        f"{'Macro-Precision':<25} {clean_metrics['macro_precision']:>10.4f} {noisy_metrics['macro_precision']:>10.4f} {noisy_metrics['macro_precision']-clean_metrics['macro_precision']:>+10.4f}",
        f"{'Latency (ms/img)':<25} {clean_metrics['mean_latency_ms']:>10.1f} {noisy_metrics['mean_latency_ms']:>10.1f} {noisy_metrics['mean_latency_ms']-clean_metrics['mean_latency_ms']:>+10.1f}",
        "",
        f"Classes degraded: {summary['n_classes_degraded']}/{summary['n_classes_total']}",
        "",
        "TOP-5 WORST CLASS DROPS (by F1)",
        "-" * 60,
        f"{'Class':<35} {'ΔF1':>10} {'ΔRecall':>10}",
        "-" * 60,
    ]

    for cls, d_f1, d_r in worst_drops[:5]:
        report_lines.append(f"{cls:<35} {d_f1:>+10.4f} {d_r:>+10.4f}")

    report_lines += [
        "",
        "PER-CLASS DETAIL",
        "-" * 80,
        f"{'Class':<35} {'Clean-R':>8} {'Noisy-R':>8} {'ΔR':>8} {'Clean-F1':>9} {'Noisy-F1':>9} {'ΔF1':>8}",
        "-" * 80,
    ]

    for cls in classes:
        c = clean_metrics["per_class"].get(cls, {})
        n = noisy_metrics["per_class"].get(cls, {})
        dr = per_class_delta[cls]["delta_recall"]
        df = per_class_delta[cls]["delta_f1"]
        report_lines.append(
            f"{cls:<35} {c.get('recall',0):>8.4f} {n.get('recall',0):>8.4f} "
            f"{dr:>+8.4f} {c.get('f1',0):>9.4f} {n.get('f1',0):>9.4f} {df:>+8.4f}"
        )

    report_lines += [
        "",
        "=" * 70,
        "  CONCLUSION",
        "=" * 70,
        f"  Accuracy drop:   {delta_accuracy*100:+.2f} percentage points",
        f"  Macro-F1 drop:   {delta_f1:+.4f}",
        f"  Classes hurt:    {summary['n_classes_degraded']}/{summary['n_classes_total']}",
        "",
    ]

    report_text = "\n".join(report_lines)
    report_path = output_dir / "robustness_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"  Report saved: {report_path}")

    # Print to console
    print(f"\n{report_text}")

    return summary


# ════════════════════════════════════════════════════════════════
# Confusion matrix plot
# ════════════════════════════════════════════════════════════════

def plot_confusion_matrix(predictions: list[dict], classes: list[str],
                          title: str, output_path: Path):
    """Normalized confusion matrix heatmap."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  WARNING: matplotlib not available — skipping confusion matrix")
        return

    n = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((n, n), dtype=int)

    for pred in predictions:
        gi = class_to_idx.get(pred["ground_truth"], -1)
        pi = class_to_idx.get(pred["predicted"], -1)
        if gi >= 0 and pi >= 0:
            cm[gi][pi] += 1

    row_sums = cm.sum(axis=1, keepdims=True).astype(float)
    row_sums[row_sums == 0] = 1
    cm_norm = cm / row_sums

    short = [c.replace("wheat_", "w.").replace("rice_", "r.").replace("healthy_", "h.")
             for c in classes]

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Oranges", vmin=0, vmax=1)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(xticks=np.arange(n), yticks=np.arange(n),
           xticklabels=short, yticklabels=short)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)

    for i in range(n):
        for j in range(n):
            val = cm_norm[i, j]
            if val > 0.005:
                color = "white" if val > 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6, color=color)

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Confusion matrix saved: {output_path}")


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Robustness evaluation: clean vs noisy")
    parser.add_argument("--model-path", type=str,
                        default="models/india_agri_cls_21class_backup.pt",
                        help="Path to YOLOv8 classification model")
    parser.add_argument("--clean-dir", type=str, default="data/training/test",
                        help="Clean test dataset directory")
    parser.add_argument("--noisy-dir", type=str, default="evaluate/noisy_dataset",
                        help="Noisy test dataset directory")
    parser.add_argument("--output-dir", type=str, default="evaluate/results",
                        help="Output directory for results")
    parser.add_argument("--generate-noisy", action="store_true",
                        help="Generate noisy dataset first (runs noise_pipeline)")
    parser.add_argument("--severity", choices=["low", "medium", "high"], default="medium",
                        help="Noise severity (if --generate-noisy)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    clean_dir = Path(args.clean_dir)
    noisy_dir = Path(args.noisy_dir)
    output_dir = Path(args.output_dir)
    model_path = Path(args.model_path)

    # Optionally generate noisy dataset first
    if args.generate_noisy or not noisy_dir.exists():
        print("Generating noisy dataset...")
        from noise_pipeline import generate_noisy_dataset
        generate_noisy_dataset(clean_dir, noisy_dir, args.severity, args.seed)

    # Validate paths
    for p, name in [(clean_dir, "clean-dir"), (noisy_dir, "noisy-dir"), (model_path, "model")]:
        if not p.exists():
            print(f"ERROR: {name} not found: {p}")
            sys.exit(1)

    # Load model
    print(f"\nLoading model: {model_path}")
    from ultralytics import YOLO
    model = YOLO(str(model_path))
    print(f"  Classes: {len(model.names)} — {list(model.names.values())[:5]}...")

    # Evaluate both datasets
    clean_metrics, clean_preds = evaluate_dataset(model, clean_dir, "Clean")
    noisy_metrics, noisy_preds = evaluate_dataset(model, noisy_dir, "Noisy")

    # Generate comparison outputs
    classes = sorted(clean_metrics["per_class"].keys())
    summary = generate_comparison(clean_metrics, noisy_metrics, output_dir)

    # Confusion matrices
    plot_confusion_matrix(
        noisy_preds, classes,
        "Confusion Matrix — Noisy Test Set",
        output_dir / "confusion_matrix_noisy.png",
    )

    # Save per-image predictions
    preds_path = output_dir / "predictions_noisy.csv"
    with open(preds_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "ground_truth", "predicted", "confidence", "correct"])
        for pred in noisy_preds:
            writer.writerow([
                pred["image_path"],
                pred["ground_truth"],
                pred["predicted"],
                pred["confidence"],
                1 if pred["ground_truth"] == pred["predicted"] else 0,
            ])
    print(f"  Predictions saved: {preds_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
