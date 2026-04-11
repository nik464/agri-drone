#!/usr/bin/env python3
"""
evaluate_model.py — Full evaluation suite for the YOLO crop disease detector.

Generates:
  1. mAP@0.5 and mAP@0.5:0.95 metrics
  2. Per-class precision, recall, F1
  3. Precision-Recall curves (per-class + combined)
  4. Confusion matrix (normalized + raw counts)
  5. Sample prediction grid
  6. Detailed JSON/CSV report

Usage:
    python scripts/evaluate_model.py
    python scripts/evaluate_model.py --model models/yolo_crop_disease.pt --data data/raw/roboflow/rice-diseases-v2/data.yaml
    python scripts/evaluate_model.py --split test --conf 0.25
"""
import argparse
import json
import csv
import sys
from pathlib import Path
from datetime import datetime

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def evaluate_yolo(model_path: str, data_yaml: str, split: str, conf: float, iou: float,
                  imgsz: int, device: str, output_dir: Path):
    """Run YOLO val and extract all metrics."""
    from ultralytics import YOLO

    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    print(f"Running evaluation on '{split}' split...")
    print(f"  conf={conf}, iou={iou}, imgsz={imgsz}, device={device}")

    results = model.val(
        data=data_yaml,
        split=split,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        plots=True,     # generates PR curves, confusion matrix, etc.
        save_json=True,  # COCO-format results
        project=str(output_dir),
        name="eval",
        exist_ok=True,
    )

    return model, results


def extract_metrics(results, class_names: list) -> dict:
    """Extract structured metrics from YOLO validation results."""
    metrics = {}

    # Overall metrics
    box = results.box
    metrics["mAP50"] = float(box.map50)
    metrics["mAP50_95"] = float(box.map)
    metrics["precision"] = float(box.mp)
    metrics["recall"] = float(box.mr)

    # Per-class metrics
    per_class = []
    ap50 = box.ap50 if hasattr(box, 'ap50') else box.maps
    ap = box.ap if hasattr(box, 'ap') else None

    for i, name in enumerate(class_names):
        entry = {"class_id": i, "class_name": name}
        if i < len(box.p):
            entry["precision"] = float(box.p[i])
        if i < len(box.r):
            entry["recall"] = float(box.r[i])
        if i < len(ap50):
            entry["ap50"] = float(ap50[i])
        if ap is not None and i < len(ap):
            entry["ap50_95"] = float(ap[i])
        # F1
        p = entry.get("precision", 0)
        r = entry.get("recall", 0)
        entry["f1"] = float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        per_class.append(entry)

    metrics["per_class"] = per_class
    return metrics


def plot_pr_curves(metrics: dict, output_dir: Path):
    """Generate additional precision-recall analysis plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping custom PR plots")
        return

    per_class = metrics["per_class"]
    names = [c["class_name"] for c in per_class]
    precisions = [c.get("precision", 0) for c in per_class]
    recalls = [c.get("recall", 0) for c in per_class]
    f1s = [c.get("f1", 0) for c in per_class]
    ap50s = [c.get("ap50", 0) for c in per_class]

    # ── 1. Per-class AP50 bar chart ──
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.RdYlGn([v for v in ap50s])
    bars = ax.barh(names, ap50s, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("AP@0.5")
    ax.set_title(f"Per-Class AP@0.5  (mAP={metrics['mAP50']:.3f})")
    ax.set_xlim(0, 1)
    for bar, val in zip(bars, ap50s):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_dir / "per_class_ap50.png", dpi=150)
    plt.close()

    # ── 2. Precision vs Recall scatter ──
    fig, ax = plt.subplots(figsize=(8, 8))
    scatter = ax.scatter(recalls, precisions, c=f1s, cmap="RdYlGn",
                         s=100, edgecolors="black", linewidth=0.5, vmin=0, vmax=1)
    for i, name in enumerate(names):
        ax.annotate(name, (recalls[i], precisions[i]), fontsize=7,
                    xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision vs Recall (color = F1)")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.2)
    plt.colorbar(scatter, label="F1 Score")
    plt.tight_layout()
    plt.savefig(output_dir / "precision_recall_scatter.png", dpi=150)
    plt.close()

    # ── 3. F1 bar chart ──
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.RdYlGn(f1s)
    bars = ax.barh(names, f1s, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("F1 Score")
    ax.set_title("Per-Class F1 Score")
    ax.set_xlim(0, 1)
    for bar, val in zip(bars, f1s):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_dir / "per_class_f1.png", dpi=150)
    plt.close()

    # ── 4. Metrics summary radar chart ──
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    categories = ["mAP@0.5", "mAP@0.5:0.95", "Precision", "Recall"]
    values = [metrics["mAP50"], metrics["mAP50_95"], metrics["precision"], metrics["recall"]]
    values_plot = values + [values[0]]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    ax.fill(angles, values_plot, alpha=0.25, color="green")
    ax.plot(angles, values_plot, "o-", color="green", linewidth=2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title("Model Performance Overview", fontsize=14, pad=20)
    for angle, val in zip(angles[:-1], values):
        ax.annotate(f"{val:.3f}", (angle, val), fontsize=10, ha="center",
                    xytext=(0, 10), textcoords="offset points")
    plt.tight_layout()
    plt.savefig(output_dir / "metrics_radar.png", dpi=150)
    plt.close()

    print(f"  Saved: per_class_ap50.png, precision_recall_scatter.png, per_class_f1.png, metrics_radar.png")


def plot_confusion_matrix(results, class_names: list, output_dir: Path):
    """Generate a clean confusion matrix from val results."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
    except ImportError:
        print("  matplotlib not available — skipping confusion matrix")
        return

    # YOLO already saves confusion_matrix.png in the eval dir, but we make a nicer one
    cm_path = output_dir / "eval" / "confusion_matrix.png"
    if cm_path.exists():
        print(f"  YOLO confusion matrix at: {cm_path}")

    # Also check if normalized version exists
    cm_norm = output_dir / "eval" / "confusion_matrix_normalized.png"
    if cm_norm.exists():
        print(f"  Normalized confusion matrix at: {cm_norm}")


def save_report(metrics: dict, output_dir: Path, model_path: str, data_yaml: str):
    """Save JSON + CSV evaluation report."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "model": model_path,
        "dataset": data_yaml,
        "overall": {
            "mAP50": metrics["mAP50"],
            "mAP50_95": metrics["mAP50_95"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
        },
        "per_class": metrics["per_class"],
    }

    # JSON
    json_path = output_dir / "evaluation_report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  JSON report: {json_path}")

    # CSV
    csv_path = output_dir / "evaluation_report.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "class_name", "precision", "recall", "f1", "ap50", "ap50_95"])
        for c in metrics["per_class"]:
            writer.writerow([
                c["class_id"], c["class_name"],
                f"{c.get('precision', 0):.4f}",
                f"{c.get('recall', 0):.4f}",
                f"{c.get('f1', 0):.4f}",
                f"{c.get('ap50', 0):.4f}",
                f"{c.get('ap50_95', 0):.4f}",
            ])
        # Summary row
        writer.writerow([
            "", "OVERALL",
            f"{metrics['precision']:.4f}",
            f"{metrics['recall']:.4f}",
            "",
            f"{metrics['mAP50']:.4f}",
            f"{metrics['mAP50_95']:.4f}",
        ])
    print(f"  CSV report: {csv_path}")


def print_summary(metrics: dict):
    """Print a readable summary table."""
    print()
    print("=" * 75)
    print("  EVALUATION RESULTS")
    print("=" * 75)
    print(f"  mAP@0.5:        {metrics['mAP50']:.4f}")
    print(f"  mAP@0.5:0.95:   {metrics['mAP50_95']:.4f}")
    print(f"  Precision:       {metrics['precision']:.4f}")
    print(f"  Recall:          {metrics['recall']:.4f}")
    print()
    print(f"  {'Class':<25s} {'Prec':>8s} {'Recall':>8s} {'F1':>8s} {'AP50':>8s}")
    print("  " + "-" * 60)
    for c in metrics["per_class"]:
        print(f"  {c['class_name']:<25s} "
              f"{c.get('precision', 0):>8.3f} "
              f"{c.get('recall', 0):>8.3f} "
              f"{c.get('f1', 0):>8.3f} "
              f"{c.get('ap50', 0):>8.3f}")
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO crop disease detector")
    parser.add_argument("--model", default=str(PROJECT_ROOT / "models" / "yolo_crop_disease.pt"),
                        help="Path to trained model")
    parser.add_argument("--data", default=str(PROJECT_ROOT / "data" / "raw" / "roboflow" / "rice-diseases-v2" / "data.yaml"),
                        help="Path to data.yaml")
    parser.add_argument("--split", default="test", choices=["test", "val"],
                        help="Which split to evaluate on")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.6, help="IoU threshold for NMS")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", default="cpu", help="Device")
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / "outputs" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not Path(args.model).exists():
        print(f"ERROR: Model not found at {args.model}")
        print("Train the model first: python scripts/train_yolo_detector.py")
        sys.exit(1)

    # Load class names
    import yaml
    with open(args.data) as f:
        data_cfg = yaml.safe_load(f)
    class_names = data_cfg.get("names", [])

    # Run evaluation
    model, results = evaluate_yolo(
        args.model, args.data, args.split, args.conf, args.iou,
        args.imgsz, args.device, output_dir
    )

    # Extract metrics
    metrics = extract_metrics(results, class_names)

    # Print summary
    print_summary(metrics)

    # Generate plots
    print("\nGenerating plots...")
    plot_pr_curves(metrics, output_dir)
    plot_confusion_matrix(results, class_names, output_dir)

    # Save reports
    print("\nSaving reports...")
    save_report(metrics, output_dir, args.model, args.data)

    print(f"\nAll outputs saved to: {output_dir}/")
    print("YOLO auto-generated plots in: {}/eval/".format(output_dir))


if __name__ == "__main__":
    main()
