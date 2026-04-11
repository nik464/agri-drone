#!/usr/bin/env python3
"""
Experiment 3 — Cross-Dataset Generalization Evaluation

Evaluates the YOLOv8n-cls model on external datasets to measure
generalization beyond the training distribution.

Supported datasets (place in data/external/):
  - PlantVillage:  data/external/plantvillage/  (folder-per-class)
  - PlantDoc:      data/external/plantdoc/      (folder-per-class)
  - Custom:        any folder with class subfolders

Class mapping: maps external dataset class names → AgriDrone 21-class labels
via a JSON mapping file. Unmapped classes are reported but excluded from
accuracy computation.

Usage:
    python evaluate/cross_dataset_eval.py --dataset-dir data/external/plantvillage
    python evaluate/cross_dataset_eval.py --dataset-dir data/external/plantdoc --mapping configs/plantdoc_map.json
    python evaluate/cross_dataset_eval.py --dataset-dir data/external/plantvillage --generate-mapping
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
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# ════════════════════════════════════════════════════════════════
# Default class mappings (external name → AgriDrone key)
# ════════════════════════════════════════════════════════════════

PLANTVILLAGE_MAP = {
    # Wheat diseases (PlantVillage subset)
    "Wheat___Brown_Rust": "wheat_brown_rust",
    "Wheat___Yellow_Rust": "wheat_yellow_rust",
    "Wheat___Healthy": "healthy_wheat",
    "Wheat___Septoria": "wheat_septoria",
    "Wheat___Powdery_Mildew": "wheat_powdery_mildew",
    # Rice diseases
    "Rice___Brown_Spot": "rice_brown_spot",
    "Rice___Blast": "rice_blast",
    "Rice___Bacterial_Blight": "rice_bacterial_blight",
    "Rice___Healthy": "healthy_rice",
    "Rice___Sheath_Blight": "rice_sheath_blight",
    "Rice___Leaf_Scald": "rice_leaf_scald",
}

PLANTDOC_MAP = {
    "wheat_brown_rust": "wheat_brown_rust",
    "wheat_yellow_rust": "wheat_yellow_rust",
    "wheat_healthy": "healthy_wheat",
    "wheat_septoria": "wheat_septoria",
    "wheat_powdery_mildew": "wheat_powdery_mildew",
    "rice_brown_spot": "rice_brown_spot",
    "rice_blast": "rice_blast",
    "rice_bacterial_blight": "rice_bacterial_blight",
    "rice_healthy": "healthy_rice",
}


def load_class_mapping(mapping_path: Path | None, dataset_dir: Path) -> dict:
    """Load or auto-detect class mapping."""
    if mapping_path and mapping_path.exists():
        return json.loads(mapping_path.read_text(encoding="utf-8"))

    # Auto-detect from known dataset structures
    dir_name = dataset_dir.name.lower()
    if "plantvillage" in dir_name:
        print("  Using built-in PlantVillage mapping")
        return PLANTVILLAGE_MAP
    elif "plantdoc" in dir_name:
        print("  Using built-in PlantDoc mapping")
        return PLANTDOC_MAP

    # Identity mapping: assume folder names match AgriDrone keys
    print("  No mapping found — using identity mapping (folder name = class key)")
    classes = [d.name for d in dataset_dir.iterdir() if d.is_dir()]
    return {c: c for c in classes}


def load_external_dataset(dataset_dir: Path, class_mapping: dict) -> tuple[list[dict], list[str]]:
    """Load external dataset with class mapping."""
    images = []
    unmapped = set()
    mapped_classes = set()

    for cls_dir in sorted(dataset_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        ext_class = cls_dir.name
        agri_class = class_mapping.get(ext_class)

        if agri_class is None:
            unmapped.add(ext_class)
            continue

        mapped_classes.add(agri_class)
        for img_path in sorted(cls_dir.glob("*")):
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}:
                images.append({
                    "path": str(img_path),
                    "external_class": ext_class,
                    "ground_truth": agri_class,
                    "crop_type": "rice" if "rice" in agri_class else "wheat",
                })

    if unmapped:
        print(f"  Unmapped classes ({len(unmapped)}): {sorted(unmapped)}")
    print(f"  Mapped classes: {sorted(mapped_classes)}")
    print(f"  Total images: {len(images)}")

    return images, sorted(mapped_classes)


def generate_mapping_template(dataset_dir: Path, model_classes: list[str], output_path: Path):
    """Generate a JSON mapping template for manual editing."""
    ext_classes = sorted([d.name for d in dataset_dir.iterdir() if d.is_dir()])
    mapping = {}
    for ec in ext_classes:
        # Try fuzzy match
        ec_lower = ec.lower().replace("___", "_").replace(" ", "_")
        matched = None
        for mc in model_classes:
            if mc in ec_lower or ec_lower in mc:
                matched = mc
                break
        mapping[ec] = matched  # null if no match

    output_path.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    print(f"  Mapping template saved → {output_path}")
    print(f"  Edit null values to map external classes → AgriDrone classes")
    return mapping


# ════════════════════════════════════════════════════════════════
# Evaluation
# ════════════════════════════════════════════════════════════════

def evaluate_model(model, images: list[dict], classes: list[str]) -> dict:
    """Run model on external images and compute metrics."""
    predictions = []
    tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)
    total_correct = 0
    latencies = []

    for i, img_info in enumerate(images):
        image_bgr = cv2.imread(img_info["path"])
        if image_bgr is None:
            warnings.warn(f"Cannot read {img_info['path']}")
            continue

        t0 = time.perf_counter()
        results = model(image_bgr, verbose=False)
        latency = (time.perf_counter() - t0) * 1000
        latencies.append(latency)

        if not results or results[0].probs is None:
            predicted = "unknown"
            confidence = 0.0
        else:
            probs = results[0].probs
            names = model.names
            predicted = names[probs.top1]
            confidence = round(probs.top1conf.item(), 4)

        gt = img_info["ground_truth"]
        correct = predicted == gt
        if correct:
            total_correct += 1
            tp[gt] += 1
        else:
            fn[gt] += 1
            fp[predicted] += 1

        predictions.append({
            "path": img_info["path"],
            "external_class": img_info["external_class"],
            "ground_truth": gt,
            "predicted": predicted,
            "confidence": confidence,
            "correct": correct,
            "latency_ms": round(latency, 1),
        })

        if (i + 1) % 100 == 0:
            acc = total_correct / len(predictions)
            print(f"  [{i+1}/{len(images)}] running acc: {acc:.1%}")

    # Compute per-class metrics
    n = len(predictions)
    accuracy = total_correct / n if n > 0 else 0

    per_class = {}
    f1_scores = []
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
        if support > 0:
            f1_scores.append(f1)

    macro_f1 = np.mean(f1_scores) if f1_scores else 0.0

    # Domain shift analysis: confidence distribution
    correct_confs = [p["confidence"] for p in predictions if p["correct"]]
    wrong_confs = [p["confidence"] for p in predictions if not p["correct"]]

    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(float(macro_f1), 4),
        "n_samples": n,
        "per_class": per_class,
        "mean_latency_ms": round(float(np.mean(latencies)), 1) if latencies else 0,
        "mean_confidence_correct": round(float(np.mean(correct_confs)), 4) if correct_confs else 0,
        "mean_confidence_wrong": round(float(np.mean(wrong_confs)), 4) if wrong_confs else 0,
        "confidence_gap": round(
            float(np.mean(correct_confs)) - float(np.mean(wrong_confs)), 4
        ) if correct_confs and wrong_confs else 0,
        "predictions": predictions,
    }


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Experiment 3: Cross-dataset evaluation")
    parser.add_argument("--dataset-dir", required=True, help="Path to external dataset (folder-per-class)")
    parser.add_argument("--model-path", default=str(PROJECT_ROOT / "models" / "india_agri_cls.pt"))
    parser.add_argument("--mapping", default=None, help="JSON file mapping external → AgriDrone class names")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "evaluate" / "results"))
    parser.add_argument("--generate-mapping", action="store_true", help="Generate mapping template and exit")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_dir.exists():
        print(f"ERROR: Dataset directory not found: {dataset_dir}")
        sys.exit(1)

    # ── Load model ──
    print("Loading YOLO classifier...")
    from ultralytics import YOLO
    model = YOLO(args.model_path, task="classify")
    model_classes = sorted(model.names.values())
    print(f"  Model: {args.model_path} ({len(model_classes)} classes)")

    # ── Generate mapping template ──
    if args.generate_mapping:
        template_path = output_dir / f"mapping_template_{dataset_dir.name}.json"
        generate_mapping_template(dataset_dir, model_classes, template_path)
        return

    # ── Load class mapping ──
    mapping_path = Path(args.mapping) if args.mapping else None
    class_mapping = load_class_mapping(mapping_path, dataset_dir)

    # ── Load dataset ──
    images, mapped_classes = load_external_dataset(dataset_dir, class_mapping)
    if not images:
        print("ERROR: No mapped images found. Check class mapping.")
        sys.exit(1)

    # ── Evaluate ──
    print(f"\n{'='*70}")
    print(f"  CROSS-DATASET EVALUATION: {dataset_dir.name}")
    print(f"{'='*70}")

    results = evaluate_model(model, images, mapped_classes)

    # Remove raw predictions from JSON output (save separately)
    predictions = results.pop("predictions")

    print(f"\n  Results:")
    print(f"    Accuracy:           {results['accuracy']:.1%}")
    print(f"    Macro-F1:           {results['macro_f1']:.3f}")
    print(f"    Mean confidence (correct): {results['mean_confidence_correct']:.3f}")
    print(f"    Mean confidence (wrong):   {results['mean_confidence_wrong']:.3f}")
    print(f"    Confidence gap:     {results['confidence_gap']:.3f}")
    print(f"    N samples:          {results['n_samples']}")

    # Per-class breakdown
    print(f"\n    Per-class F1:")
    for cls, data in sorted(results["per_class"].items()):
        print(f"      {cls:40s}: F1={data['f1']:.3f}  P={data['precision']:.3f}  "
              f"R={data['recall']:.3f}  n={data['support']}")

    # ── Save outputs ──
    ds_name = dataset_dir.name
    results_path = output_dir / f"cross_dataset_{ds_name}.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n  Metrics → {results_path}")

    # Per-image predictions CSV
    csv_path = output_dir / f"cross_dataset_{ds_name}_predictions.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "external_class", "ground_truth", "predicted",
                     "confidence", "correct", "latency_ms"])
        for p in predictions:
            w.writerow([
                Path(p["path"]).name,
                p["external_class"],
                p["ground_truth"],
                p["predicted"],
                p["confidence"],
                1 if p["correct"] else 0,
                p["latency_ms"],
            ])
    print(f"  Predictions CSV → {csv_path}")


if __name__ == "__main__":
    main()
