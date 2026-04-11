#!/usr/bin/env python3
"""
retrain_with_feedback.py — Retrain YOLOv8 classifier using feedback-corrected data (E5).

Workflow:
  1. Export corrected images from feedback DB → temporary training folder
  2. Build augmented dataset (original training data + corrected samples)
  3. Fine-tune existing model with corrected data
  4. Evaluate on a held-out set
  5. If improved, promote as new model

Usage:
    python scripts/retrain_with_feedback.py                          # dry run
    python scripts/retrain_with_feedback.py --apply                  # actually retrain
    python scripts/retrain_with_feedback.py --apply --epochs 20      # custom epochs
    python scripts/retrain_with_feedback.py --export-only            # only export images
"""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

FEEDBACK_EXPORT_DIR = PROJECT_ROOT / "data" / "feedback_export"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "retrain"
ORIGINAL_MODEL = MODELS_DIR / "india_agri_cls.pt"


def export_feedback_images() -> dict[str, int]:
    """Export corrected images from feedback DB into class-organized folders.

    Creates: data/feedback_export/{class_name}/ with JPEG images.
    Returns dict of {class_name: count}.
    """
    from agridrone.feedback.feedback_store import get_all_feedback, get_feedback_images_for_disease, init_db

    init_db()
    records = get_all_feedback(limit=10000)

    # Collect unique corrected diseases that have stored images
    disease_counts: dict[str, int] = {}
    diseases_with_images: set[str] = set()

    for r in records:
        correct = r["correct_disease"]
        disease_counts[correct] = disease_counts.get(correct, 0) + 1
        diseases_with_images.add(correct)

    # Clean export dir
    if FEEDBACK_EXPORT_DIR.exists():
        shutil.rmtree(FEEDBACK_EXPORT_DIR)
    FEEDBACK_EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    exported: dict[str, int] = {}
    total = 0

    for disease_key in diseases_with_images:
        images = get_feedback_images_for_disease(disease_key)
        if not images:
            continue

        class_dir = FEEDBACK_EXPORT_DIR / disease_key
        class_dir.mkdir(parents=True, exist_ok=True)

        for fb_id, img_bytes in images:
            img_path = class_dir / f"feedback_{fb_id}.jpg"
            img_path.write_bytes(img_bytes)
            total += 1

        exported[disease_key] = len(images)

    print(f"Exported {total} images across {len(exported)} classes to {FEEDBACK_EXPORT_DIR}")
    return exported


def build_combined_dataset(feedback_exports: dict[str, int]) -> Path:
    """Build a combined dataset with original + feedback images.

    YOLOv8 classify expects: dataset_dir/{train,val}/{class_name}/images.jpg

    Returns path to combined dataset directory.
    """
    combined_dir = OUTPUT_DIR / "combined_dataset"
    if combined_dir.exists():
        shutil.rmtree(combined_dir)
    combined_dir.mkdir(parents=True, exist_ok=True)

    train_dir = combined_dir / "train"
    val_dir = combined_dir / "val"
    train_dir.mkdir()
    val_dir.mkdir()

    # 1. Link/copy original training data
    original_data_paths = [
        PROJECT_ROOT / "data" / "raw" / "wheat" / "data" / "train",
        PROJECT_ROOT / "data" / "raw" / "rice" / "Rice_Leaf_AUG",
        PROJECT_ROOT / "data" / "training",
    ]

    for orig_path in original_data_paths:
        if not orig_path.is_dir():
            continue
        for class_dir in orig_path.iterdir():
            if not class_dir.is_dir():
                continue
            dest_train = train_dir / class_dir.name
            dest_train.mkdir(exist_ok=True)
            for img in class_dir.iterdir():
                if img.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.webp'):
                    dest = dest_train / img.name
                    if not dest.exists():
                        # Use symlink if possible, else copy
                        try:
                            dest.symlink_to(img)
                        except OSError:
                            shutil.copy2(img, dest)

    # 2. Add feedback images to training set (with copies in val at 20%)
    import random
    random.seed(42)

    for disease_key, count in feedback_exports.items():
        src_dir = FEEDBACK_EXPORT_DIR / disease_key
        if not src_dir.is_dir():
            continue

        dest_train = train_dir / disease_key
        dest_val = val_dir / disease_key
        dest_train.mkdir(exist_ok=True)
        dest_val.mkdir(exist_ok=True)

        images = list(src_dir.glob("*.jpg"))
        random.shuffle(images)
        split_idx = max(1, int(len(images) * 0.8))

        for img in images[:split_idx]:
            shutil.copy2(img, dest_train / img.name)
        for img in images[split_idx:]:
            shutil.copy2(img, dest_val / img.name)

    # Count totals
    train_total = sum(1 for _ in train_dir.rglob("*") if _.is_file())
    val_total = sum(1 for _ in val_dir.rglob("*") if _.is_file())
    print(f"Combined dataset: {train_total} train, {val_total} val images at {combined_dir}")

    return combined_dir


def run_finetune(dataset_dir: Path, epochs: int = 10, imgsz: int = 224) -> Path | None:
    """Fine-tune existing classifier on combined dataset.

    Returns path to new best.pt or None on failure.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics")
        return None

    if not ORIGINAL_MODEL.is_file():
        print(f"ERROR: Base model not found: {ORIGINAL_MODEL}")
        return None

    run_name = f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_path = OUTPUT_DIR / run_name

    print(f"\nFine-tuning {ORIGINAL_MODEL.name} for {epochs} epochs...")
    print(f"Dataset: {dataset_dir}")
    print(f"Output: {output_path}")

    model = YOLO(str(ORIGINAL_MODEL))
    results = model.train(
        data=str(dataset_dir),
        epochs=epochs,
        imgsz=imgsz,
        batch=-1,         # auto batch size
        patience=5,
        project=str(OUTPUT_DIR),
        name=run_name,
        exist_ok=True,
        pretrained=True,
        lr0=0.001,        # lower LR for fine-tuning
        lrf=0.01,
        warmup_epochs=1,
        verbose=True,
    )

    best_pt = output_path / "weights" / "best.pt"
    if best_pt.is_file():
        print(f"\nTraining complete. Best model: {best_pt}")
        return best_pt
    else:
        print("\nWARNING: best.pt not found after training")
        return None


def evaluate_model(model_path: Path, dataset_dir: Path) -> dict | None:
    """Evaluate model accuracy on validation set."""
    try:
        from ultralytics import YOLO
    except ImportError:
        return None

    if not model_path.is_file():
        return None

    model = YOLO(str(model_path))
    val_dir = dataset_dir / "val"
    if not val_dir.is_dir():
        return None

    results = model.val(data=str(dataset_dir))

    metrics = {
        "top1_accuracy": round(results.results_dict.get("metrics/accuracy_top1", 0), 4),
        "top5_accuracy": round(results.results_dict.get("metrics/accuracy_top5", 0), 4),
    }
    return metrics


def promote_model(new_model: Path) -> None:
    """Replace the production model with the newly trained one."""
    backup = MODELS_DIR / f"india_agri_cls_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    if ORIGINAL_MODEL.is_file():
        shutil.copy2(ORIGINAL_MODEL, backup)
        print(f"Original model backed up to {backup.name}")

    shutil.copy2(new_model, ORIGINAL_MODEL)
    print(f"New model promoted to {ORIGINAL_MODEL}")


def main():
    parser = argparse.ArgumentParser(description="Retrain classifier with feedback data")
    parser.add_argument("--apply", action="store_true", help="Actually retrain (default is dry run)")
    parser.add_argument("--export-only", action="store_true", help="Only export feedback images, don't train")
    parser.add_argument("--epochs", type=int, default=10, help="Fine-tune epochs (default: 10)")
    parser.add_argument("--imgsz", type=int, default=224, help="Image size (default: 224)")
    parser.add_argument("--promote", action="store_true", help="Auto-promote if accuracy improves")
    args = parser.parse_args()

    print("=" * 60)
    print("  AgriDrone — Feedback-based Model Retraining Pipeline")
    print("=" * 60)

    # Step 1: Export feedback images
    print("\n[1/5] Exporting feedback images...")
    exported = export_feedback_images()

    if not exported:
        print("No feedback images found in database. Submit feedback with images to enable retraining.")
        return

    if args.export_only:
        print("\n[Done] Export-only mode. Images are at:", FEEDBACK_EXPORT_DIR)
        return

    if not args.apply:
        print(f"\n[DRY RUN] Would retrain with {sum(exported.values())} feedback images")
        print("  Classes:", ", ".join(f"{k} ({v})" for k, v in exported.items()))
        print("  Run with --apply to actually retrain")
        return

    # Step 2: Build combined dataset
    print("\n[2/5] Building combined dataset...")
    dataset = build_combined_dataset(exported)

    # Step 3: Fine-tune
    print("\n[3/5] Fine-tuning model...")
    new_model = run_finetune(dataset, epochs=args.epochs, imgsz=args.imgsz)

    if not new_model:
        print("Training failed or produced no model.")
        return

    # Step 4: Evaluate
    print("\n[4/5] Evaluating new model...")
    new_metrics = evaluate_model(new_model, dataset)
    old_metrics = evaluate_model(ORIGINAL_MODEL, dataset)

    report = {
        "timestamp": datetime.now().isoformat(),
        "feedback_images": exported,
        "epochs": args.epochs,
        "new_model": str(new_model),
        "new_metrics": new_metrics,
        "old_metrics": old_metrics,
    }

    report_path = new_model.parent.parent / "retrain_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved: {report_path}")

    if new_metrics and old_metrics:
        old_acc = old_metrics.get("top1_accuracy", 0)
        new_acc = new_metrics.get("top1_accuracy", 0)
        print(f"\nAccuracy: {old_acc:.1%} → {new_acc:.1%} ({'+' if new_acc > old_acc else ''}{(new_acc - old_acc):.1%})")

        if new_acc > old_acc and args.promote:
            print("\n[5/5] Promoting new model...")
            promote_model(new_model)
        elif new_acc > old_acc:
            print(f"\nNew model is better! Run with --promote to replace production model.")
        else:
            print(f"\nNew model is not better. Keeping original.")
    else:
        print("\nCould not compare metrics. Manual evaluation needed.")
        print(f"New model available at: {new_model}")


if __name__ == "__main__":
    main()
