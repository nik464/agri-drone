"""
Improved YOLOv8 training pipeline for rice disease detection.

Addresses: class imbalance, underfitting, augmentation gaps.
Target: mAP@50 >= 60% (from 18.9% baseline).

Usage:
    python scripts/train_improved.py                    # Phase 1: Balanced baseline
    python scripts/train_improved.py --phase 2          # Phase 2: Heavy augmentation
    python scripts/train_improved.py --phase 3          # Phase 3: Full pipeline
    python scripts/train_improved.py --phase 3 --model yolov8m.pt  # Model upgrade
"""

import argparse
import os
import shutil
import random
import yaml
from pathlib import Path
from collections import Counter

# ──────────────────────────────────────────────────────────
# 1. CLASS-IMBALANCE ANALYSIS + OVERSAMPLING
# ──────────────────────────────────────────────────────────

CLASS_NAMES = [
    "Bacterial blight", "Bacterial leaf", "Brown spot", "Cuterpillar",
    "Drainage impact", "Grashopper damage", "Grassy stunt", "Leaf folder",
    "Sheath blight", "Stem borer", "Tungro",
]

# Observed distribution (train split):
# cls 7  (Leaf folder)         = 1011  ████████████████████
# cls 0  (Bacterial blight)    =  869  █████████████████
# cls 10 (Tungro)              =  612  ████████████
# cls 2  (Brown spot)          =  403  ████████
# cls 1  (Bacterial leaf)      =  196  ████
# cls 3  (Caterpillar)         =  175  ███
# cls 8  (Sheath blight)       =  126  ██
# cls 9  (Stem borer)          =   98  ██
# cls 4  (Drainage impact)     =   37  █
# cls 6  (Grassy stunt)        =   36  █
# cls 5  (Grasshopper damage)  =   21  ▏
#
# Imbalance ratio: 1011 / 21 = 48:1 → CATASTROPHIC
# 6 classes with 0% AP all have < 200 annotations


def count_class_distribution(label_dir: Path) -> Counter:
    """Count annotation instances per class ID."""
    dist = Counter()
    for txt in label_dir.glob("*.txt"):
        for line in txt.read_text().strip().splitlines():
            cls_id = int(line.split()[0])
            dist[cls_id] += 1
    return dist


def build_oversampled_dataset(
    src_img_dir: Path,
    src_lbl_dir: Path,
    dst_img_dir: Path,
    dst_lbl_dir: Path,
    target_per_class: int = 800,
    seed: int = 42,
):
    """
    Create an oversampled copy of the training set.

    Strategy: repeat images containing minority-class annotations until
    each class reaches ~target_per_class instances.
    """
    random.seed(seed)
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    # Map class_id → list of (img, lbl) paths
    class_to_files: dict[int, list[tuple[Path, Path]]] = {i: [] for i in range(11)}
    for lbl_path in sorted(src_lbl_dir.glob("*.txt")):
        img_name = lbl_path.stem + ".jpg"
        img_path = src_img_dir / img_name
        if not img_path.exists():
            img_path = src_img_dir / (lbl_path.stem + ".png")
        if not img_path.exists():
            continue
        classes_in_img = set()
        for line in lbl_path.read_text().strip().splitlines():
            classes_in_img.add(int(line.split()[0]))
        for c in classes_in_img:
            class_to_files[c].append((img_path, lbl_path))

    dist = count_class_distribution(src_lbl_dir)
    print("\n📊 Original class distribution:")
    for i in range(11):
        bar = "█" * (dist[i] // 50)
        print(f"  [{i:2d}] {CLASS_NAMES[i]:25s} {dist[i]:5d}  {bar}")

    # Copy all originals first
    copied = set()
    for img_path in src_img_dir.iterdir():
        if img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
            shutil.copy2(img_path, dst_img_dir / img_path.name)
            lbl_name = img_path.stem + ".txt"
            lbl_src = src_lbl_dir / lbl_name
            if lbl_src.exists():
                shutil.copy2(lbl_src, dst_lbl_dir / lbl_name)
            copied.add(img_path.stem)

    # Oversample minority classes
    dup_count = 0
    for cls_id in range(11):
        current = dist[cls_id]
        if current >= target_per_class or not class_to_files[cls_id]:
            continue
        needed = target_per_class - current
        pool = class_to_files[cls_id]
        for i in range(needed):
            img_src, lbl_src = random.choice(pool)
            suffix = img_src.suffix
            new_stem = f"{img_src.stem}_dup{cls_id}_{i}"
            shutil.copy2(img_src, dst_img_dir / f"{new_stem}{suffix}")
            shutil.copy2(lbl_src, dst_lbl_dir / f"{new_stem}.txt")
            dup_count += 1

    new_dist = count_class_distribution(dst_lbl_dir)
    print(f"\n✅ Oversampled: +{dup_count} duplicated images")
    print("📊 New class distribution:")
    for i in range(11):
        bar = "█" * (new_dist[i] // 50)
        print(f"  [{i:2d}] {CLASS_NAMES[i]:25s} {new_dist[i]:5d}  {bar}")

    return dst_img_dir, dst_lbl_dir


def create_balanced_data_yaml(
    original_yaml: Path,
    train_img_dir: Path,
    output_yaml: Path,
):
    """Write a new data.yaml pointing to the oversampled train set."""
    cfg = yaml.safe_load(original_yaml.read_text())
    cfg["train"] = str(train_img_dir)
    # Keep val/test pointing to originals
    output_yaml.write_text(yaml.dump(cfg, default_flow_style=False))
    print(f"📝 Balanced data.yaml written to {output_yaml}")
    return output_yaml


# ──────────────────────────────────────────────────────────
# 2. TRAINING CONFIGURATIONS (3 phases)
# ──────────────────────────────────────────────────────────

def get_phase1_config():
    """
    Phase 1: Fix fundamentals — balance + longer training + proper LR.
    Expected mAP improvement: 18.9% → 35-40%
    """
    return dict(
        # Model
        model="yolov8s.pt",
        task="detect",
        # Training
        epochs=200,
        patience=30,
        batch=16,
        imgsz=640,
        # Optimizer — cosine annealing, warmer final LR
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.1,          # final_lr = 0.001 * 0.1 = 0.0001 (not 0.00001)
        cos_lr=True,
        weight_decay=0.0005,
        warmup_epochs=5,
        warmup_momentum=0.5,
        # Augmentation — moderate baseline
        mosaic=1.0,
        close_mosaic=20,   # keep mosaic longer
        mixup=0.15,
        copy_paste=0.1,
        degrees=15.0,
        translate=0.15,
        scale=0.5,
        flipud=0.5,
        fliplr=0.5,
        hsv_h=0.02,
        hsv_s=0.5,
        hsv_v=0.4,
        erasing=0.3,
        # Misc
        cache="ram",
        workers=4,
        amp=True,
        deterministic=False,
        verbose=True,
        plots=True,
        val=True,
    )


def get_phase2_config():
    """
    Phase 2: Heavy augmentation + multi-scale + copy-paste.
    Expected mAP improvement: 35-40% → 50-55%
    """
    cfg = get_phase1_config()
    cfg.update(
        epochs=300,
        patience=40,
        imgsz=800,             # higher resolution
        multi_scale=0.5,       # resize between 0.5x-1.5x
        # Aggressive augmentation
        mosaic=1.0,
        close_mosaic=30,
        mixup=0.3,
        copy_paste=0.3,        # paste minority objects into other images
        degrees=25.0,
        translate=0.2,
        scale=0.7,
        shear=5.0,
        perspective=0.001,
        erasing=0.4,
        hsv_h=0.03,
        hsv_s=0.6,
        hsv_v=0.5,
    )
    return cfg


def get_phase3_config():
    """
    Phase 3: Model upgrade + fine-tuning from Phase 2 best.
    Expected mAP improvement: 50-55% → 60-65%
    """
    cfg = get_phase2_config()
    cfg.update(
        # Upgrade to medium backbone for +3-5% mAP
        model="yolov8m.pt",
        epochs=150,
        patience=30,
        batch=8,               # smaller batch for larger model
        imgsz=800,
        lr0=0.0005,            # lower LR for fine-tuning
        lrf=0.05,
        # Lighter augmentation for fine-tuning
        mosaic=0.8,
        close_mosaic=15,
        mixup=0.1,
        copy_paste=0.2,
        degrees=10.0,
        erasing=0.2,
    )
    return cfg


# ──────────────────────────────────────────────────────────
# 3. MAIN TRAINING LOOP
# ──────────────────────────────────────────────────────────

def train(args):
    from ultralytics import YOLO

    project_root = Path(__file__).resolve().parent.parent
    data_root = project_root / "data" / "raw" / "roboflow" / "rice-diseases-v2"
    original_yaml = data_root / "data.yaml"
    output_dir = project_root / "outputs" / "training" / f"phase{args.phase}"

    # ── Step 1: Build balanced dataset ──
    if args.phase >= 1 and not args.skip_balance:
        balanced_dir = data_root / "train_balanced"
        balanced_img = balanced_dir / "images"
        balanced_lbl = balanced_dir / "labels"

        if balanced_img.exists() and args.reuse_balance:
            print("♻️  Reusing existing balanced dataset")
        else:
            if balanced_dir.exists():
                shutil.rmtree(balanced_dir)
            build_oversampled_dataset(
                src_img_dir=data_root / "train" / "images",
                src_lbl_dir=data_root / "train" / "labels",
                dst_img_dir=balanced_img,
                dst_lbl_dir=balanced_lbl,
                target_per_class=800,
            )

        balanced_yaml = data_root / "data_balanced.yaml"
        create_balanced_data_yaml(original_yaml, balanced_img, balanced_yaml)
        data_yaml = str(balanced_yaml)
    else:
        data_yaml = str(original_yaml)

    # ── Step 2: Select training config ──
    phase_configs = {1: get_phase1_config, 2: get_phase2_config, 3: get_phase3_config}
    cfg = phase_configs[args.phase]()

    # Override model if specified
    if args.model:
        cfg["model"] = args.model
    if args.resume:
        cfg["model"] = args.resume
        cfg["resume"] = True

    model_name = cfg.pop("model")
    task = cfg.pop("task", "detect")

    # ── Step 3: Train ──
    print(f"\n🚀 Phase {args.phase} Training")
    print(f"   Model:  {model_name}")
    print(f"   Data:   {data_yaml}")
    print(f"   Epochs: {cfg['epochs']}")
    print(f"   ImgSz:  {cfg['imgsz']}")
    print(f"   LR:     {cfg['lr0']} → {cfg['lr0'] * cfg['lrf']:.6f}")
    print()

    model = YOLO(model_name, task=task)
    results = model.train(
        data=data_yaml,
        project=str(output_dir),
        name=f"phase{args.phase}_run",
        exist_ok=True,
        **cfg,
    )

    # ── Step 4: Evaluate ──
    print("\n📊 Final Evaluation on test set:")
    metrics = model.val(
        data=data_yaml,
        split="test",
        imgsz=cfg["imgsz"],
        batch=cfg.get("batch", 16),
        plots=True,
    )
    print(f"   mAP@50:    {metrics.box.map50:.4f}")
    print(f"   mAP@50-95: {metrics.box.map:.4f}")
    print(f"   Precision:  {metrics.box.mp:.4f}")
    print(f"   Recall:     {metrics.box.mr:.4f}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Improved rice disease detection training")
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--model", type=str, default=None, help="Override model (e.g. yolov8m.pt)")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    parser.add_argument("--skip-balance", action="store_true", help="Skip dataset balancing")
    parser.add_argument("--reuse-balance", action="store_true", help="Reuse existing balanced dataset")
    args = parser.parse_args()
    train(args)
