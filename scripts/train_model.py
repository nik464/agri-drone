#!/usr/bin/env python3
"""
train_model.py — Train a YOLOv8n-cls crop disease classifier.

Scans rice (6 explicit classes) and wheat (auto-detected) datasets,
copies max 300 images per class into data/training/, trains YOLOv8n-cls,
saves the best checkpoint, tests on Leaf Blast images, and patches
phone_connect.py to use the new model.

Usage:
    python scripts/train_model.py
"""
import os
import random
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Source paths ──
RICE_DIR = PROJECT_ROOT / "data" / "raw" / "rice" / "Rice_Leaf_AUG"
WHEAT_TRAIN_DIR = PROJECT_ROOT / "data" / "raw" / "wheat" / "data" / "train"
TRAINING_DIR = PROJECT_ROOT / "data" / "training"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}

# ── Rice class map (explicit — 6 classes) ──
RICE_CLASS_MAP = {
    "Bacterial Leaf Blight": "rice_bacterial_blight",
    "Brown Spot":            "rice_brown_spot",
    "Healthy Rice Leaf":     "healthy_rice",
    "Leaf Blast":            "rice_blast",
    "Leaf scald":            "rice_leaf_scald",
    "Leaf Scald":            "rice_leaf_scald",
    "Sheath Blight":         "rice_sheath_blight",
}

# ── Wheat auto-mapping rules (substring -> label, checked in order) ──
WHEAT_RULES: list[tuple[list[str], str]] = [
    (["fusarium", "scab"],                       "wheat_fusarium_head_blight"),
    (["yellow rust", "yellow_rust", "stripe"],    "wheat_yellow_rust"),
    (["black rust", "black_rust"],                "wheat_black_rust"),
    (["brown rust", "brown_rust"],                "wheat_brown_rust"),
    (["leaf blight", "leaf_blight"],              "wheat_leaf_blight"),
    (["mildew", "powdery"],                       "wheat_powdery_mildew"),
    (["septoria"],                                "wheat_septoria"),
    (["tan spot", "tan_spot"],                    "wheat_tan_spot"),
    (["smut", "loose"],                           "wheat_smut"),
    (["common root", "root rot", "root_rot"],     "wheat_root_rot"),
    (["aphid"],                                   "wheat_aphid"),
    (["mite"],                                    "wheat_mite"),
    (["stem fly", "stem_fly", "stemfly"],         "wheat_stem_fly"),
    (["blast"],                                   "wheat_blast"),
    (["healthy"],                                 "healthy_wheat"),
]

MAX_PER_CLASS = 300


def get_images(folder: Path) -> list[Path]:
    """Return all image files in a folder (non-recursive)."""
    return [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS]


def wheat_folder_to_label(folder_name: str) -> str:
    """Map a wheat class folder name to a unified label."""
    low = folder_name.lower()
    for keywords, label in WHEAT_RULES:
        if any(kw in low for kw in keywords):
            return label
    # Fallback: sanitize the name and prefix with wheat_
    return "wheat_" + low.replace(" ", "_").replace("-", "_")


# =====================================================================
#  PART 1 — PREPARE DATA
# =====================================================================
def prepare_data() -> dict[str, list[Path]]:
    """Scan rice + wheat datasets, return {label: [image_paths]}."""

    classes: dict[str, list[Path]] = {}

    # ── Rice (explicit map) ──
    print("\n-- Scanning rice dataset --")
    if RICE_DIR.is_dir():
        for folder in sorted(RICE_DIR.iterdir()):
            if not folder.is_dir():
                continue
            label = RICE_CLASS_MAP.get(folder.name)
            if label is None:
                # Try case-insensitive match
                for key, val in RICE_CLASS_MAP.items():
                    if key.lower() == folder.name.lower():
                        label = val
                        break
            if label is None:
                print(f"  WARNING: Skipping unmapped rice folder: {folder.name}")
                continue
            imgs = get_images(folder)
            classes.setdefault(label, []).extend(imgs)
            print(f"  {folder.name:30s} -> {label:30s}  ({len(imgs)} images)")
    else:
        print(f"  WARNING: Rice directory not found: {RICE_DIR}")

    # ── Wheat (auto-detect from train folder) ──
    print("\n-- Scanning wheat dataset --")
    if WHEAT_TRAIN_DIR.is_dir():
        for folder in sorted(WHEAT_TRAIN_DIR.iterdir()):
            if not folder.is_dir():
                continue
            label = wheat_folder_to_label(folder.name)
            imgs = get_images(folder)
            classes.setdefault(label, []).extend(imgs)
            print(f"  {folder.name:30s} -> {label:30s}  ({len(imgs)} images)")
    else:
        print(f"  WARNING: Wheat directory not found: {WHEAT_TRAIN_DIR}")

    total_images = sum(len(v) for v in classes.values())
    print(f"\nFound {len(classes)} classes, {total_images} total images")

    return classes


def copy_to_training(classes: dict[str, list[Path]]) -> list[str]:
    """Copy max MAX_PER_CLASS images per class into data/training/."""

    # Clean previous build
    if TRAINING_DIR.exists():
        print(f"\nRemoving old {TRAINING_DIR} ...")
        shutil.rmtree(TRAINING_DIR)

    print(f"\nCopying images (max {MAX_PER_CLASS} per class):\n")

    sorted_labels = sorted(classes.keys())
    for label in sorted_labels:
        imgs = classes[label]
        random.shuffle(imgs)
        selected = imgs[:MAX_PER_CLASS]

        # YOLOv8-cls expects data/train/classname/ and data/val/classname/
        # Split 80/20 for train/val
        n = len(selected)
        n_train = max(1, int(n * 0.80))
        train_imgs = selected[:n_train]
        val_imgs = selected[n_train:]

        train_dir = TRAINING_DIR / "train" / label
        val_dir = TRAINING_DIR / "val" / label
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)

        for img in train_imgs:
            shutil.copy2(img, train_dir / img.name)
        for img in val_imgs:
            shutil.copy2(img, val_dir / img.name)

        print(f"  {label:35s}  {n:>4d} selected  (train={len(train_imgs)}, val={len(val_imgs)})")

    return sorted_labels


# =====================================================================
#  PART 2 — TRAIN
# =====================================================================
def train_model():
    """Train YOLOv8n-cls on the prepared dataset."""
    from ultralytics import YOLO

    print("\nLoading YOLOv8n-cls pretrained checkpoint...")
    model = YOLO("yolov8n-cls.pt")

    print("Starting training...\n")
    results = model.train(
        data=str(TRAINING_DIR),
        epochs=50,
        imgsz=224,
        batch=16,
        device="cpu",
        project=str(PROJECT_ROOT / "outputs" / "training"),
        name="india_agri_v1",
        patience=10,
        workers=2,
        exist_ok=True,
    )
    return results


# =====================================================================
#  PART 3 — SAVE MODEL
# =====================================================================
def save_model():
    """Copy best.pt to models/india_agri_cls.pt."""
    src = PROJECT_ROOT / "outputs" / "training" / "india_agri_v1" / "weights" / "best.pt"
    dst_dir = PROJECT_ROOT / "models"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "india_agri_cls.pt"

    if src.is_file():
        shutil.copy(src, dst)
        print(f"\nModel saved to {dst}")
    else:
        # Fallback to last.pt
        last = src.parent / "last.pt"
        if last.is_file():
            shutil.copy(last, dst)
            print(f"\nlast.pt saved to {dst} (best.pt not found)")
        else:
            print(f"\nWARNING: No weights found at {src.parent}")


# =====================================================================
#  PART 4 — QUICK TEST
# =====================================================================
def quick_test():
    """Run the trained model on 5 random Leaf Blast images."""
    from ultralytics import YOLO

    model_path = PROJECT_ROOT / "models" / "india_agri_cls.pt"
    if not model_path.is_file():
        print("\nWARNING: Model not found, skipping test.")
        return

    model = YOLO(str(model_path))

    test_dir = RICE_DIR / "Leaf Blast"
    if not test_dir.is_dir():
        print(f"\nWARNING: Test folder not found: {test_dir}")
        return

    all_images = get_images(test_dir)
    if not all_images:
        print("\nWARNING: No images in Leaf Blast folder.")
        return

    samples = random.sample(all_images, min(5, len(all_images)))

    print(f"\nQuick test - 5 random images from {test_dir.name}/:\n")
    for img_path in samples:
        results = model(str(img_path), verbose=False)
        if results and results[0].probs is not None:
            probs = results[0].probs
            top1_idx = int(probs.top1)
            top1_conf = float(probs.top1conf)
            pred_label = results[0].names[top1_idx]
            print(f"  {img_path.name:40s}  predicted: {pred_label:30s}  confidence: {top1_conf:.1%}")
        else:
            print(f"  {img_path.name:40s}  WARNING: no prediction")


# =====================================================================
#  PART 5 — UPDATE phone_connect.py
# =====================================================================
def update_phone_connect():
    """Patch the default model path in scripts/phone_connect.py."""
    pc_path = PROJECT_ROOT / "scripts" / "phone_connect.py"
    if not pc_path.is_file():
        print("\nWARNING: phone_connect.py not found, skipping patch.")
        return

    old_text = pc_path.read_text(encoding="utf-8")

    # Find and replace the default model path in the argparse section
    old_default = 'default=str(_PROJECT_ROOT / "models" / "yolov8n-seg.pt")'
    new_default = 'default=str(_PROJECT_ROOT / "models" / "india_agri_cls.pt")'

    if old_default in old_text:
        new_text = old_text.replace(old_default, new_default)
        pc_path.write_text(new_text, encoding="utf-8")
        print("\nphone_connect.py updated: default model -> models/india_agri_cls.pt")
    elif new_default in old_text:
        print("\nphone_connect.py already uses india_agri_cls.pt")
    else:
        print("\nWARNING: Could not find model path line in phone_connect.py")
        print(f"   Manually set --model to: models/india_agri_cls.pt")


# =====================================================================
#  MAIN
# =====================================================================
def main():
    random.seed(42)

    print("=" * 65)
    print("  AGRI-DRONE MODEL TRAINER")
    print("  YOLOv8n-cls - India Wheat-Rice Disease Classifier")
    print("=" * 65)

    # Part 1
    print("\n" + "=" * 65)
    print("  PART 1 - PREPARE DATA")
    print("=" * 65)
    classes = prepare_data()
    if not classes:
        print("\nWARNING: No classes found. Check your data/raw/ folders.")
        return
    labels = copy_to_training(classes)

    # Part 2
    print("\n" + "=" * 65)
    print("  PART 2 - TRAIN (YOLOv8n-cls, 50 epochs, CPU)")
    print("=" * 65)
    train_model()

    # Part 3
    print("\n" + "=" * 65)
    print("  PART 3 - SAVE MODEL")
    print("=" * 65)
    save_model()

    # Part 4
    print("\n" + "=" * 65)
    print("  PART 4 - QUICK TEST")
    print("=" * 65)
    quick_test()

    # Part 5
    print("\n" + "=" * 65)
    print("  PART 5 - UPDATE phone_connect.py")
    print("=" * 65)
    update_phone_connect()

    print("\n" + "=" * 65)
    print("  DONE! Model ready at models/india_agri_cls.pt")
    print("  Run:  python scripts/phone_connect.py --crop wheat")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
