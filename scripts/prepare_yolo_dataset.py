#!/usr/bin/env python3
"""
prepare_yolo_dataset.py — Prepare YOLO detection/segmentation dataset.

This script helps you organize annotated datasets for YOLO training.
It supports two workflows:

  1. ROBOFLOW EXPORT: Download from Roboflow Universe and convert to YOLO format.
     - Go to https://universe.roboflow.com
     - Search "wheat disease detection" or "rice leaf disease"
     - Export as "YOLOv8" format → download ZIP
     - Extract into data/yolo_raw/<dataset_name>/

  2. MANUAL ANNOTATION: Annotate your existing images.
     - Upload images to https://app.roboflow.com (free academic tier)
     - Draw bounding boxes around disease areas
     - Export as "YOLOv8" format → download ZIP
     - Extract into data/yolo_raw/<dataset_name>/

After placing data, run this script to merge, validate, and split.

Usage:
    python scripts/prepare_yolo_dataset.py
"""
import os
import random
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "yolo_raw"
DATASET_DIR = PROJECT_ROOT / "data" / "yolo_dataset"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ── Unified class map ──
# Maps various dataset class names to our standard classes.
# Add new mappings as you incorporate more datasets.
CLASS_MAP = {
    # Wheat diseases
    "fusarium_head_blight": 0,
    "fusarium": 0,
    "fhb": 0,
    "scab": 0,
    "wheat_leaf_rust": 1,
    "leaf_rust": 1,
    "brown_rust": 1,
    "wheat_yellow_rust": 2,
    "yellow_rust": 2,
    "stripe_rust": 2,
    "wheat_powdery_mildew": 3,
    "powdery_mildew": 3,
    "mildew": 3,
    "wheat_septoria": 4,
    "septoria": 4,
    "wheat_blast": 5,
    "wheat_smut": 6,
    "smut": 6,
    "wheat_aphid": 7,
    "aphid": 7,
    "wheat_leaf_blight": 8,
    "leaf_blight": 8,
    "wheat_tan_spot": 9,
    "tan_spot": 9,
    # Rice diseases
    "rice_blast": 10,
    "blast": 10,
    "rice_bacterial_blight": 11,
    "bacterial_leaf_blight": 11,
    "bacterial_blight": 11,
    "rice_brown_spot": 12,
    "brown_spot": 12,
    "rice_sheath_blight": 13,
    "sheath_blight": 13,
    "rice_leaf_scald": 14,
    "leaf_scald": 14,
    # Healthy
    "healthy": 15,
    "healthy_wheat": 15,
    "healthy_rice": 15,
}

CLASS_NAMES = {
    0: "fusarium_head_blight",
    1: "wheat_leaf_rust",
    2: "wheat_yellow_rust",
    3: "wheat_powdery_mildew",
    4: "wheat_septoria",
    5: "wheat_blast",
    6: "wheat_smut",
    7: "wheat_aphid",
    8: "wheat_leaf_blight",
    9: "wheat_tan_spot",
    10: "rice_blast",
    11: "rice_bacterial_blight",
    12: "rice_brown_spot",
    13: "rice_sheath_blight",
    14: "rice_leaf_scald",
    15: "healthy",
}


def find_datasets() -> list[Path]:
    """Find all dataset directories in data/yolo_raw/."""
    if not RAW_DIR.exists():
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        return []
    datasets = [d for d in RAW_DIR.iterdir() if d.is_dir()]
    return sorted(datasets)


def read_dataset_classes(dataset_path: Path) -> dict[int, str]:
    """Read class names from a dataset's data.yaml or classes.txt."""
    classes = {}

    # Try data.yaml
    yaml_path = dataset_path / "data.yaml"
    if yaml_path.exists():
        import yaml
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        if "names" in data:
            names = data["names"]
            if isinstance(names, dict):
                classes = {int(k): v for k, v in names.items()}
            elif isinstance(names, list):
                classes = {i: n for i, n in enumerate(names)}

    # Try classes.txt
    if not classes:
        txt_path = dataset_path / "classes.txt"
        if txt_path.exists():
            lines = txt_path.read_text().strip().split("\n")
            classes = {i: line.strip() for i, line in enumerate(lines)}

    return classes


def remap_class_id(original_id: int, original_classes: dict[int, str]) -> int | None:
    """Remap a dataset's class ID to our unified class ID."""
    if original_id not in original_classes:
        return None
    name = original_classes[original_id].lower().strip().replace(" ", "_").replace("-", "_")
    return CLASS_MAP.get(name)


def collect_images_and_labels(dataset_path: Path) -> list[tuple[Path, Path, dict[int, str]]]:
    """Find all image+label pairs in a dataset directory."""
    pairs = []
    original_classes = read_dataset_classes(dataset_path)

    # Look for standard YOLO directory structures
    for split in ["train", "valid", "val", "test", ""]:
        img_dir = dataset_path / split / "images" if split else dataset_path / "images"
        lbl_dir = dataset_path / split / "labels" if split else dataset_path / "labels"

        if not img_dir.is_dir() or not lbl_dir.is_dir():
            continue

        for img_file in img_dir.iterdir():
            if img_file.suffix.lower() in IMAGE_EXTS:
                lbl_file = lbl_dir / (img_file.stem + ".txt")
                if lbl_file.exists():
                    pairs.append((img_file, lbl_file, original_classes))

    return pairs


def remap_label_file(label_path: Path, original_classes: dict[int, str]) -> list[str]:
    """Read a YOLO label file and remap class IDs to unified classes."""
    remapped = []
    for line in label_path.read_text().strip().split("\n"):
        if not line.strip():
            continue
        parts = line.strip().split()
        original_id = int(parts[0])
        new_id = remap_class_id(original_id, original_classes)
        if new_id is not None:
            remapped.append(f"{new_id} " + " ".join(parts[1:]))
    return remapped


def build_dataset():
    """Merge all raw datasets into a single unified YOLO dataset."""
    datasets = find_datasets()
    if not datasets:
        print(f"\nNo datasets found in {RAW_DIR}/")
        print("\nTo get started:")
        print("  1. Go to https://universe.roboflow.com")
        print("  2. Search for 'wheat disease detection' or 'rice leaf disease'")
        print("  3. Export as 'YOLOv8' format")
        print(f"  4. Extract the ZIP into {RAW_DIR}/<dataset_name>/")
        print("  5. Run this script again")
        return

    print(f"\nFound {len(datasets)} dataset(s):")
    all_pairs = []
    for ds in datasets:
        pairs = collect_images_and_labels(ds)
        print(f"  {ds.name}: {len(pairs)} image-label pairs")
        all_pairs.extend(pairs)

    if not all_pairs:
        print("\nNo image-label pairs found. Check directory structure.")
        print("Expected: <dataset>/train/images/*.jpg + <dataset>/train/labels/*.txt")
        return

    # Clean output dir
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)

    # Shuffle and split 80/10/10
    random.seed(42)
    random.shuffle(all_pairs)
    n = len(all_pairs)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)

    splits = {
        "train": all_pairs[:n_train],
        "val": all_pairs[n_train:n_train + n_val],
        "test": all_pairs[n_train + n_val:],
    }

    class_counts = {i: 0 for i in CLASS_NAMES}
    total_remapped = 0
    total_skipped = 0

    for split_name, pairs in splits.items():
        img_dir = DATASET_DIR / split_name / "images"
        lbl_dir = DATASET_DIR / split_name / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for idx, (img_path, lbl_path, orig_classes) in enumerate(pairs):
            remapped_lines = remap_label_file(lbl_path, orig_classes)
            if not remapped_lines:
                total_skipped += 1
                continue

            # Copy image with unique name to avoid collisions
            ext = img_path.suffix
            new_name = f"{split_name}_{idx:05d}"
            shutil.copy2(img_path, img_dir / f"{new_name}{ext}")

            # Write remapped label
            (lbl_dir / f"{new_name}.txt").write_text("\n".join(remapped_lines) + "\n")
            total_remapped += 1

            # Count classes
            for line in remapped_lines:
                cls_id = int(line.split()[0])
                class_counts[cls_id] = class_counts.get(cls_id, 0) + 1

    # Write data.yaml for YOLO training
    yaml_content = f"""# Auto-generated YOLO dataset config
path: {DATASET_DIR}
train: train/images
val: val/images
test: test/images

names:
"""
    for cls_id in sorted(CLASS_NAMES.keys()):
        yaml_content += f"  {cls_id}: {CLASS_NAMES[cls_id]}\n"

    (DATASET_DIR / "data.yaml").write_text(yaml_content)

    print(f"\nDataset built: {total_remapped} images ({total_skipped} skipped)")
    print(f"  train: {len(splits['train'])} | val: {len(splits['val'])} | test: {len(splits['test'])}")
    print(f"\nClass distribution:")
    for cls_id, count in sorted(class_counts.items()):
        if count > 0:
            print(f"  {CLASS_NAMES[cls_id]:30s}  {count:>5d} annotations")
    print(f"\nConfig saved to: {DATASET_DIR / 'data.yaml'}")


def main():
    print("=" * 65)
    print("  YOLO DATASET PREPARATION")
    print("  Merge & remap annotated datasets for detection training")
    print("=" * 65)
    build_dataset()


if __name__ == "__main__":
    main()
