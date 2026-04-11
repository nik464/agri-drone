#!/usr/bin/env python3
"""
download_all_datasets.py
========================
Downloads ALL free wheat & rice disease datasets from Roboflow Universe
and Kaggle as backup. Produces a unified data/raw/ folder ready for
prepare_yolo_dataset.py.

Usage:
    pip install roboflow
    python scripts/download_all_datasets.py

Set your FREE Roboflow API key:
    1. Go to https://roboflow.com  →  Sign up (free)
    2. Settings  →  Roboflow API  →  Private API Key
    3. Paste below or set env var  ROBOFLOW_API_KEY
"""

import os
import sys
import glob
import shutil
import subprocess
from pathlib import Path

# Fix SSL cert bundle (PostgreSQL on Windows can override this badly)
if not os.environ.get("REQUESTS_CA_BUNDLE"):
    try:
        import certifi
        os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
    except ImportError:
        pass

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "YOUR_KEY")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROBOFLOW_DIR = PROJECT_ROOT / "data" / "raw" / "roboflow"
KAGGLE_DIR = PROJECT_ROOT / "data" / "raw" / "kaggle"

# ──────────────────────────────────────────────
# ROBOFLOW DATASETS  (all free / public)
# ──────────────────────────────────────────────
ROBOFLOW_DATASETS = [
    # ── WHEAT ──
    {
        "workspace": "wheatdisease",
        "project": "wheat-disease-detection",
        "version": 1,
        "crop": "wheat",
    },
    {
        "workspace": "data-science-project-xkqtu",
        "project": "wheat-crop-disease-detection",
        "version": 1,
        "crop": "wheat",
    },
    {
        "workspace": "nfc-4xion",
        "project": "wheat-disease-detection",
        "version": 1,
        "crop": "wheat",
    },
    {
        "workspace": "mbrahim",
        "project": "wheat-disease",
        "version": 1,
        "crop": "wheat",
    },
    {
        "workspace": "loic-steven",
        "project": "wheat-disease",
        "version": 1,
        "crop": "wheat",
    },
    # ── RICE ──
    {
        "workspace": "yolo-rqava",
        "project": "rice-diseases-zoa8l",
        "version": 1,
        "crop": "rice",
    },
    {
        "workspace": "rice-disease",
        "project": "rice-leaf-disease-detection",
        "version": 1,
        "crop": "rice",
    },
    {
        "workspace": "agri-vision",
        "project": "rice-blast-detection",
        "version": 1,
        "crop": "rice",
    },
]

# ──────────────────────────────────────────────
# KAGGLE BACKUP DATASETS
# ──────────────────────────────────────────────
KAGGLE_DATASETS = [
    "kushagra3204/wheat-disease-dataset",
    "olyadgetch/wheat-leaf-dataset",
    "vbookshelf/rice-leaf-diseases",
    "minhhuy2810/rice-diseases-image-dataset",
]

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def count_images(root: Path) -> int:
    """Count image files recursively under *root*."""
    total = 0
    for ext in IMAGE_EXTS:
        total += len(list(root.rglob(f"*{ext}")))
    return total


def collect_class_names(root: Path) -> set:
    """
    Gather class names from:
      - data.yaml  (names: [...])
      - classes.txt (one per line)
      - subfolder names when dataset is image-classification layout
    """
    names: set = set()

    # 1) data.yaml
    for yaml_path in root.rglob("data.yaml"):
        try:
            import yaml  # PyYAML ships with ultralytics / roboflow

            with open(yaml_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            if isinstance(cfg, dict) and "names" in cfg:
                if isinstance(cfg["names"], list):
                    names.update(str(n) for n in cfg["names"])
                elif isinstance(cfg["names"], dict):
                    names.update(str(v) for v in cfg["names"].values())
        except Exception:
            pass

    # 2) classes.txt
    for cls_path in root.rglob("classes.txt"):
        try:
            with open(cls_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        names.add(line)
        except Exception:
            pass

    # 3) classification-style folder names (folders that contain images)
    for d in root.rglob("*"):
        if d.is_dir() and any((d / f).exists() for f in d.iterdir() if f.suffix.lower() in IMAGE_EXTS):
            # skip generic folder names
            if d.name.lower() not in {"images", "labels", "train", "valid", "test", "val"}:
                names.add(d.name)

    return names


# ──────────────────────────────────────────────
# ROBOFLOW DOWNLOAD
# ──────────────────────────────────────────────
def download_roboflow_datasets() -> tuple[int, int, set]:
    """Download all Roboflow datasets. Returns (ok_count, image_count, class_names)."""
    try:
        from roboflow import Roboflow
    except ImportError:
        print("\n[!] roboflow not installed.  Run:  pip install roboflow\n")
        return 0, 0, set()

    if ROBOFLOW_API_KEY == "YOUR_KEY":
        print(
            "\n[!] Set your Roboflow API key first!\n"
            "    export ROBOFLOW_API_KEY=<your_key>\n"
            "    or edit ROBOFLOW_API_KEY in this script.\n"
            "\n"
            "    HOW TO GET A FREE KEY:\n"
            "    1. Go to https://roboflow.com\n"
            "    2. Sign up (free)\n"
            "    3. Settings  →  Roboflow API  →  Private API Key\n"
            "    4. Copy and paste it here or set the env var.\n"
        )
        return 0, 0, set()

    try:
        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    except Exception as e:
        print(f"\n[!] Roboflow authentication failed: {e}")
        print("    Check your API key and try again.")
        print("    Go to app.roboflow.com → Settings → Roboflow API → Private Key\n")
        return 0, 0, set()

    ok = 0
    total_images = 0
    all_classes: set = set()

    for ds in ROBOFLOW_DATASETS:
        dest = str(ROBOFLOW_DIR / ds["project"])
        tag = f'{ds["crop"]}/{ds["project"]}'
        print(f"\n── Downloading  {tag}  ──")
        try:
            project = rf.workspace(ds["workspace"]).project(ds["project"])
            version = project.version(ds["version"])
            version.download("yolov8", location=dest)
            n = count_images(Path(dest))
            cls = collect_class_names(Path(dest))
            total_images += n
            all_classes.update(cls)
            ok += 1
            print(f"   ✓ {tag}  —  {n} images, classes: {sorted(cls)}")
        except Exception as e:
            print(f"   ✗ Skipped {tag}: {e}")

    return ok, total_images, all_classes


# ──────────────────────────────────────────────
# KAGGLE BACKUP DOWNLOAD
# ──────────────────────────────────────────────
def download_kaggle_datasets() -> tuple[int, int, set]:
    """Fallback: download datasets from Kaggle CLI."""
    # Check kaggle CLI available
    if shutil.which("kaggle") is None:
        print(
            "\n[!] kaggle CLI not found.  Install with:\n"
            "    pip install kaggle\n"
            "    Then place kaggle.json in ~/.kaggle/\n"
        )
        return 0, 0, set()

    ok = 0
    total_images = 0
    all_classes: set = set()

    for ds in KAGGLE_DATASETS:
        name = ds.split("/")[1]
        dest = str(KAGGLE_DIR / name)
        print(f"\n── Kaggle: {ds}  ──")
        try:
            result = subprocess.run(
                ["kaggle", "datasets", "download", ds, "-p", dest, "--unzip"],
                capture_output=True,
                text=True,
                timeout=600,
            )
            if result.returncode != 0:
                print(f"   ✗ Kaggle error: {result.stderr.strip()}")
                continue
            n = count_images(Path(dest))
            cls = collect_class_names(Path(dest))
            total_images += n
            all_classes.update(cls)
            ok += 1
            print(f"   ✓ {name}  —  {n} images, classes: {sorted(cls)}")
        except Exception as e:
            print(f"   ✗ Skipped {name}: {e}")

    return ok, total_images, all_classes


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  AgriDrone — Dataset Downloader")
    print("  Wheat + Rice Disease Datasets")
    print("=" * 60)

    # Create output dirs
    ROBOFLOW_DIR.mkdir(parents=True, exist_ok=True)
    KAGGLE_DIR.mkdir(parents=True, exist_ok=True)

    total_ok = 0
    total_images = 0
    all_classes: set = set()

    # ── Phase 1: Roboflow ──
    print("\n" + "─" * 60)
    print("PHASE 1:  Roboflow Universe downloads")
    print("─" * 60)
    r_ok, r_img, r_cls = download_roboflow_datasets()
    total_ok += r_ok
    total_images += r_img
    all_classes.update(r_cls)

    # ── Phase 2: Kaggle backup ──
    print("\n" + "─" * 60)
    print("PHASE 2:  Kaggle backup downloads")
    print("─" * 60)
    k_ok, k_img, k_cls = download_kaggle_datasets()
    total_ok += k_ok
    total_images += k_img
    all_classes.update(k_cls)

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"  Datasets downloaded : {total_ok}")
    print(f"  Total images found  : {total_images}")
    print(f"  Unique classes       : {len(all_classes)}")
    if all_classes:
        print(f"\n  Class names found:")
        for i, c in enumerate(sorted(all_classes), 1):
            print(f"    {i:>3}. {c}")
    print()
    print(f"  Roboflow data → {ROBOFLOW_DIR}")
    print(f"  Kaggle data   → {KAGGLE_DIR}")
    print()

    if total_ok == 0:
        print("  [!] No datasets were downloaded.")
        print("      Make sure your API keys are set (see instructions above).")
        sys.exit(1)

    print("  Next steps:")
    print("    1. python scripts/prepare_yolo_dataset.py   (merge & unify)")
    print("    2. python scripts/train_yolo_detector.py    (train YOLOv8)")
    print("    3. Restart backend — model auto-loads")
    print()


if __name__ == "__main__":
    main()
