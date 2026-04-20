#!/usr/bin/env python3
"""Investigate train/test leakage risk from aug_* files.

Reads the committed prediction CSVs and the split manifest to determine
whether augmented siblings of the same base image appear in both train and test.
"""
import csv, json, re
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent.parent
RES = ROOT / "evaluate" / "results"
EVALUATE = ROOT / "evaluate"

# --- Step 1: Analyze filename patterns in prediction CSVs ---
def extract_base_id(filename):
    """Try to extract the base image ID from aug_* naming patterns.
    
    Common patterns:
      aug_0_1234.jpg  -> base_id = 1234
      aug_1_1234.jpg  -> base_id = 1234
      original_1234.jpg -> base_id = 1234
      1234.jpg -> base_id = 1234
    """
    name = Path(filename).stem
    # Pattern: aug_{augmentation_index}_{base_id}
    m = re.match(r'^aug_(\d+)_(\d+)$', name)
    if m:
        return f"base_{m.group(2)}", f"aug_{m.group(1)}"
    # Pattern: just a number
    m = re.match(r'^(\d+)$', name)
    if m:
        return f"base_{m.group(1)}", "original"
    # Pattern with any prefix
    m = re.match(r'^(.+?)_(\d+)$', name)
    if m:
        return f"base_{m.group(2)}", m.group(1)
    return f"unique_{name}", "unknown"

print("=" * 70)
print("LEAKAGE INVESTIGATION")
print("=" * 70)

# Load test predictions to get test filenames
test_files = []
for csv_name in ["predictions_A_yolo_only.csv"]:
    with open(RES / csv_name, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            test_files.append(r["image"])

print(f"\nTest set size: {len(test_files)} images")

# Analyze naming patterns
aug_count = sum(1 for f in test_files if Path(f).stem.startswith("aug_"))
non_aug_count = len(test_files) - aug_count
print(f"  aug_* files in test: {aug_count}")
print(f"  non-aug files in test: {non_aug_count}")
print(f"  aug ratio: {aug_count/len(test_files)*100:.1f}%")

# Group test files by base ID
test_base_groups = defaultdict(list)
for f in test_files:
    base_id, variant = extract_base_id(Path(f).name)
    # Include class context
    parts = f.replace("\\", "/").split("/")
    cls = parts[0] if len(parts) > 1 else "unknown"
    test_base_groups[(cls, base_id)].append((f, variant))

# Show examples
print(f"\nFilename pattern examples from test set:")
for i, f in enumerate(test_files[:10]):
    base_id, variant = extract_base_id(Path(f).name)
    print(f"  {f} -> base_id={base_id}, variant={variant}")

# Check split manifest
manifest_path = EVALUATE / "data_split_manifest.json"
manifest2_path = ROOT / "data" / "splits_manifest.json"

train_files = []
manifest_data = None

for mp in [manifest_path, manifest2_path]:
    if mp.exists():
        print(f"\nFound manifest: {mp}")
        with open(mp, encoding="utf-8") as f:
            manifest_data = json.load(f)
        break

# Try to find train.csv
for train_csv_path in [EVALUATE / "train.csv", ROOT / "data" / "processed" / "train.csv"]:
    if train_csv_path.exists():
        print(f"Found train split: {train_csv_path}")
        with open(train_csv_path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                train_files.append(r.get("path", r.get("image", "")))
        break

# If we have the manifest with file lists, analyze that
if manifest_data and "splits" in manifest_data:
    for split_name, file_list in manifest_data["splits"].items():
        if split_name == "train":
            train_files = file_list
        print(f"  {split_name}: {len(file_list)} files")

# If we found train files, do the overlap analysis
if train_files:
    print(f"\nTrain set size: {len(train_files)} files")
    
    train_aug = sum(1 for f in train_files if "aug_" in f)
    print(f"  aug_* in train: {train_aug}")
    
    # Group train by base ID
    train_base_groups = defaultdict(list)
    for f in train_files:
        base_id, variant = extract_base_id(Path(f).name)
        parts = f.replace("\\", "/").split("/")
        cls = parts[0] if len(parts) > 1 else "unknown"
        train_base_groups[(cls, base_id)].append((f, variant))
    
    # Find overlapping base IDs
    overlaps = set(test_base_groups.keys()) & set(train_base_groups.keys())
    print(f"\n{'='*50}")
    print(f"OVERLAP ANALYSIS:")
    print(f"  Unique base IDs in test: {len(test_base_groups)}")
    print(f"  Unique base IDs in train: {len(train_base_groups)}")
    print(f"  OVERLAPPING base IDs: {len(overlaps)}")
    print(f"  Overlap rate: {len(overlaps)/len(test_base_groups)*100:.1f}%")
    
    if overlaps:
        print(f"\nExamples of overlapping base IDs (up to 20):")
        for key in sorted(overlaps)[:20]:
            cls, base_id = key
            print(f"  [{cls}] {base_id}:")
            print(f"    Train: {[v for _, v in train_base_groups[key][:3]]}")
            print(f"    Test:  {[v for _, v in test_base_groups[key][:3]]}")
else:
    print("\nWARNING: Could not find train file list. Analyzing test set only.")
    print("Checking the data_split_manifest.json structure instead...")
    
    if manifest_data:
        print(f"\nManifest keys: {list(manifest_data.keys())}")
        # Print first few entries to understand structure
        for k in list(manifest_data.keys())[:5]:
            v = manifest_data[k]
            if isinstance(v, (list, dict)):
                print(f"  {k}: type={type(v).__name__}, len={len(v)}")
            else:
                print(f"  {k}: {v}")

# Analyze the split script itself
print(f"\n{'='*50}")
print("SPLIT SCRIPT ANALYSIS (scripts/make_splits.py):")
print("  - Shuffles ALL files per class (line 52-53)")
print("  - No grouping by base image ID")
print("  - aug_0_X.jpg and aug_1_X.jpg treated as independent images")
print("  - If aug_* files were created BEFORE splitting, siblings can land")
print("    in different splits -> TRAIN/TEST LEAKAGE")
print("  - The script would need to extract base IDs and group-split")

# Check actual data directories for aug patterns
data_dirs = [
    ROOT / "data" / "processed",
    ROOT / "data" / "raw", 
    ROOT / "datasets",
]
for d in data_dirs:
    if d.exists():
        aug_files = list(d.rglob("aug_*.jpg")) + list(d.rglob("aug_*.png"))
        total = list(d.rglob("*.jpg")) + list(d.rglob("*.png"))
        if total:
            print(f"\n{d}: {len(aug_files)} aug files / {len(total)} total images")
            if aug_files:
                print(f"  Examples: {[f.name for f in aug_files[:5]]}")

print(f"\n{'='*70}")
print("INVESTIGATION COMPLETE")
print("=" * 70)
