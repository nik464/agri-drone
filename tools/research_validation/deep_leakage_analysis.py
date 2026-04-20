#!/usr/bin/env python3
"""Deep leakage analysis using the split manifest's per-class file lists."""
import json, re
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent.parent
MANIFEST = ROOT / "evaluate" / "data_split_manifest.json"

with open(MANIFEST, encoding="utf-8") as f:
    manifest = json.load(f)

def extract_base_id(filename):
    name = Path(filename).stem
    m = re.match(r'^aug_(\d+)_(\d+)$', name)
    if m:
        return int(m.group(2)), f"aug_{m.group(1)}"
    m = re.match(r'^(\d+)$', name)
    if m:
        return int(m.group(1)), "original"
    return None, "unknown"

print("=" * 70)
print("DEEP LEAKAGE ANALYSIS — from data_split_manifest.json")
print("=" * 70)

total_overlap = 0
total_test_base_ids = 0
total_train_base_ids = 0
overlap_examples = []

for cls in sorted(manifest["train"].keys()):
    train_files = manifest["train"].get(cls, [])
    val_files = manifest["val"].get(cls, [])
    test_files = manifest["test"].get(cls, [])
    
    # Extract base IDs per split
    train_bases = defaultdict(list)
    test_bases = defaultdict(list)
    val_bases = defaultdict(list)
    
    for f in train_files:
        bid, variant = extract_base_id(f)
        if bid is not None:
            train_bases[bid].append((f, variant))
    
    for f in test_files:
        bid, variant = extract_base_id(f)
        if bid is not None:
            test_bases[bid].append((f, variant))
    
    for f in val_files:
        bid, variant = extract_base_id(f)
        if bid is not None:
            val_bases[bid].append((f, variant))
    
    # Find overlapping base IDs between train and test
    overlap = set(train_bases.keys()) & set(test_bases.keys())
    tv_overlap = set(train_bases.keys()) & set(val_bases.keys())
    
    total_overlap += len(overlap)
    total_test_base_ids += len(test_bases)
    total_train_base_ids += len(train_bases)
    
    aug_in_train = sum(1 for f in train_files if f.startswith("aug_"))
    aug_in_test = sum(1 for f in test_files if f.startswith("aug_"))
    
    if overlap:
        print(f"\n[{cls}] train={len(train_files)} val={len(val_files)} test={len(test_files)}")
        print(f"  aug_in_train={aug_in_train} aug_in_test={aug_in_test}")
        print(f"  OVERLAPPING base IDs (train∩test): {len(overlap)}")
        print(f"  OVERLAPPING base IDs (train∩val):  {len(tv_overlap)}")
        for bid in sorted(overlap)[:3]:
            train_v = [v for _, v in train_bases[bid]]
            test_v = [v for _, v in test_bases[bid]]
            print(f"    base_id={bid}: train variants={train_v}, test variants={test_v}")
            overlap_examples.append({
                "class": cls, "base_id": bid,
                "train_variants": train_v, "test_variants": test_v
            })

print(f"\n{'='*70}")
print(f"SUMMARY:")
print(f"  Total unique base IDs in train: {total_train_base_ids}")
print(f"  Total unique base IDs in test:  {total_test_base_ids}")
print(f"  Total overlapping base IDs:     {total_overlap}")
if total_test_base_ids > 0:
    print(f"  Overlap rate:                   {total_overlap/total_test_base_ids*100:.1f}%")
print(f"\nLEAKAGE RISK: {'HIGH' if total_overlap > 50 else 'MODERATE' if total_overlap > 10 else 'LOW' if total_overlap > 0 else 'NONE'}")
print(f"{'='*70}")
