#!/usr/bin/env python3
"""Deterministically split a dataset directory into train/val/test.

Layout expected::

    input_dir/
        class_a/
            img1.jpg
            aug_0_img1.jpg   ← augmentation of img1
            aug_1_img1.jpg
            ...
        class_b/
            ...

**Group-aware splitting**: augmented images (``aug_N_BASEID.ext``) are grouped
with their base image so that ALL variants land in the SAME split.  This
prevents train/test leakage where the model memorises augmentations of a
training image and "cheats" on the test set.

Produces three CSVs ``{train,val,test}.csv`` under ``--out-dir`` with columns
``path,label``. A fixed seed makes splits reproducible; the split manifest
records SHA256 of the sorted file list so drift is detectable.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path

# Pattern: aug_<N>_<base_id>.<ext>  →  group key = base_id
_AUG_RE = re.compile(r"^aug_\d+_(.+)$")


def _base_id(filename: str) -> str:
    """Extract the group key (base image ID) from a filename.

    ``aug_0_101.jpg`` → ``101.jpg``
    ``aug_3_leaf_rust_42.png`` → ``leaf_rust_42.png``
    ``normal_image.jpg`` → ``normal_image.jpg``  (unchanged)
    """
    stem = Path(filename).stem
    ext = Path(filename).suffix
    m = _AUG_RE.match(stem)
    if m:
        return m.group(1) + ext
    return filename


def _sha256_of_file_list(files: list[Path]) -> str:
    h = hashlib.sha256()
    for f in sorted(files):
        h.update(str(f).encode("utf-8"))
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description="Group-aware train/val/test splitter")
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--ratios", type=float, nargs=3, default=(0.7, 0.15, 0.15),
                   help="Train/val/test ratios (must sum to ~1.0)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if not args.input_dir.exists():
        print(f"[err] input dir not found: {args.input_dir}")
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    splits: dict[str, list[tuple[str, str]]] = {"train": [], "val": [], "test": []}
    tot = {"train": 0, "val": 0, "test": 0}
    groups_total = 0
    rt, rv, _ = args.ratios

    for cls_dir in sorted(d for d in args.input_dir.iterdir() if d.is_dir()):
        # ── Group files by base image ID ──
        groups: dict[str, list[Path]] = defaultdict(list)
        for f in sorted(cls_dir.rglob("*")):
            if f.is_file():
                gid = _base_id(f.name)
                groups[gid].append(f)

        # ── Shuffle and split at the GROUP level ──
        group_keys = sorted(groups.keys())
        rng.shuffle(group_keys)
        n = len(group_keys)
        n_tr = int(n * rt)
        n_va = int(n * rv)
        groups_total += n

        for i, gid in enumerate(group_keys):
            bucket = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
            for f in groups[gid]:
                splits[bucket].append(
                    (str(f.relative_to(args.input_dir)), cls_dir.name)
                )
                tot[bucket] += 1

    for name, rows in splits.items():
        with (args.out_dir / f"{name}.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path", "label"])
            w.writerows(rows)

    all_files = [Path(r[0]) for rows in splits.values() for r in rows]
    manifest = {
        "seed": args.seed,
        "ratios": list(args.ratios),
        "counts": tot,
        "n_groups": groups_total,
        "group_aware": True,
        "file_list_sha256": _sha256_of_file_list(all_files),
        "input_dir": str(args.input_dir),
    }
    (args.out_dir / "splits_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"  splits: {tot}  groups: {groups_total}  manifest: {args.out_dir / 'splits_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
