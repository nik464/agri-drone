#!/usr/bin/env python3
"""Deterministically split a dataset directory into train/val/test.

Layout expected::

    input_dir/
        class_a/
            img1.jpg
            ...
        class_b/
            ...

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
from pathlib import Path


def _sha256_of_file_list(files: list[Path]) -> str:
    h = hashlib.sha256()
    for f in sorted(files):
        h.update(str(f).encode("utf-8"))
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--ratios", type=float, nargs=3, default=(0.7, 0.15, 0.15))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if not args.input_dir.exists():
        print(f"[err] input dir not found: {args.input_dir}")
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    splits = {"train": [], "val": [], "test": []}
    tot = {"train": 0, "val": 0, "test": 0}
    rt, rv, _ = args.ratios

    for cls_dir in sorted(p for p in args.input_dir.iterdir() if p.is_dir()):
        files = sorted(f for f in cls_dir.rglob("*") if f.is_file())
        rng.shuffle(files)
        n = len(files)
        n_tr = int(n * rt)
        n_va = int(n * rv)
        for i, f in enumerate(files):
            bucket = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
            splits[bucket].append((str(f.relative_to(args.input_dir)), cls_dir.name))
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
        "file_list_sha256": _sha256_of_file_list(all_files),
        "input_dir": str(args.input_dir),
    }
    (args.out_dir / "splits_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"  splits: {tot}  manifest: {args.out_dir/'splits_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
