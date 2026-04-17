#!/usr/bin/env python3
"""Download (or point to) the datasets used in the agri-drone paper.

Datasets are large (several GB) and distributed under heterogeneous licences.
We therefore **do not bundle** them. This script is a single authoritative
entry point for locating or fetching each dataset.

It prefers local caches under ``datasets/`` so CI and smoke tests never
attempt real downloads.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "datasets"

SOURCES = {
    "plantvillage": {
        "url": "https://github.com/spMohanty/PlantVillage-Dataset",
        "licence": "CC-BY-SA-4.0",
        "local_hint": DATA_DIR / "externals" / "plantvillage",
        "notes": "Close-up leaf photos, lab conditions. Used for Config A/B/v3.",
    },
    "pdt": {
        "url": "https://github.com/Kaka-Shi/PDT",
        "licence": "see repo",
        "local_hint": DATA_DIR / "externals" / "PDT_datasets",
        "notes": "Drone-altitude pest+disease images; used for cross-dataset evaluation.",
    },
    "riceleaf": {
        "url": "https://archive.ics.uci.edu/ml/datasets/Rice+Leaf+Diseases",
        "licence": "UCI-ML (open)",
        "local_hint": DATA_DIR / "externals" / "riceleaf",
        "notes": "UCI rice leaf disease dataset (Prajapati et al. 2017).",
    },
    "ricepest": {
        "url": "https://www.kaggle.com/datasets/shrupyag001/rice-leaf-disease-images",
        "licence": "Kaggle terms",
        "local_hint": DATA_DIR / "externals" / "ricepest",
        "notes": "Kaggle rice pest dataset; referenced in §5.4.",
    },
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--list", action="store_true",
                   help="list known datasets and their expected local paths")
    p.add_argument("--check", action="store_true",
                   help="check which datasets are present locally")
    args = p.parse_args()

    if args.list or not (args.check):
        for name, meta in SOURCES.items():
            print(f"[{name}]")
            print(f"  url     : {meta['url']}")
            print(f"  licence : {meta['licence']}")
            print(f"  local   : {meta['local_hint']}")
            print(f"  notes   : {meta['notes']}")
            print()

    if args.check:
        ok = 0
        for name, meta in SOURCES.items():
            present = Path(meta["local_hint"]).exists()
            marker = "[OK]" if present else "[--]"
            print(f"  {marker} {name:15s} -> {meta['local_hint']}")
            ok += int(present)
        print(f"\n{ok}/{len(SOURCES)} datasets present.")
        return 0 if ok == len(SOURCES) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
