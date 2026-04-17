#!/usr/bin/env python3
"""Smoke test — offline, CPU-only, target runtime < 60s.

Run this gate before every commit on the research-upgrade branch.

What it does
------------
1. Generates 30 synthetic 224x224 images (random noise + solid colour blobs) in
   a tmp dir, arranged as folder-per-class for 3 synthetic classes.
2. Walks the minimal public imports (knowledge base loader, feature extractor,
   rule engine, EML cost table) to make sure the package still imports cleanly.
3. Computes `evaluate.eml_analysis.compute_eml` on a tiny synthetic predictions
   CSV to verify the scoring pipeline is functional.
4. Exits 0 on success, non-zero otherwise.

Intentionally does NOT
----------------------
- Load YOLO/EfficientNet/LLaVA weights (too slow on CPU, not required for a
  smoke test).
- Hit the network.
- Write to any file outside the OS tmp dir.

Usage
-----
    python scripts/smoke_test.py
    python scripts/smoke_test.py --verbose
"""

from __future__ import annotations

import argparse
import csv
import importlib
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


SYNTHETIC_CLASSES = ["healthy_wheat", "wheat_yellow_rust", "rice_blast"]


def _gen_synthetic_images(out_dir: Path, per_class: int = 10) -> int:
    import numpy as np

    try:
        import cv2
    except Exception:
        cv2 = None

    count = 0
    for cls in SYNTHETIC_CLASSES:
        (out_dir / cls).mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            img = (np.random.default_rng(42 + count).integers(0, 255, (64, 64, 3))
                   .astype("uint8"))
            path = out_dir / cls / f"{cls}_{i:02d}.jpg"
            if cv2 is not None:
                cv2.imwrite(str(path), img)
            else:
                # Fallback: write raw bytes — we only need a file to exist.
                path.write_bytes(img.tobytes())
            count += 1
    return count


def _gen_synthetic_predictions_csv(out_path: Path) -> int:
    """A mini predictions CSV with balanced classes and mixed correctness."""
    rows = []
    for cls in SYNTHETIC_CLASSES:
        for i in range(10):
            correct = i % 3 != 0
            predicted = cls if correct else SYNTHETIC_CLASSES[(SYNTHETIC_CLASSES.index(cls) + 1) % 3]
            rows.append({
                "image": f"{cls}_{i}.jpg",
                "ground_truth": cls,
                "predicted": predicted,
                "confidence": 0.8 if correct else 0.4,
                "correct": int(correct),
                "latency_ms": 12.3,
            })
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def _check_imports(verbose: bool) -> list[str]:
    failures: list[str] = []
    to_check = [
        "agridrone",
        "agridrone.knowledge.kb_loader",
        "agridrone.vision.feature_extractor",
        "agridrone.vision.rule_engine",
    ]
    for mod in to_check:
        try:
            importlib.import_module(mod)
            if verbose:
                print(f"  [ok] import {mod}")
        except Exception as e:  # noqa: BLE001
            failures.append(f"{mod}: {e}")
            print(f"  [FAIL] import {mod}: {e}")
    return failures


def _check_eml_scoring(csv_path: Path, verbose: bool) -> list[str]:
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "evaluate"))
        from eml_analysis import compute_eml  # type: ignore
    except Exception as e:  # noqa: BLE001
        return [f"could not import evaluate.eml_analysis: {e}"]
    try:
        result = compute_eml(csv_path)
    except Exception as e:  # noqa: BLE001
        return [f"compute_eml raised: {e}"]
    if not isinstance(result, dict) or "total_eml" not in result:
        return [f"compute_eml returned malformed result: {result}"]
    if verbose:
        print(f"  [ok] EML on synthetic CSV: ₹{result['total_eml']:.2f} "
              f"across {result['n_samples']} samples")
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description="AgriDrone smoke test (< 60s, CPU only)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    print("=" * 60)
    print("  AgriDrone smoke test (offline, CPU)")
    print("=" * 60)

    all_failures: list[str] = []

    with tempfile.TemporaryDirectory(prefix="agridrone_smoke_") as tmp:
        tmp_root = Path(tmp)
        img_dir = tmp_root / "synthetic_dataset"
        n_imgs = _gen_synthetic_images(img_dir)
        print(f"  [ok] generated {n_imgs} synthetic images in {img_dir}")

        csv_path = tmp_root / "predictions.csv"
        n_rows = _gen_synthetic_predictions_csv(csv_path)
        print(f"  [ok] generated {n_rows}-row predictions CSV")

        all_failures.extend(_check_imports(args.verbose))
        all_failures.extend(_check_eml_scoring(csv_path, args.verbose))

    elapsed = time.perf_counter() - t0
    print("-" * 60)
    print(f"  Elapsed: {elapsed:.2f}s (budget: 60s)")
    if elapsed > 60:
        print("  WARNING: smoke test exceeded 60s budget")

    if all_failures:
        print(f"  [FAIL] {len(all_failures)} checks failed:")
        for f in all_failures:
            print(f"    - {f}")
        return 1

    print("  [PASS] all smoke checks green")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
