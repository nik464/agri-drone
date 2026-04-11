#!/usr/bin/env python3
"""
Noise Pipeline — Simulate real-world degradation on a clean test set.

Applies five perturbation categories to every image:
  1. Motion blur          — random-angle linear kernel
  2. Lighting variation   — brightness ± and contrast scaling
  3. Sensor noise         — Gaussian + salt-and-pepper
  4. Occlusion            — random rectangular patches
  5. Background clutter   — textured overlay on border regions

Output mirrors the input folder structure:
  <output_dir>/
    <class_a>/
      img001.jpg
      ...
    <class_b>/
      ...

Usage:
    python evaluate/noise_pipeline.py
    python evaluate/noise_pipeline.py --test-dir data/training/test --output-dir evaluate/noisy_dataset
    python evaluate/noise_pipeline.py --seed 42 --severity medium
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

# ════════════════════════════════════════════════════════════════
# Severity presets (low / medium / high)
# ════════════════════════════════════════════════════════════════

SEVERITY_PRESETS = {
    "low": {
        "blur_ksize_range": (3, 7),
        "brightness_range": (-25, 25),
        "contrast_range": (0.85, 1.15),
        "gaussian_sigma_range": (5, 15),
        "sp_amount": 0.01,
        "occlude_count": (1, 2),
        "occlude_size_frac": (0.03, 0.08),
        "clutter_alpha": 0.10,
        "clutter_border_frac": 0.10,
    },
    "medium": {
        "blur_ksize_range": (5, 15),
        "brightness_range": (-50, 50),
        "contrast_range": (0.7, 1.3),
        "gaussian_sigma_range": (10, 30),
        "sp_amount": 0.03,
        "occlude_count": (2, 5),
        "occlude_size_frac": (0.05, 0.15),
        "clutter_alpha": 0.20,
        "clutter_border_frac": 0.15,
    },
    "high": {
        "blur_ksize_range": (9, 25),
        "brightness_range": (-80, 80),
        "contrast_range": (0.5, 1.5),
        "gaussian_sigma_range": (20, 50),
        "sp_amount": 0.06,
        "occlude_count": (3, 8),
        "occlude_size_frac": (0.08, 0.25),
        "clutter_alpha": 0.35,
        "clutter_border_frac": 0.20,
    },
}


# ════════════════════════════════════════════════════════════════
# Individual perturbation functions
# ════════════════════════════════════════════════════════════════

def apply_motion_blur(img: np.ndarray, ksize_range: tuple[int, int]) -> np.ndarray:
    """Linear motion blur with random angle."""
    ksize = random.randint(ksize_range[0], ksize_range[1])
    if ksize % 2 == 0:
        ksize += 1
    angle = random.uniform(0, 360)
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    cx, cy = ksize // 2, ksize // 2
    rad = np.deg2rad(angle)
    dx, dy = np.cos(rad), np.sin(rad)
    for i in range(ksize):
        t = i - cx
        x = int(round(cx + t * dx))
        y = int(round(cy + t * dy))
        if 0 <= x < ksize and 0 <= y < ksize:
            kernel[y, x] = 1.0
    kernel /= max(kernel.sum(), 1.0)
    return cv2.filter2D(img, -1, kernel)


def apply_lighting_variation(
    img: np.ndarray,
    brightness_range: tuple[int, int],
    contrast_range: tuple[float, float],
) -> np.ndarray:
    """Random brightness shift and contrast scaling."""
    brightness = random.randint(brightness_range[0], brightness_range[1])
    contrast = random.uniform(contrast_range[0], contrast_range[1])
    result = img.astype(np.float32) * contrast + brightness
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_gaussian_noise(img: np.ndarray, sigma_range: tuple[int, int]) -> np.ndarray:
    """Additive Gaussian noise."""
    sigma = random.uniform(sigma_range[0], sigma_range[1])
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    result = img.astype(np.float32) + noise
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_salt_pepper(img: np.ndarray, amount: float) -> np.ndarray:
    """Salt-and-pepper noise (vectorized)."""
    out = img.copy()
    h, w = out.shape[:2]
    # Generate random mask: values < amount/2 become salt, > 1-amount/2 become pepper
    rng = np.random.random((h, w))
    out[rng < amount / 2] = 255
    out[rng > 1 - amount / 2] = 0
    return out


def apply_occlusion(
    img: np.ndarray,
    count_range: tuple[int, int],
    size_frac_range: tuple[float, float],
) -> np.ndarray:
    """Random rectangular occlusion patches (filled with gray/black)."""
    out = img.copy()
    h, w = out.shape[:2]
    n_patches = random.randint(count_range[0], count_range[1])
    for _ in range(n_patches):
        frac = random.uniform(size_frac_range[0], size_frac_range[1])
        pw = int(w * frac)
        ph = int(h * frac)
        x1 = random.randint(0, max(0, w - pw))
        y1 = random.randint(0, max(0, h - ph))
        # Random fill: gray, black, or mean-color
        fill_type = random.choice(["gray", "black", "mean"])
        if fill_type == "gray":
            out[y1 : y1 + ph, x1 : x1 + pw] = 128
        elif fill_type == "black":
            out[y1 : y1 + ph, x1 : x1 + pw] = 0
        else:
            mean_color = img[y1 : y1 + ph, x1 : x1 + pw].mean(axis=(0, 1))
            out[y1 : y1 + ph, x1 : x1 + pw] = mean_color.astype(np.uint8)
    return out


def apply_background_clutter(
    img: np.ndarray,
    alpha: float,
    border_frac: float,
) -> np.ndarray:
    """Add textured noise to image borders to simulate background clutter."""
    out = img.copy()
    h, w = out.shape[:2]
    bh = int(h * border_frac)
    bw = int(w * border_frac)

    # Generate random texture (Perlin-like via resized noise)
    small = np.random.randint(0, 256, (max(1, h // 8), max(1, w // 8), 3), dtype=np.uint8)
    texture = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    # Create border mask
    mask = np.zeros((h, w), dtype=np.float32)
    mask[:bh, :] = 1.0   # top
    mask[-bh:, :] = 1.0  # bottom
    mask[:, :bw] = 1.0   # left
    mask[:, -bw:] = 1.0  # right

    # Smooth transition
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(1, bh // 2))
    mask = np.clip(mask, 0, 1)

    # Blend
    mask_3ch = mask[:, :, np.newaxis] * alpha
    result = out.astype(np.float32) * (1 - mask_3ch) + texture.astype(np.float32) * mask_3ch
    return np.clip(result, 0, 255).astype(np.uint8)


# ════════════════════════════════════════════════════════════════
# Combined perturbation
# ════════════════════════════════════════════════════════════════

def apply_all_perturbations(img: np.ndarray, params: dict) -> np.ndarray:
    """Apply all five perturbation categories sequentially."""
    # 1. Motion blur
    img = apply_motion_blur(img, params["blur_ksize_range"])
    # 2. Lighting variation
    img = apply_lighting_variation(img, params["brightness_range"], params["contrast_range"])
    # 3. Sensor noise (Gaussian + salt-pepper)
    img = apply_gaussian_noise(img, params["gaussian_sigma_range"])
    img = apply_salt_pepper(img, params["sp_amount"])
    # 4. Occlusion
    img = apply_occlusion(img, params["occlude_count"], params["occlude_size_frac"])
    # 5. Background clutter
    img = apply_background_clutter(img, params["clutter_alpha"], params["clutter_border_frac"])
    return img


# ════════════════════════════════════════════════════════════════
# Dataset generation
# ════════════════════════════════════════════════════════════════

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def generate_noisy_dataset(
    test_dir: Path,
    output_dir: Path,
    severity: str = "medium",
    seed: int = 42,
) -> dict:
    """Generate a noisy copy of the test dataset.

    Returns summary dict with counts.
    """
    random.seed(seed)
    np.random.seed(seed)

    params = SEVERITY_PRESETS[severity]
    output_dir.mkdir(parents=True, exist_ok=True)

    classes = sorted([d.name for d in test_dir.iterdir() if d.is_dir()])
    total = 0
    per_class_count = {}

    for cls in classes:
        cls_in = test_dir / cls
        cls_out = output_dir / cls
        cls_out.mkdir(parents=True, exist_ok=True)
        count = 0

        for img_path in sorted(cls_in.glob("*")):
            if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # Resize to model input size (224x224) before perturbation
            # This matches YOLO inference preprocessing and prevents
            # processing multi-megapixel raw photos
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

            noisy = apply_all_perturbations(img, params)

            out_path = cls_out / img_path.name
            cv2.imwrite(str(out_path), noisy, [cv2.IMWRITE_JPEG_QUALITY, 95])
            count += 1

        per_class_count[cls] = count
        total += count
        print(f"  {cls}: {count} images")

    summary = {
        "source_dir": str(test_dir),
        "output_dir": str(output_dir),
        "severity": severity,
        "seed": seed,
        "total_images": total,
        "n_classes": len(classes),
        "per_class_count": per_class_count,
        "perturbations": [
            "motion_blur",
            "lighting_variation",
            "gaussian_noise",
            "salt_pepper_noise",
            "occlusion",
            "background_clutter",
        ],
        "params": {k: str(v) for k, v in params.items()},
    }

    # Save manifest
    manifest_path = output_dir / "noise_manifest.json"
    import json
    with open(manifest_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nManifest saved to {manifest_path}")

    return summary


# ════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Generate noisy test dataset")
    parser.add_argument("--test-dir", type=str, default="data/training/test",
                        help="Path to clean test dataset")
    parser.add_argument("--output-dir", type=str, default="evaluate/noisy_dataset",
                        help="Output directory for noisy images")
    parser.add_argument("--severity", choices=["low", "medium", "high"], default="medium",
                        help="Perturbation severity level")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    test_dir = Path(args.test_dir)
    output_dir = Path(args.output_dir)

    if not test_dir.exists():
        print(f"ERROR: Test directory not found: {test_dir}")
        return

    print(f"Generating noisy dataset (severity={args.severity}, seed={args.seed})")
    print(f"  Source: {test_dir}")
    print(f"  Output: {output_dir}")
    print()

    summary = generate_noisy_dataset(test_dir, output_dir, args.severity, args.seed)

    print(f"\nDone: {summary['total_images']} images across {summary['n_classes']} classes")


if __name__ == "__main__":
    main()
