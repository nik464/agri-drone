#!/usr/bin/env python3
"""
train_yolo_detector.py — Train YOLOv8 for crop disease DETECTION.

Optimized for CPU-only hardware (AMD Ryzen 5 / no CUDA):
  - Auto-tunes CPU threads, dataloader workers, and image caching
  - Real-time structured logging via Python logging module
  - YOLO callbacks for per-epoch metric reporting (no polling/sleeping)
  - Resume support from last checkpoint

Usage:
    python scripts/train_yolo_detector.py
    python scripts/train_yolo_detector.py --epochs 50 --imgsz 640
    python scripts/train_yolo_detector.py --resume outputs/training/yolo_crop_disease/weights/last.pt
    python scripts/train_yolo_detector.py --model yolov8s.pt --cache ram --workers 10
"""
import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

import torch

# ── Project paths ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "roboflow" / "rice-diseases-v2"
DATASET_YAML = DATASET_DIR / "data.yaml"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "training"
MODEL_DIR = PROJECT_ROOT / "models"

# ── Logging setup ──
LOG_FMT = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT, datefmt="%H:%M:%S",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("train")


# ═══════════════════════════════════════════════════════════════════
#  Hardware auto-tuning
# ═══════════════════════════════════════════════════════════════════

def configure_hardware(workers, cache):
    """Auto-detect optimal workers and cache strategy for this machine."""
    n_cores = os.cpu_count() or 4

    # Maximize PyTorch CPU parallelism
    torch.set_num_threads(n_cores)
    torch.set_num_interop_threads(max(1, n_cores // 2))
    log.info("CPU: %d cores, torch threads=%d, interop=%d",
             n_cores, n_cores, max(1, n_cores // 2))

    # Auto-tune dataloader workers
    if workers is None:
        workers = min(8, max(2, n_cores - 2))
    log.info("Dataloader workers: %d", workers)

    # Auto-detect cache strategy based on available RAM
    if cache is None:
        try:
            import psutil
            avail_gb = psutil.virtual_memory().available / (1024**3)
            cache = "ram" if avail_gb > 2.5 else "disk"
            log.info("Available RAM: %.1f GB -> cache=%s", avail_gb, cache)
        except ImportError:
            cache = "disk"
            log.info("psutil not available, defaulting to cache=disk")
    else:
        log.info("Image cache: %s (user-specified)", cache)

    return workers, cache


# ═══════════════════════════════════════════════════════════════════
#  YOLO training callbacks (real-time logging, no polling)
# ═══════════════════════════════════════════════════════════════════

def on_train_epoch_end(trainer):
    """Log metrics after each training epoch — called by YOLO automatically."""
    epoch = trainer.epoch + 1
    total = trainer.epochs
    loss = trainer.loss_items
    if loss is not None:
        box, cls, dfl = float(loss[0]), float(loss[1]), float(loss[2])
        log.info("EPOCH %d/%d  box=%.4f  cls=%.4f  dfl=%.4f",
                 epoch, total, box, cls, dfl)


def on_val_end(validator):
    """Log validation metrics — called after each validation run."""
    metrics = validator.metrics
    if hasattr(metrics, "box"):
        box = metrics.box
        log.info("  VAL  mAP50=%.4f  mAP50-95=%.4f  P=%.3f  R=%.3f",
                 float(box.map50), float(box.map),
                 float(box.mp), float(box.mr))


def on_train_end(trainer):
    """Log final summary when training completes."""
    log.info("=" * 60)
    log.info("TRAINING COMPLETE — %d epochs", trainer.epoch + 1)
    save_dir = trainer.save_dir
    log.info("Results saved to: %s", save_dir)

    # Read final metrics from results.csv
    results_csv = Path(save_dir) / "results.csv"
    if results_csv.exists():
        last_line = results_csv.read_text().strip().split("\n")[-1]
        fields = last_line.split(",")
        if len(fields) >= 9:
            log.info("  Final  mAP50=%.4f  mAP50-95=%.4f  P=%.4f  R=%.4f",
                     float(fields[7]), float(fields[8]),
                     float(fields[5]), float(fields[6]))
    log.info("=" * 60)


# ═══════════════════════════════════════════════════════════════════
#  Prerequisites check
# ═══════════════════════════════════════════════════════════════════

def check_prerequisites():
    """Verify dataset exists and report stats."""
    if not DATASET_YAML.exists():
        log.error("Dataset not found: %s", DATASET_YAML)
        log.error("Run: python scripts/prepare_yolo_dataset.py")
        return False

    train_dir = DATASET_DIR / "train" / "images"
    if not train_dir.exists() or not any(train_dir.iterdir()):
        log.error("No training images in: %s", train_dir)
        return False

    n_train = len(list(train_dir.iterdir()))
    n_val = 0
    for vname in ("val", "valid"):
        vdir = DATASET_DIR / vname / "images"
        if vdir.exists():
            n_val = len(list(vdir.iterdir()))
            break

    n_test = 0
    test_dir = DATASET_DIR / "test" / "images"
    if test_dir.exists():
        n_test = len(list(test_dir.iterdir()))

    log.info("Dataset: %d train / %d val / %d test images", n_train, n_val, n_test)
    return True


# ═══════════════════════════════════════════════════════════════════
#  Training
# ═══════════════════════════════════════════════════════════════════

def train(model_name, epochs, batch, device, imgsz,
          resume=None, workers=None, cache=None):
    """Train YOLOv8 detection model with hardware optimization and callbacks."""
    from ultralytics import YOLO

    workers, cache = configure_hardware(workers, cache)

    # Load model
    if resume:
        log.info("Resuming from checkpoint: %s", resume)
        model = YOLO(resume)
    else:
        log.info("Loading pretrained: %s", model_name)
        model = YOLO(model_name)

    # Register real-time logging callbacks
    model.add_callback("on_train_epoch_end", on_train_epoch_end)
    model.add_callback("on_val_end", on_val_end)
    model.add_callback("on_train_end", on_train_end)

    log.info("Config: epochs=%d batch=%d imgsz=%d device=%s workers=%d cache=%s",
             epochs, batch, imgsz, device, workers, cache)
    log.info("Dataset: %s", DATASET_YAML)

    results = model.train(
        data=str(DATASET_YAML),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(OUTPUT_DIR.resolve()),
        name="yolo_crop_disease",
        patience=20,
        workers=workers,
        cache=cache,
        exist_ok=True,
        resume=bool(resume),
        save_period=5,
        # Augmentation
        flipud=0.5,
        fliplr=0.5,
        degrees=15.0,
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.3,
        mosaic=0.8,
        mixup=0.1,
    )
    return results


# ═══════════════════════════════════════════════════════════════════
#  Post-training
# ═══════════════════════════════════════════════════════════════════

def save_model():
    """Copy best weights to models/ directory. Returns destination or None."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    dst = MODEL_DIR / "yolo_crop_disease.pt"

    candidates = [
        OUTPUT_DIR / "yolo_crop_disease" / "weights" / "best.pt",
    ]
    for pt in PROJECT_ROOT.rglob("yolo_crop_disease/weights/best.pt"):
        if pt not in candidates:
            candidates.append(pt)

    for src in candidates:
        if src.is_file():
            shutil.copy2(src, dst)
            log.info("Model saved: %s (%.1f MB)", dst, dst.stat().st_size / 1e6)
            return dst

    # Fallback to last.pt
    for src in candidates:
        last = src.parent / "last.pt"
        if last.is_file():
            shutil.copy2(last, dst)
            log.info("last.pt saved to %s (best.pt not found)", dst)
            return dst

    log.warning("No weights found in any expected location")
    return None


def quick_test(n_samples=5):
    """Run inference on a few test images to verify the model."""
    from ultralytics import YOLO

    model_path = MODEL_DIR / "yolo_crop_disease.pt"
    if not model_path.is_file():
        log.warning("Skipping test — model not found at %s", model_path)
        return

    model = YOLO(str(model_path))

    test_dir = DATASET_DIR / "test" / "images"
    if not test_dir.exists():
        for vname in ("val", "valid"):
            test_dir = DATASET_DIR / vname / "images"
            if test_dir.exists():
                break
    if not test_dir.exists():
        log.warning("No test images found")
        return

    import random
    images = list(test_dir.iterdir())
    samples = random.sample(images, min(n_samples, len(images)))

    log.info("Quick test on %d images:", len(samples))
    detected = 0
    for img_path in samples:
        results = model(str(img_path), verbose=False)
        if results and len(results[0].boxes) > 0:
            detected += 1
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                name = results[0].names[cls_id]
                log.info("  %-40s  %-25s  conf=%.1f%%",
                         img_path.name, name, conf * 100)
        else:
            log.info("  %-40s  No detections", img_path.name)
    log.info("Detection rate: %d/%d images", detected, len(samples))


# ═══════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 crop disease detector")
    parser.add_argument("--model", default="yolov8n.pt",
                        help="Base model (yolov8n/s/m/l/x.pt)")
    parser.add_argument("--epochs", type=int, default=80,
                        help="Training epochs")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--device", default="cpu",
                        help="Device (cpu/cuda/0)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Image size")
    parser.add_argument("--resume", default=None,
                        help="Resume from checkpoint (path to last.pt)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Dataloader workers (auto-detected if omitted)")
    parser.add_argument("--cache", default=None,
                        choices=["ram", "disk", "none"],
                        help="Image caching strategy (auto-detected if omitted)")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("YOLO CROP DISEASE DETECTOR — TRAINING")
    log.info("=" * 60)

    if not check_prerequisites():
        sys.exit(1)

    cache_val = None if args.cache == "none" else args.cache
    train(args.model, args.epochs, args.batch, args.device, args.imgsz,
          resume=args.resume, workers=args.workers, cache=cache_val)

    save_model()
    quick_test()

    log.info("=" * 60)
    log.info("ALL DONE")
    log.info("  Model: models/yolo_crop_disease.pt")
    log.info("  Evaluate: python scripts/evaluate_model.py")
    log.info("  Compare:  python scripts/compare_models.py")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
