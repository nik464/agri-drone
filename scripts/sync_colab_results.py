#!/usr/bin/env python3
"""
sync_colab_results.py — Import Colab GPU training artifacts into the local project.

After training on Google Colab, download the artifacts (from Google Drive
or the zip) and run this script to place everything where the backend
and ML Dashboard expect it.

Usage:
    # From a folder of downloaded files:
    python scripts/sync_colab_results.py  path/to/downloaded_artifacts

    # From the auto-generated zip:
    python scripts/sync_colab_results.py  path/to/agridrone_training_results.zip

    # Auto-reload the running backend after sync:
    python scripts/sync_colab_results.py  path/to/artifacts  --reload

What it does:
    1. Copies best.pt → models/yolo_crop_disease.pt
    2. Copies results.csv, confusion_matrix.png, F1_curve.png, PR_curve.png,
       results.png, args.yaml → outputs/training/yolo_crop_disease/
    3. Copies evaluation_report.json, training_curves.png,
       evaluation_plots.png → outputs/evaluation/
    4. Generates outputs/logs/training.log from results.csv
       (so the Training Logs page has data to show)
    5. Optionally hits POST /api/model/reload to hot-swap the model
"""
import argparse
import csv
import json
import logging
import os
import shutil
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sync")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Destination mapping ──────────────────────────────────────────
DEST_MAP = {
    # Model weights
    "best.pt":                        "models/yolo_crop_disease.pt",
    "yolo_crop_disease.pt":           "models/yolo_crop_disease.pt",
    # Training artifacts → outputs/training/yolo_crop_disease/
    "results.csv":                    "outputs/training/yolo_crop_disease/results.csv",
    "results.png":                    "outputs/training/yolo_crop_disease/results.png",
    "confusion_matrix.png":           "outputs/training/yolo_crop_disease/confusion_matrix.png",
    "confusion_matrix_normalized.png":"outputs/training/yolo_crop_disease/confusion_matrix_normalized.png",
    "F1_curve.png":                   "outputs/training/yolo_crop_disease/F1_curve.png",
    "PR_curve.png":                   "outputs/training/yolo_crop_disease/PR_curve.png",
    "P_curve.png":                    "outputs/training/yolo_crop_disease/P_curve.png",
    "R_curve.png":                    "outputs/training/yolo_crop_disease/R_curve.png",
    "labels.jpg":                     "outputs/training/yolo_crop_disease/labels.jpg",
    "labels_correlogram.jpg":         "outputs/training/yolo_crop_disease/labels_correlogram.jpg",
    "args.yaml":                      "outputs/training/yolo_crop_disease/args.yaml",
    "last.pt":                        "outputs/training/yolo_crop_disease/weights/last.pt",
    # Evaluation artifacts → outputs/evaluation/
    "evaluation_report.json":         "outputs/evaluation/evaluation_report.json",
    "training_curves.png":            "outputs/evaluation/training_curves.png",
    "evaluation_plots.png":           "outputs/evaluation/evaluation_plots.png",
    "sample_predictions.png":         "outputs/evaluation/sample_predictions.png",
    # Training status (from Colab pipeline)
    "training_status.json":           "outputs/training/training_status.json",
}


def copy_artifacts(src_dir: Path) -> dict:
    """Copy known artifacts from src_dir to project locations."""
    copied = {}
    skipped = []

    for filename, rel_dst in DEST_MAP.items():
        src = src_dir / filename
        if not src.is_file():
            continue

        dst = PROJECT_ROOT / rel_dst
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        size_mb = dst.stat().st_size / (1024 * 1024)
        copied[filename] = str(dst)
        log.info("  %-40s → %s  (%.1f MB)", filename, rel_dst, size_mb)

    # Report files that weren't in our map
    for f in src_dir.iterdir():
        if f.is_file() and f.name not in DEST_MAP and not f.name.startswith("."):
            skipped.append(f.name)

    if skipped:
        log.info("  Skipped unknown files: %s", ", ".join(skipped))

    return copied


def generate_training_log(results_csv: Path, log_path: Path):
    """
    Generate a training.log from results.csv so the Training Logs page
    has structured data to display.
    """
    if not results_csv.is_file():
        log.warning("No results.csv found — skipping training.log generation")
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    ts = datetime.now().strftime("%Y-%m-%d")

    lines.append(f"{'='*60}")
    lines.append(f"  YOLO CROP DISEASE DETECTOR — GPU TRAINING LOG")
    lines.append(f"  Synced from Google Colab on {ts}")
    lines.append(f"{'='*60}")
    lines.append("")

    with open(results_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    total = len(rows)
    lines.append(f"[INFO] Training: {total} epochs completed")
    lines.append("")

    for row in rows:
        r = {k.strip(): v.strip() for k, v in row.items()}
        epoch = r.get("epoch", "?")

        box = r.get("train/box_loss", "?")
        cls = r.get("train/cls_loss", "?")
        dfl = r.get("train/dfl_loss", "?")
        lines.append(
            f"[INFO] EPOCH {epoch}/{total}  box={box}  cls={cls}  dfl={dfl}"
        )

        map50 = r.get("metrics/mAP50(B)", "?")
        map95 = r.get("metrics/mAP50-95(B)", "?")
        prec = r.get("metrics/precision(B)", "?")
        rec = r.get("metrics/recall(B)", "?")
        lines.append(
            f"[INFO]   VAL  mAP50={map50}  mAP50-95={map95}  P={prec}  R={rec}"
        )

    # Final summary
    if rows:
        last = {k.strip(): v.strip() for k, v in rows[-1].items()}
        lines.append("")
        lines.append(f"{'='*60}")
        lines.append(f"[INFO] TRAINING COMPLETE — {total} epochs")
        lines.append(
            f"[INFO]   Final  mAP50={last.get('metrics/mAP50(B)', '?')}  "
            f"mAP50-95={last.get('metrics/mAP50-95(B)', '?')}"
        )
        lines.append(f"{'='*60}")

    log_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("  Generated training.log (%d lines)", len(lines))


def generate_training_status(results_csv: Path, copied: dict):
    """Generate training_status.json for the Colab Pipeline UI page."""
    status_path = PROJECT_ROOT / "outputs" / "training" / "training_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)

    status = {
        "stage": "complete",
        "message": "Training synced from Colab successfully",
        "progress": 100,
        "model": "YOLOv8s",
        "dataset": "Rice Diseases v2 (Roboflow)",
        "updated_at": datetime.now().isoformat(),
        "files_synced": len(copied),
    }

    if results_csv.is_file():
        try:
            with open(results_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            if rows:
                last = {k.strip(): v.strip() for k, v in rows[-1].items()}
                status["total_epochs"] = len(rows)
                status["epoch"] = len(rows)
                status["mAP50"] = last.get("metrics/mAP50(B)", "?")
                status["mAP50_95"] = last.get("metrics/mAP50-95(B)", "?")
                status["loss"] = last.get("train/box_loss", "?")
        except Exception as e:
            log.warning("  Failed to parse results.csv for status: %s", e)

    # Check for evaluation report
    eval_report = PROJECT_ROOT / "outputs" / "evaluation" / "evaluation_report.json"
    if eval_report.is_file():
        try:
            data = json.loads(eval_report.read_text(encoding="utf-8"))
            status["num_classes"] = data.get("num_classes", "?")
            status["num_images"] = data.get("num_images", "?")
        except Exception:
            pass

    has_model = any("best.pt" in k or "yolo_crop_disease.pt" in k for k in copied)
    status["model_synced"] = has_model

    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    log.info("  Generated training_status.json")


def reload_backend(host="127.0.0.1", port=9000):
    """Tell the running backend to hot-reload models."""
    import urllib.request
    import urllib.error

    url = f"http://{host}:{port}/api/model/reload"
    try:
        req = urllib.request.Request(url, method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            log.info("  Backend model reloaded: %s", resp.read().decode())
    except urllib.error.URLError:
        log.warning("  Backend not running at %s — reload skipped", url)
    except Exception as e:
        log.warning("  Reload failed: %s", e)


def main():
    parser = argparse.ArgumentParser(
        description="Sync Colab training artifacts into the local AgriDrone project"
    )
    parser.add_argument(
        "source",
        help="Path to downloaded artifacts folder or .zip file",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Hit /api/model/reload on the running backend after sync",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9000,
        help="Backend port for --reload (default: 9000)",
    )
    args = parser.parse_args()

    source = Path(args.source).resolve()
    if not source.exists():
        log.error("Source not found: %s", source)
        sys.exit(1)

    log.info("=" * 60)
    log.info("  SYNC COLAB RESULTS → LOCAL PROJECT")
    log.info("=" * 60)
    log.info("Source:  %s", source)
    log.info("Project: %s", PROJECT_ROOT)
    log.info("")

    # Handle zip files
    tmp_dir = None
    if source.is_file() and source.suffix == ".zip":
        tmp_dir = tempfile.mkdtemp(prefix="agridrone_sync_")
        log.info("Extracting zip to temp dir...")
        with zipfile.ZipFile(source, "r") as zf:
            zf.extractall(tmp_dir)
        # Find the actual directory with files (might be nested)
        items = list(Path(tmp_dir).iterdir())
        if len(items) == 1 and items[0].is_dir():
            source = items[0]
        else:
            source = Path(tmp_dir)

    if not source.is_dir():
        log.error("Source is not a directory: %s", source)
        sys.exit(1)

    # Copy artifacts
    copied = copy_artifacts(source)

    if not copied:
        log.error("No known artifacts found in %s", source)
        log.error("Expected files: best.pt, results.csv, confusion_matrix.png, etc.")
        sys.exit(1)

    # Generate training.log from results.csv
    results_csv = PROJECT_ROOT / "outputs" / "training" / "yolo_crop_disease" / "results.csv"
    training_log = PROJECT_ROOT / "outputs" / "logs" / "training.log"
    generate_training_log(results_csv, training_log)

    # Generate training_status.json for the Colab Pipeline UI page
    generate_training_status(results_csv, copied)

    # Reload backend if requested
    if args.reload:
        log.info("")
        log.info("Reloading backend model...")
        reload_backend(port=args.port)

    # Summary
    log.info("")
    log.info("=" * 60)
    log.info("  SYNC COMPLETE — %d files copied", len(copied))
    log.info("=" * 60)

    if "best.pt" in copied or "yolo_crop_disease.pt" in copied:
        log.info("")
        log.info("  Model ready at: models/yolo_crop_disease.pt")
        if not args.reload:
            log.info("  Run with --reload to hot-swap the running backend")

    log.info("")
    log.info("  Dashboard pages now have data:")
    if results_csv.is_file():
        log.info("    ✓ ML Dashboard  — metrics & charts from results.csv")
    if training_log.is_file():
        log.info("    ✓ Training Logs — generated from training history")
    eval_report = PROJECT_ROOT / "outputs" / "evaluation" / "evaluation_report.json"
    if eval_report.is_file():
        log.info("    ✓ Evaluation    — test set report available")

    # Clean up temp dir
    if tmp_dir:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
