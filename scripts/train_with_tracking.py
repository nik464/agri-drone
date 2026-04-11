#!/usr/bin/env python3
"""
train_with_tracking.py — YOLO training with MLflow experiment tracking.

Tracks all hyperparameters, metrics per epoch, and artifacts (model weights,
plots, evaluation results) in MLflow for reproducible experiments.

Usage:
    python scripts/train_with_tracking.py
    python scripts/train_with_tracking.py --experiment "rice-disease-v2" --epochs 100
    python scripts/train_with_tracking.py --model yolov8s.pt --batch 16

View results:
    mlflow ui --port 5001
    # Then open http://localhost:5001
"""
import argparse
import shutil
import sys
import json
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DEFAULT_DATA = str(PROJECT_ROOT / "data" / "raw" / "roboflow" / "rice-diseases-v2" / "data.yaml")
DEFAULT_MODEL = "yolov8n.pt"
MLFLOW_DIR = PROJECT_ROOT / "outputs" / "mlruns"
MODEL_DIR = PROJECT_ROOT / "models"


def setup_mlflow(experiment_name: str, tracking_uri: str):
    """Initialize MLflow tracking."""
    import mlflow
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    return mlflow


def train_with_tracking(args):
    """Train YOLO with full MLflow tracking."""
    import mlflow
    import yaml
    from ultralytics import YOLO

    tracking_uri = f"file:///{MLFLOW_DIR.as_posix()}"
    mlflow_client = setup_mlflow(args.experiment, tracking_uri)

    # Load dataset info
    with open(args.data) as f:
        data_cfg = yaml.safe_load(f)

    with mlflow.start_run(run_name=f"{args.model.replace('.pt', '')}_{args.epochs}ep_{datetime.now():%Y%m%d_%H%M}") as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")
        print(f"MLflow UI: mlflow ui --backend-store-uri {tracking_uri}")

        # ── Log hyperparameters ──
        mlflow.log_params({
            "model_arch": args.model,
            "epochs": args.epochs,
            "batch_size": args.batch,
            "imgsz": args.imgsz,
            "device": args.device,
            "patience": args.patience,
            "optimizer": "auto",
            "lr0": args.lr0,
            "lrf": args.lrf,
            "weight_decay": args.weight_decay,
            "warmup_epochs": 3,
            "mosaic": 0.8,
            "mixup": 0.1,
            "flipud": 0.5,
            "fliplr": 0.5,
            "degrees": 15.0,
            "hsv_h": 0.015,
            "hsv_s": 0.4,
            "hsv_v": 0.3,
        })

        # ── Log dataset info ──
        mlflow.log_params({
            "dataset_name": data_cfg.get("roboflow", {}).get("project", "custom"),
            "num_classes": data_cfg.get("nc", 0),
            "class_names": str(data_cfg.get("names", [])),
        })

        # Count images
        ds_root = Path(data_cfg.get("path", Path(args.data).parent))
        for split_name in ["train", "valid", "val", "test"]:
            split_dir = ds_root / split_name / "images"
            if split_dir.exists():
                n = len(list(split_dir.iterdir()))
                mlflow.log_param(f"n_{split_name}", n)

        # ── Train ──
        print(f"\nTraining {args.model} for {args.epochs} epochs...")
        model = YOLO(args.model)
        output_dir = PROJECT_ROOT / "outputs" / "training"

        results = model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=str(output_dir),
            name="yolo_crop_disease",
            patience=args.patience,
            workers=2,
            exist_ok=True,
            lr0=args.lr0,
            lrf=args.lrf,
            weight_decay=args.weight_decay,
            warmup_epochs=3,
            flipud=0.5,
            fliplr=0.5,
            degrees=15.0,
            hsv_h=0.015,
            hsv_s=0.4,
            hsv_v=0.3,
            mosaic=0.8,
            mixup=0.1,
            verbose=True,
        )

        # ── Log training metrics ──
        train_results_csv = output_dir / "yolo_crop_disease" / "results.csv"
        if train_results_csv.exists():
            import csv as csv_mod
            with open(train_results_csv) as f:
                reader = csv_mod.DictReader(f)
                for i, row in enumerate(reader):
                    step = i + 1
                    for key, val in row.items():
                        key = key.strip()
                        try:
                            mlflow.log_metric(key.replace("/", "_").replace("(", "").replace(")", ""), float(val.strip()), step=step)
                        except (ValueError, TypeError):
                            pass
            mlflow.log_artifact(str(train_results_csv), "training")

        # ── Log final box metrics ──
        val_results = model.val(data=args.data, split="val", device=args.device, verbose=False)
        box = val_results.box
        mlflow.log_metrics({
            "val_mAP50": float(box.map50),
            "val_mAP50_95": float(box.map),
            "val_precision": float(box.mp),
            "val_recall": float(box.mr),
        })

        # Test if test split exists
        test_dir = ds_root / "test" / "images"
        if test_dir.exists() and any(test_dir.iterdir()):
            test_results = model.val(data=args.data, split="test", device=args.device, verbose=False)
            tbox = test_results.box
            mlflow.log_metrics({
                "test_mAP50": float(tbox.map50),
                "test_mAP50_95": float(tbox.map),
                "test_precision": float(tbox.mp),
                "test_recall": float(tbox.mr),
            })

        # ── Log artifacts ──
        # Model weights
        best_pt = output_dir / "yolo_crop_disease" / "weights" / "best.pt"
        if best_pt.exists():
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            dst = MODEL_DIR / "yolo_crop_disease.pt"
            shutil.copy2(best_pt, dst)
            mlflow.log_artifact(str(dst), "model")
            print(f"\nModel saved: {dst}")

        # Training plots
        plot_dir = output_dir / "yolo_crop_disease"
        for ext in ["*.png", "*.jpg"]:
            for plot_file in plot_dir.glob(ext):
                mlflow.log_artifact(str(plot_file), "plots")

        # Log class-wise metrics as a table
        per_class_data = []
        class_names = data_cfg.get("names", [])
        ap50 = box.ap50 if hasattr(box, 'ap50') else box.maps
        for i, name in enumerate(class_names):
            entry = {"class": name}
            if i < len(box.p):
                entry["precision"] = round(float(box.p[i]), 4)
            if i < len(box.r):
                entry["recall"] = round(float(box.r[i]), 4)
            if i < len(ap50):
                entry["ap50"] = round(float(ap50[i]), 4)
            per_class_data.append(entry)

        per_class_path = output_dir / "yolo_crop_disease" / "per_class_metrics.json"
        with open(per_class_path, "w") as f:
            json.dump(per_class_data, f, indent=2)
        mlflow.log_artifact(str(per_class_path), "metrics")

        # ── Summary ──
        print("\n" + "=" * 65)
        print("  TRAINING COMPLETE — MLflow Tracked")
        print("=" * 65)
        print(f"  Run ID:         {run_id}")
        print(f"  Val mAP@0.5:    {float(box.map50):.4f}")
        print(f"  Val mAP@0.5:95: {float(box.map):.4f}")
        print(f"  Val Precision:  {float(box.mp):.4f}")
        print(f"  Val Recall:     {float(box.mr):.4f}")
        print(f"\n  View results:   mlflow ui --backend-store-uri {tracking_uri}")
        print("=" * 65)

    return run_id


def main():
    parser = argparse.ArgumentParser(description="Train YOLO with MLflow tracking")
    parser.add_argument("--experiment", default="agri-drone-detection",
                        help="MLflow experiment name")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Base YOLO model")
    parser.add_argument("--data", default=DEFAULT_DATA, help="data.yaml path")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", default="cpu", help="Device")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--lrf", type=float, default=0.01, help="Final learning rate factor")
    parser.add_argument("--weight-decay", type=float, default=0.0005, help="Weight decay")
    args = parser.parse_args()

    # Check mlflow
    try:
        import mlflow
        print(f"MLflow version: {mlflow.__version__}")
    except ImportError:
        print("MLflow not installed. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "mlflow", "--quiet"])
        import mlflow

    train_with_tracking(args)


if __name__ == "__main__":
    main()
