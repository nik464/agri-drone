#!/usr/bin/env python3
"""
Train EfficientNet-B0 on the 21-class Indian crop disease dataset.

Mirrors YOLOv8n-cls training setup:
  - ImageNet-pretrained backbone, fine-tuned 50 epochs
  - 224×224 input, AdamW optimizer
  - Early stopping (patience 10)
  - Same train/val/test split as YOLO

Then evaluates on test set and compares with YOLOv8 results.

Usage:
    python evaluate/train_efficientnet.py
    python evaluate/train_efficientnet.py --epochs 50 --batch-size 32
"""

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ════════════════════════════════════════════════════════════════
# Data transforms (match YOLO augmentation)
# ════════════════════════════════════════════════════════════════

def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


# ════════════════════════════════════════════════════════════════
# Model
# ════════════════════════════════════════════════════════════════

def build_model(num_classes: int):
    """EfficientNet-B0 with ImageNet-pretrained backbone."""
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model = models.efficientnet_b0(weights=weights)
    # Replace classifier head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


# ════════════════════════════════════════════════════════════════
# Training loop
# ════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def train(model, train_loader, val_loader, device, epochs=50, lr=0.001, patience=10,
          save_path=None):
    """Train with early stopping."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    best_epoch = 0
    epochs_no_improve = 0

    model.to(device)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"  Epoch {epoch:>3d}/{epochs}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
              f"lr={optimizer.param_groups[0]['lr']:.6f}  "
              f"({elapsed:.1f}s)")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            epochs_no_improve = 0
            if save_path:
                torch.save(model.state_dict(), save_path)
                print(f"    ✓ Best model saved ({val_acc:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n  Early stopping at epoch {epoch} (best={best_epoch}, val_acc={best_val_acc:.4f})")
                break

    # Load best weights
    if save_path and Path(save_path).exists():
        model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))

    return best_epoch, best_val_acc


# ════════════════════════════════════════════════════════════════
# Evaluation
# ════════════════════════════════════════════════════════════════

def evaluate_test(model, test_loader, class_names, device):
    """Evaluate and return per-image predictions + per-class metrics."""
    model.eval()
    all_preds = []
    all_labels = []
    all_latencies = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            t0 = time.perf_counter()
            outputs = model(images)
            latency = (time.perf_counter() - t0) * 1000 / images.size(0)  # per-image

            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(labels.tolist())
            all_latencies.extend([latency] * images.size(0))

    # Overall accuracy
    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    accuracy = correct / len(all_labels)

    # Per-class metrics
    per_class = {}
    for idx, name in enumerate(class_names):
        tp = sum(1 for p, l in zip(all_preds, all_labels) if p == idx and l == idx)
        fp = sum(1 for p, l in zip(all_preds, all_labels) if p == idx and l != idx)
        fn = sum(1 for p, l in zip(all_preds, all_labels) if p != idx and l == idx)
        support = sum(1 for l in all_labels if l == idx)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class[name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }

    macro_f1 = np.mean([v["f1"] for v in per_class.values()])
    mean_latency = np.mean(all_latencies)

    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(float(macro_f1), 4),
        "mean_latency_ms": round(float(mean_latency), 1),
        "per_class": per_class,
        "predictions": all_preds,
        "labels": all_labels,
    }


# ════════════════════════════════════════════════════════════════
# Config B: pipe through rule engine + ensemble
# ════════════════════════════════════════════════════════════════

SEVERITY_TIERS = {
    "wheat_fusarium_head_blight": ("critical", 10),
    "wheat_yellow_rust": ("critical", 10),
    "wheat_black_rust": ("critical", 10),
    "wheat_blast": ("critical", 10),
    "rice_blast": ("critical", 10),
    "rice_bacterial_blight": ("critical", 10),
    "wheat_brown_rust": ("high", 5),
    "wheat_septoria": ("high", 5),
    "wheat_leaf_blight": ("high", 5),
    "rice_sheath_blight": ("high", 5),
    "wheat_root_rot": ("high", 5),
    "rice_leaf_scald": ("high", 5),
    "wheat_powdery_mildew": ("moderate", 2),
    "wheat_tan_spot": ("moderate", 2),
    "wheat_aphid": ("moderate", 2),
    "wheat_mite": ("moderate", 2),
    "wheat_smut": ("moderate", 2),
    "wheat_stem_fly": ("moderate", 2),
    "rice_brown_spot": ("moderate", 2),
    "healthy_wheat": ("healthy", 1),
    "healthy_rice": ("healthy", 1),
}


def run_config_b_efficientnet(effnet_model, image_bgr, class_names, device, crop_type):
    """Run EfficientNet prediction through rule engine + ensemble voter."""
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from agridrone.vision.disease_reasoning import run_full_pipeline, diagnosis_to_dict
    from agridrone.vision.ensemble_voter import ensemble_vote

    # Step 1: EfficientNet inference
    val_tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    t0 = time.perf_counter()
    rgb = image_bgr[:, :, ::-1].copy()  # BGR→RGB
    tensor = val_tf(rgb).unsqueeze(0).to(device)

    effnet_model.eval()
    with torch.no_grad():
        outputs = effnet_model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    top5_vals, top5_idxs = probs.topk(5)
    top1_idx = top5_idxs[0].item()
    top1_conf = top5_vals[0].item()
    top_key = class_names[top1_idx]
    top_is_healthy = "healthy" in top_key.lower()
    top_severity = SEVERITY_TIERS.get(top_key, ("moderate", 2))
    health_score = 95 if top_is_healthy else max(5, round(100 - top_severity[1] * 10 * top1_conf))

    classifier_result = {
        "top_prediction": top_key,
        "top_confidence": round(top1_conf, 4),
        "confidence": round(top1_conf, 4),
        "health_score": health_score,
        "is_healthy": top_is_healthy,
        "disease_probability": round(1 - top1_conf if top_is_healthy else top1_conf, 4),
        "top5": [
            {
                "index": top5_idxs[i].item(),
                "class_key": class_names[top5_idxs[i].item()],
                "class_name": class_names[top5_idxs[i].item()].replace("_", " ").title(),
                "confidence": round(top5_vals[i].item(), 4),
            }
            for i in range(5)
        ],
    }

    # Step 2+3: Rule engine + ensemble
    try:
        import cv2
        output = run_full_pipeline(image_bgr, classifier_result, crop_type)
        reasoning_result = diagnosis_to_dict(output.diagnosis)

        ensemble_result = ensemble_vote(
            classifier_result=classifier_result,
            reasoning_result=reasoning_result,
            llm_validation=None,
            crop_type=crop_type,
        )

        latency = (time.perf_counter() - t0) * 1000
        return {
            "predicted": ensemble_result.get("final_disease", top_key),
            "confidence": round(ensemble_result.get("combined_confidence", top1_conf), 4),
            "latency_ms": round(latency, 1),
        }
    except Exception as e:
        latency = (time.perf_counter() - t0) * 1000
        return {
            "predicted": top_key,
            "confidence": round(top1_conf, 4),
            "latency_ms": round(latency, 1),
            "error": str(e),
        }


# ════════════════════════════════════════════════════════════════
# Comparison output
# ════════════════════════════════════════════════════════════════

def generate_comparison_csv(effnet_results, yolo_results_path, output_path, class_names):
    """Generate baseline_comparison.csv comparing EfficientNet vs YOLO."""

    # Load YOLO results
    yolo_summary = json.loads(yolo_results_path.read_text())
    yolo_acc = yolo_summary["config_A_accuracy"]
    yolo_f1 = yolo_summary["config_A_macro_f1"]

    # Load per-class YOLO data from ablation_table.csv
    yolo_per_class = {}
    ablation_csv = yolo_results_path.parent / "ablation_table.csv"
    if ablation_csv.exists():
        with open(ablation_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                cls = row.get("class", "")
                if cls and cls != "MACRO" and cls != "WEIGHTED":
                    yolo_per_class[cls] = {
                        "f1_A": float(row.get("A_f1", 0)),
                        "recall_A": float(row.get("A_recall", 0)),
                    }

    rows = []
    for cls in sorted(class_names):
        eff = effnet_results["per_class"].get(cls, {})
        yolo = yolo_per_class.get(cls, {})
        rows.append({
            "class": cls,
            "efficientnet_recall": eff.get("recall", 0),
            "efficientnet_f1": eff.get("f1", 0),
            "yolo_recall": yolo.get("recall_A", 0),
            "yolo_f1": yolo.get("f1_A", 0),
            "recall_delta": round(eff.get("recall", 0) - yolo.get("recall_A", 0), 4),
            "f1_delta": round(eff.get("f1", 0) - yolo.get("f1_A", 0), 4),
        })

    # Add macro row
    eff_macro_recall = np.mean([v["recall"] for v in effnet_results["per_class"].values()])
    yolo_macro_recall = np.mean([v.get("recall_A", 0) for v in yolo_per_class.values()]) if yolo_per_class else 0
    rows.append({
        "class": "MACRO_AVG",
        "efficientnet_recall": round(float(eff_macro_recall), 4),
        "efficientnet_f1": effnet_results["macro_f1"],
        "yolo_recall": round(float(yolo_macro_recall), 4),
        "yolo_f1": yolo_f1,
        "recall_delta": round(float(eff_macro_recall - yolo_macro_recall), 4),
        "f1_delta": round(effnet_results["macro_f1"] - yolo_f1, 4),
    })

    # Add accuracy row
    rows.append({
        "class": "ACCURACY",
        "efficientnet_recall": effnet_results["accuracy"],
        "efficientnet_f1": "",
        "yolo_recall": yolo_acc,
        "yolo_f1": "",
        "recall_delta": round(effnet_results["accuracy"] - yolo_acc, 4),
        "f1_delta": "",
    })

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "class", "efficientnet_recall", "efficientnet_f1",
            "yolo_recall", "yolo_f1", "recall_delta", "f1_delta",
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Comparison CSV saved: {output_path}")


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train & evaluate EfficientNet-B0 baseline")
    parser.add_argument("--train-dir", type=str, default="data/training/train")
    parser.add_argument("--val-dir", type=str, default="data/training/val")
    parser.add_argument("--test-dir", type=str, default="data/training/test")
    parser.add_argument("--output-dir", type=str, default="evaluate/results")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training, load saved model for eval only")
    parser.add_argument("--skip-config-b", action="store_true",
                        help="Skip Config B (rule engine) evaluation")
    parser.add_argument("--model-save-path", type=str,
                        default="models/efficientnet_b0_21class.pt")
    args = parser.parse_args()

    train_dir = Path(args.train_dir)
    val_dir = Path(args.val_dir)
    test_dir = Path(args.test_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_save = Path(args.model_save_path)
    model_save.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data loading ──
    train_tf, val_tf = get_transforms()

    train_dataset = datasets.ImageFolder(str(train_dir), transform=train_tf)
    val_dataset = datasets.ImageFolder(str(val_dir), transform=val_tf)
    test_dataset = datasets.ImageFolder(str(test_dir), transform=val_tf)

    class_names = train_dataset.classes
    num_classes = len(class_names)
    print(f"Classes: {num_classes} — {class_names}")
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=False)

    # ── Build model ──
    model = build_model(num_classes)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"EfficientNet-B0: {param_count:,} trainable parameters")

    # ── Train or load ──
    if args.skip_training:
        if not model_save.exists():
            print(f"ERROR: --skip-training but model not found: {model_save}")
            sys.exit(1)
        model.load_state_dict(torch.load(str(model_save), map_location=device, weights_only=True))
        print(f"Loaded saved model: {model_save}")
    else:
        print(f"\n{'='*70}")
        print(f"  TRAINING EfficientNet-B0 ({num_classes} classes, {args.epochs} epochs max)")
        print(f"{'='*70}\n")
        t_start = time.time()
        best_epoch, best_val_acc = train(
            model, train_loader, val_loader, device,
            epochs=args.epochs, lr=args.lr, patience=args.patience,
            save_path=str(model_save),
        )
        train_time = time.time() - t_start
        print(f"\n  Training complete: {train_time:.0f}s, best_epoch={best_epoch}, "
              f"best_val_acc={best_val_acc:.4f}")
        print(f"  Model saved: {model_save} ({model_save.stat().st_size / 1024 / 1024:.1f} MB)")

    model.to(device)

    # ── Evaluate on test set (Config A: EfficientNet-only) ──
    print(f"\n{'='*70}")
    print(f"  EVALUATING: Config A (EfficientNet-only) on test set")
    print(f"{'='*70}\n")

    effnet_results = evaluate_test(model, test_loader, class_names, device)

    print(f"  Accuracy:       {effnet_results['accuracy']*100:.2f}%")
    print(f"  Macro-F1:       {effnet_results['macro_f1']:.4f}")
    print(f"  Mean latency:   {effnet_results['mean_latency_ms']:.1f} ms")
    print(f"\n  Per-class metrics:")
    print(f"  {'Class':<35} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>8}")
    print(f"  {'-'*75}")
    for cls in sorted(class_names):
        m = effnet_results["per_class"][cls]
        print(f"  {cls:<35} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f} {m['support']:>8}")

    # ── Config B: EfficientNet + Rules + Ensemble (optional) ──
    effnet_b_results = None
    if not args.skip_config_b:
        print(f"\n{'='*70}")
        print(f"  EVALUATING: Config B (EfficientNet + Rules + Ensemble) on test set")
        print(f"{'='*70}\n")

        import cv2

        test_images = []
        for cls in sorted(test_dir.iterdir()):
            if not cls.is_dir():
                continue
            for img_path in sorted(cls.glob("*")):
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    test_images.append({
                        "path": str(img_path),
                        "ground_truth": cls.name,
                        "crop_type": "rice" if cls.name.startswith("rice") or cls.name == "healthy_rice" else "wheat",
                    })

        b_correct = 0
        b_total = 0
        b_per_class_tp = defaultdict(int)
        b_per_class_fp = defaultdict(int)
        b_per_class_fn = defaultdict(int)
        b_per_class_support = defaultdict(int)
        b_latencies = []

        for i, item in enumerate(test_images):
            img_bgr = cv2.imread(item["path"])
            if img_bgr is None:
                continue

            # Resize large images for speed
            h, w = img_bgr.shape[:2]
            if max(h, w) > 640:
                scale = 640 / max(h, w)
                img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))

            result = run_config_b_efficientnet(
                model, img_bgr, class_names, device, item["crop_type"]
            )

            gt = item["ground_truth"]
            pred = result["predicted"]
            b_per_class_support[gt] += 1
            b_latencies.append(result["latency_ms"])
            b_total += 1

            if pred == gt:
                b_correct += 1
                b_per_class_tp[gt] += 1
            else:
                b_per_class_fn[gt] += 1
                b_per_class_fp[pred] += 1

            if (i + 1) % 50 == 0 or i == 0:
                print(f"  [{i+1:>4d}/{len(test_images)}] acc={b_correct/b_total:.4f}  "
                      f"last: gt={gt}  pred={pred}  {'✓' if pred == gt else '✗'}")

        # Compute per-class metrics for Config B
        b_per_class = {}
        for cls in class_names:
            tp = b_per_class_tp[cls]
            fp = b_per_class_fp[cls]
            fn = b_per_class_fn[cls]
            support = b_per_class_support[cls]
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            b_per_class[cls] = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "support": support,
            }

        b_accuracy = b_correct / b_total if b_total > 0 else 0
        b_macro_f1 = float(np.mean([v["f1"] for v in b_per_class.values()]))

        effnet_b_results = {
            "accuracy": round(b_accuracy, 4),
            "macro_f1": round(b_macro_f1, 4),
            "mean_latency_ms": round(float(np.mean(b_latencies)), 1),
            "per_class": b_per_class,
        }

        print(f"\n  Config B Accuracy:  {b_accuracy*100:.2f}%")
        print(f"  Config B Macro-F1:  {b_macro_f1:.4f}")
        print(f"  Config B Latency:   {np.mean(b_latencies):.1f} ms")

    # ── Generate comparison CSV ──
    print(f"\n{'='*70}")
    print(f"  GENERATING COMPARISON")
    print(f"{'='*70}\n")

    yolo_summary_path = output_dir / "ablation_summary.json"
    csv_path = output_dir / "baseline_comparison.csv"

    if yolo_summary_path.exists():
        generate_comparison_csv(effnet_results, yolo_summary_path, csv_path, class_names)
    else:
        print(f"  WARNING: YOLO results not found at {yolo_summary_path}")
        print(f"  Writing EfficientNet-only results")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "class", "efficientnet_recall", "efficientnet_f1",
                "yolo_recall", "yolo_f1", "recall_delta", "f1_delta",
            ])
            writer.writeheader()
            for cls in sorted(class_names):
                m = effnet_results["per_class"][cls]
                writer.writerow({
                    "class": cls, "efficientnet_recall": m["recall"],
                    "efficientnet_f1": m["f1"], "yolo_recall": "", "yolo_f1": "",
                    "recall_delta": "", "f1_delta": "",
                })
        print(f"  CSV saved: {csv_path}")

    # ── Save full results JSON ──
    full_results = {
        "model": "EfficientNet-B0",
        "parameters": param_count,
        "device": str(device),
        "config_a_efficientnet": {
            "accuracy": effnet_results["accuracy"],
            "macro_f1": effnet_results["macro_f1"],
            "mean_latency_ms": effnet_results["mean_latency_ms"],
            "per_class": effnet_results["per_class"],
        },
    }
    if effnet_b_results:
        full_results["config_b_efficientnet_rules"] = {
            "accuracy": effnet_b_results["accuracy"],
            "macro_f1": effnet_b_results["macro_f1"],
            "mean_latency_ms": effnet_b_results["mean_latency_ms"],
            "per_class": effnet_b_results["per_class"],
        }
        full_results["effnet_ablation_delta"] = {
            "accuracy_delta": round(effnet_b_results["accuracy"] - effnet_results["accuracy"], 4),
            "macro_f1_delta": round(effnet_b_results["macro_f1"] - effnet_results["macro_f1"], 4),
        }

    # Compare with YOLO
    if yolo_summary_path.exists():
        yolo_summary = json.loads(yolo_summary_path.read_text())
        full_results["yolo_comparison"] = {
            "yolo_accuracy": yolo_summary["config_A_accuracy"],
            "yolo_macro_f1": yolo_summary["config_A_macro_f1"],
            "effnet_accuracy": effnet_results["accuracy"],
            "effnet_macro_f1": effnet_results["macro_f1"],
            "accuracy_delta": round(effnet_results["accuracy"] - yolo_summary["config_A_accuracy"], 4),
            "f1_delta": round(effnet_results["macro_f1"] - yolo_summary["config_A_macro_f1"], 4),
        }
        if effnet_b_results:
            full_results["yolo_comparison"]["yolo_config_b_accuracy"] = yolo_summary["config_B_accuracy"]
            full_results["yolo_comparison"]["effnet_config_b_accuracy"] = effnet_b_results["accuracy"]

    results_json_path = output_dir / "efficientnet_results.json"
    with open(results_json_path, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"  Full results JSON: {results_json_path}")

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"  BASELINE COMPARISON SUMMARY")
    print(f"{'='*70}")

    print(f"\n  {'Metric':<25} {'EfficientNet-B0':>15} {'YOLOv8n-cls':>15} {'Delta':>10}")
    print(f"  {'-'*65}")

    if yolo_summary_path.exists():
        yolo_summary = json.loads(yolo_summary_path.read_text())
        ya, yf = yolo_summary["config_A_accuracy"], yolo_summary["config_A_macro_f1"]
        ea, ef = effnet_results["accuracy"], effnet_results["macro_f1"]
        print(f"  {'Accuracy (A: model only)':<25} {ea*100:>14.2f}% {ya*100:>14.2f}% {(ea-ya)*100:>+9.2f}pp")
        print(f"  {'Macro-F1 (A: model only)':<25} {ef:>15.4f} {yf:>15.4f} {ef-yf:>+10.4f}")

        if effnet_b_results:
            yba = yolo_summary["config_B_accuracy"]
            ybf = yolo_summary["config_B_macro_f1"]
            eba = effnet_b_results["accuracy"]
            ebf = effnet_b_results["macro_f1"]
            print(f"  {'Accuracy (B: +rules)':<25} {eba*100:>14.2f}% {yba*100:>14.2f}% {(eba-yba)*100:>+9.2f}pp")
            print(f"  {'Macro-F1 (B: +rules)':<25} {ebf:>15.4f} {ybf:>15.4f} {ebf-ybf:>+10.4f}")
            print(f"\n  ABLATION DELTAS (Config B minus Config A):")
            print(f"  {'EfficientNet Δ accuracy:':<30} {(eba-ea)*100:>+.2f}pp")
            print(f"  {'YOLOv8       Δ accuracy:':<30} {(yba-ya)*100:>+.2f}pp")
            print(f"  {'EfficientNet Δ macro-F1:':<30} {ebf-ef:>+.4f}")
            print(f"  {'YOLOv8       Δ macro-F1:':<30} {ybf-yf:>+.4f}")

    print(f"\n  Parameters: EfficientNet-B0={param_count:,} vs YOLOv8n-cls=1,443,412")
    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
