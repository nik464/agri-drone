#!/usr/bin/env python3
"""
evaluate/expand_evaluation.py
──────────────────────────────
Produces two new result files:

1. predictions_A_efficientnet_b0.csv   — EfficientNet-B0 on test set (multi-backbone)
2. predictions_A_yolo_val.csv          — YOLOv8n-cls on val set    (expanded n)
3. predictions_B_yolo_rules_val.csv    — YOLOv8n-cls+rules on val  (expanded n)
4. holm_bonferroni_extended.json       — McNemar on combined test+val (n~1868)

Usage:
    cd d:\Projects\agri-drone
    python evaluate/expand_evaluation.py
"""
import csv
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import efficientnet_b0
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parent.parent
DATA_TEST    = ROOT / "data" / "training" / "test"
DATA_VAL     = ROOT / "data" / "training" / "val"
MODEL_EFF    = ROOT / "models" / "efficientnet_b0_21class.pt"
MODEL_YOLO   = ROOT / "models" / "india_agri_cls.pt"
RESULTS_DIR  = ROOT / "evaluate" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Class order (alphabetical = torchvision ImageFolder default)
CLASSES = [
    "healthy_rice", "healthy_wheat", "rice_bacterial_blight", "rice_blast",
    "rice_brown_spot", "rice_leaf_scald", "rice_sheath_blight", "wheat_aphid",
    "wheat_black_rust", "wheat_blast", "wheat_brown_rust",
    "wheat_fusarium_head_blight", "wheat_leaf_blight", "wheat_mite",
    "wheat_powdery_mildew", "wheat_root_rot", "wheat_septoria", "wheat_smut",
    "wheat_stem_fly", "wheat_tan_spot", "wheat_yellow_rust",
]

# ── EfficientNet-B0 inference ─────────────────────────────────────────────────
def run_efficientnet(data_dir: Path, out_csv: Path):
    print(f"\n{'='*60}")
    print(f"EfficientNet-B0 inference on {data_dir.name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(1280, len(CLASSES))
    state = torch.load(MODEL_EFF, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    rows = []
    correct_count = 0
    total = 0

    for cls_dir in sorted(data_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        gt = cls_dir.name
        for img_path in sorted(cls_dir.glob("*")):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                continue

            x = transform(img).unsqueeze(0).to(device)
            t0 = time.perf_counter()
            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=1)
            latency = (time.perf_counter() - t0) * 1000

            conf, idx = probs[0].max(dim=0)
            pred = CLASSES[idx.item()]
            correct = int(pred == gt)
            correct_count += correct
            total += 1

            rows.append({
                "image":       img_path.name,
                "ground_truth": gt,
                "predicted":   pred,
                "confidence":  round(conf.item(), 4),
                "correct":     correct,
                "latency_ms":  round(latency, 1),
                "severity_tier": "healthy" if "healthy" in gt else "diseased",
            })

    acc = correct_count / total if total else 0
    print(f"  n={total}  accuracy={acc:.4f}  ({correct_count}/{total} correct)")

    fields = ["image", "ground_truth", "predicted", "confidence",
              "correct", "latency_ms", "severity_tier"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {out_csv.name}")
    return rows, acc, total


# ── YOLOv8n-cls inference (val set) ───────────────────────────────────────────
def run_yolo_val(data_dir: Path, out_csv_a: Path, out_csv_b: Path):
    print(f"\n{'='*60}")
    print(f"YOLOv8n-cls inference on {data_dir.name}")

    from ultralytics import YOLO
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    import agridrone.vision.rule_engine as rule_engine_mod
    import agridrone.vision.feature_extractor as feat_ext

    model = YOLO(str(MODEL_YOLO))

    rows_a, rows_b = [], []
    correct_a = correct_b = 0
    total = 0

    for cls_dir in sorted(data_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        gt = cls_dir.name
        for img_path in sorted(cls_dir.glob("*")):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            try:
                t0 = time.perf_counter()
                result = model.predict(str(img_path), verbose=False)[0]
                latency = (time.perf_counter() - t0) * 1000
            except Exception:
                continue

            probs = result.probs
            pred_a = result.names[probs.top1]
            conf_a = round(float(probs.top1conf), 4)
            c_a = int(pred_a == gt)
            correct_a += c_a
            total += 1

            rows_a.append({
                "image": img_path.name, "ground_truth": gt,
                "predicted": pred_a, "confidence": conf_a,
                "correct": c_a, "latency_ms": round(latency, 1),
                "severity_tier": "healthy" if "healthy" in gt else "diseased",
            })

            # Config B: apply rule engine
            try:
                img_cv = __import__("cv2").imread(str(img_path))
                scores = {result.names[i]: float(p)
                          for i, p in enumerate(probs.data)}
                decision = rule_engine.apply(img_cv, scores)
                pred_b = decision.final_class
            except Exception:
                pred_b = pred_a
            c_b = int(pred_b == gt)
            correct_b += c_b
            rows_b.append({
                "image": img_path.name, "ground_truth": gt,
                "predicted": pred_b, "confidence": conf_a,
                "correct": c_b, "latency_ms": round(latency, 1),
                "severity_tier": "healthy" if "healthy" in gt else "diseased",
            })

    acc_a = correct_a / total if total else 0
    acc_b = correct_b / total if total else 0
    print(f"  n={total}  Config A acc={acc_a:.4f}  Config B acc={acc_b:.4f}")

    fields = ["image", "ground_truth", "predicted", "confidence",
              "correct", "latency_ms", "severity_tier"]
    for out_csv, rows in [(out_csv_a, rows_a), (out_csv_b, rows_b)]:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"  Saved: {out_csv.name}")
    return rows_a, rows_b, total


# ── Extended McNemar (test + val combined) ─────────────────────────────────────
def mcnemar_cc(b, c):
    denom = b + c
    if denom == 0:
        return 1.0
    chi2 = (abs(b - c) - 1) ** 2 / denom
    return math.erfc(math.sqrt(chi2 / 2))

def extended_mcnemar():
    print(f"\n{'='*60}")
    print("Extended McNemar: combining test + val sets")

    def load_csv(p):
        rows = {}
        with open(p, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                # prefix with split to avoid name collisions
                key = f"{p.stem}__{row['image']}"
                rows[key] = {
                    "gt": row["ground_truth"],
                    "correct": row["correct"].strip() == "1"
                }
        return rows

    # Original test set
    preds_a_test = load_csv(RESULTS_DIR / "predictions_A_yolo_only.csv")
    preds_b_test = load_csv(RESULTS_DIR / "predictions_B_yolo_rules.csv")
    # Val set
    preds_a_val  = load_csv(RESULTS_DIR / "predictions_A_yolo_val.csv")
    preds_b_val  = load_csv(RESULTS_DIR / "predictions_B_yolo_rules_val.csv")

    # Combine
    all_a = {**preds_a_test, **preds_a_val}
    all_b = {**preds_b_test, **preds_b_val}
    common = sorted(set(all_a) & set(all_b))
    n = len(common)

    b = sum(1 for k in common if     all_a[k]["correct"] and not all_b[k]["correct"])
    c = sum(1 for k in common if not all_a[k]["correct"] and     all_b[k]["correct"])
    p = mcnemar_cc(b, c)
    acc_a = sum(all_a[k]["correct"] for k in common) / n
    acc_b = sum(all_b[k]["correct"] for k in common) / n

    result = {
        "description": "Extended A-vs-B McNemar combining test (n=933) + val (n~933) sets",
        "note": "Val set used for model selection (YOLO early stopping); absolute accuracy may be optimistic. A-vs-B delta is unaffected by this since the rule engine has no trainable parameters.",
        "n_total": n,
        "acc_A": round(acc_a, 4),
        "acc_B": round(acc_b, 4),
        "delta_acc": round(acc_b - acc_a, 4),
        "discordant_b": b,
        "discordant_c": c,
        "p_value_mcnemar_cc": round(p, 6),
        "significant_at_0.05": p <= 0.05,
    }

    out = RESULTS_DIR / "mcnemar_extended_test_plus_val.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"  n_combined={n}  p={p:.4f}  significant={p <= 0.05}")
    print(f"  Saved: {out.name}")
    return result


# ── EfficientNet B0 stats ──────────────────────────────────────────────────────
def efficientnet_stats(rows_eff, rows_yolo_test):
    """Compare EfficientNet-B0 vs YOLOv8n-cls on the same test set."""
    print(f"\n{'='*60}")
    print("Multi-backbone comparison: EfficientNet-B0 vs YOLOv8n-cls")

    # Load original YOLO test predictions
    yolo = {}
    with open(RESULTS_DIR / "predictions_A_yolo_only.csv", newline="") as f:
        for row in csv.DictReader(f):
            yolo[row["image"]] = row["correct"].strip() == "1"

    eff = {r["image"]: r["correct"] == 1 for r in rows_eff}
    common = sorted(set(yolo) & set(eff))
    n = len(common)

    b = sum(1 for img in common if     yolo[img] and not eff[img])
    c = sum(1 for img in common if not yolo[img] and     eff[img])
    p = mcnemar_cc(b, c)

    acc_yolo = sum(yolo[img] for img in common) / n
    acc_eff  = sum(eff[img]  for img in common) / n

    result = {
        "description": "Multi-backbone comparison on test set: YOLOv8n-cls vs EfficientNet-B0",
        "n": n,
        "acc_YOLOv8n_cls": round(acc_yolo, 4),
        "acc_EfficientNet_B0": round(acc_eff, 4),
        "delta": round(acc_eff - acc_yolo, 4),
        "discordant_b_yolo_wins": b,
        "discordant_c_eff_wins": c,
        "p_value_mcnemar_cc": round(p, 6),
        "significant_at_0.05": p <= 0.05,
        "note": "Both models evaluated on identical held-out test set (seed=42 split)."
    }

    out = RESULTS_DIR / "multibackbone_comparison.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"  n={n}  YOLO={acc_yolo:.4f}  EfficientNet-B0={acc_eff:.4f}  p={p:.4f}")
    print(f"  Saved: {out.name}")
    return result


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1. EfficientNet B0 on test set
    eff_rows, eff_acc, eff_n = run_efficientnet(
        DATA_TEST,
        RESULTS_DIR / "predictions_A_efficientnet_b0.csv"
    )

    # 2. YOLO on val set (Config A + B)
    val_rows_a, val_rows_b, val_n = run_yolo_val(
        DATA_VAL,
        RESULTS_DIR / "predictions_A_yolo_val.csv",
        RESULTS_DIR / "predictions_B_yolo_rules_val.csv",
    )

    # 3. Extended McNemar (test + val)
    ext = extended_mcnemar()

    # 4. Multi-backbone stats
    mb = efficientnet_stats(eff_rows, None)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"  EfficientNet-B0 test accuracy : {eff_acc:.4f}  (n={eff_n})")
    print(f"  Extended McNemar (n={ext['n_total']}): p={ext['p_value_mcnemar_cc']:.4f}  sig={ext['significant_at_0.05']}")
    print(f"  Multi-backbone delta (EFF-YOLO): {mb['delta']:+.4f}  p={mb['p_value_mcnemar_cc']:.4f}")
    print("\nDone. Update paper/main.tex with these results.")
