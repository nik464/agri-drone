#!/usr/bin/env python3
"""
Experiment 1 — Three-Config Ablation Study

Runs every test image through:
  Config A: YOLO classifier only (raw top-1 prediction + confidence)
  Config B: YOLO + rule engine (fused score, conflict resolution active)
  Config C: Rule engine only (no YOLO input — visual features → rules → diagnosis)
  Config D: Full ensemble (B + LLaVA) — on a stratified 200-image subset
            [STUB: requires Ollama + llava:7b; skipped if unavailable]

Outputs to evaluate/results/:
  ablation_table.csv        — all metrics, all configs, all classes
  ablation_latex.tex        — publication-ready LaTeX table
  confusion_matrix_A.png    — normalized confusion matrix, 300 DPI
  confusion_matrix_B.png
  ablation_summary.json     — headline numbers

Usage:
    python evaluate/ablation_study.py
    python evaluate/ablation_study.py --test-dir data/training/test --output-dir evaluate/results
    python evaluate/ablation_study.py --with-llava   # enable Config C (slow)
"""

import argparse
import csv
import json
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ════════════════════════════════════════════════════════════════
# Severity tier definitions (from paper Section 1.3)
# ════════════════════════════════════════════════════════════════

SEVERITY_TIERS = {
    # Critical: severity >= 0.8, yield loss >= 40%
    "wheat_fusarium_head_blight": ("critical", 10),
    "wheat_yellow_rust":         ("critical", 10),
    "wheat_black_rust":          ("critical", 10),
    "wheat_blast":               ("critical", 10),
    "rice_blast":                ("critical", 10),
    "rice_bacterial_blight":     ("critical", 10),
    # High: severity >= 0.7
    "wheat_brown_rust":          ("high", 5),
    "wheat_septoria":            ("high", 5),
    "wheat_leaf_blight":         ("high", 5),
    "rice_sheath_blight":        ("high", 5),
    "wheat_root_rot":            ("high", 5),
    "rice_leaf_scald":           ("high", 5),
    # Moderate: severity >= 0.5
    "wheat_powdery_mildew":      ("moderate", 2),
    "wheat_tan_spot":            ("moderate", 2),
    "wheat_aphid":               ("moderate", 2),
    "wheat_mite":                ("moderate", 2),
    "wheat_smut":                ("moderate", 2),
    "wheat_stem_fly":            ("moderate", 2),
    "rice_brown_spot":           ("moderate", 2),
    # Healthy
    "healthy_wheat":             ("healthy", 1),
    "healthy_rice":              ("healthy", 1),
}


def get_tier_weight(class_key: str) -> int:
    return SEVERITY_TIERS.get(class_key, ("moderate", 2))[1]


def get_tier_name(class_key: str) -> str:
    return SEVERITY_TIERS.get(class_key, ("moderate",))[0]


# ════════════════════════════════════════════════════════════════
# Data loading
# ════════════════════════════════════════════════════════════════

def load_test_set(test_dir: Path) -> list[dict]:
    """Load all test images with ground truth labels (from folder names)."""
    images = []
    classes = sorted([d.name for d in test_dir.iterdir() if d.is_dir()])
    for cls in classes:
        cls_dir = test_dir / cls
        for img_path in sorted(cls_dir.glob("*")):
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}:
                images.append({
                    "path": str(img_path),
                    "ground_truth": cls,
                    "crop_type": "rice" if cls.startswith("rice") or cls == "healthy_rice" else "wheat",
                })
    return images


# ════════════════════════════════════════════════════════════════
# Config A: YOLO-only
# ════════════════════════════════════════════════════════════════

def run_config_a(model, image_bgr) -> dict:
    """Raw YOLO classifier — top-1 prediction."""
    t0 = time.perf_counter()
    results = model(image_bgr, verbose=False)
    latency = (time.perf_counter() - t0) * 1000

    if not results or results[0].probs is None:
        return {"predicted": "unknown", "confidence": 0.0, "latency_ms": latency, "top5": []}

    probs = results[0].probs
    names = model.names
    top5_indices = probs.top5
    top5_confs = probs.top5conf.tolist()

    top5 = []
    for idx, conf in zip(top5_indices, top5_confs):
        top5.append({
            "index": idx,
            "class_key": names[idx],
            "class_name": names[idx].replace("_", " ").title(),
            "confidence": round(conf, 4),
        })

    return {
        "predicted": names[probs.top1],
        "confidence": round(probs.top1conf.item(), 4),
        "latency_ms": round(latency, 1),
        "top5": top5,
    }


# ════════════════════════════════════════════════════════════════
# Config B: YOLO + Rule Engine
# ════════════════════════════════════════════════════════════════

def run_config_b(model, image_bgr, crop_type: str) -> dict:
    """YOLO classifier → rule engine → ensemble voter (2-model Bayesian fusion).

    Matches the actual system architecture: both YOLO and rule engine cast
    votes into the ensemble voter, which uses Bayesian posterior combination
    with reliability weights (YOLO=0.65, Rules=0.75).
    """
    from agridrone.vision.disease_reasoning import run_full_pipeline, diagnosis_to_dict
    from agridrone.vision.ensemble_voter import ensemble_vote

    # Step 1: YOLO
    t0 = time.perf_counter()
    results = model(image_bgr, verbose=False)
    if not results or results[0].probs is None:
        latency = (time.perf_counter() - t0) * 1000
        return {"predicted": "unknown", "confidence": 0.0, "latency_ms": latency, "top5": []}

    probs = results[0].probs
    names = model.names
    top5_indices = probs.top5
    top5_confs = probs.top5conf.tolist()

    top_key = names[probs.top1]
    top_conf = round(probs.top1conf.item(), 4)
    top_is_healthy = "healthy" in top_key.lower()
    top_severity = SEVERITY_TIERS.get(top_key, ("moderate", 2))
    health_score_yolo = 95 if top_is_healthy else max(5, round(100 - top_severity[1] * 10 * top_conf))

    # classifier_result — uses class_key for top_prediction so ensemble keys align
    classifier_result = {
        "top_prediction": top_key,
        "top_confidence": top_conf,
        "confidence": top_conf,
        "health_score": health_score_yolo,
        "is_healthy": top_is_healthy,
        "disease_probability": round(1 - top_conf if top_is_healthy else top_conf, 4),
        "top5": [
            {
                "index": idx,
                "class_key": names[idx],
                "class_name": names[idx].replace("_", " ").title(),
                "confidence": round(conf, 4),
            }
            for idx, conf in zip(top5_indices, top5_confs)
        ],
    }

    # Step 2+3: Feature extraction + Rule engine
    try:
        output = run_full_pipeline(image_bgr, classifier_result, crop_type)
        reasoning_result = diagnosis_to_dict(output.diagnosis)

        # Step 4: Ensemble voter — 2-model Bayesian combination (no LLM)
        ensemble_result = ensemble_vote(
            classifier_result=classifier_result,
            reasoning_result=reasoning_result,
            llm_validation=None,
            crop_type=crop_type,
        )

        latency = (time.perf_counter() - t0) * 1000

        return {
            "predicted": ensemble_result.final_disease,
            "confidence": round(ensemble_result.final_confidence, 4),
            "latency_ms": round(latency, 1),
            "top5": classifier_result["top5"],
            "rule_result": output.rule_result,
            "features": output.features,
            "pipeline_output": output,
            "ensemble_result": ensemble_result,
            "agreement_level": ensemble_result.agreement_level,
        }
    except Exception as e:
        latency = (time.perf_counter() - t0) * 1000
        warnings.warn(f"Config B pipeline error: {e}")
        return {
            "predicted": top_key,
            "confidence": top_conf,
            "latency_ms": round(latency, 1),
            "top5": classifier_result["top5"],
        }


# ════════════════════════════════════════════════════════════════
# Config C: Rule Engine Only (no YOLO input)
# ════════════════════════════════════════════════════════════════

def run_config_c(image_bgr, crop_type: str) -> dict:
    """Rule engine only — visual feature extraction → rules → diagnosis.

    No classifier input: the rule engine must infer disease purely from
    colour histograms, texture metrics, and hand-crafted symptom rules.
    This is the lower-bound baseline for the ablation.
    """
    from agridrone.vision.disease_reasoning import run_full_pipeline, diagnosis_to_dict

    # Empty classifier result — forces rule engine to rely solely on features
    dummy_classifier = {
        "top_prediction": "unknown",
        "top_confidence": 0.0,
        "confidence": 0.0,
        "health_score": 50,
        "is_healthy": False,
        "disease_probability": 0.0,
        "top5": [],
    }

    t0 = time.perf_counter()
    try:
        output = run_full_pipeline(image_bgr, dummy_classifier, crop_type)
        reasoning_result = diagnosis_to_dict(output.diagnosis)
        latency = (time.perf_counter() - t0) * 1000

        predicted = output.diagnosis.disease_key or "unknown"
        confidence = output.diagnosis.confidence or 0.0

        return {
            "predicted": predicted,
            "confidence": round(confidence, 4),
            "latency_ms": round(latency, 1),
            "rule_result": output.rule_result,
            "features": output.features,
        }
    except Exception as e:
        latency = (time.perf_counter() - t0) * 1000
        warnings.warn(f"Config C pipeline error: {e}")
        return {
            "predicted": "unknown",
            "confidence": 0.0,
            "latency_ms": round(latency, 1),
        }


# ════════════════════════════════════════════════════════════════
# Metrics
# ════════════════════════════════════════════════════════════════

def compute_metrics(predictions: list[dict], classes: list[str]) -> dict:
    """Compute per-class and aggregate metrics."""
    # Per-class TP, FP, FN
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    total_correct = 0
    total = len(predictions)

    # For RWA
    rwa_correct_weighted = 0.0
    rwa_total_weighted = 0.0

    # Confusion matrix
    class_to_idx = {c: i for i, c in enumerate(classes)}
    n_classes = len(classes)
    conf_matrix = np.zeros((n_classes, n_classes), dtype=int)

    for pred in predictions:
        gt = pred["ground_truth"]
        pr = pred["predicted"]

        w = get_tier_weight(gt)
        rwa_total_weighted += w

        if gt == pr:
            total_correct += 1
            tp[gt] += 1
            rwa_correct_weighted += w
        else:
            fn[gt] += 1
            fp[pr] += 1

        gi = class_to_idx.get(gt, -1)
        pi = class_to_idx.get(pr, -1)
        if gi >= 0 and pi >= 0:
            conf_matrix[gi][pi] += 1

    accuracy = total_correct / total if total > 0 else 0
    rwa = rwa_correct_weighted / rwa_total_weighted if rwa_total_weighted > 0 else 0
    safety_gap = accuracy - rwa

    # Per-class precision, recall, F1
    per_class = {}
    macro_p, macro_r, macro_f1 = 0, 0, 0
    for cls in classes:
        p = tp[cls] / (tp[cls] + fp[cls]) if (tp[cls] + fp[cls]) > 0 else 0
        r = tp[cls] / (tp[cls] + fn[cls]) if (tp[cls] + fn[cls]) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        support = tp[cls] + fn[cls]
        per_class[cls] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "support": support,
            "tier": get_tier_name(cls),
            "tier_weight": get_tier_weight(cls),
        }
        macro_p += p
        macro_r += r
        macro_f1 += f1

    n_cls = len(classes)
    macro_p /= n_cls
    macro_r /= n_cls
    macro_f1 /= n_cls

    # Matthews Correlation Coefficient (multi-class)
    cm = np.array(conf_matrix)
    c = cm.sum()
    s = cm.sum(axis=1)   # row sums (true class totals)
    t = cm.sum(axis=0)   # col sums (predicted class totals)
    pk = np.diag(cm)     # correct predictions per class
    cov_xy = c * pk.sum() - np.dot(s, t)
    cov_xx = c * c - np.dot(s, s)
    cov_yy = c * c - np.dot(t, t)
    denom = np.sqrt(float(cov_xx) * float(cov_yy))
    mcc = float(cov_xy) / denom if denom > 0 else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "rwa": round(rwa, 4),
        "safety_gap": round(safety_gap, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "macro_f1": round(macro_f1, 4),
        "mcc": round(mcc, 4),
        "per_class": per_class,
        "confusion_matrix": conf_matrix.tolist(),
        "n_samples": total,
    }


# ════════════════════════════════════════════════════════════════
# Confusion matrix plot
# ════════════════════════════════════════════════════════════════

def plot_confusion_matrix(conf_matrix, classes, title, output_path):
    """Normalized confusion matrix heatmap at 300 DPI."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  WARNING: matplotlib not installed — skipping confusion matrix plot")
        return

    cm = np.array(conf_matrix, dtype=float)
    # Normalize per row (per true class)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm / row_sums

    # Short labels
    short_labels = [c.replace("wheat_", "w.").replace("rice_", "r.").replace("healthy_", "h.")
                    for c in classes]

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(xticks=np.arange(len(classes)),
           yticks=np.arange(len(classes)),
           xticklabels=short_labels,
           yticklabels=short_labels)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)

    # Annotate cells
    for i in range(len(classes)):
        for j in range(len(classes)):
            val = cm_norm[i, j]
            if val > 0.005:
                color = "white" if val > 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6, color=color)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════
# LaTeX table generation
# ════════════════════════════════════════════════════════════════

def generate_latex_table(metrics_a, metrics_b, classes, output_path):
    """Publication-ready LaTeX ablation table."""
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Three-configuration ablation study on the 21-class AgriDrone test set "
        r"($n=" + str(metrics_a["n_samples"]) + r"$). "
        r"RWA = Risk-Weighted Accuracy with tier weights "
        r"$\tau_{\text{crit}}=10, \tau_{\text{high}}=5, \tau_{\text{mod}}=2, \tau_{\text{healthy}}=1$. "
        r"Best per-class F1 in \textbf{bold}.}",
        r"\label{tab:ablation}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l c | r r r | r r r | r}",
        r"\toprule",
        r"& & \multicolumn{3}{c|}{\textbf{Config A (YOLO)}} "
        r"& \multicolumn{3}{c|}{\textbf{Config B (YOLO+Rules)}} "
        r"& \\",
        r"\textbf{Disease} & \textbf{Tier} & P & R & F1 & P & R & F1 & $\Delta$F1 \\",
        r"\midrule",
    ]

    # Per-class rows
    for cls in classes:
        a = metrics_a["per_class"][cls]
        b = metrics_b["per_class"][cls]
        delta = b["f1"] - a["f1"]
        delta_str = f"{'+' if delta >= 0 else ''}{delta:.2f}"

        # Bold the higher F1
        f1_a_str = f"\\textbf{{{a['f1']:.2f}}}" if a["f1"] > b["f1"] else f"{a['f1']:.2f}"
        f1_b_str = f"\\textbf{{{b['f1']:.2f}}}" if b["f1"] >= a["f1"] else f"{b['f1']:.2f}"

        tier = a["tier"][:4]
        display = cls.replace("_", r"\_")
        lines.append(
            f"  {display} & {tier} & {a['precision']:.2f} & {a['recall']:.2f} & {f1_a_str} "
            f"& {b['precision']:.2f} & {b['recall']:.2f} & {f1_b_str} & {delta_str} \\\\"
        )

    lines += [
        r"\midrule",
        f"  \\textbf{{Macro}} & & {metrics_a['macro_precision']:.2f} & "
        f"{metrics_a['macro_recall']:.2f} & {metrics_a['macro_f1']:.2f} & "
        f"{metrics_b['macro_precision']:.2f} & {metrics_b['macro_recall']:.2f} & "
        f"{metrics_b['macro_f1']:.2f} & "
        f"{'+' if metrics_b['macro_f1'] >= metrics_a['macro_f1'] else ''}"
        f"{metrics_b['macro_f1'] - metrics_a['macro_f1']:.2f} \\\\",
        r"\midrule",
        f"  \\textbf{{Accuracy}} & & \\multicolumn{{3}}{{c|}}{{{metrics_a['accuracy'] * 100:.1f}\\%}} "
        f"& \\multicolumn{{3}}{{c|}}{{{metrics_b['accuracy'] * 100:.1f}\\%}} "
        f"& {'+' if metrics_b['accuracy'] >= metrics_a['accuracy'] else ''}"
        f"{(metrics_b['accuracy'] - metrics_a['accuracy']) * 100:.1f}pp \\\\",
        f"  \\textbf{{RWA}} & & \\multicolumn{{3}}{{c|}}{{{metrics_a['rwa'] * 100:.1f}\\%}} "
        f"& \\multicolumn{{3}}{{c|}}{{{metrics_b['rwa'] * 100:.1f}\\%}} "
        f"& {'+' if metrics_b['rwa'] >= metrics_a['rwa'] else ''}"
        f"{(metrics_b['rwa'] - metrics_a['rwa']) * 100:.1f}pp \\\\",
        f"  \\textbf{{Safety Gap}} & & \\multicolumn{{3}}{{c|}}{{{metrics_a['safety_gap'] * 100:.1f}pp}} "
        f"& \\multicolumn{{3}}{{c|}}{{{metrics_b['safety_gap'] * 100:.1f}pp}} & \\\\",
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table*}",
    ]
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")


# ════════════════════════════════════════════════════════════════
# CSV output
# ════════════════════════════════════════════════════════════════

def write_csv(metrics_a, metrics_b, classes, output_path):
    """Write full metrics table as CSV."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "tier", "tier_weight", "support",
                     "A_precision", "A_recall", "A_f1",
                     "B_precision", "B_recall", "B_f1",
                     "delta_f1"])
        for cls in classes:
            a = metrics_a["per_class"][cls]
            b = metrics_b["per_class"][cls]
            w.writerow([cls, a["tier"], a["tier_weight"], a["support"],
                        a["precision"], a["recall"], a["f1"],
                        b["precision"], b["recall"], b["f1"],
                        round(b["f1"] - a["f1"], 4)])
        # Aggregate row
        w.writerow(["MACRO", "", "", metrics_a["n_samples"],
                     metrics_a["macro_precision"], metrics_a["macro_recall"], metrics_a["macro_f1"],
                     metrics_b["macro_precision"], metrics_b["macro_recall"], metrics_b["macro_f1"],
                     round(metrics_b["macro_f1"] - metrics_a["macro_f1"], 4)])


# ════════════════════════════════════════════════════════════════
# Per-image predictions CSV (for downstream experiments)
# ════════════════════════════════════════════════════════════════

def write_predictions_csv(predictions, config_name, output_path):
    """Save per-image predictions for Experiments 2-4."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "ground_truth", "predicted", "confidence",
                     "correct", "latency_ms", "severity_tier"])
        for p in predictions:
            w.writerow([
                Path(p["path"]).name,
                p["ground_truth"],
                p["predicted"],
                p["confidence"],
                1 if p["ground_truth"] == p["predicted"] else 0,
                p["latency_ms"],
                get_tier_name(p["ground_truth"]),
            ])


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Experiment 1: Three-config ablation")
    parser.add_argument("--test-dir", default=str(PROJECT_ROOT / "data" / "training" / "test"))
    parser.add_argument("--model-path", default=str(PROJECT_ROOT / "models" / "india_agri_cls.pt"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "evaluate" / "results"))
    parser.add_argument("--with-llava", action="store_true", help="Enable Config C (needs Ollama)")
    parser.add_argument("--subset-size", type=int, default=200, help="Stratified subset size for Config C")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    test_dir = Path(args.test_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ──
    print("Loading YOLO classifier...")
    from ultralytics import YOLO
    model = YOLO(args.model_path, task="classify")
    classes = sorted(model.names.values())
    print(f"  Model: {args.model_path}")
    print(f"  Classes: {len(classes)}")

    # ── Load test set ──
    test_images = load_test_set(test_dir)
    print(f"  Test images: {len(test_images)}")
    if not test_images:
        print("ERROR: No test images found. Run data split first.")
        sys.exit(1)

    # Distribution check
    dist = defaultdict(int)
    for img in test_images:
        dist[img["ground_truth"]] += 1
    print("  Class distribution:")
    for cls in classes:
        print(f"    {cls:40s} {dist[cls]:4d}")

    # ══════════════════════════════════════════════════════
    # Config A: YOLO-only
    # ══════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  CONFIG A: YOLO-only")
    print(f"{'='*70}")

    preds_a = []
    for i, img_info in enumerate(test_images):
        image_bgr = cv2.imread(img_info["path"])
        if image_bgr is None:
            warnings.warn(f"Cannot read {img_info['path']}")
            continue
        result = run_config_a(model, image_bgr)
        preds_a.append({
            "path": img_info["path"],
            "ground_truth": img_info["ground_truth"],
            "predicted": result["predicted"],
            "confidence": result["confidence"],
            "latency_ms": result["latency_ms"],
        })
        if (i + 1) % 100 == 0:
            acc_so_far = sum(1 for p in preds_a if p["ground_truth"] == p["predicted"]) / len(preds_a)
            print(f"  [{i+1}/{len(test_images)}] running acc: {acc_so_far:.1%}")

    metrics_a = compute_metrics(preds_a, classes)
    mean_lat_a = np.mean([p["latency_ms"] for p in preds_a])
    print(f"  Config A — Accuracy: {metrics_a['accuracy']:.1%}, "
          f"RWA: {metrics_a['rwa']:.1%}, "
          f"Macro-F1: {metrics_a['macro_f1']:.3f}, "
          f"Mean latency: {mean_lat_a:.0f}ms")

    # Save predictions CSV
    write_predictions_csv(preds_a, "A", output_dir / "predictions_A_yolo_only.csv")

    # ══════════════════════════════════════════════════════
    # Config B: YOLO + Rules
    # ══════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  CONFIG B: YOLO + Rule Engine")
    print(f"{'='*70}")

    preds_b = []
    for i, img_info in enumerate(test_images):
        image_bgr = cv2.imread(img_info["path"])
        if image_bgr is None:
            continue
        result = run_config_b(model, image_bgr, img_info["crop_type"])
        preds_b.append({
            "path": img_info["path"],
            "ground_truth": img_info["ground_truth"],
            "predicted": result["predicted"],
            "confidence": result["confidence"],
            "latency_ms": result["latency_ms"],
        })
        if (i + 1) % 100 == 0:
            acc_so_far = sum(1 for p in preds_b if p["ground_truth"] == p["predicted"]) / len(preds_b)
            print(f"  [{i+1}/{len(test_images)}] running acc: {acc_so_far:.1%}")

    metrics_b = compute_metrics(preds_b, classes)
    mean_lat_b = np.mean([p["latency_ms"] for p in preds_b])
    print(f"  Config B — Accuracy: {metrics_b['accuracy']:.1%}, "
          f"RWA: {metrics_b['rwa']:.1%}, "
          f"Macro-F1: {metrics_b['macro_f1']:.3f}, "
          f"Mean latency: {mean_lat_b:.0f}ms")

    # Save predictions CSV
    write_predictions_csv(preds_b, "B", output_dir / "predictions_B_yolo_rules.csv")

    # ══════════════════════════════════════════════════════
    # Config C: Rule Engine Only
    # ══════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  CONFIG C: Rule Engine Only (no YOLO)")
    print(f"{'='*70}")

    preds_c = []
    for i, img_info in enumerate(test_images):
        image_bgr = cv2.imread(img_info["path"])
        if image_bgr is None:
            continue
        result = run_config_c(image_bgr, img_info["crop_type"])
        preds_c.append({
            "path": img_info["path"],
            "ground_truth": img_info["ground_truth"],
            "predicted": result["predicted"],
            "confidence": result["confidence"],
            "latency_ms": result["latency_ms"],
        })
        if (i + 1) % 100 == 0:
            acc_so_far = sum(1 for p in preds_c if p["ground_truth"] == p["predicted"]) / len(preds_c)
            print(f"  [{i+1}/{len(test_images)}] running acc: {acc_so_far:.1%}")

    metrics_c = compute_metrics(preds_c, classes)
    mean_lat_c = np.mean([p["latency_ms"] for p in preds_c])
    print(f"  Config C — Accuracy: {metrics_c['accuracy']:.1%}, "
          f"RWA: {metrics_c['rwa']:.1%}, "
          f"Macro-F1: {metrics_c['macro_f1']:.3f}, "
          f"MCC: {metrics_c['mcc']:.3f}, "
          f"Mean latency: {mean_lat_c:.0f}ms")

    write_predictions_csv(preds_c, "C", output_dir / "predictions_C_rules_only.csv")

    # ══════════════════════════════════════════════════════
    # Config D: YOLO + Rules + LLaVA (stub)
    # ══════════════════════════════════════════════════════
    metrics_d = None
    if args.with_llava:
        print(f"\n{'='*70}")
        print("  CONFIG D: Full Ensemble (YOLO + Rules + LLaVA)")
        print(f"  Running on {args.subset_size}-image stratified subset")
        print(f"{'='*70}")
        print("  [NOT IMPLEMENTED — needs --with-llava and Ollama running]")
    else:
        print(f"\n  Config D skipped (no --with-llava flag). "
              f"Re-run with --with-llava when Ollama is available.")

    # ══════════════════════════════════════════════════════
    # Output
    # ══════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  GENERATING OUTPUTS")
    print(f"{'='*70}")

    # Confusion matrices
    plot_confusion_matrix(metrics_a["confusion_matrix"], classes,
                          f"Config A: YOLO-Only (Acc={metrics_a['accuracy']:.1%})",
                          output_dir / "confusion_matrix_A.png")
    print(f"  confusion_matrix_A.png")

    plot_confusion_matrix(metrics_b["confusion_matrix"], classes,
                          f"Config B: YOLO+Rules (Acc={metrics_b['accuracy']:.1%})",
                          output_dir / "confusion_matrix_B.png")
    print(f"  confusion_matrix_B.png")

    plot_confusion_matrix(metrics_c["confusion_matrix"], classes,
                          f"Config C: Rules-Only (Acc={metrics_c['accuracy']:.1%})",
                          output_dir / "confusion_matrix_C.png")
    print(f"  confusion_matrix_C.png")

    # CSV
    write_csv(metrics_a, metrics_b, classes, output_dir / "ablation_table.csv")
    print(f"  ablation_table.csv")

    # LaTeX
    generate_latex_table(metrics_a, metrics_b, classes, output_dir / "ablation_latex.tex")
    print(f"  ablation_latex.tex")

    # B-over-A F1 deltas
    b_over_a_f1 = {}
    for cls in classes:
        b_over_a_f1[cls] = round(metrics_b["per_class"][cls]["f1"] - metrics_a["per_class"][cls]["f1"], 4)

    # Summary JSON
    summary = {
        "config_A_accuracy": metrics_a["accuracy"],
        "config_B_accuracy": metrics_b["accuracy"],
        "config_C_accuracy": metrics_c["accuracy"],
        "config_A_RWA": metrics_a["rwa"],
        "config_B_RWA": metrics_b["rwa"],
        "config_C_RWA": metrics_c["rwa"],
        "config_A_macro_f1": metrics_a["macro_f1"],
        "config_B_macro_f1": metrics_b["macro_f1"],
        "config_C_macro_f1": metrics_c["macro_f1"],
        "config_A_mcc": metrics_a["mcc"],
        "config_B_mcc": metrics_b["mcc"],
        "config_C_mcc": metrics_c["mcc"],
        "B_over_A_F1_delta": b_over_a_f1,
        "B_over_A_macro_F1_delta": round(metrics_b["macro_f1"] - metrics_a["macro_f1"], 4),
        "safety_gap_A": metrics_a["safety_gap"],
        "safety_gap_B": metrics_b["safety_gap"],
        "safety_gap_C": metrics_c["safety_gap"],
        "config_A_mean_latency_ms": round(mean_lat_a, 1),
        "config_B_mean_latency_ms": round(mean_lat_b, 1),
        "config_C_mean_latency_ms": round(mean_lat_c, 1),
        "n_test_images": len(test_images),
        "n_classes": len(classes),
    }
    (output_dir / "ablation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  ablation_summary.json")

    # ══════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  ABLATION RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"                    Config A (YOLO)  Config B (YOLO+Rules) Config C (Rules)  A vs B Delta")
    print(f"  Accuracy:         {metrics_a['accuracy']*100:5.1f}%           {metrics_b['accuracy']*100:5.1f}%              {metrics_c['accuracy']*100:5.1f}%           "
          f"{'+' if metrics_b['accuracy'] >= metrics_a['accuracy'] else ''}"
          f"{(metrics_b['accuracy']-metrics_a['accuracy'])*100:.1f}pp")
    print(f"  RWA:              {metrics_a['rwa']*100:5.1f}%           {metrics_b['rwa']*100:5.1f}%              {metrics_c['rwa']*100:5.1f}%           "
          f"{'+' if metrics_b['rwa'] >= metrics_a['rwa'] else ''}"
          f"{(metrics_b['rwa']-metrics_a['rwa'])*100:.1f}pp")
    print(f"  Macro-F1:         {metrics_a['macro_f1']:.3f}            {metrics_b['macro_f1']:.3f}               {metrics_c['macro_f1']:.3f}            "
          f"{'+' if metrics_b['macro_f1'] >= metrics_a['macro_f1'] else ''}"
          f"{metrics_b['macro_f1']-metrics_a['macro_f1']:.3f}")
    print(f"  MCC:              {metrics_a['mcc']:.3f}            {metrics_b['mcc']:.3f}               {metrics_c['mcc']:.3f}")
    print(f"  Safety Gap:       {metrics_a['safety_gap']*100:5.1f}pp          {metrics_b['safety_gap']*100:5.1f}pp             {metrics_c['safety_gap']*100:5.1f}pp")
    print(f"  Mean Latency:     {mean_lat_a:5.0f}ms          {mean_lat_b:5.0f}ms             {mean_lat_c:5.0f}ms")
    print(f"\n  All outputs → {output_dir}/")


if __name__ == "__main__":
    main()
