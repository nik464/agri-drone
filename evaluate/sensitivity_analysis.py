#!/usr/bin/env python3
"""
Experiment 3 — Threshold Sensitivity Analysis (optimized)

Grid search over three rule engine parameters:
  color_scale:             [15, 17, 20, 23, 25]   (default: 20)
  stripe_weight:           [0.3, 0.4, 0.5, 0.6, 0.7] (default: 0.5)
  yolo_override_threshold: [0.75, 0.80, 0.85, 0.90, 0.95] (default: 0.85)

→ 125 configurations evaluated on the validation set
→ macro-F1 and RWA per config
→ heatmaps + summary JSON

OPTIMIZATION: YOLO + feature extraction run ONCE per image, then only
the rule engine is re-run per config (with patched parameters).
This gives ~50-100× speed-up vs the naive approach.

Usage:
    python evaluate/sensitivity_analysis.py
    python evaluate/sensitivity_analysis.py --val-dir data/training/val --output-dir evaluate/results
"""

import argparse
import copy
import csv
import json
import logging
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from itertools import product

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ── Severity tiers (same as Experiment 1) ──
SEVERITY_TIERS = {
    "wheat_fusarium_head_blight": ("critical", 10),
    "wheat_yellow_rust":         ("critical", 10),
    "wheat_black_rust":          ("critical", 10),
    "wheat_blast":               ("critical", 10),
    "rice_blast":                ("critical", 10),
    "rice_bacterial_blight":     ("critical", 10),
    "wheat_brown_rust":          ("high", 5),
    "wheat_septoria":            ("high", 5),
    "wheat_leaf_blight":         ("high", 5),
    "rice_sheath_blight":        ("high", 5),
    "wheat_root_rot":            ("high", 5),
    "rice_leaf_scald":           ("high", 5),
    "wheat_powdery_mildew":      ("moderate", 2),
    "wheat_tan_spot":            ("moderate", 2),
    "wheat_aphid":               ("moderate", 2),
    "wheat_mite":                ("moderate", 2),
    "wheat_smut":                ("moderate", 2),
    "wheat_stem_fly":            ("moderate", 2),
    "rice_brown_spot":           ("moderate", 2),
    "healthy_wheat":             ("healthy", 1),
    "healthy_rice":              ("healthy", 1),
}


def get_tier_weight(class_key: str) -> int:
    return SEVERITY_TIERS.get(class_key, ("moderate", 2))[1]


# ════════════════════════════════════════════════════════════════
# Phase 1: Pre-compute YOLO + features (run ONCE per image)
# ════════════════════════════════════════════════════════════════

def precompute_yolo_and_features(model, loaded_images: list[tuple]) -> list[dict]:
    """Run YOLO inference + feature extraction once per image.
    Returns list of dicts with classifier_result, features, color_ratios, crop_type, gt."""
    import agridrone.vision.feature_extractor as fe_mod
    from agridrone.knowledge import kb_loader

    kb = kb_loader.get_all_profiles()
    if not kb:
        kb_loader.load()
        kb = kb_loader.get_all_profiles()

    cached = []
    for i, (image_bgr, gt, crop_type) in enumerate(loaded_images):
        # YOLO
        results = model(image_bgr, verbose=False)
        if not results or results[0].probs is None:
            cached.append({"gt": gt, "crop_type": crop_type, "classifier_result": None, "features": None})
            continue

        probs = results[0].probs
        names = model.names
        top5_indices = probs.top5
        top5_confs = probs.top5conf.tolist()

        classifier_result = {
            "top_prediction": names[probs.top1],
            "confidence": round(probs.top1conf.item(), 4),
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

        # Feature extraction
        features = fe_mod.extract_features(image_bgr, kb)

        # Save raw color_ratios for re-scaling later
        color_ratios_snapshot = dict(features.color_ratios)

        cached.append({
            "gt": gt,
            "crop_type": crop_type,
            "classifier_result": classifier_result,
            "features": features,
            "color_ratios": color_ratios_snapshot,
            "yolo_top1": names[probs.top1],
        })

        if (i + 1) % 100 == 0 or i == 0:
            print(f"    Pre-computed {i+1}/{len(loaded_images)} images")

    return cached


# ════════════════════════════════════════════════════════════════
# Phase 2: run rule engine with patched params (fast, per config)
# ════════════════════════════════════════════════════════════════

def run_rule_engine_patched(
    cached_item: dict,
    color_scale: float,
    stripe_weight: float,
    yolo_override_threshold: float,
) -> str:
    """Run rule engine on pre-computed features with patched parameters.
    Returns predicted class key."""
    import agridrone.vision.rule_engine as re_mod

    if cached_item["features"] is None or cached_item["classifier_result"] is None:
        return "unknown"

    # Deep-copy features so we don't mutate the cache
    features = copy.deepcopy(cached_item["features"])

    # Re-scale color_confidences with the new color_scale
    color_ratios = cached_item["color_ratios"]
    for key in features.color_confidences:
        if key in color_ratios:
            features.color_confidences[key] = min(1.0, color_ratios[key] * color_scale)

    # Save originals
    orig_spatial = re_mod._eval_spatial_rules
    orig_conflict = re_mod._resolve_conflict

    def patched_spatial(disease_key, profile, feats):
        matches = orig_spatial(disease_key, profile, feats)
        if stripe_weight != 0.5:
            for m in matches:
                if m.rule_name == "spatial_stripe_match":
                    if m.score_delta < 0.5:
                        combined = m.score_delta / 0.5
                    else:
                        combined = 1.0
                    object.__setattr__(m, 'score_delta', min(0.5, combined * stripe_weight))
        return matches

    def patched_conflict(yolo_top_key, yolo_top_conf, rule_top_key, rule_top_score, candidates):
        return _resolve_conflict_with_threshold(
            yolo_top_key, yolo_top_conf, rule_top_key, rule_top_score,
            candidates, yolo_override_threshold,
        )

    re_mod._eval_spatial_rules = patched_spatial
    re_mod._resolve_conflict = patched_conflict

    try:
        engine_result = re_mod.evaluate(features, cached_item["classifier_result"], cached_item["crop_type"])
        predicted = engine_result.top_disease if engine_result.top_disease else "healthy"
    except Exception as e:
        warnings.warn(f"Rule engine error: {e}")
        predicted = cached_item.get("yolo_top1", "unknown")
    finally:
        re_mod._eval_spatial_rules = orig_spatial
        re_mod._resolve_conflict = orig_conflict

    return predicted


def _resolve_conflict_with_threshold(
    yolo_top_key, yolo_top_conf, rule_top_key, rule_top_score,
    candidates, threshold,
):
    """Conflict resolution with configurable yolo override threshold."""
    from agridrone.vision.rule_engine import ConflictReport

    if yolo_top_key == rule_top_key:
        return ConflictReport(
            yolo_prediction=yolo_top_key,
            yolo_confidence=yolo_top_conf,
            rule_prediction=rule_top_key,
            rule_confidence=rule_top_score,
            winner="agree",
            reason="YOLO classifier and visual rules agree on diagnosis",
            yolo_rejections=[],
        )

    yolo_cand = candidates.get(yolo_top_key)
    rule_cand = candidates.get(rule_top_key)

    yolo_rule_score = yolo_cand.rule_score if yolo_cand else 0.0
    rule_cls_score = rule_cand.classifier_score if rule_cand else 0.0

    yolo_rejections = []
    if yolo_cand and yolo_cand.rejection:
        yolo_rejections = yolo_cand.rejection.reasons

    rules_have_strong_evidence = (rule_cand and rule_cand.rule_score > 0.3) if rule_cand else False
    yolo_very_confident = yolo_top_conf > threshold
    rules_have_weak_evidence = (rule_cand and rule_cand.rule_score < 0.15) if rule_cand else True

    if rules_have_strong_evidence and not yolo_very_confident:
        winner = "rules"
        reason = (
            f"Visual evidence strongly supports {rule_top_key} "
            f"(rule_score={rule_cand.rule_score:.2f}) while YOLO's {yolo_top_key} "
            f"({yolo_top_conf:.0%}) lacks visual confirmation"
        )
    elif yolo_very_confident and rules_have_weak_evidence:
        winner = "yolo"
        reason = (
            f"YOLO classifier very confident ({yolo_top_conf:.0%}) on {yolo_top_key} "
            f"and visual rules have weak evidence ({yolo_rule_score:.2f})"
        )
    else:
        yolo_combined = yolo_top_conf * 0.5 + yolo_rule_score * 0.5
        rule_combined = rule_cls_score * 0.5 + (rule_cand.rule_score if rule_cand else 0) * 0.5
        if rule_combined > yolo_combined:
            winner = "rules"
            reason = f"Combined evidence favors {rule_top_key}"
        else:
            winner = "yolo"
            reason = f"Combined evidence favors {yolo_top_key}"

    return ConflictReport(
        yolo_prediction=yolo_top_key,
        yolo_confidence=yolo_top_conf,
        rule_prediction=rule_top_key,
        rule_confidence=rule_top_score,
        winner=winner,
        reason=reason,
        yolo_rejections=yolo_rejections,
    )


# ════════════════════════════════════════════════════════════════
# Metrics
# ════════════════════════════════════════════════════════════════

def compute_metrics_fast(ground_truths: list[str], predictions: list[str], classes: list[str]) -> dict:
    """Compute macro-F1 and RWA from paired lists."""
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    total_correct = 0
    rwa_correct_w = 0.0
    rwa_total_w = 0.0

    for gt, pr in zip(ground_truths, predictions):
        w = get_tier_weight(gt)
        rwa_total_w += w
        if gt == pr:
            total_correct += 1
            tp[gt] += 1
            rwa_correct_w += w
        else:
            fn[gt] += 1
            fp[pr] += 1

    n = len(ground_truths)
    accuracy = total_correct / n if n > 0 else 0
    rwa = rwa_correct_w / rwa_total_w if rwa_total_w > 0 else 0

    macro_f1 = 0
    for cls in classes:
        p = tp[cls] / (tp[cls] + fp[cls]) if (tp[cls] + fp[cls]) > 0 else 0
        r = tp[cls] / (tp[cls] + fn[cls]) if (tp[cls] + fn[cls]) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        macro_f1 += f1
    macro_f1 /= len(classes) if classes else 1

    return {"accuracy": accuracy, "rwa": rwa, "macro_f1": macro_f1}


# ════════════════════════════════════════════════════════════════
# Load images
# ════════════════════════════════════════════════════════════════

def load_val_set(val_dir: Path) -> list[dict]:
    """Load validation images with ground truth labels."""
    images = []
    classes = sorted([d.name for d in val_dir.iterdir() if d.is_dir()])
    for cls in classes:
        cls_dir = val_dir / cls
        for img_path in sorted(cls_dir.glob("*")):
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}:
                images.append({
                    "path": str(img_path),
                    "ground_truth": cls,
                    "crop_type": "rice" if cls.startswith("rice") or cls == "healthy_rice" else "wheat",
                })
    return images


def preload_images(image_infos: list[dict]) -> list[tuple]:
    """Pre-read all images into memory to avoid I/O in the inner loop."""
    loaded = []
    for info in image_infos:
        bgr = cv2.imread(info["path"])
        if bgr is not None:
            loaded.append((bgr, info["ground_truth"], info["crop_type"]))
    return loaded


# ════════════════════════════════════════════════════════════════
# Plotting
# ════════════════════════════════════════════════════════════════

def _get_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_heatmap(data, x_vals, y_vals, x_label, y_label, title, output_path,
                 current_x=None, current_y=None, metric_name="macro-F1"):
    """2D heatmap with optional star marker for current config."""
    plt = _get_plt()

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto",
                   vmin=np.nanmin(data), vmax=np.nanmax(data))
    fig.colorbar(im, ax=ax, label=metric_name)

    ax.set_xticks(range(len(x_vals)))
    ax.set_xticklabels([str(v) for v in x_vals], fontsize=10)
    ax.set_yticks(range(len(y_vals)))
    ax.set_yticklabels([str(v) for v in y_vals], fontsize=10)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")

    # Annotate cells
    for i in range(len(y_vals)):
        for j in range(len(x_vals)):
            val = data[i, j]
            color = "white" if val > (np.nanmax(data) + np.nanmin(data)) / 2 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=8, color=color)

    # Star marker for current defaults
    if current_x is not None and current_y is not None:
        if current_x in x_vals and current_y in y_vals:
            xi = x_vals.index(current_x)
            yi = y_vals.index(current_y)
            ax.plot(xi, yi, marker="*", markersize=20, color="cyan",
                    markeredgecolor="black", markeredgewidth=1.5)
            ax.text(xi, yi - 0.35, "current", ha="center", va="bottom",
                    fontsize=8, fontweight="bold", color="cyan",
                    path_effects=[
                        __import__("matplotlib.patheffects", fromlist=["withStroke"]).withStroke(linewidth=2, foreground="black")
                    ])

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Experiment 3: Threshold Sensitivity")
    parser.add_argument("--val-dir", default=str(PROJECT_ROOT / "data" / "training" / "val"))
    parser.add_argument("--model-path", default=str(PROJECT_ROOT / "models" / "india_agri_cls.pt"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "evaluate" / "results"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Suppress noisy debug logging from feature_extractor and rule_engine
    logging.getLogger("agridrone.vision.feature_extractor").setLevel(logging.WARNING)
    logging.getLogger("agridrone.vision.rule_engine").setLevel(logging.WARNING)
    # Also suppress loguru if present
    try:
        from loguru import logger as loguru_logger
        loguru_logger.disable("agridrone.vision.feature_extractor")
        loguru_logger.disable("agridrone.vision.rule_engine")
    except ImportError:
        pass

    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Parameter grid ──
    color_scales = [15, 17, 20, 23, 25]
    stripe_weights = [0.3, 0.4, 0.5, 0.6, 0.7]
    yolo_thresholds = [0.75, 0.80, 0.85, 0.90, 0.95]

    # Current defaults
    CURRENT_COLOR = 20
    CURRENT_STRIPE = 0.5
    CURRENT_YOLO = 0.85

    total_configs = len(color_scales) * len(stripe_weights) * len(yolo_thresholds)

    # ── Load model ──
    print("Loading YOLO classifier...")
    from ultralytics import YOLO
    model = YOLO(args.model_path, task="classify")
    classes = sorted(model.names.values())
    print(f"  Classes: {len(classes)}")

    # ── Preload KB so it doesn't reload every iteration ──
    from agridrone.knowledge import kb_loader
    kb_loader.load()
    print("  Knowledge base loaded")

    # ── Load & preload validation images ──
    val_dir = Path(args.val_dir)
    val_images = load_val_set(val_dir)
    print(f"  Validation images: {len(val_images)}")

    print("  Pre-reading images into memory...")
    loaded = preload_images(val_images)
    print(f"  {len(loaded)} images loaded into RAM")

    # ── Phase 1: pre-compute YOLO + features (ONCE) ──
    print(f"\n{'='*70}")
    print("  PHASE 1: Pre-computing YOLO + feature extraction (once per image)")
    print(f"{'='*70}")
    t_precompute = time.perf_counter()
    cached = precompute_yolo_and_features(model, loaded)
    dt_precompute = time.perf_counter() - t_precompute
    print(f"  Pre-compute done in {dt_precompute:.1f}s ({len(cached)} images)")

    # ── Phase 2: Grid search (rule engine only) ──
    print(f"\n{'='*70}")
    print(f"  PHASE 2: GRID SEARCH — {total_configs} configurations (rule engine only)")
    print(f"  color_scale:    {color_scales}")
    print(f"  stripe_weight:  {stripe_weights}")
    print(f"  yolo_override:  {yolo_thresholds}")
    print(f"{'='*70}")

    results = []
    best_f1 = -1
    best_config = None
    config_idx = 0

    for cs, sw, yt in product(color_scales, stripe_weights, yolo_thresholds):
        config_idx += 1
        t0 = time.perf_counter()

        # Run all val images through this config
        gts = []
        preds = []
        for item in cached:
            pred = run_rule_engine_patched(item, cs, sw, yt)
            gts.append(item["gt"])
            preds.append(pred)

        elapsed = time.perf_counter() - t0
        metrics = compute_metrics_fast(gts, preds, classes)

        entry = {
            "color_scale": cs,
            "stripe_weight": sw,
            "yolo_override_threshold": yt,
            "macro_f1": round(metrics["macro_f1"], 4),
            "rwa": round(metrics["rwa"], 4),
            "accuracy": round(metrics["accuracy"], 4),
        }
        results.append(entry)

        is_current = (cs == CURRENT_COLOR and sw == CURRENT_STRIPE and yt == CURRENT_YOLO)
        marker = " ★ CURRENT" if is_current else ""
        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            best_config = entry.copy()

        if config_idx % 10 == 0 or config_idx == 1 or is_current:
            print(f"  [{config_idx:3d}/{total_configs}] "
                  f"cs={cs:2d} sw={sw:.1f} yt={yt:.2f} → "
                  f"F1={metrics['macro_f1']:.3f} RWA={metrics['rwa']:.3f} "
                  f"Acc={metrics['accuracy']:.3f} ({elapsed:.1f}s){marker}")

    # ── Current config results ──
    current_entry = next(
        (r for r in results
         if r["color_scale"] == CURRENT_COLOR
         and r["stripe_weight"] == CURRENT_STRIPE
         and r["yolo_override_threshold"] == CURRENT_YOLO),
        None
    )

    # ── Compute stability ──
    all_f1 = [r["macro_f1"] for r in results]
    all_rwa = [r["rwa"] for r in results]
    f1_std = float(np.std(all_f1))
    rwa_std = float(np.std(all_rwa))

    # ── Build 2D heatmaps (marginalised over 3rd dimension) ──
    print(f"\n{'='*70}")
    print("  GENERATING HEATMAPS")
    print(f"{'='*70}")

    # Heatmap 1: stripe_weight × color_scale (marginalised over yolo_threshold)
    heat_sc = np.zeros((len(stripe_weights), len(color_scales)))
    heat_sc_rwa = np.zeros_like(heat_sc)
    for r in results:
        si = stripe_weights.index(r["stripe_weight"])
        ci = color_scales.index(r["color_scale"])
        heat_sc[si, ci] += r["macro_f1"]
        heat_sc_rwa[si, ci] += r["rwa"]
    heat_sc /= len(yolo_thresholds)
    heat_sc_rwa /= len(yolo_thresholds)

    plot_heatmap(
        heat_sc, color_scales, stripe_weights,
        "Color Scale", "Stripe Weight",
        "Macro-F1: Stripe Weight vs Color Scale\n(averaged over YOLO thresholds)",
        output_dir / "sensitivity_stripe_vs_color.png",
        current_x=CURRENT_COLOR, current_y=CURRENT_STRIPE,
        metric_name="Macro-F1",
    )
    print(f"  sensitivity_stripe_vs_color.png")

    # Heatmap 2: stripe_weight × yolo_threshold (marginalised over color_scale)
    heat_sy = np.zeros((len(stripe_weights), len(yolo_thresholds)))
    heat_sy_rwa = np.zeros_like(heat_sy)
    for r in results:
        si = stripe_weights.index(r["stripe_weight"])
        yi = yolo_thresholds.index(r["yolo_override_threshold"])
        heat_sy[si, yi] += r["macro_f1"]
        heat_sy_rwa[si, yi] += r["rwa"]
    heat_sy /= len(color_scales)
    heat_sy_rwa /= len(color_scales)

    plot_heatmap(
        heat_sy, yolo_thresholds, stripe_weights,
        "YOLO Override Threshold", "Stripe Weight",
        "Macro-F1: Stripe Weight vs YOLO Threshold\n(averaged over color scales)",
        output_dir / "sensitivity_stripe_vs_threshold.png",
        current_x=CURRENT_YOLO, current_y=CURRENT_STRIPE,
        metric_name="Macro-F1",
    )
    print(f"  sensitivity_stripe_vs_threshold.png")

    # ── Save full grid CSV ──
    grid_path = output_dir / "sensitivity_grid.csv"
    with open(grid_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "color_scale", "stripe_weight", "yolo_override_threshold",
            "macro_f1", "rwa", "accuracy"])
        w.writeheader()
        w.writerows(results)
    print(f"  sensitivity_grid.csv ({len(results)} rows)")

    # ── Determine conclusion ──
    if current_entry:
        current_f1 = current_entry["macro_f1"]
        gap_to_best = best_f1 - current_f1
        if gap_to_best < 0.02:
            conclusion = "thresholds near-optimal"
        elif gap_to_best < 0.05:
            conclusion = "thresholds acceptable, minor tuning possible"
        else:
            conclusion = "thresholds need tuning"
    else:
        current_f1 = None
        conclusion = "current config not in grid"

    # ── Summary JSON ──
    summary = {
        "current_config": {
            "color_scale": CURRENT_COLOR,
            "stripe_weight": CURRENT_STRIPE,
            "yolo_override_threshold": CURRENT_YOLO,
        },
        "current_config_F1": current_entry["macro_f1"] if current_entry else None,
        "current_config_RWA": current_entry["rwa"] if current_entry else None,
        "current_config_accuracy": current_entry["accuracy"] if current_entry else None,
        "optimal_config": {
            "color_scale": best_config["color_scale"],
            "stripe_weight": best_config["stripe_weight"],
            "yolo_override_threshold": best_config["yolo_override_threshold"],
        },
        "optimal_config_F1": best_config["macro_f1"],
        "optimal_config_RWA": best_config["rwa"],
        "optimal_config_accuracy": best_config["accuracy"],
        "F1_std_across_configs": round(f1_std, 4),
        "RWA_std_across_configs": round(rwa_std, 4),
        "F1_range": [round(min(all_f1), 4), round(max(all_f1), 4)],
        "RWA_range": [round(min(all_rwa), 4), round(max(all_rwa), 4)],
        "n_configs_evaluated": len(results),
        "n_val_images": len(cached),
        "precompute_time_s": round(dt_precompute, 1),
        "conclusion": conclusion,
    }

    summary_path = output_dir / "sensitivity_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  sensitivity_summary.json")

    # ── Print summary ──
    print(f"\n{'='*70}")
    print(f"  SENSITIVITY ANALYSIS RESULTS")
    print(f"{'='*70}")
    if current_entry:
        print(f"  Current config (cs={CURRENT_COLOR}, sw={CURRENT_STRIPE}, yt={CURRENT_YOLO}):")
        print(f"    F1={current_entry['macro_f1']:.4f}  RWA={current_entry['rwa']:.4f}  "
              f"Acc={current_entry['accuracy']:.4f}")
    print(f"  Best config (cs={best_config['color_scale']}, sw={best_config['stripe_weight']}, "
          f"yt={best_config['yolo_override_threshold']}):")
    print(f"    F1={best_config['macro_f1']:.4f}  RWA={best_config['rwa']:.4f}  "
          f"Acc={best_config['accuracy']:.4f}")
    print(f"  F1 std across all {len(results)} configs: {f1_std:.4f}")
    print(f"  F1 range: [{min(all_f1):.4f}, {max(all_f1):.4f}]")
    print(f"  Conclusion: {conclusion}")
    print(f"\n  All outputs → {output_dir}/")


if __name__ == "__main__":
    main()
