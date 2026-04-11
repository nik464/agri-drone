#!/usr/bin/env python3
"""
ablation_study.py — Rigorous ablation study for the AgriDrone disease detection pipeline.

Experimental Configurations:

  A  YOLO-Only         — Classifier raw top-1 prediction
  B  YOLO + Rules      — Classifier → Feature Extractor → Rule Engine
  C  YOLO + Rules + LLM — Full pipeline with LLaVA validation (ensemble)

Metrics per configuration:
  - Accuracy (top-1 exact match)
  - Precision (macro + per-class)
  - Recall (macro + per-class, emphasis on severe diseases)
  - F1 (macro + per-class)
  - False Negative Rate per severity tier
  - Mean processing latency

Statistical validation:
  - McNemar's test (pairwise A↔B, B↔C, A↔C)
  - 95% confidence intervals via bootstrap
  - Cohen's κ (inter-config agreement)

Outputs:
  - CSV per-image predictions (reproducibility artifact)
  - JSON summary with all metrics
  - LaTeX-ready ablation table (for paper)
  - MLflow logging (optional, --mlflow flag)

Usage:
    python scripts/ablation_study.py
    python scripts/ablation_study.py --test-dir data/raw/roboflow/rice-diseases-v2/test/images --n 100
    python scripts/ablation_study.py --skip-llm           # skip LLaVA (slow)
    python scripts/ablation_study.py --mlflow              # log to MLflow
    python scripts/ablation_study.py --runs 5              # repeated runs for CI
"""

import argparse
import csv
import json
import sys
import time
import random
import warnings
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ─── Severity tiers for stratified FNR analysis ───
# Maps disease profile severity values to tiers
SEVERE_DISEASES = {
    "wheat_fusarium_head_blight", "wheat_yellow_rust", "wheat_black_rust",
    "wheat_blast", "rice_blast", "rice_bacterial_blight",
}
MODERATE_DISEASES = {
    "wheat_brown_rust", "wheat_powdery_mildew", "wheat_septoria",
    "wheat_leaf_blight", "rice_brown_spot", "rice_sheath_blight",
    "rice_leaf_scald",
}
# Everything else (pests, minor, healthy) = low severity


# ════════════════════════════════════════════════════════════════
# Data structures
# ════════════════════════════════════════════════════════════════

@dataclass
class Prediction:
    """Single prediction record for one image under one configuration."""
    image: str
    ground_truth: str       # class name from label file
    predicted: str          # class name from model
    confidence: float
    correct: bool
    latency_ms: float
    severity_tier: str      # "severe" | "moderate" | "low"


@dataclass
class ConfigMetrics:
    """Aggregated metrics for one experimental configuration."""
    config_name: str
    accuracy: float = 0.0
    precision_macro: float = 0.0
    recall_macro: float = 0.0
    f1_macro: float = 0.0
    fnr_overall: float = 0.0   # false negative rate
    fnr_severe: float = 0.0    # FNR for severe diseases only
    fnr_moderate: float = 0.0
    mean_latency_ms: float = 0.0
    std_latency_ms: float = 0.0
    n_samples: int = 0
    per_class: dict = field(default_factory=dict)
    predictions: list = field(default_factory=list)    # raw Prediction list
    correct_vector: list = field(default_factory=list)  # for McNemar's


# ════════════════════════════════════════════════════════════════
# Ground truth extraction
# ════════════════════════════════════════════════════════════════

def load_ground_truth(images: list[Path], label_dir: Path, class_names: list[str]) -> dict[str, str]:
    """
    Map image filename → ground truth class name.

    For classification: takes the majority class from YOLO label file.
    For images without labels, returns "unknown".
    """
    gt = {}
    for img_path in images:
        lbl_path = label_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            gt[img_path.name] = "unknown"
            continue
        # Count class occurrences (some images have multi-class annotations)
        counts = defaultdict(int)
        for line in lbl_path.read_text().strip().splitlines():
            parts = line.strip().split()
            if parts:
                cls_id = int(parts[0])
                if cls_id < len(class_names):
                    counts[class_names[cls_id]] += 1
        if counts:
            gt[img_path.name] = max(counts, key=counts.get)
        else:
            gt[img_path.name] = "unknown"
    return gt


def severity_tier(class_name: str) -> str:
    """Map a class name to a severity tier."""
    # Normalize: "Brown spot" → fuzzy match against disease keys
    normalized = class_name.lower().replace(" ", "_")
    for disease_key in SEVERE_DISEASES:
        # Match on partial — "bacterial_blight" matches "Bacterial blight"
        suffix = disease_key.split("_", 1)[-1] if "_" in disease_key else disease_key
        if suffix in normalized or normalized in suffix:
            return "severe"
    for disease_key in MODERATE_DISEASES:
        suffix = disease_key.split("_", 1)[-1] if "_" in disease_key else disease_key
        if suffix in normalized or normalized in suffix:
            return "moderate"
    return "low"


# ════════════════════════════════════════════════════════════════
# Configuration A: YOLO Only
# ════════════════════════════════════════════════════════════════

def run_yolo_only(images: list[Path], gt: dict[str, str], cls_model) -> list[Prediction]:
    """Raw YOLO classifier — top-1 prediction, no post-processing."""
    import cv2

    predictions = []
    for img_path in images:
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        start = time.perf_counter()
        results = cls_model(image, verbose=False)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Extract top-1
        if results and hasattr(results[0], "probs") and results[0].probs is not None:
            probs = results[0].probs
            top_idx = int(probs.top1)
            top_conf = float(probs.top1conf)
            top_name = results[0].names[top_idx]
        else:
            top_name = "unknown"
            top_conf = 0.0

        truth = gt.get(img_path.name, "unknown")
        predictions.append(Prediction(
            image=img_path.name,
            ground_truth=truth,
            predicted=top_name,
            confidence=top_conf,
            correct=_class_match(top_name, truth),
            latency_ms=elapsed_ms,
            severity_tier=severity_tier(truth),
        ))

    return predictions


# ════════════════════════════════════════════════════════════════
# Configuration B: YOLO + Rules
# ════════════════════════════════════════════════════════════════

def run_yolo_plus_rules(images: list[Path], gt: dict[str, str], cls_model, crop_type: str) -> list[Prediction]:
    """YOLO classifier → feature extraction → rule engine."""
    import cv2
    from agridrone.vision.disease_reasoning import run_full_pipeline

    predictions = []
    for img_path in images:
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        start = time.perf_counter()

        # Step 1: YOLO classification
        results = cls_model(image, verbose=False)
        classifier_result = None
        if results and hasattr(results[0], "probs") and results[0].probs is not None:
            probs = results[0].probs
            top5_indices = probs.top5
            top5_confs = probs.top5conf.tolist()
            classifier_result = {
                "top_prediction": results[0].names[top5_indices[0]],
                "top_confidence": float(top5_confs[0]),
                "is_healthy": "healthy" in results[0].names[top5_indices[0]].lower(),
                "top5": [
                    {"class": results[0].names[idx], "confidence": float(conf)}
                    for idx, conf in zip(top5_indices, top5_confs)
                ],
            }

        # Step 2: Feature extraction + rule engine (no LLM)
        pipeline_out = run_full_pipeline(image, classifier_result, crop_type)
        diagnosis = pipeline_out.diagnosis

        elapsed_ms = (time.perf_counter() - start) * 1000

        truth = gt.get(img_path.name, "unknown")
        predicted = diagnosis.disease_name if diagnosis else "unknown"
        confidence = diagnosis.confidence if diagnosis else 0.0

        predictions.append(Prediction(
            image=img_path.name,
            ground_truth=truth,
            predicted=predicted,
            confidence=confidence,
            correct=_class_match(predicted, truth),
            latency_ms=elapsed_ms,
            severity_tier=severity_tier(truth),
        ))

    return predictions


# ════════════════════════════════════════════════════════════════
# Configuration C: YOLO + Rules + LLM
# ════════════════════════════════════════════════════════════════

def run_full_pipeline_with_llm(
    images: list[Path], gt: dict[str, str], cls_model, crop_type: str,
    llava_url: str = "http://localhost:11434",
) -> list[Prediction]:
    """Complete pipeline: YOLO → rules → LLaVA validation → ensemble."""
    import cv2
    import base64
    import requests
    from agridrone.vision.disease_reasoning import run_full_pipeline
    from agridrone.vision.llm_validator import (
        build_validation_prompt, parse_validation_response, fuse_confidence,
    )

    predictions = []
    for img_path in images:
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        start = time.perf_counter()

        # Step 1: YOLO classification
        results = cls_model(image, verbose=False)
        classifier_result = None
        if results and hasattr(results[0], "probs") and results[0].probs is not None:
            probs = results[0].probs
            top5_indices = probs.top5
            top5_confs = probs.top5conf.tolist()
            classifier_result = {
                "top_prediction": results[0].names[top5_indices[0]],
                "top_confidence": float(top5_confs[0]),
                "is_healthy": "healthy" in results[0].names[top5_indices[0]].lower(),
                "top5": [
                    {"class": results[0].names[idx], "confidence": float(conf)}
                    for idx, conf in zip(top5_indices, top5_confs)
                ],
            }

        # Step 2: Rule engine
        pipeline_out = run_full_pipeline(image, classifier_result, crop_type)
        diagnosis = pipeline_out.diagnosis
        rule_result = pipeline_out.rule_result
        features = pipeline_out.features

        # Step 3: LLM validation
        llm_predicted = diagnosis.disease_name if diagnosis else "unknown"
        llm_confidence = diagnosis.confidence if diagnosis else 0.0

        try:
            prompt_text, scenario = build_validation_prompt(
                rule_result, features, classifier_result, crop_type,
            )

            # Encode image for Ollama
            _, buf = cv2.imencode(".jpg", image)
            img_b64 = base64.b64encode(buf).decode()

            resp = requests.post(
                f"{llava_url}/api/generate",
                json={
                    "model": "llava:7b",
                    "prompt": prompt_text,
                    "images": [img_b64],
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 512},
                },
                timeout=120,
            )
            if resp.status_code == 200:
                raw_text = resp.json().get("response", "")
                our_diagnosis = diagnosis.disease_key if diagnosis else "unknown"
                validation = parse_validation_response(raw_text, scenario, our_diagnosis)

                # Ensemble: fuse confidence
                fusion = fuse_confidence(
                    classifier_conf=classifier_result.get("top_confidence", 0) if classifier_result else 0,
                    rule_conf=rule_result.top_confidence if rule_result else 0,
                    llm_agreement=validation.agreement_score,
                )

                # If LLM disagrees strongly, it may override
                if not validation.agrees and validation.llm_diagnosis:
                    llm_predicted = validation.llm_diagnosis
                    llm_confidence = fusion.get("fused_confidence", llm_confidence)
                else:
                    llm_confidence = fusion.get("fused_confidence", llm_confidence)

        except Exception as e:
            # LLM unavailable — fall back to rules result
            warnings.warn(f"LLM call failed for {img_path.name}: {e}")

        elapsed_ms = (time.perf_counter() - start) * 1000

        truth = gt.get(img_path.name, "unknown")
        predictions.append(Prediction(
            image=img_path.name,
            ground_truth=truth,
            predicted=llm_predicted,
            confidence=llm_confidence,
            correct=_class_match(llm_predicted, truth),
            latency_ms=elapsed_ms,
            severity_tier=severity_tier(truth),
        ))

    return predictions


# ════════════════════════════════════════════════════════════════
# Metrics computation
# ════════════════════════════════════════════════════════════════

def _class_match(predicted: str, ground_truth: str) -> bool:
    """Fuzzy class matching — handles naming discrepancies between
    classifier output ('rice_brown_spot') and label names ('Brown spot')."""
    if ground_truth == "unknown":
        return False
    p = predicted.lower().replace(" ", "_").replace("-", "_")
    g = ground_truth.lower().replace(" ", "_").replace("-", "_")
    if p == g:
        return True
    # Strip crop prefix: "rice_brown_spot" → "brown_spot"
    p_short = p.split("_", 1)[-1] if "_" in p else p
    g_short = g.split("_", 1)[-1] if "_" in g else g
    return p_short == g_short or p_short in g or g_short in p


def compute_metrics(config_name: str, predictions: list[Prediction]) -> ConfigMetrics:
    """Compute all metrics from a list of predictions."""
    valid = [p for p in predictions if p.ground_truth != "unknown"]
    if not valid:
        return ConfigMetrics(config_name=config_name, n_samples=0)

    n = len(valid)
    correct = sum(1 for p in valid if p.correct)
    accuracy = correct / n

    # Per-class TP/FP/FN
    classes = sorted(set(p.ground_truth for p in valid))
    per_class = {}
    for cls in classes:
        tp = sum(1 for p in valid if p.ground_truth == cls and p.correct)
        fp = sum(1 for p in valid if p.predicted == cls and p.ground_truth != cls
                 and not _class_match(p.predicted, p.ground_truth))
        fn = sum(1 for p in valid if p.ground_truth == cls and not p.correct)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[cls] = {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn,
                          "support": tp + fn}

    # Macro averages
    n_classes = len(per_class)
    precision_macro = sum(v["precision"] for v in per_class.values()) / n_classes if n_classes else 0
    recall_macro = sum(v["recall"] for v in per_class.values()) / n_classes if n_classes else 0
    f1_macro = sum(v["f1"] for v in per_class.values()) / n_classes if n_classes else 0

    # False Negative Rate by severity tier
    def fnr_for_tier(tier: str) -> float:
        tier_preds = [p for p in valid if p.severity_tier == tier]
        if not tier_preds:
            return 0.0
        fn_count = sum(1 for p in tier_preds if not p.correct)
        return fn_count / len(tier_preds)

    fnr_severe = fnr_for_tier("severe")
    fnr_moderate = fnr_for_tier("moderate")
    fnr_overall = sum(1 for p in valid if not p.correct) / n

    latencies = [p.latency_ms for p in valid]

    return ConfigMetrics(
        config_name=config_name,
        accuracy=accuracy,
        precision_macro=precision_macro,
        recall_macro=recall_macro,
        f1_macro=f1_macro,
        fnr_overall=fnr_overall,
        fnr_severe=fnr_severe,
        fnr_moderate=fnr_moderate,
        mean_latency_ms=float(np.mean(latencies)),
        std_latency_ms=float(np.std(latencies)),
        n_samples=n,
        per_class=per_class,
        predictions=valid,
        correct_vector=[1 if p.correct else 0 for p in valid],
    )


# ════════════════════════════════════════════════════════════════
# Statistical validation
# ════════════════════════════════════════════════════════════════

def mcnemars_test(vec_a: list[int], vec_b: list[int]) -> dict:
    """
    McNemar's test for paired nominal data.

    Compares two classifiers on the same test set.
    H₀: both classifiers have the same error rate.

    Returns: chi², p-value, and whether to reject H₀ at α=0.05.
    """
    from scipy import stats

    assert len(vec_a) == len(vec_b), "Vectors must be same length"
    n = len(vec_a)

    # Build contingency: b = A correct & B wrong, c = A wrong & B correct
    b = sum(1 for a, bv in zip(vec_a, vec_b) if a == 1 and bv == 0)
    c = sum(1 for a, bv in zip(vec_a, vec_b) if a == 0 and bv == 1)

    if b + c == 0:
        return {"chi2": 0.0, "p_value": 1.0, "significant": False, "b": b, "c": c}

    # McNemar's with continuity correction
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - stats.chi2.cdf(chi2, df=1)

    return {
        "chi2": round(chi2, 4),
        "p_value": round(p_value, 6),
        "significant": p_value < 0.05,
        "b": b,
        "c": c,
        "n": n,
    }


def bootstrap_ci(correct_vector: list[int], metric_fn=np.mean,
                  n_boot: int = 2000, ci: float = 0.95, seed: int = 42) -> tuple[float, float]:
    """Bootstrap 95% confidence interval for a metric."""
    rng = np.random.RandomState(seed)
    arr = np.array(correct_vector)
    samples = [float(metric_fn(rng.choice(arr, size=len(arr), replace=True))) for _ in range(n_boot)]
    alpha = (1 - ci) / 2
    lo = float(np.percentile(samples, 100 * alpha))
    hi = float(np.percentile(samples, 100 * (1 - alpha)))
    return lo, hi


def cohens_kappa(vec_a: list[int], vec_b: list[int]) -> float:
    """Cohen's κ — agreement between two configurations beyond chance."""
    n = len(vec_a)
    if n == 0:
        return 0.0
    agree = sum(1 for a, b in zip(vec_a, vec_b) if a == b)
    p_o = agree / n  # observed agreement
    # Expected agreement
    p_a1 = sum(vec_a) / n
    p_b1 = sum(vec_b) / n
    p_e = p_a1 * p_b1 + (1 - p_a1) * (1 - p_b1)
    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1 - p_e)


# ════════════════════════════════════════════════════════════════
# Output: CSV, JSON, LaTeX, MLflow
# ════════════════════════════════════════════════════════════════

def write_predictions_csv(predictions: list[Prediction], path: Path):
    """Write per-image prediction log (reproducibility artifact)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "image", "ground_truth", "predicted", "confidence",
            "correct", "latency_ms", "severity_tier",
        ])
        w.writeheader()
        for p in predictions:
            w.writerow({
                "image": p.image, "ground_truth": p.ground_truth,
                "predicted": p.predicted, "confidence": f"{p.confidence:.4f}",
                "correct": int(p.correct), "latency_ms": f"{p.latency_ms:.1f}",
                "severity_tier": p.severity_tier,
            })


def write_summary_json(configs: list[ConfigMetrics], stat_tests: dict, output_dir: Path):
    """Write complete results JSON."""
    summary = {
        "timestamp": datetime.now().isoformat(),
        "configurations": {},
        "statistical_tests": stat_tests,
    }
    for cfg in configs:
        summary["configurations"][cfg.config_name] = {
            "accuracy": round(cfg.accuracy, 4),
            "precision_macro": round(cfg.precision_macro, 4),
            "recall_macro": round(cfg.recall_macro, 4),
            "f1_macro": round(cfg.f1_macro, 4),
            "fnr_overall": round(cfg.fnr_overall, 4),
            "fnr_severe": round(cfg.fnr_severe, 4),
            "fnr_moderate": round(cfg.fnr_moderate, 4),
            "mean_latency_ms": round(cfg.mean_latency_ms, 1),
            "std_latency_ms": round(cfg.std_latency_ms, 1),
            "n_samples": cfg.n_samples,
            "per_class": cfg.per_class,
        }
    path = output_dir / "ablation_results.json"
    path.write_text(json.dumps(summary, indent=2))
    print(f"\n📄 JSON results → {path}")


def generate_latex_table(configs: list[ConfigMetrics], stat_tests: dict, output_dir: Path):
    """
    Generate publication-ready LaTeX ablation table.

    Format matches top-tier ML venues (CVPR, ICCV, AAAI).
    """
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Ablation study: contribution of each pipeline component. "
        r"Best results in \textbf{bold}. $\dagger$ indicates $p < 0.05$ vs.\ previous row (McNemar's test).}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{l c c c c c c}",
        r"\toprule",
        r"Configuration & Acc. & Prec. & Rec. & F1 & FNR\textsubscript{sev} & Latency \\",
        r" & (\%) & (\%) & (\%) & (\%) & (\%) & (ms) \\",
        r"\midrule",
    ]

    metrics_keys = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    # Find best value per column for bolding
    best = {}
    for key in metrics_keys:
        best[key] = max(getattr(c, key) for c in configs)
    best["fnr_severe"] = min(c.fnr_severe for c in configs)  # lower is better

    prev_name = None
    for cfg in configs:
        # Check if significantly different from previous
        sig = ""
        if prev_name and f"{prev_name}_vs_{cfg.config_name}" in stat_tests:
            test = stat_tests[f"{prev_name}_vs_{cfg.config_name}"]
            if test.get("significant"):
                sig = r"$^\dagger$"

        row_vals = []
        for key in metrics_keys:
            val = getattr(cfg, key) * 100
            best_val = best[key] * 100
            s = f"{val:.1f}"
            if abs(val - best_val) < 0.05:
                s = r"\textbf{" + s + "}"
            row_vals.append(s)

        # FNR severe (lower = better)
        fnr_val = cfg.fnr_severe * 100
        fnr_best = best["fnr_severe"] * 100
        fnr_s = f"{fnr_val:.1f}"
        if abs(fnr_val - fnr_best) < 0.05:
            fnr_s = r"\textbf{" + fnr_s + "}"
        row_vals.append(fnr_s)

        # Latency
        row_vals.append(f"{cfg.mean_latency_ms:.0f}")

        name = cfg.config_name + sig
        lines.append(f"  {name} & " + " & ".join(row_vals) + r" \\")
        prev_name = cfg.config_name

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    path = output_dir / "ablation_table.tex"
    path.write_text("\n".join(lines))
    print(f"📝 LaTeX table → {path}")


def generate_per_class_latex(configs: list[ConfigMetrics], output_dir: Path):
    """Per-class recall table — emphasizes severe disease detection."""
    all_classes = sorted(set().union(*(cfg.per_class.keys() for cfg in configs)))

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Per-class recall (\%) across ablation configurations. "
        r"\colorbox{red!15}{Red} = severe diseases, \colorbox{yellow!20}{yellow} = moderate.}",
        r"\label{tab:per_class_recall}",
        r"\small",
        r"\begin{tabular}{l" + " c" * len(configs) + "}",
        r"\toprule",
        "Disease & " + " & ".join(c.config_name for c in configs) + r" \\",
        r"\midrule",
    ]

    for cls in all_classes:
        tier = severity_tier(cls)
        prefix = ""
        if tier == "severe":
            prefix = r"\cellcolor{red!15} "
        elif tier == "moderate":
            prefix = r"\cellcolor{yellow!20} "

        vals = []
        best_val = max(
            cfg.per_class.get(cls, {}).get("recall", 0) for cfg in configs
        )
        for cfg in configs:
            v = cfg.per_class.get(cls, {}).get("recall", 0) * 100
            s = f"{v:.1f}"
            if abs(v - best_val * 100) < 0.05 and v > 0:
                s = r"\textbf{" + s + "}"
            vals.append(s)

        lines.append(f"  {prefix}{cls} & " + " & ".join(vals) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]

    path = output_dir / "per_class_recall_table.tex"
    path.write_text("\n".join(lines))
    print(f"📝 Per-class LaTeX → {path}")


def log_to_mlflow(configs: list[ConfigMetrics], stat_tests: dict, output_dir: Path):
    """Log all ablation results to MLflow."""
    import mlflow

    mlflow.set_tracking_uri(str(PROJECT_ROOT / "runs" / "mlflow"))
    mlflow.set_experiment("ablation_study")

    for cfg in configs:
        with mlflow.start_run(run_name=cfg.config_name):
            mlflow.log_param("config", cfg.config_name)
            mlflow.log_param("n_samples", cfg.n_samples)
            mlflow.log_metric("accuracy", cfg.accuracy)
            mlflow.log_metric("precision_macro", cfg.precision_macro)
            mlflow.log_metric("recall_macro", cfg.recall_macro)
            mlflow.log_metric("f1_macro", cfg.f1_macro)
            mlflow.log_metric("fnr_overall", cfg.fnr_overall)
            mlflow.log_metric("fnr_severe", cfg.fnr_severe)
            mlflow.log_metric("fnr_moderate", cfg.fnr_moderate)
            mlflow.log_metric("mean_latency_ms", cfg.mean_latency_ms)

            # Log per-class metrics
            for cls, vals in cfg.per_class.items():
                safe_cls = cls.replace(" ", "_").replace("/", "_")
                mlflow.log_metric(f"recall_{safe_cls}", vals["recall"])
                mlflow.log_metric(f"precision_{safe_cls}", vals["precision"])

            # Log artifacts
            csv_path = output_dir / f"predictions_{cfg.config_name}.csv"
            if csv_path.exists():
                mlflow.log_artifact(str(csv_path))

    # Log statistical tests as a single artifact
    tests_path = output_dir / "statistical_tests.json"
    tests_path.write_text(json.dumps(stat_tests, indent=2))
    mlflow.log_artifact(str(tests_path))
    print("📊 MLflow logging complete")


# ════════════════════════════════════════════════════════════════
# Console summary
# ════════════════════════════════════════════════════════════════

def print_summary(configs: list[ConfigMetrics], stat_tests: dict, cis: dict):
    """Print publication-style summary to console."""
    print("\n" + "═" * 90)
    print("  ABLATION STUDY RESULTS")
    print("═" * 90)

    # Main table
    header = f"{'Configuration':<25} {'Acc%':>7} {'Prec%':>7} {'Rec%':>7} {'F1%':>7} {'FNR_sev%':>9} {'Lat(ms)':>8}"
    print(header)
    print("─" * 90)
    for cfg in configs:
        ci = cis.get(cfg.config_name, (0, 0))
        print(
            f"  {cfg.config_name:<23} "
            f"{cfg.accuracy * 100:6.1f}  "
            f"{cfg.precision_macro * 100:6.1f}  "
            f"{cfg.recall_macro * 100:6.1f}  "
            f"{cfg.f1_macro * 100:6.1f}  "
            f"{cfg.fnr_severe * 100:8.1f}  "
            f"{cfg.mean_latency_ms:7.0f}"
        )
        print(f"  {'':23} [95% CI: {ci[0] * 100:.1f}–{ci[1] * 100:.1f}]")

    # Statistical tests
    print("\n── Statistical Validation ──")
    for name, test in stat_tests.items():
        sig = "✅ SIGNIFICANT" if test["significant"] else "❌ not significant"
        print(f"  {name}: χ²={test['chi2']:.2f}, p={test['p_value']:.4f}  → {sig}")

    # Kappa
    if len(configs) >= 2:
        kappa_ab = cohens_kappa(configs[0].correct_vector, configs[1].correct_vector)
        print(f"\n  Cohen's κ (A↔B): {kappa_ab:.3f}")
        if len(configs) >= 3:
            kappa_bc = cohens_kappa(configs[1].correct_vector, configs[2].correct_vector)
            print(f"  Cohen's κ (B↔C): {kappa_bc:.3f}")

    print("═" * 90)


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Ablation study: YOLO → Rules → LLM")
    parser.add_argument("--test-dir",
                        default=str(PROJECT_ROOT / "data" / "raw" / "roboflow" / "rice-diseases-v2" / "test" / "images"),
                        help="Directory of test images")
    parser.add_argument("--label-dir", default=None,
                        help="Directory of test labels (auto-detected if None)")
    parser.add_argument("--cls-model",
                        default=str(PROJECT_ROOT / "yolov8n-cls.pt"),
                        help="YOLO classifier weights")
    parser.add_argument("--crop-type", default="rice", choices=["wheat", "rice", "maize"])
    parser.add_argument("--n", type=int, default=0, help="Limit to N test images (0 = all)")
    parser.add_argument("--skip-llm", action="store_true", help="Skip Config C (LLaVA — slow)")
    parser.add_argument("--mlflow", action="store_true", help="Log results to MLflow")
    parser.add_argument("--runs", type=int, default=1, help="Repeated runs for CI estimation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir",
                        default=str(PROJECT_ROOT / "outputs" / "ablation"),
                        help="Output directory")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load test images ──
    test_dir = Path(args.test_dir)
    images = sorted(f for f in test_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS)
    if args.n > 0:
        images = images[:args.n]
    print(f"📂 Test images: {len(images)} from {test_dir}")

    # ── Ground truth ──
    label_dir = Path(args.label_dir) if args.label_dir else test_dir.parent / "labels"
    class_names = [  # from data.yaml
        "Bacterial blight", "Bacterial leaf", "Brown spot", "Cuterpillar",
        "Drainage impact", "Grashopper damage", "Grassy stunt", "Leaf folder",
        "Sheath blight", "Stem borer", "Tungro",
    ]
    gt = load_ground_truth(images, label_dir, class_names)
    gt_dist = defaultdict(int)
    for v in gt.values():
        gt_dist[v] += 1
    print(f"   Ground truth distribution: {dict(gt_dist)}")

    # ── Load YOLO classifier ──
    from ultralytics import YOLO
    print(f"🔧 Loading classifier: {args.cls_model}")
    cls_model = YOLO(args.cls_model)

    # ── Run configurations ──
    all_configs = []

    # Config A: YOLO Only
    print("\n━━━ Config A: YOLO Only ━━━")
    preds_a = run_yolo_only(images, gt, cls_model)
    metrics_a = compute_metrics("A: YOLO Only", preds_a)
    all_configs.append(metrics_a)
    write_predictions_csv(preds_a, output_dir / "predictions_A_yolo_only.csv")
    print(f"   Accuracy: {metrics_a.accuracy:.1%} | F1: {metrics_a.f1_macro:.1%} | FNR_sev: {metrics_a.fnr_severe:.1%}")

    # Config B: YOLO + Rules
    print("\n━━━ Config B: YOLO + Rules ━━━")
    preds_b = run_yolo_plus_rules(images, gt, cls_model, args.crop_type)
    metrics_b = compute_metrics("B: YOLO + Rules", preds_b)
    all_configs.append(metrics_b)
    write_predictions_csv(preds_b, output_dir / "predictions_B_yolo_rules.csv")
    print(f"   Accuracy: {metrics_b.accuracy:.1%} | F1: {metrics_b.f1_macro:.1%} | FNR_sev: {metrics_b.fnr_severe:.1%}")

    # Config C: YOLO + Rules + LLM
    if not args.skip_llm:
        print("\n━━━ Config C: YOLO + Rules + LLM ━━━")
        preds_c = run_full_pipeline_with_llm(images, gt, cls_model, args.crop_type)
        metrics_c = compute_metrics("C: YOLO + Rules + LLM", preds_c)
        all_configs.append(metrics_c)
        write_predictions_csv(preds_c, output_dir / "predictions_C_full_pipeline.csv")
        print(f"   Accuracy: {metrics_c.accuracy:.1%} | F1: {metrics_c.f1_macro:.1%} | FNR_sev: {metrics_c.fnr_severe:.1%}")

    # ── Statistical tests ──
    stat_tests = {}
    if len(all_configs) >= 2:
        stat_tests["A: YOLO Only_vs_B: YOLO + Rules"] = mcnemars_test(
            metrics_a.correct_vector, metrics_b.correct_vector
        )
    if len(all_configs) >= 3:
        stat_tests["B: YOLO + Rules_vs_C: YOLO + Rules + LLM"] = mcnemars_test(
            metrics_b.correct_vector, metrics_c.correct_vector
        )
        stat_tests["A: YOLO Only_vs_C: YOLO + Rules + LLM"] = mcnemars_test(
            metrics_a.correct_vector, metrics_c.correct_vector
        )

    # ── Bootstrap CIs ──
    cis = {}
    for cfg in all_configs:
        if cfg.correct_vector:
            cis[cfg.config_name] = bootstrap_ci(cfg.correct_vector)

    # ── Output ──
    print_summary(all_configs, stat_tests, cis)
    write_summary_json(all_configs, stat_tests, output_dir)
    generate_latex_table(all_configs, stat_tests, output_dir)
    generate_per_class_latex(all_configs, output_dir)

    if args.mlflow:
        log_to_mlflow(all_configs, stat_tests, output_dir)

    print(f"\n✅ All outputs saved to {output_dir}/")


if __name__ == "__main__":
    main()
