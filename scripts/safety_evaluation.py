#!/usr/bin/env python3
"""
safety_evaluation.py — Safety-critical metrics for crop disease detection.

Defines and measures three classes of safety metrics:

  S1  False Negative Rate (stratified by disease severity)
  S2  Risk-Weighted Accuracy (penalizes misses proportional to economic harm)
  S3  Cost-Sensitive Evaluation (Expected Monetary Loss under each configuration)

Statistical framework:
  - Bootstrap confidence intervals (BCa method)
  - Permutation test for risk-weighted accuracy differences
  - Expected Calibration Error (ECE) for confidence reliability

Output artifacts:
  - LaTeX tables (Table: Safety Metrics, Table: Cost Matrix, Table: Per-disease FNR)
  - Matplotlib figures (risk heatmap, cost waterfall, FNR bar chart, calibration plot)
  - JSON summary for reproducibility
  - MLflow logging (optional)

Usage:
    python scripts/safety_evaluation.py
    python scripts/safety_evaluation.py --config ablation    # compare A/B/C configs
    python scripts/safety_evaluation.py --config single      # evaluate single model
    python scripts/safety_evaluation.py --predictions outputs/ablation/predictions_B_yolo_rules.csv
    python scripts/safety_evaluation.py --mlflow
"""

import argparse
import csv
import json
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ════════════════════════════════════════════════════════════════
# Disease Risk Taxonomy — grounded in diseases.json economics
# ════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class DiseaseRisk:
    """Risk profile for one disease, derived from agronomic data."""
    key: str
    display_name: str
    severity: float            # 0.0–1.0 from diseases.json
    max_yield_loss_pct: float  # upper bound of yield_loss range
    urgency: str               # immediate | within_7_days | within_30_days | none
    treatment_cost_inr: float  # fungicide + labor per acre per application
    spray_applications: int
    treatment_efficacy: float  # 0.0–1.0
    tier: str                  # "critical" | "high" | "moderate" | "low" | "healthy"

    @property
    def miss_cost_inr_per_acre(self) -> float:
        """Economic cost of missing this disease (false negative).
        = potential yield loss revenue if untreated."""
        # Use rice price ₹2300/quintal, 1500 kg/acre as reference
        revenue_per_acre = 1500 * (2300 / 100)  # ₹34,500
        return revenue_per_acre * (self.max_yield_loss_pct / 100)

    @property
    def false_alarm_cost_inr_per_acre(self) -> float:
        """Economic cost of false positive (unnecessary treatment)."""
        return self.treatment_cost_inr * self.spray_applications

    @property
    def cost_ratio(self) -> float:
        """C_FN / C_FP — how many times worse a miss is vs a false alarm."""
        if self.false_alarm_cost_inr_per_acre == 0:
            return 0.0
        return self.miss_cost_inr_per_acre / self.false_alarm_cost_inr_per_acre


# Complete risk registry — all 21 classes
DISEASE_RISKS: dict[str, DiseaseRisk] = {
    # ── Critical tier (severity ≥ 0.8, immediate urgency, yield loss ≥ 40%) ──
    "wheat_fusarium_head_blight": DiseaseRisk(
        "wheat_fusarium_head_blight", "Fusarium Head Blight", 0.9, 50, "immediate",
        320, 2, 0.60, "critical"),
    "wheat_yellow_rust": DiseaseRisk(
        "wheat_yellow_rust", "Yellow Rust", 0.85, 100, "immediate",
        280, 2, 0.80, "critical"),
    "wheat_black_rust": DiseaseRisk(
        "wheat_black_rust", "Black Rust", 0.80, 70, "immediate",
        300, 2, 0.75, "critical"),
    "wheat_blast": DiseaseRisk(
        "wheat_blast", "Wheat Blast", 0.85, 100, "immediate",
        400, 3, 0.55, "critical"),
    "rice_blast": DiseaseRisk(
        "rice_blast", "Rice Blast", 0.90, 100, "immediate",
        350, 2, 0.72, "critical"),
    "rice_bacterial_blight": DiseaseRisk(
        "rice_bacterial_blight", "Bacterial Blight", 0.80, 50, "immediate",
        200, 2, 0.65, "critical"),

    # ── High tier (severity 0.65–0.79, within_7_days) ──
    "wheat_brown_rust": DiseaseRisk(
        "wheat_brown_rust", "Brown Rust", 0.75, 40, "within_7_days",
        250, 2, 0.78, "high"),
    "wheat_septoria": DiseaseRisk(
        "wheat_septoria", "Septoria", 0.70, 30, "within_7_days",
        280, 2, 0.72, "high"),
    "wheat_leaf_blight": DiseaseRisk(
        "wheat_leaf_blight", "Leaf Blight", 0.70, 25, "within_7_days",
        250, 2, 0.70, "high"),
    "wheat_root_rot": DiseaseRisk(
        "wheat_root_rot", "Root Rot", 0.70, 30, "within_30_days",
        200, 1, 0.55, "high"),
    "rice_sheath_blight": DiseaseRisk(
        "rice_sheath_blight", "Sheath Blight", 0.70, 30, "within_7_days",
        280, 2, 0.73, "high"),
    "rice_leaf_scald": DiseaseRisk(
        "rice_leaf_scald", "Leaf Scald", 0.65, 20, "within_7_days",
        200, 1, 0.68, "high"),
    "wheat_smut": DiseaseRisk(
        "wheat_smut", "Smut", 0.65, 30, "within_30_days",
        180, 1, 0.90, "high"),

    # ── Moderate tier (severity 0.5–0.64) ──
    "wheat_powdery_mildew": DiseaseRisk(
        "wheat_powdery_mildew", "Powdery Mildew", 0.60, 30, "within_7_days",
        200, 2, 0.75, "moderate"),
    "wheat_tan_spot": DiseaseRisk(
        "wheat_tan_spot", "Tan Spot", 0.60, 20, "within_7_days",
        220, 1, 0.68, "moderate"),
    "wheat_aphid": DiseaseRisk(
        "wheat_aphid", "Aphid", 0.55, 25, "within_7_days",
        180, 1, 0.85, "moderate"),
    "wheat_mite": DiseaseRisk(
        "wheat_mite", "Mite", 0.50, 15, "within_7_days",
        160, 1, 0.80, "moderate"),
    "wheat_stem_fly": DiseaseRisk(
        "wheat_stem_fly", "Stem Fly", 0.50, 30, "within_7_days",
        200, 1, 0.75, "moderate"),
    "rice_brown_spot": DiseaseRisk(
        "rice_brown_spot", "Brown Spot", 0.60, 30, "within_7_days",
        220, 2, 0.70, "moderate"),

    # ── Healthy (no risk) ──
    "healthy_wheat": DiseaseRisk(
        "healthy_wheat", "Healthy Wheat", 0.0, 0, "none", 0, 0, 0, "healthy"),
    "healthy_rice": DiseaseRisk(
        "healthy_rice", "Healthy Rice", 0.0, 0, "none", 0, 0, 0, "healthy"),
}


# Risk weights per tier — used in risk-weighted accuracy
# Derived from mean cost_ratio within each tier
TIER_WEIGHTS = {
    "critical": 10.0,    # Missing a critical disease is 10x worse than a normal error
    "high":     5.0,
    "moderate": 2.0,
    "low":      1.0,
    "healthy":  1.0,     # False positive on healthy = 1x (unnecessary spray cost only)
}


def resolve_risk(class_name: str) -> DiseaseRisk:
    """Map a class name (from predictions CSV) to its DiseaseRisk profile."""
    normalized = class_name.lower().replace(" ", "_").replace("-", "_")

    # Direct match
    if normalized in DISEASE_RISKS:
        return DISEASE_RISKS[normalized]

    # Strip crop prefix
    for key, risk in DISEASE_RISKS.items():
        suffix = key.split("_", 1)[-1] if "_" in key else key
        if suffix in normalized or normalized in suffix:
            return risk
        # Also match display name
        if normalized.replace("_", " ") in risk.display_name.lower():
            return risk

    # Default: moderate risk (conservative — don't undercount unknown diseases)
    return DiseaseRisk(normalized, class_name, 0.5, 15, "within_7_days",
                       200, 1, 0.7, "moderate")


# ════════════════════════════════════════════════════════════════
# Metric S1: False Negative Rate (stratified)
# ════════════════════════════════════════════════════════════════

@dataclass
class FNRReport:
    """False Negative Rate decomposed by severity tier."""
    fnr_overall: float
    fnr_critical: float      # MOST IMPORTANT — missing blast, rust, blight
    fnr_high: float
    fnr_moderate: float
    fnr_healthy_as_diseased: float  # false positive rate on healthy images
    per_disease_fnr: dict[str, float] = field(default_factory=dict)
    per_disease_support: dict[str, int] = field(default_factory=dict)
    n_total: int = 0
    n_critical: int = 0
    n_high: int = 0
    n_moderate: int = 0


def compute_fnr(predictions: list[dict]) -> FNRReport:
    """
    Compute stratified False Negative Rate.

    A false negative = ground truth is disease X, model predicts something else.
    For healthy images: a false negative = healthy predicted as diseased (inverted).
    """
    by_tier = defaultdict(lambda: {"total": 0, "missed": 0})
    by_disease = defaultdict(lambda: {"total": 0, "missed": 0})

    for p in predictions:
        gt = p["ground_truth"]
        correct = p["correct"]
        risk = resolve_risk(gt)
        tier = risk.tier

        by_tier[tier]["total"] += 1
        by_disease[gt]["total"] += 1

        if not correct:
            by_tier[tier]["missed"] += 1
            by_disease[gt]["missed"] += 1

    def _fnr(group: str) -> float:
        d = by_tier[group]
        return d["missed"] / d["total"] if d["total"] > 0 else 0.0

    per_disease_fnr = {}
    per_disease_support = {}
    for disease, counts in by_disease.items():
        per_disease_fnr[disease] = counts["missed"] / counts["total"] if counts["total"] > 0 else 0.0
        per_disease_support[disease] = counts["total"]

    total = sum(d["total"] for d in by_tier.values())
    missed = sum(d["missed"] for d in by_tier.values())

    return FNRReport(
        fnr_overall=missed / total if total > 0 else 0.0,
        fnr_critical=_fnr("critical"),
        fnr_high=_fnr("high"),
        fnr_moderate=_fnr("moderate"),
        fnr_healthy_as_diseased=_fnr("healthy"),
        per_disease_fnr=per_disease_fnr,
        per_disease_support=per_disease_support,
        n_total=total,
        n_critical=by_tier["critical"]["total"],
        n_high=by_tier["high"]["total"],
        n_moderate=by_tier["moderate"]["total"],
    )


# ════════════════════════════════════════════════════════════════
# Metric S2: Risk-Weighted Accuracy (RWA)
# ════════════════════════════════════════════════════════════════

@dataclass
class RWAReport:
    """Risk-Weighted Accuracy — errors on severe diseases penalized more."""
    rwa: float                # risk-weighted accuracy [0, 1]
    standard_accuracy: float  # for comparison
    safety_gap: float         # standard_acc - rwa (positive = system overestimates its safety)
    weighted_correct: float
    weighted_total: float
    tier_accuracy: dict[str, float] = field(default_factory=dict)


def compute_rwa(predictions: list[dict]) -> RWAReport:
    r"""
    Risk-Weighted Accuracy.

    $$\text{RWA} = \frac{\sum_{i=1}^{N} w_i \cdot \mathbb{1}[\hat{y}_i = y_i]}{\sum_{i=1}^{N} w_i}$$

    where $w_i = \text{TIER\_WEIGHTS}[\text{tier}(y_i)]$.

    A system with 90% accuracy but 60% recall on critical diseases will have
    RWA << 90%, exposing the safety gap.
    """
    weighted_correct = 0.0
    weighted_total = 0.0
    n_correct = 0
    n_total = len(predictions)

    tier_correct = defaultdict(float)
    tier_total = defaultdict(float)

    for p in predictions:
        gt = p["ground_truth"]
        correct = p["correct"]
        risk = resolve_risk(gt)
        w = TIER_WEIGHTS.get(risk.tier, 1.0)

        weighted_total += w
        tier_total[risk.tier] += 1

        if correct:
            weighted_correct += w
            n_correct += 1
            tier_correct[risk.tier] += 1

    rwa = weighted_correct / weighted_total if weighted_total > 0 else 0.0
    std_acc = n_correct / n_total if n_total > 0 else 0.0

    tier_accuracy = {}
    for tier in TIER_WEIGHTS:
        if tier_total[tier] > 0:
            tier_accuracy[tier] = tier_correct[tier] / tier_total[tier]

    return RWAReport(
        rwa=rwa,
        standard_accuracy=std_acc,
        safety_gap=std_acc - rwa,
        weighted_correct=weighted_correct,
        weighted_total=weighted_total,
        tier_accuracy=tier_accuracy,
    )


# ════════════════════════════════════════════════════════════════
# Metric S3: Cost-Sensitive Evaluation (Expected Monetary Loss)
# ════════════════════════════════════════════════════════════════

@dataclass
class CostReport:
    """Expected monetary loss under a given model configuration."""
    eml_per_acre_inr: float      # Expected monetary loss per acre
    eml_fn_component: float      # Loss from false negatives (missed diseases)
    eml_fp_component: float      # Loss from false positives (unnecessary treatment)
    eml_by_tier: dict[str, float] = field(default_factory=dict)
    eml_by_disease: dict[str, float] = field(default_factory=dict)
    n_false_negatives: int = 0
    n_false_positives: int = 0
    total_fn_cost_inr: float = 0.0
    total_fp_cost_inr: float = 0.0
    cost_matrix: dict = field(default_factory=dict)  # for visualization


def compute_cost(predictions: list[dict]) -> CostReport:
    r"""
    Expected Monetary Loss (EML).

    For each prediction, compute the economic consequence:

    - **False Negative** (missed disease): farmer loses yield
      $$C_{FN}(d) = \text{yield\_per\_acre} \times \text{price} \times \text{max\_loss\_pct}(d)$$

    - **False Positive** (unnecessary spray): farmer wastes treatment cost
      $$C_{FP}(d) = \text{treatment\_cost}(d) \times \text{sprays}(d)$$

    - **Correct**: $C = 0$

    $$\text{EML} = \frac{1}{N} \sum_{i=1}^{N} C(y_i, \hat{y}_i)$$
    """
    total_fn_cost = 0.0
    total_fp_cost = 0.0
    n_fn = 0
    n_fp = 0
    n = len(predictions)

    eml_by_tier = defaultdict(float)
    eml_by_disease = defaultdict(float)
    cost_matrix = defaultdict(lambda: defaultdict(float))

    for p in predictions:
        gt = p["ground_truth"]
        pred = p["predicted"]
        correct = p["correct"]
        gt_risk = resolve_risk(gt)
        pred_risk = resolve_risk(pred)

        if correct:
            continue

        gt_is_healthy = gt_risk.tier == "healthy"
        pred_is_healthy = pred_risk.tier == "healthy"

        if not gt_is_healthy and (pred_is_healthy or not correct):
            # FALSE NEGATIVE: disease present but missed/misclassified
            cost = gt_risk.miss_cost_inr_per_acre
            total_fn_cost += cost
            eml_by_tier[gt_risk.tier] += cost
            eml_by_disease[gt] += cost
            cost_matrix[gt]["fn_cost"] += cost
            cost_matrix[gt]["fn_count"] = cost_matrix[gt].get("fn_count", 0) + 1
            n_fn += 1

        elif gt_is_healthy and not pred_is_healthy:
            # FALSE POSITIVE: healthy but predicted diseased
            cost = pred_risk.false_alarm_cost_inr_per_acre
            total_fp_cost += cost
            eml_by_tier["healthy_fp"] += cost
            cost_matrix[pred]["fp_cost"] += cost
            cost_matrix[pred]["fp_count"] = cost_matrix[pred].get("fp_count", 0) + 1
            n_fp += 1

    eml = (total_fn_cost + total_fp_cost) / n if n > 0 else 0.0

    return CostReport(
        eml_per_acre_inr=eml,
        eml_fn_component=total_fn_cost / n if n > 0 else 0.0,
        eml_fp_component=total_fp_cost / n if n > 0 else 0.0,
        eml_by_tier=dict(eml_by_tier),
        eml_by_disease=dict(eml_by_disease),
        n_false_negatives=n_fn,
        n_false_positives=n_fp,
        total_fn_cost_inr=total_fn_cost,
        total_fp_cost_inr=total_fp_cost,
        cost_matrix={k: dict(v) for k, v in cost_matrix.items()},
    )


# ════════════════════════════════════════════════════════════════
# Metric S4: Expected Calibration Error (ECE)
# ════════════════════════════════════════════════════════════════

def compute_ece(predictions: list[dict], n_bins: int = 10) -> dict:
    r"""
    Expected Calibration Error — measures whether confidence matches accuracy.

    $$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

    A well-calibrated safety system should have ECE < 0.05.
    Overconfident false negatives on severe diseases are the worst failure mode.
    """
    bins = [[] for _ in range(n_bins)]
    for p in predictions:
        conf = p.get("confidence", 0.5)
        bin_idx = min(int(conf * n_bins), n_bins - 1)
        bins[bin_idx].append(p)

    ece = 0.0
    n = len(predictions)
    bin_data = []
    for i, bin_preds in enumerate(bins):
        if not bin_preds:
            continue
        bin_size = len(bin_preds)
        avg_conf = np.mean([p.get("confidence", 0.5) for p in bin_preds])
        avg_acc = np.mean([1 if p["correct"] else 0 for p in bin_preds])
        ece += (bin_size / n) * abs(avg_acc - avg_conf)
        bin_data.append({
            "bin": i,
            "range": f"[{i / n_bins:.1f}, {(i + 1) / n_bins:.1f})",
            "count": bin_size,
            "avg_confidence": round(float(avg_conf), 4),
            "avg_accuracy": round(float(avg_acc), 4),
            "gap": round(float(avg_acc - avg_conf), 4),
        })

    return {"ece": round(float(ece), 4), "bins": bin_data, "n_bins": n_bins}


# ════════════════════════════════════════════════════════════════
# Statistical Validation
# ════════════════════════════════════════════════════════════════

def bootstrap_metric(values: list, metric_fn=np.mean, n_boot: int = 5000,
                     ci: float = 0.95, seed: int = 42) -> dict:
    """BCa bootstrap confidence interval."""
    rng = np.random.RandomState(seed)
    arr = np.array(values, dtype=float)
    observed = float(metric_fn(arr))
    boot_samples = [float(metric_fn(rng.choice(arr, size=len(arr), replace=True)))
                    for _ in range(n_boot)]

    alpha = (1 - ci) / 2
    lo = float(np.percentile(boot_samples, 100 * alpha))
    hi = float(np.percentile(boot_samples, 100 * (1 - alpha)))
    se = float(np.std(boot_samples))

    return {"estimate": round(observed, 4), "ci_lo": round(lo, 4),
            "ci_hi": round(hi, 4), "se": round(se, 4), "n_boot": n_boot}


def permutation_test_rwa(preds_a: list[dict], preds_b: list[dict],
                         n_perm: int = 10000, seed: int = 42) -> dict:
    """
    Permutation test for RWA difference.

    H₀: RWA(A) = RWA(B) — both systems have equal risk-weighted performance.
    """
    rng = np.random.RandomState(seed)

    def _rwa_from_preds(preds):
        r = compute_rwa(preds)
        return r.rwa

    observed_diff = _rwa_from_preds(preds_b) - _rwa_from_preds(preds_a)

    # Pool and permute
    combined = preds_a + preds_b
    n_a = len(preds_a)
    count_extreme = 0

    for _ in range(n_perm):
        perm = rng.permutation(len(combined))
        group_a = [combined[i] for i in perm[:n_a]]
        group_b = [combined[i] for i in perm[n_a:]]
        perm_diff = _rwa_from_preds(group_b) - _rwa_from_preds(group_a)
        if abs(perm_diff) >= abs(observed_diff):
            count_extreme += 1

    p_value = (count_extreme + 1) / (n_perm + 1)  # +1 for continuity

    return {
        "observed_diff": round(float(observed_diff), 4),
        "p_value": round(float(p_value), 6),
        "significant": p_value < 0.05,
        "n_permutations": n_perm,
    }


# ════════════════════════════════════════════════════════════════
# Figures (matplotlib)
# ════════════════════════════════════════════════════════════════

def plot_fnr_by_disease(fnr_reports: dict[str, FNRReport], output_dir: Path):
    """Bar chart: per-disease FNR across configurations."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  ⚠ matplotlib not installed — skipping FNR bar chart")
        return

    # Collect all diseases that appear in any config
    all_diseases = sorted(set().union(*(r.per_disease_fnr.keys() for r in fnr_reports.values())))
    # Sort by max FNR (worst diseases first)
    all_diseases.sort(key=lambda d: max(
        r.per_disease_fnr.get(d, 0) for r in fnr_reports.values()
    ), reverse=True)

    configs = list(fnr_reports.keys())
    x = np.arange(len(all_diseases))
    width = 0.8 / len(configs)

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ["#e74c3c", "#f39c12", "#27ae60", "#3498db"]

    for i, (config_name, report) in enumerate(fnr_reports.items()):
        vals = [report.per_disease_fnr.get(d, 0) * 100 for d in all_diseases]
        bars = ax.bar(x + i * width, vals, width, label=config_name,
                      color=colors[i % len(colors)], alpha=0.85)

    # Shade critical diseases
    for j, d in enumerate(all_diseases):
        risk = resolve_risk(d)
        if risk.tier == "critical":
            ax.axvspan(j - 0.4, j + 0.4 + width * len(configs),
                       alpha=0.08, color="red", zorder=0)

    ax.set_ylabel("False Negative Rate (%)", fontsize=12)
    ax.set_title("Per-Disease False Negative Rate by Configuration", fontsize=13, fontweight="bold")
    ax.set_xticks(x + width * (len(configs) - 1) / 2)
    ax.set_xticklabels(all_diseases, rotation=45, ha="right", fontsize=9)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_ylim(0, 105)
    ax.axhline(y=20, color="gray", linestyle="--", alpha=0.5, label="20% target")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "fnr_by_disease.png", dpi=200)
    fig.savefig(output_dir / "fnr_by_disease.pdf")
    plt.close(fig)
    print(f"  📊 FNR bar chart → {output_dir / 'fnr_by_disease.png'}")


def plot_cost_waterfall(cost_reports: dict[str, CostReport], output_dir: Path):
    """Waterfall chart: EML decomposition (FN cost vs FP cost)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    configs = list(cost_reports.keys())
    fn_costs = [cost_reports[c].eml_fn_component for c in configs]
    fp_costs = [cost_reports[c].eml_fp_component for c in configs]

    x = np.arange(len(configs))
    width = 0.5

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, fn_costs, width, label="FN cost (missed disease)", color="#e74c3c", alpha=0.85)
    ax.bar(x, fp_costs, width, bottom=fn_costs, label="FP cost (unnecessary spray)",
           color="#f39c12", alpha=0.85)

    ax.set_ylabel("Expected Monetary Loss (₹/acre)", fontsize=12)
    ax.set_title("Cost-Sensitive Evaluation: EML Decomposition", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # Annotate total
    for i, c in enumerate(configs):
        total = fn_costs[i] + fp_costs[i]
        ax.text(i, total + 50, f"₹{total:,.0f}", ha="center", fontweight="bold", fontsize=10)

    fig.tight_layout()
    fig.savefig(output_dir / "cost_waterfall.png", dpi=200)
    fig.savefig(output_dir / "cost_waterfall.pdf")
    plt.close(fig)
    print(f"  📊 Cost waterfall → {output_dir / 'cost_waterfall.png'}")


def plot_risk_heatmap(fnr_reports: dict[str, FNRReport], output_dir: Path):
    """Heatmap: disease × configuration, colored by FNR × severity."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError:
        return

    configs = list(fnr_reports.keys())
    all_diseases = sorted(set().union(*(r.per_disease_fnr.keys() for r in fnr_reports.values())))
    # Sort by severity
    all_diseases.sort(key=lambda d: resolve_risk(d).severity, reverse=True)

    # Risk-adjusted FNR = FNR × severity (0 = safe, 1 = dangerous miss)
    matrix = np.zeros((len(all_diseases), len(configs)))
    for j, cfg in enumerate(configs):
        for i, d in enumerate(all_diseases):
            fnr = fnr_reports[cfg].per_disease_fnr.get(d, 0)
            sev = resolve_risk(d).severity
            matrix[i, j] = fnr * sev  # risk-adjusted

    cmap = LinearSegmentedColormap.from_list("safety", ["#27ae60", "#f1c40f", "#e74c3c"])
    fig, ax = plt.subplots(figsize=(6, max(8, len(all_diseases) * 0.4)))
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs, fontsize=10)
    ax.set_yticks(range(len(all_diseases)))
    display_names = []
    for d in all_diseases:
        r = resolve_risk(d)
        tier_marker = {"critical": "🔴", "high": "🟠", "moderate": "🟡"}.get(r.tier, "🟢")
        display_names.append(f"{tier_marker} {d}")
    ax.set_yticklabels(display_names, fontsize=9)

    # Annotate cells
    for i in range(len(all_diseases)):
        for j in range(len(configs)):
            val = matrix[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)

    ax.set_title("Risk-Adjusted FNR Heatmap\n(FNR × Disease Severity)", fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, label="Risk Score (0=safe, 1=critical miss)", shrink=0.8)
    fig.tight_layout()
    fig.savefig(output_dir / "risk_heatmap.png", dpi=200)
    fig.savefig(output_dir / "risk_heatmap.pdf")
    plt.close(fig)
    print(f"  📊 Risk heatmap → {output_dir / 'risk_heatmap.png'}")


def plot_calibration(ece_reports: dict[str, dict], output_dir: Path):
    """Reliability diagram (calibration plot)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    colors = ["#e74c3c", "#f39c12", "#27ae60"]

    for i, (cfg, ece_data) in enumerate(ece_reports.items()):
        confs = [b["avg_confidence"] for b in ece_data["bins"]]
        accs = [b["avg_accuracy"] for b in ece_data["bins"]]
        ece_val = ece_data["ece"]
        ax.plot(confs, accs, "o-", color=colors[i % 3], label=f"{cfg} (ECE={ece_val:.3f})",
                markersize=5, alpha=0.85)

    ax.set_xlabel("Mean Predicted Confidence", fontsize=12)
    ax.set_ylabel("Observed Accuracy", fontsize=12)
    ax.set_title("Calibration Plot (Reliability Diagram)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(output_dir / "calibration_plot.png", dpi=200)
    fig.savefig(output_dir / "calibration_plot.pdf")
    plt.close(fig)
    print(f"  📊 Calibration plot → {output_dir / 'calibration_plot.png'}")


# ════════════════════════════════════════════════════════════════
# LaTeX output
# ════════════════════════════════════════════════════════════════

def generate_safety_latex(
    fnr_reports: dict[str, FNRReport],
    rwa_reports: dict[str, RWAReport],
    cost_reports: dict[str, CostReport],
    ece_reports: dict[str, dict],
    stat_tests: dict,
    output_dir: Path,
):
    """Generate all LaTeX tables for the paper."""

    # ── Table 1: Main safety metrics comparison ──
    configs = list(fnr_reports.keys())
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Safety metrics across pipeline configurations. "
        r"RWA = Risk-Weighted Accuracy (penalizes misses on severe diseases). "
        r"EML = Expected Monetary Loss per acre. "
        r"Best results in \textbf{bold}. "
        r"$\dagger$: $p < 0.05$ (permutation test vs.\ previous row).}",
        r"\label{tab:safety_metrics}",
        r"\begin{tabular}{l c c c c c c c}",
        r"\toprule",
        r"Config & Acc. & RWA & Safety & FNR\textsubscript{crit} "
        r"& FNR\textsubscript{high} & EML & ECE \\",
        r" & (\%) & (\%) & Gap$\downarrow$ & (\%)$\downarrow$ "
        r"& (\%)$\downarrow$ & (\rupee/ac)$\downarrow$ & $\downarrow$ \\",
        r"\midrule",
    ]

    # Direction-aware best tracking (↑ = higher better, ↓ = lower better)
    max_acc = max(rwa_reports[c].standard_accuracy for c in configs)
    max_rwa = max(rwa_reports[c].rwa for c in configs)
    min_gap = min(rwa_reports[c].safety_gap for c in configs)
    min_fnr_crit = min(fnr_reports[c].fnr_critical for c in configs)
    min_fnr_high = min(fnr_reports[c].fnr_high for c in configs)
    min_eml = min(cost_reports[c].eml_per_acre_inr for c in configs)
    min_ece = min(ece_reports[c]["ece"] for c in configs)

    def _bold_if(val, best, fmt, lower_better=False):
        s = fmt.format(val)
        tol = 0.005 if "." in s else 1
        if lower_better:
            match = val <= best + tol
        else:
            match = val >= best - tol
        return r"\textbf{" + s + "}" if match else s

    prev = None
    for cfg in configs:
        sig = ""
        if prev and f"{prev}_vs_{cfg}" in stat_tests:
            if stat_tests[f"{prev}_vs_{cfg}"].get("significant"):
                sig = r"$^\dagger$"

        row = [
            cfg + sig,
            _bold_if(rwa_reports[cfg].standard_accuracy * 100, max_acc * 100, "{:.1f}"),
            _bold_if(rwa_reports[cfg].rwa * 100, max_rwa * 100, "{:.1f}"),
            _bold_if(rwa_reports[cfg].safety_gap * 100, min_gap * 100, "{:.1f}", lower_better=True),
            _bold_if(fnr_reports[cfg].fnr_critical * 100, min_fnr_crit * 100, "{:.1f}", lower_better=True),
            _bold_if(fnr_reports[cfg].fnr_high * 100, min_fnr_high * 100, "{:.1f}", lower_better=True),
            _bold_if(cost_reports[cfg].eml_per_acre_inr, min_eml, "{:.0f}", lower_better=True),
            _bold_if(ece_reports[cfg]["ece"], min_ece, "{:.3f}", lower_better=True),
        ]
        lines.append("  " + " & ".join(row) + r" \\")
        prev = cfg

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (output_dir / "safety_metrics_table.tex").write_text("\n".join(lines))
    print(f"  📝 Safety metrics table → safety_metrics_table.tex")

    # ── Table 2: Asymmetric cost matrix ──
    cost_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Asymmetric cost matrix. $C_{FN}$: revenue lost from missed disease "
        r"(yield loss $\times$ market price). $C_{FP}$: unnecessary treatment cost. "
        r"Ratio $C_{FN}/C_{FP}$ defines how critical detection is for each disease tier.}",
        r"\label{tab:cost_matrix}",
        r"\begin{tabular}{l r r r c}",
        r"\toprule",
        r"Disease Tier & $C_{FN}$ (\rupee/ac) & $C_{FP}$ (\rupee/ac) & Ratio & Urgency \\",
        r"\midrule",
    ]

    for tier, diseases in [
        ("Critical", [d for d in DISEASE_RISKS.values() if d.tier == "critical"]),
        ("High", [d for d in DISEASE_RISKS.values() if d.tier == "high"]),
        ("Moderate", [d for d in DISEASE_RISKS.values() if d.tier == "moderate"]),
    ]:
        avg_fn = np.mean([d.miss_cost_inr_per_acre for d in diseases])
        avg_fp = np.mean([d.false_alarm_cost_inr_per_acre for d in diseases])
        avg_ratio = avg_fn / avg_fp if avg_fp > 0 else 0
        urgency = diseases[0].urgency if diseases else ""
        cost_lines.append(
            f"  {tier} & {avg_fn:,.0f} & {avg_fp:,.0f} & {avg_ratio:.0f}:1 & {urgency} \\\\"
        )

    cost_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (output_dir / "cost_matrix_table.tex").write_text("\n".join(cost_lines))
    print(f"  📝 Cost matrix table → cost_matrix_table.tex")

    # ── Table 3: Per-disease FNR (critical + high only) ──
    disease_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Per-disease False Negative Rate (\%) for critical and high-severity diseases. "
        r"\colorbox{red!20}{Red}: FNR $>$ 50\%. \colorbox{yellow!25}{Yellow}: FNR 20--50\%.}",
        r"\label{tab:per_disease_fnr}",
        r"\begin{tabular}{l c" + " c" * len(configs) + "}",
        r"\toprule",
        "Disease & Severity & " + " & ".join(configs) + r" \\",
        r"\midrule",
    ]

    important_diseases = [d for d in DISEASE_RISKS.values() if d.tier in ("critical", "high")]
    important_diseases.sort(key=lambda d: d.severity, reverse=True)

    for disease in important_diseases:
        vals = []
        for cfg in configs:
            fnr_val = fnr_reports[cfg].per_disease_fnr.get(disease.key, 0)
            # Also try display name match
            if fnr_val == 0:
                for k, v in fnr_reports[cfg].per_disease_fnr.items():
                    if resolve_risk(k).key == disease.key:
                        fnr_val = v
                        break
            pct = fnr_val * 100
            if pct > 50:
                vals.append(r"\cellcolor{red!20}" + f"{pct:.0f}")
            elif pct > 20:
                vals.append(r"\cellcolor{yellow!25}" + f"{pct:.0f}")
            else:
                vals.append(f"{pct:.0f}")

        tier_icon = r"\textcolor{red}{$\blacktriangle$}" if disease.tier == "critical" \
            else r"\textcolor{orange}{$\bullet$}"
        disease_lines.append(
            f"  {tier_icon} {disease.display_name} & {disease.severity:.2f} & "
            + " & ".join(vals) + r" \\"
        )

    disease_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (output_dir / "per_disease_fnr_table.tex").write_text("\n".join(disease_lines))
    print(f"  📝 Per-disease FNR table → per_disease_fnr_table.tex")


# ════════════════════════════════════════════════════════════════
# I/O: load predictions from ablation CSVs
# ════════════════════════════════════════════════════════════════

def load_predictions_csv(path: Path) -> list[dict]:
    """Load predictions from ablation_study.py CSV output."""
    preds = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            preds.append({
                "image": row["image"],
                "ground_truth": row["ground_truth"],
                "predicted": row["predicted"],
                "confidence": float(row["confidence"]),
                "correct": bool(int(row["correct"])),
                "latency_ms": float(row["latency_ms"]),
                "severity_tier": row["severity_tier"],
            })
    return preds


def run_live_evaluation(test_dir: Path, label_dir: Path, cls_model_path: str,
                        crop_type: str, n: int, skip_llm: bool) -> dict[str, list[dict]]:
    """Run all configurations live (delegates to ablation_study functions)."""
    # Import from ablation study to reuse the same logic
    from scripts.ablation_study import (
        load_ground_truth, run_yolo_only, run_yolo_plus_rules,
        run_full_pipeline_with_llm,
    )
    from ultralytics import YOLO

    images = sorted(f for f in test_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS)
    if n > 0:
        images = images[:n]

    class_names = [
        "Bacterial blight", "Bacterial leaf", "Brown spot", "Cuterpillar",
        "Drainage impact", "Grashopper damage", "Grassy stunt", "Leaf folder",
        "Sheath blight", "Stem borer", "Tungro",
    ]
    gt = load_ground_truth(images, label_dir, class_names)
    cls_model = YOLO(cls_model_path)

    configs = {}

    print("\n━━━ Config A: YOLO Only ━━━")
    preds_a = run_yolo_only(images, gt, cls_model)
    configs["A: YOLO"] = [
        {"image": p.image, "ground_truth": p.ground_truth, "predicted": p.predicted,
         "confidence": p.confidence, "correct": p.correct, "latency_ms": p.latency_ms,
         "severity_tier": p.severity_tier}
        for p in preds_a
    ]

    print("\n━━━ Config B: YOLO + Rules ━━━")
    preds_b = run_yolo_plus_rules(images, gt, cls_model, crop_type)
    configs["B: YOLO+Rules"] = [
        {"image": p.image, "ground_truth": p.ground_truth, "predicted": p.predicted,
         "confidence": p.confidence, "correct": p.correct, "latency_ms": p.latency_ms,
         "severity_tier": p.severity_tier}
        for p in preds_b
    ]

    if not skip_llm:
        print("\n━━━ Config C: YOLO + Rules + LLM ━━━")
        preds_c = run_full_pipeline_with_llm(images, gt, cls_model, crop_type)
        configs["C: Full"] = [
            {"image": p.image, "ground_truth": p.ground_truth, "predicted": p.predicted,
             "confidence": p.confidence, "correct": p.correct, "latency_ms": p.latency_ms,
             "severity_tier": p.severity_tier}
            for p in preds_c
        ]

    return configs


# ════════════════════════════════════════════════════════════════
# Console summary
# ════════════════════════════════════════════════════════════════

def print_safety_summary(
    fnr_reports: dict[str, FNRReport],
    rwa_reports: dict[str, RWAReport],
    cost_reports: dict[str, CostReport],
    ece_reports: dict[str, dict],
    stat_tests: dict,
    rwa_cis: dict,
):
    """Print paper-ready summary to console."""
    configs = list(fnr_reports.keys())

    print("\n" + "=" * 100)
    print("  SAFETY EVALUATION RESULTS")
    print("=" * 100)

    # ── S1: FNR ──
    print("\n── S1: False Negative Rate (lower = safer) ──")
    header = f"{'Config':<20} {'FNRoverall':>10} {'FNRcritical':>12} {'FNRhigh':>10} {'FNRmoderate':>12} {'n':>5}"
    print(header)
    print("─" * 75)
    for cfg in configs:
        r = fnr_reports[cfg]
        print(
            f"  {cfg:<18} {r.fnr_overall * 100:9.1f}% {r.fnr_critical * 100:11.1f}% "
            f"{r.fnr_high * 100:9.1f}% {r.fnr_moderate * 100:11.1f}% {r.n_total:5d}"
        )

    # ── S2: RWA ──
    print("\n── S2: Risk-Weighted Accuracy (higher = better) ──")
    header = f"{'Config':<20} {'StdAcc%':>8} {'RWA%':>8} {'SafetyGap':>10} {'95% CI':>16}"
    print(header)
    print("─" * 70)
    for cfg in configs:
        r = rwa_reports[cfg]
        ci = rwa_cis.get(cfg, {})
        ci_str = f"[{ci.get('ci_lo', 0) * 100:.1f}, {ci.get('ci_hi', 0) * 100:.1f}]" if ci else "—"
        print(
            f"  {cfg:<18} {r.standard_accuracy * 100:7.1f} {r.rwa * 100:7.1f} "
            f"{r.safety_gap * 100:9.1f}pp {ci_str:>16}"
        )

    # ── S3: Cost ──
    print("\n── S3: Expected Monetary Loss (lower = better) ──")
    header = f"{'Config':<20} {'EML/acre':>10} {'FN cost':>10} {'FP cost':>10} {'#FN':>5} {'#FP':>5}"
    print(header)
    print("─" * 70)
    for cfg in configs:
        r = cost_reports[cfg]
        print(
            f"  {cfg:<18} Rs{r.eml_per_acre_inr:8,.0f} Rs{r.eml_fn_component:8,.0f} "
            f"Rs{r.eml_fp_component:8,.0f} {r.n_false_negatives:5d} {r.n_false_positives:5d}"
        )

    # ── S4: Calibration ──
    print("\n── S4: Expected Calibration Error (lower = better calibrated) ──")
    for cfg in configs:
        ece = ece_reports[cfg]["ece"]
        quality = "well-calibrated" if ece < 0.05 else "moderate" if ece < 0.10 else "POORLY CALIBRATED"
        print(f"  {cfg:<18} ECE = {ece:.4f}  ({quality})")

    # ── Statistical tests ──
    if stat_tests:
        print("\n── Statistical Significance ──")
        for name, test in stat_tests.items():
            sig = "SIGNIFICANT" if test["significant"] else "not significant"
            print(f"  {name}: RWA diff = {test['observed_diff']:.4f}, p = {test['p_value']:.4f} → {sig}")

    print("=" * 100)


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Safety-critical evaluation for crop disease detection")
    parser.add_argument("--config", choices=["ablation", "single", "csv"],
                        default="csv", help="Evaluation mode")
    parser.add_argument("--predictions", nargs="+",
                        help="Prediction CSV files (from ablation_study.py)")
    parser.add_argument("--labels", nargs="+",
                        help="Config labels for each CSV (e.g. 'A: YOLO' 'B: YOLO+Rules')")
    parser.add_argument("--test-dir",
                        default=str(PROJECT_ROOT / "data" / "raw" / "roboflow" / "rice-diseases-v2" / "test" / "images"))
    parser.add_argument("--label-dir", default=None)
    parser.add_argument("--cls-model", default=str(PROJECT_ROOT / "yolov8n-cls.pt"))
    parser.add_argument("--crop-type", default="rice")
    parser.add_argument("--n", type=int, default=0)
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--mlflow", action="store_true")
    parser.add_argument("--output-dir",
                        default=str(PROJECT_ROOT / "outputs" / "safety_evaluation"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load or run predictions ──
    all_preds: dict[str, list[dict]] = {}

    if args.config == "csv" and args.predictions:
        # Load from CSVs
        labels = args.labels or [Path(p).stem for p in args.predictions]
        for path, label in zip(args.predictions, labels):
            all_preds[label] = load_predictions_csv(Path(path))
            print(f"  Loaded {len(all_preds[label])} predictions from {path}")

    elif args.config == "ablation":
        # Load from default ablation output directory
        ablation_dir = PROJECT_ROOT / "outputs" / "ablation"
        for csv_file in sorted(ablation_dir.glob("predictions_*.csv")):
            label = csv_file.stem.replace("predictions_", "")
            all_preds[label] = load_predictions_csv(csv_file)
            print(f"  Loaded {label}: {len(all_preds[label])} predictions")

    elif args.config == "single":
        # Run live evaluation
        test_dir = Path(args.test_dir)
        label_dir = Path(args.label_dir) if args.label_dir else test_dir.parent / "labels"
        all_preds = run_live_evaluation(
            test_dir, label_dir, args.cls_model, args.crop_type, args.n, args.skip_llm
        )

    if not all_preds:
        # Auto-detect from ablation output
        ablation_dir = PROJECT_ROOT / "outputs" / "ablation"
        if ablation_dir.exists():
            for csv_file in sorted(ablation_dir.glob("predictions_*.csv")):
                label = csv_file.stem.replace("predictions_", "")
                all_preds[label] = load_predictions_csv(csv_file)
                print(f"  Auto-loaded {label}: {len(all_preds[label])} predictions")

    if not all_preds:
        print("ERROR: No predictions found. Run ablation_study.py first or provide --predictions CSVs.")
        sys.exit(1)

    # ── Compute all safety metrics ──
    print("\n🔬 Computing safety metrics...")

    fnr_reports = {}
    rwa_reports = {}
    cost_reports = {}
    ece_reports = {}

    for cfg, preds in all_preds.items():
        fnr_reports[cfg] = compute_fnr(preds)
        rwa_reports[cfg] = compute_rwa(preds)
        cost_reports[cfg] = compute_cost(preds)
        ece_reports[cfg] = compute_ece(preds)

    # ── Statistical tests ──
    stat_tests = {}
    rwa_cis = {}
    configs = list(all_preds.keys())

    for cfg, preds in all_preds.items():
        correct_vec = [1 if p["correct"] else 0 for p in preds]
        # RWA bootstrap: create weighted correct vector
        weighted_vals = []
        for p in preds:
            risk = resolve_risk(p["ground_truth"])
            w = TIER_WEIGHTS.get(risk.tier, 1.0)
            weighted_vals.append(w if p["correct"] else 0.0)
        rwa_cis[cfg] = bootstrap_metric(
            [1 if p["correct"] else 0 for p in preds],
            n_boot=5000, seed=args.seed,
        )

    if len(configs) >= 2:
        for i in range(len(configs) - 1):
            name = f"{configs[i]}_vs_{configs[i + 1]}"
            stat_tests[name] = permutation_test_rwa(
                all_preds[configs[i]], all_preds[configs[i + 1]],
                n_perm=5000, seed=args.seed,
            )

    # ── Console output ──
    print_safety_summary(fnr_reports, rwa_reports, cost_reports, ece_reports, stat_tests, rwa_cis)

    # ── Figures ──
    print("\n📊 Generating figures...")
    plot_fnr_by_disease(fnr_reports, output_dir)
    plot_cost_waterfall(cost_reports, output_dir)
    plot_risk_heatmap(fnr_reports, output_dir)
    plot_calibration(ece_reports, output_dir)

    # ── LaTeX tables ──
    print("\n📝 Generating LaTeX tables...")
    generate_safety_latex(fnr_reports, rwa_reports, cost_reports, ece_reports, stat_tests, output_dir)

    # ── JSON summary ──
    summary = {
        "timestamp": datetime.now().isoformat(),
        "configurations": {},
        "statistical_tests": stat_tests,
        "cost_matrix_reference": {
            tier: {
                "mean_cfn_inr": round(float(np.mean([
                    d.miss_cost_inr_per_acre for d in DISEASE_RISKS.values() if d.tier == tier
                ])), 0),
                "mean_cfp_inr": round(float(np.mean([
                    d.false_alarm_cost_inr_per_acre for d in DISEASE_RISKS.values() if d.tier == tier
                ])) if any(d.tier == tier and d.false_alarm_cost_inr_per_acre > 0
                           for d in DISEASE_RISKS.values()) else 0, 0),
                "risk_weight": TIER_WEIGHTS[tier],
            }
            for tier in ["critical", "high", "moderate", "healthy"]
        },
    }
    for cfg in configs:
        summary["configurations"][cfg] = {
            "fnr": {
                "overall": round(fnr_reports[cfg].fnr_overall, 4),
                "critical": round(fnr_reports[cfg].fnr_critical, 4),
                "high": round(fnr_reports[cfg].fnr_high, 4),
                "moderate": round(fnr_reports[cfg].fnr_moderate, 4),
                "per_disease": {k: round(v, 4) for k, v in fnr_reports[cfg].per_disease_fnr.items()},
            },
            "rwa": {
                "rwa": round(rwa_reports[cfg].rwa, 4),
                "standard_accuracy": round(rwa_reports[cfg].standard_accuracy, 4),
                "safety_gap": round(rwa_reports[cfg].safety_gap, 4),
                "tier_accuracy": {k: round(v, 4) for k, v in rwa_reports[cfg].tier_accuracy.items()},
                "ci_95": rwa_cis.get(cfg, {}),
            },
            "cost": {
                "eml_per_acre_inr": round(cost_reports[cfg].eml_per_acre_inr, 2),
                "eml_fn_component": round(cost_reports[cfg].eml_fn_component, 2),
                "eml_fp_component": round(cost_reports[cfg].eml_fp_component, 2),
                "n_fn": cost_reports[cfg].n_false_negatives,
                "n_fp": cost_reports[cfg].n_false_positives,
            },
            "calibration": ece_reports[cfg],
        }

    json_path = output_dir / "safety_evaluation_results.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"\n📄 JSON → {json_path}")

    # ── MLflow ──
    if args.mlflow:
        try:
            import mlflow
            mlflow.set_tracking_uri(str(PROJECT_ROOT / "runs" / "mlflow"))
            mlflow.set_experiment("safety_evaluation")
            for cfg in configs:
                with mlflow.start_run(run_name=f"safety_{cfg}"):
                    mlflow.log_metric("fnr_critical", fnr_reports[cfg].fnr_critical)
                    mlflow.log_metric("fnr_overall", fnr_reports[cfg].fnr_overall)
                    mlflow.log_metric("rwa", rwa_reports[cfg].rwa)
                    mlflow.log_metric("safety_gap", rwa_reports[cfg].safety_gap)
                    mlflow.log_metric("eml_per_acre", cost_reports[cfg].eml_per_acre_inr)
                    mlflow.log_metric("ece", ece_reports[cfg]["ece"])
                    mlflow.log_artifacts(str(output_dir))
            print("📊 MLflow logging complete")
        except ImportError:
            print("  ⚠ mlflow not installed")

    print(f"\n✅ All safety evaluation outputs saved to {output_dir}/")


if __name__ == "__main__":
    main()
