#!/usr/bin/env python3
"""
Experiment 2 — Statistical Significance Tests

Provides:
  1. Bootstrap confidence intervals (n=10,000) for accuracy, macro-F1, MCC
  2. McNemar's test (chi-squared) between any two configurations
  3. Per-class bootstrap CI on F1

Reads the per-image prediction CSVs produced by ablation_study.py:
  evaluate/results/predictions_A_yolo_only.csv
  evaluate/results/predictions_B_yolo_rules.csv
  evaluate/results/predictions_C_rules_only.csv

Usage:
    python evaluate/statistical_tests.py
    python evaluate/statistical_tests.py --results-dir evaluate/results --n-boot 10000
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ════════════════════════════════════════════════════════════════
# Data loading
# ════════════════════════════════════════════════════════════════

def load_predictions(csv_path: Path) -> list[dict]:
    """Load per-image predictions from ablation CSV."""
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "image": row["image"],
                "ground_truth": row["ground_truth"],
                "predicted": row["predicted"],
                "confidence": float(row["confidence"]),
                "correct": int(row["correct"]),
                "latency_ms": float(row["latency_ms"]),
            })
    return rows


# ════════════════════════════════════════════════════════════════
# Metric computation helpers
# ════════════════════════════════════════════════════════════════

def accuracy(preds: list[dict]) -> float:
    if not preds:
        return 0.0
    return sum(p["correct"] for p in preds) / len(preds)


def macro_f1(preds: list[dict]) -> float:
    tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)
    classes = set()
    for p in preds:
        gt, pr = p["ground_truth"], p["predicted"]
        classes.add(gt)
        if gt == pr:
            tp[gt] += 1
        else:
            fn[gt] += 1
            fp[pr] += 1

    f1_scores = []
    for cls in classes:
        prec = tp[cls] / (tp[cls] + fp[cls]) if (tp[cls] + fp[cls]) > 0 else 0
        rec = tp[cls] / (tp[cls] + fn[cls]) if (tp[cls] + fn[cls]) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        f1_scores.append(f1)
    return np.mean(f1_scores) if f1_scores else 0.0


def mcc_multiclass(preds: list[dict]) -> float:
    """Multi-class Matthews Correlation Coefficient via confusion matrix."""
    classes = sorted(set(p["ground_truth"] for p in preds))
    idx_map = {c: i for i, c in enumerate(classes)}
    k = len(classes)
    cm = np.zeros((k, k), dtype=float)
    for p in preds:
        gi = idx_map.get(p["ground_truth"], -1)
        pi = idx_map.get(p["predicted"], -1)
        if gi >= 0 and pi >= 0:
            cm[gi, pi] += 1

    c = cm.sum()
    s = cm.sum(axis=1)
    t = cm.sum(axis=0)
    pk = np.diag(cm)
    cov_xy = c * pk.sum() - np.dot(s, t)
    cov_xx = c * c - np.dot(s, s)
    cov_yy = c * c - np.dot(t, t)
    denom = np.sqrt(float(cov_xx) * float(cov_yy))
    return float(cov_xy) / denom if denom > 0 else 0.0


def per_class_f1(preds: list[dict]) -> dict[str, float]:
    tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)
    for p in preds:
        gt, pr = p["ground_truth"], p["predicted"]
        if gt == pr:
            tp[gt] += 1
        else:
            fn[gt] += 1
            fp[pr] += 1

    classes = sorted(set(p["ground_truth"] for p in preds))
    result = {}
    for cls in classes:
        prec = tp[cls] / (tp[cls] + fp[cls]) if (tp[cls] + fp[cls]) > 0 else 0
        rec = tp[cls] / (tp[cls] + fn[cls]) if (tp[cls] + fn[cls]) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        result[cls] = f1
    return result


# ════════════════════════════════════════════════════════════════
# Bootstrap Confidence Intervals
# ════════════════════════════════════════════════════════════════

def bootstrap_ci(preds: list[dict], metric_fn, n_boot: int = 10000,
                 alpha: float = 0.05, seed: int = 42) -> dict:
    """Compute bootstrap (1-alpha)% CI for a given metric function.

    Returns: {point, ci_lower, ci_upper, se, n_boot}
    """
    rng = np.random.RandomState(seed)
    n = len(preds)
    point_estimate = metric_fn(preds)

    boot_values = np.empty(n_boot)
    for b in range(n_boot):
        indices = rng.randint(0, n, size=n)
        boot_sample = [preds[i] for i in indices]
        boot_values[b] = metric_fn(boot_sample)

    lo = np.percentile(boot_values, 100 * alpha / 2)
    hi = np.percentile(boot_values, 100 * (1 - alpha / 2))
    se = np.std(boot_values, ddof=1)

    return {
        "point": round(point_estimate, 4),
        "ci_lower": round(float(lo), 4),
        "ci_upper": round(float(hi), 4),
        "se": round(float(se), 4),
        "n_boot": n_boot,
        "alpha": alpha,
    }


def bootstrap_per_class_f1(preds: list[dict], n_boot: int = 10000,
                           alpha: float = 0.05, seed: int = 42) -> dict:
    """Bootstrap CI on per-class F1 scores."""
    rng = np.random.RandomState(seed)
    n = len(preds)
    classes = sorted(set(p["ground_truth"] for p in preds))
    point_f1 = per_class_f1(preds)

    # Collect bootstrap samples
    boot_f1s = {cls: [] for cls in classes}
    for b in range(n_boot):
        indices = rng.randint(0, n, size=n)
        boot_sample = [preds[i] for i in indices]
        bf1 = per_class_f1(boot_sample)
        for cls in classes:
            boot_f1s[cls].append(bf1.get(cls, 0.0))

    result = {}
    for cls in classes:
        vals = np.array(boot_f1s[cls])
        result[cls] = {
            "point": round(point_f1.get(cls, 0.0), 4),
            "ci_lower": round(float(np.percentile(vals, 100 * alpha / 2)), 4),
            "ci_upper": round(float(np.percentile(vals, 100 * (1 - alpha / 2))), 4),
            "se": round(float(np.std(vals, ddof=1)), 4),
        }
    return result


# ════════════════════════════════════════════════════════════════
# McNemar's Test
# ════════════════════════════════════════════════════════════════

def mcnemar_test(preds_x: list[dict], preds_y: list[dict]) -> dict:
    """McNemar's chi-squared test with continuity correction.

    Compares two classifiers on the SAME test set.
    H0: both classifiers have the same error rate.

    Contingency table:
                    Y correct    Y wrong
        X correct    n11          n10
        X wrong      n01          n00

    Statistic = (|n01 - n10| - 1)^2 / (n01 + n10)  [with continuity correction]

    Returns: {n01, n10, n11, n00, chi2, p_value, significant_005}
    """
    assert len(preds_x) == len(preds_y), "Prediction lists must have same length"

    n11, n10, n01, n00 = 0, 0, 0, 0
    for px, py in zip(preds_x, preds_y):
        x_ok = px["correct"]
        y_ok = py["correct"]
        if x_ok and y_ok:
            n11 += 1
        elif x_ok and not y_ok:
            n10 += 1
        elif not x_ok and y_ok:
            n01 += 1
        else:
            n00 += 1

    # McNemar statistic with continuity correction
    discordant = n01 + n10
    if discordant == 0:
        chi2 = 0.0
        p_value = 1.0
    else:
        chi2 = (abs(n01 - n10) - 1) ** 2 / discordant
        # p-value from chi-squared distribution (1 df)
        # Use survival function: P(X > chi2) for chi2(1)
        try:
            from scipy.stats import chi2 as chi2_dist
            p_value = float(chi2_dist.sf(chi2, df=1))
        except ImportError:
            # Fallback: approximate using normal distribution
            z = np.sqrt(chi2)
            # Two-tailed p via error function approximation
            p_value = float(2 * (1 - _norm_cdf(z)))

    return {
        "n11_both_correct": n11,
        "n10_x_correct_y_wrong": n10,
        "n01_x_wrong_y_correct": n01,
        "n00_both_wrong": n00,
        "discordant_pairs": discordant,
        "chi2_statistic": round(chi2, 4),
        "p_value": round(p_value, 6),
        "significant_005": p_value < 0.05,
        "significant_001": p_value < 0.01,
    }


def _norm_cdf(x: float) -> float:
    """Standard normal CDF approximation (Abramowitz & Stegun)."""
    import math
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Experiment 2: Statistical tests")
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "evaluate" / "results"))
    parser.add_argument("--n-boot", type=int, default=10000, help="Bootstrap iterations")
    parser.add_argument("--alpha", type=float, default=0.05, help="CI significance level")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--v2",
        action="store_true",
        help=(
            "Route to the v2 statistical protocol (per-class bootstrap, "
            "Holm-Bonferroni, Dietterich 5x2cv, Friedman-Nemenyi). Writes to "
            "evaluate/results/v2/statistics/. Does not touch v1 artifacts."
        ),
    )
    args = parser.parse_args()

    if args.v2:
        # Lazy import keeps v1 path dependency-free.
        from statistical_tests_v2 import run_v2  # type: ignore
        return run_v2(
            results_dir=Path(args.results_dir),
            n_boot=args.n_boot,
            alpha=args.alpha,
            seed=args.seed,
        )

    results_dir = Path(args.results_dir)
    output = {}

    # ── Load predictions ──
    configs = {}
    config_files = {
        "A": "predictions_A_yolo_only.csv",
        "B": "predictions_B_yolo_rules.csv",
        "C": "predictions_C_rules_only.csv",
    }
    for name, fname in config_files.items():
        fpath = results_dir / fname
        if fpath.exists():
            configs[name] = load_predictions(fpath)
            print(f"  Loaded Config {name}: {len(configs[name])} predictions from {fname}")
        else:
            print(f"  WARNING: {fname} not found — skipping Config {name}")

    if not configs:
        print("ERROR: No prediction files found. Run ablation_study.py first.")
        sys.exit(1)

    # ── Bootstrap CIs ──
    print(f"\n{'='*70}")
    print(f"  BOOTSTRAP CONFIDENCE INTERVALS (n={args.n_boot}, alpha={args.alpha})")
    print(f"{'='*70}")

    metrics = {"accuracy": accuracy, "macro_f1": macro_f1, "mcc": mcc_multiclass}
    output["bootstrap_ci"] = {}

    for cfg_name, preds in sorted(configs.items()):
        output["bootstrap_ci"][cfg_name] = {}
        print(f"\n  Config {cfg_name}:")
        for metric_name, metric_fn in metrics.items():
            ci = bootstrap_ci(preds, metric_fn, n_boot=args.n_boot,
                              alpha=args.alpha, seed=args.seed)
            output["bootstrap_ci"][cfg_name][metric_name] = ci
            print(f"    {metric_name:12s}: {ci['point']:.4f}  "
                  f"[{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]  "
                  f"SE={ci['se']:.4f}")

    # ── Per-class F1 CIs ──
    print(f"\n{'='*70}")
    print(f"  PER-CLASS F1 BOOTSTRAP CIs")
    print(f"{'='*70}")

    output["per_class_f1_ci"] = {}
    for cfg_name, preds in sorted(configs.items()):
        pcf1 = bootstrap_per_class_f1(preds, n_boot=args.n_boot,
                                       alpha=args.alpha, seed=args.seed)
        output["per_class_f1_ci"][cfg_name] = pcf1
        print(f"\n  Config {cfg_name}:")
        for cls, ci in sorted(pcf1.items()):
            width = ci["ci_upper"] - ci["ci_lower"]
            print(f"    {cls:40s}: {ci['point']:.3f}  "
                  f"[{ci['ci_lower']:.3f}, {ci['ci_upper']:.3f}]  "
                  f"width={width:.3f}")

    # ── McNemar's tests (all pairwise) ──
    print(f"\n{'='*70}")
    print(f"  McNEMAR'S TEST (pairwise)")
    print(f"{'='*70}")

    output["mcnemar"] = {}
    config_names = sorted(configs.keys())
    for i, cx in enumerate(config_names):
        for cy in config_names[i + 1:]:
            key = f"{cx}_vs_{cy}"
            result = mcnemar_test(configs[cx], configs[cy])
            output["mcnemar"][key] = result
            sig_marker = "***" if result["significant_001"] else ("**" if result["significant_005"] else "ns")
            print(f"\n  {cx} vs {cy}:")
            print(f"    Discordant pairs: {result['discordant_pairs']} "
                  f"(n01={result['n01_x_wrong_y_correct']}, n10={result['n10_x_correct_y_wrong']})")
            print(f"    chi2 = {result['chi2_statistic']:.4f}, "
                  f"p = {result['p_value']:.6f} {sig_marker}")

    # ── Save results ──
    out_path = results_dir / "statistical_tests.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\n  Results saved → {out_path}")

    # ── LaTeX snippet ──
    latex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Bootstrap 95\% confidence intervals ($B=10{,}000$) and McNemar's test.}",
        r"\label{tab:statistical}",
        r"\begin{tabular}{l l r @{\,}c@{\,} r r}",
        r"\toprule",
        r"\textbf{Config} & \textbf{Metric} & \multicolumn{3}{c}{\textbf{95\% CI}} & \textbf{Point} \\",
        r"\midrule",
    ]
    for cfg_name in sorted(output["bootstrap_ci"].keys()):
        for metric_name, ci in output["bootstrap_ci"][cfg_name].items():
            latex_lines.append(
                f"  {cfg_name} & {metric_name.replace('_', ' ')} & "
                f"[{ci['ci_lower']:.4f} & -- & {ci['ci_upper']:.4f}] & {ci['point']:.4f} \\\\"
            )
        latex_lines.append(r"\midrule")

    # McNemar rows
    if output["mcnemar"]:
        latex_lines.append(r"\multicolumn{6}{l}{\textit{McNemar's $\chi^2$ test (continuity-corrected)}} \\")
        latex_lines.append(r"\midrule")
        for key, result in output["mcnemar"].items():
            sig = "\\checkmark" if result["significant_005"] else "$\\times$"
            latex_lines.append(
                f"  \\multicolumn{{2}}{{l}}{{{key.replace('_', ' ')}}} & "
                f"\\multicolumn{{3}}{{c}}{{$\\chi^2={result['chi2_statistic']:.2f}$, "
                f"$p={result['p_value']:.4f}$}} & {sig} \\\\"
            )

    latex_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    latex_path = results_dir / "statistical_tests_latex.tex"
    latex_path.write_text("\n".join(latex_lines), encoding="utf-8")
    print(f"  LaTeX table → {latex_path}")


if __name__ == "__main__":
    main()
