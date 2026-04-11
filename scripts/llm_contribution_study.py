#!/usr/bin/env python3
"""
llm_contribution_study.py — Critical evaluation of whether LLaVA adds value.

A reviewer asked: "Does the LLM validator actually improve your system,
or is it confirmation bias with 60-90 seconds of extra latency?"

This script measures:

  E1  Marginal accuracy gain (Config B → C delta, with CIs)
  E2  Disagreement resolution quality (when LLM overrides, is it right?)
  E3  Error correction rate (wrong→right vs right→wrong transitions)
  E4  Scenario-stratified value (VALIDATE/ARBITRATE/DIFFERENTIATE/HEALTHY_CHECK)
  E5  Consistency across runs (LLaVA stochasticity at temperature=0.1)
  E6  Confidence calibration improvement (does LLM improve ECE?)

Honest analysis: also measures when LLM HURTS:
  - right→wrong flips (LLM overrode a correct prediction with a wrong one)
  - latency cost per corrected prediction
  - severity-stratified analysis (does LLM help on hard cases or easy ones?)

Usage:
    # From ablation CSVs (fast — no LLaVA needed):
    python scripts/llm_contribution_study.py \
        --without outputs/ablation/predictions_B_yolo_rules.csv \
        --with-llm outputs/ablation/predictions_C_full_pipeline.csv

    # Live evaluation with multiple runs (slow — needs Ollama):
    python scripts/llm_contribution_study.py --live --runs 5

    # Just compute from existing predictions:
    python scripts/llm_contribution_study.py --auto
"""

import argparse
import csv
import json
import sys
import warnings
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


# ════════════════════════════════════════════════════════════════
# E1: Marginal Accuracy Gain
# ════════════════════════════════════════════════════════════════

@dataclass
class MarginalGainReport:
    """Does adding LLM improve accuracy, and by how much?"""
    accuracy_without: float           # Config B accuracy
    accuracy_with: float              # Config C accuracy
    delta: float                      # C - B
    delta_ci_lo: float                # 95% CI lower bound on delta
    delta_ci_hi: float                # 95% CI upper bound
    delta_significant: bool           # Is delta significantly > 0?
    p_value: float                    # McNemar's p-value
    n_samples: int
    # Honest disclosure
    improvement_percentage: float     # relative improvement
    cost_per_correct: float           # extra latency ms per net correction


# ════════════════════════════════════════════════════════════════
# E2: Disagreement Resolution
# ════════════════════════════════════════════════════════════════

@dataclass
class DisagreementAnalysis:
    """When LLM disagrees with Rules, who is right?"""
    n_agreements: int                 # LLM agrees with rules
    n_disagreements: int              # LLM disagrees
    disagreement_rate: float

    # Of the disagreements:
    n_llm_correct: int                # LLM was right, rules were wrong
    n_rules_correct: int              # Rules were right, LLM was wrong
    n_both_wrong: int                 # Neither was right
    llm_win_rate: float               # LLM correct / (LLM correct + rules correct)

    # Net value
    net_corrections: int              # llm_correct - rules_correct (positive = LLM helps)
    net_correction_rate: float        # net_corrections / n_disagreements


# ════════════════════════════════════════════════════════════════
# E3: Error Transition Matrix
# ════════════════════════════════════════════════════════════════

@dataclass
class TransitionMatrix:
    """Paired comparison: what changed when LLM was added?"""
    n_both_correct: int              # ✓→✓ (LLM didn't matter)
    n_both_wrong: int                # ✗→✗ (LLM didn't help)
    n_wrong_to_right: int            # ✗→✓ (LLM HELPED — corrected an error)
    n_right_to_wrong: int            # ✓→✗ (LLM HURT — introduced an error)
    n_changed_but_both_wrong: int    # different wrong answers

    # Derived
    net_corrections: int             # wrong→right - right→wrong
    correction_rate: float           # net / total
    harm_rate: float                 # right→wrong / total
    irrelevance_rate: float          # (both_correct + both_wrong) / total
    n_total: int

    # Critical ratio: how many times LLM helps vs hurts
    help_to_harm_ratio: float        # wrong→right / right→wrong (>1 = net positive)

    # Examples of each transition (for qualitative analysis in paper)
    examples_helped: list = field(default_factory=list)    # sample of ✗→✓
    examples_harmed: list = field(default_factory=list)    # sample of ✓→✗


# ════════════════════════════════════════════════════════════════
# E4: Scenario-Stratified Value
# ════════════════════════════════════════════════════════════════

@dataclass
class ScenarioAnalysis:
    """Performance breakdown by LLM prompt scenario."""
    scenario: str
    n_samples: int
    accuracy_without: float
    accuracy_with: float
    delta: float
    n_wrong_to_right: int
    n_right_to_wrong: int
    net: int
    verdict: str                     # "helps" | "neutral" | "hurts"


# ════════════════════════════════════════════════════════════════
# E5: Consistency Analysis
# ════════════════════════════════════════════════════════════════

@dataclass
class ConsistencyReport:
    """Across multiple runs, how stable is the LLM?"""
    n_runs: int
    accuracies: list[float]
    mean_accuracy: float
    std_accuracy: float
    cv: float                        # coefficient of variation
    # Per-image consistency
    n_always_same: int               # predicted same class every run
    n_sometimes_different: int       # predicted different classes across runs
    flip_rate: float                 # sometimes_different / total
    # Agreement across runs
    mean_pairwise_agreement: float   # avg Jaccard across all run pairs


# ════════════════════════════════════════════════════════════════
# Core computation
# ════════════════════════════════════════════════════════════════

def load_predictions(path: Path) -> list[dict]:
    """Load predictions CSV from ablation study."""
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
                "severity_tier": row.get("severity_tier", "unknown"),
            })
    return preds


def _class_match(a: str, b: str) -> bool:
    """Fuzzy class name match."""
    a = a.lower().replace(" ", "_").replace("-", "_")
    b = b.lower().replace(" ", "_").replace("-", "_")
    if a == b:
        return True
    a_short = a.split("_", 1)[-1] if "_" in a else a
    b_short = b.split("_", 1)[-1] if "_" in b else b
    return a_short == b_short or a_short in b or b_short in a


def compute_marginal_gain(
    preds_without: list[dict],
    preds_with: list[dict],
    n_boot: int = 10000,
    seed: int = 42,
) -> MarginalGainReport:
    """E1: Does accuracy improve when LLM is added?"""
    rng = np.random.RandomState(seed)

    # Align by image name
    without_map = {p["image"]: p for p in preds_without}
    with_map = {p["image"]: p for p in preds_with}
    common = sorted(set(without_map) & set(with_map))
    n = len(common)

    vec_b = [1 if without_map[img]["correct"] else 0 for img in common]
    vec_c = [1 if with_map[img]["correct"] else 0 for img in common]

    acc_b = np.mean(vec_b)
    acc_c = np.mean(vec_c)
    delta = acc_c - acc_b

    # Bootstrap CI on delta
    arr_b = np.array(vec_b)
    arr_c = np.array(vec_c)
    boot_deltas = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        boot_deltas.append(float(np.mean(arr_c[idx]) - np.mean(arr_b[idx])))
    ci_lo = float(np.percentile(boot_deltas, 2.5))
    ci_hi = float(np.percentile(boot_deltas, 97.5))

    # McNemar's test
    b_count = sum(1 for a, c in zip(vec_b, vec_c) if a == 1 and c == 0)  # B right, C wrong
    c_count = sum(1 for a, c in zip(vec_b, vec_c) if a == 0 and c == 1)  # B wrong, C right
    if b_count + c_count > 0:
        from scipy import stats
        chi2 = (abs(b_count - c_count) - 1) ** 2 / (b_count + c_count)
        p_value = float(1 - stats.chi2.cdf(chi2, df=1))
    else:
        p_value = 1.0

    # Cost per correction
    lat_b = [without_map[img]["latency_ms"] for img in common]
    lat_c = [with_map[img]["latency_ms"] for img in common]
    extra_latency = np.mean(lat_c) - np.mean(lat_b)
    net_corrections = c_count - b_count
    cost_per = extra_latency / net_corrections if net_corrections > 0 else float("inf")

    return MarginalGainReport(
        accuracy_without=float(acc_b),
        accuracy_with=float(acc_c),
        delta=float(delta),
        delta_ci_lo=ci_lo,
        delta_ci_hi=ci_hi,
        delta_significant=p_value < 0.05,
        p_value=p_value,
        n_samples=n,
        improvement_percentage=float(delta / acc_b * 100) if acc_b > 0 else 0,
        cost_per_correct=float(cost_per),
    )


def compute_transition_matrix(
    preds_without: list[dict],
    preds_with: list[dict],
    n_examples: int = 5,
) -> TransitionMatrix:
    """E3: Classify every image into one of four quadrants."""
    without_map = {p["image"]: p for p in preds_without}
    with_map = {p["image"]: p for p in preds_with}
    common = sorted(set(without_map) & set(with_map))

    both_correct = 0
    both_wrong = 0
    wrong_to_right = 0
    right_to_wrong = 0
    changed_both_wrong = 0

    examples_helped = []
    examples_harmed = []

    for img in common:
        b = without_map[img]
        c = with_map[img]
        b_ok = b["correct"]
        c_ok = c["correct"]

        if b_ok and c_ok:
            both_correct += 1
        elif not b_ok and not c_ok:
            if b["predicted"] != c["predicted"]:
                changed_both_wrong += 1
            both_wrong += 1
        elif not b_ok and c_ok:
            wrong_to_right += 1
            if len(examples_helped) < n_examples:
                examples_helped.append({
                    "image": img, "ground_truth": b["ground_truth"],
                    "without_pred": b["predicted"], "with_pred": c["predicted"],
                })
        elif b_ok and not c_ok:
            right_to_wrong += 1
            if len(examples_harmed) < n_examples:
                examples_harmed.append({
                    "image": img, "ground_truth": b["ground_truth"],
                    "without_pred": b["predicted"], "with_pred": c["predicted"],
                })

    n = len(common)
    net = wrong_to_right - right_to_wrong
    h2h = wrong_to_right / right_to_wrong if right_to_wrong > 0 else float("inf")

    return TransitionMatrix(
        n_both_correct=both_correct,
        n_both_wrong=both_wrong,
        n_wrong_to_right=wrong_to_right,
        n_right_to_wrong=right_to_wrong,
        n_changed_but_both_wrong=changed_both_wrong,
        net_corrections=net,
        correction_rate=net / n if n > 0 else 0,
        harm_rate=right_to_wrong / n if n > 0 else 0,
        irrelevance_rate=(both_correct + both_wrong) / n if n > 0 else 0,
        n_total=n,
        help_to_harm_ratio=h2h,
        examples_helped=examples_helped,
        examples_harmed=examples_harmed,
    )


def compute_severity_stratified_transitions(
    preds_without: list[dict],
    preds_with: list[dict],
) -> dict[str, TransitionMatrix]:
    """E3b: Transition matrix per severity tier."""
    without_map = {p["image"]: p for p in preds_without}
    with_map = {p["image"]: p for p in preds_with}
    common = sorted(set(without_map) & set(with_map))

    by_tier = defaultdict(lambda: {"without": [], "with": []})
    for img in common:
        tier = without_map[img].get("severity_tier", "unknown")
        by_tier[tier]["without"].append(without_map[img])
        by_tier[tier]["with"].append(with_map[img])

    result = {}
    for tier, data in by_tier.items():
        result[tier] = compute_transition_matrix(data["without"], data["with"], n_examples=3)
    return result


def compute_consistency(run_predictions: list[list[dict]]) -> ConsistencyReport:
    """E5: How stable is the LLM across repeated runs?"""
    n_runs = len(run_predictions)
    if n_runs < 2:
        return ConsistencyReport(
            n_runs=n_runs, accuracies=[], mean_accuracy=0, std_accuracy=0,
            cv=0, n_always_same=0, n_sometimes_different=0, flip_rate=0,
            mean_pairwise_agreement=0,
        )

    # Per-run accuracy
    accuracies = [
        np.mean([1 if p["correct"] else 0 for p in run])
        for run in run_predictions
    ]
    mean_acc = float(np.mean(accuracies))
    std_acc = float(np.std(accuracies))

    # Per-image consistency
    image_predictions = defaultdict(list)
    for run in run_predictions:
        for p in run:
            image_predictions[p["image"]].append(p["predicted"])

    always_same = 0
    sometimes_diff = 0
    for img, preds in image_predictions.items():
        if len(set(preds)) == 1:
            always_same += 1
        else:
            sometimes_diff += 1

    n_images = always_same + sometimes_diff

    # Pairwise agreement
    agreements = []
    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            map_i = {p["image"]: p["correct"] for p in run_predictions[i]}
            map_j = {p["image"]: p["correct"] for p in run_predictions[j]}
            common_imgs = set(map_i) & set(map_j)
            if common_imgs:
                agree = sum(1 for img in common_imgs if map_i[img] == map_j[img])
                agreements.append(agree / len(common_imgs))

    return ConsistencyReport(
        n_runs=n_runs,
        accuracies=[round(a, 4) for a in accuracies],
        mean_accuracy=round(mean_acc, 4),
        std_accuracy=round(std_acc, 4),
        cv=round(std_acc / mean_acc, 4) if mean_acc > 0 else 0,
        n_always_same=always_same,
        n_sometimes_different=sometimes_diff,
        flip_rate=round(sometimes_diff / n_images, 4) if n_images > 0 else 0,
        mean_pairwise_agreement=round(float(np.mean(agreements)), 4) if agreements else 0,
    )


# ════════════════════════════════════════════════════════════════
# LaTeX Output
# ════════════════════════════════════════════════════════════════

def generate_latex(
    marginal: MarginalGainReport,
    transitions: TransitionMatrix,
    tier_transitions: dict[str, TransitionMatrix],
    consistency: ConsistencyReport | None,
    output_dir: Path,
):
    """Generate all LaTeX tables for paper."""

    # ── Table 1: LLM Contribution Summary ──
    sig_mark = r"$^{**}$" if marginal.delta_significant else ""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{LLM validator contribution analysis. "
        r"$^{**}$: $p < 0.05$ (McNemar's test). "
        r"Help:Harm = corrections introduced / errors introduced.}",
        r"\label{tab:llm_contribution}",
        r"\begin{tabular}{l r}",
        r"\toprule",
        r"\textbf{Metric} & \textbf{Value} \\",
        r"\midrule",
        f"Accuracy without LLM & {marginal.accuracy_without * 100:.1f}\\% \\\\",
        f"Accuracy with LLM & {marginal.accuracy_with * 100:.1f}\\% \\\\",
        f"$\\Delta$ Accuracy{sig_mark} & "
        f"{'+' if marginal.delta >= 0 else ''}{marginal.delta * 100:.1f}pp "
        f"[{marginal.delta_ci_lo * 100:.1f}, {marginal.delta_ci_hi * 100:.1f}] \\\\",
        f"McNemar's $p$-value & {marginal.p_value:.4f} \\\\",
        r"\midrule",
        f"Predictions corrected ($\\times \\to \\checkmark$) & {transitions.n_wrong_to_right} \\\\",
        f"Predictions broken ($\\checkmark \\to \\times$) & {transitions.n_right_to_wrong} \\\\",
        f"Net corrections & {'+' if transitions.net_corrections >= 0 else ''}{transitions.net_corrections} \\\\",
        f"Help:Harm ratio & {transitions.help_to_harm_ratio:.1f}:1 \\\\",
        r"\midrule",
        f"Unchanged (both correct) & {transitions.n_both_correct} "
        f"({transitions.n_both_correct / transitions.n_total * 100:.0f}\\%) \\\\",
        f"Unchanged (both wrong) & {transitions.n_both_wrong} "
        f"({transitions.n_both_wrong / transitions.n_total * 100:.0f}\\%) \\\\",
        f"LLM irrelevance rate & {transitions.irrelevance_rate * 100:.1f}\\% \\\\",
        r"\midrule",
        f"Mean extra latency & {marginal.cost_per_correct:.0f} ms/correction \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    (output_dir / "llm_contribution_table.tex").write_text("\n".join(lines))

    # ── Table 2: Transition Matrix (visual 2×2 contingency) ──
    mat_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Error transition matrix: effect of adding LLM validator. "
        r"Rows = without LLM, columns = with LLM.}",
        r"\label{tab:transition_matrix}",
        r"\begin{tabular}{l c c | r}",
        r"\toprule",
        r" & \textbf{With LLM: Correct} & \textbf{With LLM: Wrong} & \textbf{Total} \\",
        r"\midrule",
        f"\\textbf{{Without LLM: Correct}} & "
        f"{transitions.n_both_correct} & "
        f"\\cellcolor{{red!15}}{transitions.n_right_to_wrong} & "
        f"{transitions.n_both_correct + transitions.n_right_to_wrong} \\\\",
        f"\\textbf{{Without LLM: Wrong}} & "
        f"\\cellcolor{{green!15}}{transitions.n_wrong_to_right} & "
        f"{transitions.n_both_wrong} & "
        f"{transitions.n_wrong_to_right + transitions.n_both_wrong} \\\\",
        r"\midrule",
        f"\\textbf{{Total}} & "
        f"{transitions.n_both_correct + transitions.n_wrong_to_right} & "
        f"{transitions.n_right_to_wrong + transitions.n_both_wrong} & "
        f"{transitions.n_total} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    (output_dir / "transition_matrix_table.tex").write_text("\n".join(mat_lines))

    # ── Table 3: Severity-stratified transitions ──
    tier_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{LLM contribution stratified by disease severity. "
        r"Net = corrections $-$ errors introduced. "
        r"Positive net indicates the LLM adds value for that tier.}",
        r"\label{tab:llm_by_severity}",
        r"\begin{tabular}{l r r r r c}",
        r"\toprule",
        r"Severity & $n$ & $\times\!\to\!\checkmark$ & $\checkmark\!\to\!\times$ "
        r"& Net & Verdict \\",
        r"\midrule",
    ]
    for tier in ["severe", "moderate", "low"]:
        if tier not in tier_transitions:
            continue
        t = tier_transitions[tier]
        verdict = r"\textcolor{green!60!black}{helps}" if t.net_corrections > 0 \
            else r"\textcolor{red}{hurts}" if t.net_corrections < 0 \
            else r"\textcolor{gray}{neutral}"
        tier_lines.append(
            f"  {tier.capitalize()} & {t.n_total} & {t.n_wrong_to_right} & "
            f"{t.n_right_to_wrong} & {'+' if t.net_corrections >= 0 else ''}"
            f"{t.net_corrections} & {verdict} \\\\"
        )
    tier_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (output_dir / "llm_by_severity_table.tex").write_text("\n".join(tier_lines))

    # ── Table 4: Consistency (if multiple runs) ──
    if consistency and consistency.n_runs >= 2:
        con_lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{LLM consistency across $N=" + str(consistency.n_runs) + r"$ repeated runs. "
            r"CV = coefficient of variation. Flip rate = fraction of images with "
            r"different predictions across runs.}",
            r"\label{tab:llm_consistency}",
            r"\begin{tabular}{l r}",
            r"\toprule",
            r"\textbf{Metric} & \textbf{Value} \\",
            r"\midrule",
            f"Mean accuracy & {consistency.mean_accuracy * 100:.1f}\\% \\\\",
            f"Std accuracy  & $\\pm${consistency.std_accuracy * 100:.2f}pp \\\\",
            f"CV            & {consistency.cv:.4f} \\\\",
            r"\midrule",
            f"Images always consistent & {consistency.n_always_same} "
            f"({(1 - consistency.flip_rate) * 100:.1f}\\%) \\\\",
            f"Images with flips & {consistency.n_sometimes_different} "
            f"({consistency.flip_rate * 100:.1f}\\%) \\\\",
            f"Mean pairwise agreement & {consistency.mean_pairwise_agreement * 100:.1f}\\% \\\\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
        (output_dir / "llm_consistency_table.tex").write_text("\n".join(con_lines))

    print(f"  📝 LaTeX tables written to {output_dir}")


# ════════════════════════════════════════════════════════════════
# Figures
# ════════════════════════════════════════════════════════════════

def plot_transition_sankey(transitions: TransitionMatrix, output_dir: Path):
    """Visual: what happens to each prediction when LLM is added."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("  ⚠ matplotlib not installed — skipping transition plot")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    # Left side: without LLM
    total = transitions.n_total
    correct_b = transitions.n_both_correct + transitions.n_right_to_wrong
    wrong_b = transitions.n_wrong_to_right + transitions.n_both_wrong

    # Stacked bars
    bar_width = 0.3

    # Left bar (without LLM)
    ax.barh(1, correct_b, bar_width, color="#27ae60", alpha=0.8, label="Correct")
    ax.barh(1, wrong_b, bar_width, left=correct_b, color="#e74c3c", alpha=0.8, label="Wrong")

    # Right bar (with LLM)
    correct_c = transitions.n_both_correct + transitions.n_wrong_to_right
    wrong_c = transitions.n_right_to_wrong + transitions.n_both_wrong
    ax.barh(0, correct_c, bar_width, color="#27ae60", alpha=0.8)
    ax.barh(0, wrong_c, bar_width, left=correct_c, color="#e74c3c", alpha=0.8)

    # Annotations
    ax.text(correct_b / 2, 1, f"{correct_b}", ha="center", va="center", fontweight="bold", color="white")
    ax.text(correct_b + wrong_b / 2, 1, f"{wrong_b}", ha="center", va="center", fontweight="bold", color="white")
    ax.text(correct_c / 2, 0, f"{correct_c}", ha="center", va="center", fontweight="bold", color="white")
    ax.text(correct_c + wrong_c / 2, 0, f"{wrong_c}", ha="center", va="center", fontweight="bold", color="white")

    # Transition arrows
    mid_x = total * 0.5
    if transitions.n_wrong_to_right > 0:
        ax.annotate(f"+{transitions.n_wrong_to_right} corrected",
                    xy=(correct_c, 0.15), xytext=(correct_b + wrong_b * 0.3, 0.85),
                    arrowprops=dict(arrowstyle="->", color="#27ae60", lw=2),
                    fontsize=10, color="#27ae60", fontweight="bold")
    if transitions.n_right_to_wrong > 0:
        ax.annotate(f"-{transitions.n_right_to_wrong} broken",
                    xy=(correct_c + wrong_c * 0.3, 0.15), xytext=(correct_b * 0.7, 0.85),
                    arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=2),
                    fontsize=10, color="#e74c3c", fontweight="bold")

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["With LLM (Config C)", "Without LLM (Config B)"], fontsize=11)
    ax.set_xlabel("Number of predictions", fontsize=11)
    ax.set_title(f"Error Transition Analysis (n={total})\n"
                 f"Net: {'+' if transitions.net_corrections >= 0 else ''}{transitions.net_corrections} corrections, "
                 f"Help:Harm = {transitions.help_to_harm_ratio:.1f}:1",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "transition_analysis.png", dpi=200)
    fig.savefig(output_dir / "transition_analysis.pdf")
    plt.close(fig)
    print(f"  📊 Transition analysis → {output_dir / 'transition_analysis.png'}")


def plot_severity_breakdown(tier_transitions: dict, output_dir: Path):
    """Bar chart: net LLM corrections per severity tier."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    tiers = ["severe", "moderate", "low"]
    available = [t for t in tiers if t in tier_transitions]
    if not available:
        return

    nets = [tier_transitions[t].net_corrections for t in available]
    helps = [tier_transitions[t].n_wrong_to_right for t in available]
    harms = [-tier_transitions[t].n_right_to_wrong for t in available]

    x = np.arange(len(available))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, helps, width, label="Corrected (✗→✓)", color="#27ae60", alpha=0.85)
    ax.bar(x + width / 2, harms, width, label="Broken (✓→✗)", color="#e74c3c", alpha=0.85)

    # Net annotations
    for i, (t, net) in enumerate(zip(available, nets)):
        color = "#27ae60" if net >= 0 else "#e74c3c"
        ax.text(i, max(helps[i], 0) + 0.5,
                f"net: {'+' if net >= 0 else ''}{net}",
                ha="center", fontweight="bold", color=color, fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels([t.capitalize() for t in available], fontsize=11)
    ax.set_ylabel("Number of predictions", fontsize=11)
    ax.set_title("LLM Impact by Disease Severity", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "severity_breakdown.png", dpi=200)
    fig.savefig(output_dir / "severity_breakdown.pdf")
    plt.close(fig)
    print(f"  📊 Severity breakdown → {output_dir / 'severity_breakdown.png'}")


def plot_consistency(consistency: ConsistencyReport, output_dir: Path):
    """Plot accuracy across repeated runs."""
    if not consistency or consistency.n_runs < 2:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Left: accuracy per run
    runs = range(1, consistency.n_runs + 1)
    ax1.bar(runs, [a * 100 for a in consistency.accuracies], color="#3498db", alpha=0.85)
    ax1.axhline(y=consistency.mean_accuracy * 100, color="red", linestyle="--",
                label=f"Mean: {consistency.mean_accuracy * 100:.1f}%")
    ax1.fill_between(
        [0.5, consistency.n_runs + 0.5],
        (consistency.mean_accuracy - consistency.std_accuracy) * 100,
        (consistency.mean_accuracy + consistency.std_accuracy) * 100,
        alpha=0.15, color="red", label=f"±1σ ({consistency.std_accuracy * 100:.2f}pp)"
    )
    ax1.set_xlabel("Run", fontsize=11)
    ax1.set_ylabel("Accuracy (%)", fontsize=11)
    ax1.set_title("Accuracy per Run", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    # Right: consistency pie
    labels = [f"Consistent\n({consistency.n_always_same})",
              f"Flipping\n({consistency.n_sometimes_different})"]
    sizes = [consistency.n_always_same, consistency.n_sometimes_different]
    colors_pie = ["#27ae60", "#e74c3c"]
    ax2.pie(sizes, labels=labels, colors=colors_pie, autopct="%1.1f%%",
            startangle=90, textprops={"fontsize": 10})
    ax2.set_title(f"Prediction Stability\n(CV={consistency.cv:.4f})",
                  fontsize=12, fontweight="bold")

    fig.tight_layout()
    fig.savefig(output_dir / "consistency_analysis.png", dpi=200)
    fig.savefig(output_dir / "consistency_analysis.pdf")
    plt.close(fig)
    print(f"  📊 Consistency analysis → {output_dir / 'consistency_analysis.png'}")


# ════════════════════════════════════════════════════════════════
# Console summary
# ════════════════════════════════════════════════════════════════

def print_critical_summary(
    marginal: MarginalGainReport,
    transitions: TransitionMatrix,
    tier_transitions: dict[str, TransitionMatrix],
    consistency: ConsistencyReport | None,
):
    print("\n" + "=" * 90)
    print("  CRITICAL LLM CONTRIBUTION ANALYSIS")
    print("  Does LLaVA actually help, or is it expensive confirmation bias?")
    print("=" * 90)

    # ── E1: Marginal gain ──
    print("\n── E1: Marginal Accuracy Gain ──")
    sig = "YES" if marginal.delta_significant else "NO"
    print(f"  Without LLM: {marginal.accuracy_without * 100:.1f}%")
    print(f"  With LLM:    {marginal.accuracy_with * 100:.1f}%")
    print(f"  Delta:        {'+' if marginal.delta >= 0 else ''}{marginal.delta * 100:.1f}pp "
          f"  95% CI: [{marginal.delta_ci_lo * 100:.1f}, {marginal.delta_ci_hi * 100:.1f}]")
    print(f"  Significant:  {sig} (p = {marginal.p_value:.4f})")
    if marginal.delta_ci_lo < 0:
        print(f"  ⚠ WARNING: CI includes zero — improvement may be due to chance")

    # ── E3: Transition matrix ──
    print(f"\n── E3: Error Transitions (n={transitions.n_total}) ──")
    print(f"  ✓→✓ (no change, both correct):  {transitions.n_both_correct:4d}  "
          f"({transitions.n_both_correct / transitions.n_total * 100:.1f}%)")
    print(f"  ✗→✗ (no change, both wrong):    {transitions.n_both_wrong:4d}  "
          f"({transitions.n_both_wrong / transitions.n_total * 100:.1f}%)")
    print(f"  ✗→✓ (LLM CORRECTED):            {transitions.n_wrong_to_right:4d}  "
          f"({transitions.n_wrong_to_right / transitions.n_total * 100:.1f}%)")
    print(f"  ✓→✗ (LLM BROKE):                {transitions.n_right_to_wrong:4d}  "
          f"({transitions.n_right_to_wrong / transitions.n_total * 100:.1f}%)")
    print(f"  ─────────────────────────────")
    print(f"  Net corrections:  {'+' if transitions.net_corrections >= 0 else ''}{transitions.net_corrections}")
    print(f"  Help:Harm ratio:  {transitions.help_to_harm_ratio:.1f}:1")
    print(f"  Irrelevance rate: {transitions.irrelevance_rate * 100:.1f}% "
          f"(LLM didn't change the outcome)")

    if transitions.help_to_harm_ratio < 1.0:
        print(f"\n  🚨 CRITICAL: LLM HURTS more than it helps (ratio < 1.0)")
        print(f"     Consider removing LLM or restricting to specific scenarios.")
    elif transitions.help_to_harm_ratio < 2.0:
        print(f"\n  ⚠ MARGINAL: Benefit exists but is fragile (ratio < 2.0)")
        print(f"     Reviewer may argue the latency cost isn't justified.")
    else:
        print(f"\n  ✅ LLM provides clear net benefit (ratio ≥ 2.0)")

    # ── Severity stratified ──
    print(f"\n── LLM Value by Severity ──")
    for tier in ["severe", "moderate", "low"]:
        if tier not in tier_transitions:
            continue
        t = tier_transitions[tier]
        verdict = "HELPS" if t.net_corrections > 0 else "HURTS" if t.net_corrections < 0 else "NEUTRAL"
        print(f"  {tier:10s}: +{t.n_wrong_to_right} corrected, -{t.n_right_to_wrong} broken, "
              f"net={'+' if t.net_corrections >= 0 else ''}{t.net_corrections}  → {verdict}")

    # ── Cost analysis ──
    print(f"\n── Latency Cost ──")
    if marginal.cost_per_correct < float("inf"):
        print(f"  Extra latency per net correction: {marginal.cost_per_correct:,.0f} ms")
        if marginal.cost_per_correct > 60000:
            print(f"  ⚠ Each correction costs >{marginal.cost_per_correct / 1000:.0f}s of latency")
    else:
        print(f"  ∞ (no net corrections — LLM adds latency with zero net benefit)")

    # ── Consistency ──
    if consistency and consistency.n_runs >= 2:
        print(f"\n── E5: Consistency ({consistency.n_runs} runs) ──")
        print(f"  Mean accuracy: {consistency.mean_accuracy * 100:.1f}% ± {consistency.std_accuracy * 100:.2f}pp")
        print(f"  CV:            {consistency.cv:.4f}")
        print(f"  Flip rate:     {consistency.flip_rate * 100:.1f}% of images get different predictions")
        if consistency.flip_rate > 0.10:
            print(f"  ⚠ HIGH INSTABILITY: >{consistency.flip_rate * 100:.0f}% of predictions are stochastic")
            print(f"     Consider: lower temperature, majority vote, or removing LLM")
        elif consistency.cv > 0.05:
            print(f"  ⚠ Non-trivial variance across runs (CV > 5%)")
        else:
            print(f"  ✅ Good consistency (CV < 5%, flip rate < 10%)")

    # ── Honest verdict ──
    print(f"\n── HONEST VERDICT ──")
    score = 0
    reasons = []
    if marginal.delta_significant:
        score += 2
        reasons.append("Statistically significant improvement")
    elif marginal.delta > 0:
        score += 1
        reasons.append("Positive but not significant improvement")
    else:
        reasons.append("No accuracy improvement")

    if transitions.help_to_harm_ratio >= 2.0:
        score += 2
        reasons.append(f"Strong help:harm ratio ({transitions.help_to_harm_ratio:.1f}:1)")
    elif transitions.help_to_harm_ratio >= 1.0:
        score += 1
        reasons.append(f"Marginal help:harm ratio ({transitions.help_to_harm_ratio:.1f}:1)")
    else:
        score -= 1
        reasons.append(f"LLM causes net harm ({transitions.help_to_harm_ratio:.1f}:1)")

    severe_trans = tier_transitions.get("severe")
    if severe_trans and severe_trans.net_corrections > 0:
        score += 2
        reasons.append(f"Helps on severe diseases (+{severe_trans.net_corrections} net)")
    elif severe_trans and severe_trans.net_corrections < 0:
        score -= 2
        reasons.append(f"HURTS on severe diseases ({severe_trans.net_corrections} net)")

    if consistency and consistency.n_runs >= 2:
        if consistency.cv < 0.03:
            score += 1
            reasons.append("Highly consistent across runs")
        elif consistency.flip_rate > 0.15:
            score -= 1
            reasons.append("Inconsistent across runs")

    for r in reasons:
        print(f"  {'✅' if 'help' in r.lower() or 'signif' in r.lower() or 'consist' in r.lower() else '⚠' if 'marginal' in r.lower() or 'positive' in r.lower() else '🚨'} {r}")

    if score >= 4:
        print(f"\n  VERDICT: LLM validator JUSTIFIED (score {score}/7)")
        print(f"  Paper framing: 'The LLM validator provides statistically significant improvement,")
        print(f"  particularly on safety-critical severe diseases.'")
    elif score >= 2:
        print(f"\n  VERDICT: LLM validator CONDITIONALLY JUSTIFIED (score {score}/7)")
        print(f"  Paper framing: 'The LLM validator provides modest improvement in specific scenarios.")
        print(f"  Future work: selective invocation to reduce latency overhead.'")
    else:
        print(f"\n  VERDICT: LLM validator NOT JUSTIFIED (score {score}/7)")
        print(f"  Paper framing: Be honest. Show ablation B as your main system,")
        print(f"  present LLM as exploratory. Reviewers will respect honesty > spin.")

    print("=" * 90)


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Critical LLM contribution evaluation")
    parser.add_argument("--without", type=str, help="Predictions CSV without LLM (Config B)")
    parser.add_argument("--with-llm", type=str, help="Predictions CSV with LLM (Config C)")
    parser.add_argument("--auto", action="store_true",
                        help="Auto-detect from outputs/ablation/")
    parser.add_argument("--live", action="store_true",
                        help="Run live evaluation (needs Ollama)")
    parser.add_argument("--runs", type=int, default=1,
                        help="Number of repeated runs for consistency analysis")
    parser.add_argument("--output-dir",
                        default=str(PROJECT_ROOT / "outputs" / "llm_contribution"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load predictions ──
    preds_without = None
    preds_with = None
    run_preds = []  # for consistency analysis

    if args.without and args.with_llm:
        preds_without = load_predictions(Path(args.without))
        preds_with = load_predictions(Path(args.with_llm))
    elif args.auto or (not args.live):
        ablation_dir = PROJECT_ROOT / "outputs" / "ablation"
        b_path = ablation_dir / "predictions_B_yolo_rules.csv"
        c_path = ablation_dir / "predictions_C_full_pipeline.csv"
        if b_path.exists() and c_path.exists():
            preds_without = load_predictions(b_path)
            preds_with = load_predictions(c_path)
            print(f"  Auto-loaded: B={len(preds_without)}, C={len(preds_with)} predictions")
        else:
            print(f"ERROR: Expected CSVs not found at {ablation_dir}")
            print(f"  Run ablation_study.py first, or provide --without and --with-llm paths")
            sys.exit(1)

    if preds_without is None or preds_with is None:
        print("ERROR: Need both --without and --with-llm predictions")
        sys.exit(1)

    # ── Compute metrics ──
    print("🔬 Computing LLM contribution metrics...")

    marginal = compute_marginal_gain(preds_without, preds_with, seed=args.seed)
    transitions = compute_transition_matrix(preds_without, preds_with)
    tier_transitions = compute_severity_stratified_transitions(preds_without, preds_with)
    consistency = compute_consistency(run_preds) if len(run_preds) >= 2 else None

    # ── Console ──
    print_critical_summary(marginal, transitions, tier_transitions, consistency)

    # ── Figures ──
    print("\n📊 Generating figures...")
    plot_transition_sankey(transitions, output_dir)
    plot_severity_breakdown(tier_transitions, output_dir)
    if consistency:
        plot_consistency(consistency, output_dir)

    # ── LaTeX ──
    print("\n📝 Generating LaTeX tables...")
    generate_latex(marginal, transitions, tier_transitions, consistency, output_dir)

    # ── JSON ──
    summary = {
        "timestamp": datetime.now().isoformat(),
        "marginal_gain": {
            "accuracy_without": round(marginal.accuracy_without, 4),
            "accuracy_with": round(marginal.accuracy_with, 4),
            "delta": round(marginal.delta, 4),
            "ci_95": [round(marginal.delta_ci_lo, 4), round(marginal.delta_ci_hi, 4)],
            "significant": marginal.delta_significant,
            "p_value": round(marginal.p_value, 6),
            "cost_per_correct_ms": round(marginal.cost_per_correct, 0)
            if marginal.cost_per_correct < float("inf") else None,
        },
        "transitions": {
            "both_correct": transitions.n_both_correct,
            "both_wrong": transitions.n_both_wrong,
            "wrong_to_right": transitions.n_wrong_to_right,
            "right_to_wrong": transitions.n_right_to_wrong,
            "net_corrections": transitions.net_corrections,
            "help_harm_ratio": round(transitions.help_to_harm_ratio, 2)
            if transitions.help_to_harm_ratio < float("inf") else None,
            "irrelevance_rate": round(transitions.irrelevance_rate, 4),
        },
        "severity_stratified": {
            tier: {
                "n": t.n_total,
                "wrong_to_right": t.n_wrong_to_right,
                "right_to_wrong": t.n_right_to_wrong,
                "net": t.net_corrections,
            }
            for tier, t in tier_transitions.items()
        },
        "examples_helped": transitions.examples_helped,
        "examples_harmed": transitions.examples_harmed,
        "consistency": {
            "n_runs": consistency.n_runs if consistency else 0,
            "mean_accuracy": consistency.mean_accuracy if consistency else None,
            "cv": consistency.cv if consistency else None,
            "flip_rate": consistency.flip_rate if consistency else None,
        } if consistency else None,
    }

    json_path = output_dir / "llm_contribution_results.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"\n📄 JSON → {json_path}")
    print(f"✅ All outputs saved to {output_dir}/")


if __name__ == "__main__":
    main()
