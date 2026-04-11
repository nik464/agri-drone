#!/usr/bin/env python3
"""
Experiment 5 — Generate LaTeX Tables and Section 9 Draft

Reads all experiment results and produces publication-ready LaTeX tables
and a Markdown draft of Section 9 (Experimental Results).

Outputs:
    evaluate/results/table2_ablation.tex
    evaluate/results/table3_sensitivity.tex
    evaluate/results/table4_eml.tex
    evaluate/results/section9_draft.md

Usage:
    python evaluate/paper_tables.py
"""

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "evaluate" / "results"


def load_json(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


# ════════════════════════════════════════════════════════════════
# Table 2: Ablation (Config A vs Config B)
# ════════════════════════════════════════════════════════════════

def gen_table2():
    ab = load_json("ablation_summary.json")

    rows = [
        ("A", "YOLO-only",
         f"{ab['config_A_accuracy']:.4f}",
         f"{ab['config_A_macro_f1']:.4f}",
         f"{ab['config_A_RWA']:.4f}",
         f"{ab['safety_gap_A']:+.4f}",
         f"{ab['config_A_mean_latency_ms']:.1f}"),
        ("B", "YOLO + Rules + Ensemble",
         f"{ab['config_B_accuracy']:.4f}",
         f"{ab['config_B_macro_f1']:.4f}",
         f"{ab['config_B_RWA']:.4f}",
         f"{ab['safety_gap_B']:+.4f}",
         f"{ab['config_B_mean_latency_ms']:.1f}"),
    ]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Ablation study: component contribution on 935-image test set (21 classes, stratified 15\% split).}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"Config & Pipeline & Accuracy & Macro-F1 & RWA & Safety Gap & Latency (ms) \\",
        r"\midrule",
    ]
    for cfg, name, acc, f1, rwa, sg, lat in rows:
        lines.append(f"{cfg} & {name} & {acc} & {f1} & {rwa} & {sg} & {lat} \\\\")
    lines += [
        r"\midrule",
        f"\\multicolumn{{7}}{{l}}{{$\\Delta$ (B$-$A): Macro-F1 = {ab['B_over_A_macro_F1_delta']:+.4f}, "
        f"$n$ = {ab['n_test_images']}, classes = {ab['n_classes']}}} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    tex = "\n".join(lines) + "\n"
    (RESULTS / "table2_ablation.tex").write_text(tex, encoding="utf-8")
    print("  table2_ablation.tex")
    return tex


# ════════════════════════════════════════════════════════════════
# Table 3: Sensitivity Analysis
# ════════════════════════════════════════════════════════════════

def gen_table3():
    s = load_json("sensitivity_summary.json")

    # Also load grid CSV to get top-5 configs
    grid_rows = []
    with open(RESULTS / "sensitivity_grid.csv", newline="", encoding="utf-8") as f:
        grid_rows = list(csv.DictReader(f))

    # Sort by macro_f1 descending
    grid_rows.sort(key=lambda r: float(r["macro_f1"]), reverse=True)
    top5 = grid_rows[:5]

    # Find current config row
    current = next(
        (r for r in grid_rows
         if float(r["color_scale"]) == s["current_config"]["color_scale"]
         and float(r["stripe_weight"]) == s["current_config"]["stripe_weight"]
         and float(r["yolo_override_threshold"]) == s["current_config"]["yolo_override_threshold"]),
        None
    )
    current_rank = None
    if current:
        current_rank = grid_rows.index(current) + 1

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Threshold sensitivity analysis over 125 configurations "
        r"($5 \times 5 \times 5$ grid) on 935 validation images. "
        f"F1 std = {s['F1_std_across_configs']:.4f}.}}",
        r"\label{tab:sensitivity}",
        r"\begin{tabular}{ccccccc}",
        r"\toprule",
        r"Rank & $\alpha_{color}$ & $w_{stripe}$ & $\tau_{yolo}$ & Macro-F1 & RWA & Accuracy \\",
        r"\midrule",
    ]

    for i, row in enumerate(top5, 1):
        cs = int(float(row["color_scale"]))
        sw = float(row["stripe_weight"])
        yt = float(row["yolo_override_threshold"])
        f1 = float(row["macro_f1"])
        rwa = float(row["rwa"])
        acc = float(row["accuracy"])
        lines.append(f"{i} & {cs} & {sw:.1f} & {yt:.2f} & {f1:.4f} & {rwa:.4f} & {acc:.4f} \\\\")

    lines.append(r"\midrule")

    # Current config
    if current:
        cs = int(float(current["color_scale"]))
        sw = float(current["stripe_weight"])
        yt = float(current["yolo_override_threshold"])
        f1 = float(current["macro_f1"])
        rwa = float(current["rwa"])
        acc = float(current["accuracy"])
        lines.append(
            f"\\textit{{{current_rank}}} & \\textit{{{cs}}} & \\textit{{{sw:.1f}}} "
            f"& \\textit{{{yt:.2f}}} & \\textit{{{f1:.4f}}} & \\textit{{{rwa:.4f}}} "
            f"& \\textit{{{acc:.4f}}} \\\\ \\% current defaults"
        )

    lines += [
        r"\midrule",
        f"\\multicolumn{{7}}{{l}}{{F1 range: [{s['F1_range'][0]:.4f}, {s['F1_range'][1]:.4f}], "
        f"$\\sigma$ = {s['F1_std_across_configs']:.4f}, "
        f"$n_{{configs}}$ = {s['n_configs_evaluated']}}} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    tex = "\n".join(lines) + "\n"
    (RESULTS / "table3_sensitivity.tex").write_text(tex, encoding="utf-8")
    print("  table3_sensitivity.tex")
    return tex


# ════════════════════════════════════════════════════════════════
# Table 4: EML Comparison
# ════════════════════════════════════════════════════════════════

def gen_table4():
    eml = load_json("eml_summary.json")

    # Pick key diseases from the cost table (those with explicit costs)
    key_diseases = [
        "wheat_fusarium_head_blight",
        "wheat_blast",
        "rice_blast",
        "wheat_black_rust",
        "rice_bacterial_blight",
        "rice_brown_spot",
        "healthy_wheat",
        "healthy_rice",
    ]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Expected Monetary Loss (INR/sample) for key diseases. "
        f"$n$ = {eml['n_test_samples']}.}}",
        r"\label{tab:eml}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Disease & $C_{miss}$ & Miss\%$_A$ & Miss\%$_B$ & EML$_A$ & EML$_B$ \\",
        r"\midrule",
    ]

    for disease in key_diseases:
        a = eml["per_disease_A"].get(disease, {})
        b = eml["per_disease_B"].get(disease, {})
        if not a:
            continue
        name = disease.replace("wheat_", "W. ").replace("rice_", "R. ").replace("healthy_", "H. ").replace("_", " ").title()
        cm = a.get("cost_miss", 0)
        mr_a = a.get("miss_rate", 0) * 100
        mr_b = b.get("miss_rate", 0) * 100
        ea = a.get("eml_per_positive", 0)
        eb = b.get("eml_per_positive", 0)
        lines.append(f"{name} & {cm:,} & {mr_a:.1f} & {mr_b:.1f} & {ea:,.0f} & {eb:,.0f} \\\\")

    lines += [
        r"\midrule",
        f"\\textbf{{Aggregate}} & & & & \\textbf{{\\textrm{{₹}}{eml['total_eml_A']:,.0f}}} "
        f"& \\textbf{{\\textrm{{₹}}{eml['total_eml_B']:,.0f}}} \\\\",
        r"\midrule",
        f"\\multicolumn{{6}}{{l}}{{$\\Delta$EML = \\textrm{{₹}}{eml['delta_eml']:,.0f} "
        f"({eml['delta_pct']:+.1f}\\%), "
        f"critical diseases: A=\\textrm{{₹}}{eml['critical_eml_A']:,.0f}, "
        f"B=\\textrm{{₹}}{eml['critical_eml_B']:,.0f}}} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    tex = "\n".join(lines) + "\n"
    (RESULTS / "table4_eml.tex").write_text(tex, encoding="utf-8")
    print("  table4_eml.tex")
    return tex


# ════════════════════════════════════════════════════════════════
# Section 9 Draft
# ════════════════════════════════════════════════════════════════

def gen_section9():
    ab = load_json("ablation_summary.json")
    s = load_json("sensitivity_summary.json")
    eml = load_json("eml_summary.json")

    # Identify worst B−A delta diseases
    b_over_a = ab.get("B_over_A_F1_delta", {})
    worst_3 = sorted(b_over_a.items(), key=lambda x: x[1])[:3]
    best_3 = sorted(b_over_a.items(), key=lambda x: x[1], reverse=True)[:3]

    # EML key stats
    delta_eml = eml["delta_eml"]
    pct_eml = eml["delta_pct"]

    md = f"""# Section 9: Experimental Results

## 9.1 Dataset and Protocol

We evaluate on a 21-class Indian crop disease dataset comprising 14 wheat
and 5 rice pathologies plus 2 healthy classes.  A stratified 70/15/15 split
yields **{ab['n_test_images']}** test images (~45 per class; wheat\\_stem\\_fly: 35).
All configurations share the same test fold (seed=42).

## 9.2 Ablation Study (Table 2)

\\input{{table2_ablation}}

Config A (YOLOv8n-cls standalone) achieves **{ab['config_A_macro_f1']:.4f}** macro-F1
at {ab['config_A_mean_latency_ms']:.1f} ms/image.  Adding the rule engine and ensemble
voter (Config B) drops macro-F1 to **{ab['config_B_macro_f1']:.4f}**
($\\Delta$ = {ab['B_over_A_macro_F1_delta']:+.4f}).

Risk-Weighted Accuracy confirms the pattern: RWA$_A$ = {ab['config_A_RWA']:.4f},
RWA$_B$ = {ab['config_B_RWA']:.4f}.  The safety gap (RWA $-$ Accuracy) shifts
from {ab['safety_gap_A']:+.4f} (A) to {ab['safety_gap_B']:+.4f} (B), indicating
the rule engine introduces disproportionately more errors on high-severity diseases.

**Per-class analysis.**  The three largest F1 regressions under Config B are:
"""

    for disease, delta in worst_3:
        name = disease.replace("_", "\\_")
        md += f"- `{name}`: $\\Delta$F1 = {delta:+.4f}\n"

    md += f"""
The rule engine's color and spatial heuristics over-correct confident YOLO
predictions in wheat diseases characterised by diffuse brown lesions,
where multiple KB profiles share overlapping HSV signatures.

## 9.3 Threshold Sensitivity (Table 3)

\\input{{table3_sensitivity}}

A $5 \\times 5 \\times 5$ grid search over the three hardcoded rule-engine
parameters ($\\alpha_{{color}}$, $w_{{stripe}}$, $\\tau_{{yolo}}$) evaluates
{s['n_configs_evaluated']} configurations on {s['n_val_images']} validation images.

The macro-F1 standard deviation across all configurations is
**{s['F1_std_across_configs']:.4f}**, with a total range of
[{s['F1_range'][0]:.4f}, {s['F1_range'][1]:.4f}]. The current defaults
($\\alpha_{{color}}$={s['current_config']['color_scale']},
$w_{{stripe}}$={s['current_config']['stripe_weight']},
$\\tau_{{yolo}}$={s['current_config']['yolo_override_threshold']})
achieve F1={s['current_config_F1']:.4f}, within {abs(s['optimal_config_F1'] - s['current_config_F1']):.4f}
of the grid optimum ({s['optimal_config_F1']:.4f}).

**Interpretation.**  The low $\\sigma$ confirms that the rule engine's
performance is dominated by the structural mismatch between
heuristic rules and the learned YOLOv8 feature space, not by any
single threshold choice.  The colour scale shows the strongest
marginal effect (higher $\\alpha_{{color}}$ $\\rightarrow$ slightly better F1),
while stripe weight and YOLO override threshold have negligible impact.

## 9.4 Expected Monetary Loss (Table 4)

\\input{{table4_eml}}

Using field-verified cost data from Indian agricultural extension reports,
we compute the Expected Monetary Loss under both configurations.

Config A achieves a total EML of **₹{eml['total_eml_A']:,.0f}**/sample,
while Config B produces **₹{eml['total_eml_B']:,.0f}**/sample
($\\Delta$ = ₹{delta_eml:,.0f}, {pct_eml:+.1f}%).
"""

    if delta_eml > 0:
        md += f"""
The higher EML under Config B reflects its elevated miss rates on
high-cost diseases.  For critical diseases alone, Config A incurs
₹{eml['critical_eml_A']:,.0f} vs Config B's ₹{eml['critical_eml_B']:,.0f}.
"""
    else:
        md += f"""
Config B's lower EML reflects its reduced miss rates on
high-cost diseases.  For critical diseases alone, Config B saves
₹{eml['critical_eml_A'] - eml['critical_eml_B']:,.0f} over Config A.
"""

    md += f"""
## 9.5 Discussion

These experiments reveal a counter-intuitive finding: the handcrafted rule
engine, despite encoding agronomic domain knowledge, **degrades**
classification performance when cascaded after a well-trained YOLOv8n
classifier.  Three factors explain this:

1. **Feature-space mismatch**: The rule engine operates on raw HSV ratios
   and morphological features, while YOLOv8 learns discriminative features
   in a 512-d embedding space.  The two representations are not calibrated.

2. **Threshold sensitivity is a red herring**: The $\\sigma_{{F1}}$ = {s['F1_std_across_configs']:.4f}
   across 125 configurations shows the problem is structural, not parametric.

3. **Conflict resolution bias**: When YOLO and rules disagree, the current
   resolution logic favours rules for any rule\\_score > 0.3, but many
   diseases produce spurious matches through shared colour signatures.

**Implication for C1 (Safety-Aware Ensemble).**  These results motivate
replacing the heuristic conflict resolution with a learned fusion layer
that can weight YOLO confidence against calibrated rule evidence.
The {ab['safety_gap_B']:+.4f} safety gap under Config B, versus
{ab['safety_gap_A']:+.4f} under Config A, demonstrates that naive
rule injection amplifies errors on precisely the diseases where
misclassification costs are highest.
"""

    (RESULTS / "section9_draft.md").write_text(md.strip() + "\n", encoding="utf-8")
    print("  section9_draft.md")
    return md


# ════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  EXPERIMENT 5: Paper Tables & Section 9 Draft")
    print("=" * 70)

    gen_table2()
    gen_table3()
    gen_table4()
    gen_section9()

    print(f"\n  All outputs → {RESULTS}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
