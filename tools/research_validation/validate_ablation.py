#!/usr/bin/env python3
"""Validate all headline claims from committed prediction CSVs.

Reads:
  evaluate/results/predictions_A_yolo_only.csv
  evaluate/results/predictions_B_yolo_rules.csv
  evaluate/results/predictions_C_rules_only.csv
  evaluate/results/ablation_summary.json
  evaluate/results/statistical_tests.json
  evaluate/results/eml_summary.json

Outputs:
  evaluate/results/override_decomposition.json
  evaluate/results/override_decomposition.csv
  evaluate/results/claim_traceability_table.csv
  (stdout) full validation report
"""
import csv, json, sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent.parent
RES = ROOT / "evaluate" / "results"

def load_csv(name):
    rows = []
    with open(RES / name, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows

def accuracy(rows):
    return sum(1 for r in rows if int(r["correct"])) / len(rows)

def macro_f1(rows):
    tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)
    classes = set()
    for r in rows:
        gt, pr = r["ground_truth"], r["predicted"]
        classes.add(gt)
        if gt == pr:
            tp[gt] += 1
        else:
            fn[gt] += 1
            fp[pr] += 1
    f1s = []
    for c in classes:
        p = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0
        rec = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0
        f1s.append(2 * p * rec / (p + rec) if (p + rec) > 0 else 0)
    return sum(f1s) / len(f1s)

def mcnemar_chi2(a_rows, b_rows):
    """Continuity-corrected McNemar chi2 and p-value from paired predictions."""
    b_right_a_wrong = 0
    a_right_b_wrong = 0
    for a, b in zip(a_rows, b_rows):
        ac, bc = int(a["correct"]), int(b["correct"])
        if ac == 1 and bc == 0:
            a_right_b_wrong += 1
        elif ac == 0 and bc == 1:
            b_right_a_wrong += 1
    n_disc = a_right_b_wrong + b_right_a_wrong
    if n_disc == 0:
        return 0.0, 1.0, a_right_b_wrong, b_right_a_wrong
    chi2 = (abs(a_right_b_wrong - b_right_a_wrong) - 1) ** 2 / n_disc
    # p-value from chi2(1) — use simple approximation
    import math
    # chi2 CDF(1) approximation via normal
    z = math.sqrt(chi2)
    # two-tailed p from standard normal
    p = 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))
    return chi2, p, a_right_b_wrong, b_right_a_wrong

def override_decomposition(a_rows, b_rows):
    """Compute where B overrides A's prediction and classify as rescue/corruption/neutral."""
    overrides = []
    for a, b in zip(a_rows, b_rows):
        if a["predicted"] != b["predicted"]:
            a_correct = int(a["correct"])
            b_correct = int(b["correct"])
            if b_correct == 1 and a_correct == 0:
                cat = "rescue"
            elif b_correct == 0 and a_correct == 1:
                cat = "corruption"
            elif b_correct == 1 and a_correct == 1:
                cat = "neutral_both_right"
            else:
                cat = "neutral_both_wrong"
            overrides.append({
                "image": a["image"],
                "ground_truth": a["ground_truth"],
                "pred_A": a["predicted"],
                "pred_B": b["predicted"],
                "conf_A": float(a["confidence"]),
                "conf_B": float(b["confidence"]),
                "A_correct": a_correct,
                "B_correct": b_correct,
                "category": cat,
            })
    return overrides

print("=" * 70)
print("CORE ABLATION VALIDATION — from committed CSVs")
print("=" * 70)

a = load_csv("predictions_A_yolo_only.csv")
b = load_csv("predictions_B_yolo_rules.csv")
c = load_csv("predictions_C_rules_only.csv")

print(f"\nSample counts: A={len(a)}, B={len(b)}, C={len(c)}")

acc_a, acc_b, acc_c = accuracy(a), accuracy(b), accuracy(c)
f1_a, f1_b, f1_c = macro_f1(a), macro_f1(b), macro_f1(c)

print(f"\nAccuracy:  A={acc_a:.4f}  B={acc_b:.4f}  C={acc_c:.4f}")
print(f"Macro-F1:  A={f1_a:.4f}  B={f1_b:.4f}  C={f1_c:.4f}")

chi2, p, arb, bra = mcnemar_chi2(a, b)
print(f"\nMcNemar A-vs-B: chi2={chi2:.4f}, p={p:.6f}")
print(f"  A-right-B-wrong={arb}, B-right-A-wrong={bra}, total discordant={arb+bra}")

# Override decomposition
overrides = override_decomposition(a, b)
cats = defaultdict(int)
for o in overrides:
    cats[o["category"]] += 1

print(f"\nOverride decomposition (A→B prediction changes): {len(overrides)} total")
for cat in ["rescue", "corruption", "neutral_both_right", "neutral_both_wrong"]:
    print(f"  {cat}: {cats[cat]}")

# Save override decomposition
ov_json = {
    "total_overrides": len(overrides),
    "rescues": cats["rescue"],
    "corruptions": cats["corruption"],
    "neutral_both_right": cats["neutral_both_right"],
    "neutral_both_wrong": cats["neutral_both_wrong"],
    "details": overrides,
}
with open(RES / "override_decomposition.json", "w", encoding="utf-8") as f:
    json.dump(ov_json, f, indent=2)
print(f"\nSaved: evaluate/results/override_decomposition.json")

with open(RES / "override_decomposition.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["image","ground_truth","pred_A","pred_B","conf_A","conf_B","A_correct","B_correct","category"])
    w.writeheader()
    w.writerows(overrides)
print(f"Saved: evaluate/results/override_decomposition.csv")

# Load committed ablation_summary for cross-check
with open(RES / "ablation_summary.json", encoding="utf-8") as f:
    ablation = json.load(f)

print(f"\n--- Cross-check vs ablation_summary.json ---")
print(f"  ablation_summary n_test_images: {ablation.get('n_test_images')}")
print(f"  ablation_summary A acc: {ablation.get('config_A_accuracy')}")
print(f"  ablation_summary B acc: {ablation.get('config_B_accuracy')}")
print(f"  ablation_summary C acc: {ablation.get('config_C_accuracy')}")

# Load committed statistical_tests.json for McNemar cross-check
with open(RES / "statistical_tests.json", encoding="utf-8") as f:
    stats = json.load(f)
mcn = stats.get("mcnemar", {}).get("A_vs_B", {})
print(f"\n--- Cross-check vs statistical_tests.json McNemar ---")
print(f"  committed p_value: {mcn.get('p_value')}")
print(f"  committed chi2: {mcn.get('chi2')}")

# EML check
with open(RES / "eml_summary.json", encoding="utf-8") as f:
    eml = json.load(f)
print(f"\n--- EML from eml_summary.json ---")
print(f"  n_test_samples: {eml.get('n_test_samples')}")
print(f"  total_eml_A: {eml.get('total_eml_A')}")
print(f"  total_eml_B: {eml.get('total_eml_B')}")
print(f"  delta_pct: {eml.get('delta_pct')}")

# Build claim traceability table
claims = [
    ("Test set size n=935", f"CSVs have {len(a)} rows", "WRONG — actual n=934", "934"),
    (f"Config A accuracy 96.15%", f"Recomputed: {acc_a:.4f}", "EXACT" if abs(acc_a - 0.9615) < 0.0001 else "MISMATCH", f"{acc_a:.4f}"),
    (f"Config B accuracy 95.72%", f"Recomputed: {acc_b:.4f}", "EXACT" if abs(acc_b - 0.9572) < 0.0001 else "MISMATCH", f"{acc_b:.4f}"),
    (f"Config C accuracy 13.38%", f"Recomputed: {acc_c:.4f}", "EXACT" if abs(acc_c - 0.1338) < 0.001 else "MISMATCH", f"{acc_c:.4f}"),
    (f"McNemar p≈0.134", f"Recomputed: p={p:.6f}", "APPROXIMATE" if abs(p - 0.134) < 0.01 else "CHECK", f"{p:.6f}"),
    (f"EML A=294.33", f"eml_summary.json: {eml.get('total_eml_A')}", "EXACT" if eml.get('total_eml_A') == 294.33 else "CHECK", str(eml.get('total_eml_A'))),
    (f"EML B=2769.06", f"eml_summary.json: {eml.get('total_eml_B')}", "EXACT" if eml.get('total_eml_B') == 2769.06 else "CHECK", str(eml.get('total_eml_B'))),
    (f"Override decomp: 0 rescues, 4 corruptions", f"Recomputed: {cats['rescue']} rescues, {cats['corruption']} corruptions", "EXACT" if cats["rescue"]==0 and cats["corruption"]==4 else "MISMATCH", f"{cats['rescue']} rescues, {cats['corruption']} corruptions"),
]

with open(RES / "claim_traceability_table.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["manuscript_claim", "evidence", "status", "corrected_value"])
    w.writerows(claims)
print(f"\nSaved: evaluate/results/claim_traceability_table.csv")

print("\n" + "=" * 70)
print("VALIDATION COMPLETE")
print("=" * 70)
