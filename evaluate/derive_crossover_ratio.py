#!/usr/bin/env python3
"""
derive_crossover_ratio.py — Verify the EML crossover ratio r*.

The crossover ratio is the miss-to-alarm cost ratio at which Config B
(YOLO + rules) breaks even with Config A (YOLO only) on Expected Monetary
Loss. Below r*, A wins. Above r*, B wins.

This script reads the eml_comparison.csv and computes r* analytically from
the per-class miss/false-alarm rates, confirming (or correcting) the
r≈18.7 claim in the paper.

Usage:
    python evaluate/derive_crossover_ratio.py
"""
from __future__ import annotations

import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "evaluate" / "results"


def main() -> int:
    eml_path = RESULTS_DIR / "eml_comparison.csv"
    if not eml_path.exists():
        print(f"ERROR: {eml_path} not found")
        return 1

    rows = []
    with open(eml_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    print(f"Loaded {len(rows)} disease rows from eml_comparison.csv\n")

    # EML_A(r) = Σ_d [ miss_rate_A_d * r * base_miss_d  +  fa_rate_A_d * base_alarm_d ] * n_d
    # EML_B(r) = Σ_d [ miss_rate_B_d * r * base_miss_d  +  fa_rate_B_d * base_alarm_d ] * n_d
    # Crossover: EML_A(r*) = EML_B(r*)
    # Σ (miss_A - miss_B) * base_miss * n * r  = Σ (fa_B - fa_A) * base_alarm * n
    # r* = Σ (fa_B - fa_A) * alarm * n  /  Σ (miss_A - miss_B) * miss * n
    #   ... but only when the denominator sign supports crossover.

    # Simpler: just sweep r and find when EML_A(r) = EML_B(r)
    base_alarm = 640  # fixed

    def eml(config: str, r: float) -> float:
        total = 0.0
        for row in rows:
            n = int(row["n_samples"])
            miss_rate = float(row[f"miss_rate_{config}"])
            fa_rate = float(row[f"false_alarm_rate_{config}"])
            cost_miss = float(row["cost_miss"])
            total += n * (miss_rate * r * cost_miss + fa_rate * base_alarm)
        return total

    # Sweep r from 0.1 to 100
    best_r = None
    best_diff = float("inf")
    for r_x10 in range(1, 10000):
        r = r_x10 / 100.0
        diff = eml("A", r) - eml("B", r)
        if abs(diff) < best_diff:
            best_diff = abs(diff)
            best_r = r
        # Detect sign change
        if r_x10 > 1:
            prev_r = (r_x10 - 1) / 100.0
            prev_diff = eml("A", prev_r) - eml("B", prev_r)
            if prev_diff * diff < 0:
                # Linear interpolation for more precise crossover
                alpha = abs(prev_diff) / (abs(prev_diff) + abs(diff))
                best_r = prev_r + alpha * 0.01
                break

    if best_r:
        print(f"EML crossover ratio r* ≈ {best_r:.2f}")
        print(f"  At r={best_r:.2f}: EML_A={eml('A', best_r):.0f}, EML_B={eml('B', best_r):.0f}")
        print(f"  At r=1.0:    EML_A={eml('A', 1.0):.0f}, EML_B={eml('B', 1.0):.0f}")
        print(f"  At r=18.75:  EML_A={eml('A', 18.75):.0f}, EML_B={eml('B', 18.75):.0f}")
    else:
        print("No crossover found in range [0.01, 100]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
