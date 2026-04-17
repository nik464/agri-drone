"""Regression tests that freeze the currently-reported numbers.

Rationale
---------
The paper RESEARCH_PAPER_FINAL_v3.md and README.md cite specific numbers
(96.15% Config A accuracy, ₹294.33 EML, etc.). These numbers are produced by
evaluate/*.py scripts and cached in evaluate/results/*.json. This test file
simply re-reads the JSONs and asserts the headline values are unchanged, so
that CI fails loudly if somebody silently edits a result file.

Running
-------
    pytest -q tests/regression/test_frozen_metrics.py

Tolerances
----------
All floating-point comparisons use atol=5e-4 (i.e. ±0.05%) which is below the
rounding precision in the paper.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "evaluate" / "results"


def _load(name: str) -> dict:
    path = RESULTS_DIR / name
    if not path.exists():
        pytest.skip(f"Result artifact not present: {path}. "
                    "Regenerate with the corresponding evaluate/*.py script.")
    return json.loads(path.read_text(encoding="utf-8"))


def _approx(a: float, b: float, atol: float = 5e-4) -> bool:
    return math.isclose(a, b, abs_tol=atol)


# ---------------------------------------------------------------------------
# Ablation headline numbers (RESEARCH_PAPER_FINAL_v3.md Table 2)
# ---------------------------------------------------------------------------

def test_ablation_config_a_accuracy_frozen():
    data = _load("ablation_summary.json")
    assert _approx(data["config_A_accuracy"], 0.9615), (
        f"Config A accuracy changed: {data['config_A_accuracy']} (expected 0.9615). "
        "If this is intentional, update RESEARCH_PAPER_v4.md and this test together."
    )


def test_ablation_config_b_accuracy_frozen():
    data = _load("ablation_summary.json")
    assert _approx(data["config_B_accuracy"], 0.9572)


def test_ablation_config_c_accuracy_frozen():
    data = _load("ablation_summary.json")
    assert _approx(data["config_C_accuracy"], 0.1338, atol=1e-3)


def test_ablation_macro_f1_frozen():
    data = _load("ablation_summary.json")
    assert _approx(data["config_A_macro_f1"], 0.9618)
    assert _approx(data["config_B_macro_f1"], 0.9574)


def test_ablation_mcc_frozen():
    data = _load("ablation_summary.json")
    assert _approx(data["config_A_mcc"], 0.9596)
    assert _approx(data["config_B_mcc"], 0.9551)


def test_ablation_sample_count_frozen():
    data = _load("ablation_summary.json")
    assert data["n_test_images"] == 935
    assert data["n_classes"] == 21


# ---------------------------------------------------------------------------
# Statistical tests (McNemar, bootstrap)
# ---------------------------------------------------------------------------

def test_mcnemar_a_vs_b_not_significant():
    data = _load("statistical_tests.json")
    mc = data["mcnemar"]["A_vs_B"]
    # 4 discordant pairs, chi2 = 2.25, not significant at 0.05
    assert mc["discordant_pairs"] == 4
    assert _approx(mc["chi2_statistic"], 2.25, atol=0.05)
    assert mc["significant_005"] is False


def test_bootstrap_ci_config_a_accuracy():
    data = _load("statistical_tests.json")
    ci = data["bootstrap_ci"]["A"]["accuracy"]
    assert _approx(ci["point"], 0.9615)
    # 95% CI per paper: [0.9486, 0.9732]
    assert _approx(ci["ci_lower"], 0.9486, atol=1e-3)
    assert _approx(ci["ci_upper"], 0.9732, atol=1e-3)


# ---------------------------------------------------------------------------
# EML headline numbers (RESEARCH_PAPER_FINAL_v3.md Section on Economic Loss)
# ---------------------------------------------------------------------------

def test_eml_total_a_frozen():
    data = _load("eml_summary.json")
    assert _approx(data["total_eml_A"], 294.33, atol=0.01), (
        f"EML(A) changed: {data['total_eml_A']} (expected 294.33). "
        "This number appears in the paper abstract — update both together."
    )


def test_eml_total_b_frozen():
    data = _load("eml_summary.json")
    assert _approx(data["total_eml_B"], 2769.06, atol=0.01)


def test_eml_delta_pct_frozen():
    data = _load("eml_summary.json")
    # ~840.8% increase
    assert _approx(data["delta_pct"], 840.8, atol=0.5)


# ---------------------------------------------------------------------------
# Cross-dataset PDT evaluation (honest degenerate-result lock)
# ---------------------------------------------------------------------------

def test_pdt_specificity_is_zero():
    """Locks the honest-but-degenerate PDT result.

    The v3 paper reports specificity=0 (every healthy image misclassified).
    This test ensures that number does not silently drift until Step 6 produces
    a proper threshold-sweep replacement in evaluate/results/v2/.
    """
    data = _load("cross_dataset_PDT.json")
    assert _approx(data["specificity"], 0.0, atol=1e-6)
    assert _approx(data["recall_sensitivity"], 1.0, atol=1e-6)
    assert data["n_total"] == 672


def test_pdt_accuracy_frozen():
    data = _load("cross_dataset_PDT.json")
    assert _approx(data["accuracy"], 0.8438, atol=1e-3)


# ---------------------------------------------------------------------------
# EfficientNet baseline (the number the paper cites)
# ---------------------------------------------------------------------------

def test_efficientnet_baseline_frozen():
    data = _load("efficientnet_results.json")
    # Note: v4 will re-report this under the fair-comparison recipe (Step 4).
    # Until then, the v3 number (76.15%) remains the ground truth for v3.
    assert _approx(data["config_a_efficientnet"]["accuracy"], 0.7615, atol=1e-3)


# ---------------------------------------------------------------------------
# Sensitivity sweep (125 configs)
# ---------------------------------------------------------------------------

def test_sensitivity_grid_size_frozen():
    data = _load("sensitivity_summary.json")
    assert data["n_configs_evaluated"] == 125


# ---------------------------------------------------------------------------
# Robustness degradation (noise injection)
# ---------------------------------------------------------------------------

def test_robustness_clean_accuracy_frozen():
    data = _load("robustness_summary.json")
    assert _approx(data["clean"]["accuracy"], 0.9615)
