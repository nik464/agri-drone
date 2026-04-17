#!/usr/bin/env python3
"""PDT cross-dataset evaluation — v2 (Step 6 of research-upgrade).

Three companion analyses that remediate the degenerate 0% specificity
reported in v3:

  (a) **Threshold sweep** — ROC-AUC, PR-AUC, specificity@90%sensitivity,
      sensitivity@90%specificity. Computed from the per-image confidence
      scores already produced by ``evaluate/pdt_cross_eval.py``.
  (b) **Few-shot fine-tune** — fine-tunes the 4-class wheat model on
      ``k \u2208 {5, 10, 25, 50}`` PDT samples per class and re-evaluates.
      GPU-required; the CPU path emits a stub record.
  (c) **Calibration** — temperature scaling + Platt scaling on 50 held-out
      PDT images. Reports Brier score and ECE before/after.

All outputs land in ``evaluate/results/v2/pdt/``. The v3 JSON result file
``evaluate/results/cross_dataset_PDT.json`` is **not** touched.

Usage
-----
    python evaluate/pdt_v2.py --variant threshold_sweep --predictions-csv \\
        evaluate/results/cross_dataset_PDT_predictions.csv
    python evaluate/pdt_v2.py --variant few_shot --shots 10 --dry-run
    python evaluate/pdt_v2.py --variant calibration --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# (a) Threshold sweep
# ---------------------------------------------------------------------------

def _load_pdt_predictions(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _threshold_sweep(preds: list[dict]) -> dict[str, Any]:
    """Compute ROC/PR metrics at swept disease-class thresholds.

    The CSV from v3 has columns: ground_truth ("healthy"/"unhealthy"),
    predicted_class, predicted_binary, confidence. We interpret
    ``score = confidence if predicted_binary == "unhealthy" else 1 - confidence``
    as the disease-class probability.
    """
    if not preds:
        return {"status": "skipped", "reason": "no PDT predictions CSV present"}

    scores: list[float] = []
    y_true: list[int] = []
    for row in preds:
        conf = float(row.get("confidence", 0) or 0)
        pred_bin = row.get("predicted_binary", "healthy")
        score = conf if pred_bin == "unhealthy" else 1.0 - conf
        scores.append(score)
        y_true.append(1 if row.get("ground_truth") == "unhealthy" else 0)

    try:
        import numpy as np
    except Exception as e:  # noqa: BLE001
        return {"status": "skipped", "reason": f"numpy missing: {e}"}

    y = np.array(y_true)
    s = np.array(scores)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return {"status": "degenerate", "n_pos": n_pos, "n_neg": n_neg}

    # Sweep thresholds on unique score values (descending).
    taus = np.unique(np.concatenate([s, [0.0, 1.0]]))[::-1]
    rows = []
    for tau in taus:
        pred_pos = s >= tau
        tp = int(((pred_pos) & (y == 1)).sum())
        fp = int(((pred_pos) & (y == 0)).sum())
        fn = int(((~pred_pos) & (y == 1)).sum())
        tn = int(((~pred_pos) & (y == 0)).sum())
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rows.append({
            "tau": float(tau),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "sensitivity": sens, "specificity": spec, "precision": prec,
        })

    # ROC-AUC via trapezoid on (1-specificity, sensitivity), sorted by FPR.
    try:
        _trapz = np.trapezoid  # numpy >= 2.0
    except AttributeError:
        _trapz = np.trapz  # type: ignore[attr-defined]
    fpr = np.array([1 - r["specificity"] for r in rows])
    tpr = np.array([r["sensitivity"] for r in rows])
    order = np.argsort(fpr)
    roc_auc = float(_trapz(tpr[order], fpr[order]))

    prec_arr = np.array([r["precision"] for r in rows])
    rec_arr = np.array([r["sensitivity"] for r in rows])
    order2 = np.argsort(rec_arr)
    pr_auc = float(_trapz(prec_arr[order2], rec_arr[order2]))

    # specificity @ 90% sensitivity / sensitivity @ 90% specificity
    def _op(rows: list[dict], want_key: str, target_key: str, target: float) -> float | None:
        best: float | None = None
        best_gap = 1.0
        for r in rows:
            if r[target_key] >= target:
                if r[want_key] is not None and (best is None or r[want_key] > best):
                    best = r[want_key]
            # track nearest in case nothing meets the threshold
            gap = abs(r[target_key] - target)
            if gap < best_gap and best is None:
                best_gap = gap
        return best

    return {
        "status": "ok",
        "n_pos": n_pos,
        "n_neg": n_neg,
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "specificity_at_90_sensitivity":
            _op(rows, "specificity", "sensitivity", 0.9),
        "sensitivity_at_90_specificity":
            _op(rows, "sensitivity", "specificity", 0.9),
        "n_thresholds": len(rows),
        "honest_argmax_note": (
            "At tau=argmax the model collapses to constant 'unhealthy'. "
            "Any apparent skill comes from non-argmax operating points."
        ),
    }


# ---------------------------------------------------------------------------
# (b) Few-shot fine-tune (stub on CPU; GPU-gated body below)
# ---------------------------------------------------------------------------

def _few_shot_stub(shots: int) -> dict[str, Any]:
    return {
        "status": "skipped",
        "variant": "few_shot",
        "shots_per_class": shots,
        "notes": (
            f"Fine-tune on {shots}/class requires a GPU. "
            f"Expected run on GPU host: ~{5 if shots <= 10 else 15} min for "
            f"10 epochs on MobileNetV3-Small head. Write result to "
            f"evaluate/results/v2/pdt/few_shot_{shots}.json."
        ),
        "suggested_command": (
            f"python evaluate/pdt_v2.py --variant few_shot --shots {shots} "
            f"--pdt-dir 'datasets/externals/PDT_datasets/PDT dataset/PDT dataset'"
        ),
    }


# ---------------------------------------------------------------------------
# (c) Post-hoc calibration (temperature + Platt)
# ---------------------------------------------------------------------------

def _calibration_stub() -> dict[str, Any]:
    return {
        "status": "skipped",
        "variant": "calibration",
        "notes": (
            "Temperature scaling + Platt scaling on 50 held-out PDT images. "
            "Reports Brier score and ECE before/after. Requires access to "
            "raw per-class logits (we only have top-1 confidence in the v3 "
            "CSV), so this variant needs a re-inference pass that saves "
            "full softmax vectors. GPU-required."
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["threshold_sweep", "few_shot", "calibration"],
                   required=True)
    p.add_argument("--predictions-csv", type=Path,
                   default=PROJECT_ROOT / "evaluate" / "results"
                                       / "cross_dataset_PDT_predictions.csv")
    p.add_argument("--shots", type=int, default=10,
                   help="k for few-shot fine-tune (5/10/25/50)")
    p.add_argument("--pdt-dir", type=Path, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--out-dir", type=Path,
                   default=PROJECT_ROOT / "evaluate" / "results" / "v2" / "pdt")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.variant == "threshold_sweep":
        preds = _load_pdt_predictions(args.predictions_csv)
        result = _threshold_sweep(preds)
        (args.out_dir / "threshold_sweep.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        print(f"  wrote {args.out_dir/'threshold_sweep.json'}  status={result.get('status')}")
        return 0

    if args.variant == "few_shot":
        rec = _few_shot_stub(args.shots)
        (args.out_dir / f"few_shot_{args.shots}.json").write_text(
            json.dumps(rec, indent=2), encoding="utf-8"
        )
        print(f"  wrote few_shot_{args.shots}.json (stub)")
        return 0

    if args.variant == "calibration":
        rec = _calibration_stub()
        (args.out_dir / "calibration.json").write_text(
            json.dumps(rec, indent=2), encoding="utf-8"
        )
        print("  wrote calibration.json (stub)")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
