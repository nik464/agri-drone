"""
detector.py — Uncertainty-aware inference wrapper for the YOLOv8 classifier.

Uses Monte-Carlo Dropout (MC-Dropout) at inference time: the model is
switched to train() mode so that dropout layers stay active, but no
gradients are computed.  Running the same image N times produces a
distribution of predictions whose spread quantifies epistemic uncertainty.
"""

import json
import time
import threading
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from loguru import logger

# ── Constants ──
_DEFAULT_N_FORWARD = 20
_UNCERTAIN_STD_THRESHOLD = 0.12
_ACTIVE_LEARNING_PATH = Path("data/uncertain_cases.jsonl")


@dataclass
class UncertaintyResult:
    """Output of predict_with_uncertainty()."""
    mean_confidence: float
    std_confidence: float
    confidence_interval: list[float]   # [low, high] 95% CI
    prediction_consistency: float      # fraction of runs agreeing on top class
    is_uncertain: bool
    uncertainty_reason: str
    top_class: str
    top_class_display: str
    n_forward_passes: int
    elapsed_ms: float


_file_lock = threading.Lock()


def predict_with_uncertainty(
    model,
    image_bgr: "np.ndarray",
    n_forward: int = _DEFAULT_N_FORWARD,
    class_display: dict | None = None,
    class_severity: dict | None = None,
) -> UncertaintyResult:
    """
    Run MC-Dropout uncertainty quantification on the YOLOv8 classifier.

    Parameters
    ----------
    model : ultralytics.YOLO
        A loaded YOLOv8 classification model (task="classify").
    image_bgr : np.ndarray
        BGR image (OpenCV format).
    n_forward : int
        Number of forward passes (default 20).
    class_display : dict | None
        Optional {class_key: "Display Name"} mapping.
    class_severity : dict | None
        Optional {class_key: float} mapping (unused here, reserved).

    Returns
    -------
    UncertaintyResult with statistics across N stochastic forward passes.
    """
    t0 = time.time()

    names = model.names  # {0: 'class_key', ...}
    class_display = class_display or {}

    # Grab the underlying PyTorch module
    torch_model = model.model
    was_training = torch_model.training

    confidences = []
    predictions = []

    try:
        # Enable dropout (train mode) but disable gradients
        torch_model.train()

        with torch.no_grad():
            for _ in range(n_forward):
                results = model(image_bgr, verbose=False)
                if not results or results[0].probs is None:
                    continue
                probs = results[0].probs
                top_idx = probs.top1
                top_conf = float(probs.top1conf)
                class_key = names[top_idx]
                confidences.append(top_conf)
                predictions.append(class_key)
    finally:
        # Restore original mode
        if not was_training:
            torch_model.eval()

    elapsed = (time.time() - t0) * 1000

    if not confidences:
        return UncertaintyResult(
            mean_confidence=0.0,
            std_confidence=1.0,
            confidence_interval=[0.0, 0.0],
            prediction_consistency=0.0,
            is_uncertain=True,
            uncertainty_reason="Model produced no output across all forward passes",
            top_class="unknown",
            top_class_display="Unknown",
            n_forward_passes=n_forward,
            elapsed_ms=round(elapsed, 1),
        )

    arr = np.array(confidences)
    mean_conf = float(np.mean(arr))
    std_conf = float(np.std(arr))

    # 95% confidence interval (mean ± 1.96σ)
    ci_low = max(0.0, mean_conf - 1.96 * std_conf)
    ci_high = min(1.0, mean_conf + 1.96 * std_conf)

    # Most frequent prediction
    from collections import Counter
    pred_counts = Counter(predictions)
    top_class, top_count = pred_counts.most_common(1)[0]
    consistency = top_count / len(predictions)

    # Uncertainty decision
    is_uncertain = std_conf > _UNCERTAIN_STD_THRESHOLD or consistency < 0.6

    # Human-readable reason
    reasons = []
    if std_conf > _UNCERTAIN_STD_THRESHOLD:
        reasons.append(f"high variance across runs (σ={std_conf:.3f})")
    if consistency < 0.6:
        n_classes = len(pred_counts)
        reasons.append(
            f"inconsistent predictions ({n_classes} different classes across {len(predictions)} runs, "
            f"top class only {consistency:.0%})"
        )
    if mean_conf < 0.5:
        reasons.append(f"low mean confidence ({mean_conf:.1%})")

    if not reasons:
        uncertainty_reason = "Prediction is stable and confident"
    else:
        uncertainty_reason = "Low confidence — " + "; ".join(reasons) + \
            ". Please photograph from 30cm distance in good light."

    display = class_display.get(top_class, top_class.replace("_", " ").title())

    return UncertaintyResult(
        mean_confidence=round(mean_conf, 4),
        std_confidence=round(std_conf, 4),
        confidence_interval=[round(ci_low, 4), round(ci_high, 4)],
        prediction_consistency=round(consistency, 4),
        is_uncertain=is_uncertain,
        uncertainty_reason=uncertainty_reason,
        top_class=top_class,
        top_class_display=display,
        n_forward_passes=n_forward,
        elapsed_ms=round(elapsed, 1),
    )


def flag_uncertain_case(
    uncertainty: UncertaintyResult,
    image_hash: str,
    classifier_result: dict | None = None,
    filename: str = "",
) -> None:
    """
    Append an uncertain case to data/uncertain_cases.jsonl for active learning.
    Only called when uncertainty.is_uncertain is True.
    """
    _ACTIVE_LEARNING_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "image_hash": image_hash,
        "filename": filename,
        "top_class": uncertainty.top_class,
        "mean_confidence": uncertainty.mean_confidence,
        "std_confidence": uncertainty.std_confidence,
        "prediction_consistency": uncertainty.prediction_consistency,
        "uncertainty_reason": uncertainty.uncertainty_reason,
        "classifier_top": classifier_result.get("top_prediction") if classifier_result else None,
        "classifier_conf": classifier_result.get("top_confidence") if classifier_result else None,
    }
    try:
        with _file_lock:
            with open(_ACTIVE_LEARNING_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info(f"Flagged uncertain case: {image_hash[:8]} ({uncertainty.top_class}, σ={uncertainty.std_confidence:.3f})")
    except Exception as e:
        logger.warning(f"Failed to log uncertain case: {e}")


def uncertainty_to_dict(u: UncertaintyResult) -> dict:
    """Serialize UncertaintyResult to a JSON-friendly dict."""
    return asdict(u)
