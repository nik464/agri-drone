"""
crop_type_gate.py — Layer 2: Crop-Type Gate.

Determines whether an image contains wheat, rice, or an unknown/unsupported
crop.  Uses the existing 21-class YOLOv8n-cls softmax distribution grouped by
crop family, combined with entropy-based OOD rejection.

Design:
  * The 21-class classifier always produces a softmax distribution that sums
    to 1.0 across all wheat + rice classes.
  * For IN-DISTRIBUTION images (real wheat or rice), probability mass is
    heavily concentrated in one crop family (>0.85 for a single group).
  * For OUT-OF-DISTRIBUTION images (corn, tomato, non-crop objects that
    slipped past Layer 1), the model produces:
      - Lower top-1 confidence (spreading mass across classes),
      - Higher Shannon entropy,
      - Cross-group confusion (top-5 spans both wheat and rice).
  * We combine crop-group dominance + entropy + cross-group confusion
    into a single accept/reject decision.

Return:
    CropGateResult with crop_type ('wheat' | 'rice' | 'unknown'),
    confidence, and diagnostic info.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from loguru import logger

# ── Crop family definitions (must match classifier training classes) ──
WHEAT_CLASSES: frozenset[str] = frozenset({
    "healthy_wheat",
    "wheat_fusarium_head_blight",
    "wheat_yellow_rust",
    "wheat_black_rust",
    "wheat_brown_rust",
    "wheat_leaf_blight",
    "wheat_powdery_mildew",
    "wheat_septoria",
    "wheat_tan_spot",
    "wheat_smut",
    "wheat_root_rot",
    "wheat_blast",
    "wheat_aphid",
    "wheat_mite",
    "wheat_stem_fly",
})  # 15 classes

RICE_CLASSES: frozenset[str] = frozenset({
    "healthy_rice",
    "rice_bacterial_blight",
    "rice_brown_spot",
    "rice_blast",
    "rice_leaf_scald",
    "rice_sheath_blight",
})  # 6 classes

# Thresholds (calibrated on held-out in-distribution images)
CROP_DOMINANCE_THRESHOLD = 0.70    # min probability mass in dominant crop group
MAX_ENTROPY_RATIO = 0.75           # max normalised entropy (0=certain, 1=uniform)
MIN_TOP1_CONFIDENCE = 0.25         # absolute minimum for top-1 class
MAX_CROSS_GROUP_TOP5 = 2           # max classes from non-dominant group in top-5


@dataclass
class CropGateResult:
    """Output of the crop-type gate (Layer 2)."""
    crop_type: str                   # 'wheat' | 'rice' | 'unknown'
    accepted: bool                   # True if crop type is supported
    confidence: float                # dominant group's probability sum
    wheat_prob: float = 0.0
    rice_prob: float = 0.0
    entropy: float = 0.0            # Shannon entropy of full distribution
    normalised_entropy: float = 0.0  # entropy / log(n_classes)
    top1_class: str = ""
    top1_confidence: float = 0.0
    cross_group_top5: int = 0        # how many of top-5 from non-dominant group
    reason: str = ""


def _shannon_entropy(probs: list[float]) -> float:
    """Compute Shannon entropy H = -sum(p * log2(p)) for non-zero probs."""
    return -sum(p * math.log2(p) for p in probs if p > 1e-10)


def classify_crop_type(
    class_names: dict[int, str],
    probs_tensor,
) -> CropGateResult:
    """Run the crop-type gate on a classification result.

    Args:
        class_names: {index: class_key} from the YOLO model.
        probs_tensor: The ``results[0].probs`` object from ultralytics,
                      exposing ``.data`` (full softmax), ``.top5``, ``.top5conf``.

    Returns:
        CropGateResult indicating wheat, rice, or unknown.
    """
    # Extract full softmax distribution
    full_probs = probs_tensor.data.cpu().tolist()
    n_classes = len(full_probs)

    # Sum probabilities per crop group
    wheat_prob = 0.0
    rice_prob = 0.0
    for idx, p in enumerate(full_probs):
        cls_name = class_names.get(idx, "")
        if cls_name in WHEAT_CLASSES:
            wheat_prob += p
        elif cls_name in RICE_CLASSES:
            rice_prob += p

    # Top-1 info
    top5_indices = probs_tensor.top5
    top5_confs = probs_tensor.top5conf.tolist()
    top1_class = class_names.get(top5_indices[0], "unknown")
    top1_conf = top5_confs[0] if top5_confs else 0.0

    # Determine dominant crop group
    if wheat_prob >= rice_prob:
        dominant_group = "wheat"
        dominant_prob = wheat_prob
        dominant_classes = WHEAT_CLASSES
    else:
        dominant_group = "rice"
        dominant_prob = rice_prob
        dominant_classes = RICE_CLASSES

    # Cross-group confusion: count top-5 classes NOT in dominant group
    cross_group = 0
    for idx in top5_indices:
        cls_name = class_names.get(idx, "")
        if cls_name not in dominant_classes:
            cross_group += 1

    # Shannon entropy (normalised to [0, 1])
    entropy = _shannon_entropy(full_probs)
    max_entropy = math.log2(n_classes) if n_classes > 1 else 1.0
    norm_entropy = entropy / max_entropy

    # ── Decision logic ──
    accepted = True
    reason = ""

    if top1_conf < MIN_TOP1_CONFIDENCE:
        accepted = False
        reason = (
            f"Low classifier confidence ({top1_conf:.2f} < {MIN_TOP1_CONFIDENCE}). "
            "Image may not contain a supported crop (wheat or rice)."
        )
    elif dominant_prob < CROP_DOMINANCE_THRESHOLD:
        accepted = False
        reason = (
            f"Crop-type ambiguous: wheat_prob={wheat_prob:.2f}, rice_prob={rice_prob:.2f}. "
            f"Neither group dominates (threshold={CROP_DOMINANCE_THRESHOLD}). "
            "Image may not be wheat or rice."
        )
    elif norm_entropy > MAX_ENTROPY_RATIO:
        accepted = False
        reason = (
            f"High prediction entropy ({norm_entropy:.2f} > {MAX_ENTROPY_RATIO}). "
            "Model is unsure — image may be out-of-distribution."
        )
    elif cross_group > MAX_CROSS_GROUP_TOP5:
        accepted = False
        reason = (
            f"Cross-crop confusion: {cross_group} of top-5 predictions from "
            f"non-{dominant_group} group. Image may not be a clear {dominant_group} sample."
        )

    crop_type = dominant_group if accepted else "unknown"

    result = CropGateResult(
        crop_type=crop_type,
        accepted=accepted,
        confidence=dominant_prob,
        wheat_prob=round(wheat_prob, 4),
        rice_prob=round(rice_prob, 4),
        entropy=round(entropy, 4),
        normalised_entropy=round(norm_entropy, 4),
        top1_class=top1_class,
        top1_confidence=round(top1_conf, 4),
        cross_group_top5=cross_group,
        reason=reason,
    )

    logger.info(
        f"Crop-type gate: {crop_type} (wheat={wheat_prob:.2f} rice={rice_prob:.2f} "
        f"H_norm={norm_entropy:.2f} cross={cross_group} top1={top1_class}@{top1_conf:.2f}) "
        f"→ {'PASS' if accepted else 'REJECT'}"
    )
    return result
