"""
disease_reasoning.py — Orchestrator for the crop disease reasoning pipeline.

Delegates to:
  - feature_extractor.py  → extract visual features from image
  - rule_engine.py         → score diseases, resolve conflicts, explain rejections
  - kb_loader.py           → knowledge base queries

Public API (unchanged — backward compatible):
  - reason_diagnosis(image_bgr, classifier_result, crop_type) → DiagnosisResult
  - diagnosis_to_dict(d) → dict
"""

import numpy as np
from dataclasses import dataclass, field
from loguru import logger

from agridrone.knowledge import kb_loader
from agridrone.knowledge.kb_loader import DiseaseProfile
from agridrone.vision.feature_extractor import extract_features, ImageFeatures
from agridrone.vision.rule_engine import evaluate as rule_evaluate, result_to_dict as rule_result_to_dict, RuleEngineResult
from agridrone.core.spectral_features import extract_spectral_indices, spectral_to_dict, SpectralResult


# ════════════════════════════════════════════════════════════════
# Public data structures (kept for backward compatibility)
# ════════════════════════════════════════════════════════════════

@dataclass
class DiagnosisResult:
    """Complete diagnosis with reasoning chain."""
    disease_key: str
    disease_name: str
    confidence: float
    health_score: int
    risk_level: str
    symptoms_detected: list[str]
    symptoms_matched: list[str]
    reasoning_chain: list[str]
    differential: list[dict]
    treatment: list[str]
    urgency: str
    yield_loss: str
    affected_parts: list[str]
    # New fields from rule engine (optional for backward compat)
    conflict: dict | None = None
    rejections: list[dict] = field(default_factory=list)
    rule_engine_detail: dict | None = None
    spectral_analysis: dict | None = None


@dataclass
class PipelineOutput:
    """Full pipeline output including intermediates for LLM validation."""
    diagnosis: DiagnosisResult
    features: ImageFeatures
    rule_result: RuleEngineResult
    conflict: dict | None = None
    rejections: list[dict] = field(default_factory=list)
    rule_engine_detail: dict | None = None
    spectral: SpectralResult | None = None


# ════════════════════════════════════════════════════════════════
# Main entry point
# ════════════════════════════════════════════════════════════════

def reason_diagnosis(
    image_bgr: np.ndarray,
    classifier_result: dict | None,
    crop_type: str = "wheat",
) -> DiagnosisResult:
    """
    Main reasoning function — orchestrates feature extraction → rule engine → explanation.
    Backward-compatible: returns only DiagnosisResult.
    """
    output = run_full_pipeline(image_bgr, classifier_result, crop_type)
    return output.diagnosis


def run_full_pipeline(
    image_bgr: np.ndarray,
    classifier_result: dict | None,
    crop_type: str = "wheat",
) -> PipelineOutput:
    """
    Full pipeline returning DiagnosisResult + intermediate objects (features, rule_result)
    for use by the LLM validator.
    """
    # ── Step 1: Extract features ──
    kb = kb_loader.get_all_profiles()
    if not kb:
        kb_loader.load()
        kb = kb_loader.get_all_profiles()

    features = extract_features(image_bgr, kb)

    logger.info(
        f"Features: {len(features.color_confidences)} color sigs, "
        f"stripe={features.has_stripe_pattern}, spots={features.has_spot_pattern}, "
        f"vivid={features.has_vivid_yellow}, green={features.green_ratio:.1%}"
    )

    # ── Step 1b: Extract spectral vegetation indices ──
    spectral = None
    try:
        spectral = extract_spectral_indices(image_bgr)
        logger.info(
            f"Spectral: VARI={spectral.indices.get('VARI', None) and spectral.indices['VARI'].mean:.3f}, "
            f"RGRI={spectral.indices.get('RGRI', None) and spectral.indices['RGRI'].mean:.3f}, "
            f"stress={spectral.stress_level}"
        )
    except Exception as sp_exc:
        logger.warning(f"Spectral extraction failed (non-critical): {sp_exc}")

    # ── Step 2: Run rule engine ──
    engine_result = rule_evaluate(features, classifier_result, crop_type, spectral=spectral)

    if not engine_result.candidates:
        diagnosis = DiagnosisResult(
            disease_key="healthy",
            disease_name="Healthy Crop",
            confidence=0.5,
            health_score=85,
            risk_level="low",
            symptoms_detected=[],
            symptoms_matched=[],
            reasoning_chain=[
                "Step 1: Analyzed image for disease color signatures — none significant found",
                "Step 2: Classifier predictions suggest healthy crop",
                "Conclusion: No clear disease symptoms detected",
            ],
            differential=[],
            treatment=["No treatment needed — continue regular monitoring"],
            urgency="within_30_days",
            yield_loss="0%",
            affected_parts=[],
        )
        return PipelineOutput(diagnosis=diagnosis, features=features, rule_result=engine_result, spectral=spectral)

    # ── Step 3: Build output from rule engine result ──
    top_cand = engine_result.candidates[0]
    top_key = top_cand.disease_key
    top_score = top_cand.final_score
    top_profile = kb[top_key]

    # Symptoms detected (from features)
    symptoms_detected = []
    for k, v in features.color_confidences.items():
        if v > 0.1:
            sig_name = k.split(":")[1] if ":" in k else k
            symptoms_detected.append(f"Color: {sig_name.replace('_', ' ')}")
    if features.has_stripe_pattern:
        symptoms_detected.append(f"Texture: linear stripe pattern ({features.stripe_confidence:.0%})")
    if features.has_spot_pattern:
        symptoms_detected.append(f"Texture: discrete spots ({features.spot_confidence:.0%})")
    if features.has_vivid_yellow:
        symptoms_detected.append(f"Texture: vivid yellow-orange ({features.vivid_yellow_orange_ratio:.1%})")
    if features.has_bleaching:
        symptoms_detected.append(f"Texture: bleaching ({features.bleaching_ratio:.1%})")

    # Spectral stress symptoms
    if spectral and spectral.stress_detected:
        for sig in spectral.stress_signals[:3]:
            symptoms_detected.append(f"Spectral: {sig}")

    # Symptoms matched (from rule matches for top disease)
    symptoms_matched = [
        m.explanation for m in top_cand.matches if m.score_delta > 0
    ]

    # Build differential (other possibilities) — enriched with KB differential rules
    differential = []
    for cand in engine_result.candidates[1:4]:
        diff_rule = kb_loader.get_differential(top_key, cand.disease_key)
        if diff_rule:
            key_diff = diff_rule.rule
        else:
            p = kb.get(cand.disease_key)
            if p and p.symptoms:
                key_diff = f"Would expect: {p.symptoms[0]}"
            else:
                key_diff = "Different symptom pattern"
        differential.append({
            "disease": cand.disease_name,
            "confidence": round(cand.final_score, 3),
            "key_difference": key_diff,
        })

    # Health score from severity + confidence
    health_score = max(5, round(100 - top_profile.severity * 100 * min(top_score, 1.0)))

    # Risk level
    risk_level = (
        "low" if health_score >= 70
        else "medium" if health_score >= 40
        else "high" if health_score >= 20
        else "critical"
    )

    # ── Step 4: Build reasoning chain ──
    reasoning = [
        f"Step 1 — OBSERVE: Analyzed image for visual disease markers "
        f"({len(features.color_confidences)} color patterns, "
        f"stripe={features.has_stripe_pattern}, spots={features.has_spot_pattern})",
    ]

    if symptoms_matched:
        reasoning.append(
            f"Step 2 — SYMPTOMS FOUND: {'; '.join(symptoms_matched[:3])}"
        )
    else:
        reasoning.append(f"Step 2 — CLASSIFIER: Top prediction is {top_profile.display_name}")

    reasoning.append(
        f"Step 3 — MATCH: Symptoms best match {top_profile.display_name} "
        f"(severity: {top_profile.severity:.0%}, yield loss: {top_profile.yield_loss_pct})"
    )

    # Conflict explanation
    if engine_result.conflict and engine_result.conflict.winner != "agree":
        c = engine_result.conflict
        reasoning.append(
            f"Step 4 — CONFLICT RESOLVED: {c.reason}"
        )

    # Rejection explanations for top YOLO pick if it was overridden
    if engine_result.conflict and engine_result.conflict.winner == "rules":
        yolo_key = engine_result.conflict.yolo_prediction
        for rej in engine_result.rejections:
            if rej.disease_key == yolo_key and rej.reasons:
                reasoning.append(
                    f"Step 5 — REJECTED {rej.disease_name}: {'; '.join(rej.reasons[:2])}"
                )
                break

    # Confusion note (backward compat)
    confusion_note = ""
    if classifier_result and classifier_result.get("top5"):
        cls_top = classifier_result["top5"][0]
        cls_name = cls_top.get("class_key", "")
        if cls_name != top_key and cls_name in (top_profile.confusion_diseases or []):
            confusion_note = (
                f"Note: Classifier initially suggested {cls_top.get('class_name', cls_name)}, "
                f"which is a common confusion with {top_profile.display_name}. "
                f"Visual symptom analysis corrected the diagnosis."
            )
            if not any("CONFLICT" in r or "REJECTED" in r for r in reasoning):
                reasoning.append(f"Step 4 — CORRECTION: {confusion_note}")

    step_n = len(reasoning) + 1
    reasoning.append(
        f"Step {step_n} — DIAGNOSIS: {top_profile.display_name} "
        f"with {top_score:.0%} confidence. {top_profile.urgency.replace('_', ' ').title()} action needed."
    )

    # Serialize conflict and rejections for the response
    conflict_dict = None
    if engine_result.conflict and engine_result.conflict.winner != "agree":
        c = engine_result.conflict
        conflict_dict = {
            "yolo_prediction": c.yolo_prediction,
            "yolo_confidence": round(c.yolo_confidence, 3),
            "rule_prediction": c.rule_prediction,
            "winner": c.winner,
            "reason": c.reason,
        }

    rejection_dicts = [
        {
            "disease": r.disease_name,
            "reasons": r.reasons,
            "missing_features": r.missing_features,
            "contradicting_features": r.contradicting_features,
        }
        for r in engine_result.rejections[:5]
    ]

    diagnosis = DiagnosisResult(
        disease_key=top_key,
        disease_name=top_profile.display_name,
        confidence=round(min(top_score, 1.0), 3),
        health_score=health_score,
        risk_level=risk_level,
        symptoms_detected=symptoms_detected,
        symptoms_matched=symptoms_matched,
        reasoning_chain=reasoning,
        differential=differential,
        treatment=top_profile.treatment,
        urgency=top_profile.urgency,
        yield_loss=top_profile.yield_loss_pct,
        affected_parts=top_profile.affected_parts,
        conflict=conflict_dict,
        rejections=rejection_dicts,
        rule_engine_detail=rule_result_to_dict(engine_result),
        spectral_analysis=spectral_to_dict(spectral) if spectral else None,
    )

    return PipelineOutput(
        diagnosis=diagnosis,
        features=features,
        rule_result=engine_result,
        spectral=spectral,
    )


def diagnosis_to_dict(d: DiagnosisResult) -> dict:
    """Convert DiagnosisResult to JSON-serializable dict."""
    result = {
        "disease_key": d.disease_key,
        "disease_name": d.disease_name,
        "confidence": d.confidence,
        "health_score": d.health_score,
        "risk_level": d.risk_level,
        "symptoms_detected": d.symptoms_detected,
        "symptoms_matched": d.symptoms_matched,
        "reasoning_chain": d.reasoning_chain,
        "differential_diagnosis": d.differential,
        "treatment": d.treatment,
        "urgency": d.urgency,
        "yield_loss": d.yield_loss,
        "affected_parts": d.affected_parts,
    }
    # New fields — only include if present
    if d.conflict:
        result["conflict"] = d.conflict
    if d.rejections:
        result["rejections"] = d.rejections
    if d.rule_engine_detail:
        result["rule_engine_detail"] = d.rule_engine_detail
    if d.spectral_analysis:
        result["spectral_analysis"] = d.spectral_analysis
    return result
