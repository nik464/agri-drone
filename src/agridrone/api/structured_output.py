"""
structured_output.py — Build the final structured API response (Task Group D).

Takes outputs from ALL pipeline stages and produces a single, clean, frontend-friendly
JSON structure. This is the "final word" — the frontend can render from this one object
without needing priority fallback logic.

Input sources:
  - Classifier result (YOLO-cls top-5)
  - Rule engine result (scored candidates, conflicts, rejections)
  - Reasoning engine result (diagnosis, chain, differential)
  - LLM validation (agree/disagree, agreement score, scenario)
  - Confidence fusion (weighted multi-signal confidence)

Output: A single dict with clean sections:
  - diagnosis: disease name + fused confidence + grade
  - health: score + risk + urgency + yield loss
  - confidence_breakdown: per-source scores + weights
  - reasoning: step-by-step chain
  - evidence: supporting + contradicting features
  - rejected: diseases ruled out with reasons
  - differential: alternative diagnoses
  - treatment: recommendations + urgency + products
  - metadata: models used, agreement, timing, version
"""

from __future__ import annotations
from loguru import logger


# ════════════════════════════════════════════════════════════════
# Confidence grading
# ════════════════════════════════════════════════════════════════

def _confidence_grade(conf: float) -> str:
    """Map a 0.0–1.0 confidence to a human-readable grade."""
    if conf >= 0.90:
        return "VERY_HIGH"
    if conf >= 0.75:
        return "HIGH"
    if conf >= 0.55:
        return "MODERATE"
    if conf >= 0.35:
        return "LOW"
    return "UNCERTAIN"


# ════════════════════════════════════════════════════════════════
# Main builder
# ════════════════════════════════════════════════════════════════

def build_structured_output(
    classifier_result: dict | None,
    reasoning_result: dict | None,
    llm_validation_dict: dict | None,
    confidence_fusion: dict | None,
    ensemble: dict | None,
    processing_time_ms: float = 0,
    *,
    gradcam_data: dict | None = None,
    research_papers: list[dict] | None = None,
    ensemble_voting: dict | None = None,
    temporal_data: dict | None = None,
) -> dict:
    """Build the final structured output from all pipeline signals.

    All inputs are plain dicts (already serialized from dataclasses).
    Returns a clean dict ready for JSON response.
    """

    # ── 1. Diagnosis: pick the best source ──
    diagnosis = _build_diagnosis(reasoning_result, llm_validation_dict, confidence_fusion, classifier_result)

    # ── 2. Health ──
    health = _build_health(reasoning_result, llm_validation_dict, ensemble)

    # ── 3. Confidence breakdown ──
    confidence = _build_confidence_breakdown(
        classifier_result, reasoning_result, llm_validation_dict, confidence_fusion
    )

    # ── 4. Reasoning chain ──
    reasoning_chain = []
    if reasoning_result and reasoning_result.get("reasoning_chain"):
        reasoning_chain = reasoning_result["reasoning_chain"]

    # ── 5. Evidence ──
    evidence = _build_evidence(reasoning_result)

    # ── 6. Rejected diagnoses ──
    rejected = _build_rejected(reasoning_result)

    # ── 7. Differential ──
    differential = []
    if reasoning_result and reasoning_result.get("differential_diagnosis"):
        differential = reasoning_result["differential_diagnosis"]

    # ── 8. Treatment ──
    treatment = _build_treatment(reasoning_result, llm_validation_dict)

    # ── 9. Metadata ──
    models_used = ensemble.get("models_used", []) if ensemble else []
    model_agreement = ensemble.get("model_agreement", "none") if ensemble else "none"

    metadata = {
        "models_used": models_used,
        "model_agreement": model_agreement,
        "processing_time_ms": round(processing_time_ms, 0),
        "pipeline_version": "3.0",
        "ensemble_note": ensemble.get("note", "") if ensemble else "",
    }

    result = {
        "diagnosis": diagnosis,
        "health": health,
        "confidence_breakdown": confidence,
        "reasoning_chain": reasoning_chain,
        "evidence": evidence,
        "rejected_diagnoses": rejected,
        "differential_diagnosis": differential,
        "treatment": treatment,
        "metadata": metadata,
    }

    # ── AI Validation (LLaVA) ──
    if llm_validation_dict:
        result["ai_validation"] = {
            "agrees": llm_validation_dict.get("agrees"),
            "agreement_score": llm_validation_dict.get("agreement_score"),
            "llm_diagnosis": llm_validation_dict.get("llm_diagnosis"),
            "scenario": llm_validation_dict.get("scenario"),
            "reasoning_text": llm_validation_dict.get("reasoning_text", ""),
            "health_score": llm_validation_dict.get("health_score"),
            "risk_level": llm_validation_dict.get("risk_level"),
            "model": "LLaVA",
        }

    # ── F1: Grad-CAM heatmap ──
    if gradcam_data:
        result["gradcam"] = gradcam_data

    # ── F2: Research papers ──
    if research_papers:
        result["research_papers"] = research_papers

    # ── F3: Ensemble voting details ──
    if ensemble_voting:
        result["ensemble_voting"] = ensemble_voting

    # ── F4: Temporal tracking ──
    if temporal_data:
        result["temporal"] = temporal_data

    # ── F5: Spectral analysis ──
    if reasoning_result and reasoning_result.get("spectral_analysis"):
        result["spectral_analysis"] = reasoning_result["spectral_analysis"]

    return result


# ════════════════════════════════════════════════════════════════
# Section builders
# ════════════════════════════════════════════════════════════════

def _build_diagnosis(
    reasoning: dict | None,
    llm_val: dict | None,
    fusion: dict | None,
    classifier: dict | None,
) -> dict:
    """Build the diagnosis section — single best answer."""

    # Primary confidence comes from fusion if available
    fused_conf = fusion.get("fused_confidence", 0) if fusion else None

    # Disease name from reasoning engine (most reliable)
    if reasoning and reasoning.get("disease_key"):
        disease_key = reasoning["disease_key"]
        disease_name = reasoning.get("disease_name", disease_key)
        base_conf = reasoning.get("confidence", 0)
    elif classifier:
        disease_key = classifier.get("top5", [{}])[0].get("class_key", "unknown")
        disease_name = classifier.get("top_prediction", "Unknown")
        base_conf = classifier.get("top_confidence", 0)
    else:
        disease_key = "unknown"
        disease_name = "Unknown"
        base_conf = 0

    # Use fused confidence if available, otherwise fall back to reasoning confidence
    final_conf = fused_conf if fused_conf is not None else base_conf

    # LLM validation enrichment
    llm_agrees = None
    llm_alt_diagnosis = None
    if llm_val:
        llm_agrees = llm_val.get("agrees")
        if not llm_agrees and llm_val.get("llm_diagnosis"):
            llm_alt_diagnosis = llm_val["llm_diagnosis"]

    return {
        "disease_key": disease_key,
        "disease_name": disease_name,
        "confidence": round(final_conf, 3),
        "confidence_grade": _confidence_grade(final_conf),
        "is_healthy": disease_key.startswith("healthy"),
        "llm_agrees": llm_agrees,
        "llm_alt_diagnosis": llm_alt_diagnosis,
    }


def _build_health(
    reasoning: dict | None,
    llm_val: dict | None,
    ensemble: dict | None,
) -> dict:
    """Build the health section."""
    # Priority: ensemble > reasoning > llm > default
    if ensemble and ensemble.get("ensemble_health_score") is not None:
        score = ensemble["ensemble_health_score"]
        risk = ensemble.get("ensemble_risk_level", "medium")
    elif reasoning:
        score = reasoning.get("health_score", 50)
        risk = reasoning.get("risk_level", "medium")
    elif llm_val:
        score = llm_val.get("health_score", 50)
        risk = llm_val.get("risk_level", "medium")
    else:
        score = 50
        risk = "medium"

    urgency = "within_30_days"
    yield_loss = None
    affected_parts = []
    if reasoning:
        urgency = reasoning.get("urgency", "within_30_days")
        yield_loss = reasoning.get("yield_loss")
        affected_parts = reasoning.get("affected_parts", [])

    return {
        "score": score,
        "risk_level": risk,
        "urgency": urgency,
        "yield_loss": yield_loss,
        "affected_parts": affected_parts,
    }


def _build_confidence_breakdown(
    classifier: dict | None,
    reasoning: dict | None,
    llm_val: dict | None,
    fusion: dict | None,
) -> dict:
    """Build per-source confidence breakdown."""
    sources = []

    # Classifier source
    if classifier:
        cls_disease = classifier.get("top_prediction", "Unknown")
        cls_conf = classifier.get("top_confidence", 0)
        sources.append({
            "source": "classifier",
            "label": "YOLO-CLS",
            "disease": cls_disease,
            "score": round(cls_conf, 3),
            "weight": fusion.get("weights", {}).get("classifier", 0.20) if fusion else 0.20,
        })

    # Rule engine source
    if reasoning:
        re_disease = reasoning.get("disease_name", "Unknown")
        re_conf = reasoning.get("confidence", 0)
        sources.append({
            "source": "rule_engine",
            "label": "Rule Engine",
            "disease": re_disease,
            "score": round(re_conf, 3),
            "weight": fusion.get("weights", {}).get("rule", 0.50) if fusion else 0.50,
        })

    # LLM validator source
    if llm_val:
        llm_diag = llm_val.get("llm_diagnosis", "Unknown")
        llm_score = llm_val.get("agreement_score", 0)
        sources.append({
            "source": "llm_validator",
            "label": "LLM Validator",
            "disease": llm_diag,
            "score": round(llm_score, 3),
            "agrees": llm_val.get("agrees", False),
            "scenario": llm_val.get("scenario", ""),
            "weight": fusion.get("weights", {}).get("llm", 0.30) if fusion else 0.30,
        })

    # Fused result
    fused = fusion.get("fused_confidence", 0) if fusion else None

    return {
        "sources": sources,
        "fused_confidence": round(fused, 3) if fused is not None else None,
        "fusion_note": fusion.get("note", "") if fusion else "No fusion available",
    }


def _build_evidence(reasoning: dict | None) -> dict:
    """Build evidence section from reasoning result."""
    supporting = []
    contradicting = []

    if reasoning:
        # Symptoms matched = supporting evidence
        for s in reasoning.get("symptoms_matched", []):
            supporting.append(s)

        # Symptoms detected but not matched = observational
        for s in reasoning.get("symptoms_detected", []):
            if s not in supporting:
                supporting.append(s)

        # Conflict info → contradicting
        if reasoning.get("conflict"):
            c = reasoning["conflict"]
            if c.get("winner") == "rules":
                contradicting.append(
                    f"Classifier predicted {c.get('yolo_prediction', 'other disease')} "
                    f"({c.get('yolo_confidence', 0):.0%}) but visual evidence contradicts this"
                )

    return {
        "supporting": supporting,
        "contradicting": contradicting,
    }


def _build_rejected(reasoning: dict | None) -> list:
    """Build rejected diagnoses list."""
    if not reasoning or not reasoning.get("rejections"):
        return []

    rejected = []
    for r in reasoning["rejections"]:
        entry = {
            "disease": r.get("disease", "Unknown"),
            "reasons": r.get("reasons", []),
        }
        if r.get("missing_features"):
            entry["missing_features"] = r["missing_features"]
        if r.get("contradicting_features"):
            entry["contradicting_features"] = r["contradicting_features"]
        rejected.append(entry)
    return rejected


def _build_treatment(
    reasoning: dict | None,
    llm_val: dict | None,
) -> dict:
    """Build treatment recommendations."""
    recommendations = []
    urgency = "within_30_days"

    if reasoning:
        recommendations = reasoning.get("treatment", [])
        urgency = reasoning.get("urgency", "within_30_days")

    # LLM may add complementary recommendations
    llm_recs = []
    if llm_val and llm_val.get("recommendations"):
        llm_recs = llm_val["recommendations"]
        # Only add LLM recs that aren't already covered
        existing_lower = {r.lower() for r in recommendations}
        for rec in llm_recs:
            if rec.lower() not in existing_lower:
                recommendations.append(rec)

    return {
        "recommendations": recommendations,
        "urgency": urgency,
        "urgency_display": urgency.replace("_", " ").title(),
    }
