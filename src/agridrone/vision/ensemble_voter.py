"""
ensemble_voter.py — Multi-model ensemble voting with Bayesian combination (F3).

Replaces simple weighted averaging with:
  1. Confidence-weighted majority vote
  2. Per-model reliability tracking (from feedback data)
  3. Bayesian posterior combination
  4. Agreement/disagreement analysis
  5. Full voting audit trail

Models participating:
  - YOLOv8n-cls (trained classifier): fast, trained on 21 classes
  - Symptom Reasoning Engine: rule-based, uses visual features + KB
  - LLM Validator (LLaVA): vision LLM, validates/arbitrates
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from loguru import logger


# ════════════════════════════════════════════════════════════════
# Data structures
# ════════════════════════════════════════════════════════════════

@dataclass
class ModelVote:
    """A single model's opinion on the diagnosis."""
    model_name: str
    disease_key: str           # e.g. "wheat_yellow_rust" or "healthy_wheat"
    confidence: float          # 0-1
    health_score: int          # 0-100
    is_healthy: bool
    reliability: float = 1.0   # Per-model reliability weight (updated from feedback)
    details: dict = field(default_factory=dict)


@dataclass
class EnsembleResult:
    """Full result of ensemble voting."""
    final_disease: str
    final_confidence: float
    final_health_score: int
    final_risk_level: str
    agreement_level: str       # "unanimous", "majority", "split", "single"
    voting_method: str         # "bayesian", "majority", "weighted_avg", "single_model"
    votes: list[ModelVote] = field(default_factory=list)
    vote_tally: dict = field(default_factory=dict)   # disease_key → {count, total_weight}
    disagreement_details: str = ""
    models_used: list[str] = field(default_factory=list)
    safety_overrides: list[str] = field(default_factory=list)


# ════════════════════════════════════════════════════════════════
# Severe diseases (cap health score for safety)
# ════════════════════════════════════════════════════════════════

_SEVERE_DISEASES = {
    "wheat_fusarium_head_blight", "wheat_blast", "wheat_black_rust",
    "rice_blast", "rice_bacterial_blight",
}


# ════════════════════════════════════════════════════════════════
# Default model reliability weights (overridden by feedback data)
# ════════════════════════════════════════════════════════════════

_DEFAULT_RELIABILITY = {
    "YOLOv8n-cls": 0.96,        # Validated: 96.2% on test set
    "Reasoning Engine": 0.60,    # Validated: 60.2% on test set (degrades YOLO)
    "LLM Validator": 0.55,
}


def _load_feedback_reliability() -> dict[str, float]:
    """Try to load per-model reliability from feedback data."""
    try:
        feedback_db = Path(__file__).resolve().parent.parent / "feedback" / "feedback.db"
        if not feedback_db.is_file():
            return _DEFAULT_RELIABILITY.copy()

        from ..feedback.correction_aggregator import compute_model_accuracy
        from ..feedback.feedback_store import FeedbackStore
        store = FeedbackStore(str(feedback_db))
        records = store.get_all_feedback()
        if len(records) < 10:
            return _DEFAULT_RELIABILITY.copy()

        model_acc = compute_model_accuracy(records)
        rel = _DEFAULT_RELIABILITY.copy()
        for model_name, stats in model_acc.items():
            acc = stats.get("accuracy", 0.5)
            # Map model accuracy names to our standard names
            if "classifier" in model_name.lower() or "yolo" in model_name.lower():
                rel["YOLOv8n-cls"] = max(0.3, min(0.95, acc))
            elif "reasoning" in model_name.lower() or "rule" in model_name.lower():
                rel["Reasoning Engine"] = max(0.3, min(0.95, acc))
            elif "llm" in model_name.lower() or "llava" in model_name.lower():
                rel["LLM Validator"] = max(0.3, min(0.95, acc))
        return rel
    except Exception as exc:
        logger.debug(f"Feedback reliability load failed (using defaults): {exc}")
        return _DEFAULT_RELIABILITY.copy()


# ════════════════════════════════════════════════════════════════
# Ensemble voting
# ════════════════════════════════════════════════════════════════

def build_votes(
    classifier_result: dict | None,
    reasoning_result: dict | None,
    llm_validation: object | None,   # LLMValidation dataclass
    crop_type: str = "wheat",
) -> list[ModelVote]:
    """Collect votes from each available model."""
    votes: list[ModelVote] = []

    # ── Classifier vote ──
    if classifier_result:
        top = classifier_result.get("top_prediction", "unknown")
        conf = classifier_result.get("top_confidence", 0.5)
        is_h = classifier_result.get("is_healthy", False)
        hs = classifier_result.get("health_score", 50)

        votes.append(ModelVote(
            model_name="YOLOv8n-cls",
            disease_key=top,
            confidence=conf,
            health_score=hs,
            is_healthy=is_h,
            details={
                "top5": classifier_result.get("top5_predictions", []),
                "disease_probability": classifier_result.get("disease_probability", 0),
            },
        ))

    # ── Reasoning engine vote ──
    if reasoning_result:
        dk = reasoning_result.get("disease_key", "unknown")
        conf = reasoning_result.get("confidence", 0.5)
        is_h = dk.startswith("healthy")
        hs = reasoning_result.get("health_score", 50)

        votes.append(ModelVote(
            model_name="Reasoning Engine",
            disease_key=dk,
            confidence=conf,
            health_score=hs,
            is_healthy=is_h,
            details={
                "evidence": reasoning_result.get("evidence", []),
                "rejected": reasoning_result.get("rejected_diseases", []),
            },
        ))

    # ── LLM Validator vote ──
    if llm_validation:
        # llm_validation is an LLMValidation dataclass
        llm_diag = getattr(llm_validation, "llm_diagnosis", "unknown")
        llm_conf = getattr(llm_validation, "agreement_score", 0.5)
        llm_hs = getattr(llm_validation, "health_score", 50)
        llm_agrees = getattr(llm_validation, "agrees", True)

        votes.append(ModelVote(
            model_name="LLM Validator",
            disease_key=llm_diag if llm_diag and llm_diag != "unknown" else (
                reasoning_result.get("disease_key", "unknown") if reasoning_result and llm_agrees else "unknown"
            ),
            confidence=llm_conf,
            health_score=llm_hs,
            is_healthy=llm_hs >= 70,
            details={
                "agrees": llm_agrees,
                "reasoning": getattr(llm_validation, "explanation", ""),
                "risk_level": getattr(llm_validation, "risk_level", "medium"),
            },
        ))

    return votes


def ensemble_vote(
    classifier_result: dict | None,
    reasoning_result: dict | None,
    llm_validation: object | None,
    crop_type: str = "wheat",
) -> EnsembleResult:
    """Run ensemble voting across all available models.

    Strategy:
      1. Collect votes from each model
      2. Weight by reliability × confidence
      3. Bayesian combination: multiply weighted likelihoods per disease
      4. Safety-first: severe diseases cap health score
      5. Track agreement and provide audit trail
    """
    votes = build_votes(classifier_result, reasoning_result, llm_validation, crop_type)
    reliability = _load_feedback_reliability()

    # Assign reliability weights
    for v in votes:
        v.reliability = reliability.get(v.model_name, 0.5)

    if not votes:
        return EnsembleResult(
            final_disease="unknown",
            final_confidence=0.0,
            final_health_score=50,
            final_risk_level="medium",
            agreement_level="none",
            voting_method="none",
            models_used=[],
        )

    if len(votes) == 1:
        v = votes[0]
        hs = v.health_score
        overrides = []
        if v.disease_key in _SEVERE_DISEASES and hs > 45:
            hs = min(hs, 45)
            overrides.append(f"Severe disease {v.disease_key} — health capped to {hs}")
        return EnsembleResult(
            final_disease=v.disease_key,
            final_confidence=v.confidence,
            final_health_score=hs,
            final_risk_level=_risk_level(hs),
            agreement_level="single",
            voting_method="single_model",
            votes=votes,
            vote_tally={v.disease_key: {"count": 1, "total_weight": v.reliability * v.confidence}},
            models_used=[v.model_name],
            safety_overrides=overrides,
        )

    # ── Tally votes with weighted scoring ──
    tally: dict[str, dict] = {}  # disease_key → {count, total_weight, health_scores, voters}
    for v in votes:
        dk = v.disease_key
        if dk not in tally:
            tally[dk] = {"count": 0, "total_weight": 0.0, "health_scores": [], "voters": []}
        w = v.reliability * v.confidence
        tally[dk]["count"] += 1
        tally[dk]["total_weight"] += w
        tally[dk]["health_scores"].append(v.health_score)
        tally[dk]["voters"].append(v.model_name)

    # ── Bayesian posterior combination ──
    # P(disease | votes) ∝ Π (reliability_i × confidence_i) for each vote for that disease
    # Plus a small prior for diseases with more votes
    best_disease = None
    best_score = -1.0
    for dk, info in tally.items():
        # Bayesian-ish: product of weights × count bonus
        posterior = info["total_weight"] * (1 + 0.2 * (info["count"] - 1))
        info["posterior"] = round(posterior, 4)
        if posterior > best_score:
            best_score = posterior
            best_disease = dk

    # ── Determine agreement level ──
    unique_diseases = set(v.disease_key for v in votes if v.disease_key != "unknown")
    health_agree = all(v.is_healthy == votes[0].is_healthy for v in votes)

    if len(unique_diseases) <= 1 and health_agree:
        agreement = "unanimous"
    elif len(unique_diseases) <= 1 or health_agree:
        agreement = "majority"
    else:
        agreement = "split"

    # ── Compute final health score ──
    if agreement == "unanimous":
        # Weighted average health
        total_w = sum(v.reliability * v.confidence for v in votes)
        if total_w > 0:
            final_health = round(sum(v.health_score * v.reliability * v.confidence for v in votes) / total_w)
        else:
            final_health = round(sum(v.health_score for v in votes) / len(votes))
        voting_method = "bayesian"
    elif agreement == "majority":
        # Trust the majority, weight by reliability
        total_w = sum(v.reliability * v.confidence for v in votes)
        if total_w > 0:
            final_health = round(sum(v.health_score * v.reliability * v.confidence for v in votes) / total_w)
        else:
            final_health = round(sum(v.health_score for v in votes) / len(votes))
        voting_method = "majority"
    else:
        # Split: SAFETY-FIRST — use the most pessimistic health score
        final_health = min(v.health_score for v in votes)
        voting_method = "safety_first_min"

    # ── Final confidence from Bayesian posterior ──
    total_posterior = sum(info["posterior"] for info in tally.values())
    final_confidence = best_score / total_posterior if total_posterior > 0 else 0.5

    # ── Safety overrides ──
    overrides: list[str] = []
    if best_disease in _SEVERE_DISEASES:
        if final_health > 55:
            final_health = min(final_health, 55)
            overrides.append(f"Severe disease {best_disease} — health capped to 55")

    # Check if any model flagged a severe disease even if not the winner
    for v in votes:
        if v.disease_key in _SEVERE_DISEASES and v.disease_key != best_disease:
            if v.confidence > 0.4 and v.reliability > 0.5:
                # A credible model sees a severe disease — don't let health be too high
                if final_health > 60:
                    final_health = min(final_health, 60)
                    overrides.append(
                        f"{v.model_name} flagged severe {v.disease_key} (conf={v.confidence:.2f}) — health capped"
                    )

    # ── Disagreement details ──
    dis_details = ""
    if agreement == "split":
        parts = []
        for v in votes:
            parts.append(f"{v.model_name}: {v.disease_key} (conf={v.confidence:.2f}, health={v.health_score})")
        dis_details = " | ".join(parts)

    # Serialize tally for output (remove health_scores list)
    tally_out = {}
    for dk, info in tally.items():
        tally_out[dk] = {
            "count": info["count"],
            "total_weight": round(info["total_weight"], 3),
            "posterior": info["posterior"],
            "voters": info["voters"],
        }

    return EnsembleResult(
        final_disease=best_disease or "unknown",
        final_confidence=round(min(final_confidence, 1.0), 3),
        final_health_score=final_health,
        final_risk_level=_risk_level(final_health),
        agreement_level=agreement,
        voting_method=voting_method,
        votes=votes,
        vote_tally=tally_out,
        disagreement_details=dis_details,
        models_used=[v.model_name for v in votes],
        safety_overrides=overrides,
    )


def ensemble_to_dict(result: EnsembleResult) -> dict:
    """Convert EnsembleResult to JSON-serializable dict for API response."""
    return {
        "final_disease": result.final_disease,
        "final_confidence": result.final_confidence,
        "final_health_score": result.final_health_score,
        "final_risk_level": result.final_risk_level,
        "agreement_level": result.agreement_level,
        "voting_method": result.voting_method,
        "models_used": result.models_used,
        "num_models": len(result.votes),
        "vote_tally": result.vote_tally,
        "disagreement_details": result.disagreement_details,
        "safety_overrides": result.safety_overrides,
        "individual_votes": [
            {
                "model": v.model_name,
                "disease": v.disease_key,
                "confidence": round(v.confidence, 3),
                "health_score": v.health_score,
                "is_healthy": v.is_healthy,
                "reliability": round(v.reliability, 3),
            }
            for v in result.votes
        ],
    }


def _risk_level(health_score: int) -> str:
    if health_score >= 70:
        return "low"
    if health_score >= 40:
        return "medium"
    if health_score >= 20:
        return "high"
    return "critical"
