"""
rule_engine.py — Feature-to-disease matching with scoring, conflict resolution,
and rejection explanation.

This is the brain of the diagnostic pipeline:
  - Takes structured ImageFeatures + YOLO classifier predictions
  - Applies rules from the knowledge base to score each candidate disease
  - Resolves conflicts when YOLO disagrees with visual evidence
  - Generates human-readable rejection explanations

Architecture:
  feature_extractor.py  →  rule_engine.py  →  disease_reasoning.py (orchestrator)
       (eyes)                 (brain)              (mouth — explains)

Builds on Task Group A: diseases.json + kb_loader.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from loguru import logger

from agridrone.knowledge import kb_loader
from agridrone.knowledge.kb_loader import DiseaseProfile
from agridrone.vision.feature_extractor import ImageFeatures


# ════════════════════════════════════════════════════════════════
# Data structures
# ════════════════════════════════════════════════════════════════

@dataclass
class RuleMatch:
    """A single rule evaluation result for one disease candidate."""
    disease_key: str
    rule_name: str
    score_delta: float              # Positive = supports, negative = contradicts
    explanation: str                 # Human-readable evidence sentence


@dataclass
class Rejection:
    """Explanation for why a disease was rejected or penalized."""
    disease_key: str
    disease_name: str
    reasons: list[str]              # Human-readable rejection reasons
    missing_features: list[str]     # What visual evidence was absent
    contradicting_features: list[str]  # What visual evidence contradicts this disease


@dataclass
class ConflictReport:
    """When YOLO and rules disagree, this explains the conflict."""
    yolo_prediction: str
    yolo_confidence: float
    rule_prediction: str
    rule_confidence: float
    winner: str                     # "yolo" | "rules" | "agree"
    reason: str                     # Why one was chosen over the other
    yolo_rejections: list[str]      # Why YOLO's pick was demoted (if rules won)


@dataclass
class CandidateScore:
    """Fully scored disease candidate with all evidence."""
    disease_key: str
    disease_name: str
    classifier_score: float         # Raw YOLO/classifier confidence (0.0–1.0)
    rule_score: float               # Rule-derived visual evidence score (-1.0 to +2.0)
    seasonal_multiplier: float      # Season-based adjustment (0.5–1.2)
    final_score: float              # Combined weighted score (0.0–1.0)
    matches: list[RuleMatch]        # All rules that fired for this disease
    rejection: Rejection | None     # If rejected/penalized, why


@dataclass
class RuleEngineResult:
    """Complete output of the rule engine."""
    candidates: list[CandidateScore]  # All scored candidates, sorted by final_score desc
    top_disease: str                   # Key of highest-scoring disease
    top_confidence: float              # Final score of top disease
    conflict: ConflictReport | None    # Non-None when YOLO and rules disagree
    rejections: list[Rejection]        # All diseases that were rejected with explanations


# ════════════════════════════════════════════════════════════════
# Rule evaluation functions
# ════════════════════════════════════════════════════════════════

def _eval_color_rules(
    disease_key: str,
    profile: DiseaseProfile,
    features: ImageFeatures,
) -> list[RuleMatch]:
    """Check if the image's color signatures match this disease's expected colors."""
    matches = []

    for sig in profile.color_signatures:
        sig_key = f"{disease_key}:{sig['name']}"
        if sig_key in features.color_confidences:
            strength = features.color_confidences[sig_key]
            matches.append(RuleMatch(
                disease_key=disease_key,
                rule_name=f"color_{sig['name']}",
                score_delta=strength * 0.4,
                explanation=(
                    f"{sig['name'].replace('_', ' ').title()} color pattern detected "
                    f"({strength:.0%} confidence)"
                ),
            ))

    return matches


def _eval_texture_rules(
    disease_key: str,
    profile: DiseaseProfile,
    features: ImageFeatures,
) -> list[RuleMatch]:
    """Check texture evidence: bleaching, spots, edge density."""
    matches = []
    symptoms_lower = [s.lower() for s in profile.symptoms]
    keywords_lower = [k.lower() for k in profile.texture_keywords]

    # Bleaching rule
    if features.has_bleaching and any("bleach" in s or "whiten" in s for s in symptoms_lower):
        conf = min(1.0, features.bleaching_ratio * 10)
        matches.append(RuleMatch(
            disease_key=disease_key,
            rule_name="texture_bleaching",
            score_delta=conf * 0.3,
            explanation=f"Bleaching detected ({features.bleaching_ratio:.1%} of image) — matches expected symptom",
        ))

    # Spot/pustule rule
    if features.has_significant_spots and any(
        "pustule" in s or "spot" in s for s in symptoms_lower
    ):
        conf = min(1.0, features.spot_count / 100)
        matches.append(RuleMatch(
            disease_key=disease_key,
            rule_name="texture_spots",
            score_delta=conf * 0.2,
            explanation=f"{features.spot_count} spot/pustule structures found — matches expected symptom",
        ))

    return matches


def _eval_spatial_rules(
    disease_key: str,
    profile: DiseaseProfile,
    features: ImageFeatures,
) -> list[RuleMatch]:
    """Stripe-vs-spot spatial pattern rules — critical for rust vs tan spot."""
    matches = []
    display = profile.display_name.lower()
    is_stripe_disease = "stripe" in display or "rust" in display
    is_spot_disease = "spot" in display
    is_head_disease = any(kw in display for kw in ("blight", "blast", "scab", "smut", "bunting", "bunt"))

    # ── Stripe pattern boosts stripe/rust diseases ──
    if features.has_stripe_pattern and is_stripe_disease:
        combined = (
            features.stripe_confidence
            + (features.directionality - 1.0) * 0.2 if features.directionality > 1.5 else features.stripe_confidence
        )
        delta = min(0.5, combined * 0.5)
        matches.append(RuleMatch(
            disease_key=disease_key,
            rule_name="spatial_stripe_match",
            score_delta=delta,
            explanation=(
                f"Linear stripe pattern detected (conf={features.stripe_confidence:.0%}, "
                f"{features.hough_line_count} lines) — consistent with {profile.display_name}"
            ),
        ))

    # ── Stripe pattern penalizes spot diseases ──
    if features.has_stripe_pattern and is_spot_disease:
        penalty = features.stripe_confidence * 0.3
        matches.append(RuleMatch(
            disease_key=disease_key,
            rule_name="spatial_stripe_contradicts_spot",
            score_delta=-penalty,
            explanation=(
                f"Linear stripe pattern found but {profile.display_name} expects discrete spots — "
                f"stripes contradict spot disease"
            ),
        ))

    # ── Stripe pattern penalizes head/blight diseases (FHB, Blast, Smut) ──
    if features.has_stripe_pattern and is_head_disease and not is_stripe_disease:
        penalty = features.stripe_confidence * 0.35
        matches.append(RuleMatch(
            disease_key=disease_key,
            rule_name="spatial_stripe_contradicts_head_disease",
            score_delta=-penalty,
            explanation=(
                f"Linear stripe pattern on leaves contradicts {profile.display_name} "
                f"(head/spike disease, not leaf stripe)"
            ),
        ))

    # ── Spot pattern boosts spot diseases ──
    if features.has_spot_pattern and is_spot_disease:
        delta = features.spot_confidence * 0.3
        matches.append(RuleMatch(
            disease_key=disease_key,
            rule_name="spatial_spot_match",
            score_delta=delta,
            explanation=f"Discrete circular spots detected — consistent with {profile.display_name}",
        ))

    # ── Spot pattern penalizes stripe diseases ──
    if features.has_spot_pattern and is_stripe_disease:
        penalty = features.spot_confidence * 0.2
        matches.append(RuleMatch(
            disease_key=disease_key,
            rule_name="spatial_spot_contradicts_stripe",
            score_delta=-penalty,
            explanation=(
                f"Discrete spots found but {profile.display_name} expects linear stripes"
            ),
        ))

    return matches


def _eval_saturation_rules(
    disease_key: str,
    profile: DiseaseProfile,
    features: ImageFeatures,
) -> list[RuleMatch]:
    """Vivid vs dull saturation rules — separates rust from tan spot/blight."""
    matches = []
    display = profile.display_name.lower()
    is_rust = "rust" in display
    is_spot = "spot" in display
    is_blight = "blight" in display and "sheath" not in display

    if features.has_vivid_yellow and is_rust:
        viv = min(1.0, features.vivid_yellow_orange_ratio * 15)
        matches.append(RuleMatch(
            disease_key=disease_key,
            rule_name="saturation_vivid_rust",
            score_delta=viv * 0.4,
            explanation=(
                f"Vivid yellow-orange color ({features.vivid_yellow_orange_ratio:.1%} of image) — "
                f"typical of rust pustules, not dull lesions"
            ),
        ))

    if features.has_vivid_yellow and (is_spot or is_blight):
        viv = min(1.0, features.vivid_yellow_orange_ratio * 15)
        penalty = viv * 0.25
        matches.append(RuleMatch(
            disease_key=disease_key,
            rule_name="saturation_vivid_contradicts_dull",
            score_delta=-penalty,
            explanation=(
                f"Vivid yellow-orange detected but {profile.display_name} typically shows "
                f"dull/muted coloration"
            ),
        ))

    return matches


def _eval_greenness_rule(
    disease_key: str,
    profile: DiseaseProfile,
    features: ImageFeatures,
) -> list[RuleMatch]:
    """High greenness supports healthy, penalizes severe diseases.

    Only fires for healthy profiles if green covers >70% of pixels,
    ensuring background grass/stem green doesn't trigger false healthy.
    """
    matches = []

    if profile.type == "healthy" and features.green_ratio > 0.70:
        matches.append(RuleMatch(
            disease_key=disease_key,
            rule_name="greenness_healthy",
            score_delta=min(0.2, (features.green_ratio - 0.70) * 0.5),
            explanation=f"High green coverage ({features.green_ratio:.0%}) supports healthy diagnosis",
        ))

    if profile.severity >= 0.7 and features.green_ratio > 0.6:
        penalty = (features.green_ratio - 0.5) * 0.3
        matches.append(RuleMatch(
            disease_key=disease_key,
            rule_name="greenness_contradicts_severe",
            score_delta=-penalty,
            explanation=(
                f"Mostly green tissue ({features.green_ratio:.0%}) is unusual for "
                f"severe disease {profile.display_name}"
            ),
        ))

    return matches


def _eval_spectral_rules(
    disease_key: str,
    profile: DiseaseProfile,
    features: ImageFeatures,
    spectral: "SpectralResult | None" = None,
) -> list[RuleMatch]:
    """Score diseases using pseudo-hyperspectral vegetation indices.

    Uses VARI, RGRI and other indices from SpectralResult to detect
    early stress patterns that precede visible symptoms.
    """
    matches = []
    if spectral is None:
        return matches

    vari = spectral.indices.get("VARI")
    rgri = spectral.indices.get("RGRI")
    gli = spectral.indices.get("GLI")
    ngrdi = spectral.indices.get("NGRDI")

    # ── Chlorosis evidence (low VARI + negative GLI) → supports rust, nutrient deficiency ──
    is_chlorosis_disease = any(
        kw in disease_key for kw in ("rust", "deficiency", "yellowing", "mosaic")
    )
    if vari and vari.mean < 0.1:
        if is_chlorosis_disease:
            boost = 0.15 if vari.mean < 0.0 else 0.08
            matches.append(RuleMatch(
                disease_key=disease_key,
                rule_name="spectral_chlorosis_support",
                score_delta=boost,
                explanation=(
                    f"Low VARI ({vari.mean:.3f}) indicates reduced photosynthetic "
                    f"capacity — consistent with {profile.display_name}"
                ),
            ))
        elif profile.type == "healthy":
            matches.append(RuleMatch(
                disease_key=disease_key,
                rule_name="spectral_chlorosis_contradicts_healthy",
                score_delta=-0.10,
                explanation=(
                    f"Low VARI ({vari.mean:.3f}) suggests early stress — "
                    f"contradicts healthy diagnosis"
                ),
            ))

    # ── Necrosis evidence (high RGRI) → supports blast, blight, spot diseases ──
    is_necrosis_disease = any(
        kw in disease_key for kw in ("blast", "blight", "spot", "scald", "burn")
    )
    if rgri and rgri.mean > 1.3:
        if is_necrosis_disease:
            boost = 0.18 if rgri.mean > 1.8 else 0.10
            matches.append(RuleMatch(
                disease_key=disease_key,
                rule_name="spectral_necrosis_support",
                score_delta=boost,
                explanation=(
                    f"Elevated RGRI ({rgri.mean:.3f}) indicates tissue reddening — "
                    f"consistent with {profile.display_name}"
                ),
            ))
        elif profile.type == "healthy":
            matches.append(RuleMatch(
                disease_key=disease_key,
                rule_name="spectral_necrosis_contradicts_healthy",
                score_delta=-0.12,
                explanation=(
                    f"Elevated RGRI ({rgri.mean:.3f}) suggests necrosis — "
                    f"contradicts healthy diagnosis"
                ),
            ))

    # ── Strong green signal supports healthy ──
    if profile.type == "healthy" and vari and vari.mean > 0.2 and gli and gli.mean > 0.1:
        matches.append(RuleMatch(
            disease_key=disease_key,
            rule_name="spectral_healthy_vegetation",
            score_delta=0.12,
            explanation=(
                f"Strong vegetation indices (VARI={vari.mean:.3f}, GLI={gli.mean:.3f}) "
                f"support healthy assessment"
            ),
        ))

    # ── Severe stress penalizes low-severity diseases ──
    if spectral.stress_level == "severe" and profile.severity < 0.3 and profile.type != "healthy":
        matches.append(RuleMatch(
            disease_key=disease_key,
            rule_name="spectral_severity_mismatch",
            score_delta=-0.08,
            explanation=(
                f"Severe spectral stress ({spectral.stress_type}) is unlikely for "
                f"low-severity {profile.display_name}"
            ),
        ))

    return matches


# ════════════════════════════════════════════════════════════════
# B3: Conflict Resolver
# ════════════════════════════════════════════════════════════════

def _resolve_conflict(
    yolo_top_key: str,
    yolo_top_conf: float,
    rule_top_key: str,
    rule_top_score: float,
    candidates: dict[str, CandidateScore],
) -> ConflictReport:
    """Determine winner when YOLO classifier and rule engine disagree.

    Decision logic:
    1. If rule engine has strong visual evidence (>0.3 rule_score) AND
       YOLO confidence is moderate (<0.7), rules win.
    2. If YOLO has very high confidence (>0.85) AND rules have weak evidence (<0.15),
       YOLO wins — the model may know something we can't see.
    3. Otherwise: combine scores — higher combined total wins.
    """
    if yolo_top_key == rule_top_key:
        return ConflictReport(
            yolo_prediction=yolo_top_key,
            yolo_confidence=yolo_top_conf,
            rule_prediction=rule_top_key,
            rule_confidence=rule_top_score,
            winner="agree",
            reason="YOLO classifier and visual rules agree on diagnosis",
            yolo_rejections=[],
        )

    yolo_cand = candidates.get(yolo_top_key)
    rule_cand = candidates.get(rule_top_key)

    yolo_rule_score = yolo_cand.rule_score if yolo_cand else 0.0
    rule_cls_score = rule_cand.classifier_score if rule_cand else 0.0

    yolo_rejections = []
    if yolo_cand and yolo_cand.rejection:
        yolo_rejections = yolo_cand.rejection.reasons

    # Decision thresholds
    # YOLO auto-win at ≥0.95 confidence: ablation shows YOLO at 96.15% accuracy,
    # any prediction above 95% confidence should be trusted unconditionally.
    rules_have_strong_evidence = (rule_cand and rule_cand.rule_score > 0.3) if rule_cand else False
    yolo_very_confident = yolo_top_conf >= 0.95
    rules_have_weak_evidence = (rule_cand and rule_cand.rule_score < 0.15) if rule_cand else True

    if yolo_very_confident:
        # YOLO ≥95% confidence: automatic win, no rule engine involvement
        winner = "yolo"
        reason = (
            f"YOLO classifier high-confidence ({yolo_top_conf:.0%}) on {yolo_top_key} — "
            f"auto-accepted (threshold ≥95%)"
        )
    elif rules_have_strong_evidence and yolo_top_conf < 0.70:
        winner = "rules"
        reason = (
            f"Visual evidence strongly supports {rule_top_key} "
            f"(rule_score={rule_cand.rule_score:.2f}) while YOLO's {yolo_top_key} "
            f"({yolo_top_conf:.0%}) lacks visual confirmation"
        )
        if yolo_rejections:
            reason += f". YOLO's pick rejected because: {'; '.join(yolo_rejections[:2])}"
    elif yolo_top_conf > 0.85 and rules_have_weak_evidence:
        winner = "yolo"
        reason = (
            f"YOLO classifier very confident ({yolo_top_conf:.0%}) on {yolo_top_key} "
            f"and visual rules have weak evidence ({yolo_rule_score:.2f})"
        )
    else:
        # Combined score comparison — YOLO-dominant (70/30)
        yolo_combined = yolo_top_conf * 0.70 + yolo_rule_score * 0.30
        rule_combined = rule_cls_score * 0.70 + (rule_cand.rule_score if rule_cand else 0) * 0.30
        if rule_combined > yolo_combined:
            winner = "rules"
            reason = (
                f"Combined evidence favors {rule_top_key} "
                f"(combined={rule_combined:.2f}) over YOLO's {yolo_top_key} "
                f"(combined={yolo_combined:.2f})"
            )
        else:
            winner = "yolo"
            reason = (
                f"Combined evidence favors YOLO's {yolo_top_key} "
                f"(combined={yolo_combined:.2f}) over {rule_top_key} "
                f"(combined={rule_combined:.2f})"
            )

    return ConflictReport(
        yolo_prediction=yolo_top_key,
        yolo_confidence=yolo_top_conf,
        rule_prediction=rule_top_key,
        rule_confidence=rule_top_score,
        winner=winner,
        reason=reason,
        yolo_rejections=yolo_rejections,
    )


# ════════════════════════════════════════════════════════════════
# B4: Rejection Explainer
# ════════════════════════════════════════════════════════════════

def _build_rejection(
    disease_key: str,
    profile: DiseaseProfile,
    features: ImageFeatures,
    matches: list[RuleMatch],
) -> Rejection | None:
    """Build a human-readable rejection explanation for a disease.

    Returns None if the disease has no contradictions worth explaining.
    Only generates rejections for diseases that were plausible candidates
    but got penalized by visual evidence.
    """
    contradictions = [m for m in matches if m.score_delta < -0.05]
    if not contradictions:
        return None

    display = profile.display_name
    reasons = []
    missing = []
    contradicting = []

    # Check what color signatures were expected but absent
    for sig in profile.color_signatures:
        sig_key = f"{disease_key}:{sig['name']}"
        if sig_key not in features.color_confidences:
            missing.append(sig["name"].replace("_", " "))

    # Spatial pattern contradictions
    is_stripe_disease = "stripe" in display.lower() or "rust" in display.lower()
    is_spot_disease = "spot" in display.lower()

    if is_stripe_disease and not features.has_stripe_pattern:
        missing.append("linear stripe pattern")
    if is_spot_disease and not features.has_spot_pattern:
        missing.append("discrete circular spots")

    if is_spot_disease and features.has_stripe_pattern:
        contradicting.append("linear stripes found (expects spots)")
    if is_stripe_disease and features.has_spot_pattern:
        contradicting.append("discrete spots found (expects stripes)")

    # Saturation contradictions
    if is_spot_disease and features.has_vivid_yellow:
        contradicting.append("vivid yellow-orange detected (expects dull tan)")

    # Build human-readable reasons
    for c in contradictions:
        reasons.append(c.explanation)

    if missing:
        reasons.append(f"Missing expected features: {', '.join(missing[:3])}")

    if not reasons:
        return None

    return Rejection(
        disease_key=disease_key,
        disease_name=display,
        reasons=reasons,
        missing_features=missing,
        contradicting_features=contradicting,
    )


# ════════════════════════════════════════════════════════════════
# Main entry point
# ════════════════════════════════════════════════════════════════

def evaluate(
    features: ImageFeatures,
    classifier_result: dict | None,
    crop_type: str = "wheat",
    spectral: "SpectralResult | None" = None,
) -> RuleEngineResult:
    """Run all rules against extracted features and classifier predictions.

    This is the main function called by the reasoning orchestrator.

    Args:
        features: Extracted image features from feature_extractor.
        classifier_result: YOLO classifier output with "top5" predictions.
        crop_type: "wheat" or "rice".

    Returns:
        RuleEngineResult with scored candidates, conflict report, rejections.
    """
    # ── Load KB ──
    kb = kb_loader.get_all_profiles()
    if not kb:
        kb_loader.load()
        kb = kb_loader.get_all_profiles()

    # ── Build candidate set from classifier + all crop diseases ──
    cls_predictions = []
    if classifier_result and "top5" in classifier_result:
        cls_predictions = classifier_result["top5"]

    # Map classifier predictions to known KB keys
    classifier_scores: dict[str, float] = {}
    # Build the active model's vocabulary (only classes the classifier knows about)
    active_vocab: set[str] = set()
    for pred in cls_predictions:
        key = pred.get("class_key", "")
        conf = pred.get("confidence", 0)
        active_vocab.add(key)
        if key in kb:
            classifier_scores[key] = conf
        elif "healthy" not in key.lower():
            # Fuzzy match
            for dk in kb:
                if key.replace("wheat_", "").replace("rice_", "") in dk:
                    classifier_scores[dk] = conf
                    active_vocab.add(dk)
                    break

    # Add crop diseases as candidates ONLY if they are in the active model's vocabulary.
    # This prevents the rule engine from voting for classes that the classifier doesn't know.
    # If no vocabulary info (no classifier), fall back to all crop diseases.
    for dk, profile in kb.items():
        if profile.crop == crop_type or profile.crop == "both":
            if not active_vocab or dk in active_vocab:
                classifier_scores.setdefault(dk, 0.0)
            # else: skip — this disease is not in the model's label set

    # ── Evaluate rules for each candidate ──
    scored_candidates: dict[str, CandidateScore] = {}
    all_rejections: list[Rejection] = []

    for disease_key, cls_score in classifier_scores.items():
        profile = kb.get(disease_key)
        if not profile:
            continue

        # Run all rule groups
        all_matches: list[RuleMatch] = []
        all_matches.extend(_eval_color_rules(disease_key, profile, features))
        all_matches.extend(_eval_texture_rules(disease_key, profile, features))
        all_matches.extend(_eval_spatial_rules(disease_key, profile, features))
        all_matches.extend(_eval_saturation_rules(disease_key, profile, features))
        all_matches.extend(_eval_greenness_rule(disease_key, profile, features))
        all_matches.extend(_eval_spectral_rules(disease_key, profile, features, spectral))

        # Sum rule scores
        rule_score = sum(m.score_delta for m in all_matches)

        # Seasonal adjustment
        seasonal_mult = kb_loader.get_seasonal_adjustment(disease_key, crop_type)

        # Color evidence base boost (from _detect_color_symptoms equivalent)
        color_boost = 0.0
        for sig in profile.color_signatures:
            sig_key = f"{disease_key}:{sig['name']}"
            if sig_key in features.color_confidences and features.color_confidences[sig_key] > 0.1:
                color_boost += features.color_confidences[sig_key] * 0.3
        cls_score_adjusted = cls_score + color_boost

        # Apply seasonal
        cls_score_adjusted *= seasonal_mult

        # Combined score: YOLO-dominant weighting (validated by ablation study)
        # YOLO achieves 96.2% alone; rules degrade to 60.2% when weighted too heavily.
        # Use performance-proportional weights: YOLO acc / (YOLO acc + Rules acc)
        positive_matches = [m for m in all_matches if m.score_delta > 0]
        if positive_matches:
            final = min(1.0, cls_score_adjusted * 0.70 + rule_score * 0.30)
        else:
            final = min(1.0, cls_score_adjusted * 0.85 + rule_score * 0.15)
        final = max(0.0, final)

        # Build rejection if applicable
        rejection = _build_rejection(disease_key, profile, features, all_matches)
        if rejection:
            all_rejections.append(rejection)

        scored_candidates[disease_key] = CandidateScore(
            disease_key=disease_key,
            disease_name=profile.display_name,
            classifier_score=cls_score,
            rule_score=rule_score,
            seasonal_multiplier=seasonal_mult,
            final_score=final,
            matches=all_matches,
            rejection=rejection,
        )

    # Sort by final score
    sorted_candidates = sorted(
        scored_candidates.values(),
        key=lambda c: c.final_score,
        reverse=True,
    )

    if not sorted_candidates:
        return RuleEngineResult(
            candidates=[],
            top_disease="healthy",
            top_confidence=0.5,
            conflict=None,
            rejections=[],
        )

    top = sorted_candidates[0]

    # ── Resolve YOLO vs Rules conflict ──
    conflict = None
    if cls_predictions:
        yolo_top_key = cls_predictions[0].get("class_key", "")
        yolo_top_conf = cls_predictions[0].get("confidence", 0.0)

        # Find "pure rule" winner — highest rule_score regardless of classifier
        rule_only_sorted = sorted(
            scored_candidates.values(),
            key=lambda c: c.rule_score,
            reverse=True,
        )
        rule_top = rule_only_sorted[0]

        if yolo_top_key != top.disease_key or yolo_top_key != rule_top.disease_key:
            conflict = _resolve_conflict(
                yolo_top_key=yolo_top_key,
                yolo_top_conf=yolo_top_conf,
                rule_top_key=rule_top.disease_key,
                rule_top_score=rule_top.rule_score,
                candidates=scored_candidates,
            )

            # If rules win the conflict, reorder so rule winner is on top
            if conflict.winner == "rules" and top.disease_key != rule_top.disease_key:
                logger.info(
                    f"Conflict resolved: Rules override YOLO ({yolo_top_key} → {rule_top.disease_key})"
                )
                # Boost rule winner's final score to be above YOLO winner
                if rule_top.final_score < top.final_score:
                    boost = top.final_score - rule_top.final_score + 0.05
                    rule_top_key = rule_top.disease_key
                    scored_candidates[rule_top_key] = CandidateScore(
                        disease_key=rule_top_key,
                        disease_name=rule_top.disease_name,
                        classifier_score=rule_top.classifier_score,
                        rule_score=rule_top.rule_score,
                        seasonal_multiplier=rule_top.seasonal_multiplier,
                        final_score=min(1.0, rule_top.final_score + boost),
                        matches=rule_top.matches,
                        rejection=rule_top.rejection,
                    )
                    sorted_candidates = sorted(
                        scored_candidates.values(),
                        key=lambda c: c.final_score,
                        reverse=True,
                    )
                    top = sorted_candidates[0]

    logger.info(
        f"Rule engine: top={top.disease_key} (final={top.final_score:.2f}, "
        f"cls={top.classifier_score:.2f}, rules={top.rule_score:.2f}, "
        f"season={top.seasonal_multiplier:.1f}), "
        f"{len(all_rejections)} rejections, "
        f"conflict={'yes' if conflict and conflict.winner != 'agree' else 'no'}"
    )

    return RuleEngineResult(
        candidates=sorted_candidates,
        top_disease=top.disease_key,
        top_confidence=min(1.0, top.final_score),
        conflict=conflict,
        rejections=all_rejections,
    )


# ════════════════════════════════════════════════════════════════
# Serialization helpers
# ════════════════════════════════════════════════════════════════

def result_to_dict(result: RuleEngineResult) -> dict:
    """Convert RuleEngineResult to JSON-serializable dict."""
    return {
        "top_disease": result.top_disease,
        "top_confidence": round(result.top_confidence, 3),
        "candidates": [
            {
                "disease_key": c.disease_key,
                "disease_name": c.disease_name,
                "classifier_score": round(c.classifier_score, 3),
                "rule_score": round(c.rule_score, 3),
                "seasonal_multiplier": round(c.seasonal_multiplier, 2),
                "final_score": round(c.final_score, 3),
                "evidence": [m.explanation for m in c.matches if m.score_delta > 0],
                "contradictions": [m.explanation for m in c.matches if m.score_delta < 0],
            }
            for c in result.candidates[:5]
        ],
        "conflict": {
            "yolo_prediction": result.conflict.yolo_prediction,
            "yolo_confidence": round(result.conflict.yolo_confidence, 3),
            "rule_prediction": result.conflict.rule_prediction,
            "winner": result.conflict.winner,
            "reason": result.conflict.reason,
        } if result.conflict else None,
        "rejections": [
            {
                "disease": r.disease_name,
                "reasons": r.reasons,
                "missing_features": r.missing_features,
                "contradicting_features": r.contradicting_features,
            }
            for r in result.rejections[:5]
        ],
    }
