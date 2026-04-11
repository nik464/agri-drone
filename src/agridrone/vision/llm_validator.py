"""
llm_validator.py — Use LLaVA as a VALIDATOR, not a predictor.

Instead of asking LLaVA "what disease is this?", we tell it what the rule engine
found and ask "do you agree?". This produces:
  - Structured validation (agree/disagree/partial with reasons)
  - LLM agreement score (0.0–1.0) for confidence fusion
  - Better prompts that leverage the image + our analysis together

Prompt templates for different scenarios:
  - VALIDATE: High-confidence → just confirm
  - ARBITRATE: YOLO vs Rules conflict → ask LLaVA to break the tie
  - DIFFERENTIATE: Multiple close candidates → ask LLaVA to distinguish
  - HEALTHY_CHECK: Classifier says healthy → verify no hidden symptoms

This module does NOT call Ollama directly. It builds the prompt and parses
the response. The actual HTTP call stays in app.py.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from loguru import logger

from agridrone.vision.feature_extractor import ImageFeatures
from agridrone.vision.rule_engine import RuleEngineResult


# ════════════════════════════════════════════════════════════════
# Data structures
# ════════════════════════════════════════════════════════════════

@dataclass
class LLMValidation:
    """Parsed LLaVA validation response."""
    agrees: bool                        # Does LLaVA agree with our diagnosis?
    agreement_score: float              # 0.0 (fully disagrees) to 1.0 (fully agrees)
    llm_diagnosis: str                  # What LLaVA thinks the disease is
    llm_confidence: str                 # "low" | "medium" | "high"
    reasons: list[str]                  # Why LLaVA agrees/disagrees
    visible_symptoms: str               # What LLaVA sees in the image
    health_score: int                   # LLaVA's health score estimate
    risk_level: str                     # "low" | "medium" | "high" | "critical"
    recommendations: list[str]          # Treatment recommendations
    urgency: str                        # "immediate" | "within_7_days" | "within_30_days"
    scenario: str                       # Which prompt template was used
    raw_response: str = ""              # Original LLaVA text (for debugging)


# ════════════════════════════════════════════════════════════════
# C2: Prompt templates for different scenarios
# ════════════════════════════════════════════════════════════════

def _template_validate(
    top_disease: str,
    top_confidence: float,
    evidence: list[str],
    crop_type: str,
) -> str:
    """High-confidence diagnosis — ask LLaVA to confirm or deny."""
    evidence_text = "\n".join(f"  - {e}" for e in evidence[:5]) if evidence else "  - (no specific visual evidence)"
    return f"""You are an expert plant pathologist validating an AI crop disease diagnosis.

OUR AI ANALYSIS found:
  Disease: {top_disease}
  Confidence: {top_confidence:.0%}
  Crop: {crop_type}
  Visual evidence detected:
{evidence_text}

YOUR TASK: Look at this {crop_type} image carefully and VALIDATE our diagnosis.

Answer these questions:
1. Do you AGREE with the diagnosis "{top_disease}"? (yes/partially/no)
2. What specific symptoms do you see in the image?
3. If you disagree, what disease do you think it is instead?
4. How confident are you? (low/medium/high)

Respond ONLY in this JSON format (no markdown fences):
{{
  "agrees": <true or false>,
  "agreement_level": "<full|partial|disagree>",
  "your_diagnosis": "<disease name you see>",
  "confidence": "<low|medium|high>",
  "visible_symptoms": "<describe exactly what you see in the image>",
  "reasons": ["<reason 1>", "<reason 2>"],
  "health_score": <integer 0-100>,
  "risk_level": "<low|medium|high|critical>",
  "recommendations": ["<treatment 1>", "<treatment 2>"],
  "urgency": "<immediate|within_7_days|within_30_days|seasonal>"
}}"""


def _template_arbitrate(
    yolo_disease: str,
    yolo_confidence: float,
    rule_disease: str,
    rule_confidence: float,
    yolo_rejections: list[str],
    crop_type: str,
) -> str:
    """YOLO vs Rules conflict — ask LLaVA to break the tie."""
    rejections_text = "\n".join(f"  - {r}" for r in yolo_rejections[:3]) if yolo_rejections else "  - (none)"
    return f"""You are an expert plant pathologist resolving a CONFLICT between two AI models analyzing a {crop_type} image.

MODEL 1 (YOLO Classifier) says:
  Disease: {yolo_disease}
  Confidence: {yolo_confidence:.0%}

MODEL 2 (Visual Rule Engine) says:
  Disease: {rule_disease}
  Confidence: {rule_confidence:.0%}
  Reasons YOLO's pick was rejected:
{rejections_text}

YOUR TASK: Look at this {crop_type} image carefully and decide which model is CORRECT.

Key visual clues to check:
- If you see LINEAR STRIPES of yellow-orange along leaf veins → supports Stripe/Yellow Rust
- If you see DISCRETE CIRCULAR spots with dark center → supports Tan Spot or Brown Spot
- If you see VIVID bright yellow-orange → supports Rust diseases
- If you see DULL muted tan/brown → supports Tan Spot or Septoria
- If you see BLEACHED heads → supports FHB or Blast
- If you see WHITE POWDER on leaves → supports Powdery Mildew

Respond ONLY in this JSON format (no markdown fences):
{{
  "agrees_with": "<yolo|rules|neither>",
  "your_diagnosis": "<disease name you see>",
  "confidence": "<low|medium|high>",
  "visible_symptoms": "<describe exactly what you see>",
  "reasons": ["<why you chose this model>", "<key visual evidence>"],
  "health_score": <integer 0-100>,
  "risk_level": "<low|medium|high|critical>",
  "recommendations": ["<treatment 1>", "<treatment 2>"],
  "urgency": "<immediate|within_7_days|within_30_days|seasonal>"
}}"""


def _template_differentiate(
    candidates: list[dict],
    crop_type: str,
) -> str:
    """Multiple close candidates — ask LLaVA to distinguish."""
    cand_text = "\n".join(
        f"  {i+1}. {c['name']} ({c['score']:.0%}) — {c['evidence']}"
        for i, c in enumerate(candidates[:4])
    )
    return f"""You are an expert plant pathologist choosing between multiple possible diagnoses for a {crop_type} image.

Our AI found these candidates (all close in confidence):
{cand_text}

YOUR TASK: Look at the image carefully and tell us which disease you see.

Pay special attention to:
- Color: vivid yellow-orange vs dull tan-brown vs dark reddish-brown
- Pattern: linear stripes vs scattered circular spots vs diamond-shaped lesions
- Distribution: along veins (stripe) vs random (spot/pustule) vs on heads only
- Texture: powdery vs rough/gritty vs water-soaked

Respond ONLY in this JSON format (no markdown fences):
{{
  "your_diagnosis": "<disease name you see>",
  "confidence": "<low|medium|high>",
  "visible_symptoms": "<describe exactly what you see>",
  "reasons": ["<why this disease>", "<what rules it out>"],
  "candidate_ranking": [1, 2, 3],
  "health_score": <integer 0-100>,
  "risk_level": "<low|medium|high|critical>",
  "recommendations": ["<treatment 1>", "<treatment 2>"],
  "urgency": "<immediate|within_7_days|within_30_days|seasonal>"
}}"""


def _template_healthy_check(
    classifier_confidence: float,
    features_summary: str,
    crop_type: str,
) -> str:
    """Classifier says healthy — verify no hidden symptoms."""
    return f"""You are an expert plant pathologist checking if a {crop_type} crop is truly healthy.

Our AI classifier says this {crop_type} is HEALTHY with {classifier_confidence:.0%} confidence.

Visual features detected:
  {features_summary}

YOUR TASK: Look carefully for ANY hidden or early-stage disease symptoms that the classifier might have missed:
- Small spots, even tiny ones
- Slight color changes (yellowing, browning)
- Any powder-like coating
- Water-soaked areas
- Early lesions at leaf tips or margins

It is CRITICAL to catch early infections before they spread. If you see even slight symptoms, report them.

Respond ONLY in this JSON format (no markdown fences):
{{
  "truly_healthy": <true or false>,
  "your_diagnosis": "<healthy or disease name>",
  "confidence": "<low|medium|high>",
  "visible_symptoms": "<describe what you see — or 'no symptoms visible'>",
  "reasons": ["<why healthy or why early disease>"],
  "health_score": <integer 0-100>,
  "risk_level": "<low|medium|high|critical>",
  "recommendations": ["<action>"],
  "urgency": "<immediate|within_7_days|within_30_days|seasonal>"
}}"""


# ════════════════════════════════════════════════════════════════
# C1: Build the right prompt based on scenario
# ════════════════════════════════════════════════════════════════

def build_validation_prompt(
    rule_result: RuleEngineResult,
    features: ImageFeatures,
    classifier_result: dict | None,
    crop_type: str = "wheat",
) -> tuple[str, str]:
    """Choose the right prompt template based on the diagnostic scenario.

    Returns (prompt_text, scenario_name).
    """
    top = rule_result.candidates[0] if rule_result.candidates else None
    conflict = rule_result.conflict

    # ── Scenario 1: CONFLICT — YOLO and rules disagree ──
    if conflict and conflict.winner != "agree":
        scenario = "arbitrate"
        yolo_rejections = conflict.yolo_rejections or []
        prompt = _template_arbitrate(
            yolo_disease=conflict.yolo_prediction,
            yolo_confidence=conflict.yolo_confidence,
            rule_disease=conflict.rule_prediction,
            rule_confidence=conflict.rule_confidence,
            yolo_rejections=yolo_rejections,
            crop_type=crop_type,
        )
        return prompt, scenario

    # ── Scenario 2: HEALTHY — check for hidden symptoms ──
    if top and top.disease_key.startswith("healthy"):
        scenario = "healthy_check"
        features_summary = _summarize_features(features)
        cls_conf = top.classifier_score
        prompt = _template_healthy_check(
            classifier_confidence=cls_conf,
            features_summary=features_summary,
            crop_type=crop_type,
        )
        return prompt, scenario

    # ── Scenario 3: LOW CONFIDENCE / CLOSE CANDIDATES — differentiate ──
    if (
        top
        and len(rule_result.candidates) >= 2
        and rule_result.candidates[1].final_score > top.final_score * 0.6
    ):
        scenario = "differentiate"
        cands = [
            {
                "name": c.disease_name,
                "score": c.final_score,
                "evidence": "; ".join(
                    m.explanation for m in c.matches if m.score_delta > 0
                )[:120] or "classifier prediction only",
            }
            for c in rule_result.candidates[:4]
        ]
        prompt = _template_differentiate(candidates=cands, crop_type=crop_type)
        return prompt, scenario

    # ── Scenario 4: HIGH CONFIDENCE — just validate ──
    scenario = "validate"
    evidence_list = [m.explanation for m in top.matches if m.score_delta > 0] if top else []
    prompt = _template_validate(
        top_disease=top.disease_name if top else "Unknown",
        top_confidence=top.final_score if top else 0,
        evidence=evidence_list,
        crop_type=crop_type,
    )
    return prompt, scenario


def _summarize_features(features: ImageFeatures) -> str:
    """One-line summary of extracted features for prompts."""
    parts = []
    if features.has_stripe_pattern:
        parts.append(f"stripe pattern ({features.stripe_confidence:.0%})")
    if features.has_spot_pattern:
        parts.append(f"spot pattern ({features.spot_confidence:.0%})")
    if features.has_vivid_yellow:
        parts.append(f"vivid yellow-orange ({features.vivid_yellow_orange_ratio:.1%})")
    if features.has_bleaching:
        parts.append(f"bleaching ({features.bleaching_ratio:.1%})")
    if features.green_ratio > 0.3:
        parts.append(f"green coverage {features.green_ratio:.0%}")
    parts.append(f"{len(features.color_confidences)} color signatures detected")
    return ", ".join(parts) if parts else "no significant visual features detected"


# ════════════════════════════════════════════════════════════════
# C3: Parse LLaVA validation response + compute agreement score
# ════════════════════════════════════════════════════════════════

def parse_validation_response(
    raw: str,
    scenario: str,
    our_diagnosis: str,
) -> LLMValidation:
    """Parse LLaVA's structured validation response into LLMValidation.

    Robust: handles JSON, markdown fences, and regex fallback.
    """
    text = raw.strip()

    # Remove markdown code fences
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        parts = text.split("```")
        for part in parts[1:]:
            stripped = part.strip()
            if stripped.startswith("{"):
                text = stripped
                break

    # Fix common JSON errors
    text = re.sub(r",\s*}", "}", text)
    text = re.sub(r",\s*]", "]", text)

    parsed = None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Regex fallback
        parsed = _regex_parse_fallback(raw)

    if not parsed:
        logger.warning(f"Failed to parse LLaVA validation response ({len(raw)} chars)")
        return LLMValidation(
            agrees=True,
            agreement_score=0.5,
            llm_diagnosis=our_diagnosis,
            llm_confidence="low",
            reasons=["LLaVA response could not be parsed"],
            visible_symptoms=raw[:200] if raw else "",
            health_score=50,
            risk_level="medium",
            recommendations=[],
            urgency="within_7_days",
            scenario=scenario,
            raw_response=raw,
        )

    # Extract common fields
    llm_diagnosis = (
        parsed.get("your_diagnosis", "")
        or parsed.get("disease", "")
        or our_diagnosis
    )
    confidence = parsed.get("confidence", "medium")
    visible = parsed.get("visible_symptoms", "")
    reasons = parsed.get("reasons", [])
    if isinstance(reasons, str):
        reasons = [reasons]
    health_score = parsed.get("health_score", 50)
    risk_level = parsed.get("risk_level", "medium")
    recommendations = parsed.get("recommendations", [])
    urgency = parsed.get("urgency", "within_7_days")

    # ── Compute agreement score based on scenario ──
    agrees, agreement_score = _compute_agreement(parsed, scenario, our_diagnosis, llm_diagnosis)

    return LLMValidation(
        agrees=agrees,
        agreement_score=agreement_score,
        llm_diagnosis=llm_diagnosis,
        llm_confidence=confidence,
        reasons=reasons,
        visible_symptoms=visible,
        health_score=health_score,
        risk_level=risk_level,
        recommendations=recommendations,
        urgency=urgency,
        scenario=scenario,
        raw_response=raw,
    )


def _compute_agreement(
    parsed: dict,
    scenario: str,
    our_diagnosis: str,
    llm_diagnosis: str,
) -> tuple[bool, float]:
    """Compute agreement boolean + score (0.0–1.0) based on scenario.

    Returns (agrees: bool, agreement_score: float).
    """
    our_lower = our_diagnosis.lower().strip()
    llm_lower = llm_diagnosis.lower().strip()

    # ── VALIDATE scenario ──
    if scenario == "validate":
        explicit_agree = parsed.get("agrees", None)
        level = parsed.get("agreement_level", "").lower()

        if explicit_agree is True or level == "full":
            return True, 1.0
        elif level == "partial":
            return True, 0.6
        elif explicit_agree is False or level == "disagree":
            # LLaVA disagrees — check if same disease family
            if _same_disease_family(our_lower, llm_lower):
                return False, 0.3
            return False, 0.1
        else:
            # Heuristic: check if LLaVA named the same disease
            if _disease_name_match(our_lower, llm_lower):
                return True, 0.85
            elif _same_disease_family(our_lower, llm_lower):
                return True, 0.5
            return False, 0.2

    # ── ARBITRATE scenario ──
    elif scenario == "arbitrate":
        agrees_with = parsed.get("agrees_with", "").lower()
        if agrees_with == "rules":
            return True, 0.9     # LLaVA agrees with our rule engine
        elif agrees_with == "yolo":
            return False, 0.15   # LLaVA agrees with YOLO over rules
        else:
            # "neither" or LLaVA named a third disease
            if _disease_name_match(our_lower, llm_lower):
                return True, 0.7
            return False, 0.2

    # ── DIFFERENTIATE scenario ──
    elif scenario == "differentiate":
        ranking = parsed.get("candidate_ranking", [])
        if ranking and ranking[0] == 1:
            return True, 0.85    # LLaVA ranks our top pick first
        elif _disease_name_match(our_lower, llm_lower):
            return True, 0.8
        elif _same_disease_family(our_lower, llm_lower):
            return True, 0.5
        return False, 0.2

    # ── HEALTHY_CHECK scenario ──
    elif scenario == "healthy_check":
        truly_healthy = parsed.get("truly_healthy", True)
        if truly_healthy and "healthy" in our_lower:
            return True, 0.95    # Both say healthy
        elif not truly_healthy and "healthy" in our_lower:
            return False, 0.1    # LLaVA found hidden disease!
        elif truly_healthy and "healthy" not in our_lower:
            return False, 0.2
        return True, 0.5

    # Default
    if _disease_name_match(our_lower, llm_lower):
        return True, 0.8
    return False, 0.3


def _disease_name_match(name_a: str, name_b: str) -> bool:
    """Check if two disease names refer to the same disease (fuzzy match)."""
    if not name_a or not name_b:
        return False
    # Exact match
    if name_a == name_b:
        return True
    # One contains the other
    if name_a in name_b or name_b in name_a:
        return True
    # Key word overlap
    keywords_a = set(name_a.replace("/", " ").replace("-", " ").replace("_", " ").split())
    keywords_b = set(name_b.replace("/", " ").replace("-", " ").replace("_", " ").split())
    # Remove common filler words
    filler = {"wheat", "rice", "leaf", "head", "crop", "the", "a", "of"}
    keywords_a -= filler
    keywords_b -= filler
    if not keywords_a or not keywords_b:
        return False
    overlap = keywords_a & keywords_b
    return len(overlap) >= 1


def _same_disease_family(name_a: str, name_b: str) -> bool:
    """Check if two diseases are in the same family (e.g., all rusts)."""
    families = [
        {"rust", "yellow rust", "stripe rust", "brown rust", "leaf rust", "black rust", "stem rust"},
        {"blast", "rice blast", "wheat blast"},
        {"blight", "fhb", "fusarium", "scab", "head blight", "leaf blight", "bacterial blight"},
        {"spot", "tan spot", "brown spot"},
        {"mildew", "powdery mildew"},
        {"healthy"},
    ]
    for family in families:
        in_a = any(term in name_a for term in family)
        in_b = any(term in name_b for term in family)
        if in_a and in_b:
            return True
    return False


def _regex_parse_fallback(raw: str) -> dict | None:
    """Extract JSON fields via regex when JSON parsing fails."""
    try:
        result = {}
        # Boolean/string fields
        for field_name in ["agrees", "truly_healthy"]:
            m = re.search(rf'"{field_name}"\s*:\s*(true|false)', raw, re.IGNORECASE)
            if m:
                result[field_name] = m.group(1).lower() == "true"

        for field_name in [
            "your_diagnosis", "confidence", "visible_symptoms",
            "risk_level", "urgency", "agreement_level", "agrees_with",
        ]:
            m = re.search(rf'"{field_name}"\s*:\s*"([^"]*)"', raw)
            if m:
                result[field_name] = m.group(1)

        for field_name in ["health_score"]:
            m = re.search(rf'"{field_name}"\s*:\s*(\d+)', raw)
            if m:
                result[field_name] = int(m.group(1))

        for field_name in ["reasons", "recommendations"]:
            m = re.search(rf'"{field_name}"\s*:\s*\[([^\]]*)\]', raw)
            if m:
                result[field_name] = re.findall(r'"([^"]*)"', m.group(1))

        # candidate_ranking
        m = re.search(r'"candidate_ranking"\s*:\s*\[([^\]]*)\]', raw)
        if m:
            result["candidate_ranking"] = [int(x) for x in re.findall(r'\d+', m.group(1))]

        return result if result else None
    except Exception:
        return None


# ════════════════════════════════════════════════════════════════
# C3: Fuse LLM agreement into confidence
# ════════════════════════════════════════════════════════════════

def fuse_confidence(
    rule_confidence: float,
    llm_validation: LLMValidation | None,
    classifier_confidence: float = 0.0,
) -> dict:
    """Fuse rule engine confidence with LLM agreement score.

    Weights:
      - Rule engine: 50% (visual evidence is the foundation)
      - LLM validation: 30% (if available, 0% otherwise)
      - Classifier: 20% (baseline, redistributed if no LLM)

    Returns dict with fused_confidence, component scores, and explanation.
    """
    if llm_validation and llm_validation.agreement_score > 0:
        # All three sources available
        rule_weight = 0.50
        llm_weight = 0.30
        cls_weight = 0.20

        fused = (
            rule_confidence * rule_weight
            + llm_validation.agreement_score * llm_weight
            + classifier_confidence * cls_weight
        )

        # Agreement bonus: if LLM strongly agrees, boost confidence
        if llm_validation.agrees and llm_validation.agreement_score >= 0.8:
            fused = min(1.0, fused * 1.1)
            note = "LLM strongly validates diagnosis — confidence boosted"
        # Disagreement penalty: if LLM disagrees, cap confidence
        elif not llm_validation.agrees and llm_validation.agreement_score < 0.3:
            fused = min(fused, 0.6)
            note = "LLM disagrees — confidence capped for safety"
        else:
            note = "LLM partially validates diagnosis"

        return {
            "fused_confidence": round(min(1.0, fused), 3),
            "rule_confidence": round(rule_confidence, 3),
            "llm_agreement_score": round(llm_validation.agreement_score, 3),
            "classifier_confidence": round(classifier_confidence, 3),
            "weights": {"rule": rule_weight, "llm": llm_weight, "classifier": cls_weight},
            "llm_agrees": llm_validation.agrees,
            "llm_scenario": llm_validation.scenario,
            "note": note,
        }
    else:
        # No LLM — redistribute weight to rule engine
        rule_weight = 0.65
        cls_weight = 0.35

        fused = rule_confidence * rule_weight + classifier_confidence * cls_weight

        return {
            "fused_confidence": round(min(1.0, fused), 3),
            "rule_confidence": round(rule_confidence, 3),
            "llm_agreement_score": None,
            "classifier_confidence": round(classifier_confidence, 3),
            "weights": {"rule": rule_weight, "llm": 0, "classifier": cls_weight},
            "llm_agrees": None,
            "llm_scenario": None,
            "note": "No LLM validation available — using rule + classifier only",
        }


def validation_to_dict(v: LLMValidation) -> dict:
    """Convert LLMValidation to JSON-serializable dict."""
    return {
        "agrees": v.agrees,
        "agreement_score": round(v.agreement_score, 3),
        "llm_diagnosis": v.llm_diagnosis,
        "llm_confidence": v.llm_confidence,
        "reasons": v.reasons,
        "visible_symptoms": v.visible_symptoms,
        "health_score": v.health_score,
        "risk_level": v.risk_level,
        "recommendations": v.recommendations,
        "urgency": v.urgency,
        "scenario": v.scenario,
    }
