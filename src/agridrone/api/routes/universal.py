"""
universal.py - Universal Object Analyzer using LLaVA.

Provides endpoints to analyze ANY object (not just crops) via LLaVA vision model.
Supports: cropped object analysis, full-frame deep analysis, and auto-routing
between crop disease pipeline and general object analysis.
"""
import asyncio
import base64
import time
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

router = APIRouter(prefix="/universal", tags=["universal"])

# ── Crop-related YOLO classes (trigger disease pipeline) ──
CROP_CLASSES = {
    "healthy_crop", "leaf_rust", "goss_wilt", "fusarium_head_blight",
    "nitrogen_deficiency", "water_stress", "wheat_lodging",
    "volunteer_corn", "broadleaf_weed", "grass_weed",
    "good_plant_spacing", "poor_plant_spacing",
}

# ── Person classes (limited analysis) ──
PERSON_CLASSES = {"person"}

# ── Object category prompts ──
_GENERAL_OBJECT_PROMPT = """Analyze this image in detail. Identify:

1. **Object**: What is this object? Be specific (brand, model, type if recognizable)
2. **Category**: electronics / clothing / food / tool / document / plant / animal / furniture / vehicle / other
3. **Condition**: new / good / worn / damaged / unknown
4. **Details**: Color, material, size estimate, any text/logos visible
5. **Notable Features**: Anything interesting or unusual about this object
6. **Use Case**: What is this typically used for?

Respond ONLY in this JSON format (no markdown, no extra text):
{
  "object_name": "<specific name>",
  "category": "<category>",
  "brand": "<brand if visible, else null>",
  "model": "<model if recognizable, else null>",
  "condition": "<condition>",
  "color": "<primary color(s)>",
  "material": "<material if identifiable>",
  "text_visible": "<any text/labels visible, else null>",
  "notable_features": ["<list of notable features>"],
  "use_case": "<typical use>",
  "confidence": "<low|medium|high>",
  "description": "<2-3 sentence detailed description>"
}"""

_DEEP_ANALYZE_PROMPT = """Analyze this entire image in extreme detail. You are an expert visual analyst. Report:

1. **Scene**: What environment/setting is this? (office, field, lab, home, outdoor, etc.)
2. **All Objects**: List EVERY visible object with approximate position (left/center/right, foreground/background)
3. **People**: If any people visible, describe what they're doing (no personal identification)
4. **Text**: Any text, labels, signs, or writing visible
5. **Colors & Patterns**: Dominant colors, any patterns or textures
6. **Lighting**: Natural/artificial, time of day if outdoor
7. **Technical Details**: Image quality, camera angle, depth of field
8. **Agricultural Relevance**: If any plants, crops, soil, or agricultural elements visible, analyze their health
9. **Concerns**: Any safety issues, damage, or notable problems visible
10. **Recommendations**: Based on what you see, any actionable suggestions

Respond ONLY in this JSON format (no markdown, no extra text):
{
  "scene_type": "<environment description>",
  "scene_description": "<2-3 sentence overview>",
  "objects": [
    {"name": "<object>", "position": "<location>", "details": "<brief details>"}
  ],
  "people_count": <number>,
  "people_activity": "<what they're doing, or null>",
  "text_found": ["<any visible text>"],
  "dominant_colors": ["<color1>", "<color2>"],
  "lighting": "<lighting description>",
  "agricultural_elements": {
    "present": true/false,
    "details": "<description if present, else null>",
    "health_assessment": "<if crops visible>"
  },
  "concerns": ["<any issues noticed>"],
  "recommendations": ["<actionable suggestions>"],
  "confidence": "<low|medium|high>",
  "analysis_depth": "comprehensive"
}"""

_PLANT_ANALYZE_PROMPT = """You are an expert botanist and plant pathologist. Analyze this plant image in extreme detail.

Identify:
1. **Species**: What plant/crop species is this? Be as specific as possible
2. **Growth Stage**: seedling / vegetative / flowering / fruiting / mature / senescent
3. **Health Status**: healthy / stressed / diseased / damaged / dying
4. **Disease Symptoms**: Any spots, discoloration, wilting, lesions, fungal growth, pest damage?
5. **Pest Signs**: Any visible insects, eggs, webs, feeding damage?
6. **Nutrient Status**: Signs of deficiency (yellowing, purpling, stunting)?
7. **Environmental Stress**: Drought stress, waterlogging, heat damage, frost damage?
8. **Soil Visible**: If soil visible, assess: dry/moist/waterlogged, color, texture
9. **Severity**: How severe is any detected issue (0-100%)?
10. **Recommendations**: Specific treatment recommendations

Respond ONLY in this JSON format (no markdown, no extra text):
{
  "plant_species": "<species name>",
  "common_name": "<common name>",
  "growth_stage": "<stage>",
  "health_status": "<status>",
  "health_score": <0-100>,
  "diseases_detected": [
    {"name": "<disease>", "confidence": "<low|medium|high>", "severity_pct": <0-100>, "symptoms": "<visible symptoms>"}
  ],
  "pest_signs": [
    {"pest": "<pest name>", "evidence": "<what you see>"}
  ],
  "nutrient_status": "<normal|deficient|excess>",
  "nutrient_issues": ["<specific deficiencies>"],
  "environmental_stress": ["<stress types>"],
  "soil_assessment": "<if visible>",
  "overall_severity": <0-100>,
  "recommendations": ["<specific treatments with dosages>"],
  "confidence": "<low|medium|high>",
  "description": "<detailed 2-3 sentence analysis>"
}"""

_PERSON_PROMPT = """Briefly describe this scene. Focus on activity and environment only. Do NOT identify any person.
JSON only: {"activity":"<doing what>","setting":"<place>","objects_nearby":["<items>"],"description":"<1 sentence>"}"""


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class AnalyzeObjectRequest(BaseModel):
    """Request to analyze a cropped object region."""
    image_b64: str = Field(..., description="Base64-encoded JPEG image")
    object_class: Optional[str] = Field(None, description="YOLO detected class (if any)")
    bbox: Optional[dict] = Field(None, description="Bounding box {x1,y1,x2,y2}")


class DeepAnalyzeRequest(BaseModel):
    """Request for full-frame deep analysis."""
    image_b64: str = Field(..., description="Base64-encoded JPEG of full frame")


class AnalyzeResponse(BaseModel):
    """Universal analysis response."""
    analysis_type: str  # 'general_object' | 'plant' | 'person' | 'deep_scene' | 'crop_disease'
    result: dict
    raw_text: Optional[str] = None
    processing_ms: float
    model: str = "llava"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decode_b64_image(b64: str) -> np.ndarray:
    """Decode base64 to BGR numpy array."""
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode image")
    return image


def _classify_object(yolo_class: str | None, image: np.ndarray) -> str:
    """Determine what type of analysis to run based on YOLO class and image content.
    
    Returns: 'crop_disease' | 'plant' | 'person' | 'general_object'
    """
    if yolo_class:
        cls = yolo_class.lower().replace(" ", "_")
        if cls in CROP_CLASSES:
            return "crop_disease"
        if cls in PERSON_CLASSES:
            return "person"
        # Check if it looks plant-related by name
        plant_keywords = ["plant", "tree", "flower", "leaf", "crop", "grass", "weed", "seed", "fruit", "vegetable"]
        if any(kw in cls for kw in plant_keywords):
            return "plant"
        return "general_object"
    
    # No YOLO class — check if image has green (likely plant)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    green_pct = green_mask.sum() / (255 * green_mask.size)
    if green_pct > 0.3:
        return "plant"
    
    return "general_object"


def _get_prompt(analysis_type: str) -> str:
    """Get the appropriate LLaVA prompt for the analysis type."""
    prompts = {
        "general_object": _GENERAL_OBJECT_PROMPT,
        "plant": _PLANT_ANALYZE_PROMPT,
        "person": _PERSON_PROMPT,
        "deep_scene": _DEEP_ANALYZE_PROMPT,
    }
    return prompts.get(analysis_type, _GENERAL_OBJECT_PROMPT)


def _shrink(image: np.ndarray, max_dim: int = 384) -> np.ndarray:
    """Shrink image so largest side <= max_dim. Saves huge LLaVA time."""
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image
    scale = max_dim / max(h, w)
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


async def _run_llava(image: np.ndarray, prompt: str, max_dim: int = 384) -> dict | None:
    """Send image to LLaVA and return parsed result. Shrinks image first."""
    from ..app import _llava_analyze_sync
    small = _shrink(image, max_dim)
    logger.info(f"LLaVA input size: {small.shape[1]}x{small.shape[0]} (from {image.shape[1]}x{image.shape[0]})")
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _llava_analyze_sync, small, prompt)
    return result


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/analyze-object", response_model=AnalyzeResponse)
async def analyze_object(req: AnalyzeObjectRequest):
    """
    Analyze a single object from the camera feed.
    
    Auto-routes between:
    - Crop disease pipeline (for agricultural objects)
    - Plant analysis (for any vegetation)
    - Person description (for people)
    - General object analysis (for everything else)
    """
    t0 = time.time()
    
    try:
        image = _decode_b64_image(req.image_b64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")
    
    # Determine analysis type
    analysis_type = _classify_object(req.object_class, image)
    logger.info(f"Universal analyze: class={req.object_class} → type={analysis_type}")
    
    # For crop disease, redirect to the disease pipeline
    if analysis_type == "crop_disease":
        # Use the plant prompt but with crop focus
        analysis_type = "plant"
    
    prompt = _get_prompt(analysis_type)
    # Use smaller images for faster LLaVA — person/object don't need detail
    max_dim = 256 if analysis_type == "person" else 384
    result = await _run_llava(image, prompt, max_dim=max_dim)
    
    if not result:
        raise HTTPException(status_code=503, detail="LLaVA analysis failed. Is Ollama running?")
    
    processing_ms = round((time.time() - t0) * 1000, 1)
    
    return AnalyzeResponse(
        analysis_type=analysis_type,
        result=result.get("parsed") or {"raw_text": result.get("raw", "")},
        raw_text=result.get("raw", ""),
        processing_ms=processing_ms,
    )


@router.post("/deep-analyze", response_model=AnalyzeResponse)
async def deep_analyze(req: DeepAnalyzeRequest):
    """
    Full-frame deep analysis using LLaVA.
    
    Analyzes everything visible in the image: objects, text, colors,
    environment, agricultural elements, and provides recommendations.
    """
    t0 = time.time()
    
    try:
        image = _decode_b64_image(req.image_b64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")
    
    logger.info(f"Deep analysis requested: {image.shape[1]}x{image.shape[0]}")
    
    result = await _run_llava(image, _DEEP_ANALYZE_PROMPT, max_dim=512)
    
    if not result:
        raise HTTPException(status_code=503, detail="LLaVA analysis failed. Is Ollama running?")
    
    processing_ms = round((time.time() - t0) * 1000, 1)
    
    return AnalyzeResponse(
        analysis_type="deep_scene",
        result=result.get("parsed") or {"raw_text": result.get("raw", "")},
        raw_text=result.get("raw", ""),
        processing_ms=processing_ms,
    )
