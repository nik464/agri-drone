"""
app.py - FastAPI application and routes.
"""
import asyncio
import base64
import csv
import hashlib
import io
import json
import os
import re
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import cv2
import numpy as np
import requests
from fastapi import FastAPI, Form, HTTPException, File, UploadFile, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from .. import __version__


# ── Background LLaVA results cache (keyed by image hash) ──
_LLAVA_RESULTS: dict[str, dict | None] = {}   # hash -> result
_LLAVA_PENDING: set[str] = set()                # hashes currently being analyzed
_LLAVA_CONTEXT: dict[str, dict] = {}            # hash -> {scenario, our_diagnosis} for validation parsing

MAX_IMAGE_DIM = 1280  # Downsize images larger than this before inference

# ── Plant image gatekeeper ──
# Face detector (loaded once, reused)
_FACE_CASCADE = None

def _get_face_cascade():
    """Lazy-load OpenCV Haar cascade face detector."""
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _FACE_CASCADE = cv2.CascadeClassifier(cascade_path)
        logger.info(f"Loaded face detector: {cascade_path}")
    return _FACE_CASCADE


def _detect_faces(image_bgr: np.ndarray) -> list:
    """Detect faces using Haar cascade. Returns list of (x,y,w,h) rectangles."""
    cascade = _get_face_cascade()
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    # Resize for speed if image is large
    h_img, w_img = gray.shape[:2]
    scale = 1.0
    if max(h_img, w_img) > 600:
        scale = 600 / max(h_img, w_img)
        gray = cv2.resize(gray, (int(w_img * scale), int(h_img * scale)))
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    if len(faces) == 0:
        return []
    # Scale back to original coordinates
    return [(int(x / scale), int(y / scale), int(w / scale), int(h_val / scale))
            for (x, y, w, h_val) in faces]


def _is_plant_image(image_bgr: np.ndarray) -> dict:
    """Layer 1 — Physics / Spectral Gate.

    Rejects non-vegetation images using multi-signal analysis:
    1. Face detection (Haar cascade)
    2. Skin-pixel ratio (YCrCb)
    3. Green + brown vegetation pixel ratio (HSV)
    4. Largest green blob analysis — a real leaf forms a large connected region
    5. Achromatic ratio — paper/pens/screens are mostly gray/white/dark
    6. Spectral vegetation indices (GLI, ExG, RGRI, NGRDI, VARI)
    7. Texture uniformity — catches green-painted surfaces / artificial turf

    Core principle: if the image contains a green connected region covering >2%
    of the frame, it almost certainly contains a real plant → PASS.
    If there's no meaningful green blob, use spectral + achromatic signals to reject.

    Returns dict with diagnostic signals.
    """
    img_h, img_w = image_bgr.shape[:2]
    total_pixels = img_h * img_w

    # ── 1. FACE DETECTION ──
    faces = _detect_faces(image_bgr)
    face_count = len(faces)
    face_area = sum(w * h_val for (_, _, w, h_val) in faces)
    face_area_pct = face_area / total_pixels if total_pixels > 0 else 0

    # ── 2. Skin detection (YCrCb color space — catches fingers/hands) ──
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
    skin_mask = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)
    skin_ratio = float(np.count_nonzero(skin_mask) / total_pixels)

    # ── 3. Green vegetation check ──
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    green_mask = (h >= 30) & (h <= 90) & (s > 25) & (v > 30)
    green_ratio = float(np.count_nonzero(green_mask) / total_pixels)

    # ── 4. Brown vegetation (dried leaves, stems, soil) ──
    # Tightened: exclude paper/cardboard by requiring V < 200 and S > 60
    brown_veg = (h >= 10) & (h <= 28) & (s > 60) & (v > 40) & (v < 200)
    brown_ratio = float(np.count_nonzero(brown_veg) / total_pixels)
    vegetation_ratio = green_ratio + brown_ratio

    # ── 5. Texture uniformity within green region ──
    texture_std = 0.0
    green_pixel_count = np.count_nonzero(green_mask)
    if green_pixel_count > 100:
        sat_in_green = s[green_mask].astype(np.float32)
        texture_std = float(np.std(sat_in_green))

    # ── 6. Largest green blob — THE key signal ──
    # A real plant leaf forms a large connected green region.
    # Paper, pens, office items, random objects have no large green blob.
    # If the largest green blob covers >2% of the image, it's almost certainly a real plant.
    largest_green_blob_ratio = 0.0
    green_coherence = 0.0
    if green_pixel_count > 100:
        green_u8 = green_mask.astype(np.uint8) * 255
        n_labels, _labels, stats, _ = cv2.connectedComponentsWithStats(
            green_u8, connectivity=8
        )
        if n_labels > 1:
            largest_blob_area = int(stats[1:, cv2.CC_STAT_AREA].max())
            largest_green_blob_ratio = largest_blob_area / total_pixels
            green_coherence = largest_blob_area / green_pixel_count
        else:
            green_coherence = 1.0

    has_plant_region = largest_green_blob_ratio > 0.02  # 2% of image

    # ── 7. Achromatic ratio — paper/pen/screens are mostly unsaturated ──
    # Pixels with S < 25 are gray/white/black (no color → not a plant)
    achromatic_mask = s < 25
    achromatic_ratio = float(np.count_nonzero(achromatic_mask) / total_pixels)

    # ── 8. Edge density (diagnostic only, not used for rejection) ──
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.count_nonzero(edges) / total_pixels)

    # ══════════════════════════════════════════════════════════════
    # DECISION LOGIC
    # Core principle: default is REJECT. Image must prove it has a plant.
    # A real plant image has a large connected green region (has_plant_region).
    # ══════════════════════════════════════════════════════════════
    is_plant = False  # guilty until proven innocent
    reason = "No plant or crop features detected in this image."

    # ── D1. HAS a real green plant region → PASS (highest priority) ──
    # If there's a real leaf blob (>2% of image), it's a plant regardless of
    # faces, skin, edges, or background complexity. This covers:
    # - Farmer holding a leaf (hand + plant)
    # - Plant in an office/lab setting
    # - Field photo with people in background
    if has_plant_region:
        is_plant = True
        reason = ""
        logger.info(
            f"Green plant region found (largest blob {largest_green_blob_ratio:.1%} of image, "
            f"coherence {green_coherence:.0%}) — accepting as plant"
        )

    # ── D2. No green blob — check if face/person without plant ──
    elif face_count > 0 and face_area_pct > 0.01:
        is_plant = False
        reason = (
            f"Human face detected ({face_count} face{'s' if face_count > 1 else ''}, "
            f"covering {face_area_pct:.0%}) without a visible crop. "
            "Please upload a close-up photo of a crop leaf."
        )

    # ── D3. No green blob, mostly skin → person without plant ──
    elif skin_ratio > 0.25 and not has_plant_region:
        is_plant = False
        reason = (
            f"Person detected without crop (skin {skin_ratio:.0%}). "
            "Please upload a crop photo."
        )

    # ── D4. No green blob + mostly achromatic → paper/pen/screen/wall ──
    elif achromatic_ratio > 0.50 and not has_plant_region:
        is_plant = False
        reason = (
            f"No plant detected — image is mostly gray/white/dark "
            f"({achromatic_ratio:.0%} achromatic). Please upload a crop photo."
        )

    # ── D5. Some scattered vegetation but no cohesive blob → not a crop ──
    elif vegetation_ratio > 0.03 and not has_plant_region:
        is_plant = False
        reason = (
            f"Some green detected ({green_ratio:.0%}) but no crop leaf region found. "
            "Please upload a close-up photo of a crop leaf."
        )

    # ── D6. Fallthrough — no green blob, no special signals → reject ──
    # Default is already reject (is_plant = False)

    # ── SPECTRAL VEGETATION GATE (hardened with RGRI + texture) ──
    # Only applies if image passed via blob detection. If has_plant_region,
    # do NOT let the spectral gate override — real plants can have odd spectral.
    spectral_info: dict = {}
    if is_plant and not has_plant_region:
        try:
            from ..core.spectral_features import extract_spectral_indices
            spectral = extract_spectral_indices(image_bgr)
            vari = spectral.indices.get("VARI")
            gli = spectral.indices.get("GLI")
            exg = spectral.indices.get("ExG")
            ngrdi = spectral.indices.get("NGRDI")
            rgri = spectral.indices.get("RGRI")

            # Count non-vegetation signals (5 indices now)
            non_veg_count = 0
            if vari and vari.mean < 0.10:
                non_veg_count += 1
            if gli and gli.mean < 0.01:
                non_veg_count += 1
            if exg and exg.mean < 0.02:
                non_veg_count += 1
            if ngrdi and ngrdi.mean < 0.01:
                non_veg_count += 1
            # RGRI > 1.3 means red dominates green — not healthy vegetation
            if rgri and rgri.mean > 1.3:
                non_veg_count += 1

            spectral_info = {
                "vari": round(vari.mean, 4) if vari else None,
                "gli": round(gli.mean, 4) if gli else None,
                "exg": round(exg.mean, 4) if exg else None,
                "ngrdi": round(ngrdi.mean, 4) if ngrdi else None,
                "rgri": round(rgri.mean, 4) if rgri else None,
                "non_veg_signals": non_veg_count,
                "texture_std": round(texture_std, 2),
            }

            gli_ok = gli and gli.mean >= 0.01
            exg_ok = exg and exg.mean >= 0.02

            should_reject = False
            reject_reason = ""

            # Rule 1: low green + both GLI & ExG fail → not a crop
            if green_ratio < 0.15 and not gli_ok and not exg_ok:
                should_reject = True
                reject_reason = "low green + GLI/ExG both below threshold"
            # Rule 2: very low green + majority of indices fail
            elif green_ratio < 0.08 and non_veg_count >= 3:
                should_reject = True
                reject_reason = f"very low green + {non_veg_count}/5 non-veg signals"
            # Rule 3: green is present but texture is too uniform → painted surface
            elif green_ratio > 0.15 and texture_std < 8.0 and not gli_ok and not exg_ok:
                should_reject = True
                reject_reason = f"uniform texture (std={texture_std:.1f}) + spectral fail"
            # Rule 4: high RGRI (red >> green) with low GLI — necrotic non-plant
            elif rgri and rgri.mean > 1.5 and not gli_ok and green_ratio < 0.10:
                should_reject = True
                reject_reason = f"high RGRI ({rgri.mean:.2f}) + low GLI"

            if should_reject:
                is_plant = False
                vals = (
                    f"VARI={vari.mean:.3f}, GLI={gli.mean:.3f}, "
                    f"ExG={exg.mean:.3f}, NGRDI={ngrdi.mean:.3f}"
                )
                if rgri:
                    vals += f", RGRI={rgri.mean:.3f}"
                reason = (
                    f"Not a crop — {reject_reason}. "
                    f"Spectral: {vals}. "
                    "Please upload a photo of a crop leaf or plant."
                )
                logger.info(
                    f"Spectral gate REJECTED: {reject_reason} | "
                    f"gli_ok={gli_ok} exg_ok={exg_ok} "
                    f"non_veg={non_veg_count}/5 green={green_ratio:.1%} "
                    f"texture_std={texture_std:.1f}"
                )
        except Exception as e:
            logger.warning(f"Spectral gatekeeper check failed (non-critical): {e}")

    logger.info(
        f"Layer-1 Physics Gate: faces={face_count} face_area={face_area_pct:.1%} "
        f"skin={skin_ratio:.1%} green={green_ratio:.1%} brown={brown_ratio:.1%} "
        f"texture_std={texture_std:.1f} edge_density={edge_density:.1%} "
        f"green_coherence={green_coherence:.1%} blob={largest_green_blob_ratio:.1%} "
        f"achromatic={achromatic_ratio:.1%} has_plant={has_plant_region} "
        f"→ {'PASS' if is_plant else 'REJECT'}"
    )
    return {
        "is_plant": is_plant,
        "reason": reason,
        "green_ratio": round(green_ratio, 4),
        "face_count": face_count,
        "face_area_pct": round(face_area_pct, 4),
        "skin_ratio": round(skin_ratio, 4),
        "edge_density": round(edge_density, 4),
        "green_coherence": round(green_coherence, 4),
        "largest_green_blob_ratio": round(largest_green_blob_ratio, 4),
        "achromatic_ratio": round(achromatic_ratio, 4),
        "has_plant_region": has_plant_region,
        **spectral_info,
    }


MIN_IMAGE_DIM = 32  # Minimum pixels on shortest side for meaningful classification


def _check_minimum_size(image: np.ndarray) -> dict | None:
    """Return rejection dict if image is too small for meaningful classification, else None."""
    h, w = image.shape[:2]
    if min(h, w) < MIN_IMAGE_DIM:
        return {
            "rejected": True,
            "is_plant": False,
            "rejection_reason": (
                f"Image too small ({w}x{h} px). "
                f"Minimum {MIN_IMAGE_DIM}x{MIN_IMAGE_DIM} px required for crop disease analysis."
            ),
            "rejection_layer": 0,
            "width": w,
            "height": h,
        }
    return None


def _resize_if_large(image: np.ndarray, max_dim: int = MAX_IMAGE_DIM) -> np.ndarray:
    """Downsize image if either dimension exceeds max_dim. Preserves aspect ratio."""
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image
    scale = max_dim / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    logger.info(f"Resizing image from {w}x{h} to {new_w}x{new_h} for faster inference")
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _image_hash(image: np.ndarray) -> str:
    """Fast perceptual hash of image bytes for caching."""
    small = cv2.resize(image, (64, 64))
    return hashlib.md5(small.tobytes()).hexdigest()


# ── LLaVA Configuration ──
_OLLAMA_URL = "http://localhost:11434"
_LLAVA_MODEL = "llava"

_LLAVA_PROMPT = """You are an expert plant pathologist specializing in Indian and global wheat and rice diseases. Analyze this image carefully.

IMPORTANT — FIRST CHECK: Before any diagnosis, determine if this image actually contains a plant, crop, or agricultural field.
- If the image shows a person, animal, vehicle, building, food, or any non-plant subject, respond ONLY with:
  {"is_plant": false, "health_score": 0, "risk_level": "none", "diseases_found": [], "confidence": "high", "visible_symptoms": "This image does not contain a plant or crop", "affected_area_pct": 0, "recommendations": ["Please upload a photo of a crop leaf or agricultural field"], "urgency": "none"}
- If the image is unclear or ambiguous, set "confidence": "low" and note your uncertainty.
- Only proceed with disease analysis if you can clearly see plant/crop material.

DISEASE REFERENCE GUIDE:

1. Fusarium Head Blight (FHB / Scab):
   - Orange, salmon-pink, or tan spore masses on wheat heads
   - Bleached/whitened spikelets while adjacent spikelets remain green
   - Shriveled, chalky-white or pink-tinged kernels

2. Wheat Leaf Rust (Puccinia triticina):
   - Small round orange-brown pustules scattered on upper leaf surface
   - Leaves feel rough/gritty to touch

3. Yellow/Stripe Rust (Puccinia striiformis):
   - Bright yellow-orange pustules in distinct linear stripes along leaf veins

4. Powdery Mildew (Blumeria graminis):
   - White to gray powdery fungal growth on leaf surfaces and stems

5. Rice Blast (Magnaporthe oryzae):
   - Diamond/spindle-shaped lesions with gray center and brown margin

6. Healthy Crop:
   - Uniform dark green color, upright stems, no visible spots or discoloration

ANALYSIS INSTRUCTIONS:
- Be very specific. If you see ANY orange, pink, brown discoloration on wheat heads or leaves, identify it as a disease.
- A wheat head with mixed bleached and green spikelets is a STRONG indicator of Fusarium Head Blight.
- Orange/salmon coloring on wheat heads is almost certainly FHB or rust — NEVER call it healthy.
- Do not default to "healthy" unless the crop is uniformly green with zero visible symptoms.

HEALTH SCORE CALIBRATION (follow strictly):
- Fusarium Head Blight: score 25-50 (FHB causes >45% yield loss — always rate as high/critical risk)
- Wheat Blast / Rice Blast: score 20-45 (devastating, spreads rapidly)
- Yellow/Stripe Rust: score 30-50 (aggressive, spreads fast in cool weather)
- Black Rust / Stem Rust: score 25-45 (historically catastrophic)
- Brown Rust / Leaf Rust: score 40-60 (significant but slower spread)
- Powdery Mildew: score 50-65 (moderate if caught early)
- Bacterial Leaf Blight: score 30-50 (serious in rice)
- Multiple diseases: score 15-35 (compound damage)
- Healthy crop (zero symptoms): score 85-100
- Minor/early stage single issue: score 60-75
NEVER give a score above 60 for Fusarium Head Blight, any Blast, or Stem Rust — these are economically devastating diseases.

Respond ONLY in this JSON format (no markdown fences, no extra text):
{
  "health_score": <integer 0-100, use calibration above>,
  "risk_level": "<low|medium|high|critical>",
  "diseases_found": ["list of diseases seen"],
  "confidence": "<low|medium|high>",
  "visible_symptoms": "<describe exactly what you see>",
  "affected_area_pct": <integer 0-100>,
  "recommendations": ["list of actions using Indian inputs like Propiconazole 25% EC, Tricyclazole 75% WP, Tebuconazole 25.9% EC"],
  "urgency": "<immediate|within_7_days|within_30_days|seasonal>"
}"""


def _llava_analyze_sync(image_bgr: np.ndarray, prompt: str | None = None) -> dict | None:
    """Send image to LLaVA via Ollama for visual disease diagnosis (blocking).

    If prompt is provided, uses that instead of the default _LLAVA_PROMPT.
    """  
    try:
        success, encoded = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 60])
        if not success:
            return None
        image_b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")

        actual_prompt = prompt or _LLAVA_PROMPT
        logger.info("Sending image to LLaVA for visual diagnosis...")
        resp = requests.post(
            f"{_OLLAMA_URL}/api/chat",
            json={
                "model": _LLAVA_MODEL,
                "messages": [{
                    "role": "user",
                    "content": actual_prompt,
                    "images": [image_b64],
                }],
                "stream": False,
            },
            timeout=120,
        )
        resp.raise_for_status()
        raw = resp.json()["message"]["content"].strip()
        logger.info(f"LLaVA raw response length: {len(raw)} chars")
        return {"raw": raw, "parsed": _parse_llava_response(raw)}
    except Exception as exc:
        logger.warning(f"LLaVA analysis failed: {exc}")
        return None


def _parse_llava_response(raw: str) -> dict | None:
    """Robustly parse LLaVA output (handles markdown fences, trailing commas, etc.)."""
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

    # Try standard JSON parse
    try:
        result = json.loads(text)
        result.setdefault("health_score", 50)
        result.setdefault("risk_level", "medium")
        result.setdefault("diseases_found", [])
        result.setdefault("confidence", "medium")
        result.setdefault("visible_symptoms", "")
        result.setdefault("affected_area_pct", 0)
        result.setdefault("recommendations", [])
        result.setdefault("urgency", "within_7_days")
        return result
    except json.JSONDecodeError:
        pass

    # Regex fallback
    try:
        health_m = re.search(r'"health_score"\s*:\s*(\d+)', raw)
        risk_m = re.search(r'"risk_level"\s*:\s*"([^"]+)"', raw)
        diseases_m = re.search(r'"diseases_found"\s*:\s*\[([^\]]+)\]', raw)
        symptoms_m = re.search(r'"visible_symptoms"\s*:\s*"([^"]+)"', raw)
        affected_m = re.search(r'"affected_area_pct"\s*:\s*(\d+)', raw)
        urgency_m = re.search(r'"urgency"\s*:\s*"([^"]+)"', raw)
        confidence_m = re.search(r'"confidence"\s*:\s*"([^"]+)"', raw)
        diseases = []
        if diseases_m:
            diseases = re.findall(r'"([^"]+)"', diseases_m.group(1))
        recs = []
        recs_m = re.search(r'"recommendations"\s*:\s*\[([^\]]+)\]', raw)
        if recs_m:
            recs = re.findall(r'"([^"]+)"', recs_m.group(1))
        return {
            "health_score": int(health_m.group(1)) if health_m else 50,
            "risk_level": risk_m.group(1) if risk_m else "medium",
            "diseases_found": diseases or [],
            "confidence": confidence_m.group(1) if confidence_m else "medium",
            "visible_symptoms": symptoms_m.group(1) if symptoms_m else raw[:300],
            "affected_area_pct": int(affected_m.group(1)) if affected_m else 0,
            "recommendations": recs or ["Apply recommended fungicide"],
            "urgency": urgency_m.group(1) if urgency_m else "within_7_days",
        }
    except Exception:
        return None


def _safe_config():
    """Get config, tolerating import errors."""
    try:
        from .. import get_config, setup_logging
        config = get_config()
        setup_logging(
            log_level=config.get_env().log_level, log_file=config.get_env().log_file
        )
        return config
    except Exception as e:
        logger.warning(f"Config init skipped: {e}")
        return None


def _safe_validation_to_dict(llm_validation) -> dict | None:
    """Safely convert LLMValidation to dict, handling import/None cases."""
    if llm_validation is None:
        return None
    try:
        from ..vision.llm_validator import validation_to_dict
        return validation_to_dict(llm_validation)
    except Exception:
        return None


async def _llava_analyze_background(image_bgr: np.ndarray, img_hash: str, prompt: str | None = None):
    """Run LLaVA analysis in background thread, store result in cache."""
    _LLAVA_PENDING.add(img_hash)
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _llava_analyze_sync, image_bgr, prompt)
        _LLAVA_RESULTS[img_hash] = result
        parsed = result.get("parsed") if result else None
        logger.info(f"Background LLaVA done for {img_hash[:8]}: score={parsed.get('health_score') if parsed else 'N/A'}")
    except Exception as exc:
        logger.warning(f"Background LLaVA failed: {exc}")
        _LLAVA_RESULTS[img_hash] = None
    finally:
        _LLAVA_PENDING.discard(img_hash)


# ── Trained Classifier (india_agri_cls.pt) ──
_CLASSIFIER_MODEL = None
_CLASSIFIER_NAMES = None

# Severity map for classifier predictions
_CLASS_SEVERITY: dict[str, float] = {
    "healthy_wheat": 0.0,
    "healthy_rice": 0.0,
    "wheat_fusarium_head_blight": 0.9,
    "wheat_yellow_rust": 0.85,
    "wheat_black_rust": 0.8,
    "wheat_brown_rust": 0.75,
    "wheat_leaf_blight": 0.7,
    "wheat_powdery_mildew": 0.6,
    "wheat_septoria": 0.7,
    "wheat_tan_spot": 0.6,
    "wheat_smut": 0.65,
    "wheat_root_rot": 0.7,
    "wheat_blast": 0.85,
    "wheat_aphid": 0.55,
    "wheat_mite": 0.5,
    "wheat_stem_fly": 0.5,
    "rice_bacterial_blight": 0.8,
    "rice_brown_spot": 0.6,
    "rice_blast": 0.9,
    "rice_leaf_scald": 0.65,
    "rice_sheath_blight": 0.7,
}

# Human-friendly display names
_CLASS_DISPLAY: dict[str, str] = {
    "healthy_wheat": "Healthy Wheat",
    "healthy_rice": "Healthy Rice",
    "wheat_fusarium_head_blight": "Fusarium Head Blight",
    "wheat_yellow_rust": "Yellow / Stripe Rust",
    "wheat_black_rust": "Black Rust",
    "wheat_brown_rust": "Brown Rust",
    "wheat_leaf_blight": "Wheat Leaf Blight",
    "wheat_powdery_mildew": "Powdery Mildew",
    "wheat_septoria": "Septoria Leaf Blotch",
    "wheat_tan_spot": "Tan Spot",
    "wheat_smut": "Wheat Smut",
    "wheat_root_rot": "Common Root Rot",
    "wheat_blast": "Wheat Blast",
    "wheat_aphid": "Aphid Infestation",
    "wheat_mite": "Mite Damage",
    "wheat_stem_fly": "Stem Fly",
    "rice_bacterial_blight": "Bacterial Leaf Blight",
    "rice_brown_spot": "Rice Brown Spot",
    "rice_blast": "Rice Blast",
    "rice_leaf_scald": "Rice Leaf Scald",
    "rice_sheath_blight": "Sheath Blight",
}


def _get_classifier():
    """Lazy-load the trained crop disease classifier."""
    global _CLASSIFIER_MODEL, _CLASSIFIER_NAMES
    if _CLASSIFIER_MODEL is not None:
        return _CLASSIFIER_MODEL, _CLASSIFIER_NAMES

    from pathlib import Path
    model_path = Path(__file__).resolve().parent.parent.parent.parent / "models" / "india_agri_cls.pt"
    if not model_path.is_file():
        logger.warning(f"Classifier not found at {model_path}")
        return None, None
    try:
        from ultralytics import YOLO
        _CLASSIFIER_MODEL = YOLO(str(model_path), task="classify")
        _CLASSIFIER_NAMES = _CLASSIFIER_MODEL.names  # {0: 'class_name', ...}
        logger.info(f"Loaded crop classifier: {model_path.name} ({len(_CLASSIFIER_NAMES)} classes)")
        return _CLASSIFIER_MODEL, _CLASSIFIER_NAMES
    except Exception as exc:
        logger.warning(f"Failed to load classifier: {exc}")
        return None, None


def _classify_image(image_bgr: np.ndarray) -> dict | None:
    """Run the trained classifier on the image, return top predictions.

    Also stores the raw probs tensor in the result under '_raw_probs' so
    that the crop-type gate (Layer 2) can inspect the full softmax
    distribution without a second forward pass.
    """
    model, names = _get_classifier()
    if model is None:
        return None
    try:
        results = model(image_bgr, verbose=False)
        if not results or results[0].probs is None:
            return None
        probs = results[0].probs

        # Top-5 predictions
        top5_indices = probs.top5
        top5_confs = probs.top5conf.tolist()
        predictions = []
        for idx, conf in zip(top5_indices, top5_confs):
            class_key = names[idx]
            predictions.append({
                "index": idx,
                "class_key": class_key,
                "class_name": _CLASS_DISPLAY.get(class_key, class_key.replace("_", " ").title()),
                "confidence": round(conf, 4),
                "severity": _CLASS_SEVERITY.get(class_key, 0.5),
            })

        top = predictions[0]
        top_is_healthy = "healthy" in top["class_key"].lower()

        # Sum up all disease class probabilities from top-5
        disease_prob = sum(
            p["confidence"] for p in predictions if "healthy" not in p["class_key"].lower()
        )
        healthy_prob = sum(
            p["confidence"] for p in predictions if "healthy" in p["class_key"].lower()
        )

        # Find the leading disease prediction for use below
        top_disease = next((p for p in predictions if "healthy" not in p["class_key"].lower()), None)

        # ── Decision logic (safety-first: false-negative is worse than false-positive) ──
        #
        # Rule 1: "Healthy" must be confident (>= 70%) AND diseases must be low (< 25%)
        #         A barely-50% "healthy" with 47% disease is NOT healthy.
        # Rule 2: If healthy confidence < 70%, always report disease.
        # Rule 3: If combined disease probability > 30%, report disease regardless.

        if top_is_healthy and healthy_prob >= 0.70 and disease_prob < 0.25:
            # High-confidence healthy — genuinely clean field
            health_score = 95
            is_healthy = True
        elif top_is_healthy and healthy_prob >= 0.70 and disease_prob < 0.40:
            # Moderate-confidence healthy with some disease signal
            health_score = max(60, round(95 * (1 - disease_prob)))
            is_healthy = True
        elif top_is_healthy:
            # Classifier says healthy but NOT confident enough — treat as DISEASED
            # This catches the common failure: 50% healthy + 47% disease = NOT healthy
            is_healthy = False
            if top_disease:
                top = top_disease  # Report the top disease instead
                health_score = max(10, round(100 - top["severity"] * 100 * top["confidence"]))
                # Boost severity when disease_prob is close to or exceeds healthy_prob
                if disease_prob > healthy_prob * 0.8:
                    health_score = min(health_score, max(10, round(50 * (1 - disease_prob))))
            else:
                health_score = max(30, round(95 * (1 - disease_prob)))
        else:
            # Top prediction IS a disease — straightforward
            is_healthy = False
            health_score = max(5, round(100 - top["severity"] * 100 * top["confidence"]))

        risk_level = "low" if health_score >= 70 else "medium" if health_score >= 40 else "high" if health_score >= 20 else "critical"

        return {
            "top_prediction": top["class_name"],
            "top_confidence": top["confidence"],
            "health_score": health_score,
            "risk_level": risk_level,
            "is_healthy": is_healthy,
            "disease_probability": round(disease_prob, 4),
            "top5": predictions,
            "model": "india_agri_cls.pt (YOLOv8n-cls, 21 crop diseases)",
            "_raw_probs": probs,  # Layer 2 crop-type gate consumes this
        }
    except Exception as exc:
        logger.warning(f"Classifier inference failed: {exc}")
        return None


def _run_crop_type_gate(classifier_result: dict | None) -> dict | None:
    """Layer 2 — Crop-Type Gate.

    Uses the softmax distribution from the 21-class classifier to determine
    whether the image is wheat, rice, or an unknown/unsupported crop.
    Returns None if the classifier didn't run, otherwise a dict with gate info.
    """
    if classifier_result is None:
        return None
    raw_probs = classifier_result.get("_raw_probs")
    if raw_probs is None:
        return None
    _, names = _get_classifier()
    if names is None:
        return None
    try:
        from ..core.crop_type_gate import classify_crop_type
        gate = classify_crop_type(names, raw_probs)
        return {
            "crop_type": gate.crop_type,
            "accepted": gate.accepted,
            "confidence": gate.confidence,
            "wheat_prob": gate.wheat_prob,
            "rice_prob": gate.rice_prob,
            "entropy": gate.entropy,
            "normalised_entropy": gate.normalised_entropy,
            "top1_class": gate.top1_class,
            "top1_confidence": gate.top1_confidence,
            "cross_group_top5": gate.cross_group_top5,
            "reason": gate.reason,
        }
    except Exception as exc:
        logger.warning(f"Crop-type gate failed (non-fatal): {exc}")
        return None


# ── YOLO Detection Models (bounding boxes) ──
# Supports 3 models: rice_disease.pt, wheat_disease.pt, crop_disease.pt
_DETECTOR_MODELS: dict = {}  # crop_type -> YOLO model

# Model file lookup: crop_type -> list of filenames to try (in priority order)
_DETECTOR_FILES = {
    "rice":     ["rice_disease.pt", "crop_disease.pt", "yolo_crop_disease.pt"],
    "wheat":    ["wheat_disease.pt", "crop_disease.pt", "yolo_crop_disease.pt"],
    "combined": ["crop_disease.pt", "yolo_crop_disease.pt"],
    "auto":     ["crop_disease.pt", "yolo_crop_disease.pt"],
}


def _get_detector(crop_type: str = "auto"):
    """Lazy-load the YOLO detector for a given crop type."""
    global _DETECTOR_MODELS
    crop_key = crop_type.lower().strip()
    if crop_key in _DETECTOR_MODELS:
        return _DETECTOR_MODELS[crop_key]

    from pathlib import Path
    models_dir = Path(__file__).resolve().parent.parent.parent.parent / "models"
    candidates = _DETECTOR_FILES.get(crop_key, _DETECTOR_FILES["auto"])

    for fname in candidates:
        model_path = models_dir / fname
        if model_path.is_file():
            try:
                from ultralytics import YOLO
                model = YOLO(str(model_path))
                _DETECTOR_MODELS[crop_key] = model
                logger.info(f"Loaded YOLO detector for '{crop_key}': {model_path.name} ({len(model.names)} classes)")
                return model
            except Exception as exc:
                logger.warning(f"Failed to load {model_path.name}: {exc}")
                continue

    logger.info(f"No detector model found for crop '{crop_key}' — YOLO detection disabled")
    _DETECTOR_MODELS[crop_key] = None
    return None


def _yolo_detect(image_bgr: np.ndarray, conf: float, draw_boxes: bool, crop_type: str = "auto") -> tuple[list, str | None]:
    """Run YOLO detection and optionally draw bounding boxes on image."""
    model = _get_detector(crop_type)
    if model is None:
        return [], None

    try:
        results = model(image_bgr, conf=conf, verbose=False)
        if not results or len(results[0].boxes) == 0:
            return [], None

        boxes = results[0].boxes
        detections = []
        for box in boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            name = results[0].names.get(cls_id, f"class_{cls_id}")
            detections.append({
                "class_name": name,
                "confidence": round(confidence, 4),
                "x1": int(x1), "y1": int(y1),
                "x2": int(x2), "y2": int(y2),
            })

        # Draw annotated image with boxes
        annotated_b64 = None
        if draw_boxes and detections:
            annotated = results[0].plot()  # YOLO built-in box drawing
            success, encoded = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if success:
                annotated_b64 = f"data:image/jpeg;base64,{base64.b64encode(encoded).decode('utf-8')}"

        logger.info(f"YOLO detected {len(detections)} disease region(s)")
        return detections, annotated_b64
    except Exception as exc:
        logger.warning(f"YOLO detection failed: {exc}")
        return [], None


# Known severe diseases — if ANY model detects these, lower the score
_SEVERE_DISEASES = {
    "fusarium", "fhb", "scab", "head blight",
    "blast", "stem rust", "black rust",
    "yellow rust", "stripe rust",
    "bacterial leaf blight", "brown planthopper",
}


def _has_severe_disease(llava_result: dict | None, cls_result: dict | None) -> bool:
    """Check if any model detected a known severe disease."""
    texts = []
    if llava_result:
        for d in llava_result.get("diseases_found", []):
            texts.append(d.lower())
    if cls_result:
        texts.append(cls_result.get("top_prediction", "").lower())
        # Also check all top-5 predictions (classifier may rank disease #2 or #3)
        for p in cls_result.get("top5", []):
            if p.get("confidence", 0) > 0.10:  # Only if > 10% confidence
                texts.append(p.get("class_name", "").lower())
                texts.append(p.get("class_key", "").lower())
    combined = " ".join(texts)
    return any(kw in combined for kw in _SEVERE_DISEASES)


def _compute_ensemble(llava_result: dict | None, cls_result: dict | None) -> dict:
    """Combine LLaVA + Classifier into a conservative ensemble verdict.

    Safety-first principle: when models disagree, trust the MORE pessimistic
    score. In agriculture, a false-negative (missing a disease) is far worse
    than a false-positive (spraying a healthy crop).
    """
    if llava_result and cls_result:
        llava_score = llava_result.get("health_score", 50)
        cls_score = cls_result.get("health_score", 50)
        llava_healthy = llava_score >= 70
        cls_healthy = cls_result.get("is_healthy", False)
        both_agree = llava_healthy == cls_healthy
        severe = _has_severe_disease(llava_result, cls_result)

        # Measure classifier uncertainty: high disease_probability with healthy
        # top pick means the classifier is not confident
        cls_disease_prob = cls_result.get("disease_probability", 0)
        cls_uncertain = cls_healthy and cls_disease_prob > 0.3

        if both_agree:
            # Models agree: weighted average (LLaVA 60%, Cls 40%)
            ensemble_health = round(llava_score * 0.6 + cls_score * 0.4)
            agreement = "high"
            note = "Models agree — high reliability"
        elif cls_uncertain:
            # Classifier is uncertain (barely "healthy") — trust LLaVA heavily
            ensemble_health = round(llava_score * 0.85 + cls_score * 0.15)
            agreement = "low"
            note = "Classifier uncertain — LLaVA visual analysis weighted higher"
        else:
            # Models DISAGREE clearly: use the LOWER score (safety-first)
            ensemble_health = min(llava_score, cls_score)
            agreement = "low"
            note = "Models disagree — using conservative (lower) score for safety"

        # Severe disease penalty: cap health at 55 if a devastating disease is found
        if severe and ensemble_health > 55:
            ensemble_health = min(ensemble_health, 55)
            note += " | Severe disease detected — score capped for safety"

        ensemble_risk = (
            "low" if ensemble_health >= 70
            else "medium" if ensemble_health >= 40
            else "high" if ensemble_health >= 20
            else "critical"
        )

        return {
            "ensemble_health_score": ensemble_health,
            "ensemble_risk_level": ensemble_risk,
            "model_agreement": agreement,
            "note": note,
            "models_used": ["LLaVA (Vision LLM)", "YOLOv8n-cls (Trained Classifier)"],
        }
    elif llava_result:
        score = llava_result.get("health_score", 50)
        severe = _has_severe_disease(llava_result, None)
        if severe and score > 55:
            score = min(score, 55)
        return {
            "ensemble_health_score": score,
            "ensemble_risk_level": "low" if score >= 70 else "medium" if score >= 40 else "high" if score >= 20 else "critical",
            "model_agreement": "single_model",
            "note": "Only LLaVA available" + (" | Severe disease detected" if severe else ""),
            "models_used": ["LLaVA (Vision LLM)"],
        }
    elif cls_result:
        score = cls_result.get("health_score", 50)
        severe = _has_severe_disease(None, cls_result)
        if severe and score > 45:
            score = min(score, 45)
        # When classifier is uncertain (low confidence healthy), indicate it
        disease_prob = cls_result.get("disease_probability", 0)
        uncertain = cls_result.get("is_healthy", False) and disease_prob > 0.30
        note = "Only classifier available"
        if severe:
            note += " | Severe disease detected in top predictions — score capped for safety"
        if uncertain:
            note += " | Classifier uncertain — recommend LLaVA for confirmation"
        return {
            "ensemble_health_score": score,
            "ensemble_risk_level": "low" if score >= 70 else "medium" if score >= 40 else "high" if score >= 20 else "critical",
            "model_agreement": "single_model",
            "note": note,
            "models_used": ["YOLOv8n-cls (Trained Classifier)"],
        }
    else:
        return {
            "ensemble_health_score": 50,
            "ensemble_risk_level": "medium",
            "model_agreement": "none",
            "note": "No AI models available",
            "models_used": [],
        }


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application.

    Returns:
        Configured FastAPI app
    """
    # Initialize config and logging (tolerating missing deps)
    config = _safe_config()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Preload models at startup for fast first-request response."""
        logger.info("Preloading models at startup...")
        t0 = time.time()
        _get_classifier()
        _get_detector("auto")
        logger.info(f"Models preloaded in {(time.time() - t0) * 1000:.0f}ms")
        yield

    app = FastAPI(
        title="AgriDrone API",
        description="Research prototype for site-specific crop protection",
        version=__version__,
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers (lazy import to avoid cascading failures)
    try:
        from .routes import analysis, chat, detection, field, reports, stream, universal, voice
        app.include_router(detection.router, prefix="/api")
        app.include_router(analysis.router, prefix="/api")
        app.include_router(stream.router, prefix="/api")
        app.include_router(reports.router, prefix="/api")
        app.include_router(chat.router, prefix="/api")
        app.include_router(field.router, prefix="/api")
        app.include_router(voice.router, prefix="/api")
        app.include_router(universal.router, prefix="/api")
    except Exception as e:
        logger.warning(f"Some API routers could not be loaded: {e}")

    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "status": "ok",
            "app": "agridrone",
            "version": __version__,
        }

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        env = config.get_env() if config else None
        return {
            "status": "ok",
            "dry_run": getattr(env, "dry_run", True),
            "test_fluid_only": getattr(env, "safe_test_fluid_only", True),
        }

    @app.get("/system")
    async def system_info():
        """Get system information."""
        env = config.get_env() if config else None
        return {
            "version": __version__,
            "dry_run": getattr(env, "dry_run", True),
            "test_fluid_only": getattr(env, "safe_test_fluid_only", True),
            "device": getattr(env, "device", "cpu"),
        }

    @app.get("/config")
    async def get_configuration():
        """Get current configuration."""
        if config:
            return config.get_env().model_dump()
        return {"status": "config not loaded"}

    # ============================================================
    # Detection History Storage
    # ============================================================
    _HISTORY_DIR = Path(__file__).resolve().parent.parent.parent.parent / "outputs" / "history"
    _HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    _HISTORY_FILE = _HISTORY_DIR / "detection_history.json"
    _ACTIVITY_LOG = _HISTORY_DIR / "activity.json"

    def _load_history():
        if _HISTORY_FILE.is_file():
            try:
                return json.loads(_HISTORY_FILE.read_text(encoding="utf-8"))
            except Exception:
                return []
        return []

    def _save_history(history):
        _HISTORY_FILE.write_text(json.dumps(history, default=str, indent=2), encoding="utf-8")

    def _log_activity(action: str, detail: str = ""):
        activities = []
        if _ACTIVITY_LOG.is_file():
            try:
                activities = json.loads(_ACTIVITY_LOG.read_text(encoding="utf-8"))
            except Exception:
                activities = []
        activities.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "detail": detail,
        })
        # Keep last 200 entries
        activities = activities[-200:]
        _ACTIVITY_LOG.write_text(json.dumps(activities, default=str, indent=2), encoding="utf-8")

    # ============================================================
    # Dashboard API Endpoints (simplified interface)
    # ============================================================

    @app.post("/detect")
    async def detect_image_simple(
        file: UploadFile = File(...),
        use_mock: bool = Form(False),
        use_llava: bool = Form(False),
        confidence_threshold: float = Form(0.3),
        crop_type: str = Form("wheat"),
        include_image: bool = Form(True),
        area_acres: float = Form(1.0),
        growth_stage: str = Form("unknown"),
    ):
        """
        Detection endpoint for dashboard with LLaVA vision analysis.
        """
        try:
            if not file.filename:
                raise HTTPException(status_code=400, detail="No filename provided")

            contents = await file.read()
            if not contents:
                raise HTTPException(status_code=400, detail="Empty file")

            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                raise HTTPException(status_code=400, detail="Failed to decode image")

            # Reject images too small for meaningful classification
            size_reject = _check_minimum_size(image)
            if size_reject:
                logger.warning(f"REJECTED tiny image: {file.filename} shape={image.shape}")
                return size_reject

            # Downsize large images for faster inference
            image = _resize_if_large(image)
            img_hash = _image_hash(image)

            logger.info(f"Dashboard detection: {file.filename} shape={image.shape} (llava={use_llava}, crop={crop_type})")

            # Convert image to base64 for response
            image_b64_dataurl = None
            if include_image:
                success, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if success:
                    image_base64 = base64.b64encode(encoded).decode('utf-8')
                    image_b64_dataurl = f"data:image/jpeg;base64,{image_base64}"

            start_time = time.time()

            # ── GATEKEEPER: Is this a plant image? ──
            plant_check = _is_plant_image(image)
            if not plant_check["is_plant"]:
                processing_time = (time.time() - start_time) * 1000
                logger.warning(
                    f"REJECTED non-plant image: {file.filename} "
                    f"(skin={plant_check['skin_ratio']:.1%}, green={plant_check['green_ratio']:.1%})"
                )
                return {
                    "rejected": True,
                    "is_plant": False,
                    "rejection_reason": plant_check["reason"],
                    "green_ratio": plant_check.get("green_ratio", 0),
                    "skin_ratio": plant_check.get("skin_ratio", 0),
                    "face_count": plant_check.get("face_count", 0),
                    "face_area_pct": plant_check.get("face_area_pct", 0),
                    "vegetation_ratio": plant_check.get("green_ratio", 0),
                    "spectral": {
                        "vari": plant_check.get("vari"),
                        "gli": plant_check.get("gli"),
                        "exg": plant_check.get("exg"),
                        "ngrdi": plant_check.get("ngrdi"),
                    },
                    "classifier_top_confidence": 0,
                    "image": image_b64_dataurl,
                    "filename": file.filename,
                    "processing_time_ms": processing_time,
                    "structured": None,
                    "detections": [],
                    "ensemble": {
                        "ensemble_health_score": 0,
                        "ensemble_risk_level": "none",
                        "model_agreement": "rejected",
                        "note": "Image rejected — does not appear to be a plant/crop",
                        "models_used": ["Plant Gatekeeper", "Spectral Verification"],
                    },
                }

            # ── Model 1: Trained Classifier (21 crop disease classes) ── FAST ~100ms
            classifier_result = _classify_image(image)

            # ── Layer 2: Crop-Type Gate ──
            # Uses the classifier's full softmax distribution to determine
            # wheat vs rice vs unknown. Rejects OOD crops (corn, tomato, etc.)
            # that passed the physics gate but aren't wheat or rice.
            crop_gate_result = _run_crop_type_gate(classifier_result)
            if crop_gate_result and not crop_gate_result["accepted"]:
                processing_time = (time.time() - start_time) * 1000
                logger.warning(
                    f"Layer-2 REJECTED: {file.filename} — {crop_gate_result['reason']}"
                )
                return {
                    "rejected": True,
                    "is_plant": True,
                    "rejection_layer": 2,
                    "rejection_reason": crop_gate_result["reason"],
                    "crop_gate": crop_gate_result,
                    "classifier_top_prediction": classifier_result.get("top_prediction", "") if classifier_result else "",
                    "classifier_top_confidence": classifier_result.get("top_confidence", 0) if classifier_result else 0,
                    "image": image_b64_dataurl,
                    "filename": file.filename,
                    "processing_time_ms": processing_time,
                    "structured": None,
                    "detections": [],
                    "ensemble": {
                        "ensemble_health_score": 0,
                        "ensemble_risk_level": "none",
                        "model_agreement": "rejected",
                        "note": f"Image rejected — {crop_gate_result['reason']}",
                        "models_used": ["Layer-1 Physics Gate", "Layer-2 Crop-Type Gate"],
                    },
                }
            # Override crop_type from form with auto-detected crop type
            if crop_gate_result and crop_gate_result["accepted"]:
                detected_crop = crop_gate_result["crop_type"]
                if detected_crop in ("wheat", "rice") and crop_type == "wheat":
                    # Auto-correct: user may have left default "wheat" but image is rice
                    crop_type = detected_crop
                    logger.info(f"Layer-2 auto-detected crop_type={detected_crop}")

            # Strip internal _raw_probs from classifier_result before serialisation
            if classifier_result and "_raw_probs" in classifier_result:
                del classifier_result["_raw_probs"]

            # ── Model 1b: Uncertainty Quantification (MC-Dropout) ──
            uncertainty_data = None
            try:
                from ..core.detector import predict_with_uncertainty, flag_uncertain_case, uncertainty_to_dict
                cls_model, cls_names = _get_classifier()
                if cls_model is not None:
                    uq = predict_with_uncertainty(
                        cls_model, image,
                        n_forward=20,
                        class_display=_CLASS_DISPLAY,
                        class_severity=_CLASS_SEVERITY,
                    )
                    uncertainty_data = uncertainty_to_dict(uq)
                    logger.info(
                        f"Uncertainty: μ={uq.mean_confidence:.2f} σ={uq.std_confidence:.3f} "
                        f"consistent={uq.prediction_consistency:.0%} uncertain={uq.is_uncertain}"
                    )
                    if uq.is_uncertain:
                        flag_uncertain_case(uq, img_hash, classifier_result, file.filename)
            except Exception as uq_exc:
                logger.warning(f"Uncertainty quantification failed (non-fatal): {uq_exc}")

            # ── Model 2: YOLO detection (bounding boxes around diseases) ── FAST ~200ms
            detections = []
            annotated_b64 = None
            if not use_mock:
                detections, annotated_b64 = _yolo_detect(image, confidence_threshold, include_image, crop_type)

            # ── Model 3: Symptom Reasoning Engine ── FAST ~50ms
            # Runs BEFORE LLaVA so we can build a validation prompt
            reasoning_result = None
            pipeline_output = None
            try:
                from ..vision.disease_reasoning import run_full_pipeline, diagnosis_to_dict
                pipeline_output = run_full_pipeline(image, classifier_result, crop_type)
                diagnosis = pipeline_output.diagnosis
                reasoning_result = diagnosis_to_dict(diagnosis)
                logger.info(
                    f"Reasoning engine: {diagnosis.disease_name} "
                    f"(conf={diagnosis.confidence:.2f}, health={diagnosis.health_score})"
                )
            except Exception as re_exc:
                logger.warning(f"Reasoning engine failed: {re_exc}")

            # ── Model 4: LLaVA Vision LLM (as VALIDATOR, not predictor) ──
            # Build a structured validation prompt using rule engine results
            llava_result = None
            llm_validation = None
            validation_prompt = None
            if pipeline_output:
                try:
                    from ..vision.llm_validator import (
                        build_validation_prompt,
                        parse_validation_response,
                        validation_to_dict,
                    )
                    validation_prompt, scenario = build_validation_prompt(
                        pipeline_output.rule_result,
                        pipeline_output.features,
                        classifier_result,
                        crop_type,
                    )
                    logger.info(f"LLM validation scenario: {scenario}")
                except Exception as vp_exc:
                    logger.warning(f"Failed to build validation prompt: {vp_exc}")

            if use_llava:
                # Check cache first
                if img_hash in _LLAVA_RESULTS:
                    llava_result = _LLAVA_RESULTS[img_hash]
                    logger.info(f"LLaVA cache hit for {img_hash[:8]}")
                else:
                    llava_result = _llava_analyze_sync(image, validation_prompt)
                    _LLAVA_RESULTS[img_hash] = llava_result
            else:
                # Fire background LLaVA with validation prompt
                if img_hash not in _LLAVA_RESULTS and img_hash not in _LLAVA_PENDING:
                    _LLAVA_PENDING.add(img_hash)  # Mark pending BEFORE task starts (avoid race)
                    # Store context needed for parsing validation response later
                    _LLAVA_CONTEXT[img_hash] = {
                        "scenario": scenario if pipeline_output else None,
                        "our_diagnosis": pipeline_output.diagnosis.disease_name if pipeline_output else None,
                    }
                    asyncio.create_task(_llava_analyze_background(image.copy(), img_hash, validation_prompt))

            # ── Parse LLM validation response ──
            if llava_result and pipeline_output and validation_prompt:
                try:
                    from ..vision.llm_validator import (
                        parse_validation_response,
                        validation_to_dict,
                        fuse_confidence,
                    )
                    raw_text = llava_result.get("raw", "")
                    our_diagnosis = pipeline_output.diagnosis.disease_name
                    llm_validation = parse_validation_response(raw_text, scenario, our_diagnosis)
                    logger.info(
                        f"LLM validation: agrees={llm_validation.agrees}, "
                        f"score={llm_validation.agreement_score:.2f}, "
                        f"llm_says={llm_validation.llm_diagnosis}"
                    )
                except Exception as lv_exc:
                    logger.warning(f"LLM validation parsing failed: {lv_exc}")

            # ── Ensemble: combine all model opinions ──
            # Extract parsed LLaVA result for backward-compatible ensemble
            llava_parsed = llava_result.get("parsed") if llava_result else None
            ensemble = _compute_ensemble(llava_parsed, classifier_result)

            # Rule engine is used as a MONITOR, not an override.
            # The ablation study proved it degrades accuracy from 96.2% → 60.2%.
            # We still run it to provide reasoning chains and differentials in the UI,
            # but we do NOT let it override the classifier's health score.
            if reasoning_result:
                if "Symptom Reasoning" not in ensemble.get("models_used", []):
                    ensemble.setdefault("models_used", []).append("Symptom Reasoning Engine (monitor)")
                # Log disagreement but don't override
                if reasoning_result.get("health_score", 100) < ensemble.get("ensemble_health_score", 100):
                    logger.info(
                        f"Rule engine disagrees (health={reasoning_result['health_score']} vs "
                        f"ensemble={ensemble['ensemble_health_score']}) — logged but NOT overriding"
                    )
                    ensemble["rule_engine_disagreement"] = {
                        "rule_health_score": reasoning_result["health_score"],
                        "rule_disease": reasoning_result.get("disease_name", ""),
                        "rule_risk_level": reasoning_result.get("risk_level", ""),
                    }

            # ── Apply LLM validation to ensemble ──
            if llm_validation:
                try:
                    from ..vision.llm_validator import fuse_confidence, validation_to_dict
                    rule_conf = pipeline_output.diagnosis.confidence if pipeline_output else 0.5
                    cls_conf = classifier_result.get("top_confidence", 0.5) if classifier_result else 0.5
                    fusion = fuse_confidence(rule_conf, llm_validation, cls_conf)
                    ensemble["llm_validation"] = validation_to_dict(llm_validation)
                    ensemble["confidence_fusion"] = fusion

                    # If LLM disagrees and has lower health score, apply safety-first
                    if not llm_validation.agrees and llm_validation.health_score < ensemble["ensemble_health_score"]:
                        ensemble["ensemble_health_score"] = llm_validation.health_score
                        ensemble["ensemble_risk_level"] = llm_validation.risk_level
                        ensemble["note"] = (ensemble.get("note", "") + " | LLM validator override (safety-first)").strip(" | ")

                    if "LLM Validator" not in ensemble.get("models_used", []):
                        ensemble.setdefault("models_used", []).append("LLM Validator (LLaVA)")
                except Exception as fuse_exc:
                    logger.warning(f"Confidence fusion failed: {fuse_exc}")

            processing_time = (time.time() - start_time) * 1000
            active_models = sum([
                llava_result is not None,
                classifier_result is not None,
                len(detections) > 0,
                reasoning_result is not None,
            ])
            logger.info(
                f"Ensemble result: health={ensemble['ensemble_health_score']}, "
                f"agreement={ensemble['model_agreement']}, "
                f"models_active={active_models}/4, "
                f"time={processing_time:.0f}ms"
            )

            # ── D2: Build structured output (single clean object for frontend) ──
            structured = None
            try:
                from .structured_output import build_structured_output

                # ── F1: Grad-CAM heatmap ──
                gradcam_data = None
                try:
                    from ..vision.gradcam import generate_gradcam_response
                    cls_model, cls_names = _get_classifier()
                    target_idx = None
                    if classifier_result and classifier_result.get("top5"):
                        target_idx = classifier_result["top5"][0].get("index")
                    if target_idx is not None:
                        gradcam_data = generate_gradcam_response(cls_model, image, target_idx)
                        logger.info(f"Grad-CAM generated: coverage={gradcam_data.get('cam_coverage', 0):.1%}")
                except Exception as gc_exc:
                    logger.warning(f"Grad-CAM failed (non-critical): {gc_exc}")

                # ── F2: RAG research papers ──
                research_papers = None
                try:
                    from ..knowledge.research_rag import retrieve_for_diagnosis
                    diag_key = reasoning_result.get("disease_key", "") if reasoning_result else ""
                    diag_name = reasoning_result.get("disease_name", "") if reasoning_result else ""
                    if diag_key and not diag_key.startswith("healthy"):
                        evidence_list = reasoning_result.get("evidence", []) if reasoning_result else []
                        research_papers = retrieve_for_diagnosis(
                            disease_key=diag_key,
                            disease_name=diag_name,
                            evidence=evidence_list,
                        )
                        logger.info(f"RAG retrieved {len(research_papers)} papers for {diag_key}")
                except Exception as rag_exc:
                    logger.warning(f"RAG retrieval failed (non-critical): {rag_exc}")

                # ── F3: Enhanced ensemble voting ──
                ensemble_voting_data = None
                try:
                    from ..vision.ensemble_voter import ensemble_vote, ensemble_to_dict
                    vote_result = ensemble_vote(
                        classifier_result=classifier_result,
                        reasoning_result=reasoning_result,
                        llm_validation=llm_validation,
                        crop_type=crop_type,
                    )
                    ensemble_voting_data = ensemble_to_dict(vote_result)
                    logger.info(
                        f"Ensemble voting: {vote_result.final_disease} "
                        f"(agreement={vote_result.agreement_level}, method={vote_result.voting_method})"
                    )
                except Exception as ev_exc:
                    logger.warning(f"Ensemble voting failed (non-critical): {ev_exc}")

                # ── F4: Temporal tracking ──
                temporal_data = None
                try:
                    from ..feedback.temporal_tracker import get_temporal_context
                    t_disease = reasoning_result.get("disease_name", "Unknown") if reasoning_result else "Unknown"
                    t_health = ensemble.get("ensemble_health_score", 50)
                    t_conf = reasoning_result.get("confidence", 0.5) if reasoning_result else 0.5
                    temporal_data = get_temporal_context(
                        current_filename=file.filename,
                        current_disease=t_disease,
                        current_health=t_health,
                        current_confidence=t_conf,
                        crop_type=crop_type,
                    )
                    logger.info(f"Temporal: trend={temporal_data.get('trend')}, prev_scans={temporal_data.get('num_previous_scans', 0)}")
                except Exception as tt_exc:
                    logger.warning(f"Temporal tracking failed (non-critical): {tt_exc}")

                llm_val_dict = _safe_validation_to_dict(llm_validation)
                fusion_dict = ensemble.get("confidence_fusion") if ensemble else None
                structured = build_structured_output(
                    classifier_result=classifier_result,
                    reasoning_result=reasoning_result,
                    llm_validation_dict=llm_val_dict,
                    confidence_fusion=fusion_dict,
                    ensemble=ensemble,
                    processing_time_ms=processing_time,
                    gradcam_data=gradcam_data,
                    research_papers=research_papers,
                    ensemble_voting=ensemble_voting_data,
                    temporal_data=temporal_data,
                )
            except Exception as so_exc:
                logger.warning(f"Structured output build failed: {so_exc}")

            # ── Save to detection history ──
            disease_name = "Healthy"
            confidence_val = 0
            if structured:
                disease_name = structured["diagnosis"].get("disease_name", "Healthy")
                confidence_val = round(structured["diagnosis"].get("confidence", 0) * 100)
            elif reasoning_result and reasoning_result.get("disease_key") != "healthy":
                disease_name = reasoning_result.get("disease_name", "Unknown")
                confidence_val = round(reasoning_result.get("confidence", 0) * 100)
            elif classifier_result:
                disease_name = classifier_result.get("top_prediction", "Unknown")
                confidence_val = round(classifier_result.get("top_confidence", 0) * 100)
            elif detections:
                disease_name = detections[0].get("class_name", "Unknown")
                confidence_val = round(detections[0].get("confidence", 0) * 100)

            history_entry = {
                "id": int(time.time() * 1000),
                "filename": file.filename,
                "timestamp": datetime.now().isoformat(),
                "disease": disease_name,
                "confidence": confidence_val,
                "health_score": ensemble.get("ensemble_health_score", 50),
                "risk_level": ensemble.get("ensemble_risk_level", "medium"),
                "num_detections": len(detections),
                "crop_type": crop_type,
            }
            try:
                history = _load_history()
                history.append(history_entry)
                _save_history(history)
            except Exception as he:
                logger.warning(f"Failed to save history: {he}")

            _log_activity("upload", f"Image uploaded: {file.filename}")
            _log_activity("detection", f"Disease detected: {disease_name} ({confidence_val}%)")
            _log_activity("scan_complete", f"Scan completed — Health: {ensemble.get('ensemble_health_score', 50)}%")

            # ── Yield & cost estimator ──
            yield_estimate = None
            try:
                from ..core.yield_estimator import get_estimator, estimate_to_dict
                disease_key = (
                    reasoning_result.get("disease_key", "") if reasoning_result
                    else (classifier_result.get("top_prediction", "") if classifier_result else "")
                )
                health = ensemble.get("ensemble_health_score", 50)
                if health >= 70:
                    sev = "mild"
                elif health >= 40:
                    sev = "moderate"
                else:
                    sev = "severe"
                if disease_key and not disease_key.startswith("healthy"):
                    est = get_estimator().estimate(
                        disease=disease_key,
                        severity=sev,
                        crop=crop_type,
                        area_acres=max(0.01, area_acres),
                        stage=growth_stage,
                    )
                    yield_estimate = estimate_to_dict(est)
                    logger.info(
                        f"Yield estimate: loss={yield_estimate['yield_loss_percent']}%, "
                        f"revenue_loss=₹{yield_estimate['revenue_loss_inr']:.0f}, "
                        f"rec={yield_estimate['recommendation']}"
                    )
            except Exception as ye_exc:
                logger.warning(f"Yield estimator failed (non-critical): {ye_exc}")

            return {
                "rejected": False,
                "is_plant": True,
                "low_confidence": (
                    classifier_result is not None
                    and classifier_result.get("top_confidence", 1.0) < 0.40
                ),
                "defense_layers": {
                    "layer_1_physics_gate": "pass",
                    "layer_2_crop_type_gate": crop_gate_result.get("crop_type", "unknown") if crop_gate_result else "skipped",
                    "layer_2_confidence": crop_gate_result.get("confidence", 0) if crop_gate_result else 0,
                    "layer_3_disease_classifier": classifier_result.get("top_prediction", "") if classifier_result else "skipped",
                },
                "crop_gate": crop_gate_result,
                "structured": structured,
                "detections": detections,
                "image": image_b64_dataurl,
                "annotated_image": annotated_b64,
                "processing_time_ms": processing_time,
                "num_detections": len(detections),
                "filename": file.filename,
                "llava_analysis": llava_result.get("parsed") if llava_result else None,
                "llm_validation": _safe_validation_to_dict(llm_validation),
                "classifier_result": classifier_result,
                "reasoning": reasoning_result,
                "ensemble": ensemble,
                "uncertainty": uncertainty_data,
                "yield_estimate": yield_estimate,
                "llava_pending": img_hash in _LLAVA_PENDING,
                "llava_hash": img_hash,
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Error in detect endpoint: {e}")
            raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

    @app.get("/detect/llava-status/{img_hash}")
    async def get_llava_status(img_hash: str):
        """Poll for background LLaVA analysis result."""
        if img_hash in _LLAVA_RESULTS:
            result = _LLAVA_RESULTS[img_hash]
            parsed = result.get("parsed") if result else None

            # Also parse LLM validation if context is available
            llm_val_dict = None
            ctx = _LLAVA_CONTEXT.get(img_hash)
            if result and ctx and ctx.get("scenario") and ctx.get("our_diagnosis"):
                try:
                    from ..vision.llm_validator import parse_validation_response, validation_to_dict
                    raw_text = result.get("raw", "")
                    validation = parse_validation_response(
                        raw_text, ctx["scenario"], ctx["our_diagnosis"]
                    )
                    llm_val_dict = validation_to_dict(validation)
                except Exception:
                    pass

            return {
                "status": "complete",
                "llava_analysis": parsed,
                "llm_validation": llm_val_dict,
            }
        elif img_hash in _LLAVA_PENDING:
            return {"status": "pending"}
        else:
            return {"status": "not_found"}

    # ============================================================
    # Detection History / Reports
    # ============================================================
    @app.get("/api/reports/history")
    async def get_detection_history():
        """Get stored detection history for the reports page."""
        history = _load_history()
        return {"history": history, "total": len(history)}

    @app.delete("/api/reports/history")
    async def clear_detection_history():
        """Clear detection history."""
        _save_history([])
        return {"status": "cleared"}

    # ============================================================
    # Feedback Loop (E1 — agronomist corrections)
    # ============================================================
    from ..feedback.feedback_store import (
        init_db as _init_feedback_db,
        save_feedback as _save_feedback,
        get_all_feedback as _get_all_feedback,
        get_feedback_by_id as _get_feedback_by_id,
        get_feedback_count as _get_feedback_count,
        delete_feedback as _delete_feedback,
        FeedbackRecord,
    )
    from ..feedback.correction_aggregator import generate_full_report as _generate_feedback_report
    from ..feedback.kb_updater import (
        run_full_update as _run_kb_update,
        list_backups as _list_kb_backups,
        restore_backup as _restore_kb_backup,
    )

    # Initialize feedback DB on startup
    _init_feedback_db()

    @app.post("/api/feedback")
    async def submit_feedback(payload: dict):
        """Submit agronomist correction for a detection result.

        Expected body:
        {
          "detection_id": 1712681234567,        // from history entry id
          "image_hash": "abc123...",             // from llava_hash
          "filename": "field_001.jpg",
          "predicted_disease": "wheat_yellow_rust",
          "predicted_confidence": 0.87,
          "correct_disease": "wheat_tan_spot",   // ground-truth label
          "severity_rating": 4,                  // 1-5 (optional)
          "notes": "Circular lesions visible",   // optional
          "classifier_prediction": "wheat_yellow_rust",
          "rule_engine_prediction": "wheat_yellow_rust",
          "llm_prediction": "wheat_tan_spot",
          "crop_type": "wheat",
          "image_data": "base64..."              // optional — original image for retrain
        }
        """
        correct_disease = payload.get("correct_disease", "").strip()
        if not correct_disease:
            raise HTTPException(status_code=400, detail="correct_disease is required")

        predicted = payload.get("predicted_disease", "").strip()
        if not predicted:
            raise HTTPException(status_code=400, detail="predicted_disease is required")

        record = FeedbackRecord(
            detection_id=payload.get("detection_id"),
            image_hash=payload.get("image_hash", ""),
            filename=payload.get("filename", ""),
            predicted_disease=predicted,
            predicted_confidence=payload.get("predicted_confidence", 0),
            correct_disease=correct_disease,
            severity_rating=payload.get("severity_rating"),
            notes=payload.get("notes", ""),
            classifier_prediction=payload.get("classifier_prediction", ""),
            rule_engine_prediction=payload.get("rule_engine_prediction", ""),
            llm_prediction=payload.get("llm_prediction", ""),
            crop_type=payload.get("crop_type", ""),
        )

        # Decode optional image data
        image_bytes = None
        image_b64 = payload.get("image_data", "")
        if image_b64:
            try:
                # Strip data URI prefix if present
                if "," in image_b64:
                    image_b64 = image_b64.split(",", 1)[1]
                image_bytes = base64.b64decode(image_b64)
            except Exception:
                logger.warning("Failed to decode feedback image data")

        fb_id = _save_feedback(record, image_bytes)
        _log_activity("feedback", f"Agronomist correction: {predicted} → {correct_disease}")

        return {
            "status": "saved",
            "feedback_id": fb_id,
            "predicted": predicted,
            "corrected": correct_disease,
            "is_correction": predicted != correct_disease,
        }

    @app.get("/api/feedback")
    async def list_feedback(
        limit: int = Query(50, ge=1, le=1000),
        offset: int = Query(0, ge=0),
        disease: Optional[str] = Query(None),
    ):
        """List submitted feedback records."""
        records = _get_all_feedback(limit=limit, offset=offset, disease_filter=disease)
        total = _get_feedback_count()
        return {"feedback": records, "total": total, "limit": limit, "offset": offset}

    @app.get("/api/feedback/{fb_id}")
    async def get_single_feedback(fb_id: int):
        """Get a single feedback record."""
        record = _get_feedback_by_id(fb_id)
        if not record:
            raise HTTPException(status_code=404, detail="Feedback not found")
        return record

    @app.delete("/api/feedback/{fb_id}")
    async def remove_feedback(fb_id: int):
        """Delete a feedback record."""
        deleted = _delete_feedback(fb_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Feedback not found")
        return {"status": "deleted", "id": fb_id}

    @app.get("/api/feedback/analysis/report")
    async def get_feedback_analysis():
        """Get full feedback analysis report (confusion matrix, accuracy, recommendations)."""
        return _generate_feedback_report()

    @app.post("/api/feedback/kb-update")
    async def trigger_kb_update(payload: dict = None):
        """Run KB weight updater based on accumulated feedback.

        Body (optional): {"dry_run": true}
        """
        payload = payload or {}
        dry_run = payload.get("dry_run", True)  # Default to dry run for safety
        return _run_kb_update(dry_run=dry_run)

    @app.get("/api/feedback/kb-backups")
    async def get_kb_backups():
        """List available KB backups."""
        return {"backups": _list_kb_backups()}

    @app.post("/api/feedback/kb-restore")
    async def restore_kb(payload: dict):
        """Restore a KB backup.

        Body: {"filename": "diseases_20260409_123456.json"}
        """
        filename = payload.get("filename", "")
        if not filename:
            raise HTTPException(status_code=400, detail="filename is required")
        # Validate filename format to prevent path traversal
        if ".." in filename or "/" in filename or "\\" in filename:
            raise HTTPException(status_code=400, detail="Invalid filename")
        success = _restore_kb_backup(filename)
        if not success:
            raise HTTPException(status_code=404, detail="Backup not found")
        return {"status": "restored", "filename": filename}

    # ============================================================
    # Activity Feed
    # ============================================================
    @app.get("/api/activity/feed")
    async def get_activity_feed():
        """Get activity log for the activity feed."""
        if _ACTIVITY_LOG.is_file():
            try:
                activities = json.loads(_ACTIVITY_LOG.read_text(encoding="utf-8"))
                return {"feed": activities[-50:]}  # Last 50
            except Exception:
                pass
        return {"feed": []}

    # ============================================================
    # ML Dashboard Endpoints
    # ============================================================
    @app.get("/api/ml/metrics")
    async def get_ml_metrics():
        """Read training metrics from results.csv."""
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        # Search for results.csv in common training output dirs
        search_paths = [
            project_root / "outputs" / "training" / "yolo_crop_disease",
            project_root / "outputs" / "training" / "rice_v2",
            project_root / "outputs" / "training" / "india_agri_v1",
            project_root / "outputs" / "training" / "rice_disease_gpu",
            project_root / "outputs" / "training",
            project_root / "runs" / "detect" / "train",
            project_root / "runs",
        ]
        for base in search_paths:
            csv_path = base / "results.csv"
            if csv_path.is_file():
                try:
                    rows = []
                    with open(csv_path, "r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            cleaned = {k.strip(): v.strip() for k, v in row.items()}
                            rows.append(cleaned)
                    if rows:
                        last = rows[-1]
                        metrics = {
                            "mAP50": float(last.get("metrics/mAP50(B)", last.get("mAP50", 0))),
                            "mAP50_95": float(last.get("metrics/mAP50-95(B)", last.get("mAP50-95", 0))),
                            "precision": float(last.get("metrics/precision(B)", last.get("precision", 0))),
                            "recall": float(last.get("metrics/recall(B)", last.get("recall", 0))),
                            "epochs": len(rows),
                            "history": [
                                {
                                    "epoch": i + 1,
                                    "mAP50": float(r.get("metrics/mAP50(B)", r.get("mAP50", 0))),
                                    "mAP50_95": float(r.get("metrics/mAP50-95(B)", r.get("mAP50-95", 0))),
                                    "precision": float(r.get("metrics/precision(B)", r.get("precision", 0))),
                                    "recall": float(r.get("metrics/recall(B)", r.get("recall", 0))),
                                    "train_loss": float(r.get("train/box_loss", r.get("train_loss", 0))),
                                }
                                for i, r in enumerate(rows)
                            ],
                        }
                        return metrics
                except Exception as e:
                    logger.warning(f"Failed to parse results.csv: {e}")

        # Fallback: try evaluation_report.json (from Colab test evaluation)
        eval_report = project_root / "outputs" / "evaluation" / "evaluation_report.json"
        if eval_report.is_file():
            try:
                import json as _json
                data = _json.loads(eval_report.read_text(encoding="utf-8"))
                return {
                    "mAP50": data.get("mAP50", 0),
                    "mAP50_95": data.get("mAP50_95", 0),
                    "precision": data.get("precision", 0),
                    "recall": data.get("recall", 0),
                    "epochs": 0,
                    "history": [],
                    "source": "evaluation_report",
                    "per_class": data.get("per_class", []),
                }
            except Exception as e:
                logger.warning(f"Failed to parse evaluation_report.json: {e}")

        return {"mAP50": 0, "mAP50_95": 0, "precision": 0, "recall": 0, "epochs": 0, "history": []}

    @app.get("/api/ml/model-info")
    async def get_model_info():
        """Get info about loaded models."""
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        models_dir = project_root / "models"

        model_files = []
        if models_dir.is_dir():
            for f in models_dir.iterdir():
                if f.suffix == ".pt":
                    stat = f.stat()
                    model_files.append({
                        "name": f.name,
                        "size_mb": round(stat.st_size / (1024 * 1024), 2),
                        "last_modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    })

        # Dataset info
        dataset_info = {"total_images": 0, "classes": 0, "dataset_name": "Unknown"}
        data_yaml_paths = [
            project_root / "data" / "raw" / "roboflow" / "rice-diseases-v2" / "data.yaml",
            project_root / "data" / "data.yaml",
        ]
        for dp in data_yaml_paths:
            if dp.is_file():
                try:
                    import yaml
                    with open(dp, "r") as yf:
                        data = yaml.safe_load(yf)
                    dataset_info = {
                        "total_images": data.get("train_count", data.get("nc", 0)),
                        "classes": data.get("nc", len(data.get("names", []))),
                        "dataset_name": dp.parent.name,
                        "class_names": data.get("names", []),
                    }
                except Exception:
                    pass
                break

        # Classifier info
        cls_model, cls_names = _get_classifier()
        classifier_info = None
        if cls_model:
            classifier_info = {
                "name": "india_agri_cls.pt",
                "task": "classify",
                "num_classes": len(cls_names) if cls_names else 0,
                "classes": list(cls_names.values()) if cls_names else [],
            }

        return {
            "models": model_files,
            "dataset": dataset_info,
            "classifier": classifier_info,
            "detector_loaded": any(m is not None for m in _DETECTOR_MODELS.values()) if _DETECTOR_MODELS else False,
        }

    @app.get("/api/ml/training-images/{image_name:path}")
    async def get_training_image(image_name: str):
        """Serve training output images (results.png, confusion_matrix.png)."""
        from fastapi.responses import FileResponse
        # Sanitize: block path traversal
        if ".." in image_name:
            raise HTTPException(status_code=400, detail="Invalid filename")

        project_root = Path(__file__).resolve().parent.parent.parent.parent
        # Try the exact relative path under outputs/training first
        direct = project_root / "outputs" / "training" / image_name
        if direct.is_file():
            return FileResponse(str(direct))

        # Fallback: search by basename in known dirs
        safe_name = Path(image_name).name
        # Try alternate naming conventions (e.g. BoxF1_curve.png for F1_curve.png)
        alt_names = [safe_name]
        if safe_name.startswith("Box"):
            alt_names.append(safe_name[3:])  # BoxF1_curve.png -> F1_curve.png
        else:
            alt_names.append("Box" + safe_name)  # F1_curve.png -> BoxF1_curve.png
            alt_names.append("BoxP_curve.png" if safe_name == "P_curve.png" else safe_name)
            alt_names.append("BoxR_curve.png" if safe_name == "R_curve.png" else safe_name)
        search_paths = [
            project_root / "outputs" / "training" / "yolo_crop_disease",
            project_root / "outputs" / "training" / "rice_v2",
            project_root / "outputs" / "training" / "india_agri_v1",
            project_root / "outputs" / "training" / "rice_disease_gpu",
            project_root / "outputs" / "training",
            project_root / "runs" / "detect" / "train",
        ]
        for base in search_paths:
            for name in alt_names:
                img_path = base / name
                if img_path.is_file():
                    return FileResponse(str(img_path))

        raise HTTPException(status_code=404, detail=f"Image {image_name} not found")

    # ============================================================
    # Training Logs
    # ============================================================
    @app.get("/api/ml/logs")
    async def get_training_logs():
        """Read training log file."""
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        log_paths = [
            project_root / "outputs" / "logs" / "training.log",
            project_root / "logs" / "training.log",
            project_root / "outputs" / "training" / "training.log",
            project_root / "outputs" / "training" / "yolo_crop_disease" / "training.log",
            project_root / "outputs" / "logs" / "agridrone.log",
        ]
        for lp in log_paths:
            if lp.is_file():
                try:
                    content = lp.read_text(encoding="utf-8", errors="replace")
                    lines = content.strip().split("\n")[-200:]  # Last 200 lines
                    return {"logs": lines, "file": str(lp.name)}
                except Exception:
                    pass
        return {"logs": ["No training logs found. Run training to generate logs."], "file": "none"}

    # ============================================================
    # Matrix progress (per_run.jsonl from Colab, read live)
    # ============================================================
    @app.get("/api/ml/matrix")
    async def get_matrix_progress():
        """Read the newest per_run.jsonl from a matrix run.

        Scans local ``evaluate/results/v2/matrix/`` and, if present, common
        Google Drive for Desktop mount points (so the UI can watch a Colab
        run live without a separate sync step).
        """
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        search_roots = [
            project_root / "evaluate" / "results" / "v2" / "matrix",
        ]
        # Google Drive for desktop (Windows): "G:/My Drive/agri-drone/results_v2/matrix"
        drive_candidates = [
            Path(r"G:/My Drive/agri-drone/results_v2/matrix"),
            Path(r"H:/My Drive/agri-drone/results_v2/matrix"),
            Path(os.path.expanduser("~/My Drive/agri-drone/results_v2/matrix")),
            Path(os.path.expanduser("~/Google Drive/agri-drone/results_v2/matrix")),
        ]
        for d in drive_candidates:
            try:
                if d.is_dir():
                    search_roots.append(d)
            except Exception:
                pass

        newest_file: Path | None = None
        newest_mtime = 0.0
        for root in search_roots:
            if not root.is_dir():
                continue
            try:
                for run_dir in root.iterdir():
                    if not run_dir.is_dir():
                        continue
                    jf = run_dir / "per_run.jsonl"
                    if jf.is_file():
                        mt = jf.stat().st_mtime
                        if mt > newest_mtime:
                            newest_mtime = mt
                            newest_file = jf
            except Exception:
                continue

        if newest_file is None:
            return {
                "found": False,
                "message": ("No matrix per_run.jsonl found yet. The Colab run writes to "
                            "Drive; mount Drive via 'Google Drive for desktop' or sync it "
                            "locally to evaluate/results/v2/matrix/ to see live progress."),
                "run_id": None,
                "total": 0,
                "ok": 0,
                "failed": 0,
                "skipped": 0,
                "cells": [],
            }

        cells: list[dict] = []
        try:
            for line in newest_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    cells.append(json.loads(line))
                except Exception:
                    continue
        except Exception as e:
            return {"found": False, "message": f"Read failed: {e}"}

        status_counts = {"ok": 0, "failed": 0, "skipped": 0, "smoke": 0, "other": 0}
        for c in cells:
            s = c.get("status", "other")
            status_counts[s] = status_counts.get(s, 0) + 1

        run_id = cells[-1].get("run_id") if cells else newest_file.parent.name
        return {
            "found": True,
            "run_id": run_id,
            "file": str(newest_file),
            "updated_at": datetime.fromtimestamp(newest_mtime).isoformat(),
            "total": len(cells),
            "ok": status_counts.get("ok", 0),
            "failed": status_counts.get("failed", 0),
            "skipped": status_counts.get("skipped", 0),
            "cells": cells[-50:],  # last 50 rows
        }

    # ============================================================
    # Training Pipeline Status (Colab → UI bridge)
    # ============================================================
    _STATUS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "outputs" / "training"
    _STATUS_DIR.mkdir(parents=True, exist_ok=True)
    _STATUS_FILE = _STATUS_DIR / "training_status.json"

    @app.get("/api/training/status")
    async def get_training_status():
        """Get current Colab training pipeline status."""
        if _STATUS_FILE.is_file():
            try:
                data = json.loads(_STATUS_FILE.read_text(encoding="utf-8"))
                return data
            except Exception as e:
                return {"stage": "error", "message": f"Failed to read status: {e}"}
        return {
            "stage": "idle",
            "message": "Ready. Open the Colab notebook, run training, then sync results to see live progress here.",
            "hint": "Click 'Setup Guide' tab below for step-by-step instructions."
        }

    @app.post("/api/training/status")
    async def update_training_status(payload: dict):
        """Update training status (called by sync script or Colab webhook)."""
        payload["updated_at"] = datetime.now().isoformat()
        _STATUS_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        _log_activity("training_status", payload.get("message", "Status updated"))
        return {"status": "ok"}

    @app.get("/api/training/artifacts")
    async def list_training_artifacts():
        """List available training artifacts (images, logs, models)."""
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        training_dir = project_root / "outputs" / "training"
        artifacts = {"images": [], "logs": [], "models": [], "csvs": []}
        if training_dir.is_dir():
            for f in training_dir.rglob("*"):
                if not f.is_file():
                    continue
                rel = str(f.relative_to(training_dir)).replace("\\", "/")
                size_kb = round(f.stat().st_size / 1024, 1)
                entry = {"name": rel, "size_kb": size_kb, "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat()}
                if f.suffix in (".png", ".jpg", ".jpeg"):
                    artifacts["images"].append(entry)
                elif f.suffix in (".log", ".txt"):
                    artifacts["logs"].append(entry)
                elif f.suffix == ".pt":
                    artifacts["models"].append(entry)
                elif f.suffix == ".csv":
                    artifacts["csvs"].append(entry)
        return artifacts

    # ============================================================
    # Dataset Collector
    # ============================================================
    def _datasets_root() -> Path:
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        return project_root / "datasets"

    def _user_uploads_root() -> Path:
        root = _datasets_root() / "user_uploads"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _sanitize_class_name(name: str) -> str:
        name = (name or "").strip()
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)[:64]
        return safe or "unlabeled"

    @app.get("/api/dataset/stats")
    async def dataset_stats():
        """Summarize every ImageFolder dataset under ``datasets/``.

        Walks ``datasets/externals/`` (one level deep) and ``datasets/user_uploads/``,
        counting JPEG/PNG files per class-folder. Safe on empty installs.
        """
        root = _datasets_root()
        IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        def _summarize(folder: Path) -> dict:
            classes = []
            total_images = 0
            total_bytes = 0
            if not folder.is_dir():
                return {"classes": [], "total_images": 0, "total_bytes": 0}
            for cls in sorted(folder.iterdir()):
                if not cls.is_dir():
                    continue
                imgs = [p for p in cls.iterdir() if p.suffix.lower() in IMG_EXTS]
                if not imgs:
                    continue
                n = len(imgs)
                size = sum(p.stat().st_size for p in imgs)
                classes.append({"name": cls.name, "n_images": n,
                                "size_mb": round(size / (1024 * 1024), 2)})
                total_images += n
                total_bytes += size
            return {"classes": classes, "total_images": total_images,
                    "total_bytes": total_bytes}

        datasets_info: list[dict] = []

        # user uploads (always listed even if empty so UI can render upload target)
        uploads = _summarize(_user_uploads_root())
        datasets_info.append({
            "name": "user_uploads",
            "path": "datasets/user_uploads",
            "writable": True,
            **uploads,
        })

        externals = root / "externals"
        if externals.is_dir():
            for ds in sorted(externals.iterdir()):
                if not ds.is_dir():
                    continue
                # Handle the PlantVillage "color/" nested-folder layout.
                color_sub = ds / "color"
                target = color_sub if color_sub.is_dir() else ds
                info = _summarize(target)
                if not info["classes"]:
                    # May be one-level deeper (e.g. plantvillage dataset/color/).
                    for nested in ds.iterdir():
                        if nested.is_dir():
                            sub = _summarize(nested)
                            if sub["classes"]:
                                info = sub
                                target = nested
                                break
                if info["classes"]:
                    datasets_info.append({
                        "name": ds.name,
                        "path": str(target.relative_to(root.parent)).replace("\\", "/"),
                        "writable": False,
                        **info,
                    })

        return {
            "root": str(root),
            "datasets": datasets_info,
            "uploads_dir": str(_user_uploads_root()),
        }

    @app.post("/api/dataset/upload")
    async def dataset_upload(
        class_name: str = Form(...),
        files: List[UploadFile] = File(...),
    ):
        """Append user-labeled images to ``datasets/user_uploads/<class_name>/``.

        Idempotent per filename (overwrites). Rejects non-image content types.
        Returns the updated per-class count so the UI can refresh without a
        second round-trip.
        """
        safe_cls = _sanitize_class_name(class_name)
        dest = _user_uploads_root() / safe_cls
        dest.mkdir(parents=True, exist_ok=True)

        saved = []
        skipped = []
        for up in files:
            ctype = (up.content_type or "").lower()
            if not ctype.startswith("image/"):
                skipped.append({"name": up.filename, "reason": f"not an image ({ctype})"})
                continue
            # sanitize filename
            base = os.path.basename(up.filename or "upload.jpg")
            base = "".join(c if c.isalnum() or c in "-_." else "_" for c in base)[:120]
            target = dest / base
            try:
                content = await up.read()
                if not content:
                    skipped.append({"name": base, "reason": "empty file"})
                    continue
                target.write_bytes(content)
                saved.append({"name": base, "bytes": len(content)})
            except Exception as e:
                skipped.append({"name": base, "reason": f"{type(e).__name__}: {e}"})

        n_total = sum(1 for p in dest.iterdir() if p.suffix.lower()
                      in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
        return {
            "class": safe_cls,
            "saved": saved,
            "skipped": skipped,
            "total_in_class": n_total,
        }

    @app.delete("/api/dataset/class/{class_name}")
    async def dataset_delete_class(class_name: str):
        """Delete all images in ``user_uploads/<class_name>/`` (irreversible)."""
        safe_cls = _sanitize_class_name(class_name)
        target = _user_uploads_root() / safe_cls
        if not target.is_dir():
            raise HTTPException(status_code=404, detail=f"class '{safe_cls}' not found")
        import shutil as _sh
        _sh.rmtree(target, ignore_errors=True)
        return {"deleted": safe_cls}

    # ============================================================
    # Model Reload
    # ============================================================
    @app.post("/api/model/reload")
    async def reload_model():
        """Hot-reload the YOLO detection model without restarting."""
        global _DETECTOR_MODELS, _CLASSIFIER_MODEL, _CLASSIFIER_NAMES
        try:
            _DETECTOR_MODELS.clear()
            _CLASSIFIER_MODEL = None
            _CLASSIFIER_NAMES = None
            det = _get_detector("auto")
            cls, names = _get_classifier()
            _log_activity("model_reload", "Models reloaded successfully")
            return {
                "status": "reloaded",
                "detector_loaded": det is not None,
                "classifier_loaded": cls is not None,
                "classifier_classes": len(names) if names else 0,
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Reload failed: {str(e)}")

    # ============================================================
    # Batch Detection (process multiple images in one request)
    # ============================================================
    @app.post("/detect/batch")
    async def detect_batch(
        files: List[UploadFile] = File(...),
        confidence_threshold: float = Form(0.3),
        crop_type: str = Form("wheat"),
        area_acres: float = Form(1.0),
        growth_stage: str = Form("unknown"),
    ):
        """
        Batch detection: accepts multiple images, returns results for each.
        Each image goes through the full pipeline (classifier + YOLO + reasoning).
        """
        if len(files) > 50:
            raise HTTPException(status_code=400, detail="Maximum 50 images per batch")

        results = []
        summary = {
            "total_images": len(files),
            "healthy_count": 0,
            "diseased_count": 0,
            "rejected_count": 0,
            "disease_distribution": {},
            "avg_health_score": 0,
            "total_detections": 0,
        }
        health_scores = []

        for idx, file in enumerate(files):
            try:
                contents = await file.read()
                if not contents:
                    results.append({"index": idx, "filename": file.filename, "error": "Empty file", "rejected": True})
                    summary["rejected_count"] += 1
                    continue

                nparr = np.frombuffer(contents, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if image is None:
                    results.append({"index": idx, "filename": file.filename, "error": "Failed to decode", "rejected": True})
                    summary["rejected_count"] += 1
                    continue

                size_reject = _check_minimum_size(image)
                if size_reject:
                    size_reject.update({"index": idx, "filename": file.filename})
                    results.append(size_reject)
                    summary["rejected_count"] += 1
                    continue

                image = _resize_if_large(image)

                # Plant check
                plant_check = _is_plant_image(image)
                if not plant_check["is_plant"]:
                    # Return thumbnail for rejected
                    thumb = cv2.resize(image, (200, 200))
                    _, enc = cv2.imencode('.jpg', thumb, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    thumb_b64 = f"data:image/jpeg;base64,{base64.b64encode(enc).decode()}"
                    results.append({
                        "index": idx, "filename": file.filename,
                        "rejected": True, "rejection_reason": plant_check["reason"],
                        "thumbnail": thumb_b64,
                    })
                    summary["rejected_count"] += 1
                    continue

                start_t = time.time()

                # Classifier
                classifier_result = _classify_image(image)

                # YOLO detection
                detections, annotated_b64 = _yolo_detect(image, confidence_threshold, True, crop_type)

                # Reasoning engine
                reasoning_result = None
                try:
                    from ..vision.disease_reasoning import run_full_pipeline, diagnosis_to_dict
                    pipeline_output = run_full_pipeline(image, classifier_result, crop_type)
                    diagnosis = pipeline_output.diagnosis
                    reasoning_result = diagnosis_to_dict(diagnosis)
                except Exception:
                    pass

                # Ensemble
                ensemble = _compute_ensemble(None, classifier_result)

                proc_ms = (time.time() - start_t) * 1000

                # Determine disease
                disease_name = "Healthy"
                confidence_val = 0.0
                if reasoning_result and reasoning_result.get("disease_key", "healthy") != "healthy":
                    disease_name = reasoning_result.get("disease_name", "Unknown")
                    confidence_val = reasoning_result.get("confidence", 0)
                elif classifier_result:
                    disease_name = classifier_result.get("top_prediction", "Unknown")
                    confidence_val = classifier_result.get("top_confidence", 0)

                is_healthy = disease_name.lower().startswith("healthy")
                health_score = ensemble.get("ensemble_health_score", 50)
                health_scores.append(health_score)

                # Update summary
                if is_healthy:
                    summary["healthy_count"] += 1
                else:
                    summary["diseased_count"] += 1
                summary["total_detections"] += len(detections)
                summary["disease_distribution"][disease_name] = summary["disease_distribution"].get(disease_name, 0) + 1

                # Build thumbnail with annotations
                thumb_img = image.copy()
                # Draw YOLO bboxes on thumbnail
                for det in detections:
                    x1 = int(det.get("x1", 0))
                    y1 = int(det.get("y1", 0))
                    x2 = int(det.get("x2", 0))
                    y2 = int(det.get("y2", 0))
                    cv2.rectangle(thumb_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{det.get('class_name', '')} {det.get('confidence', 0):.0%}"
                    cv2.putText(thumb_img, label, (x1, max(y1-5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                _, enc = cv2.imencode('.jpg', thumb_img, [cv2.IMWRITE_JPEG_QUALITY, 80])
                annotated_thumb = f"data:image/jpeg;base64,{base64.b64encode(enc).decode()}"

                # Original image thumbnail (clean)
                _, enc2 = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 80])
                original_b64 = f"data:image/jpeg;base64,{base64.b64encode(enc2).decode()}"

                results.append({
                    "index": idx,
                    "filename": file.filename,
                    "rejected": False,
                    "disease_name": disease_name,
                    "confidence": round(confidence_val, 3),
                    "health_score": health_score,
                    "risk_level": ensemble.get("ensemble_risk_level", "medium"),
                    "detections": detections,
                    "num_detections": len(detections),
                    "annotated_image": annotated_thumb,
                    "original_image": original_b64,
                    "classifier_top5": classifier_result.get("top5", []) if classifier_result else [],
                    "processing_time_ms": round(proc_ms, 1),
                    "treatment": reasoning_result.get("treatment", {}) if reasoning_result else None,
                    "evidence": reasoning_result.get("evidence", []) if reasoning_result else [],
                })

            except Exception as e:
                logger.warning(f"Batch item {idx} ({file.filename}) failed: {e}")
                results.append({"index": idx, "filename": file.filename, "error": str(e), "rejected": True})
                summary["rejected_count"] += 1

        summary["avg_health_score"] = round(sum(health_scores) / len(health_scores), 1) if health_scores else 0
        return {"results": results, "summary": summary}

    # ============================================================
    # Video Detection (upload video, extract frames, detect)
    # ============================================================
    @app.post("/detect/video")
    async def detect_video(
        file: UploadFile = File(...),
        confidence_threshold: float = Form(0.3),
        crop_type: str = Form("wheat"),
        frame_interval: int = Form(30),
        max_frames: int = Form(20),
    ):
        """
        Video detection: upload a video file, extract frames at intervals,
        run detection on each frame.
        frame_interval: extract every Nth frame (default: every 30th frame = ~1/sec at 30fps)
        max_frames: maximum frames to process (default: 20)
        """
        import tempfile

        if max_frames > 100:
            max_frames = 100

        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty video file")

        # Write to temp file for OpenCV
        suffix = Path(file.filename).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        try:
            cap = cv2.VideoCapture(tmp_path)
            if not cap.isOpened():
                raise HTTPException(status_code=400, detail="Could not open video file")

            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_sec = total_frames / fps if fps > 0 else 0

            frames_results = []
            frame_idx = 0
            processed = 0
            health_scores = []
            disease_dist = {}

            while processed < max_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break

                frame = _resize_if_large(frame)
                timestamp_sec = round(frame_idx / fps, 2) if fps > 0 else 0

                # Skip frames too small for classification
                if _check_minimum_size(frame):
                    frame_idx += frame_interval
                    continue

                # Plant check (skip non-plant frames silently)
                plant_check = _is_plant_image(frame)
                if not plant_check["is_plant"]:
                    frame_idx += frame_interval
                    continue

                start_t = time.time()

                # Classifier
                classifier_result = _classify_image(frame)

                # YOLO detection
                detections, _ = _yolo_detect(frame, confidence_threshold, False, crop_type)

                # Reasoning
                reasoning_result = None
                try:
                    from ..vision.disease_reasoning import run_full_pipeline, diagnosis_to_dict
                    po = run_full_pipeline(frame, classifier_result, crop_type)
                    reasoning_result = diagnosis_to_dict(po.diagnosis)
                except Exception:
                    pass

                # Ensemble
                ensemble = _compute_ensemble(None, classifier_result)

                proc_ms = (time.time() - start_t) * 1000

                disease_name = "Healthy"
                confidence_val = 0.0
                if reasoning_result and reasoning_result.get("disease_key", "healthy") != "healthy":
                    disease_name = reasoning_result.get("disease_name", "Unknown")
                    confidence_val = reasoning_result.get("confidence", 0)
                elif classifier_result:
                    disease_name = classifier_result.get("top_prediction", "Unknown")
                    confidence_val = classifier_result.get("top_confidence", 0)

                health_score = ensemble.get("ensemble_health_score", 50)
                health_scores.append(health_score)
                disease_dist[disease_name] = disease_dist.get(disease_name, 0) + 1

                # Draw detections on frame for annotated thumbnail
                annotated_frame = frame.copy()
                for det in detections:
                    x1 = int(det.get("x1", 0))
                    y1 = int(det.get("y1", 0))
                    x2 = int(det.get("x2", 0))
                    y2 = int(det.get("y2", 0))
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{det.get('class_name', '')} {det.get('confidence', 0):.0%}"
                    cv2.putText(annotated_frame, label, (x1, max(y1-5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                _, enc = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                annotated_b64 = f"data:image/jpeg;base64,{base64.b64encode(enc).decode()}"
                _, enc2 = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                original_b64 = f"data:image/jpeg;base64,{base64.b64encode(enc2).decode()}"

                frames_results.append({
                    "frame_index": frame_idx,
                    "timestamp_sec": timestamp_sec,
                    "disease_name": disease_name,
                    "confidence": round(confidence_val, 3),
                    "health_score": health_score,
                    "detections": detections,
                    "num_detections": len(detections),
                    "annotated_image": annotated_b64,
                    "original_image": original_b64,
                    "processing_time_ms": round(proc_ms, 1),
                })

                processed += 1
                frame_idx += frame_interval

            cap.release()

            summary = {
                "filename": file.filename,
                "fps": round(fps, 1),
                "total_video_frames": total_frames,
                "duration_sec": round(duration_sec, 1),
                "frames_processed": len(frames_results),
                "frame_interval": frame_interval,
                "avg_health_score": round(sum(health_scores) / len(health_scores), 1) if health_scores else 0,
                "disease_distribution": disease_dist,
                "health_timeline": [{"timestamp": f["timestamp_sec"], "health": f["health_score"]} for f in frames_results],
            }

            return {"frames": frames_results, "summary": summary}

        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    @app.post("/missions")
    async def create_mission(mission_data: dict):
        """Create a new mission."""
        logger.info(f"Creating mission: {mission_data}")
        return {"mission_id": "test_mission_001", "status": "created"}

    @app.get("/missions/{mission_id}")
    async def get_mission(mission_id: str):
        """Get mission details."""
        return {"mission_id": mission_id, "status": "pending"}

    @app.post("/missions/{mission_id}/prescribe")
    async def generate_prescription(mission_id: str):
        """Generate prescription map."""
        logger.info(f"Generating prescription for mission {mission_id}")
        return {"mission_id": mission_id, "status": "prescribed"}

    logger.info("FastAPI application initialized")
    return app


# Lazy app instance — created on first access (e.g. by uvicorn)
_app_instance = None

def get_app():
    global _app_instance
    if _app_instance is None:
        _app_instance = create_app()
    return _app_instance

# Module-level __getattr__ so `from agridrone.api.app import app` and
# uvicorn agridrone.api.app:app both work without eager creation.
def __getattr__(name):
    if name == "app":
        return get_app()
    raise AttributeError(name)

