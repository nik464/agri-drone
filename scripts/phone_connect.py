#!/usr/bin/env python
"""
phone_connect.py — Wireless phone-to-laptop crop analysis.

Start a local server, scan QR code on your phone, take a photo of your crop,
and get an instant health analysis — no cables or cloud services needed.

Usage:
    python scripts/phone_connect.py --crop wheat
    python scripts/phone_connect.py --crop rice --port 8765
"""
import argparse
import asyncio
import base64
import hashlib
import io
import json
import os
import platform
import re
import socket
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import ollama
import qrcode
import uvicorn
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response, JSONResponse
from PIL import Image
from rich.console import Console
from rich.panel import Panel

# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Globals initialised in main()
# ---------------------------------------------------------------------------
console = Console()
app = FastAPI(title="Agri-Drone Phone Connect")

# Allow dashboard frontend to call session APIs
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None  # ultralytics YOLO detector model
_cls_model = None  # ultralytics YOLO classifier model (india_agri_cls.pt)
_cls_names = None  # class names from classifier
_crop_type: str = "wheat"
_output_dir: Path = _PROJECT_ROOT / "outputs" / "phone_test"
_annotated_dir: Path = _output_dir / "annotated"

# Stores the latest result per upload so the phone can poll for it
_results: dict[str, dict] = {}

# Job tracking for async analysis with timeout
_jobs: dict[str, dict] = {}  # job_id -> {status, started_at, result, ...}
_jobs_lock = threading.Lock()

# Simple global dict for job results — accessible from any thread
_job_results: dict[str, dict] = {}
_job_lock = threading.Lock()


def safe_str(s):
    """Remove surrogate characters that break UTF-8 encoding."""
    if not isinstance(s, str):
        return str(s) if s else ""
    return s.encode('utf-8', errors='ignore').decode('utf-8')


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------
_SESSIONS_FILE: Path = _PROJECT_ROOT / "outputs" / "sessions.json"
_SESSIONS_DIR: Path = _PROJECT_ROOT / "outputs" / "sessions"
_sessions: dict[str, dict] = {}  # session_id -> session data
_activity_feed: list[dict] = []  # most recent activity events (capped at 200)
_sessions_lock = threading.Lock()
_server_port: int = 8765  # Set in main()


def _load_sessions() -> None:
    """Load sessions from disk."""
    global _sessions, _activity_feed
    if _SESSIONS_FILE.is_file():
        try:
            data = json.loads(_SESSIONS_FILE.read_text(encoding="utf-8"))
            _sessions = data.get("sessions", {})
            _activity_feed = data.get("activity_feed", [])
        except Exception:
            _sessions = {}
            _activity_feed = []


def _save_sessions() -> None:
    """Persist sessions to disk."""
    _SESSIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _SESSIONS_FILE.write_text(
        json.dumps({"sessions": _sessions, "activity_feed": _activity_feed}, indent=2, default=str),
        encoding="utf-8",
    )


def _save_session_file(session_id: str) -> None:
    """Save individual session JSON to outputs/sessions/[date]/[session_id].json."""
    sess = _sessions.get(session_id)
    if not sess:
        return
    date_str = datetime.now().strftime("%Y-%m-%d")
    day_dir = _SESSIONS_DIR / date_str
    day_dir.mkdir(parents=True, exist_ok=True)
    fpath = day_dir / f"{session_id}.json"
    fpath.write_text(json.dumps(sess, indent=2, default=str), encoding="utf-8")


def _get_or_create_session(request: Request, device_name: str = "", location: str = "", crop: str = "") -> str:
    """Get existing session for this device or create a new one."""
    # Build a stable device fingerprint from user-agent + IP
    ua = request.headers.get("user-agent", "unknown")
    ip = request.client.host if request.client else "0.0.0.0"
    fingerprint = hashlib.sha256(f"{ua}:{ip}".encode()).hexdigest()[:16]

    with _sessions_lock:
        # Check if there's already an active session for this fingerprint
        for sid, sess in _sessions.items():
            if sess.get("device_id") == fingerprint and sess.get("status") == "active":
                # Update last_activity
                sess["last_activity"] = datetime.now().isoformat()
                if device_name:
                    sess["device_name"] = device_name
                if location:
                    sess["location"] = location
                if crop:
                    sess["crop_type"] = crop
                _save_sessions()
                return sid

        # Create new session
        session_id = uuid.uuid4().hex[:12]
        browser = "Unknown"
        ua_lower = ua.lower()
        if "chrome" in ua_lower and "edg" not in ua_lower:
            browser = "Chrome"
        elif "safari" in ua_lower and "chrome" not in ua_lower:
            browser = "Safari"
        elif "firefox" in ua_lower:
            browser = "Firefox"
        elif "edg" in ua_lower:
            browser = "Edge"

        _sessions[session_id] = {
            "session_id": session_id,
            "device_id": fingerprint,
            "ip": ip,
            "device_name": device_name or f"Phone-{fingerprint[:6]}",
            "browser": browser,
            "user_agent": ua[:200],
            "connected_at": datetime.now().isoformat(),
            "last_activity": datetime.now().isoformat(),
            "location": location or "Unknown",
            "crop_type": crop or _crop_type,
            "photos_uploaded": 0,
            "photos": [],
            "diseases_found": [],
            "health_scores": [],
            "status": "active",
        }
        _add_activity("connect", session_id, f"New device connected: {_sessions[session_id]['device_name']}")
        _save_sessions()
        return session_id


def _add_activity(event_type: str, session_id: str, message: str) -> None:
    """Add an event to the activity feed (caller must hold _sessions_lock or call inside one)."""
    _activity_feed.insert(0, {
        "id": uuid.uuid4().hex[:8],
        "type": event_type,
        "session_id": session_id,
        "message": message,
        "timestamp": datetime.now().isoformat(),
    })
    # Cap at 200 entries
    if len(_activity_feed) > 200:
        _activity_feed[:] = _activity_feed[:200]

# ---------------------------------------------------------------------------
# Inline class metadata (from YOLOv8Detector)
# ---------------------------------------------------------------------------
_CLASS_SEVERITY = {
    "healthy_wheat": 0.0, "wheat_lodging": 0.8, "wheat_leaf_rust": 0.75,
    "wheat_yellow_rust": 0.85, "wheat_powdery_mildew": 0.6, "wheat_nitrogen_def": 0.55,
    "wheat_weed": 0.5, "healthy_rice": 0.0, "rice_blast": 0.9,
    "rice_brown_planthopper": 0.8, "rice_water_stress": 0.5, "rice_weed": 0.45,
    "poor_row_spacing": 0.3, "good_row_spacing": 0.0,
}
_CLASS_CATEGORIES = {
    "healthy_wheat": "health", "wheat_lodging": "lodging", "wheat_leaf_rust": "disease",
    "wheat_yellow_rust": "disease", "wheat_powdery_mildew": "disease",
    "wheat_nitrogen_def": "nutrient", "wheat_weed": "weed", "healthy_rice": "health",
    "rice_blast": "disease", "rice_brown_planthopper": "disease",
    "rice_water_stress": "stress", "rice_weed": "weed",
    "poor_row_spacing": "stand", "good_row_spacing": "health",
}
_CLASS_CROP_TYPE = {
    "healthy_wheat": "wheat", "wheat_lodging": "wheat", "wheat_leaf_rust": "wheat",
    "wheat_yellow_rust": "wheat", "wheat_powdery_mildew": "wheat",
    "wheat_nitrogen_def": "wheat", "wheat_weed": "wheat", "healthy_rice": "rice",
    "rice_blast": "rice", "rice_brown_planthopper": "rice", "rice_water_stress": "rice",
    "rice_weed": "rice", "poor_row_spacing": "rice", "good_row_spacing": "rice",
}

# ---------------------------------------------------------------------------
# LLaVA vision analysis (PRIMARY) — sends actual image to multimodal model
# ---------------------------------------------------------------------------
_LLAVA_MODEL = "llava"

_LLAVA_PROMPT = """You are an expert plant pathologist specializing in Indian and global wheat and rice diseases. Analyze this crop field image carefully.

DISEASE REFERENCE GUIDE — know exactly what to look for:

1. Fusarium Head Blight (FHB / Scab):
   - Orange, salmon-pink, or tan spore masses (sporodochia) on wheat heads
   - Bleached/whitened spikelets while adjacent spikelets remain green
   - Shriveled, chalky-white or pink-tinged kernels
   - Typically appears during flowering (anthesis) in warm humid weather
   - In severe cases entire head turns straw-colored prematurely

2. Wheat Leaf Rust (Puccinia triticina):
   - Small round orange-brown pustules (uredinia) scattered on upper leaf surface
   - Pustules rupture epidermis, releasing powdery orange spores
   - Leaves feel rough/gritty to touch
   - Distinguished from stripe rust by random (not striped) pustule arrangement

3. Yellow/Stripe Rust (Puccinia striiformis):
   - Bright yellow-orange pustules arranged in distinct linear stripes along leaf veins
   - Stripes run parallel to the leaf midrib
   - Often starts on lower leaves and progresses upward
   - Favored by cool (10-15C) and wet conditions

4. Powdery Mildew (Blumeria graminis):
   - White to gray powdery fungal growth on leaf surfaces and stems
   - Starts as small white patches, spreads to cover entire leaf
   - Affected leaves may yellow and die prematurely
   - Favored by high humidity and moderate temperatures

5. Wheat Lodging:
   - Stems bent or fallen over at the base or mid-stem
   - Creates a flattened, tangled appearance across the field
   - May show root exposure at the base
   - Often follows heavy rain or wind events

6. Nitrogen Deficiency:
   - General yellowing (chlorosis) starting from older/lower leaves
   - Yellowing progresses from leaf tips inward in a V-pattern
   - Stunted growth, thin stems, reduced tillering
   - Pale green to yellow color overall vs. healthy dark green

7. Rice Blast (Magnaporthe oryzae):
   - Diamond/spindle-shaped lesions with gray center and brown margin on leaves
   - Neck blast: brown-black lesion at panicle base causing white/empty panicles
   - Node blast: black lesions at stem nodes

8. Rice Brown Planthopper damage:
   - "Hopper burn": circular patches of dried, brown plants in the field
   - Starts as yellowing, rapidly becomes brown and crispy
   - Often in patches radiating outward

9. Healthy Crop:
   - Uniform dark green color across the field
   - Upright stems with no visible spots, lesions, or discoloration
   - Full, plump heads (wheat) or panicles (rice) with no bleaching

ANALYSIS INSTRUCTIONS:
- Be very specific. If you see ANY orange, pink, brown discoloration on wheat heads or leaves, identify it as a disease.
- A wheat head with mixed bleached and green spikelets is a STRONG indicator of Fusarium Head Blight.
- Orange/salmon coloring on wheat heads is almost certainly FHB or rust — NEVER call it healthy.
- Do not default to "healthy" unless the crop is uniformly green with zero visible symptoms.

Respond ONLY in this JSON format (no markdown fences, no extra text):
{
  "health_score": <integer 0-100>,
  "risk_level": "<low|medium|high|critical>",
  "diseases_found": ["list of diseases seen"],
  "confidence": "<low|medium|high>",
  "visible_symptoms": "<describe exactly what you see in the image>",
  "affected_area_pct": <integer 0-100>,
  "recommendations": ["list of actions using Indian inputs like Propiconazole 25% EC, Tricyclazole 75% WP, Tebuconazole 25.9% EC, Urea 46-0-0, DAP 18-46-0, Imidacloprid 17.8% SL, Pendimethalin 30% EC"],
  "urgency": "<immediate|within_7_days|seasonal>"
}"""


def _llava_analyze(image_bgr: np.ndarray) -> dict | None:
    """Send image to LLaVA via Ollama for visual disease diagnosis.

    Returns parsed dict on success, None on failure.
    """
    try:
        # Encode image as base64 JPEG
        success, encoded = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not success:
            console.print("[red]Failed to encode image for LLaVA[/red]")
            return None
        image_b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")

        console.print("[cyan]  Sending image to LLaVA for visual diagnosis...[/cyan]")
        response = ollama.chat(
            model=_LLAVA_MODEL,
            messages=[{
                "role": "user",
                "content": _LLAVA_PROMPT,
                "images": [image_b64],
            }],
        )

        raw = response["message"]["content"].strip()
        console.print(f"[dim]  LLaVA raw response length: {len(raw)} chars[/dim]")

        result = _parse_llava_response(raw)
        if result is None:
            console.print("[yellow]  LLaVA response could not be parsed at all[/yellow]")
            return None
        return result

    except Exception as exc:
        console.print(f"[yellow]  LLaVA analysis failed: {exc}[/yellow]")
        return None


def _parse_llava_response(raw: str) -> dict | None:
    """Robustly parse LLaVA output which may be wrapped in markdown code fences.

    Handles:
      - Clean JSON
      - ```json ... ``` wrapped blocks
      - ``` ... ``` wrapped blocks (no language tag)
      - Trailing commas, missing commas
      - Complete parse failure → regex field extraction fallback
    """
    text = raw.strip()

    # ── Step 1: Remove markdown code fences ──
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        # May be ```\n{...}\n``` or ```{...}```
        parts = text.split("```")
        # Take the first non-empty part after a fence
        for part in parts[1:]:
            stripped = part.strip()
            if stripped.startswith("{"):
                text = stripped
                break
        else:
            # Fallback: just strip leading/trailing fences
            if text.startswith("```"):
                text = text.split("\n", 1)[-1]
            if text.endswith("```"):
                text = text.rsplit("```", 1)[0]
            text = text.strip()

    # ── Step 2: Fix common JSON errors ──
    text = re.sub(r",\s*}", "}", text)   # trailing comma before }
    text = re.sub(r",\s*]", "]", text)   # trailing comma before ]

    # ── Step 3: Try standard JSON parse ──
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
    except json.JSONDecodeError as e:
        console.print(f"[yellow]  LLaVA JSON parse failed ({e}), trying regex fallback...[/yellow]")

    # ── Step 4: Regex field extraction fallback ──
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

        result = {
            "health_score": int(health_m.group(1)) if health_m else 50,
            "risk_level": risk_m.group(1) if risk_m else "medium",
            "diseases_found": diseases if diseases else [],
            "confidence": confidence_m.group(1) if confidence_m else "medium",
            "visible_symptoms": symptoms_m.group(1) if symptoms_m else raw[:300],
            "affected_area_pct": int(affected_m.group(1)) if affected_m else 0,
            "recommendations": recs if recs else ["Apply recommended fungicide"],
            "urgency": urgency_m.group(1) if urgency_m else "within_7_days",
        }
        console.print(f"[green]  Regex fallback extracted: health={result['health_score']}, "
                      f"diseases={result['diseases_found']}[/green]")
        return result
    except Exception as exc:
        console.print(f"[red]  Regex fallback also failed: {exc}[/red]")
        return {
            "health_score": 40,
            "risk_level": "high",
            "diseases_found": ["Unable to parse — manual review needed"],
            "confidence": "low",
            "visible_symptoms": raw[:500],
            "affected_area_pct": 0,
            "recommendations": ["Manually inspect the field"],
            "urgency": "within_7_days",
        }


# ---------------------------------------------------------------------------
# Inline LLM text helpers (fallback when LLaVA unavailable)
# ---------------------------------------------------------------------------
_OLLAMA_URL = "http://localhost:11434/api/generate"
_OLLAMA_MODEL = "llama3.2"
_CLAUDE_URL = "https://api.anthropic.com/v1/messages"
_CLAUDE_MODEL = "claude-opus-4-5"
_LLM_TIMEOUT = 120.0


def _build_prompt(detections: list[dict], crop_type: str, field_id: str) -> str:
    det_json = json.dumps(detections, indent=2, default=str)
    return (
        "You are a precision agriculture research scientist advising farmers "
        "in North India (Punjab / Haryana / western UP). "
        "Analyze the following drone-based crop detection results and return "
        "ONLY valid JSON (no markdown fences, no commentary).\n\n"
        f"Field ID: {field_id}\nCrop type: {crop_type}\n"
        f"Detection results:\n{det_json}\n\n"
        "Return JSON with exactly these keys:\n"
        '{"overall_field_health": <int 0-100>, "risk_level": "<low|medium|high|critical>", '
        '"primary_issues": [{"issue": "<string>", "severity": <float 0-1>}], '
        '"recommendations": [{"action": "<string>", "priority": "<low|medium|high|urgent>", '
        '"input": "<Indian product name>", "rate": "<amount in kg/ha or ml/ha>"}], '
        '"research_notes": "<paragraph>", "follow_up_scan_days": <integer>}\n'
    )


async def _call_llm(detections: list[dict], crop_type: str) -> dict:
    """Call Claude (if API key set) or Ollama for analysis."""
    import httpx

    prompt = _build_prompt(detections, crop_type, "PHONE_SCAN")
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    async with httpx.AsyncClient(timeout=_LLM_TIMEOUT) as client:
        if api_key:
            resp = await client.post(
                _CLAUDE_URL,
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": _CLAUDE_MODEL,
                    "max_tokens": 4096,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            resp.raise_for_status()
            raw = resp.json()["content"][0]["text"]
        else:
            resp = await client.post(
                _OLLAMA_URL,
                json={"model": _OLLAMA_MODEL, "prompt": prompt, "stream": False, "format": "json"},
            )
            resp.raise_for_status()
            raw = resp.json().get("response", "{}")

    # Parse JSON from LLM response
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return {"overall_field_health": 0, "risk_level": "unknown",
                "primary_issues": [], "recommendations": [],
                "research_notes": text, "follow_up_scan_days": 7}


# ---------------------------------------------------------------------------
# Fast classifier helper (replaces LLaVA as primary analysis)
# ---------------------------------------------------------------------------
_CLS_DISPLAY = {
    "healthy_wheat": "Healthy Wheat", "healthy_rice": "Healthy Rice",
    "wheat_fusarium_head_blight": "Fusarium Head Blight (FHB / Scab)",
    "wheat_yellow_rust": "Yellow / Stripe Rust", "wheat_black_rust": "Black / Stem Rust",
    "wheat_brown_rust": "Brown / Leaf Rust", "wheat_leaf_blight": "Leaf Blight",
    "wheat_powdery_mildew": "Powdery Mildew", "wheat_septoria": "Septoria Leaf Blotch",
    "wheat_tan_spot": "Tan Spot", "wheat_smut": "Smut (Loose/Flag)",
    "wheat_root_rot": "Root Rot", "wheat_blast": "Wheat Blast",
    "wheat_aphid": "Aphid Infestation", "wheat_mite": "Mite Damage",
    "wheat_stem_fly": "Stem Fly", "rice_bacterial_blight": "Bacterial Blight",
    "rice_brown_spot": "Brown Spot", "rice_blast": "Rice Blast",
    "rice_leaf_scald": "Rice Leaf Scald", "rice_sheath_blight": "Sheath Blight",
}
_CLS_SEVERITY_21 = {
    "healthy_wheat": 0.0, "healthy_rice": 0.0,
    "wheat_fusarium_head_blight": 0.90, "wheat_yellow_rust": 0.85,
    "wheat_black_rust": 0.80, "wheat_brown_rust": 0.75, "wheat_leaf_blight": 0.70,
    "wheat_powdery_mildew": 0.60, "wheat_septoria": 0.70, "wheat_tan_spot": 0.65,
    "wheat_smut": 0.75, "wheat_root_rot": 0.80, "wheat_blast": 0.95,
    "wheat_aphid": 0.55, "wheat_mite": 0.50, "wheat_stem_fly": 0.60,
    "rice_bacterial_blight": 0.85, "rice_brown_spot": 0.65,
    "rice_blast": 0.90, "rice_leaf_scald": 0.70, "rice_sheath_blight": 0.75,
}


def _classify_with_model(model, names, image_bgr: np.ndarray) -> dict | None:
    """Run the 21-class classifier on an image. Returns a result dict."""
    results = model(image_bgr, verbose=False)
    if not results or results[0].probs is None:
        return None
    probs = results[0].probs
    top5_indices = probs.top5
    top5_confs = probs.top5conf.tolist()
    predictions = []
    for idx, conf in zip(top5_indices, top5_confs):
        class_key = names[idx]
        predictions.append({
            "index": idx,
            "class_key": class_key,
            "class_name": _CLS_DISPLAY.get(class_key, class_key.replace("_", " ").title()),
            "confidence": round(conf, 4),
            "severity": _CLS_SEVERITY_21.get(class_key, 0.5),
        })
    top = predictions[0]
    top_is_healthy = "healthy" in top["class_key"].lower()
    disease_prob = sum(p["confidence"] for p in predictions if "healthy" not in p["class_key"].lower())
    healthy_prob = sum(p["confidence"] for p in predictions if "healthy" in p["class_key"].lower())
    top_disease = next((p for p in predictions if "healthy" not in p["class_key"].lower()), None)

    if top_is_healthy and healthy_prob >= 0.70 and disease_prob < 0.25:
        health_score, is_healthy = 95, True
    elif top_is_healthy and healthy_prob >= 0.70 and disease_prob < 0.40:
        health_score, is_healthy = max(60, round(95 * (1 - disease_prob))), True
    elif top_is_healthy:
        is_healthy = False
        if top_disease:
            top = top_disease
            health_score = max(10, round(100 - top["severity"] * 100 * top["confidence"]))
            if disease_prob > healthy_prob * 0.8:
                health_score = min(health_score, max(10, round(50 * (1 - disease_prob))))
        else:
            health_score = max(30, round(95 * (1 - disease_prob)))
    else:
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
    }


# ---------------------------------------------------------------------------
# Inline YOLO detection helper (SECONDARY — bounding boxes only)
# ---------------------------------------------------------------------------
def _detect_image(image_bgr: np.ndarray, conf: float = 0.4) -> list[dict]:
    """Run YOLO on image_bgr, return list of detection dicts."""
    if _model is None:
        return []
    results = _model(image_bgr, conf=conf, verbose=False)
    if not results:
        return []

    r = results[0]
    h, w = image_bgr.shape[:2]
    image_area = h * w
    dets = []

    if hasattr(r, "boxes") and r.boxes is not None:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            class_name = r.names.get(cls_id, f"class_{cls_id}")
            confidence = float(box.conf[0])
            xyxy = box.xyxy[0]
            x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
            bbox_area = (x2 - x1) * (y2 - y1)

            # Filter small detections
            if bbox_area < 50:
                continue

            base_sev = _CLASS_SEVERITY.get(class_name, 0.5)
            severity = round(min(1.0, base_sev * confidence / 0.5), 4)
            area_pct = round((bbox_area / image_area) * 100, 4) if image_area else 0.0

            dets.append({
                "class_name": class_name,
                "confidence": round(confidence, 3),
                "severity_score": severity,
                "category": _CLASS_CATEGORIES.get(class_name, "health"),
                "crop_type": _CLASS_CROP_TYPE.get(class_name, "unknown"),
                "area_pct": area_pct,
                "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
            })

    return dets

# ---------------------------------------------------------------------------
# Mobile HTML page
# ---------------------------------------------------------------------------
_MOBILE_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no"/>
<title>AgriDrone Field Connect</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
     background:#0f172a;color:#e2e8f0;min-height:100vh;display:flex;
     flex-direction:column;align-items:center;padding:16px}
h1{font-size:1.5rem;margin:12px 0 4px;color:#22d3ee}
.subtitle{font-size:.85rem;color:#94a3b8;margin-bottom:20px}
.card{background:#1e293b;border-radius:16px;padding:24px;width:100%;
      max-width:420px;text-align:center;margin-bottom:16px}
.form-group{text-align:left;margin-bottom:14px}
.form-group label{display:block;font-size:.8rem;color:#94a3b8;margin-bottom:4px;font-weight:600}
.form-group input,.form-group select{width:100%;padding:12px 14px;border:1px solid #334155;
  border-radius:10px;background:#0f172a;color:#e2e8f0;font-size:1rem;outline:none;
  transition:border-color .2s}
.form-group input:focus,.form-group select:focus{border-color:#22d3ee}
.btn{display:inline-block;width:100%;padding:16px;border:none;border-radius:12px;
     font-size:1.1rem;font-weight:700;cursor:pointer;margin-top:10px;
     transition:transform .1s,box-shadow .15s}
.btn:active{transform:scale(.97)}
.btn-primary{background:linear-gradient(135deg,#22c55e,#16a34a);color:#fff;
             box-shadow:0 4px 14px rgba(34,197,94,.4)}
.btn-secondary{background:#334155;color:#cbd5e1}
.btn-whatsapp{background:#25D366;color:#fff;box-shadow:0 4px 14px rgba(37,211,102,.3)}
.btn-sm{padding:12px;font-size:.95rem;margin-top:8px}
input[type=file]{display:none}
.progress-wrap{width:100%;background:#334155;border-radius:8px;height:10px;
               margin-top:14px;overflow:hidden;display:none}
.progress-bar{height:100%;width:0%;background:linear-gradient(90deg,#22c55e,#16a34a);
              border-radius:8px;transition:width .25s}
.status{margin-top:14px;font-size:.9rem;min-height:1.4em;color:#94a3b8}
.connected-badge{display:none;background:#166534;color:#bbf7d0;padding:6px 16px;
                 border-radius:999px;font-size:.8rem;font-weight:700;margin-bottom:14px}
.results-list{width:100%;max-width:420px}
.result-mini{background:#1e293b;border-radius:14px;padding:16px;margin-bottom:12px;text-align:center}
.score-sm{width:64px;height:64px;border-radius:50%;margin:0 auto 8px;
          display:flex;align-items:center;justify-content:center;
          font-size:1.4rem;font-weight:800}
.score-good{border:4px solid #22c55e;color:#22c55e}
.score-mid{border:4px solid #eab308;color:#eab308}
.score-bad{border:4px solid #ef4444;color:#ef4444}
.disease-tag{display:inline-block;font-size:.75rem;padding:3px 10px;border-radius:999px;
             margin:2px;background:rgba(239,68,68,.15);color:#fca5a5}
.rec-text{text-align:left;font-size:.8rem;color:#94a3b8;margin-top:8px;line-height:1.4}
.rec-text b{color:#22d3ee}
.photo-counter{font-size:.75rem;color:#94a3b8;margin-top:4px}
@keyframes spin{to{transform:rotate(360deg)}}
</style>
</head>
<body>
<h1>🌾 AgriDrone Field Connect</h1>
<p class="subtitle">Take crop photos for instant health analysis</p>

<!-- Step 1: Connect form -->
<div class="card" id="connectCard">
  <p style="font-size:.95rem;color:#cbd5e1;margin-bottom:16px;font-weight:600">Step 1: Enter your details</p>
  <div class="form-group">
    <label>Your Name</label>
    <input type="text" id="deviceName" placeholder="e.g. Ashutosh" autocomplete="name"/>
  </div>
  <div class="form-group">
    <label>Field Location</label>
    <input type="text" id="fieldLocation" placeholder="e.g. North Field, Karnal" autocomplete="off"/>
  </div>
  <div class="form-group">
    <label>Crop Type</label>
    <select id="cropSelect">
      <option value="wheat">🌾 Wheat</option>
      <option value="rice">🍚 Rice</option>
    </select>
  </div>
  <button class="btn btn-primary" onclick="connectSession()">Connect & Start Scanning</button>
</div>

<!-- Step 2: Camera (stays visible after each photo) -->
<div class="card" id="uploadCard" style="display:none">
  <div class="connected-badge" id="connBadge">✓ Connected</div>
  <p class="photo-counter" id="photoCounter">0 photos taken</p>
  <label class="btn btn-primary" id="captureBtn">
    📷 Take Photo
    <input type="file" id="fileInput" accept="image/*" capture="environment"/>
  </label>
  <label class="btn btn-secondary btn-sm" id="galleryBtn">
    🖼️ Pick from Gallery
    <input type="file" id="galleryInput" accept="image/*"/>
  </label>
  <div class="progress-wrap" id="progressWrap">
    <div class="progress-bar" id="progressBar"></div>
  </div>
  <div class="status" id="status"></div>
  <!-- Spinner card shown during analysis -->
  <div id="analyzeCard" style="display:none;margin-top:12px;padding:20px;border-radius:16px;background:rgba(76,175,80,0.06);border:1px solid rgba(76,175,80,0.15);text-align:center">
    <div style="display:inline-block;width:40px;height:40px;border:3px solid #e5e7eb;border-top-color:#4CAF50;border-radius:50%;animation:spin 0.8s linear infinite;margin-bottom:10px"></div>
    <p style="font-size:.95rem;font-weight:600;color:#1a3a1a;margin:4px 0" id="analyzeMsg">AI is analyzing your photo...</p>
    <p style="font-size:.8rem;color:#64748b" id="analyzeTimer">Usually takes 20-40 seconds</p>
    <div style="margin-top:10px;height:4px;background:#e5e7eb;border-radius:4px;overflow:hidden">
      <div id="analyzeProgress" style="height:100%;width:0%;background:linear-gradient(90deg,#4CAF50,#66BB6A);transition:width 1s linear;border-radius:4px"></div>
    </div>
  </div>
</div>

<!-- Results stack here (one card per photo) -->
<div class="results-list" id="resultsList"></div>

<script>
let sessionId=null;
let photoCount=0;
const results=[];

const connectCard=document.getElementById('connectCard');
const uploadCard=document.getElementById('uploadCard');
const connBadge=document.getElementById('connBadge');
const photoCounter=document.getElementById('photoCounter');
const fileInput=document.getElementById('fileInput');
const galleryInput=document.getElementById('galleryInput');
const progressWrap=document.getElementById('progressWrap');
const progressBar=document.getElementById('progressBar');
const status=document.getElementById('status');
const resultsList=document.getElementById('resultsList');
const analyzeCard=document.getElementById('analyzeCard');
const analyzeMsg=document.getElementById('analyzeMsg');
const analyzeTimer=document.getElementById('analyzeTimer');
const analyzeProgress=document.getElementById('analyzeProgress');

fileInput.addEventListener('change',handleFile);
galleryInput.addEventListener('change',handleFile);

async function connectSession(){
  const name=document.getElementById('deviceName').value.trim();
  const loc=document.getElementById('fieldLocation').value.trim();
  const crop=document.getElementById('cropSelect').value;
  try{
    const resp=await fetch('/api/connect',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({device_name:name,location:loc,crop:crop})
    });
    const data=await resp.json();
    sessionId=data.session_id;
    connectCard.style.display='none';
    uploadCard.style.display='block';
    connBadge.style.display='inline-block';
    connBadge.textContent='\u2713 Connected'+(name ? ' as '+name : '');
  }catch(e){
    alert('Connection failed. Check WiFi.');
  }
}

function handleFile(e){
  const file=e.target.files[0];
  if(!file)return;
  uploadFile(file);
}

async function uploadFile(file){
  const formData=new FormData();
  formData.append('photo',file);
  progressWrap.style.display='block';
  progressBar.style.width='0%';
  status.textContent='Uploading…';
  analyzeCard.style.display='none';

  try{
    const xhr=new XMLHttpRequest();
    xhr.open('POST','/api/upload');
    xhr.upload.onprogress=function(e){
      if(e.lengthComputable){
        const pct=Math.round(e.loaded/e.total*100);
        progressBar.style.width=pct+'%';
        if(pct>=100) status.textContent='Photo received! Starting analysis…';
      }
    };
    xhr.onload=function(){
      if(xhr.status===200){
        const data=JSON.parse(xhr.responseText);
        if(data.job_id){
          photoCount++;
          photoCounter.textContent=photoCount+' photo'+(photoCount>1?'s':'')+' taken';
          progressWrap.style.display='none';
          status.textContent='';
          showAnalyzeSpinner();
          pollJobStatus(data.job_id);
        }
      } else {
        status.textContent='Upload failed. Try again.';
        progressWrap.style.display='none';
      }
    };
    xhr.onerror=function(){
      status.textContent='Connection error. Check WiFi.';
      progressWrap.style.display='none';
    };
    xhr.send(formData);
  }catch(err){
    status.textContent='Error: '+err.message;
    progressWrap.style.display='none';
  }
}

let analyzeStart=0;
let analyzeInterval=null;

function showAnalyzeSpinner(){
  analyzeCard.style.display='block';
  analyzeMsg.textContent='AI is analyzing your photo…';
  analyzeTimer.textContent='Usually takes 20-40 seconds';
  analyzeProgress.style.width='0%';
  analyzeStart=Date.now();
  if(analyzeInterval) clearInterval(analyzeInterval);
  analyzeInterval=setInterval(()=>{
    const elapsed=Math.round((Date.now()-analyzeStart)/1000);
    const pct=Math.min(95,Math.round(elapsed/120*100));
    analyzeProgress.style.width=pct+'%';
    if(elapsed<20){
      analyzeTimer.textContent='Elapsed: '+elapsed+'s — Usually takes 20-40 seconds';
    } else if(elapsed<60){
      analyzeTimer.textContent='Elapsed: '+elapsed+'s — Almost done…';
    } else if(elapsed<120){
      analyzeMsg.textContent='Still working… LLaVA is doing a detailed analysis';
      analyzeTimer.textContent='Elapsed: '+elapsed+'s — Hang tight!';
    } else {
      analyzeMsg.textContent='Taking longer than usual…';
      analyzeTimer.textContent='Elapsed: '+elapsed+'s — Falling back to quick detection';
    }
  },1000);
}

function hideAnalyzeSpinner(){
  analyzeCard.style.display='none';
  if(analyzeInterval){clearInterval(analyzeInterval);analyzeInterval=null;}
}

async function pollJobStatus(jobId){
  const maxRetries=30;
  for(let i=0;i<maxRetries;i++){
    await new Promise(r=>setTimeout(r,1000));
    try{
      const resp=await fetch('/api/status/'+jobId);
      if(!resp.ok) continue;
      const data=await resp.json();
      console.log('Poll attempt',i,jobId,data);

      if(data.status==='complete'){
        hideAnalyzeSpinner();
        addResultCard(data);
        status.textContent='Done! Take another photo.';
        fileInput.value='';
        galleryInput.value='';
        return;
      }
    }catch(e){console.error('Poll error',e)}
  }
  hideAnalyzeSpinner();
  status.textContent='Analysis timed out. Try again.';
  fileInput.value='';
  galleryInput.value='';
}

function addResultCard(data){
  results.unshift(data);
  const score=data.health||0;
  const disease=data.disease||'None';
  const diseases=data.diseases||[];
  const risk=(data.risk||'low').toUpperCase();
  const symptoms=data.symptoms||'';
  const affected=data.affected_area||0;
  const treatment=typeof data.treatment==='string'?data.treatment:(data.treatment&&data.treatment.action?data.treatment.action:'');
  const allTreatments=data.all_treatments||[];
  const urgency=(data.urgency||'').toUpperCase();
  const confidence=data.confidence||'';
  const line='\u2501'.repeat(22);

  let html='<div class="result-mini" style="text-align:left;font-family:monospace;font-size:.85rem;line-height:1.6;padding:16px">';
  html+='<div style="color:#64748b;font-size:.7rem;margin-bottom:8px">Photo #'+results.length+'</div>';
  html+='<div style="color:#475569">'+line+'</div>';
  html+='<div><b>Disease:</b> <span style="color:#ef4444">'+disease+'</span></div>';
  html+='<div><b>Health:</b> <span style="color:'+(score>=70?'#22c55e':score>=40?'#eab308':'#ef4444')+'">'+score+'/100</span></div>';
  html+='<div><b>Risk:</b> <span style="color:'+(risk==='CRITICAL'||risk==='HIGH'?'#ef4444':risk==='MEDIUM'?'#eab308':'#22c55e')+'">'+risk+'</span></div>';
  html+='<div style="color:#475569">'+line+'</div>';
  if(symptoms) html+='<div><b>Symptoms:</b> '+symptoms.substring(0,120)+'</div>';
  html+='<div><b>Affected:</b> '+affected+'%</div>';
  html+='<div style="color:#475569">'+line+'</div>';
  if(treatment){
    html+='<div><b>Treatment:</b> <span style="color:#22c55e">'+treatment+'</span></div>';
    html+='<div><b>Dose:</b> 1ml per litre</div>';
  }
  if(urgency) html+='<div><b>Spray within:</b> '+urgency+'</div>';
  html+='<div style="color:#475569">'+line+'</div>';

  html+='<button class="btn" style="margin-top:10px;width:100%;padding:12px;background:#0f172a;color:#e2e8f0;border:1px solid #334155;border-radius:10px;font-size:.9rem" onclick="fileInput.click()">Take Another Photo</button>';
  html+='<button class="btn btn-whatsapp" style="margin-top:8px;width:100%;padding:12px;border-radius:10px;font-size:.9rem" onclick="shareViaWhatsApp('+results.length+')">Share on WhatsApp</button>';

  html+='</div>';
  resultsList.insertAdjacentHTML('afterbegin',html);
}

function shareViaWhatsApp(index){
  const data=results[index-1];
  if(!data) return;
  const score=data.health||0;
  const disease=data.disease||'None';
  const treatment=data.treatment||'';
  const risk=(data.risk||'low').toUpperCase();
  let text='[AgriDrone Report] Disease: '+disease+' Health: '+score+'/100 Risk: '+risk;
  if(treatment) text+=' Treatment: '+treatment;
  text+=' Scanned via AgriDrone AI';
  window.open('https://wa.me/?text='+encodeURIComponent(text),'_blank');
}
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _get_local_ip() -> str:
    """Get the machine's LAN IP address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def _print_qr(url: str) -> None:
    """Render QR code in the terminal."""
    qr = qrcode.QRCode(box_size=1, border=1)
    qr.add_data(url)
    qr.make(fit=True)
    # Print using Rich for nice formatting
    f = io.StringIO()
    qr.print_ascii(out=f, invert=True)
    f.seek(0)
    qr_text = f.read()
    console.print()
    console.print(Panel(
        qr_text,
        title="[bold cyan]Scan with your phone[/bold cyan]",
        subtitle=f"[dim]{url}[/dim]",
        border_style="cyan",
        padding=(0, 2),
    ))
    console.print()


def _run_analysis(image_bgr: np.ndarray, filename: str) -> dict:
    """Call the main /detect API (port 9000) for the full pipeline, map to mobile format.

    Uses the exact same pipeline as the dashboard upload:
    Classifier → Reasoning Engine → Grad-CAM → RAG → Ensemble Voting → Temporal
    Then maps the rich structured output to the simple mobile card format.
    """
    import requests as _req

    # Encode image to JPEG bytes for the API call
    success, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not success:
        console.print("[red]  Failed to encode image[/red]")
        return _fallback_result("Could not encode image for analysis")

    console.print("[cyan]  Calling /detect API (full pipeline)...[/cyan]")
    start = time.time()

    try:
        resp = _req.post(
            "http://127.0.0.1:9000/detect",
            files={"file": (filename, buf.tobytes(), "image/jpeg")},
            data={
                "crop_type": _crop_type,
                "use_llava": "false",
                "include_image": "false",
            },
            timeout=30,
        )
        resp.raise_for_status()
    except _req.ConnectionError:
        console.print("[yellow]  Backend API not reachable on port 9000 — using local fallback[/yellow]")
        return _local_fallback_analysis(image_bgr, filename)
    except Exception as exc:
        console.print(f"[yellow]  /detect API failed: {exc} — using local fallback[/yellow]")
        return _local_fallback_analysis(image_bgr, filename)

    elapsed = time.time() - start
    data = resp.json()
    s = data.get("structured") or {}
    diag = s.get("diagnosis", {})
    health_sec = s.get("health", {})
    treatment_sec = s.get("treatment", {})

    # Also log classifier + reasoning from the API for debugging
    cls_r = data.get("classifier_result", {})
    reasoning = data.get("reasoning", {})
    if cls_r:
        console.print(
            f"[dim]  API classifier: {cls_r.get('top_prediction')} "
            f"({cls_r.get('top_confidence', 0) * 100:.0f}%)[/dim]"
        )
    if reasoning:
        console.print(
            f"[dim]  API reasoning: {reasoning.get('disease_name', '?')} "
            f"(rule_score={reasoning.get('confidence', 0):.0%})[/dim]"
        )

    disease_name = diag.get("disease_name", "Unknown")
    is_healthy = "healthy" in disease_name.lower()
    diseases = [] if is_healthy else [disease_name]
    conf = diag.get("confidence", 0)

    # Extract symptoms from evidence
    evidence = s.get("evidence", {})
    symptoms_list = evidence.get("supporting", [])
    symptoms_text = " • ".join(symptoms_list[:5]) if symptoms_list else ""

    # Extract treatments
    recs = treatment_sec.get("recommendations", [])
    urgency = treatment_sec.get("urgency", "within_7_days")

    # Yield/affected area from health section
    yield_loss = health_sec.get("yield_loss_estimate", "0%")
    affected = _parse_yield_loss(yield_loss)

    console.print(
        f"[green]  Pipeline done in {elapsed:.1f}s: {disease_name} "
        f"(conf={conf:.0%}, health={health_sec.get('score', 50)})[/green]"
    )

    # Save annotated image if YOLO found anything
    dets = data.get("detections", [])
    annotated = image_bgr.copy()
    for d in dets:
        x1 = int(d.get("x1", d.get("bbox", [0])[0] if "bbox" in d else 0))
        y1 = int(d.get("y1", d.get("bbox", [0, 0])[1] if "bbox" in d else 0))
        x2 = int(d.get("x2", d.get("bbox", [0, 0, 0])[2] if "bbox" in d else 0))
        y2 = int(d.get("y2", d.get("bbox", [0, 0, 0, 0])[3] if "bbox" in d else 0))
        color = (0, 0, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
    _annotated_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(_annotated_dir / filename), annotated)

    return {
        "ready": True,
        "health_score": health_sec.get("score", 50),
        "risk_level": health_sec.get("risk_level", "medium"),
        "diseases_found": diseases,
        "visible_symptoms": symptoms_text or "See detailed analysis on dashboard",
        "affected_area_pct": affected,
        "top_issue": diseases[0] if diseases else "Healthy crop",
        "recommendations": recs if isinstance(recs, list) else [recs],
        "urgency": urgency,
        "confidence": "high" if conf >= 0.75 else "medium" if conf >= 0.4 else "low",
        "num_detections": len(dets),
        "yolo_detections": dets,
    }


def _parse_yield_loss(val: str) -> int:
    """Parse yield_loss like '40-100%' -> take the first number."""
    s = str(val).replace("%", "").strip()
    if "-" in s:
        s = s.split("-")[0].strip()
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return 0


def _fallback_result(msg: str) -> dict:
    """Return a generic error result for mobile display."""
    return {
        "ready": True,
        "health_score": 50,
        "risk_level": "medium",
        "diseases_found": [],
        "visible_symptoms": msg,
        "affected_area_pct": 0,
        "top_issue": "Analysis failed",
        "recommendations": ["Manually inspect the field"],
        "urgency": "within_7_days",
        "confidence": "low",
        "num_detections": 0,
        "yolo_detections": [],
    }


def _local_fallback_analysis(image_bgr: np.ndarray, filename: str) -> dict:
    """Fallback: classifier + full reasoning engine when backend API is unreachable."""
    classifier_result = None
    if _cls_model is not None:
        try:
            classifier_result = _classify_with_model(_cls_model, _cls_names, image_bgr)
            console.print(
                f"[green]  Local classifier: {classifier_result.get('top_prediction')} "
                f"({classifier_result.get('top_confidence', 0) * 100:.0f}%)[/green]"
            )
        except Exception as exc:
            console.print(f"[yellow]  Local classifier failed: {exc}[/yellow]")

    reasoning_result = None
    try:
        import sys
        sys.path.insert(0, str(_PROJECT_ROOT / "src"))
        from agridrone.vision.disease_reasoning import run_full_pipeline, diagnosis_to_dict
        # Downscale for faster feature extraction (phone images are huge ~4000px)
        h, w = image_bgr.shape[:2]
        max_dim = 800
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            small = cv2.resize(image_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            small = image_bgr
        pipeline_output = run_full_pipeline(small, classifier_result, _crop_type)
        reasoning_result = diagnosis_to_dict(pipeline_output.diagnosis)
        console.print(
            f"[green]  Reasoning: {reasoning_result.get('disease_name')} "
            f"(conf={reasoning_result.get('confidence', 0):.0%}, "
            f"health={reasoning_result.get('health_score')})[/green]"
        )
    except Exception as exc:
        console.print(f"[yellow]  Reasoning engine failed: {exc}[/yellow]")

    _annotated_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(_annotated_dir / filename), image_bgr)

    if reasoning_result:
        dk = reasoning_result.get("disease_key", "")
        diseases = [] if dk.startswith("healthy") else [reasoning_result.get("disease_name", "Unknown")]
        recs = reasoning_result.get("treatment", [])
        symptoms = reasoning_result.get("symptoms_detected", [])
        if not symptoms:
            symptoms = reasoning_result.get("symptoms_matched", [])
        return {
            "ready": True,
            "health_score": reasoning_result.get("health_score", 50),
            "risk_level": reasoning_result.get("risk_level", "medium"),
            "diseases_found": diseases,
            "visible_symptoms": " • ".join(symptoms)[:500] or "See detailed analysis",
            "affected_area_pct": _parse_yield_loss(reasoning_result.get("yield_loss", "0")),
            "top_issue": diseases[0] if diseases else "Healthy crop",
            "recommendations": recs if isinstance(recs, list) else [recs],
            "urgency": reasoning_result.get("urgency", "within_7_days"),
            "confidence": "high" if reasoning_result.get("confidence", 0) >= 0.75 else "medium",
            "num_detections": 0,
            "yolo_detections": [],
        }

    if classifier_result:
        diseases = [] if classifier_result.get("is_healthy") else [classifier_result.get("top_prediction", "Unknown")]
        return {
            "ready": True,
            "health_score": classifier_result.get("health_score", 50),
            "risk_level": classifier_result.get("risk_level", "medium"),
            "diseases_found": diseases,
            "visible_symptoms": f"{classifier_result.get('top_prediction')} ({classifier_result.get('top_confidence', 0) * 100:.0f}%)",
            "affected_area_pct": 0,
            "top_issue": diseases[0] if diseases else "Healthy crop",
            "recommendations": ["Upload to dashboard for detailed analysis"],
            "urgency": "within_7_days",
            "confidence": "medium",
            "num_detections": 0,
            "yolo_detections": [],
        }

    return _fallback_result("No models could analyze this image. Manual inspection recommended.")


# ---------------------------------------------------------------------------
# FastAPI routes
# ---------------------------------------------------------------------------

@app.get("/")
async def mobile_page():
    """Serve the mobile interface."""
    html_bytes = _MOBILE_HTML.encode("utf-8", errors="replace")
    return Response(content=html_bytes, media_type="text/html; charset=utf-8")


@app.post("/api/connect")
async def connect_session(request: Request):
    """Phone calls this when user fills out the connect form."""
    body = await request.json()
    device_name = body.get("device_name", "")
    location = body.get("location", "")
    crop = body.get("crop", "")
    session_id = _get_or_create_session(request, device_name, location, crop)
    return {"session_id": session_id, "message": "Connected!"}


@app.post("/api/upload")
async def upload_photo(request: Request, photo: UploadFile = File(...)):
    """Receive a photo from the phone, immediately return a job_id.

    The phone then polls GET /api/status/{job_id} every 3 seconds.
    Analysis runs in a background thread (~1-2s with fast pipeline).
    """
    upload_id = uuid.uuid4().hex[:12]
    job_id = uuid.uuid4().hex[:16]
    _results[upload_id] = {"ready": False}

    # Register the job immediately so polling can start
    with _jobs_lock:
        _jobs[job_id] = {
            "status": "analyzing",
            "upload_id": upload_id,
            "started_at": time.monotonic(),
            "message": "Photo received! AI is analyzing your crop...",
            "result": None,
        }

    # Session tracking
    session_id = _get_or_create_session(request)
    with _sessions_lock:
        if session_id in _sessions:
            _sessions[session_id]["photos_uploaded"] += 1
            _sessions[session_id]["last_activity"] = datetime.now().isoformat()
            _add_activity("upload", session_id, f"Photo uploaded by {_sessions[session_id]['device_name']}")
            _save_sessions()

    # Read image bytes
    contents = await photo.read()
    nparr = np.frombuffer(contents, np.uint8)
    image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image_bgr is None:
        with _jobs_lock:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["message"] = "Could not decode image"
        return JSONResponse({"error": "Could not decode image"}, status_code=400)

    # Save original
    _output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{ts}_{upload_id}.jpg"
    cv2.imwrite(str(_output_dir / filename), image_bgr)

    console.print(f"\n[bold green]📷 Photo received![/bold green]  ({photo.filename})")
    console.print(f"   Saved to [cyan]{_output_dir / filename}[/cyan]")
    console.print(f"   Session: [cyan]{session_id}[/cyan]")
    console.print(f"   Job: [cyan]{job_id}[/cyan]")
    console.print("   [yellow]Analyzing with fast pipeline (Classifier + Reasoning Engine)...[/yellow]")

    # Run analysis in a thread so we don't block the event loop
    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        None, _background_analyze, upload_id, image_bgr, filename, session_id, job_id,
    )

    return {
        "status": "analyzing",
        "job_id": job_id,
        "upload_id": upload_id,
        "session_id": session_id,
        "message": "Photo received! Analyzing...",
    }


def _yolo_only_result(image_bgr: np.ndarray) -> dict:
    """Quick YOLO-only analysis as fallback when LLaVA times out."""
    try:
        yolo_dets = _detect_image(image_bgr, conf=0.3)
    except Exception:
        yolo_dets = []
    if not yolo_dets:
        return {
            "ready": True,
            "health_score": 60,
            "risk_level": "medium",
            "diseases_found": [],
            "visible_symptoms": "LLaVA timed out. YOLO found no detections. Manual inspection recommended.",
            "affected_area_pct": 0,
            "top_issue": "Analysis timed out — basic detection only",
            "recommendations": ["Manually inspect the crop area"],
            "urgency": "within_7_days",
            "confidence": "low",
            "num_detections": 0,
            "yolo_detections": [],
        }
    worst = max(yolo_dets, key=lambda d: d.get("severity_score", 0))
    health = max(0, 100 - int(worst.get("severity_score", 0.5) * 100))
    risk = "high" if health < 50 else "medium" if health < 75 else "low"
    diseases = [
        d["class_name"].replace("_", " ").title()
        for d in yolo_dets if d.get("category") == "disease"
    ]
    return {
        "ready": True,
        "health_score": health,
        "risk_level": risk,
        "diseases_found": diseases or ["Possible issue (YOLO)"],
        "visible_symptoms": f"YOLO detected {len(yolo_dets)} objects (LLaVA timed out)",
        "affected_area_pct": int(sum(d.get("area_pct", 0) for d in yolo_dets)),
        "top_issue": worst["class_name"].replace("_", " ").title(),
        "recommendations": [f"Investigate {worst['class_name']}"],
        "urgency": "immediate" if health < 40 else "within_7_days",
        "confidence": "medium",
        "num_detections": len(yolo_dets),
        "yolo_detections": yolo_dets,
    }


def _background_analyze(
    upload_id: str,
    image_bgr: np.ndarray,
    filename: str,
    session_id: str = "",
    job_id: str = "",
):
    """Run Classifier + Reasoning Engine pipeline in a background thread.

    This is now fast (~1-2s) since it uses local models instead of LLaVA.
    No timeout wrapper needed.
    """
    try:
        result = _run_analysis(image_bgr, filename)
    except Exception as exc:
        console.print(f"[bold red]Analysis error:[/bold red] {exc}")
        result = {
            "ready": True,
            "health_score": 0,
            "risk_level": "critical",
            "diseases_found": [f"Error: {exc}"],
            "visible_symptoms": "",
            "affected_area_pct": 0,
            "top_issue": f"Analysis error: {exc}",
            "recommendations": [],
            "urgency": "immediate",
            "confidence": "low",
            "num_detections": 0,
            "yolo_detections": [],
        }

    if job_id:
        with _jobs_lock:
            if job_id in _jobs:
                _jobs[job_id]["status"] = "complete"
                _jobs[job_id]["message"] = "Analysis complete"
                _jobs[job_id]["result"] = result

    _results[upload_id] = result

    # ── Store in global _job_results so /api/status/{job_id} can find it ──
    if job_id:
        yolo_count = result.get("num_detections", 0)
        _diseases = result.get("diseases_found", [])
        _recs = result.get("recommendations", [])
        with _job_lock:
            _job_results[job_id] = {
                "status": "complete",
                "health": int(result.get("health_score", 50)),
                "risk": safe_str(result.get("risk_level", "medium")),
                "disease": safe_str(_diseases[0] if _diseases else "Unknown"),
                "diseases": [safe_str(d) for d in _diseases],
                "symptoms": safe_str(result.get("visible_symptoms", ""))[:500],
                "affected_area": int(result.get("affected_area_pct", 0)),
                "treatment": safe_str(_recs[0] if _recs else ""),
                "all_treatments": [safe_str(t) for t in _recs],
                "urgency": safe_str(result.get("urgency", "within_7_days")),
                "confidence": safe_str(result.get("confidence", "medium")),
                "yolo_detections": int(yolo_count),
            }

    # Update session with analysis results
    if session_id:
        with _sessions_lock:
            sess = _sessions.get(session_id)
            if sess:
                health = result.get("health_score", 0)
                diseases = result.get("diseases_found", [])
                risk = result.get("risk_level", "low")
                recs = result.get("recommendations", [])
                top_rec = ""
                if recs:
                    top_rec = recs[0] if isinstance(recs[0], str) else recs[0].get("action", str(recs[0]))

                # Per-photo record
                photo_record = {
                    "photo_id": upload_id,
                    "taken_at": datetime.now().isoformat(),
                    "filename": filename,
                    "health_score": health,
                    "risk_level": risk,
                    "diseases": diseases,
                    "treatment": top_rec,
                    "visible_symptoms": result.get("visible_symptoms", ""),
                    "zone": sess.get("location", "Unknown"),
                }
                sess["photos"].append(photo_record)
                sess["health_scores"].append(health)
                sess["diseases_found"].extend(diseases)
                sess["last_activity"] = datetime.now().isoformat()

                _add_activity(
                    "analysis",
                    session_id,
                    f"Analysis complete: health {health}/100, risk {risk}"
                    + (f", found {', '.join(diseases[:3])}" if diseases else ""),
                )
                if diseases and top_rec:
                    _add_activity(
                        "treatment",
                        session_id,
                        f"Recommended: {top_rec}",
                    )
                _save_sessions()
                _save_session_file(session_id)

    # ── Detailed terminal output ──
    health = result.get("health_score", 0)
    risk = result.get("risk_level", "unknown")
    diseases = result.get("diseases_found", [])
    symptoms = result.get("visible_symptoms", "")
    affected = result.get("affected_area_pct", 0)
    recs = result.get("recommendations", [])
    urgency = result.get("urgency", "unknown")
    confidence = result.get("confidence", "unknown")
    n_yolo = result.get("num_detections", 0)

    if health >= 70:
        health_color = "green"
    elif health >= 40:
        health_color = "yellow"
    else:
        health_color = "red"

    risk_colors = {"low": "green", "medium": "yellow", "high": "red", "critical": "bold red"}
    risk_color = risk_colors.get(risk, "white")

    console.print()
    console.print(f"[bold green]{'=' * 60}[/bold green]")
    console.print(f"[bold green]  ANALYSIS COMPLETE[/bold green]  [{upload_id}]")
    console.print(f"[bold green]{'=' * 60}[/bold green]")
    console.print(f"   Health: [{health_color}][bold]{health}/100[/bold][/{health_color}]    "
                   f"Risk: [{risk_color}][bold]{risk.upper()}[/bold][/{risk_color}]    "
                   f"Confidence: [bold]{confidence}[/bold]")
    if diseases:
        console.print(f"   Diseases: [bold red]{', '.join(diseases)}[/bold red]")
    else:
        console.print("   Diseases: [green]None detected[/green]")
    if symptoms:
        sym_display = symptoms[:200] + "..." if len(symptoms) > 200 else symptoms
        console.print(f"   Symptoms seen: [italic]{sym_display}[/italic]")
    console.print(f"   Affected area: [bold]{affected}%[/bold]")
    if recs:
        top_rec = recs[0] if isinstance(recs[0], str) else recs[0].get("action", str(recs[0]))
        console.print(f"   Action needed: [bold cyan]{top_rec}[/bold cyan]")
    console.print(f"   Urgency: [bold]{urgency}[/bold]")
    console.print(f"   YOLO detections: {n_yolo}")
    console.print(f"[bold green]{'=' * 60}[/bold green]")
    console.print()


@app.get("/api/result/{upload_id}")
async def get_result(upload_id: str):
    """Poll endpoint — phone checks this until analysis is done."""
    result = _results.get(upload_id)
    if result is None:
        return JSONResponse({"error": "Unknown upload ID"}, status_code=404)
    return result


@app.get("/api/status/{job_id}")
async def get_job_status(job_id: str):
    """Poll endpoint — phone checks this until analysis is done."""
    with _job_lock:
        data = _job_results.get(job_id, {"status": "processing"})
    safe = json.dumps(data, ensure_ascii=True, default=str)
    return Response(content=safe.encode("utf-8"), media_type="application/json")


# ---------------------------------------------------------------------------
# Session API endpoints (consumed by dashboard frontend)
# ---------------------------------------------------------------------------

@app.get("/api/sessions/active")
async def get_active_sessions():
    """Return enriched session data for dashboard multi-device view."""
    with _sessions_lock:
        active = []
        total_photos = 0
        total_diseases = 0
        for s in _sessions.values():
            if s.get("status") != "active":
                continue
            photos_list = s.get("photos", [])
            health_scores = s.get("health_scores", [])
            all_diseases = s.get("diseases_found", [])
            avg_health = (
                round(sum(health_scores) / len(health_scores))
                if health_scores else None
            )
            top_disease = None
            if all_diseases:
                # Most frequently occurring disease
                from collections import Counter
                counts = Counter(d.lower() for d in all_diseases if d)
                if counts:
                    top_disease = counts.most_common(1)[0][0].title()

            # Build per-photo summary with optional thumbnail
            photo_summaries = []
            for p in photos_list[-20:]:  # cap at 20 most recent
                thumb_b64 = _make_thumbnail_b64(p.get("filename", ""))
                photo_summaries.append({
                    "photo_id": p.get("photo_id", ""),
                    "time": p.get("taken_at", ""),
                    "health": p.get("health_score", 0),
                    "disease": ", ".join(p.get("diseases", [])) or "None",
                    "treatment": p.get("treatment", ""),
                    "risk_level": p.get("risk_level", "low"),
                    "thumbnail_b64": thumb_b64,
                })

            worst_risk = "low"
            if any(p.get("risk_level") == "critical" for p in photos_list):
                worst_risk = "critical"
            elif any(p.get("risk_level") == "high" for p in photos_list):
                worst_risk = "high"
            elif any(p.get("risk_level") == "medium" for p in photos_list):
                worst_risk = "medium"

            active.append({
                "session_id": s.get("session_id", ""),
                "user_name": s.get("device_name", "Unknown"),
                "device_name": s.get("device_name", "Unknown"),
                "device": s.get("browser", "Unknown"),
                "browser": s.get("browser", ""),
                "ip": s.get("ip", ""),
                "field_location": s.get("location", ""),
                "location": s.get("location", ""),
                "crop_type": s.get("crop_type", ""),
                "photos_count": s.get("photos_uploaded", 0),
                "photos_uploaded": s.get("photos_uploaded", 0),
                "last_photo_time": s.get("last_activity", ""),
                "last_activity": s.get("last_activity", ""),
                "avg_health": avg_health,
                "health_scores": health_scores,
                "risk_level": worst_risk,
                "top_disease": top_disease,
                "diseases_found": all_diseases,
                "status": "active",
                "photos": photo_summaries,
            })
            total_photos += s.get("photos_uploaded", 0)
            total_diseases += len(set(d.lower() for d in all_diseases if d))

    return {
        "sessions": active,
        "count": len(active),
        "total_active": len(active),
        "total_photos_today": total_photos,
        "total_diseases_found": total_diseases,
    }


def _make_thumbnail_b64(filename: str, max_size: int = 120) -> str:
    """Generate a small base64 JPEG thumbnail for a photo, or empty string."""
    if not filename:
        return ""
    filepath = _output_dir / filename
    if not filepath.is_file():
        return ""
    try:
        img = cv2.imread(str(filepath))
        if img is None:
            return ""
        h, w = img.shape[:2]
        scale = min(max_size / w, max_size / h, 1.0)
        thumb = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        _, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 60])
        return base64.b64encode(buf.tobytes()).decode("utf-8")
    except Exception:
        return ""


@app.get("/api/sessions/photos/{session_id}")
async def get_session_photos(session_id: str):
    """Return all photos + analysis results for one session."""
    with _sessions_lock:
        sess = _sessions.get(session_id)
    if sess is None:
        return JSONResponse({"error": "Session not found"}, status_code=404)
    photos = sess.get("photos", [])
    enriched = []
    for p in photos:
        thumb_b64 = _make_thumbnail_b64(p.get("filename", ""))
        enriched.append({
            "photo_id": p.get("photo_id", ""),
            "taken_at": p.get("taken_at", ""),
            "filename": p.get("filename", ""),
            "health_score": p.get("health_score", 0),
            "risk_level": p.get("risk_level", "low"),
            "diseases": p.get("diseases", []),
            "treatment": p.get("treatment", ""),
            "visible_symptoms": p.get("visible_symptoms", ""),
            "zone": p.get("zone", ""),
            "thumbnail_b64": thumb_b64,
        })
    return {
        "session_id": session_id,
        "device_name": sess.get("device_name", ""),
        "location": sess.get("location", ""),
        "crop_type": sess.get("crop_type", ""),
        "photos_count": len(enriched),
        "photos": enriched,
    }


@app.get("/api/sessions/all")
async def get_all_sessions():
    """Return every session (active + disconnected)."""
    with _sessions_lock:
        all_sessions = list(_sessions.values())
    return {"sessions": all_sessions, "count": len(all_sessions)}


@app.get("/api/sessions/stats")
async def get_session_stats():
    """Aggregate stats for the dashboard stat cards."""
    with _sessions_lock:
        active_count = sum(1 for s in _sessions.values() if s.get("status") == "active")
        total_photos = sum(s.get("photos_uploaded", 0) for s in _sessions.values())
        all_diseases = []
        for s in _sessions.values():
            all_diseases.extend(s.get("diseases_found", []))
        unique_diseases = len(set(d.lower() for d in all_diseases if d))
        unique_locations = len(set(
            s.get("location", "").lower()
            for s in _sessions.values()
            if s.get("location") and s["location"] != "Unknown"
        ))
    return {
        "active_devices": active_count,
        "photos_today": total_photos,
        "diseases_found": unique_diseases,
        "fields_scanned": unique_locations,
    }


@app.get("/api/sessions/feed")
async def get_activity_feed():
    """Return the live activity feed (most recent first)."""
    with _sessions_lock:
        return {"feed": _activity_feed[:50]}


@app.get("/api/sessions/{session_id}")
async def get_session_detail(session_id: str):
    """Return details for a specific session."""
    with _sessions_lock:
        sess = _sessions.get(session_id)
    if sess is None:
        return JSONResponse({"error": "Session not found"}, status_code=404)
    return sess


@app.delete("/api/sessions/{session_id}")
async def disconnect_session(session_id: str):
    """Mark a session as disconnected."""
    with _sessions_lock:
        sess = _sessions.get(session_id)
        if sess is None:
            return JSONResponse({"error": "Session not found"}, status_code=404)
        sess["status"] = "disconnected"
        sess["disconnected_at"] = datetime.now().isoformat()
        _add_activity("disconnect", session_id, f"Session ended: {sess['device_name']}")
        _save_sessions()
    return {"message": "Session disconnected", "session_id": session_id}


@app.get("/api/qr-code")
async def get_qr_code():
    """Return QR code as base64 PNG + server info for dashboard display."""
    ip = _get_local_ip()
    url = f"http://{ip}:{_server_port}"

    qr_img = qrcode.make(url)
    buffer = io.BytesIO()
    qr_img.save(buffer, format="PNG")
    qr_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    with _sessions_lock:
        active_count = sum(1 for s in _sessions.values() if s.get("status") == "active")

    return {
        "qr_code_base64": qr_b64,
        "url": url,
        "server_ip": ip,
        "port": _server_port,
        "connected_devices": active_count,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Wireless phone-to-laptop crop analysis server",
    )
    parser.add_argument("--crop", type=str, default="wheat",
                        help="Crop type for analysis (default: wheat)")
    parser.add_argument("--port", type=int, default=8765,
                        help="Server port (default: 8765)")
    parser.add_argument("--model", type=str,
                        default=str(_PROJECT_ROOT / "models" / "india_agri_cls.pt"),
                        help="Path to YOLO model")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"],
                        help="Compute device (default: cpu)")
    return parser.parse_args()


def main():
    global _model, _cls_model, _cls_names, _crop_type, _server_port

    args = _parse_args()
    _crop_type = args.crop
    _server_port = args.port

    # Add src to path for agridrone imports
    import sys
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

    # Load persisted sessions from disk
    _load_sessions()
    console.print(f"[dim]Loaded {len(_sessions)} previous sessions from disk[/dim]")

    console.print(Panel(
        "[bold cyan]Agri-Drone Phone Connect[/bold cyan]\n"
        f"Crop: [green]{_crop_type}[/green]  •  Device: [green]{args.device}[/green]  •  "
        f"Port: [green]{args.port}[/green]",
        border_style="cyan",
    ))

    # Load YOLO detector model (bounding boxes)
    from ultralytics import YOLO

    console.print("[yellow]Loading YOLO detector model...[/yellow]")
    model_path = Path(args.model)
    if model_path.is_file():
        _model = YOLO(str(model_path), task="segment")
    else:
        _model = YOLO(model_path.stem, task="segment")
    _model.to(args.device)
    console.print("[green]YOLO detector ready.[/green]")

    # Load classifier model (fast disease classification — the main brain)
    console.print("[yellow]Loading crop classifier (india_agri_cls.pt)...[/yellow]")
    cls_path = _PROJECT_ROOT / "models" / "india_agri_cls.pt"
    if cls_path.is_file():
        _cls_model = YOLO(str(cls_path), task="classify")
        _cls_model.to(args.device)
        _cls_names = _cls_model.names if hasattr(_cls_model, "names") else None
        console.print(f"[green]Classifier loaded: {len(_cls_names or {})} classes.[/green]")
    else:
        console.print(f"[yellow]Classifier not found at {cls_path} — will use YOLO detector only[/yellow]")

    # Check LLaVA availability (informational — no longer blocks analysis)
    try:
        models = ollama.list()
        model_names = [m.model for m in models.models] if hasattr(models, "models") else []
        llava_available = any("llava" in str(n).lower() for n in model_names)
        if llava_available:
            console.print("[green]LLaVA available (used as background validator on dashboard).[/green]")
        else:
            console.print("[dim]LLaVA not found — not needed for phone analysis.[/dim]")
    except Exception:
        console.print("[dim]Ollama not reachable — not needed for phone analysis.[/dim]")

    # Determine URL and show QR code
    ip = _get_local_ip()
    url = f"http://{ip}:{args.port}"

    console.print(f"\n[bold]Server starting at:[/bold] [link={url}]{url}[/link]")
    console.print("[dim]Both phone and laptop must be on the same WiFi network.[/dim]")
    _print_qr(url)
    console.print("[bold green]Ready![/bold green] Scan the QR code above with your phone.\n")

    # Start server
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
