"""
stream.py - Live drone camera streaming via WebSocket + REST fallback.

Upgraded pipeline:  YOLO detection → Feature extraction → Rule engine →
Fused scoring → Smart LLaVA trigger  (all per-frame, <50 ms for rules).
"""
import asyncio
import base64
import collections
import hashlib
import json
import time
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from loguru import logger

from .detection import _draw_detections_on_image, _image_to_base64, get_detector

# Reasoning pipeline imports
from ...knowledge import kb_loader
from ...vision.feature_extractor import extract_features, ImageFeatures
from ...vision.rule_engine import evaluate as rule_evaluate, RuleEngineResult

router = APIRouter(prefix="/stream", tags=["stream"])

# Rolling history of the last N frame results for trend tracking
_HISTORY_SIZE = 30
_frame_history: collections.deque[dict] = collections.deque(maxlen=_HISTORY_SIZE)
_frame_counter: int = 0

# ── Smart LLaVA trigger state ──
_last_disease: str | None = None
_last_fused_score: float = 0.0
_llava_in_flight: bool = False          # True while a background LLaVA call is running
_llava_last_result: dict | None = None  # Most recent LLaVA verdict (persisted across frames)

# ── KB cache (loaded once) ──
_kb_profiles: dict | None = None


def _ensure_kb() -> dict:
    """Load knowledge-base profiles once and cache."""
    global _kb_profiles
    if _kb_profiles is None:
        kb_loader.load()
        _kb_profiles = kb_loader.get_all_profiles()
    return _kb_profiles


def _decode_base64_frame(data: str) -> np.ndarray:
    """Decode a base64-encoded JPEG string to a BGR numpy image.

    Also resizes oversized frames to 640px max-side for speed.
    """
    # Strip optional data-URL header (e.g. "data:image/jpeg;base64,...")
    if "," in data:
        data = data.split(",", 1)[1]

    raw = base64.b64decode(data)
    arr = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode base64 frame as JPEG")

    # Downscale large frames to keep feature extraction fast
    h, w = image.shape[:2]
    max_side = 640
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    return image


def _image_hash(image: np.ndarray) -> str:
    """Fast perceptual hash: MD5 of 32×32 grayscale thumbnail."""
    tiny = cv2.resize(image, (32, 32))
    gray = cv2.cvtColor(tiny, cv2.COLOR_BGR2GRAY)
    return hashlib.md5(gray.tobytes()).hexdigest()


def _should_trigger_llava(disease: str, fused_score: float) -> bool:
    """Decide whether this frame warrants a (slow) LLaVA call.

    Triggers when:
      - Top disease changed from last frame, OR
      - Fused confidence jumped by >15% since last LLaVA run
    Skips when a LLaVA call is already in flight.
    """
    global _last_disease, _last_fused_score, _llava_in_flight

    if _llava_in_flight:
        return False

    disease_changed = disease != _last_disease and _last_disease is not None
    score_jumped = abs(fused_score - _last_fused_score) > 0.15

    return disease_changed or score_jumped


async def _run_llava_background(image: np.ndarray, disease: str, fused_score: float):
    """Fire-and-forget LLaVA validation in a background thread."""
    global _llava_in_flight, _llava_last_result, _last_disease, _last_fused_score

    _llava_in_flight = True
    _last_disease = disease
    _last_fused_score = fused_score

    try:
        # Import here to avoid circular dependency at module load
        from ..app import _llava_analyze_sync

        prompt = (
            f"Our real-time system detected '{disease}' with {fused_score:.0%} confidence. "
            "Look at this crop image and reply with a short JSON: "
            '{"agrees": true/false, "brief": "<one sentence>"}.'
        )
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _llava_analyze_sync, image, prompt)
        _llava_last_result = result
        logger.info(f"Stream LLaVA done: {result.get('raw', '')[:80] if result else 'None'}")
    except Exception as exc:
        logger.warning(f"Stream LLaVA failed: {exc}")
        _llava_last_result = None
    finally:
        _llava_in_flight = False


def _process_frame(image: np.ndarray, frame_id: int) -> dict:
    """
    Run detection + reasoning on a single frame.

    Pipeline (target <50 ms for steps 1-4 on CPU):
      1. YOLO detection  → bounding boxes + class predictions
      2. Feature extraction  → ImageFeatures (color, texture, spatial)
      3. Rule engine  → per-disease scores, conflict resolution
      4. Score fusion  → YOLO 60 % + Rules 40 % (streaming blend)
      5. Smart LLaVA trigger  → only on significant changes (async)

    Returns enriched dict with disease, reasoning_summary, fused_score, etc.
    """
    global _last_disease, _last_fused_score

    detector = get_detector()
    kb = _ensure_kb()

    t0 = time.time()

    # ── Step 1: YOLO detection ──
    batch = detector.detect(image, confidence_threshold=0.5)
    t_yolo = time.time()

    # Annotate image
    if batch.num_detections > 0:
        annotated = _draw_detections_on_image(image, batch.detections)
        annotated_b64 = _image_to_base64(annotated)
    else:
        annotated_b64 = _image_to_base64(image)

    # Per-category stats
    by_category: dict[str, int] = {}
    for det in batch.detections:
        cat = getattr(det, "category", det.class_name)
        by_category[cat] = by_category.get(cat, 0) + 1

    # Serialise detections list (always returned)
    detections_list = [
        {
            "id": det.detection_id,
            "class_name": det.class_name,
            "confidence": round(float(det.confidence), 4),
            "severity_score": round(float(det.severity_score), 4),
            "category": getattr(det, "category", ""),
            "area_pct": round(float(getattr(det, "area_pct", 0.0)), 4),
            "bbox": {
                "x1": float(det.bbox.x1),
                "y1": float(det.bbox.y1),
                "x2": float(det.bbox.x2),
                "y2": float(det.bbox.y2),
            },
        }
        for det in batch.detections
    ]

    # ── Find top YOLO detection ──
    top_det = max(batch.detections, key=lambda d: d.confidence) if batch.detections else None
    yolo_class = top_det.class_name if top_det else "healthy"
    yolo_conf = float(top_det.confidence) if top_det else 0.0

    # ── Steps 2-4: Reasoning (only when YOLO is confident enough) ──
    reasoning_summary = "No significant detection"
    rule_score = 0.0
    fused_score = yolo_conf
    disease = yolo_class
    ran_rules = False

    if yolo_conf > 0.65 and kb:
        try:
            # Step 2 — Feature extraction (fast: ~10-20 ms)
            features = extract_features(image, kb)

            # Step 3 — Rule engine (fast: ~5-15 ms)
            classifier_result = {
                "top5": [
                    {"class_key": det.class_name, "confidence": float(det.confidence)}
                    for det in sorted(batch.detections, key=lambda d: -d.confidence)[:5]
                ]
            }
            crop_type = "wheat"  # default; could be inferred or passed by client
            engine_result: RuleEngineResult = rule_evaluate(features, classifier_result, crop_type)

            # Step 4 — Fused score: YOLO 60% + Rules 40% (faster streaming blend)
            if engine_result.candidates:
                top_cand = engine_result.candidates[0]
                rule_score = round(max(0.0, min(1.0, top_cand.rule_score)), 4)
                fused_score = round(yolo_conf * 0.6 + rule_score * 0.4, 4)
                disease = top_cand.disease_name

                # Conflict indicator
                if engine_result.conflict and engine_result.conflict.winner == "rules":
                    disease = engine_result.conflict.rule_prediction
                    reasoning_summary = (
                        f"Rules override YOLO: {disease} "
                        f"(visual evidence {rule_score:.0%} vs YOLO {yolo_conf:.0%})"
                    )
                elif engine_result.conflict and engine_result.conflict.winner == "yolo":
                    reasoning_summary = (
                        f"YOLO confident: {disease} "
                        f"(YOLO {yolo_conf:.0%}, rules {rule_score:.0%})"
                    )
                else:
                    reasoning_summary = (
                        f"YOLO+Rules agree: {disease} "
                        f"(fused {fused_score:.0%})"
                    )

                ran_rules = True
            else:
                reasoning_summary = f"YOLO: {yolo_class} ({yolo_conf:.0%}), rules found no candidates"

        except Exception as exc:
            logger.warning(f"Frame {frame_id} rule engine error (non-fatal): {exc}")
            reasoning_summary = f"YOLO only: {yolo_class} ({yolo_conf:.0%}) — rules skipped"

    elif yolo_conf > 0.0:
        reasoning_summary = f"YOLO low conf: {yolo_class} ({yolo_conf:.0%}) — rules skipped"

    t_rules = time.time()

    # ── Step 5: Smart LLaVA trigger (async, never blocks the frame) ──
    llava_ran = False
    if ran_rules and _should_trigger_llava(disease, fused_score):
        llava_ran = True
        asyncio.ensure_future(_run_llava_background(image.copy(), disease, fused_score))

    processing_ms = round((time.time() - t0) * 1000, 2)
    yolo_ms = round((t_yolo - t0) * 1000, 2)
    rules_ms = round((t_rules - t_yolo) * 1000, 2)

    result = {
        "frame_id": frame_id,
        # ── Core enriched fields ──
        "disease": disease,
        "confidence": round(yolo_conf, 4),
        "rule_score": rule_score,
        "fused_score": fused_score,
        "reasoning_summary": reasoning_summary,
        "llava_ran": llava_ran,
        "llava_last": (
            _llava_last_result.get("raw", "")[:200]
            if _llava_last_result else None
        ),
        # ── Existing fields (backward-compatible) ──
        "detections": detections_list,
        "annotated_frame_b64": annotated_b64,
        "stats": {
            "total": batch.num_detections,
            "by_category": by_category,
        },
        "processing_ms": processing_ms,
        "timing": {
            "yolo_ms": yolo_ms,
            "rules_ms": rules_ms,
        },
    }

    _frame_history.append(result)
    return result


# ---------------------------------------------------------------------------
# WebSocket — live stream
# ---------------------------------------------------------------------------


@router.websocket("/live")
async def live_stream(ws: WebSocket):
    """
    Accept base64-encoded JPEG frames over WebSocket and return
    detection results as JSON for each frame.
    """
    await ws.accept()
    global _frame_counter
    logger.info("WebSocket stream connected")

    try:
        while True:
            data = await ws.receive_text()

            try:
                image = _decode_base64_frame(data)
            except Exception as e:
                await ws.send_json({"error": f"Bad frame: {e}"})
                continue

            _frame_counter += 1
            try:
                result = _process_frame(image, _frame_counter)
                await ws.send_json(result)
            except Exception as e:
                logger.error(f"Frame {_frame_counter} processing error: {e}")
                await ws.send_json(
                    {"frame_id": _frame_counter, "error": f"Processing failed: {e}"}
                )

    except WebSocketDisconnect:
        logger.info("WebSocket stream disconnected")


# ---------------------------------------------------------------------------
# WebSocket — server-side video / YouTube streaming
# ---------------------------------------------------------------------------


@router.websocket("/video")
async def video_stream(ws: WebSocket):
    """
    Server-side video streaming endpoint.

    Accepts a JSON config message: {"source": "<url>", "type": "youtube"|"file"}
    For YouTube URLs, extracts the real stream URL using yt-dlp.
    Reads frames with OpenCV, runs YOLO detection, and pushes annotated
    results back as JSON (same schema as /live).
    """
    await ws.accept()
    global _frame_counter
    logger.info("Video stream WebSocket connected")

    try:
        raw = await ws.receive_text()
        config = json.loads(raw)
        source = config.get("source", "").strip()
        source_type = config.get("type", "file")

        if not source:
            await ws.send_json({"error": "No source provided"})
            return

        if source_type == "youtube":
            try:
                import yt_dlp  # noqa: PLC0415
                ydl_opts = {
                    "format": "best[ext=mp4]/best",
                    "quiet": True,
                    "no_warnings": True,
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(source, download=False)
                    source = info["url"]
                logger.info("Resolved YouTube URL for streaming")
            except ImportError:
                await ws.send_json({"error": "yt-dlp not installed. Run: pip install yt-dlp"})
                return
            except Exception as exc:
                await ws.send_json({"error": f"YouTube URL extraction failed: {exc}"})
                return

        loop = asyncio.get_event_loop()
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            await ws.send_json({"error": "Cannot open video source."})
            return

        logger.info(f"Video capture opened: {source[:80]}")
        try:
            while True:
                ret, frame = await loop.run_in_executor(None, cap.read)
                if not ret:
                    logger.info("Video stream ended (no more frames)")
                    break

                frame = cv2.resize(frame, (640, 480))
                _frame_counter += 1
                try:
                    result = _process_frame(frame, _frame_counter)
                    await ws.send_json(result)
                except Exception as exc:
                    logger.error(f"Video frame {_frame_counter} processing error: {exc}")
                    await ws.send_json({"frame_id": _frame_counter, "error": str(exc)})

                await asyncio.sleep(0.1)  # ~10 fps
        finally:
            cap.release()

    except WebSocketDisconnect:
        logger.info("Video stream WebSocket disconnected")
    except Exception as exc:
        logger.error(f"Video stream unexpected error: {exc}", exc_info=True)


# ---------------------------------------------------------------------------
# REST fallback — single frame upload
# ---------------------------------------------------------------------------


@router.post("/frame")
async def process_single_frame(
    file: UploadFile = File(..., description="JPEG image from drone camera"),
):
    """
    REST fallback for environments where WebSocket is unavailable.

    Accepts a single image upload and returns the same detection payload
    as the WebSocket endpoint.
    """
    global _frame_counter

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")

    arr = np.frombuffer(contents, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Failed to decode image. Ensure file is a valid JPEG/PNG.",
        )

    _frame_counter += 1
    try:
        return _process_frame(image, _frame_counter)
    except Exception as e:
        logger.error(f"Frame processing error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")
