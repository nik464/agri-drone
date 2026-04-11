"""
detection.py - Detection API endpoints.
"""

import base64
import hashlib
import io
import json
from typing import Optional

import cv2
import numpy as np
import requests
from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from loguru import logger

from ...vision import YOLOv8Detector
from ...types.detections import DetectionBatch
from ...core.temporal_tracker import get_tracker
from ... import get_config
from ..schemas import (
    DetectionResponseSchema,
    ErrorResponseSchema,
    HealthCheckSchema,
    ResetResponseSchema,
)

router = APIRouter(prefix="/detect", tags=["detection"])

# Global detector instance (lazy-loaded)
_detector: Optional[YOLOv8Detector] = None


def _draw_detections_on_image(image: np.ndarray, detections) -> np.ndarray:
    """
    Draw bounding boxes on image.

    Args:
        image: Input image (numpy array, BGR)
        detections: List of Detection objects

    Returns:
        Image with drawn bounding boxes
    """
    img_copy = image.copy()

    # Color palette for 12 precision agriculture classes (BGR)
    colors_map = {
        "healthy_crop": (0, 200, 0),          # Green
        "wheat_lodging": (0, 100, 255),       # Orange
        "volunteer_corn": (0, 0, 200),        # Red
        "broadleaf_weed": (0, 0, 255),        # Bright red
        "grass_weed": (50, 50, 200),          # Dark red
        "leaf_rust": (0, 165, 255),           # Orange
        "goss_wilt": (0, 200, 255),           # Yellow-orange
        "fusarium_head_blight": (128, 0, 255),# Purple
        "nitrogen_deficiency": (0, 255, 255), # Yellow
        "water_stress": (255, 200, 0),        # Cyan-blue
        "good_plant_spacing": (200, 200, 0),  # Teal
        "poor_plant_spacing": (255, 0, 200),  # Magenta
    }

    for det in detections:
        # Get bounding box
        x1 = int(det.bbox.x1)
        y1 = int(det.bbox.y1)
        x2 = int(det.bbox.x2)
        y2 = int(det.bbox.y2)

        # Get color (default to white)
        color = colors_map.get(det.class_name, (255, 255, 255))

        # Draw rectangle
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)

        # Draw label
        label = f"{det.class_name} {det.confidence:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        label_y = max(y1 - 5, label_size[1] + 5)

        # Draw label background
        cv2.rectangle(
            img_copy,
            (x1, label_y - label_size[1] - 5),
            (x1 + label_size[0], label_y + 5),
            color,
            -1,
        )

        # Draw label text
        cv2.putText(
            img_copy,
            label,
            (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            1,
        )

    return img_copy


def _image_to_base64(image: np.ndarray) -> str:
    """
    Convert OpenCV image to base64 JPEG string.

    Args:
        image: Input image (numpy array, BGR)

    Returns:
        Base64-encoded JPEG string
    """
    success, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not success:
        raise ValueError("Failed to encode image to JPEG")
    return base64.b64encode(encoded).decode('utf-8')


async def _llava_analyze(image_bgr: np.ndarray, crop_type: str = "wheat") -> Optional[dict]:
    """
    Send image to LLaVA via Ollama for visual disease diagnosis.

    Args:
        image_bgr: Image in BGR format (OpenCV)
        crop_type: Crop type ('wheat' or 'rice')

    Returns:
        Dict with health_score, diseases_found, visible_symptoms, recommendations
        or None if analysis fails
    """
    try:
        # Encode image to base64
        _, jpeg_data = cv2.imencode('.jpg', image_bgr)
        image_base64 = base64.b64encode(jpeg_data).decode('utf-8')

        # Prepare LLaVA prompt based on crop type
        crop_name = crop_type.capitalize() if crop_type else "Wheat"
        llava_prompt = f"""You are an expert plant pathologist specializing in {crop_name} crop diseases. 
        Analyze this {crop_name} field image carefully and provide a JSON response with:
        {{
            "health_score": (0-100 numeric score),
            "diseases_found": "names of detected diseases or 'None detected'",
            "visible_symptoms": "description of visible symptoms",
            "recommendations": "treatment recommendations"
        }}
        Respond ONLY with valid JSON, no markdown or extra text."""

        # Call Ollama LLaVA via HTTP
        ollama_url = "http://localhost:11434/api/generate"
        request_payload = {
            "model": "llava",
            "prompt": llava_prompt,
            "images": [image_base64],
            "stream": False,
        }

        response = requests.post(ollama_url, json=request_payload, timeout=60)
        if response.status_code != 200:
            logger.warning(f"Ollama returned status {response.status_code}")
            return None

        result = response.json()
        raw_text = result.get("response", "").strip()

        # Try to parse JSON from response
        try:
            # Remove markdown code fences if present
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            if raw_text.startswith("```"):
                raw_text = raw_text[3:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]
            
            analysis = json.loads(raw_text)
            logger.info(f"LLaVA analysis: health_score={analysis.get('health_score')}")
            return analysis
        except json.JSONDecodeError:
            logger.warning(f"Could not parse LLaVA response as JSON: {raw_text[:100]}")
            return None

    except requests.exceptions.RequestException as e:
        logger.warning(f"LLaVA analysis failed (Ollama unavailable?): {e}")
        return None
    except Exception as e:
        logger.warning(f"LLaVA analysis error: {e}")
        return None


def get_detector() -> YOLOv8Detector:
    """
    Get or initialize the global YOLOv8Detector instance.

    Returns:
        Initialized YOLOv8Detector

    Raises:
        HTTPException: If detector initialization fails
    """
    global _detector

    if _detector is None:
        try:
            config = get_config()
            model_name = config.get("model.backbone", "yolov8n-seg")
            model_path = config.get("model.checkpoint", "models/yolov8n-seg.pt")
            device = config.get_env().device

            logger.info(f"Initializing YOLOv8Detector: model={model_name}, device={device}")
            _detector = YOLOv8Detector(model_name, model_path, device=device)
            logger.info("YOLOv8Detector initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize detector: {e}")
            raise HTTPException(status_code=500, detail=f"Detector initialization failed: {str(e)}")

    return _detector


@router.post("/", response_model=DetectionResponseSchema)
async def detect_image(
    file: UploadFile = File(..., description="Image file (JPG, PNG, BMP)"),
    confidence_threshold: float = Query(0.3, ge=0.0, le=1.0, description="Detection confidence threshold [0.0-1.0]"),
    include_image: bool = Query(True, description="Include annotated image in response (base64)"),
    crop_type: str = Query("wheat", description="Crop type: 'wheat' or 'rice'"),
    use_llava: bool = Query(False, description="Enable LLaVA vision analysis"),
    lat: Optional[float] = Query(None, ge=-90, le=90, description="GPS latitude for zone tracking"),
    lng: Optional[float] = Query(None, ge=-180, le=180, description="GPS longitude for zone tracking"),
) -> DetectionResponseSchema:
    """
    Run hotspot detection on uploaded image.

    **Endpoint**: `POST /api/detect/`

    **Description**:
    Analyzes an uploaded field image to detect crop stress hotspots
    (weeds, disease, pest damage). Returns detection results with
    bounding boxes, confidence scores, and optional annotated image.
    Optionally includes LLaVA AI vision analysis for disease diagnosis.

    **Parameters**:
    - `file`: Image file (multipart/form-data, required)
    - `confidence_threshold`: Minimum confidence [0.0-1.0] (default: 0.3)
    - `include_image`: Include annotated image in response (default: true)
    - `crop_type`: Crop type 'wheat' or 'rice' (default: wheat)
    - `use_llava`: Enable LLaVA vision analysis (default: false)

    **Returns**:
    - `status`: "success" or "error"
    - `batch_id`: Unique batch identifier
    - `source_image`: Original filename
    - `num_detections`: Number of detections found
    - `processing_time_ms`: Inference time in milliseconds
    - `detections`: List of detection objects with bounding boxes
    - `annotated_image_base64`: Annotated image as base64 JPEG (optional)
    - `llava_analysis`: LLaVA AI vision analysis (optional, if use_llava=true)
    - `metadata`: Additional information

    **Raises**:
    - 400: Invalid or empty image file
    - 500: Detection processing error
    """
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")

        # Read uploaded file into memory
        contents = await file.read()

        if not contents:
            raise HTTPException(status_code=400, detail="Empty file")

        # Convert bytes to numpy array
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(
                status_code=400,
                detail="Failed to decode image. Ensure file is a valid image format (JPG, PNG, BMP)"
            )

        # Downsize large images for faster inference
        h, w = image.shape[:2]
        max_dim = 1280
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            logger.info(f"Resized image from {w}x{h} to {image.shape[1]}x{image.shape[0]}")

        # Log detection request
        logger.info(
            f"Detection request | file={file.filename} | shape={image.shape} | "
            f"confidence_threshold={confidence_threshold} | include_image={include_image}"
        )

        # Get detector instance
        detector = get_detector()

        # Run detection
        batch = detector.detect(image, confidence_threshold=confidence_threshold)

        if batch is None:
            raise HTTPException(status_code=500, detail="Detection returned no results")

        # Log results
        logger.info(
            f"Detection complete | detections={batch.num_detections} | "
            f"processing_time_ms={batch.processing_time_ms or 0:.1f} | batch_id={batch.batch_id}"
        )

        # Draw annotations on image if requested
        annotated_image_base64 = None
        if include_image and batch.num_detections > 0:
            try:
                annotated_image = _draw_detections_on_image(image, batch.detections)
                annotated_image_base64 = _image_to_base64(annotated_image)
                logger.debug(f"Annotated image encoded to base64: {len(annotated_image_base64)} chars")
            except Exception as e:
                logger.warning(f"Failed to annotate image: {e}")

        # Optionally run LLaVA vision analysis
        llava_analysis = None
        if use_llava:
            logger.info("Running LLaVA vision analysis...")
            llava_analysis = await _llava_analyze(image, crop_type)
            if llava_analysis:
                logger.info(f"LLaVA analysis complete: {llava_analysis}")

        # ── Disease progression tracking ──
        progression_data = None
        if lat is not None and lng is not None:
            try:
                tracker = get_tracker()
                zone_id = tracker.gps_to_zone(lat, lng)
                # Pick top detection for recording
                top_det = batch.detections[0] if batch.detections else None
                disease_name = top_det.class_name if top_det else "unknown"
                det_confidence = float(top_det.confidence) if top_det else 0.0
                det_severity = float(top_det.severity_score) if top_det else 0.0
                img_hash = hashlib.sha256(contents).hexdigest()[:16]

                tracker.record(
                    zone_id=zone_id,
                    disease=disease_name,
                    confidence=det_confidence,
                    severity=det_severity,
                    image_hash=img_hash,
                )
                prog = tracker.analyze_progression(zone_id)
                progression_data = {
                    "zone_id": prog.zone_id,
                    "spread_rate": prog.spread_rate,
                    "trend": prog.trend,
                    "days_since_first_detection": prog.days_since_first_detection,
                    "urgency_override": prog.urgency_override,
                    "readings_count": len(prog.history),
                }
                logger.info(
                    f"Progression: zone={zone_id} trend={prog.trend} "
                    f"spread_rate={prog.spread_rate}%/day"
                )
            except Exception as e:
                logger.warning(f"Progression tracking failed (non-fatal): {e}")

        # Build response using schema
        response = DetectionResponseSchema(
            status="success",
            batch_id=batch.batch_id,
            source_image=file.filename,
            num_detections=batch.num_detections,
            processing_time_ms=batch.processing_time_ms or 0.0,
            detections=[
                {
                    "id": det.detection_id,
                    "class_name": det.class_name,
                    "confidence": float(det.confidence),
                    "severity_score": float(det.severity_score),
                    "category": det.category,
                    "area_pct": float(det.area_pct),
                    "bbox": {
                        "x1": float(det.bbox.x1),
                        "y1": float(det.bbox.y1),
                        "x2": float(det.bbox.x2),
                        "y2": float(det.bbox.y2),
                        "width": float(det.bbox.width),
                        "height": float(det.bbox.height),
                        "area": float(det.bbox.area),
                    },
                    "polygon": [
                        [float(x), float(y)] for x, y in det.polygon.points
                    ] if det.polygon else None,
                    "timestamp": det.timestamp.isoformat() if det.timestamp else None,
                }
                for det in batch.detections
            ],
            annotated_image_base64=annotated_image_base64,
            llava_analysis=llava_analysis,
            metadata={**(batch.metadata or {}), **({"progression": progression_data} if progression_data else {})},
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error during detection: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


@router.get("/health", response_model=HealthCheckSchema)
async def detector_health() -> HealthCheckSchema:
    """
    Check detector health and status.

    Returns detector availability and model information.
    """
    try:
        detector = get_detector()
        return HealthCheckSchema(
            status="healthy",
            detector_loaded=True,
            model_name=detector.model_name,
            device=detector.device,
        )
    except Exception as e:
        logger.warning(f"Detector health check failed: {e}")
        return HealthCheckSchema(
            status="unhealthy",
            detector_loaded=False,
            error=str(e),
        )


@router.post("/reset", response_model=ResetResponseSchema)
async def reset_detector() -> ResetResponseSchema:
    """
    Reset the detector (unload model from memory).

    Useful for freeing GPU memory between requests.
    """
    global _detector

    try:
        if _detector is not None:
            logger.info("Resetting detector")
            _detector = None
            return ResetResponseSchema(
                status="success",
                message="Detector reset successfully"
            )
        else:
            return ResetResponseSchema(
                status="already_reset",
                message="Detector was not loaded"
            )
    except Exception as e:
        logger.error(f"Failed to reset detector: {e}")
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")
