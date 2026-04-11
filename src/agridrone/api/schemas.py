"""
schemas.py - Pydantic request/response models for API endpoints.

Provides type-safe data contracts for all API operations.
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class BoundingBoxSchema(BaseModel):
    """Bounding box coordinates and dimensions."""

    x1: float = Field(..., description="Top-left X coordinate (pixels)")
    y1: float = Field(..., description="Top-left Y coordinate (pixels)")
    x2: float = Field(..., description="Bottom-right X coordinate (pixels)")
    y2: float = Field(..., description="Bottom-right Y coordinate (pixels)")
    width: float = Field(..., description="Box width (pixels)")
    height: float = Field(..., description="Box height (pixels)")
    area: float = Field(..., description="Box area (pixels²)")

    class Config:
        json_schema_extra = {
            "example": {
                "x1": 100.5,
                "y1": 150.3,
                "x2": 250.8,
                "y2": 300.2,
                "width": 150.3,
                "height": 149.9,
                "area": 22515.0,
            }
        }


class DetectionSchema(BaseModel):
    """Single hotspot detection result."""

    id: str = Field(..., description="Unique detection identifier")
    class_name: str = Field(
        ...,
        description="Detection class (e.g., 'broadleaf_weed', 'leaf_rust', 'wheat_lodging')",
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score [0.0-1.0]")
    severity_score: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Severity score [0.0-1.0]: 0 = no problem, 1 = critical",
    )
    category: str = Field(
        "health",
        description="Agronomic category: health, disease, weed, lodging, nutrient, stress, or stand",
    )
    area_pct: float = Field(
        0.0,
        ge=0.0,
        description="Percentage of image area covered by this detection",
    )
    bbox: BoundingBoxSchema = Field(..., description="Bounding box")
    polygon: Optional[List[List[float]]] = Field(
        None, description="Polygon coordinates [[x1,y1], [x2,y2], ...]"
    )
    timestamp: Optional[datetime] = Field(None, description="Detection timestamp (ISO-8601)")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "det_001",
                "class_name": "broadleaf_weed",
                "confidence": 0.87,
                "severity_score": 0.75,
                "category": "weed",
                "area_pct": 2.34,
                "bbox": {
                    "x1": 100.5,
                    "y1": 150.3,
                    "x2": 250.8,
                    "y2": 300.2,
                    "width": 150.3,
                    "height": 149.9,
                    "area": 22515.0,
                },
                "polygon": [[100.5, 150.3], [250.8, 150.3], [250.8, 300.2], [100.5, 300.2]],
                "timestamp": "2024-03-18T10:30:45.123456",
            }
        }


class DetectionResponseSchema(BaseModel):
    """Successful detection API response."""

    status: str = Field("success", description="Response status")
    batch_id: str = Field(..., description="Unique batch identifier")
    source_image: str = Field(..., description="Source image filename")
    num_detections: int = Field(..., ge=0, description="Number of detections found")
    processing_time_ms: float = Field(..., description="Inference time (milliseconds)")
    detections: List[DetectionSchema] = Field(..., description="List of detections")
    annotated_image_base64: Optional[str] = Field(
        None, description="Annotated image as base64-encoded JPEG (optional)"
    )
    llava_analysis: Optional[dict] = Field(
        None, description="LLaVA AI vision analysis (optional, includes health_score, diseases_found, visible_symptoms, recommendations)"
    )
    metadata: dict = Field(default_factory=dict, description="Additional metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "batch_id": "batch_abc123",
                "source_image": "field_001.jpg",
                "num_detections": 3,
                "processing_time_ms": 245.3,
                "detections": [
                    {
                        "id": "det_001",
                        "class_name": "weed",
                        "confidence": 0.87,
                        "severity_score": 0.75,
                        "bbox": {
                            "x1": 100.5,
                            "y1": 150.3,
                            "x2": 250.8,
                            "y2": 300.2,
                            "width": 150.3,
                            "height": 149.9,
                            "area": 22515.0,
                        },
                        "polygon": None,
                        "timestamp": "2024-03-18T10:30:45.123456",
                    }
                ],
                "annotated_image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
                "metadata": {},
            }
        }


class ErrorResponseSchema(BaseModel):
    """Error response schema."""

    status: str = Field("error", description="Response status")
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "error",
                "error": "Invalid image format",
                "detail": "File must be JPEG, PNG, or BMP",
            }
        }


class HealthCheckSchema(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    detector_loaded: bool = Field(..., description="Is detector model loaded")
    model_name: Optional[str] = Field(None, description="Model name")
    device: Optional[str] = Field(None, description="Compute device (cpu/cuda)")
    error: Optional[str] = Field(None, description="Error message if unhealthy")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "detector_loaded": True,
                "model_name": "yolov8n-seg",
                "device": "cuda:0",
                "error": None,
            }
        }


class ResetResponseSchema(BaseModel):
    """Reset detector response."""

    status: str = Field(..., description="Reset status (success/already_reset)")
    message: str = Field(..., description="Status message")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "message": "Detector reset successfully",
            }
        }
