"""
detections.py - Detection and segmentation output models.
"""
from datetime import datetime
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Bounding box in normalized or pixel coordinates."""
    x1: float = Field(..., description="Top-left X")
    y1: float = Field(..., description="Top-left Y")
    x2: float = Field(..., description="Bottom-right X")
    y2: float = Field(..., description="Bottom-right Y")
    is_normalized: bool = Field(default=False, description="True if 0-1, False if pixels")

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


class Polygon(BaseModel):
    """Polygon as list of (x, y) coordinate pairs."""
    points: list[tuple[float, float]] = Field(..., description="List of (x, y) vertices")
    is_normalized: bool = Field(default=False)

    @property
    def area(self) -> float:
        """Compute polygon area using shoelace formula."""
        if len(self.points) < 3:
            return 0.0
        area = 0.0
        for i in range(len(self.points)):
            x1, y1 = self.points[i]
            x2, y2 = self.points[(i + 1) % len(self.points)]
            area += x1 * y2 - x2 * y1
        return abs(area) / 2.0

    @property
    def centroid(self) -> tuple[float, float]:
        """Compute polygon centroid."""
        if not self.points:
            return (0.0, 0.0)
        cx = sum(p[0] for p in self.points) / len(self.points)
        cy = sum(p[1] for p in self.points) / len(self.points)
        return (cx, cy)


# Valid agronomic categories for detection classification
VALID_CATEGORIES = ("health", "disease", "weed", "lodging", "nutrient", "stress", "stand")


class Detection(BaseModel):
    """Single detection/hotspot output."""
    detection_id: str = Field(default_factory=lambda: str(uuid4()))
    class_name: str = Field(
        ...,
        description="Class label (e.g. healthy_crop, wheat_lodging, broadleaf_weed, leaf_rust, …)",
    )
    confidence: float = Field(..., description="Confidence score 0-1")
    severity_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Severity score 0.0 (no problem) to 1.0 (critical). "
                    "Defaults to confidence when not explicitly set.",
    )
    category: str = Field(
        default="health",
        description="Agronomic category: health, disease, weed, lodging, nutrient, stress, or stand",
    )
    crop_type: str = Field(
        default="unknown",
        description="Crop type this detection belongs to (e.g. wheat, rice)",
    )
    area_pct: float = Field(
        default=0.0,
        ge=0.0,
        description="Percentage of the source image area covered by this detection",
    )
    uncertainty: Optional[float] = Field(None, description="Uncertainty estimate if available")
    bbox: BoundingBox = Field(..., description="Bounding box coordinates")
    polygon: Optional[Polygon] = Field(None, description="Segmentation mask as polygon")
    source_image: str = Field(..., description="Path or ID of source image")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    model_name: str = Field(default="unknown")
    model_version: str = Field(default="unknown")


class DetectionBatch(BaseModel):
    """Batch of detections from one image."""
    batch_id: str = Field(default_factory=lambda: str(uuid4()))
    source_image: str = Field(...)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    model_name: str
    model_version: str
    detections: list[Detection] = Field(default_factory=list)
    num_detections: int = Field(default=0)
    processing_time_ms: Optional[float] = None
    metadata: dict = Field(default_factory=dict)

    def add_detection(self, detection: Detection) -> None:
        """Add detection to batch."""
        self.detections.append(detection)
        self.num_detections = len(self.detections)

    def filter_by_confidence(self, threshold: float) -> list[Detection]:
        """Return detections above confidence threshold."""
        return [d for d in self.detections if d.confidence >= threshold]

    def filter_by_class(self, class_name: str) -> list[Detection]:
        """Return detections of specific class."""
        return [d for d in self.detections if d.class_name == class_name]

    def filter_by_area(self, min_area: float, max_area: Optional[float] = None) -> list[Detection]:
        """Return detections within area bounds."""
        result = [d for d in self.detections if d.bbox.area >= min_area]
        if max_area is not None:
            result = [d for d in result if d.bbox.area <= max_area]
        return result
