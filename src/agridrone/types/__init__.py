"""
types - Shared Pydantic models and data contracts.
"""

from .actuation import (
    ActuationEvent,
    ActuationLog,
    ActuationStatus,
    SafetyCheckResult,
    SafetyReport,
    SafetyState,
)
from .detections import BoundingBox, Detection, DetectionBatch, Polygon
from .mapdata import ActuationPlan, GeoCoordinate, GridCell, PrescriptionMap
from .mission import CameraFrame, GNSSData, MissionLog, MissionMetadata

__all__ = [
    # Mission
    "MissionLog",
    "MissionMetadata",
    "CameraFrame",
    "GNSSData",
    # Detections
    "Detection",
    "DetectionBatch",
    "BoundingBox",
    "Polygon",
    # Map data
    "PrescriptionMap",
    "GridCell",
    "GeoCoordinate",
    "ActuationPlan",
    # Actuation
    "ActuationEvent",
    "ActuationLog",
    "ActuationStatus",
    "SafetyState",
    "SafetyCheckResult",
    "SafetyReport",
]
