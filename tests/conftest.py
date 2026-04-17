"""
conftest.py - Pytest fixtures and configuration.
"""
import os
import pytest
from pathlib import Path
from datetime import datetime

from agridrone.types import (
    MissionMetadata, MissionLog, GNSSData, CameraFrame,
    Detection, DetectionBatch, BoundingBox,
    GridCell, PrescriptionMap, GeoCoordinate,
)


def _cuda_available() -> bool:
    """Return True iff a usable CUDA device is visible to the process.

    Honours the conventional ``CUDA_VISIBLE_DEVICES=""`` opt-out even when
    torch would otherwise report a device, so CI runners can force CPU.
    """
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) == "":
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests marked ``gpu`` on CPU-only hosts."""
    if _cuda_available():
        return
    skip_gpu = pytest.mark.skip(reason="no CUDA device available (gpu-marked test)")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)


@pytest.fixture
def mission_log():
    """Create a test mission log."""
    metadata = MissionMetadata(
        mission_name="test_mission",
        field_location=GNSSData(latitude=45.5, longitude=-122.5, altitude_m=50),
        flight_altitude_m=50,
    )
    return MissionLog(metadata=metadata)


@pytest.fixture
def sample_image_sizes():
    """Sample image dimensions."""
    return (640, 480)  # width, height


@pytest.fixture
def sample_detection():
    """Create a sample detection."""
    return Detection(
        class_name="weed",
        confidence=0.85,
        bbox=BoundingBox(x1=100, y1=100, x2=200, y2=200),
        source_image="test_image.jpg",
    )


@pytest.fixture
def sample_detection_batch(sample_detection):
    """Create a sample detection batch."""
    batch = DetectionBatch(
        source_image="test_image.jpg",
        model_name="yolov8n-seg",
        model_version="8.0",
    )
    batch.add_detection(sample_detection)
    return batch


@pytest.fixture
def sample_grid_cell():
    """Create a sample grid cell."""
    return GridCell(
        row=0,
        col=0,
        geometry_wkt="POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))",
        center=GeoCoordinate(x=5, y=5),
        area_m2=100.0,
    )


@pytest.fixture
def sample_prescription_map():
    """Create a sample prescription map."""
    return PrescriptionMap(
        field_center=GeoCoordinate(x=45.5, y=-122.5),
    )
