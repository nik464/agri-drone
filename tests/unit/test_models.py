"""
Unit tests for data models.
"""
import pytest
from agridrone.types import (
    Detection, BoundingBox, DetectionBatch,
    GridCell, PrescriptionMap, GeoCoordinate,
)


@pytest.mark.unit
def test_bounding_box_properties():
    """Test bounding box calculations."""
    bbox = BoundingBox(x1=100, y1=100, x2=200, y2=200)

    assert bbox.width == 100
    assert bbox.height == 100
    assert bbox.area == 10000
    assert bbox.center == (150, 150)


@pytest.mark.unit
def test_detection_batch_filtering(sample_detection):
    """Test detection batch filtering."""
    batch = DetectionBatch(
        source_image="test.jpg",
        model_name="test",
        model_version="1.0",
    )

    det1 = sample_detection
    det1.confidence = 0.9

    det2 = Detection(
        class_name="disease",
        confidence=0.3,
        bbox=BoundingBox(x1=50, y1=50, x2=100, y2=100),
        source_image="test.jpg",
    )

    batch.add_detection(det1)
    batch.add_detection(det2)

    # Filter by confidence
    high_conf = batch.filter_by_confidence(0.5)
    assert len(high_conf) == 1
    assert high_conf[0].class_name == "weed"

    # Filter by class
    diseases = batch.filter_by_class("disease")
    assert len(diseases) == 1
    assert diseases[0].class_name == "disease"


@pytest.mark.unit
def test_grid_cell_creation():
    """Test grid cell creation."""
    cell = GridCell(
        row=0,
        col=0,
        geometry_wkt="POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))",
        center=GeoCoordinate(x=5, y=5),
        area_m2=100.0,
    )

    assert cell.row == 0
    assert cell.col == 0
    assert cell.area_m2 == 100.0
    assert cell.recommended_action == "none"
    assert cell.spray_rate == 0.0
