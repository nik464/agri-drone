"""
test_yolov8_detector.py - Tests for YOLOv8 detection inference.
"""
import pytest
import numpy as np
from pathlib import Path

from agridrone.vision.infer import YOLOv8Detector, HotspotDetector
from agridrone.vision.postprocess import DetectionPostProcessor
from agridrone.types import BoundingBox, Detection, DetectionBatch


@pytest.mark.unit
def test_hotspot_detector_base_class():
    """Test base HotspotDetector class.

    Device resolution is CPU-safe: when CUDA is unavailable, the detector must
    fall back to ``cpu``. We therefore assert against the actually-available
    device rather than hard-coding ``cuda``.
    """
    import torch

    expected = "cuda" if torch.cuda.is_available() else "cpu"
    detector = HotspotDetector("test_model", Path("test.pt"))
    assert detector.model_name == "test_model"
    assert detector.device == expected


@pytest.mark.unit
@pytest.mark.gpu
def test_hotspot_detector_cuda_when_available():
    """Sanity check: when running on a GPU host the detector picks CUDA."""
    detector = HotspotDetector("test_model", Path("test.pt"), device="cuda")
    assert detector.device == "cuda"


@pytest.mark.unit
def test_hotspot_detector_detect_not_implemented():
    """Test that base detect method raises NotImplementedError."""
    detector = HotspotDetector("test_model", Path("test.pt"))
    image = np.zeros((480, 640, 3), dtype=np.uint8)

    with pytest.raises(NotImplementedError):
        detector.detect(image)


@pytest.mark.unit
def test_detection_postprocessor_filter_by_confidence():
    """Test filtering detections by confidence."""
    batch = DetectionBatch(
        source_image="test.jpg",
        model_name="test",
        model_version="1.0",
    )

    # Add detections with varying confidence
    for conf in [0.9, 0.7, 0.4, 0.2]:
        det = Detection(
            class_name="weed",
            confidence=conf,
            bbox=BoundingBox(x1=10, y1=10, x2=110, y2=110),
            source_image="test.jpg",
        )
        batch.add_detection(det)

    # Filter with confidence threshold 0.5
    filtered = DetectionPostProcessor.filter_batch(batch, min_confidence=0.5)

    assert filtered.num_detections == 2  # Only 0.9 and 0.7
    assert all(d.confidence >= 0.5 for d in filtered.detections)


@pytest.mark.unit
def test_detection_postprocessor_filter_by_area():
    """Test filtering detections by bounding box area."""
    batch = DetectionBatch(
        source_image="test.jpg",
        model_name="test",
        model_version="1.0",
    )

    # Add detections with varying sizes
    sizes = [100, 50, 20, 10]
    for size in sizes:
        det = Detection(
            class_name="weed",
            confidence=0.8,
            bbox=BoundingBox(x1=0, y1=0, x2=size, y2=size),
            source_image="test.jpg",
        )
        batch.add_detection(det)

    # Filter with min area 2500 (50x50)
    filtered = DetectionPostProcessor.filter_batch(batch, min_area_px=2500)

    assert filtered.num_detections == 2  # Only 100x100 and 50x50


@pytest.mark.unit
def test_detection_postprocessor_nms():
    """Test Non-Maximum Suppression."""
    batch = DetectionBatch(
        source_image="test.jpg",
        model_name="test",
        model_version="1.0",
    )

    # Add overlapping detections
    det1 = Detection(
        class_name="weed",
        confidence=0.9,
        bbox=BoundingBox(x1=0, y1=0, x2=100, y2=100),
        source_image="test.jpg",
    )
    batch.add_detection(det1)

    # Highly overlapping (should be suppressed)
    det2 = Detection(
        class_name="weed",
        confidence=0.7,
        bbox=BoundingBox(x1=10, y1=10, x2=110, y2=110),
        source_image="test.jpg",
    )
    batch.add_detection(det2)

    # Non-overlapping (should be kept)
    det3 = Detection(
        class_name="disease",
        confidence=0.8,
        bbox=BoundingBox(x1=200, y1=200, x2=300, y2=300),
        source_image="test.jpg",
    )
    batch.add_detection(det3)

    # Apply NMS
    result = DetectionPostProcessor.nms(batch, iou_threshold=0.5)

    assert result.num_detections == 2  # det1 and det3
    assert det1 in result.detections
    assert det3 in result.detections


@pytest.mark.unit
def test_detection_postprocessor_compute_iou():
    """Test IoU computation."""
    bbox1 = BoundingBox(x1=0, y1=0, x2=100, y2=100)
    bbox2 = BoundingBox(x1=50, y1=50, x2=150, y2=150)

    iou = DetectionPostProcessor._compute_iou(bbox1, bbox2)

    # Intersection: 50x50=2500, Union: 10000+10000-2500=17500
    # IoU = 2500/17500 ≈ 0.143
    assert 0.14 < iou < 0.15


@pytest.mark.unit
def test_detection_postprocessor_compute_iou_no_overlap():
    """Test IoU when boxes don't overlap."""
    bbox1 = BoundingBox(x1=0, y1=0, x2=100, y2=100)
    bbox2 = BoundingBox(x1=200, y1=200, x2=300, y2=300)

    iou = DetectionPostProcessor._compute_iou(bbox1, bbox2)
    assert iou == 0.0


@pytest.mark.unit
def test_detection_postprocessor_merge_duplicates():
    """Test merging duplicate detections."""
    batch = DetectionBatch(
        source_image="test.jpg",
        model_name="test",
        model_version="1.0",
    )

    # Add nearly identical detections
    det1 = Detection(
        class_name="weed",
        confidence=0.9,
        bbox=BoundingBox(x1=0, y1=0, x2=100, y2=100),
        source_image="test.jpg",
    )
    batch.add_detection(det1)

    det2 = Detection(
        class_name="weed",
        confidence=0.85,
        bbox=BoundingBox(x1=1, y1=1, x2=101, y2=101),
        source_image="test.jpg",
    )
    batch.add_detection(det2)

    # Merge with high IoU threshold
    result = DetectionPostProcessor.merge_duplicates(batch, iou_threshold=0.85)

    assert result.num_detections == 1  # Should merge the two
    merged = result.detections[0]
    assert merged.confidence == 0.9  # Takes highest confidence


@pytest.mark.unit
def test_detection_batch_properties():
    """Test DetectionBatch properties."""
    batch = DetectionBatch(
        source_image="test.jpg",
        model_name="yolov8",
        model_version="8.0",
    )

    assert batch.num_detections == 0

    # Add detections
    for _ in range(5):
        det = Detection(
            class_name="weed",
            confidence=0.8,
            bbox=BoundingBox(x1=0, y1=0, x2=100, y2=100),
            source_image="test.jpg",
        )
        batch.add_detection(det)

    assert batch.num_detections == 5


@pytest.mark.unit
def test_detection_batch_filtering():
    """Test DetectionBatch filtering methods."""
    batch = DetectionBatch(
        source_image="test.jpg",
        model_name="test",
        model_version="1.0",
    )

    # Add diverse detections
    classes = ["weed", "disease", "pest", "weed"]
    for i, cls in enumerate(classes):
        det = Detection(
            class_name=cls,
            confidence=0.7 + i * 0.05,
            bbox=BoundingBox(x1=i*100, y1=0, x2=i*100+100, y2=100),
            source_image="test.jpg",
        )
        batch.add_detection(det)

    # Filter by confidence
    high_conf = batch.filter_by_confidence(0.75)
    assert len(high_conf) == 3

    # Filter by class
    weeds = batch.filter_by_class("weed")
    assert len(weeds) == 2

    # Filter by area
    specific_area = batch.filter_by_area(min_area=10000)
    assert len(specific_area) == 4  # All are exactly 10000 px
