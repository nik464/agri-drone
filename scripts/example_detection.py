#!/usr/bin/env python
"""
example_detection.py - Example usage of YOLOv8 detection pipeline.

This script demonstrates:
1. Creating synthetic test images
2. Running detection inference
3. Filtering and post-processing detections
4. Exporting results
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from agridrone import init_config, setup_logging, get_logger
from agridrone.vision.infer import YOLOv8Detector
from agridrone.vision.postprocess import DetectionPostProcessor
from agridrone.types import Detection, DetectionBatch, BoundingBox
from agridrone.io.exporters import DetectionExporter


def create_synthetic_test_image(width: int = 640, height: int = 480) -> np.ndarray:
    """
    Create a synthetic test image with colored regions.

    Args:
        width: Image width
        height: Image height

    Returns:
        Image as numpy array (BGR)
    """
    image = np.ones((height, width, 3), dtype=np.uint8) * 100  # Gray background

    # Green rectangle (simulating healthy crops)
    cv2.rectangle(image, (50, 50), (200, 200), (0, 255, 0), -1)

    # Red rectangle (simulating diseased area)
    cv2.rectangle(image, (300, 150), (450, 300), (0, 0, 255), -1)

    # Yellow rectangle (simulating pest damage)
    cv2.rectangle(image, (100, 350), (300, 450), (0, 255, 255), -1)

    # Add noise
    noise = np.random.normal(0, 5, image.shape).astype(np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return image


def create_mock_detections() -> DetectionBatch:
    """
    Create mock detections for testing (when model not available).

    Returns:
        DetectionBatch with sample detections
    """
    batch = DetectionBatch(
        source_image="synthetic_test.jpg",
        model_name="yolov8n-seg (mock)",
        model_version="8.0",
        processing_time_ms=42.5,
    )

    # Mock weed detection
    det1 = Detection(
        class_name="weed",
        confidence=0.87,
        bbox=BoundingBox(x1=50, y1=50, x2=200, y2=200),
        source_image="synthetic_test.jpg",
    )
    batch.add_detection(det1)

    # Mock disease detection
    det2 = Detection(
        class_name="disease",
        confidence=0.92,
        bbox=BoundingBox(x1=300, y1=150, x2=450, y2=300),
        source_image="synthetic_test.jpg",
    )
    batch.add_detection(det2)

    # Mock pest detection
    det3 = Detection(
        class_name="pest",
        confidence=0.78,
        bbox=BoundingBox(x1=100, y1=350, x2=300, y2=450),
        source_image="synthetic_test.jpg",
    )
    batch.add_detection(det3)

    # Low confidence detection (will be filtered)
    det4 = Detection(
        class_name="anomaly",
        confidence=0.35,
        bbox=BoundingBox(x1=400, y1=50, x2=500, y2=100),
        source_image="synthetic_test.jpg",
    )
    batch.add_detection(det4)

    return batch


def main():
    """Run detection example."""
    parser = argparse.ArgumentParser(
        description="Example: YOLOv8 detection pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--use-mock", action="store_true", help="Use mock detections (no model required)"
    )
    parser.add_argument("--model", type=Path, default=Path("models/yolov8n-seg.pt"))
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/example_detection"))

    args = parser.parse_args()

    # Initialize
    config = init_config()
    setup_logging(log_level="INFO")
    logger_obj = get_logger()

    logger_obj.info("=" * 60)
    logger_obj.info("AgriDrone Detection Example")
    logger_obj.info("=" * 60)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Create synthetic test image
    logger_obj.info("\n[Step 1] Creating synthetic test image...")
    test_image_path = args.output_dir / "synthetic_test.jpg"
    test_image = create_synthetic_test_image(640, 480)
    cv2.imwrite(str(test_image_path), test_image)
    logger_obj.info(f"Created test image: {test_image_path}")

    # Step 2: Load or create detections
    logger_obj.info("\n[Step 2] Running detection inference...")

    if args.use_mock:
        logger_obj.info("Using mock detections (model not required)")
        detections = create_mock_detections()
    else:
        logger_obj.info(f"Loading model: {args.model}")
        try:
            detector = YOLOv8Detector(
                model_name="yolov8n-seg",
                model_path=args.model,
                device=args.device,
            )

            if detector.model is None:
                logger_obj.error("Failed to load model, falling back to mock detections")
                detections = create_mock_detections()
            else:
                detections = detector.detect(test_image, confidence_threshold=args.confidence)

        except Exception as e:
            logger_obj.error(f"Error loading model: {e}, using mock detections")
            detections = create_mock_detections()

    logger_obj.info(f"Found {detections.num_detections} detections")
    for i, det in enumerate(detections.detections):
        logger_obj.info(
            f"  {i+1}. {det.class_name}: conf={det.confidence:.2f}, "
            f"area={det.bbox.area:.0f}px²"
        )

    # Step 3: Filter detections
    logger_obj.info("\n[Step 3] Filtering detections...")
    min_confidence = 0.5
    min_area = 500

    filtered = DetectionPostProcessor.filter_batch(
        detections,
        min_confidence=min_confidence,
        min_area_px=min_area,
    )
    logger_obj.info(f"After filtering: {filtered.num_detections} detections")

    # Step 4: Apply NMS
    logger_obj.info("\n[Step 4] Applying Non-Maximum Suppression...")
    nms_result = DetectionPostProcessor.nms(filtered, iou_threshold=0.5)
    logger_obj.info(f"After NMS: {nms_result.num_detections} detections")

    # Step 5: Export results
    logger_obj.info("\n[Step 5] Exporting results...")

    # Export as JSON
    json_path = args.output_dir / "detections.json"
    DetectionExporter.to_json(nms_result, json_path)
    logger_obj.info(f"Exported JSON: {json_path}")

    # Export as CSV
    csv_path = args.output_dir / "detections.csv"
    DetectionExporter.to_csv(nms_result, csv_path)
    logger_obj.info(f"Exported CSV: {csv_path}")

    # Step 6: Visualize
    logger_obj.info("\n[Step 6] Creating visualization...")
    visualization = visualize_detections(test_image, nms_result)
    viz_path = args.output_dir / "detection_result.jpg"
    cv2.imwrite(str(viz_path), visualization)
    logger_obj.info(f"Saved visualization: {viz_path}")

    # Step 7: Summary
    logger_obj.info("\n[Summary]")
    logger_obj.info(f"Original detections: {detections.num_detections}")
    logger_obj.info(f"Final detections: {nms_result.num_detections}")

    # Safe handling for processing_time_ms (may be None)
    if detections.processing_time_ms is not None:
        logger_obj.info(f"Processing time: {detections.processing_time_ms:.1f}ms")
    else:
        logger_obj.info("Processing time: not available")

    logger_obj.info(f"Output directory: {args.output_dir}")

    logger_obj.info("\n" + "=" * 60)
    logger_obj.info("Example complete!")
    logger_obj.info("=" * 60)


def visualize_detections(image: np.ndarray, batch: DetectionBatch) -> np.ndarray:
    """
    Draw detections on image.

    Args:
        image: Input image (BGR)
        batch: DetectionBatch

    Returns:
        Image with detections drawn
    """
    output = image.copy()

    colors = {
        "weed": (0, 255, 255),      # Cyan
        "disease": (0, 0, 255),     # Red
        "pest": (0, 165, 255),      # Orange
        "anomaly": (255, 0, 0),     # Blue
    }

    for det in batch.detections:
        color = colors.get(det.class_name, (200, 200, 200))

        # Draw bounding box
        pt1 = (int(det.bbox.x1), int(det.bbox.y1))
        pt2 = (int(det.bbox.x2), int(det.bbox.y2))
        cv2.rectangle(output, pt1, pt2, color, 2)

        # Draw label
        label = f"{det.class_name} {det.confidence:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        label_pt1 = (pt1[0], pt1[1] - 5)
        label_pt2 = (pt1[0] + label_size[0], pt1[1] - 20)

        cv2.rectangle(output, label_pt1, label_pt2, color, -1)
        cv2.putText(
            output,
            label,
            (pt1[0], pt1[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )

    return output


if __name__ == "__main__":
    main()
