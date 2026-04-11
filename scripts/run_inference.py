#!/usr/bin/env python
"""
run_inference.py - Run hotspot detection on image files.

Usage:
    python scripts/run_inference.py --image data/sample/field.tif --model models/yolov8n-seg.pt
    python scripts/run_inference.py --image data/raw/ --confidence 0.4 --device cpu
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from agridrone import init_config, setup_logging, get_logger
from agridrone.io.image_loader import ImageLoader
from agridrone.io.exporters import DetectionExporter
from agridrone.vision.infer import YOLOv8Detector
from agridrone.vision.postprocess import DetectionPostProcessor
from agridrone.types import DetectionBatch


# BGR color palette matching the 12 precision agriculture classes
_COLORS_BGR = {
    "healthy_crop": (0, 200, 0),
    "wheat_lodging": (0, 100, 255),
    "volunteer_corn": (0, 0, 200),
    "broadleaf_weed": (0, 0, 255),
    "grass_weed": (50, 50, 200),
    "leaf_rust": (0, 165, 255),
    "goss_wilt": (0, 200, 255),
    "fusarium_head_blight": (128, 0, 255),
    "nitrogen_deficiency": (0, 255, 255),
    "water_stress": (255, 200, 0),
    "good_plant_spacing": (200, 200, 0),
    "poor_plant_spacing": (255, 0, 200),
}


def draw_detections(image: np.ndarray, detections) -> np.ndarray:
    """Draw bounding boxes with labels on image (BGR)."""
    img = image.copy()
    for det in detections:
        x1, y1 = int(det.bbox.x1), int(det.bbox.y1)
        x2, y2 = int(det.bbox.x2), int(det.bbox.y2)
        color = _COLORS_BGR.get(det.class_name, (255, 255, 255))

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        label = f"{det.class_name} {det.confidence:.2f}"
        sz, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y = max(y1 - 4, sz[1] + 4)
        cv2.rectangle(img, (x1, label_y - sz[1] - 4), (x1 + sz[0] + 2, label_y + 4), color, -1)
        cv2.putText(img, label, (x1 + 1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return img


def print_summary(all_batches: list[DetectionBatch], elapsed_s: float) -> None:
    """Print a Rich summary table of inference results."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    console.print()

    # Per-image table
    table = Table(title="Detection Results", show_lines=True)
    table.add_column("Image", style="cyan")
    table.add_column("Detections", justify="right", style="green")
    table.add_column("Time (ms)", justify="right", style="yellow")
    table.add_column("Top Class", style="magenta")
    table.add_column("Max Conf", justify="right")

    total_dets = 0
    for batch in all_batches:
        total_dets += batch.num_detections
        if batch.detections:
            top = max(batch.detections, key=lambda d: d.confidence)
            top_class = top.class_name
            max_conf = f"{top.confidence:.3f}"
        else:
            top_class = "—"
            max_conf = "—"

        table.add_row(
            Path(batch.source_image).name,
            str(batch.num_detections),
            f"{batch.processing_time_ms:.1f}" if batch.processing_time_ms else "—",
            top_class,
            max_conf,
        )

    console.print(table)

    # Class breakdown
    class_counts: dict[str, int] = {}
    for batch in all_batches:
        for det in batch.detections:
            class_counts[det.class_name] = class_counts.get(det.class_name, 0) + 1

    if class_counts:
        breakdown = Table(title="Class Breakdown")
        breakdown.add_column("Class", style="cyan")
        breakdown.add_column("Count", justify="right", style="green")
        breakdown.add_column("Category", style="yellow")
        for cls in sorted(class_counts, key=class_counts.get, reverse=True):
            cat = YOLOv8Detector.CLASS_CATEGORIES.get(cls, "")
            breakdown.add_row(cls, str(class_counts[cls]), cat)
        console.print(breakdown)

    console.print(
        f"\n[bold]Summary:[/bold] {len(all_batches)} images, "
        f"{total_dets} total detections, {elapsed_s:.1f}s elapsed\n"
    )


def main():
    """Run inference on images."""
    parser = argparse.ArgumentParser(description="Run hotspot detection on images")
    parser.add_argument("--image", type=Path, required=True, help="Image file or directory")
    parser.add_argument("--model", type=Path, default=Path("models/yolov8n-seg.pt"))
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=Path("outputs/inference"))

    args = parser.parse_args()

    # Initialize
    config = init_config()
    setup_logging(log_level=config.get_env().log_level)
    logger = get_logger()

    logger.info("Starting inference pipeline...")
    logger.info(f"Image path: {args.image}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Confidence threshold: {args.confidence}")

    # Load images
    if not args.image.exists():
        logger.error(f"Image path does not exist: {args.image}")
        return

    if args.image.is_file():
        images = [args.image]
    else:
        loader = ImageLoader(args.image)
        images = loader.images

    if not images:
        logger.warning("No images found")
        return

    logger.info(f"Found {len(images)} images")

    # Load model
    detector = YOLOv8Detector(
        model_name="yolov8n-seg",
        model_path=args.model,
        device=args.device,
    )

    postprocessor = DetectionPostProcessor()

    # Prepare output directories
    args.output.mkdir(parents=True, exist_ok=True)
    annotated_dir = args.output / "annotated"
    annotated_dir.mkdir(parents=True, exist_ok=True)

    # Run inference on each image
    all_batches: list[DetectionBatch] = []
    pipeline_start = time.time()

    for image_path in images:
        logger.info(f"Processing {image_path}...")

        image = cv2.imread(str(image_path))
        if image is None:
            logger.warning(f"Failed to load image: {image_path}")
            continue

        # Detect
        batch = detector.detect(image, confidence_threshold=args.confidence)
        batch.source_image = str(image_path)

        # Post-process: filter + NMS + merge duplicates
        batch = postprocessor.filter_batch(batch, min_confidence=args.confidence)
        batch = postprocessor.nms(batch, iou_threshold=0.5)
        batch = postprocessor.merge_duplicates(batch, iou_threshold=0.9)

        all_batches.append(batch)

        # Draw annotated image and save
        annotated = draw_detections(image, batch.detections)
        out_name = f"{image_path.stem}_detections{image_path.suffix}"
        cv2.imwrite(str(annotated_dir / out_name), annotated)

        logger.info(f"Completed {image_path.name}: {batch.num_detections} detections")

    elapsed = time.time() - pipeline_start

    if not all_batches:
        logger.warning("No images were processed successfully")
        return

    # Merge all batches into one for unified export
    merged = DetectionBatch(
        source_image=str(args.image),
        model_name=all_batches[0].model_name,
        model_version=all_batches[0].model_version,
    )
    for batch in all_batches:
        for det in batch.detections:
            merged.add_detection(det)

    # Export detections.json and detections.csv
    DetectionExporter.to_json(merged, args.output / "detections.json")
    DetectionExporter.to_csv(merged, args.output / "detections.csv")

    # Print summary table
    print_summary(all_batches, elapsed)

    logger.info("Inference complete")


if __name__ == "__main__":
    main()
