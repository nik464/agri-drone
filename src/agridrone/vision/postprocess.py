"""
postprocess.py - Post-processing for detection results.

Handles:
- Non-maximum suppression (NMS)
- Filtering by area
- Removing duplicate detections
- Confidence thresholding
"""
from typing import Optional

import numpy as np
from loguru import logger

from ..types import Detection, DetectionBatch


class DetectionPostProcessor:
    """Clean and filter detection results."""

    @staticmethod
    def filter_batch(
        batch: DetectionBatch,
        min_confidence: float = 0.3,
        min_area_px: float = 50,
        max_area_px: Optional[float] = None,
    ) -> DetectionBatch:
        """
        Filter detections by confidence and area.

        Args:
            batch: Input detection batch
            min_confidence: Minimum confidence threshold
            min_area_px: Minimum bounding box area in pixels
            max_area_px: Maximum bounding box area in pixels (optional)

        Returns:
            Filtered detection batch
        """
        filtered = []

        for detection in batch.detections:
            # Check confidence
            if detection.confidence < min_confidence:
                continue

            # Check area
            bbox_area = detection.bbox.area
            if bbox_area < min_area_px:
                continue

            if max_area_px is not None and bbox_area > max_area_px:
                continue

            filtered.append(detection)

        # Create new batch with filtered detections
        result_batch = DetectionBatch(
            source_image=batch.source_image,
            model_name=batch.model_name,
            model_version=batch.model_version,
            processing_time_ms=batch.processing_time_ms,
        )

        for det in filtered:
            result_batch.add_detection(det)

        logger.info(
            f"Filtered detections: {len(batch.detections)} → {len(filtered)} "
            f"(min_conf={min_confidence}, min_area={min_area_px})"
        )

        return result_batch

    @staticmethod
    def nms(batch: DetectionBatch, iou_threshold: float = 0.5) -> DetectionBatch:
        """
        Apply Non-Maximum Suppression to remove overlapping detections.

        Args:
            batch: Input detection batch
            iou_threshold: IoU threshold for suppression

        Returns:
            Detection batch with NMS applied
        """
        if batch.num_detections == 0:
            return batch

        detections = sorted(batch.detections, key=lambda x: x.confidence, reverse=True)
        keep_indices = []

        for i, det_i in enumerate(detections):
            # Check if this detection overlaps with any kept detection
            suppressed = False
            for j in keep_indices:
                det_j = detections[j]
                iou = DetectionPostProcessor._compute_iou(det_i.bbox, det_j.bbox)

                if iou > iou_threshold:
                    suppressed = True
                    break

            if not suppressed:
                keep_indices.append(i)

        # Build result batch
        result_batch = DetectionBatch(
            source_image=batch.source_image,
            model_name=batch.model_name,
            model_version=batch.model_version,
            processing_time_ms=batch.processing_time_ms,
        )

        for idx in keep_indices:
            result_batch.add_detection(detections[idx])

        logger.info(f"NMS: {len(detections)} → {len(keep_indices)} detections (IoU={iou_threshold})")

        return result_batch

    @staticmethod
    def _compute_iou(bbox1, bbox2) -> float:
        """
        Compute Intersection over Union (IoU) between two bounding boxes.

        Args:
            bbox1, bbox2: BoundingBox objects

        Returns:
            IoU score (0-1)
        """
        # Compute intersection
        x1_inter = max(bbox1.x1, bbox2.x1)
        y1_inter = max(bbox1.y1, bbox2.y1)
        x2_inter = min(bbox1.x2, bbox2.x2)
        y2_inter = min(bbox1.y2, bbox2.y2)

        if x2_inter < x1_inter or y2_inter < y1_inter:
            return 0.0  # No intersection

        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)

        # Compute union
        bbox1_area = bbox1.area
        bbox2_area = bbox2.area
        union_area = bbox1_area + bbox2_area - inter_area

        if union_area == 0:
            return 0.0

        return inter_area / union_area

    @staticmethod
    def merge_duplicates(batch: DetectionBatch, iou_threshold: float = 0.9) -> DetectionBatch:
        """
        Merge duplicate detections (very high IoU).

        Args:
            batch: Input detection batch
            iou_threshold: IoU threshold for merging

        Returns:
            Detection batch with duplicates merged
        """
        if batch.num_detections <= 1:
            return batch

        merged = []
        used = set()

        for i, det_i in enumerate(batch.detections):
            if i in used:
                continue

            # Find all similar detections
            similar = [i]
            for j in range(i + 1, len(batch.detections)):
                if j in used:
                    continue

                det_j = batch.detections[j]
                iou = DetectionPostProcessor._compute_iou(det_i.bbox, det_j.bbox)

                if iou > iou_threshold:
                    similar.append(j)

            # Merge similar detections
            if len(similar) == 1:
                merged.append(det_i)
            else:
                merged_det = DetectionPostProcessor._merge_detections(
                    [batch.detections[idx] for idx in similar]
                )
                merged.append(merged_det)
                used.update(similar)

        # Build result batch
        result_batch = DetectionBatch(
            source_image=batch.source_image,
            model_name=batch.model_name,
            model_version=batch.model_version,
            processing_time_ms=batch.processing_time_ms,
        )

        for det in merged:
            result_batch.add_detection(det)

        logger.info(f"Merge duplicates: {len(batch.detections)} → {len(merged)} detections")
        return result_batch

    @staticmethod
    def _merge_detections(detections: list[Detection]) -> Detection:
        """
        Merge multiple similar detections into one.

        Args:
            detections: List of detections to merge

        Returns:
            Merged detection
        """
        if not detections:
            raise ValueError("Cannot merge empty list")

        if len(detections) == 1:
            return detections[0]

        # Average bounding box
        x1_avg = np.mean([d.bbox.x1 for d in detections])
        y1_avg = np.mean([d.bbox.y1 for d in detections])
        x2_avg = np.mean([d.bbox.x2 for d in detections])
        y2_avg = np.mean([d.bbox.y2 for d in detections])

        # Take highest confidence
        max_det = max(detections, key=lambda d: d.confidence)

        # Create merged detection
        from ..types import BoundingBox

        merged = Detection(
            class_name=max_det.class_name,
            confidence=max_det.confidence,
            bbox=BoundingBox(x1=x1_avg, y1=y1_avg, x2=x2_avg, y2=y2_avg, is_normalized=False),
            polygon=max_det.polygon,
            source_image=max_det.source_image,
            model_name=max_det.model_name,
            model_version=max_det.model_version,
        )

        return merged
