"""
infer.py - Hotspot detection inference pipeline using YOLOv8.
"""
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from loguru import logger

from ..types import BoundingBox, Detection, DetectionBatch, Polygon


class HotspotDetector:
    """
    Base hotspot detector interface.

    Subclasses implement specific model backends (YOLO, etc).
    """

    def __init__(self, model_name: str, model_path: Path, device: str = "auto"):
        """
        Initialize detector.

        Args:
            model_name: Model name (e.g., "yolov8n-seg")
            model_path: Path to model checkpoint
            device: Compute device ("cuda", "cpu", or "auto")
        """
        self.model_name = model_name
        self.model_path = Path(model_path)
        # Auto-detect device: use CUDA only if available
        if device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "cuda":
            import torch
            if not torch.cuda.is_available():
                logger.warning("CUDA requested but not available, falling back to CPU")
                self.device = "cpu"
            else:
                self.device = "cuda"
        else:
            self.device = device
        logger.info(f"Detector device resolved to: {self.device}")
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load model checkpoint. Implement in subclass."""
        logger.warning(f"Model loading not implemented for {self.model_name}")

    def detect(self, image: np.ndarray, confidence_threshold: float = 0.5) -> DetectionBatch:
        """
        Run inference on image.

        Args:
            image: Input image (RGB, shape [H, W, 3])
            confidence_threshold: Confidence threshold for detections

        Returns:
            DetectionBatch with detections
        """
        raise NotImplementedError("Subclass must implement detect()")

    def detect_batch(
        self, images: list[np.ndarray], confidence_threshold: float = 0.5
    ) -> list[DetectionBatch]:
        """
        Run inference on batch of images.

        Args:
            images: List of images
            confidence_threshold: Confidence threshold

        Returns:
            List of DetectionBatch objects
        """
        return [self.detect(img, confidence_threshold) for img in images]


class YOLOv8Detector(HotspotDetector):
    """YOLOv8 detector wrapper for instance segmentation."""

    # 12 precision agriculture class mappings
    # 14 India-specific precision agriculture classes (North India wheat-rice belt)
    DEFAULT_CLASSES = {
        # Wheat classes (0-6)
        0: "healthy_wheat",
        1: "wheat_lodging",
        2: "wheat_leaf_rust",
        3: "wheat_yellow_rust",
        4: "wheat_powdery_mildew",
        5: "wheat_nitrogen_def",
        6: "wheat_weed",
        # Rice classes (7-13)
        7: "healthy_rice",
        8: "rice_blast",
        9: "rice_brown_planthopper",
        10: "rice_water_stress",
        11: "rice_weed",
        12: "poor_row_spacing",
        13: "good_row_spacing",
    }

    # Map each class to its agronomic category
    CLASS_CATEGORIES: dict[str, str] = {
        "healthy_wheat": "health",
        "wheat_lodging": "lodging",
        "wheat_leaf_rust": "disease",
        "wheat_yellow_rust": "disease",
        "wheat_powdery_mildew": "disease",
        "wheat_nitrogen_def": "nutrient",
        "wheat_weed": "weed",
        "healthy_rice": "health",
        "rice_blast": "disease",
        "rice_brown_planthopper": "disease",
        "rice_water_stress": "stress",
        "rice_weed": "weed",
        "poor_row_spacing": "stand",
        "good_row_spacing": "health",
    }

    # Baseline severity for each class (0 = benign, 1 = critical)
    CLASS_SEVERITY: dict[str, float] = {
        "healthy_wheat": 0.0,
        "wheat_lodging": 0.8,          # severe — Flores 2020
        "wheat_leaf_rust": 0.75,       # orange pustules, common Punjab
        "wheat_yellow_rust": 0.85,     # stripe rust, spreads fast
        "wheat_powdery_mildew": 0.6,   # moderate if caught early
        "wheat_nitrogen_def": 0.55,    # yellowing, correctable
        "wheat_weed": 0.5,             # broadleaf weeds
        "healthy_rice": 0.0,
        "rice_blast": 0.9,             # neck/leaf blast — major India issue
        "rice_brown_planthopper": 0.8, # hopper burn, rapid
        "rice_water_stress": 0.5,      # rolled leaves
        "rice_weed": 0.45,             # paddy weeds
        "poor_row_spacing": 0.3,       # uneven transplanting
        "good_row_spacing": 0.0,
    }

    # Map each class to its crop type
    CLASS_CROP_TYPE: dict[str, str] = {
        "healthy_wheat": "wheat",
        "wheat_lodging": "wheat",
        "wheat_leaf_rust": "wheat",
        "wheat_yellow_rust": "wheat",
        "wheat_powdery_mildew": "wheat",
        "wheat_nitrogen_def": "wheat",
        "wheat_weed": "wheat",
        "healthy_rice": "rice",
        "rice_blast": "rice",
        "rice_brown_planthopper": "rice",
        "rice_water_stress": "rice",
        "rice_weed": "rice",
        "poor_row_spacing": "rice",
        "good_row_spacing": "rice",
    }

    def _load_model(self) -> None:
        """Load YOLOv8 model."""
        try:
            from ultralytics import YOLO

            # Check if model file exists
            if not self.model_path.exists():
                logger.info(f"Model file not found at {self.model_path}, downloading...")
                # YOLO will auto-download from ultralytics if model_path is a model name
                self.model = YOLO(self.model_name, task="segment")
            else:
                self.model = YOLO(str(self.model_path), task="segment")

            logger.info(f"Successfully loaded YOLOv8 model: {self.model_name}")
            logger.info(f"Model device: {self.device}")

            # Set device
            if self.device == "cuda":
                self.model.to("cuda")
            else:
                self.model.to("cpu")

        except ImportError:
            logger.error("ultralytics not installed. Install with: pip install ultralytics")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}", exc_info=True)

    def detect(self, image: np.ndarray, confidence_threshold: float = 0.5) -> DetectionBatch:
        """
        Run YOLOv8 inference on image.

        Args:
            image: Input image (BGR or RGB, shape [H, W, 3])
            confidence_threshold: Confidence threshold for detections

        Returns:
            DetectionBatch with structured detections
        """
        start_time = time.time()

        if self.model is None:
            logger.error("Model not loaded - cannot run inference")
            return DetectionBatch(
                source_image="unknown",
                model_name=self.model_name,
                model_version="unknown",
            )

        try:
            # Ensure image is RGB (convert BGR to RGB if needed)
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Assume BGR input from OpenCV, convert to RGB
                if image.dtype == np.uint8:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Run inference
            results = self.model(
                image,
                conf=confidence_threshold,
                verbose=False,
                device=self.device,
            )

            processing_time_ms = (time.time() - start_time) * 1000

            # Create detection batch
            batch = DetectionBatch(
                source_image="inference_input",
                model_name=self.model_name,
                model_version=self._get_model_version(),
                processing_time_ms=processing_time_ms,
            )

            if not results or len(results) == 0:
                logger.debug("No results from inference")
                return batch

            result = results[0]
            image_height, image_width = image.shape[:2]

            # Extract detections from results
            if hasattr(result, "boxes") and result.boxes is not None:
                boxes = result.boxes
                masks = result.masks if hasattr(result, "masks") and result.masks is not None else None

                for idx, box in enumerate(boxes):
                    try:
                        detection = self._parse_box_and_mask(
                            box=box,
                            masks=masks,
                            box_idx=idx,
                            image_width=image_width,
                            image_height=image_height,
                            class_names=result.names,
                        )
                        batch.add_detection(detection)

                    except Exception as e:
                        logger.warning(f"Failed to parse detection {idx}: {e}")
                        continue

            logger.info(
                f"Detection complete: {batch.num_detections} hotspots found in {processing_time_ms:.1f}ms"
            )
            return batch

        except Exception as e:
            logger.error(f"Inference error: {e}", exc_info=True)
            return DetectionBatch(
                source_image="inference_input",
                model_name=self.model_name,
                model_version=self._get_model_version(),
            )

    def _parse_box_and_mask(
        self,
        box,
        masks: Optional[np.ndarray],
        box_idx: int,
        image_width: int,
        image_height: int,
        class_names: dict,
    ) -> Detection:
        """
        Parse YOLO detection box and optional segmentation mask.

        Args:
            box: YOLO box object
            masks: Segmentation masks array (optional)
            box_idx: Index of box in results
            image_width: Image width in pixels
            image_height: Image height in pixels
            class_names: Dictionary of class names

        Returns:
            Detection object with bbox and optional polygon
        """
        # Extract class name
        class_id = int(box.cls[0]) if hasattr(box, "cls") else 0
        class_name = class_names.get(class_id, f"class_{class_id}") if class_names else "unknown"

        # Extract confidence
        confidence = float(box.conf[0]) if hasattr(box, "conf") else 0.5

        # Extract bounding box coordinates (normalized to image size)
        xyxy = box.xyxy[0] if hasattr(box, "xyxy") else None
        if xyxy is None:
            raise ValueError("No xyxy coordinates in box")

        x1 = float(xyxy[0])
        y1 = float(xyxy[1])
        x2 = float(xyxy[2])
        y2 = float(xyxy[3])

        # Create bounding box (pixel coordinates)
        bbox = BoundingBox(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            is_normalized=False,
        )

        # Extract segmentation mask if available
        polygon = None
        if masks is not None and box_idx < len(masks.data):
            try:
                mask = masks.data[box_idx]
                polygon = self._mask_to_polygon(mask, image_height, image_width)
            except Exception as e:
                logger.debug(f"Failed to extract mask polygon: {e}")

        # Compute severity: scale class baseline by detection confidence
        base_severity = self.CLASS_SEVERITY.get(class_name, 0.5)
        severity_score = round(min(1.0, base_severity * confidence / 0.5), 4)

        # Determine agronomic category
        category = self.CLASS_CATEGORIES.get(class_name, "health")

        # Determine crop type
        crop_type = self.CLASS_CROP_TYPE.get(class_name, "unknown")

        # Compute area_pct: detection bbox area as % of image area
        image_area = image_width * image_height
        area_pct = round((bbox.area / image_area) * 100, 4) if image_area > 0 else 0.0

        # Create detection
        detection = Detection(
            class_name=class_name,
            confidence=confidence,
            severity_score=severity_score,
            category=category,
            crop_type=crop_type,
            area_pct=area_pct,
            bbox=bbox,
            polygon=polygon,
            source_image="inference_input",
            model_name=self.model_name,
            model_version=self._get_model_version(),
        )

        return detection

    @staticmethod
    def _mask_to_polygon(mask: np.ndarray, height: int, width: int) -> Optional[Polygon]:
        """
        Convert segmentation mask to polygon.

        Args:
            mask: Binary segmentation mask (H, W)
            height: Image height
            width: Image width

        Returns:
            Polygon object or None if mask is empty
        """
        try:
            # Resize mask to image size if needed
            if mask.shape != (height, width):
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

            # Convert mask to uint8
            mask_uint8 = (mask * 255).astype(np.uint8)

            # Find contours
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return None

            # Get largest contour
            largest_contour = max(contours, key=cv2.contourArea)

            # Simplify contour
            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            simplified = cv2.approxPolyDP(largest_contour, epsilon, True)

            # Convert to list of (x, y) tuples
            points = [(float(pt[0][0]), float(pt[0][1])) for pt in simplified]

            if len(points) >= 3:
                return Polygon(points=points, is_normalized=False)

            return None

        except Exception as e:
            logger.debug(f"Mask to polygon conversion failed: {e}")
            return None

    def _get_model_version(self) -> str:
        """Get model version string."""
        try:
            if self.model is not None and hasattr(self.model, "model"):
                return self.model.model.yaml.get("version", "unknown")
        except Exception:
            pass
        return "8.0"  # Default YOLOv8 version
