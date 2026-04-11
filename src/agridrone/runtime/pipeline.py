"""
pipeline.py - End-to-end mission execution pipeline.
"""
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger

from ..environment.features import EnvironmentalFeatureAttacher
from ..geo.grid import FieldGridGenerator
from ..io.image_loader import ImageLoader
from ..prescription.rules import PrescriptionEngine
from ..types import (
    CameraFrame,
    DetectionBatch,
    GeoCoordinate,
    GNSSData,
    MissionLog,
    PrescriptionMap,
)
from ..vision.infer import YOLOv8Detector
from ..vision.postprocess import DetectionPostProcessor


class MissionPipeline:
    """
    Execute complete detection -> prescription -> actuation pipeline.

    Flow:
        1. Load imagery and telemetry
        2. Run detection inference
        3. Attach environmental context
        4. Generate prescription map
        5. Create actuation plan
        6. Execute with safety checks
    """

    def __init__(
        self,
        mission_log: MissionLog,
        model_path: Path = Path("yolov8n-seg.pt"),
        model_name: str = "yolov8n-seg",
        device: str = "cpu",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        cell_size_m: float = 10.0,
        temperature_c: float = 20.0,
        humidity_percent: float = 60.0,
    ):
        """
        Initialize pipeline.

        Args:
            mission_log: MissionLog to process
            model_path: Path to YOLOv8 model weights
            model_name: Model name identifier
            device: Compute device ("cuda" or "cpu")
            confidence_threshold: Detection confidence threshold
            iou_threshold: NMS IoU threshold
            cell_size_m: Grid cell size in meters
            temperature_c: Ambient temperature for environmental features
            humidity_percent: Ambient humidity for environmental features
        """
        self.mission_log = mission_log
        self.model_path = Path(model_path)
        self.model_name = model_name
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.cell_size_m = cell_size_m
        self.temperature_c = temperature_c
        self.humidity_percent = humidity_percent

        self.detections: list[DetectionBatch] = []
        self.prescription_map: Optional[PrescriptionMap] = None
        self._image_loader: Optional[ImageLoader] = None
        self._detector: Optional[YOLOv8Detector] = None
        self._image_size: Optional[tuple[int, int]] = None

        logger.info(f"Initialized pipeline for mission {mission_log.metadata.mission_id}")

    def load_images(self, image_dir: Path) -> bool:
        """
        Load mission images and register CameraFrames in the mission log.

        Args:
            image_dir: Directory containing drone images

        Returns:
            True if at least one image was loaded
        """
        try:
            self._image_loader = ImageLoader(image_dir, recursive=True)

            if not self._image_loader.images:
                logger.warning(f"No images found in {image_dir}")
                return False

            # Build CameraFrame entries for each discovered image
            field_location = self.mission_log.metadata.field_location
            for img_path in self._image_loader.images:
                size = self._image_loader.get_image_size(img_path)
                if size is None:
                    continue
                width, height = size
                if self._image_size is None:
                    self._image_size = (width, height)

                gnss = GNSSData(
                    latitude=field_location.latitude if field_location else 0.0,
                    longitude=field_location.longitude if field_location else 0.0,
                    altitude_m=self.mission_log.metadata.flight_altitude_m,
                )
                frame = CameraFrame(
                    image_path=str(img_path),
                    image_size_px=(width, height),
                    gnss=gnss,
                )
                self.mission_log.add_frame(frame)

            logger.info(
                f"Loaded {self.mission_log.total_frames} images from {image_dir}"
            )
            return self.mission_log.total_frames > 0

        except Exception as e:
            logger.error(f"Failed to load images: {e}")
            return False

    def run_detection(self) -> bool:
        """
        Run YOLOv8 hotspot detection on all loaded images.

        Returns:
            True if detection completed (even with zero detections)
        """
        if self._image_loader is None or not self.mission_log.frames:
            logger.error("No images loaded — call load_images() first")
            return False

        logger.info("Initializing YOLOv8 detector...")
        try:
            self._detector = YOLOv8Detector(
                model_name=self.model_name,
                model_path=self.model_path,
                device=self.device,
            )
        except Exception as e:
            logger.error(f"Failed to initialize detector: {e}")
            return False

        logger.info(
            f"Running detection on {self.mission_log.total_frames} images "
            f"(conf={self.confidence_threshold}, iou={self.iou_threshold})"
        )

        total_detections = 0
        for i, frame in enumerate(self.mission_log.frames):
            image = self._image_loader.load_image(Path(frame.image_path))
            if image is None:
                logger.warning(f"Skipping unreadable image: {frame.image_path}")
                continue

            # Run inference
            batch = self._detector.detect(image, self.confidence_threshold)
            batch.source_image = frame.image_path

            # Post-process: NMS then confidence/area filter
            batch = DetectionPostProcessor.nms(batch, self.iou_threshold)
            batch = DetectionPostProcessor.filter_batch(
                batch,
                min_confidence=self.confidence_threshold,
                min_area_px=50,
            )

            self.detections.append(batch)
            total_detections += batch.num_detections
            logger.debug(
                f"Image {i + 1}/{self.mission_log.total_frames}: "
                f"{batch.num_detections} detections in {batch.processing_time_ms:.1f}ms"
            )

        logger.info(
            f"Detection complete: {total_detections} total hotspots "
            f"across {len(self.detections)} images"
        )
        return True

    def _estimate_field_bounds(self) -> tuple[float, float, float, float]:
        """
        Estimate field bounds from flight altitude and image dimensions.

        Uses a simple pinhole-camera approximation with ~70° FOV.

        Returns:
            (minx, miny, maxx, maxy) in projected coordinates (meters)
        """
        altitude = self.mission_log.metadata.flight_altitude_m
        # Approximate ground footprint: 2 * altitude * tan(FOV/2) with ~70° FOV
        ground_width = 2.0 * altitude * math.tan(math.radians(35))
        if self._image_size:
            aspect = self._image_size[1] / self._image_size[0]
        else:
            aspect = 1.0
        ground_height = ground_width * aspect

        loc = self.mission_log.metadata.field_location
        cx = loc.longitude if loc else 0.0
        cy = loc.latitude if loc else 0.0

        half_w = ground_width / 2
        half_h = ground_height / 2
        return (cx - half_w, cy - half_h, cx + half_w, cy + half_h)

    def _map_detections_to_cells(
        self,
        prescription_map: PrescriptionMap,
        field_bounds: tuple[float, float, float, float],
    ) -> None:
        """
        Map pixel-space detections into geographic grid cells.

        Sets severity_score, num_detections, detection_classes, and
        hotspot_fraction on each cell that contains detections.
        """
        minx, miny, maxx, maxy = field_bounds
        field_w = maxx - minx
        field_h = maxy - miny

        # Build a lookup by (row, col) for O(1) cell access
        cell_lookup: dict[tuple[int, int], "GridCell"] = {}
        for cell in prescription_map.cells:
            cell_lookup[(cell.row, cell.col)] = cell

        for batch in self.detections:
            # Determine source image dimensions
            frame = next(
                (f for f in self.mission_log.frames if f.image_path == batch.source_image),
                None,
            )
            if frame is None:
                img_w, img_h = self._image_size or (640, 640)
            else:
                img_w, img_h = frame.image_size_px

            for det in batch.detections:
                cx_px, cy_px = det.bbox.center
                # Map pixel center to field coordinates
                geo_x = minx + (cx_px / img_w) * field_w
                geo_y = miny + (cy_px / img_h) * field_h

                col = int((geo_x - minx) / self.cell_size_m)
                row = int((geo_y - miny) / self.cell_size_m)

                cell = cell_lookup.get((row, col))
                if cell is None:
                    continue

                cell.num_detections += 1
                cell.detection_classes[det.class_name] = (
                    cell.detection_classes.get(det.class_name, 0) + 1
                )
                # Severity: take the maximum confidence seen in the cell
                cell.severity_score = max(cell.severity_score, det.severity_score)

        # Compute hotspot fractions (capped at 1.0)
        max_per_cell = max((c.num_detections for c in prescription_map.cells), default=1) or 1
        for cell in prescription_map.cells:
            if cell.num_detections > 0:
                cell.hotspot_fraction = min(1.0, cell.num_detections / max_per_cell)

    def attach_environment(self) -> bool:
        """
        Build the prescription grid, map detections to cells, and attach
        environmental features.

        Returns:
            True if the prescription map was created successfully
        """
        if not self.detections:
            logger.warning("No detection batches — skipping environment attachment")
            return False

        logger.info("Building field grid and mapping detections...")
        try:
            field_bounds = self._estimate_field_bounds()
            loc = self.mission_log.metadata.field_location
            field_center = GeoCoordinate(
                x=loc.longitude if loc else 0.0,
                y=loc.latitude if loc else 0.0,
            )

            grid_gen = FieldGridGenerator(cell_size_m=self.cell_size_m)
            self.prescription_map = grid_gen.generate_grid(
                field_bounds=field_bounds,
                field_center=field_center,
            )
            self.prescription_map.mission_id = self.mission_log.metadata.mission_id
            self.prescription_map.field_name = self.mission_log.metadata.field_name
            self.prescription_map.confidence_threshold = self.confidence_threshold

            logger.info(f"Generated {self.prescription_map.num_cells} grid cells")

            # Map detections into grid cells
            self._map_detections_to_cells(self.prescription_map, field_bounds)
            cells_with_detections = sum(
                1 for c in self.prescription_map.cells if c.num_detections > 0
            )
            logger.info(f"Mapped detections to {cells_with_detections} cells")

            # Attach environmental features
            attacher = EnvironmentalFeatureAttacher()
            attacher.attach_static_features(
                self.prescription_map,
                temperature_c=self.temperature_c,
                humidity_percent=self.humidity_percent,
                timestamp=datetime.utcnow(),
            )

            return True

        except Exception as e:
            logger.error(f"Failed to attach environment: {e}")
            return False

    def generate_prescription(self) -> bool:
        """
        Run the PrescriptionEngine to assign severity categories and
        spray rates to each grid cell.

        Returns:
            True if prescription was generated successfully
        """
        if self.prescription_map is None:
            logger.error("No prescription map — call attach_environment() first")
            return False

        logger.info("Generating prescription map...")
        try:
            engine = PrescriptionEngine()
            engine.prescribe(self.prescription_map)

            spray_zones = self.prescription_map.get_spray_zones()
            logger.info(
                f"Prescription complete: {len(spray_zones)} spray zones, "
                f"treatment ratio {self.prescription_map.treatment_ratio:.1%}, "
                f"hotspot ratio {self.prescription_map.hotspot_ratio:.1%}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to generate prescription: {e}")
            return False

    def execute(self, image_dir: Path) -> bool:
        """
        Execute the full mission pipeline end-to-end.

        Args:
            image_dir: Directory containing drone images

        Returns:
            True if the pipeline completed successfully
        """
        mission_id = self.mission_log.metadata.mission_id
        logger.info(f"=== Starting mission pipeline: {mission_id} ===")
        start = time.time()

        if not self.load_images(image_dir):
            logger.error("Pipeline aborted: image loading failed")
            return False

        if not self.run_detection():
            logger.error("Pipeline aborted: detection failed")
            return False

        if not self.attach_environment():
            logger.error("Pipeline aborted: environment attachment failed")
            return False

        if not self.generate_prescription():
            logger.error("Pipeline aborted: prescription generation failed")
            return False

        # Finalize mission log
        self.mission_log.finalize()
        elapsed = time.time() - start

        total_dets = sum(b.num_detections for b in self.detections)
        logger.info(
            f"=== Pipeline complete in {elapsed:.1f}s: "
            f"{self.mission_log.total_frames} images, "
            f"{total_dets} detections, "
            f"{self.prescription_map.num_cells} grid cells ==="
        )
        return True

    def get_results(self) -> dict:
        """
        Return a summary of pipeline results.

        Returns:
            Dict with mission_log, detections, and prescription_map
        """
        return {
            "mission_log": self.mission_log,
            "detections": self.detections,
            "prescription_map": self.prescription_map,
        }
