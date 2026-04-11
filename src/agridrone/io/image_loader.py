"""
image_loader.py - Load and preprocess drone imagery.
"""
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from loguru import logger
from PIL import Image as PILImage
from PIL.ExifTags import TAGS, GPSTAGS

from ..types import CameraFrame, GNSSData


# ── GPS / EXIF helpers ────────────────────────────────────────────────

def _dms_to_decimal(dms_tuple: tuple, ref: str) -> float:
    """Convert EXIF GPS DMS (degrees, minutes, seconds) to decimal degrees."""
    degrees = float(dms_tuple[0])
    minutes = float(dms_tuple[1])
    seconds = float(dms_tuple[2])
    decimal = degrees + minutes / 60.0 + seconds / 3600.0
    if ref in ("S", "W"):
        decimal = -decimal
    return decimal


def extract_gps_from_exif(image_path: str) -> Optional[dict[str, float]]:
    """Extract GPS coordinates from image EXIF data.

    Works with drone images (DJI, Autel, etc.) and smartphone photos
    that embed GPS EXIF tags.

    Args:
        image_path: Path to the image file.

    Returns:
        ``{"lat": float, "lon": float, "alt": float}`` or ``None``
        if GPS data is not present.
    """
    try:
        img = PILImage.open(image_path)
        exif_data = img._getexif()
        if exif_data is None:
            return None

        # Collect GPS IFD
        gps_info: dict = {}
        for tag_id, value in exif_data.items():
            tag_name = TAGS.get(tag_id, tag_id)
            if tag_name == "GPSInfo":
                for gps_tag_id, gps_value in value.items():
                    gps_tag_name = GPSTAGS.get(gps_tag_id, gps_tag_id)
                    gps_info[gps_tag_name] = gps_value
                break

        if not gps_info:
            return None

        # Latitude
        lat_dms = gps_info.get("GPSLatitude")
        lat_ref = gps_info.get("GPSLatitudeRef", "N")
        # Longitude
        lon_dms = gps_info.get("GPSLongitude")
        lon_ref = gps_info.get("GPSLongitudeRef", "E")

        if lat_dms is None or lon_dms is None:
            return None

        lat = _dms_to_decimal(lat_dms, lat_ref)
        lon = _dms_to_decimal(lon_dms, lon_ref)

        # Altitude (optional)
        alt = 0.0
        alt_val = gps_info.get("GPSAltitude")
        if alt_val is not None:
            alt = float(alt_val)
            # Below-sea-level flag
            if gps_info.get("GPSAltitudeRef", 0) == 1:
                alt = -alt

        logger.debug(f"EXIF GPS: lat={lat:.6f} lon={lon:.6f} alt={alt:.1f}m — {image_path}")
        return {"lat": round(lat, 7), "lon": round(lon, 7), "alt": round(alt, 2)}

    except Exception as e:
        logger.warning(f"Could not read EXIF GPS from {image_path}: {e}")
        return None


# ── Indian state lookup ───────────────────────────────────────────────

# Approximate bounding boxes for major North-India agricultural states.
# Ordered so that smaller / more specific regions are tested first.
_INDIA_STATES: list[tuple[str, tuple[float, float, float, float]]] = [
    # (name, (lat_min, lat_max, lon_min, lon_max))
    ("Punjab",              (29.5, 32.5, 73.8, 76.9)),
    ("Haryana",             (27.5, 31.0, 74.5, 77.6)),
    ("Uttar Pradesh",       (23.8, 30.5, 77.0, 84.7)),
    ("Bihar",               (24.3, 27.6, 83.3, 88.2)),
    ("West Bengal",         (21.5, 27.2, 86.0, 89.9)),
    ("Madhya Pradesh",      (21.0, 26.9, 74.0, 82.8)),
    ("Rajasthan",           (23.0, 30.2, 69.5, 78.3)),
    ("Gujarat",             (20.1, 24.7, 68.2, 74.5)),
    ("Maharashtra",         (15.6, 22.1, 72.6, 80.9)),
    ("Andhra Pradesh",      (12.6, 19.9, 77.0, 84.8)),
    ("Telangana",           (15.8, 19.9, 77.2, 81.3)),
    ("Karnataka",           (11.6, 18.5, 74.0, 78.6)),
    ("Tamil Nadu",          (8.0,  13.6, 76.2, 80.4)),
    ("Kerala",              (8.2,  12.8, 74.8, 77.4)),
    ("Odisha",              (17.8, 22.6, 81.3, 87.5)),
    ("Chhattisgarh",        (17.8, 24.1, 80.2, 84.4)),
    ("Jharkhand",           (21.9, 25.3, 83.3, 87.9)),
    ("Assam",               (24.1, 28.0, 89.7, 96.0)),
    ("Uttarakhand",         (28.7, 31.5, 77.5, 81.0)),
    ("Himachal Pradesh",    (30.4, 33.3, 75.6, 79.1)),
    ("Jammu & Kashmir",     (32.2, 37.1, 73.7, 80.3)),
]

_INDIA_BOUNDS = (8.0, 37.1, 68.0, 97.5)  # lat_min, lat_max, lon_min, lon_max


def format_india_location(lat: float, lon: float) -> str:
    """Return an approximate Indian state name for the given coordinates.

    Args:
        lat: Latitude (WGS84 decimal degrees).
        lon: Longitude (WGS84 decimal degrees).

    Returns:
        ``"India - <State>"`` if matched, ``"India - Unknown State"`` if
        within India but no state matched, or ``"Outside India"``.
    """
    lat_min, lat_max, lon_min, lon_max = _INDIA_BOUNDS
    if not (lat_min <= lat <= lat_max and lon_min <= lon <= lon_max):
        return "Outside India"

    for state_name, (s_lat_min, s_lat_max, s_lon_min, s_lon_max) in _INDIA_STATES:
        if s_lat_min <= lat <= s_lat_max and s_lon_min <= lon <= s_lon_max:
            return f"India - {state_name}"

    return "India - Unknown State"


class ImageLoader:
    """Load drone images with metadata extraction."""

    SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

    def __init__(self, image_dir: Path, recursive: bool = True):
        """
        Initialize image loader.

        Args:
            image_dir: Directory containing images
            recursive: Whether to search subdirectories
        """
        self.image_dir = Path(image_dir)
        self.recursive = recursive
        self.images: list[Path] = []
        self._discover_images()

    def _discover_images(self) -> None:
        """Discover all supported images in directory."""
        pattern = "**/*" if self.recursive else "*"
        for ext in self.SUPPORTED_FORMATS:
            self.images.extend(self.image_dir.glob(pattern + ext))
            self.images.extend(self.image_dir.glob(pattern + ext.upper()))
        self.images = sorted(list(set(self.images)))
        logger.info(f"Discovered {len(self.images)} images in {self.image_dir}")

    def load_image(self, image_path: Path) -> Optional[np.ndarray]:
        """
        Load image as numpy array.

        Args:
            image_path: Path to image file

        Returns:
            Image as numpy array (BGR), or None if failed
        """
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                logger.warning(f"Failed to load image: {image_path}")
                return None
            return image
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return None

    def load_as_rgb(self, image_path: Path) -> Optional[np.ndarray]:
        """Load image and convert to RGB."""
        image = self.load_image(image_path)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def load_as_float32(self, image_path: Path, normalize: bool = True) -> Optional[np.ndarray]:
        """
        Load image as float32 normalized [0, 1].

        Args:
            image_path: Path to image
            normalize: If True, normalize to [0, 1]; else return [0, 255]

        Returns:
            Image as float32 array
        """
        image = self.load_as_rgb(image_path)
        if image is None:
            return None

        image = image.astype(np.float32)
        if normalize:
            image /= 255.0

        return image

    def get_image_size(self, image_path: Path) -> Optional[tuple[int, int]]:
        """Get image dimensions (width, height)."""
        image = self.load_image(image_path)
        if image is None:
            return None
        height, width = image.shape[:2]
        return (width, height)

    def load_with_gps(self, image_path: Path) -> tuple[Optional[np.ndarray], Optional[dict]]:
        """Load an image and extract GPS from EXIF in one call.

        Returns:
            (image_bgr, gps_dict) where gps_dict is
            ``{"lat": …, "lon": …, "alt": …}`` or ``None``.
        """
        image = self.load_image(image_path)
        gps = extract_gps_from_exif(str(image_path))
        return image, gps


class CameraFrameBuilder:
    """Build CameraFrame objects with metadata."""

    def __init__(self, source_image: str, image_size: tuple[int, int]):
        """
        Initialize frame builder.

        Args:
            source_image: Image file path or identifier
            image_size: (width, height) in pixels
        """
        self.source_image = source_image
        self.image_size = image_size

    def with_gnss(
        self,
        latitude: float,
        longitude: float,
        altitude_m: float,
        h_accuracy: Optional[float] = None,
        v_accuracy: Optional[float] = None,
    ) -> "CameraFrameBuilder":
        """Add GNSS coordinates."""
        self.gnss = GNSSData(
            latitude=latitude,
            longitude=longitude,
            altitude_m=altitude_m,
            horizontal_accuracy_m=h_accuracy,
            vertical_accuracy_m=v_accuracy,
        )
        return self

    def with_camera_model(
        self,
        model: str,
        focal_length_mm: Optional[float] = None,
        sensor_size_mm: Optional[tuple[float, float]] = None,
    ) -> "CameraFrameBuilder":
        """Add camera model information."""
        self.camera_model = model
        self.focal_length_mm = focal_length_mm
        self.sensor_size_mm = sensor_size_mm
        return self

    def with_attitude(
        self,
        roll_deg: Optional[float] = None,
        pitch_deg: Optional[float] = None,
        yaw_deg: Optional[float] = None,
    ) -> "CameraFrameBuilder":
        """Add drone attitude (orientation)."""
        self.roll_deg = roll_deg
        self.pitch_deg = pitch_deg
        self.yaw_deg = yaw_deg
        return self

    def with_speed(self, speed_mps: float) -> "CameraFrameBuilder":
        """Add flight speed."""
        self.flight_speed_mps = speed_mps
        return self

    def build(self) -> CameraFrame:
        """Build CameraFrame object."""
        return CameraFrame(
            image_path=self.source_image,
            image_size_px=self.image_size,
            gnss=getattr(self, "gnss", GNSSData(latitude=0, longitude=0, altitude_m=0)),
            camera_model=getattr(self, "camera_model", "unknown"),
            focal_length_mm=getattr(self, "focal_length_mm", None),
            sensor_size_mm=getattr(self, "sensor_size_mm", None),
            roll_deg=getattr(self, "roll_deg", None),
            pitch_deg=getattr(self, "pitch_deg", None),
            yaw_deg=getattr(self, "yaw_deg", None),
            flight_speed_mps=getattr(self, "flight_speed_mps", None),
        )
