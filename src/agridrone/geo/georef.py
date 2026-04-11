"""
georef.py - Georeferencing drone imagery to field coordinates.
"""
from typing import Optional, Tuple

import numpy as np
from loguru import logger
from pyproj import Transformer

from ..types import CameraFrame, GeoCoordinate


class Georeferencer:
    """Transform image pixel coordinates to geographic coordinates."""

    def __init__(self, src_crs: str = "EPSG:4326", dst_crs: str = "EPSG:32633"):
        """
        Initialize georeferencer.

        Args:
            src_crs: Source coordinate system
            dst_crs: Destination coordinate system
        """
        self.src_crs = src_crs
        self.dst_crs = dst_crs
        self.transformer: Optional[Transformer] = None
        if src_crs != dst_crs:
            try:
                self.transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
            except Exception as e:
                logger.warning(f"Failed to create transformer: {e}")

    def transform_coords(
        self, lon: float, lat: float
    ) -> Tuple[float, float]:
        """
        Transform coordinates between CRS systems.

        Args:
            lon: Longitude (WGS84)
            lat: Latitude (WGS84)

        Returns:
            Transformed (x, y) coordinates
        """
        if self.transformer is None:
            return (lon, lat)

        try:
            x, y = self.transformer.transform(lon, lat)
            return (x, y)
        except Exception as e:
            logger.error(f"Transformation error: {e}")
            return (lon, lat)

    def pixel_to_geo(
        self, frame: CameraFrame, pixel_x: float, pixel_y: float
    ) -> GeoCoordinate:
        """
        Convert pixel coordinates to geographic coordinates.

        Simplified approach assuming nadir (overhead) view.

        Args:
            frame: CameraFrame with GNSS and altitude
            pixel_x: X pixel coordinate
            pixel_y: Y pixel coordinate

        Returns:
            GeoCoordinate
        """
        try:
            # Extremely simplified: assume orthogonal projection
            # Real implementation would account for camera optics, attitude, etc
            image_width, image_height = frame.image_size_px
            altitude = frame.gnss.altitude_m

            # Field of view (assuming 60 degrees)
            fov_degrees = 60
            fov_rad = np.radians(fov_degrees)

            # Ground width/height at altitude
            ground_width = 2 * altitude * np.tan(fov_rad / 2)
            ground_height = 2 * altitude * np.tan(fov_rad / 2)

            # Pixel to normalized coordinates
            norm_x = pixel_x / image_width
            norm_y = pixel_y / image_height

            # Offset from nadir in meters
            offset_x = (norm_x - 0.5) * ground_width
            offset_y = (norm_y - 0.5) * ground_height

            # Apply transformation
            lon, lat = self.transform_coords(
                frame.gnss.longitude + offset_x / 111000,
                frame.gnss.latitude + offset_y / 111000,
            )

            return GeoCoordinate(x=lon, y=lat, crs=self.dst_crs)

        except Exception as e:
            logger.error(f"Pixel-to-geo conversion error: {e}")
            return GeoCoordinate(
                x=frame.gnss.longitude, y=frame.gnss.latitude, crs=self.src_crs
            )
