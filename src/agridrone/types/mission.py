"""
mission.py - Mission and telemetry data models.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class GNSSData(BaseModel):
    """Global Navigation Satellite System coordinates."""
    latitude: float = Field(..., description="Latitude in WGS84")
    longitude: float = Field(..., description="Longitude in WGS84")
    altitude_m: float = Field(..., description="Altitude above mean sea level in meters")
    horizontal_accuracy_m: Optional[float] = None
    vertical_accuracy_m: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class CameraFrame(BaseModel):
    """Camera frame metadata and telemetry at acquisition time."""
    frame_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    image_path: str = Field(..., description="Path to image file")
    image_size_px: tuple[int, int] = Field(..., description="(width, height) in pixels")
    camera_model: str = Field(default="unknown")
    focal_length_mm: Optional[float] = None
    sensor_size_mm: Optional[tuple[float, float]] = None
    gnss: GNSSData = Field(..., description="GPS coordinates when frame was captured")
    gps_lat: Optional[float] = Field(None, description="Latitude extracted from image EXIF")
    gps_lon: Optional[float] = Field(None, description="Longitude extracted from image EXIF")
    gps_alt: Optional[float] = Field(None, description="Altitude (m) extracted from image EXIF")
    roll_deg: Optional[float] = Field(None, description="Roll angle in degrees")
    pitch_deg: Optional[float] = Field(None, description="Pitch angle in degrees")
    yaw_deg: Optional[float] = Field(None, description="Yaw/heading angle in degrees")
    flight_speed_mps: Optional[float] = None


class MissionMetadata(BaseModel):
    """High-level mission information."""
    mission_id: str = Field(default_factory=lambda: str(uuid4()))
    mission_name: str
    timestamp_start: datetime = Field(default_factory=datetime.utcnow)
    timestamp_end: Optional[datetime] = None
    field_name: Optional[str] = None
    field_location: Optional[GNSSData] = None
    operator: Optional[str] = None
    aircraft_type: str = Field(default="unknown")
    aircraft_id: Optional[str] = None
    flight_altitude_m: float = Field(..., description="Flight altitude in meters AWL")
    research_team: str = Field(default="AgriDrone")
    version: str = Field(default="0.1.0")
    notes: Optional[str] = None
    tags: list[str] = Field(default_factory=list)


class MissionLog(BaseModel):
    """Complete mission record."""
    metadata: MissionMetadata
    frames: list[CameraFrame] = Field(default_factory=list)
    total_frames: int = Field(default=0)
    total_distance_m: Optional[float] = None
    total_area_m2: Optional[float] = None
    conditions: dict = Field(default_factory=dict)

    def add_frame(self, frame: CameraFrame) -> None:
        """Add a frame to mission log."""
        self.frames.append(frame)
        self.total_frames = len(self.frames)

    def finalize(self, timestamp: Optional[datetime] = None) -> None:
        """Mark mission as complete."""
        self.metadata.timestamp_end = timestamp or datetime.utcnow()
