"""
mapdata.py - Geospatial and prescription map models.
"""
from datetime import datetime
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class GeoCoordinate(BaseModel):
    """Geographic coordinate with CRS information."""
    x: float = Field(..., description="X coordinate (longitude or easting)")
    y: float = Field(..., description="Y coordinate (latitude or northing)")
    crs: str = Field(default="EPSG:4326", description="Coordinate Reference System")


class GridCell(BaseModel):
    """Single grid cell in prescription map."""
    cell_id: str = Field(default_factory=lambda: str(uuid4()))
    row: int = Field(..., description="Row index in grid")
    col: int = Field(..., description="Column index in grid")
    geometry_wkt: str = Field(..., description="WKT representation of cell geometry")
    center: GeoCoordinate = Field(...)
    area_m2: float = Field(..., description="Cell area in square meters")

    # Detection/infestation data
    hotspot_fraction: float = Field(default=0.0, description="Fraction of cell with hotspots (0-1)")
    num_detections: int = Field(default=0)
    detection_classes: dict[str, int] = Field(default_factory=dict)

    # Environmental context
    env_features: dict = Field(default_factory=dict, description="Temperature, humidity, etc.")

    # Prescription output
    recommended_action: str = Field(default="none", description="none, spray, inspect")
    severity_score: float = Field(default=0.0, description="Overall severity 0-1")
    spray_rate: float = Field(default=0.0, description="Spray rate 0-1")
    reason_codes: list[str] = Field(default_factory=list)

    # Metadata
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class PrescriptionMap(BaseModel):
    """Complete prescription map for a field."""
    map_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    mission_id: Optional[str] = None
    field_name: Optional[str] = None
    field_center: GeoCoordinate = Field(...)
    crs: str = Field(default="EPSG:4326")

    grid_metadata: dict = Field(default_factory=dict)
    cells: list[GridCell] = Field(default_factory=list)
    num_cells: int = Field(default=0)

    # Summary statistics
    total_area_m2: float = Field(default=0.0)
    treated_area_m2: float = Field(default=0.0)
    treatment_ratio: float = Field(default=0.0)
    hotspot_area_m2: float = Field(default=0.0)
    hotspot_ratio: float = Field(default=0.0)

    # Processing info
    model_version: str = Field(default="unknown")
    config_version: str = Field(default="unknown")
    confidence_threshold: float = Field(default=0.5)

    def add_cell(self, cell: GridCell) -> None:
        """Add grid cell to map."""
        self.cells.append(cell)
        self.num_cells = len(self.cells)

    def compute_statistics(self) -> None:
        """Compute map-level statistics."""
        if not self.cells:
            return

        self.total_area_m2 = sum(c.area_m2 for c in self.cells)
        self.treated_area_m2 = sum(c.area_m2 for c in self.cells if c.recommended_action == "spray")
        self.treatment_ratio = self.treated_area_m2 / self.total_area_m2 if self.total_area_m2 > 0 else 0.0
        self.hotspot_area_m2 = sum(c.area_m2 * c.hotspot_fraction for c in self.cells)
        self.hotspot_ratio = self.hotspot_area_m2 / self.total_area_m2 if self.total_area_m2 > 0 else 0.0

    def get_spray_zones(self) -> list[GridCell]:
        """Return all cells recommended for spraying."""
        return [c for c in self.cells if c.recommended_action == "spray"]

    def get_high_severity_cells(self, threshold: float = 0.7) -> list[GridCell]:
        """Return high-severity cells above threshold."""
        return [c for c in self.cells if c.severity_score >= threshold]


class ActuationPlan(BaseModel):
    """Plan for sprayer actuation."""
    plan_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    map_id: str = Field(...)
    mission_id: Optional[str] = None

    spray_zones: list[GridCell] = Field(default_factory=list)
    total_zones: int = Field(default=0)
    estimated_fluid_ml: float = Field(default=0.0)
    estimated_duration_seconds: float = Field(default=0.0)

    safety_checks_passed: bool = Field(default=False)
    dry_run: bool = Field(default=True)
    test_fluid_only: bool = Field(default=True)
    requires_human_review: bool = Field(default=True)

    approval_status: str = Field(default="pending", description="pending, approved, rejected, executed")
    approved_by: Optional[str] = None
    approval_timestamp: Optional[datetime] = None

    def is_actuatable(self) -> bool:
        """Check if plan can be actuated."""
        return (
            self.safety_checks_passed
            and not self.dry_run
            and self.test_fluid_only
            and self.approval_status == "approved"
        )
