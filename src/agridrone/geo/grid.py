"""
grid.py - Generate regular grid tiles over fields for prescription mapping.
"""
from typing import Optional

from shapely.geometry import Polygon as ShapelyPolygon

from ..types import GeoCoordinate, GridCell, PrescriptionMap


class FieldGridGenerator:
    """Generate regular grid cells tiling a field."""

    def __init__(self, cell_size_m: float = 10.0):
        """
        Initialize grid generator.

        Args:
            cell_size_m: Size of each grid cell in meters
        """
        self.cell_size_m = cell_size_m

    def generate_grid(
        self,
        field_bounds: tuple[float, float, float, float],
        field_center: GeoCoordinate,
        crs: str = "EPSG:32633",
    ) -> PrescriptionMap:
        """
        Generate grid covering field bounds.

        Args:
            field_bounds: (minx, miny, maxx, maxy) in geographic coordinates
            field_center: Center point of field
            crs: Coordinate reference system

        Returns:
            PrescriptionMap with empty grid cells
        """
        minx, miny, maxx, maxy = field_bounds
        prescription_map = PrescriptionMap(field_center=field_center, crs=crs)

        row = 0
        y = miny
        while y < maxy:
            col = 0
            x = minx
            while x < maxx:
                # Create cell
                cell_id = f"grid_{row}_{col}"
                center = GeoCoordinate(x=x + self.cell_size_m / 2, y=y + self.cell_size_m / 2, crs=crs)

                cell = GridCell(
                    cell_id=cell_id,
                    row=row,
                    col=col,
                    geometry_wkt=f"POLYGON(({x} {y}, {x + self.cell_size_m} {y}, {x + self.cell_size_m} {y + self.cell_size_m}, {x} {y + self.cell_size_m}, {x} {y}))",
                    center=center,
                    area_m2=self.cell_size_m * self.cell_size_m,
                )

                prescription_map.add_cell(cell)
                col += 1
                x += self.cell_size_m

            row += 1
            y += self.cell_size_m

        prescription_map.compute_statistics()
        return prescription_map
