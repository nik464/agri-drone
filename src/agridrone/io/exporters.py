"""
exporters.py - Export maps and detection results in various formats.
"""
import json
from pathlib import Path
from typing import Optional

import geopandas as gpd
import pandas as pd
from loguru import logger
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry import box

from ..types import DetectionBatch, GridCell, PrescriptionMap


class MapExporter:
    """Export prescription maps and grid data in multiple formats."""

    @staticmethod
    def to_geojson(prescription_map: PrescriptionMap, output_path: Path) -> bool:
        """
        Export prescription map to GeoJSON format.

        Args:
            prescription_map: PrescriptionMap object
            output_path: Output file path

        Returns:
            True if successful, False otherwise
        """
        try:
            features = []
            for cell in prescription_map.cells:
                # Parse WKT geometry (simplified - assumes rectangular)
                feature = {
                    "type": "Feature",
                    "properties": {
                        "cell_id": cell.cell_id,
                        "row": cell.row,
                        "col": cell.col,
                        "hotspot_fraction": cell.hotspot_fraction,
                        "severity_score": cell.severity_score,
                        "recommended_action": cell.recommended_action,
                        "spray_rate": cell.spray_rate,
                    },
                    "geometry": {
                        "type": "Point",
                        "coordinates": [cell.center.x, cell.center.y],
                    },
                }
                features.append(feature)

            geojson = {"type": "FeatureCollection", "features": features}

            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(geojson, f, indent=2)

            logger.info(f"Exported GeoJSON to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export GeoJSON: {e}")
            return False

    @staticmethod
    def to_csv(prescription_map: PrescriptionMap, output_path: Path) -> bool:
        """
        Export grid cells to CSV format.

        Args:
            prescription_map: PrescriptionMap object
            output_path: Output file path

        Returns:
            True if successful, False otherwise
        """
        try:
            rows = []
            for cell in prescription_map.cells:
                row = {
                    "cell_id": cell.cell_id,
                    "row": cell.row,
                    "col": cell.col,
                    "center_x": cell.center.x,
                    "center_y": cell.center.y,
                    "area_m2": cell.area_m2,
                    "hotspot_fraction": cell.hotspot_fraction,
                    "severity_score": cell.severity_score,
                    "recommended_action": cell.recommended_action,
                    "spray_rate": cell.spray_rate,
                }
                rows.append(row)

            df = pd.DataFrame(rows)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)

            logger.info(f"Exported CSV to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export CSV: {e}")
            return False

    @staticmethod
    def to_shapefile(prescription_map: PrescriptionMap, output_path: Path) -> bool:
        """
        Export grid to shapefile format.

        Args:
            prescription_map: PrescriptionMap object
            output_path: Output directory path

        Returns:
            True if successful, False otherwise
        """
        try:
            geometries = []
            attributes = []

            for cell in prescription_map.cells:
                # Create point geometry at cell center
                geom = ShapelyPolygon([
                    (cell.center.x - 5, cell.center.y - 5),
                    (cell.center.x + 5, cell.center.y - 5),
                    (cell.center.x + 5, cell.center.y + 5),
                    (cell.center.x - 5, cell.center.y + 5),
                ])
                geometries.append(geom)

                attributes.append({
                    "cell_id": str(cell.cell_id),
                    "row": cell.row,
                    "col": cell.col,
                    "severity": float(cell.severity_score),
                    "action": cell.recommended_action,
                })

            gdf = gpd.GeoDataFrame(attributes, geometry=geometries, crs=prescription_map.crs)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            gdf.to_file(str(output_path))

            logger.info(f"Exported Shapefile to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export Shapefile: {e}")
            return False


class DetectionExporter:
    """Export detection results in multiple formats."""

    @staticmethod
    def to_json(detections: DetectionBatch, output_path: Path) -> bool:
        """Export detections to JSON."""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(detections.model_dump(), f, indent=2, default=str)
            logger.info(f"Exported detections JSON to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to export detections JSON: {e}")
            return False

    @staticmethod
    def to_csv(detections: DetectionBatch, output_path: Path) -> bool:
        """Export detections to CSV."""
        try:
            rows = []
            for det in detections.detections:
                rows.append({
                    "detection_id": det.detection_id,
                    "class": det.class_name,
                    "confidence": det.confidence,
                    "x1": det.bbox.x1,
                    "y1": det.bbox.y1,
                    "x2": det.bbox.x2,
                    "y2": det.bbox.y2,
                    "area": det.bbox.area,
                })

            df = pd.DataFrame(rows)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            logger.info(f"Exported detections CSV to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to export detections CSV: {e}")
            return False
