#!/usr/bin/env python
"""
build_prescription_map.py - Generate prescription map from detections.

Usage:
    python scripts/build_prescription_map.py --detections outputs/inference/detections.json
    python scripts/build_prescription_map.py --detections outputs/inference/detections.json --grid-size 5.0
"""
import argparse
import json
from pathlib import Path

from agridrone import init_config, setup_logging, get_logger
from agridrone.geo.grid import FieldGridGenerator
from agridrone.prescription.rules import PrescriptionEngine
from agridrone.io.exporters import MapExporter
from agridrone.types import (
    BoundingBox,
    Detection,
    DetectionBatch,
    GeoCoordinate,
)


def _load_detections(path: Path) -> DetectionBatch:
    """Load a DetectionBatch from a JSON file exported by DetectionExporter."""
    with open(path) as f:
        data = json.load(f)
    return DetectionBatch.model_validate(data)


def _estimate_field_bounds(
    detections: list[Detection],
    cell_size_m: float,
) -> tuple[float, float, float, float]:
    """
    Estimate field bounds from detection bounding boxes.

    Uses pixel coordinates as a proxy for spatial extent.  When real
    geo-referenced data is available this should be replaced with
    actual GNSS-derived bounds.

    Returns:
        (minx, miny, maxx, maxy) padded by one cell size.
    """
    if not detections:
        return (0.0, 0.0, cell_size_m, cell_size_m)

    all_x1 = [d.bbox.x1 for d in detections]
    all_y1 = [d.bbox.y1 for d in detections]
    all_x2 = [d.bbox.x2 for d in detections]
    all_y2 = [d.bbox.y2 for d in detections]

    minx = min(all_x1) - cell_size_m
    miny = min(all_y1) - cell_size_m
    maxx = max(all_x2) + cell_size_m
    maxy = max(all_y2) + cell_size_m

    return (minx, miny, maxx, maxy)


def _map_detections_to_cells(
    detections: list[Detection],
    prescription_map,
    cell_size_m: float,
) -> None:
    """Assign detection statistics to each grid cell that overlaps."""
    for cell in prescription_map.cells:
        cx, cy = cell.center.x, cell.center.y
        half = cell_size_m / 2.0

        cell_x1 = cx - half
        cell_y1 = cy - half
        cell_x2 = cx + half
        cell_y2 = cy + half

        matching_dets: list[Detection] = []
        for det in detections:
            # Simple AABB overlap test
            if (
                det.bbox.x2 > cell_x1
                and det.bbox.x1 < cell_x2
                and det.bbox.y2 > cell_y1
                and det.bbox.y1 < cell_y2
            ):
                matching_dets.append(det)

        cell.num_detections = len(matching_dets)

        if matching_dets:
            class_counts: dict[str, int] = {}
            severity_sum = 0.0
            area_sum = 0.0

            for det in matching_dets:
                class_counts[det.class_name] = class_counts.get(det.class_name, 0) + 1
                severity_sum += det.severity_score
                area_sum += det.area_pct

            cell.detection_classes = class_counts
            cell.severity_score = severity_sum / len(matching_dets)
            cell.hotspot_fraction = min(1.0, area_sum / 100.0)


def print_summary(prescription_map) -> None:
    """Print a Rich summary table of the prescription map."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    console.print()

    # Overview
    overview = Table(title="Prescription Map Summary", show_lines=True)
    overview.add_column("Metric", style="cyan")
    overview.add_column("Value", justify="right", style="green")

    overview.add_row("Total Cells", str(prescription_map.num_cells))
    overview.add_row("Total Area (m²)", f"{prescription_map.total_area_m2:.1f}")
    overview.add_row("Treated Area (m²)", f"{prescription_map.treated_area_m2:.1f}")
    overview.add_row("Treatment Ratio", f"{prescription_map.treatment_ratio:.1%}")
    overview.add_row("Hotspot Area (m²)", f"{prescription_map.hotspot_area_m2:.1f}")
    overview.add_row("Hotspot Ratio", f"{prescription_map.hotspot_ratio:.1%}")
    overview.add_row("Spray Zones", str(len(prescription_map.get_spray_zones())))
    console.print(overview)

    # Spray zone details
    spray_zones = prescription_map.get_spray_zones()
    if spray_zones:
        zones_table = Table(title="Spray Zones", show_lines=True)
        zones_table.add_column("Cell", style="cyan")
        zones_table.add_column("Severity", justify="right", style="yellow")
        zones_table.add_column("Spray Rate", justify="right", style="red")
        zones_table.add_column("Detections", justify="right", style="green")
        zones_table.add_column("Reasons", style="magenta")

        for cell in spray_zones[:20]:  # Cap at 20 rows for readability
            zones_table.add_row(
                cell.cell_id,
                f"{cell.severity_score:.2f}",
                f"{cell.spray_rate:.2f}",
                str(cell.num_detections),
                ", ".join(cell.reason_codes),
            )

        if len(spray_zones) > 20:
            zones_table.add_row("…", f"({len(spray_zones) - 20} more)", "", "", "")

        console.print(zones_table)

    console.print()


def main():
    """Build prescription map."""
    parser = argparse.ArgumentParser(description="Build prescription map from detections")
    parser.add_argument("--detections", type=Path, required=True,
                        help="Path to detections.json from run_inference.py")
    parser.add_argument("--grid-size", type=float, default=10.0,
                        help="Grid cell size in metres")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/maps"),
                        help="Output directory for exported map files")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Ambient temperature °C (optional environmental modifier)")
    parser.add_argument("--humidity", type=float, default=None,
                        help="Relative humidity %% (optional environmental modifier)")

    args = parser.parse_args()

    # Initialize
    config = init_config()
    setup_logging(log_level=config.get_env().log_level)
    logger = get_logger()

    logger.info("Starting prescription map generation...")

    # Load detections
    if not args.detections.exists():
        logger.error(f"Detections file not found: {args.detections}")
        return

    try:
        batch = _load_detections(args.detections)
        logger.info(f"Loaded {batch.num_detections} detections from {args.detections}")
    except Exception as e:
        logger.error(f"Failed to load detections: {e}")
        return

    if batch.num_detections == 0:
        logger.warning("No detections — prescription map will be empty")

    # Estimate field bounds from detection coordinates
    bounds = _estimate_field_bounds(batch.detections, args.grid_size)
    center_x = (bounds[0] + bounds[2]) / 2
    center_y = (bounds[1] + bounds[3]) / 2
    field_center = GeoCoordinate(x=center_x, y=center_y, crs="EPSG:32633")

    logger.info(f"Field bounds: {bounds}")
    logger.info(f"Generating grid with {args.grid_size}m cells...")

    # Generate grid
    grid_gen = FieldGridGenerator(cell_size_m=args.grid_size)
    prescription_map = grid_gen.generate_grid(
        field_bounds=bounds,
        field_center=field_center,
        crs="EPSG:32633",
    )
    logger.info(f"Generated {prescription_map.num_cells} grid cells")

    # Map detections to grid cells
    _map_detections_to_cells(batch.detections, prescription_map, args.grid_size)

    # Inject environmental features if provided
    if args.temperature is not None or args.humidity is not None:
        for cell in prescription_map.cells:
            if args.temperature is not None:
                cell.env_features["temperature_c"] = args.temperature
            if args.humidity is not None:
                cell.env_features["humidity_percent"] = args.humidity

    # Run prescription engine
    engine = PrescriptionEngine()
    engine.prescribe(prescription_map)

    # Export results
    args.output_dir.mkdir(parents=True, exist_ok=True)

    MapExporter.to_geojson(prescription_map, args.output_dir / "prescription_map.geojson")
    MapExporter.to_csv(prescription_map, args.output_dir / "prescription_map.csv")

    # Print summary table
    print_summary(prescription_map)

    logger.info("Prescription map generation complete")


if __name__ == "__main__":
    main()
