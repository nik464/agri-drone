"""
Unit tests for prescription engine logic.
"""
import pytest
from agridrone.prescription.rules import PrescriptionEngine
from agridrone.types import GridCell, GeoCoordinate, PrescriptionMap


@pytest.mark.unit
def test_prescription_engine_high_severity(sample_grid_cell):
    """Test high severity prescription."""
    engine = PrescriptionEngine(
        high_severity_threshold=0.7,
        medium_severity_threshold=0.4,
    )

    cell = sample_grid_cell
    cell.severity_score = 0.85

    engine._prescribe_cell(cell)

    assert cell.recommended_action == "spray"
    assert cell.spray_rate == 1.0
    assert "high_severity" in cell.reason_codes


@pytest.mark.unit
def test_prescription_engine_medium_severity(sample_grid_cell):
    """Test medium severity prescription."""
    engine = PrescriptionEngine(
        high_severity_threshold=0.7,
        medium_severity_threshold=0.4,
    )

    cell = sample_grid_cell
    cell.severity_score = 0.5

    engine._prescribe_cell(cell)

    assert cell.recommended_action == "spray"
    assert cell.spray_rate == 0.5
    assert "medium_severity" in cell.reason_codes


@pytest.mark.unit
def test_prescription_engine_low_severity(sample_grid_cell):
    """Test low severity prescription."""
    engine = PrescriptionEngine(
        high_severity_threshold=0.7,
        medium_severity_threshold=0.4,
        low_severity_threshold=0.1,
    )

    cell = sample_grid_cell
    cell.severity_score = 0.2

    engine._prescribe_cell(cell)

    assert cell.recommended_action == "spray"
    assert cell.spray_rate == 0.2
    assert "low_severity" in cell.reason_codes


@pytest.mark.unit
def test_prescription_engine_below_threshold(sample_grid_cell):
    """Test below threshold (no spray)."""
    engine = PrescriptionEngine(
        low_severity_threshold=0.1,
    )

    cell = sample_grid_cell
    cell.severity_score = 0.05

    engine._prescribe_cell(cell)

    assert cell.recommended_action == "none"
    assert cell.spray_rate == 0.0
    assert "below_threshold" in cell.reason_codes


@pytest.mark.unit
def test_prescription_engine_environmental_modifier(sample_grid_cell):
    """Test environmental modifiers."""
    engine = PrescriptionEngine(high_severity_threshold=0.7)

    cell = sample_grid_cell
    cell.severity_score = 0.8
    cell.env_features = {"temperature_c": 2.0}  # Too cold

    engine._prescribe_cell(cell)

    assert cell.recommended_action == "spray"
    assert cell.spray_rate < 1.0  # Reduced due to temperature
    assert "suboptimal_temperature" in cell.reason_codes


@pytest.mark.unit
def test_prescription_map_statistics(sample_prescription_map, sample_grid_cell):
    """Test prescription map statistics computation."""
    map = sample_prescription_map

    for i in range(5):
        cell = GridCell(
            row=i, col=0,
            geometry_wkt=f"POLYGON((0 {i} 10 {i} 10 {i+10} 0 {i+10} 0 {i}))",
            center=GeoCoordinate(x=5, y=i+5),
            area_m2=100.0,
        )
        cell.severity_score = 0.5 if i < 3 else 0.2
        cell.recommended_action = "spray" if i < 3 else "none"
        map.add_cell(cell)

    map.compute_statistics()

    assert map.num_cells == 5
    assert map.total_area_m2 == 500.0
    assert map.treated_area_m2 == 300.0  # 3 cells sprayed
    assert map.treatment_ratio == 0.6
