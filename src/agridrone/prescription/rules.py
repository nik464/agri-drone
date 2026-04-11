"""
rules.py - Deterministic rule-based prescription engine.
"""
from typing import Optional

from loguru import logger

from ..types import GridCell, PrescriptionMap


class PrescriptionEngine:
    """Apply deterministic rules to generate spray prescriptions."""

    def __init__(
        self,
        high_severity_threshold: float = 0.7,
        medium_severity_threshold: float = 0.4,
        low_severity_threshold: float = 0.1,
    ):
        """
        Initialize prescription engine.

        Args:
            high_severity_threshold: Severity threshold for high category
            medium_severity_threshold: Severity threshold for medium category
            low_severity_threshold: Severity threshold for low category
        """
        self.high_threshold = high_severity_threshold
        self.medium_threshold = medium_severity_threshold
        self.low_threshold = low_severity_threshold

    def prescribe(self, prescription_map: PrescriptionMap) -> None:
        """
        Apply prescription rules to all grid cells.

        Args:
            prescription_map: PrescriptionMap to prescribe (modified in-place)
        """
        for cell in prescription_map.cells:
            self._prescribe_cell(cell)

        prescription_map.compute_statistics()
        logger.info(f"Prescribed {len(prescription_map.get_spray_zones())} spray zones")

    def _prescribe_cell(self, cell: GridCell) -> None:
        """Apply prescription logic to single grid cell."""
        # Severity is already computed upstream
        severity = cell.severity_score

        # Determine action and spray rate
        if severity >= self.high_threshold:
            cell.recommended_action = "spray"
            cell.spray_rate = 1.0
            cell.reason_codes.append("high_severity")

        elif severity >= self.medium_threshold:
            cell.recommended_action = "spray"
            cell.spray_rate = 0.5
            cell.reason_codes.append("medium_severity")

        elif severity >= self.low_threshold:
            cell.recommended_action = "spray"
            cell.spray_rate = 0.2
            cell.reason_codes.append("low_severity")

        else:
            cell.recommended_action = "none"
            cell.spray_rate = 0.0
            cell.reason_codes.append("below_threshold")

        # Apply environmental modifiers
        self._apply_environmental_modifiers(cell)

    def _apply_environmental_modifiers(self, cell: GridCell) -> None:
        """Apply environmental rules to adjust prescription."""
        if not cell.env_features:
            return

        # Temperature modifier
        if "temperature_c" in cell.env_features:
            temp = cell.env_features["temperature_c"]
            if temp < 5 or temp > 35:
                # Outside optimal conditions
                cell.reason_codes.append("suboptimal_temperature")
                cell.spray_rate *= 0.8

        # Humidity modifier
        if "humidity_percent" in cell.env_features:
            humidity = cell.env_features["humidity_percent"]
            if humidity < 30 or humidity > 90:
                cell.reason_codes.append("suboptimal_humidity")
                cell.spray_rate *= 0.8

        # Re-evaluate action based on modified rate
        if cell.spray_rate <= 0.0:
            cell.recommended_action = "none"
