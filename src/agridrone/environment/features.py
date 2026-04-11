"""
features.py - Compute environmental features and attach to grid cells.
"""
from datetime import datetime
from typing import Optional

import pandas as pd
from loguru import logger

from ..types import GridCell, PrescriptionMap


class EnvironmentalFeatureAttacher:
    """Attach environmental context to prescription cells."""

    def __init__(self):
        """Initialize feature attacher."""
        pass

    def attach_static_features(
        self,
        prescription_map: PrescriptionMap,
        temperature_c: float,
        humidity_percent: float,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """
        Attach static environmental features to all cells.

        Args:
            prescription_map: Map to enrich
            temperature_c: Ambient temperature in Celsius
            humidity_percent: Relative humidity percentage
            timestamp: Measurement timestamp
        """
        for cell in prescription_map.cells:
            if not cell.env_features:
                cell.env_features = {}

            cell.env_features["temperature_c"] = temperature_c
            cell.env_features["humidity_percent"] = humidity_percent
            if timestamp:
                cell.env_features["timestamp"] = timestamp.isoformat()

        logger.info(f"Attached static features: temp={temperature_c}C, humidity={humidity_percent}%")

    def attach_sensor_data(
        self, prescription_map: PrescriptionMap, sensor_data: pd.DataFrame
    ) -> None:
        """
        Attach sensor data by proximity.

        Args:
            prescription_map: Map to enrich
            sensor_data: DataFrame with sensor measurements
        """
        if sensor_data.empty:
            logger.warning("No sensor data to attach")
            return

        # Simple approach: attach nearest sensor values to each cell
        for cell in prescription_map.cells:
            # Find nearest sensor measurement (in real implementation, use spatial index)
            if not cell.env_features:
                cell.env_features = {}

            # Simple aggregation: use median values
            for column in sensor_data.columns:
                if column not in ["timestamp", "time"]:
                    try:
                        cell.env_features[column] = float(sensor_data[column].median())
                    except (ValueError, TypeError):
                        pass

        logger.info(f"Attached sensor data to {len(prescription_map.cells)} cells")
