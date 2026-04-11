"""
sensor_loader.py - Load environmental sensor data (temperature, humidity, etc).
"""
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger


class SensorLoader:
    """Load and parse environmental sensor measurements."""

    def load_csv_sensor_data(
        self, filepath: Path, time_column: str = "timestamp"
    ) -> Optional[pd.DataFrame]:
        """
        Load sensor data from CSV file.

        Args:
            filepath: Path to CSV file
            time_column: Name of timestamp column

        Returns:
            DataFrame with sensor data or None if failed
        """
        try:
            df = pd.read_csv(filepath)
            if time_column in df.columns:
                df[time_column] = pd.to_datetime(df[time_column])
            logger.info(f"Loaded {len(df)} sensor records from {filepath}")
            return df
        except Exception as e:
            logger.error(f"Failed to load sensor data from {filepath}: {e}")
            return None

    def get_nearest_measurements(
        self, sensor_data: pd.DataFrame, timestamp: datetime, tolerance_seconds: int = 60
    ) -> dict:
        """
        Get sensor measurements nearest to given timestamp.

        Args:
            sensor_data: DataFrame with timestamp column
            timestamp: Query timestamp
            tolerance_seconds: Max time difference to accept

        Returns:
            Dictionary of sensor values
        """
        if sensor_data.empty:
            return {}

        # Simple nearest-neighbor search
        time_col = None
        for col in sensor_data.columns:
            if col.lower() in ["timestamp", "time", "datetime"]:
                time_col = col
                break

        if time_col is None:
            return {}

        # Find nearest row
        diffs = (sensor_data[time_col] - timestamp).abs()
        nearest_idx = diffs.idxmin()
        nearest_diff = diffs[nearest_idx].total_seconds()

        if abs(nearest_diff) > tolerance_seconds:
            return {}

        return sensor_data.iloc[nearest_idx].to_dict()

    def compute_statistics(self, sensor_data: pd.DataFrame) -> dict:
        """Compute summary statistics for sensor data."""
        return sensor_data.describe().to_dict()
