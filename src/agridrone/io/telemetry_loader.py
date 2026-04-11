"""
telemetry_loader.py - Load and parse mission telemetry data.
"""
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger

from ..types import MissionLog, MissionMetadata


class TelemetryLoader:
    """Load mission telemetry from files (CSV, JSON, binary logs)."""

    def load_mission_metadata(self, filepath: Path) -> Optional[MissionMetadata]:
        """
        Load mission metadata from file.

        Args:
            filepath: Path to metadata file (JSON or YAML)

        Returns:
            MissionMetadata object or None if failed
        """
        try:
            import json
            if filepath.suffix == ".json":
                with open(filepath) as f:
                    data = json.load(f)
                return MissionMetadata(**data)
        except Exception as e:
            logger.error(f"Failed to load mission metadata from {filepath}: {e}")
            return None

    def load_mission_log(self, log_dir: Path) -> Optional[MissionLog]:
        """
        Load complete mission log from directory.

        Args:
            log_dir: Directory containing mission files

        Returns:
            MissionLog object or None if failed
        """
        try:
            # Try to find metadata file
            metadata_files = list(log_dir.glob("*metadata.json"))
            if not metadata_files:
                logger.warning(f"No metadata found in {log_dir}")
                return None

            metadata = self.load_mission_metadata(metadata_files[0])
            if metadata is None:
                return None

            mission_log = MissionLog(metadata=metadata)
            logger.info(f"Loaded mission {metadata.mission_id}")
            return mission_log

        except Exception as e:
            logger.error(f"Failed to load mission log from {log_dir}: {e}")
            return None
