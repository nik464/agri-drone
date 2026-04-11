"""
field_generator.py - Generate synthetic field layouts and hotspots.
"""
import random
from typing import Optional

import numpy as np
from loguru import logger

from ..types import Detection, PrescriptionMap


class SyntheticFieldGenerator:
    """Generate synthetic fields with simulated hotspots."""

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize field generator.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def generate_hotspots(
        self,
        field_width_m: float,
        field_height_m: float,
        density: float = 0.15,
        class_distribution: Optional[dict] = None,
    ) -> list[Detection]:
        """
        Generate synthetic hotspot detections.

        Args:
            field_width_m: Field width in meters
            field_height_m: Field height in meters
            density: Fraction of field with hotspots (0-1)
            class_distribution: Distribution of hotspot classes

        Returns:
            List of synthetic Detection objects
        """
        if class_distribution is None:
            class_distribution = {"weed": 0.5, "disease": 0.3, "pest": 0.15, "anomaly": 0.05}

        num_hotspots = max(1, int(field_width_m * field_height_m * density / 100))
        detections = []

        for i in range(num_hotspots):
            # Random location
            x = random.uniform(0, field_width_m)
            y = random.uniform(0, field_height_m)
            size = random.uniform(1, 20)

            # Random class
            class_name = random.choices(
                list(class_distribution.keys()), weights=class_distribution.values()
            )[0]

            # Random confidence
            confidence = random.uniform(0.5, 1.0)

            # Create detection
            from ..types import BoundingBox, Detection
            detection = Detection(
                class_name=class_name,
                confidence=confidence,
                bbox=BoundingBox(
                    x1=max(0, x - size / 2),
                    y1=max(0, y - size / 2),
                    x2=min(field_width_m, x + size / 2),
                    y2=min(field_height_m, y + size / 2),
                    is_normalized=False,
                ),
                source_image="synthetic",
                model_name="synthetic_generator",
            )
            detections.append(detection)

        logger.info(f"Generated {num_hotspots} synthetic hotspots")
        return detections
