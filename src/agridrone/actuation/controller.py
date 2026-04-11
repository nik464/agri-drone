"""
controller.py - Mock sprayer controller for testing and simulation.
"""
from datetime import datetime
from typing import Optional

from loguru import logger

from ..types import ActuationEvent, ActuationLog, ActuationStatus


class MockSprayerController:
    """Simulated sprayer for testing without real hardware."""

    def __init__(self, dry_run: bool = True, test_fluid_only: bool = True):
        """
        Initialize mock controller.

        Args:
            dry_run: If True, no actual actuation
            test_fluid_only: If True, only test fluid allowed
        """
        self.dry_run = dry_run
        self.test_fluid_only = test_fluid_only
        self.status = ActuationStatus.IDLE
        self.current_rate = 0.0

    def start_spray(
        self, mission_id: str, pump_rate: float = 0.5, duration_seconds: float = 10.0
    ) -> ActuationEvent:
        """
        Simulate spray actuation.

        Args:
            mission_id: Mission identifier
            pump_rate: Pump duty cycle (0-1)
            duration_seconds: Spray duration in seconds

        Returns:
            ActuationEvent recording the spray
        """
        event = ActuationEvent(
            mission_id=mission_id,
            status=ActuationStatus.SPRAYING if not self.dry_run else ActuationStatus.IDLE,
            pump_rate=pump_rate if not self.dry_run else 0.0,
            spray_duration_seconds=duration_seconds,
            dry_run=self.dry_run,
            safety_approved=True,
        )

        if not self.dry_run:
            # Simulate fluid dispensing
            flow_rate_ml_per_min = 500  # From config
            event.fluid_dispensed_ml = (flow_rate_ml_per_min * duration_seconds) / 60.0
            logger.info(f"Sprayed {event.fluid_dispensed_ml:.1f} ml for {duration_seconds}s")
        else:
            logger.info(f"DRY RUN: Would spray {pump_rate * 100:.0f}% for {duration_seconds}s")

        return event

    def stop_spray(self, mission_id: str) -> ActuationEvent:
        """Stop sprayer."""
        event = ActuationEvent(
            mission_id=mission_id,
            status=ActuationStatus.STOPPED,
            pump_rate=0.0,
            dry_run=self.dry_run,
        )
        logger.info("Sprayer stopped")
        return event

    def emergency_stop(self, mission_id: str) -> ActuationEvent:
        """Emergency stop sprayer."""
        event = ActuationEvent(
            mission_id=mission_id,
            status=ActuationStatus.EMERGENCY_STOP,
            pump_rate=0.0,
            dry_run=self.dry_run,
        )
        logger.critical("EMERGENCY STOP triggered")
        return event
