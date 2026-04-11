"""
actuation.py - Sprayer control and safety models.
"""
from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class ActuationStatus(str, Enum):
    """Status of actuation event."""
    IDLE = "idle"
    ARMED = "armed"
    SPRAYING = "spraying"
    PAUSED = "paused"
    STOPPED = "stopped"
    FAULT = "fault"
    EMERGENCY_STOP = "emergency_stop"


class SafetyState(str, Enum):
    """Overall safety state."""
    SAFE = "safe"
    WARNING = "warning"
    FAULT = "fault"
    SHUTDOWN = "shutdown"


class ActuationEvent(BaseModel):
    """Single sprayer actuation event."""
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    mission_id: str = Field(...)
    status: ActuationStatus = Field(default=ActuationStatus.IDLE)

    # Spray parameters
    pump_rate: float = Field(default=0.0, description="Pump duty cycle 0-1")
    valve_position: float = Field(default=0.0, description="Valve position 0-1")
    spray_duration_seconds: Optional[float] = None
    fluid_dispensed_ml: Optional[float] = None

    # Location
    location_lat: Optional[float] = None
    location_lon: Optional[float] = None
    altitude_m: Optional[float] = None

    # Telemetry
    pressure_bar: Optional[float] = None
    temperature_c: Optional[float] = None
    humidity_percent: Optional[float] = None

    # Safety/logging
    safety_approved: bool = Field(default=False)
    dry_run: bool = Field(default=True)
    error_code: Optional[str] = None
    notes: Optional[str] = None

    def is_actuating(self) -> bool:
        """Check if sprayer is actively dispensing."""
        return self.status == ActuationStatus.SPRAYING and self.pump_rate > 0.0


class SafetyCheckResult(BaseModel):
    """Result of a safety check."""
    check_name: str = Field(...)
    passed: bool = Field(...)
    severity: str = Field(default="info", description="info, warning, error")
    message: str = Field(default="")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SafetyReport(BaseModel):
    """Comprehensive safety check report."""
    report_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    mission_id: str = Field(...)
    overall_state: SafetyState = Field(default=SafetyState.SAFE)

    checks: list[SafetyCheckResult] = Field(default_factory=list)

    dry_run_enabled: bool = Field(default=True)
    test_fluid_enabled: bool = Field(default=True)
    hardware_ready: bool = Field(default=False)
    software_ready: bool = Field(default=False)
    all_checks_passed: bool = Field(default=False)

    def add_check(self, check: SafetyCheckResult) -> None:
        """Add safety check result."""
        self.checks.append(check)

    def finalize(self) -> None:
        """Compute final safety state."""
        if not self.checks:
            return

        errors = [c for c in self.checks if c.severity == "error"]
        warnings = [c for c in self.checks if c.severity == "warning"]

        if errors:
            self.overall_state = SafetyState.FAULT
        elif warnings:
            self.overall_state = SafetyState.WARNING
        else:
            self.overall_state = SafetyState.SAFE

        all_passed = all(c.passed for c in self.checks)
        self.all_checks_passed = all_passed and self.dry_run_enabled and self.test_fluid_enabled


class ActuationLog(BaseModel):
    """Complete actuation log for a mission."""
    log_id: str = Field(default_factory=lambda: str(uuid4()))
    mission_id: str = Field(...)
    timestamp_start: datetime = Field(default_factory=datetime.utcnow)
    timestamp_end: Optional[datetime] = None

    events: list[ActuationEvent] = Field(default_factory=list)
    safety_reports: list[SafetyReport] = Field(default_factory=list)

    total_spray_time_seconds: float = Field(default=0.0)
    total_fluid_dispensed_ml: float = Field(default=0.0)
    num_spray_zones: int = Field(default=0)
    num_errors: int = Field(default=0)

    def add_event(self, event: ActuationEvent) -> None:
        """Add actuation event to log."""
        self.events.append(event)
        if event.status == ActuationStatus.SPRAYING:
            if event.spray_duration_seconds:
                self.total_spray_time_seconds += event.spray_duration_seconds
            if event.fluid_dispensed_ml:
                self.total_fluid_dispensed_ml += event.fluid_dispensed_ml

    def add_safety_report(self, report: SafetyReport) -> None:
        """Add safety check result."""
        self.safety_reports.append(report)

    def finalize(self, timestamp: Optional[datetime] = None) -> None:
        """Mark log as complete."""
        self.timestamp_end = timestamp or datetime.utcnow()
