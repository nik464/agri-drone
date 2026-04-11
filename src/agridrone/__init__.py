"""
AgriDrone: Low-cost research prototype for site-specific crop protection.

A modular system for aerial hotspot detection, environmental fusion,
prescription mapping, and controlled actuation with safety-first design.
"""

__version__ = "0.1.0"
__author__ = "AgriDrone Research Team"

# Lazy imports — defer heavy modules so that import agridrone never crashes
# on missing optional deps.  config + logging are lightweight.
from .config import ConfigManager, get_config, init_config
from .logging import get_logger, setup_logging


def _lazy_types():
    from .types import (
        ActuationEvent,
        ActuationLog,
        Detection,
        DetectionBatch,
        GridCell,
        MissionLog,
        PrescriptionMap,
    )
    return {
        "ActuationEvent": ActuationEvent,
        "ActuationLog": ActuationLog,
        "Detection": Detection,
        "DetectionBatch": DetectionBatch,
        "GridCell": GridCell,
        "MissionLog": MissionLog,
        "PrescriptionMap": PrescriptionMap,
    }


def __getattr__(name: str):
    types = _lazy_types()
    if name in types:
        return types[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "__version__",
    "ConfigManager",
    "get_config",
    "init_config",
    "setup_logging",
    "get_logger",
    "MissionLog",
    "Detection",
    "DetectionBatch",
    "PrescriptionMap",
    "GridCell",
    "ActuationEvent",
    "ActuationLog",
]
