"""
vision - Detection and segmentation using deep learning models.

Core Components:
- infer.py: YOLOv8 detector for instance segmentation
- postprocess.py: NMS, filtering, and duplicate merging
- train.py: Model training pipeline (future)
- dataset.py: Dataset loading and preparation (future)
- augment.py: Data augmentation (future)
- uncertainty.py: Uncertainty estimation (future)
"""

from .infer import HotspotDetector, YOLOv8Detector
from .postprocess import DetectionPostProcessor

__all__ = [
    "HotspotDetector",
    "YOLOv8Detector",
    "DetectionPostProcessor",
]
