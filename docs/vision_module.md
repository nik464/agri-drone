"""
Vision Module Documentation
===========================

## Overview

The vision module handles hotspot detection using deep learning models,
specifically YOLOv8 for instance segmentation.

## Architecture

```
Input Image (RGB/BGR)
    ↓
[YOLOv8Detector]
    ├─ Load pretrained model
    ├─ Run inference (GPU/CPU)
    ├─ Parse boxes and masks
    └─ Output: DetectionBatch
    ↓
[DetectionPostProcessor]
    ├─ Filter by confidence
    ├─ Filter by area
    ├─ Merge duplicates
    ├─ Non-Maximum Suppression (NMS)
    └─ Output: Cleaned DetectionBatch
```

## Components

### 1. HotspotDetector (Base Class)

Abstract base class defining the detector interface.

**Usage:**
```python
from agridrone.vision import HotspotDetector

# Implement subclass for specific model
class CustomDetector(HotspotDetector):
    def _load_model(self):
        # Load model
        pass

    def detect(self, image, confidence_threshold=0.5):
        # Run inference
        pass
```

**Methods:**
- `__init__(model_name, model_path, device)` - Initialize detector
- `_load_model()` - Load model checkpoint (implement in subclass)
- `detect(image, confidence_threshold)` - Run inference on single image
- `detect_batch(images, confidence_threshold)` - Run inference on batch

### 2. YOLOv8Detector

Full implementation for YOLOv8 architecture.

**Features:**
- Automatic model download if file not found
- GPU/CPU device selection
- Instance segmentation with mask extraction
- Polygon conversion from binary masks
- Comprehensive error handling and logging
- Processing time tracking

**Usage:**
```python
from agridrone.vision import YOLOv8Detector
import cv2

# Initialize detector
detector = YOLOv8Detector(
    model_name="yolov8n-seg",
    model_path="models/yolov8n-seg.pt",
    device="cuda"  # or "cpu"
)

# Load image
image = cv2.imread("field.jpg")

# Run detection
detections = detector.detect(
    image,
    confidence_threshold=0.5
)

# Access results
print(f"Found {detections.num_detections} hotspots")
for det in detections.detections:
    print(f"  {det.class_name}: {det.confidence:.2f}")
    print(f"    - BBox: ({det.bbox.x1}, {det.bbox.y1}) to ({det.bbox.x2}, {det.bbox.y2})")
    if det.polygon:
        print(f"    - Segmentation: {len(det.polygon.points)} points")
```

**Model Options:**
- `yolov8n-seg` - Nano (fastest, lower accuracy)
- `yolov8s-seg` - Small (balanced)
- `yolov8m-seg` - Medium (slower, higher accuracy)
- `yolov8l-seg` - Large
- `yolov8x-seg` - Extra Large (slowest, highest accuracy)

**Configuration:**
```yaml
# configs/model.yaml
model:
  type: yolov8
  backbone: yolov8n  # Model size
  pretrained: true
  checkpoint: models/yolov8n-seg.pt

inference:
  confidence_threshold: 0.5
  iou_threshold: 0.45
  device: cuda
  batch_size: 8
  half_precision: true
```

### 3. DetectionPostProcessor

Cleans and filters detection results.

**Operations:**
1. **Filtering** - By confidence and area
2. **NMS** - Remove overlapping detections
3. **Merging** - Combine near-duplicate detections

**Usage:**
```python
from agridrone.vision import DetectionPostProcessor

# Filter detections
filtered = DetectionPostProcessor.filter_batch(
    batch,
    min_confidence=0.3,
    min_area_px=100,
    max_area_px=10000
)

# Apply NMS
nms_result = DetectionPostProcessor.nms(
    filtered,
    iou_threshold=0.5
)

# Merge duplicates
merged = DetectionPostProcessor.merge_duplicates(
    nms_result,
    iou_threshold=0.9
)
```

**IoU Calculation:**
```
IoU = Intersection / Union
    = (overlap_area) / (box1_area + box2_area - overlap_area)
```

## Data Structures

### Detection

Single detection output.

**Fields:**
- `detection_id` - Unique identifier
- `class_name` - Class label (weed, disease, pest, anomaly)
- `confidence` - Confidence score [0, 1]
- `uncertainty` - Uncertainty estimate (optional)
- `bbox` - Bounding box (BoundingBox object)
- `polygon` - Segmentation mask (Polygon object, optional)
- `source_image` - Source image path/ID
- `timestamp` - Detection timestamp
- `model_name` - Model used
- `model_version` - Model version

### DetectionBatch

Batch of detections from one image.

**Fields:**
- `batch_id` - Unique identifier
- `source_image` - Source image
- `timestamp` - Processing timestamp
- `model_name` - Model used
- `model_version` - Model version
- `detections` - List of Detection objects
- `num_detections` - Count of detections
- `processing_time_ms` - Inference time

**Methods:**
- `add_detection(detection)` - Add detection to batch
- `filter_by_confidence(threshold)` - Filter by confidence
- `filter_by_class(class_name)` - Filter by class
- `filter_by_area(min_area, max_area)` - Filter by area

### BoundingBox

Bounding box representation.

**Fields:**
- `x1, y1` - Top-left corner
- `x2, y2` - Bottom-right corner
- `is_normalized` - True if coordinates are [0, 1], False if pixels

**Properties:**
- `width` - Box width
- `height` - Box height
- `area` - Total area
- `center` - Center point (x, y)

### Polygon

Polygon segmentation mask.

**Fields:**
- `points` - List of (x, y) vertices
- `is_normalized` - Coordinate system

**Properties:**
- `area` - Polygon area (shoelace formula)
- `centroid` - Center point

## Workflow Examples

### Example 1: Single Image Inference

```python
from agridrone.vision import YOLOv8Detector, DetectionPostProcessor
import cv2

# Load detector
detector = YOLOv8Detector("yolov8n-seg", "models/yolov8n-seg.pt")

# Load image
image = cv2.imread("field.jpg")

# Detect
detections = detector.detect(image, confidence_threshold=0.5)

# Post-process
detections = DetectionPostProcessor.filter_batch(
    detections, min_confidence=0.4, min_area_px=50
)
detections = DetectionPostProcessor.nms(detections, iou_threshold=0.5)
detections = DetectionPostProcessor.merge_duplicates(detections)

# Use results
print(f"Final detections: {detections.num_detections}")
```

### Example 2: Batch Processing

```python
from pathlib import Path
from agridrone.io.image_loader import ImageLoader
from agridrone.vision import YOLOv8Detector

# Load images
loader = ImageLoader("data/raw_images", recursive=True)
images = [loader.load_as_rgb(p) for p in loader.images]

# Batch inference
detector = YOLOv8Detector("yolov8n-seg", "models/yolov8n-seg.pt")
batch_results = detector.detect_batch(images)

# Process all
for result in batch_results:
    print(f"{result.source_image}: {result.num_detections} detections")
```

### Example 3: Export Results

```python
from agridrone.io.exporters import DetectionExporter

# Export to different formats
DetectionExporter.to_json(detections, "detections.json")
DetectionExporter.to_csv(detections, "detections.csv")

# Can also visualize with OpenCV:
import cv2

def draw_detections(image, batch):
    output = image.copy()
    for det in batch.detections:
        # Draw box
        pt1 = (int(det.bbox.x1), int(det.bbox.y1))
        pt2 = (int(det.bbox.x2), int(det.bbox.y2))
        cv2.rectangle(output, pt1, pt2, (0, 255, 0), 2)

        # Draw label
        cv2.putText(
            output, f"{det.class_name} {det.confidence:.2f}",
            pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1
        )
    return output

result_image = draw_detections(image, detections)
cv2.imwrite("detections.jpg", result_image)
```

## Performance Tuning

### Accuracy vs Speed Trade-off

| Model | Speed | Accuracy | Memory |
|-------|-------|----------|--------|
| yolov8n | ⚡⚡⚡ | ⭐⭐ | Small |
| yolov8s | ⚡⚡ | ⭐⭐⭐ | Medium |
| yolov8m | ⚡ | ⭐⭐⭐⭐ | Large |
| yolov8l | 🐢 | ⭐⭐⭐⭐⭐ | XLarge |

### Optimization Tips

```python
# Use half precision (FP16) for faster inference
detector = YOLOv8Detector("yolov8n-seg", model_path, device="cuda")
# Half precision is enabled by default in configs/model.yaml

# Batch processing (multiple images)
detections_batch = detector.detect_batch(images)

# CPU fallback for CPU-only systems
detector_cpu = YOLOv8Detector("yolov8n-seg", model_path, device="cpu")

# Skip optional mask extraction if not needed
# (Remove polygon creation in _parse_box_and_mask)
```

### Profiling

```python
import time

start = time.time()
detections = detector.detect(image)
elapsed = (time.time() - start) * 1000

print(f"Inference time: {elapsed:.1f}ms")
print(f"Processing time from batch: {detections.processing_time_ms:.1f}ms")
```

## Troubleshooting

### Model Download Issues

```bash
# If automatic download fails:
python -c "from ultralytics import YOLO; YOLO('yolov8n-seg')"

# Or download manually:
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt
```

### CUDA/GPU Issues

```python
import torch
if not torch.cuda.is_available():
    print("CUDA not available, using CPU")
    detector = YOLOv8Detector(..., device="cpu")
```

### Memory Issues

```python
# Use smaller model
detector = YOLOv8Detector("yolov8n-seg", ...)  # Nano is smallest

# Process images one at a time instead of batch
# Reduce batch size in config
```

## Testing

```bash
# Run unit tests
pytest tests/unit/test_vision.py -v

# Test with mock detections (no model required)
python scripts/example_detection.py --use-mock

# Benchmark
python scripts/example_detection.py --device cuda
```

## Future Enhancements

- [ ] Multi-scale inference
- [ ] Uncertainty estimation (Monte Carlo dropout)
- [ ] Model fine-tuning on custom datasets
- [ ] ONNX model export for edge deployment
- [ ] Real-time video stream processing
- [ ] Multi-model ensemble
- [ ] Active learning sample selection
