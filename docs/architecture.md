"""
AgriDrone System Architecture
==============================

## Overview

AgriDrone is a research prototype for site-specific crop protection using aerial imagery,
environmental sensing, and controlled selective spraying.

## System Architecture

```
Image Input
    ↓
[Vision Module]  → Detection hotspots
    ↓
[Geo Module]  → Georeferenced locations
    ↓
[Prescription Engine] → Severity scores
    ↓
[Environmental Fusion] → Risk modifiers
    ↓
[Safety Checks] → Validation
    ↓
[Actuation Controller] → Spray zones
    ↓
[Logging] → Mission record
```

## Core Modules

### 1. Vision Module (`src/agridrone/vision/`)

Detects hotspot classes using YOLO or similar deep learning models.

**Key Components:**
- `infer.py` - Inference pipeline
- `train.py` - Model training (future)
- `postprocess.py` - Post-processing and NMS
- `uncertainty.py` - Uncertainty estimation (optional)

**Interfaces:**
- Input: RGB image (numpy array)
- Output: DetectionBatch with Detection objects

**Configuration:**
- `configs/model.yaml` - Model architecture and inference parameters

### 2. Geospatial Module (`src/agridrone/geo/`)

Links image detections to field coordinates and generates tiled grids.

**Key Components:**
- `georef.py` - Pixel-to-geo coordinate transformation
- `grid.py` - Regular grid generation
- `shapefile.py` - Shapefile export

**Interfaces:**
- Input: Detection + CameraFrame (with GNSS)
- Output: GeoCoordinate or GridCell

**Configuration:**
- `configs/base.yaml` - CRS and grid parameters

### 3. Prescription Engine (`src/agridrone/prescription/`)

Converts detections into actionable spray recommendations using deterministic rules.

**Key Components:**
- `rules.py` - Rule-based prescription engine
- `severity.py` - Severity scoring
- `optimize.py` - Optimization methods (future)

**Interfaces:**
- Input: PrescriptionMap with GridCell severity scores
- Output: PrescriptionMap with recommended_action and spray_rate

**Configuration:**
- `configs/prescription.yaml` - Thresholds and spray rates

### 4. Environmental Fusion (`src/agridrone/environment/`)

Attaches environmental context and modifies prescriptions.

**Key Components:**
- `features.py` - Feature extraction and attachment
- `fusion.py` - Multi-sensor fusion

**Interfaces:**
- Input: Temperature, humidity, wind speed, etc.
- Output: GridCell with env_features attached

### 5. Actuation Module (`src/agridrone/actuation/`)

Controls sprayer hardware with mandatory safety interlocks.

**Key Components:**
- `controller.py` - Sprayer controller interface
- `safety.py` - Safety checks and validation
- `nozzle_logic.py` - Nozzle control logic
- `mock_controller.py` - Mock for testing

**Interfaces:**
- Input: ActuationPlan
- Output: ActuationEvent log

**Configuration:**
- `configs/actuation.yaml` - Hardware pins, safety settings

### 6. Simulation Module (`src/agridrone/sim/`)

Generates synthetic fields and enables closed-loop testing.

**Key Components:**
- `field_generator.py` - Synthetic field generation
- `infestation.py` - Hotspot distribution
- `spraying.py` - Spray outcome simulation
- `metrics.py` - Evaluation metrics

### 7. Runtime Module (`src/agridrone/runtime/`)

Orchestrates end-to-end pipeline execution.

**Key Components:**
- `pipeline.py` - Main execution pipeline
- `mission_state.py` - Mission state tracking
- `decision_engine.py` - Decision logic

### 8. API Module (`src/agridrone/api/`)

FastAPI-based REST API for mission control and monitoring.

**Endpoints:**
- `GET /` - Root health check
- `POST /missions` - Create mission
- `GET /missions/{id}` - Get mission details
- `POST /missions/{id}/detect` - Run detection
- `POST /missions/{id}/prescribe` - Generate prescription

## Data Flow

### Offline Research Workflow

```
Raw Images → Detection → Labels
                ↓
                ├→ Model Training
                ├→ Visualization
                └→ Metrics
```

### Runtime Workflow

```
Drone Capture
    ↓
[Image Ingestion]
    ↓
[Detection Inference]
    ↓
[Georeferencing]
    ↓
[Grid Generation]
    ↓
[Prescription Engine]
    ↓
[Environmental Fusion]
    ↓
[Safety Validation]
    ↓
[Human Review]
    ↓
[Actuation] (if approved)
    ↓
[Logging]
```

## Safety Design

### Safety Hierarchy

1. **Hardware Failsafe** - Sprayer defaults to OFF
2. **Software Interlocks** - Mandatory flags must be set
3. **Configuration Lockouts** - DRY_RUN and TEST_FLUID_ONLY modes
4. **Deterministic Rules** - No black-box decisions for actuation
5. **Human Review** - Operator approval before spray
6. **Audit Trail** - Complete logging of decisions

### Required Safety Flags

- `DRY_RUN=true` - Disables all actuation
- `SAFE_TEST_FLUID_ONLY=true` - Only allows test fluid
- Manual approval required before any real spray

### Decision Chain

Every spray decision must pass through:

1. Detection model (confidence score)
2. Postprocessing (overlap resolution)
3. Environmental fusion (risk modifiers)
4. Prescription engine (deterministic rules)
5. Safety validation (interlock checks)
6. Human review (operator approval)
7. Logging (complete audit trail)

## Configuration System

Settings loaded from (in order):
1. `.env` file (environment variables)
2. `configs/*.yaml` files (structured config)
3. Runtime overrides

Use `ConfigManager` to access:
```python
from agridrone import get_config
config = get_config()
severity_threshold = config.get("prescription.thresholds.high_severity")
```

## Type System

All modules use Pydantic models for safe data interchange:

- `Mission` - Mission metadata and telemetry
- `Detection` - Single detection output
- `DetectionBatch` - Batch of detections
- `GridCell` - Single prescription cell
- `PrescriptionMap` - Complete prescription map
- `ActuationEvent` - Spray actuation record
- `ActuationLog` - Mission actuation history

## Testing Strategy

### Unit Tests (`tests/unit/`)
- Prescription rule logic
- Coordinate transformations
- Data model validation
- Utility functions

### Integration Tests (`tests/integration/`)
- End-to-end pipeline
- Image→detection→prescription→actuation
- Data export formats

### Simulation Tests (`tests/sim/`)
- Synthetic field generation
- Closed-loop mission replay
- Reproducibility under fixed seeds

## Performance Targets

- **Inference**: <100ms per image on GPU
- **Prescription**: <1s for 1000-cell grid
- **Memory**: <2GB for typical mission
- **Throughput**: Real-time during flight

## Future Extensions

- **Multi-spectral Support** - Thermal, near-infrared
- **Machine Learning Prescription** - Learn optimal thresholds
- **Reinforcement Learning** - Adaptive spray policies
- **ROS Integration** - Drone middleware support
- **Edge Deployment** - ONNX runtime on Jetson
