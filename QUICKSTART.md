"""
QUICKSTART.md - Getting Started with AgriDrone
==============================================

## Installation

### 1. Clone and Setup

```bash
git clone <repository-url>
cd agri-drone

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install package
pip install -e .

# Optional: development tools
pip install -e .[dev]
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit for your setup (most defaults are safe)
# IMPORTANT: Leave DRY_RUN=true and SAFE_TEST_FLUID_ONLY=true by default
```

### 3. Download Model

```bash
# Create models directory
mkdir models

# Download YOLOv8 model (approx 25MB)
# Option 1: Using ultralytics
python -c "from ultralytics import YOLO; YOLO('yolov8n-seg')"

# Option 2: Manual download
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt -O models/yolov8n-seg.pt
```

## Quick Test

### Run Detection on Sample Image

```bash
python scripts/run_inference.py \
    --image data/sample/ \
    --model models/yolov8n-seg.pt \
    --confidence 0.5 \
    --output outputs/inference
```

### Generate Prescription Map

```bash
python scripts/build_prescription_map.py \
    --detections outputs/inference/detections.json \
    --output-dir outputs/maps
```

### Start API Server

```bash
# Terminal 1: Start server
uvicorn src.agridrone.api.app:app --reload --port 8000

# Terminal 2: Test endpoint
curl http://localhost:8000/health
```

### Run Simulation

```bash
python scripts/simulate_field.py --config configs/sim.yaml
```

### Run Tests

```bash
# All tests
pytest tests/

# Specific test
pytest tests/unit/test_prescription.py -v

# With coverage
pytest tests/ --cov=src/agridrone --cov-report=html
```

## Project Structure

```
agri-drone/
├── configs/              ← YAML configuration files
├── data/                 ← Input/output data
│   ├── raw/             ← Original images
│   ├── processed/       ← Prepared datasets
│   └── sample/          ← Example data
├── docs/                ← Documentation
├── src/agridrone/       ← Main package
│   ├── vision/          ← Detection inference
│   ├── geo/             ← Georeferencing & grid
│   ├── prescription/    ← Prescription engine
│   ├── actuation/       ← Sprayer control
│   ├── io/              ← Data loading/export
│   └── api/             ← FastAPI
├── scripts/             ← Standalone scripts
├── tests/               ← Unit and integration tests
└── outputs/             ← Results (logs, maps, reports)
```

## Core Workflows

### Offline Image Processing Pipeline

```python
from agridrone import init_config, setup_logging
from agridrone.io.image_loader import ImageLoader
from agridrone.vision.infer import YOLOv8Detector

# Setup
config = init_config()
setup_logging()

# Load images
loader = ImageLoader("data/raw_images")
images = loader.images

# Run inference
detector = YOLOv8Detector("yolov8n-seg", "models/yolov8n-seg.pt")
for image_path in images:
    img = loader.load_as_rgb(image_path)
    detections = detector.detect(img)
    # Process detections...
```

### Generate Prescription Map

```python
from agridrone.geo.grid import FieldGridGenerator
from agridrone.prescription.rules import PrescriptionEngine
from agridrone.types import PrescriptionMap, GeoCoordinate

# Create grid
grid_gen = FieldGridGenerator(cell_size_m=10.0)
prescription_map = grid_gen.generate_grid(
    field_bounds=(400000, 5200000, 400100, 5200100),
    field_center=GeoCoordinate(x=400050, y=5200050)
)

# Prescribe
engine = PrescriptionEngine()
engine.prescribe(prescription_map)

# Export
from agridrone.io.exporters import MapExporter
MapExporter.to_csv(prescription_map, "output.csv")
MapExporter.to_geojson(prescription_map, "output.geojson")
```

## Safety-First Development

### Remember the Safety Rules

```env
# In .env - KEEP THESE SAFE BY DEFAULT
DRY_RUN=true                    # Never remove this
SAFE_TEST_FLUID_ONLY=true      # Never disable this
```

### Check Safety Status

```python
from agridrone.actuation.safety import SafetyChecker
from agridrone.types import ActuationPlan

checker = SafetyChecker()
report = checker.check_actuation_safety(plan)

if checker.is_safe_to_actuate(report):
    print("Safe to execute")
else:
    print("Safety checks failed")
    print(report)
```

## Common Tasks

### Load Environmental Sensor Data

```python
from agridrone.io.sensor_loader import SensorLoader
from agridrone.environment.features import EnvironmentalFeatureAttacher

loader = SensorLoader()
sensor_data = loader.load_csv_sensor_data("sensor_log.csv")

attacher = EnvironmentalFeatureAttacher()
attacher.attach_sensor_data(prescription_map, sensor_data)
```

### Generate Synthetic Field for Testing

```python
from agridrone.sim.field_generator import SyntheticFieldGenerator

gen = SyntheticFieldGenerator(seed=42)
hotspots = gen.generate_hotspots(
    field_width_m=100,
    field_height_m=100,
    density=0.15
)
```

### Export Results

```python
from agridrone.io.exporters import MapExporter, DetectionExporter

# Map exports
MapExporter.to_geojson(map, "field_prescription.geojson")
MapExporter.to_csv(map, "field_prescription.csv")
MapExporter.to_shapefile(map, "field_prescription/")

# Detection exports
DetectionExporter.to_json(detections, "detections.json")
DetectionExporter.to_csv(detections, "detections.csv")
```

## Troubleshooting

### Model Download Issues

```bash
# If automatic download fails, download manually:
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt

# Then set path in .env
MODEL_PATH=path/to/yolov8n-seg.pt
```

### CUDA/GPU Issues

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Fall back to CPU
DEVICE=cpu python scripts/run_inference.py ...
```

### Import Errors

```bash
# Reinstall package
pip install -e . --force-reinstall

# Check installation
python -c "import agridrone; print(agridrone.__version__)"
```

## Next Steps

1. **Review Documentation**
   - `docs/architecture.md` - System design
   - `docs/safety.md` - Safety interlocks
   - `docs/api.md` - API endpoints

2. **Explore Code Examples**
   - Check `scripts/` for standalone utilities
   - Review `tests/unit/` for usage patterns
   - Look at `notebooks/` for notebooks (coming soon)

3. **Start Development**
   - Modify thresholds in `configs/prescription.yaml`
   - Implement custom detection post-processing
   - Add new export formats

4. **Deploy**
   - Follow `docs/field_protocol.md`
   - Ensure safety checks pass
   - Work with field supervisor

## Getting Help

- Check existing documentation in `docs/`
- Review test files in `tests/` for usage examples
- Check logs in `outputs/logs/`
- File issues with error messages and logs attached

## Contributing

Follow these guidelines:

1. Write tests for new features
2. Follow coding style: `black` formatting, type hints
3. Update relevant docstrings
4. Run tests before committing: `pytest tests/ -v`
5. Update documentation if adding modules

```bash
# Setup pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

---

**Happy researching! Remember: Safety first, demos second.** 🌾🤖
