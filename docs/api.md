# FastAPI Backend API Reference

## Overview

Production-ready FastAPI backend for the AgriDrone hotspot detection system. Provides RESTful endpoints for image upload, real-time hotspot detection, and result export.

**Base URL**: `http://localhost:8000`

---

## Quick Start

### 1. Start the Server

```bash
# From project root
uvicorn agridrone.api.app:app --reload --host 127.0.0.1 --port 8000
```

### 2. Interactive API Documentation

- **Swagger UI**: http://127.0.0.1:8000/docs
- **ReDoc**: http://127.0.0.1:8000/redoc

### 3. Test the API

```bash
# Health check
curl http://127.0.0.1:8000/api/detect/health

# Run detection with image file
curl -X POST http://127.0.0.1:8000/api/detect/ \
  -F "file=@field_image.jpg" \
  -F "confidence_threshold=0.5"
```

---

## Current Implementation (Phase 1)

### Core Endpoints

#### POST /api/detect/

**Run hotspot detection on an uploaded image.**

**Method**: `POST`
**Content-Type**: `multipart/form-data`

**Parameters**:

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `file` | File | Yes | - | Image file (JPG, PNG, BMP). Max 10MB recommended. |
| `confidence_threshold` | float | No | 0.5 | Minimum confidence score [0.0-1.0] for detections |
| `include_image` | boolean | No | true | Include annotated image as base64 in response |

**Response (200 OK)**:

```json
{
  "status": "success",
  "batch_id": "batch_abc123xyz",
  "source_image": "field_001.jpg",
  "num_detections": 3,
  "processing_time_ms": 245.3,
  "detections": [
    {
      "id": "det_001",
      "class_name": "weed",
      "confidence": 0.87,
      "severity_score": 0.75,
      "bbox": {
        "x1": 100.5,
        "y1": 150.3,
        "x2": 250.8,
        "y2": 300.2,
        "width": 150.3,
        "height": 149.9,
        "area": 22515.0
      },
      "polygon": [[100.5, 150.3], [250.8, 150.3], [250.8, 300.2], [100.5, 300.2]],
      "timestamp": "2024-03-18T10:30:45.123456"
    }
  ],
  "annotated_image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
  "metadata": {}
}
```

**Error Response (400/500)**:

```json
{
  "detail": "Failed to decode image. Ensure file is a valid image format (JPG, PNG, BMP)"
}
```

**cURL Examples**:

```bash
# Basic detection
curl -X POST http://localhost:8000/api/detect/ \
  -F "file=@field_image.jpg"

# Custom confidence threshold
curl -X POST http://localhost:8000/api/detect/ \
  -F "file=@field_image.jpg" \
  -F "confidence_threshold=0.7"

# Without annotated image in response
curl -X POST http://localhost:8000/api/detect/ \
  -F "file=@field_image.jpg" \
  -F "include_image=false"
```

**Python Example**:

```python
import requests

with open("field_image.jpg", "rb") as f:
    files = {"file": f}
    params = {"confidence_threshold": 0.5}

    response = requests.post(
        "http://localhost:8000/api/detect/",
        files=files,
        params=params
    )

    result = response.json()
    print(f"Detections: {result['num_detections']}")
    for det in result['detections']:
        print(f"  - {det['class_name']}: {det['confidence']:.2f}")
```

---

#### GET /api/detect/health

**Check detector health and availability.**

**Method**: `GET`

**Response (200 OK)**:

```json
{
  "status": "healthy",
  "detector_loaded": true,
  "model_name": "yolov8n-seg",
  "device": "cuda:0"
}
```

**Example**:

```bash
curl http://localhost:8000/api/detect/health
```

---

#### POST /api/detect/reset

**Reset the detector (unload model from memory).**

**Method**: `POST`

**Response (200 OK)**:

```json
{
  "status": "success",
  "message": "Detector reset successfully"
}
```

**Use Cases**:
- Free GPU memory after batch processing
- Reload model with different weights
- Troubleshoot stuck processes

**Example**:

```bash
curl -X POST http://localhost:8000/api/detect/reset
```

---

### General Health & Status

#### GET /
Root endpoint providing basic health check.

**Response:**
```json
{
  "status": "ok",
  "app": "agridrone",
  "version": "0.1.0"
}
```

#### GET /health
Comprehensive health status.

**Response:**
```json
{
  "status": "healthy",
  "dry_run": true,
  "test_fluid_only": true
}
```

#### GET /config
Current configuration values.

**Response:**
```json
{
  "app_name": "agridrone",
  "debug": false,
  "dry_run": true,
  "log_level": "INFO",
  ...
}
```

---

### Mission Management

#### POST /missions
Create a new mission.

**Request:**
```json
{
  "mission_name": "field_a_2024",
  "field_name": "Test Field",
  "operator": "John Doe",
  "aircraft_type": "DJI Air 3"
}
```

**Response:**
```json
{
  "mission_id": "mission_abc123",
  "status": "created",
  "timestamp": "2024-03-17T10:30:00Z"
}
```

#### GET /missions/{mission_id}
Get mission details.

**Response:**
```json
{
  "mission_id": "mission_abc123",
  "name": "field_a_2024",
  "status": "active",
  "timestamp_start": "2024-03-17T10:30:00Z",
  "frames_captured": 150,
  "total_area_m2": 12500
}
```

---

### Detection & Inference

#### POST /missions/{mission_id}/detect
Run hotspot detection on mission images.

**Request:**
```json
{
  "confidence_threshold": 0.5,
  "device": "cuda"
}
```

**Response:**
```json
{
  "mission_id": "mission_abc123",
  "status": "detecting",
  "progress": "35%",
  "detections": 247
}
```

#### GET /missions/{mission_id}/detections
Get detection results.

**Response:**
```json
{
  "mission_id": "mission_abc123",
  "total_detections": 247,
  "by_class": {
    "weed": 120,
    "disease": 89,
    "pest": 38
  },
  "mean_confidence": 0.78,
  "detections": [...]
}
```

---

### Prescription Mapping

#### POST /missions/{mission_id}/prescribe
Generate prescription map from detections.

**Request:**
```json
{
  "grid_size_m": 10.0,
  "confidence_threshold": 0.5
}
```

**Response:**
```json
{
  "mission_id": "mission_abc123",
  "status": "prescribed",
  "total_cells": 1250,
  "spray_cells": 245,
  "treatment_ratio": 0.196,
  "total_area_m2": 12500,
  "treated_area_m2": 2450
}
```

#### GET /missions/{mission_id}/map
Get prescription map details.

**Response:**
```json
{
  "map_id": "map_xyz789",
  "mission_id": "mission_abc123",
  "total_cells": 1250,
  "cells_to_spray": 245,
  "grid_metadata": {
    "cell_size_m": 10.0,
    "rows": 50,
    "cols": 25
  },
  "summary": {
    "high_severity": 89,
    "medium_severity": 156,
    "low_severity": 64,
    "no_treatment": 941
  }
}
```

#### GET /missions/{mission_id}/map/cells
Get all grid cells with details.

**Query Parameters:**
- `recommended_action`: Filter by action (none, spray, inspect)
- `min_severity`: Minimum severity score
- `format`: Export format (json, geojson, csv)

**Response:**
```json
[
  {
    "cell_id": "grid_0_0",
    "row": 0,
    "col": 0,
    "center": {"x": 45.5001, "y": -122.5001},
    "severity_score": 0.82,
    "recommended_action": "spray",
    "spray_rate": 1.0,
    "reason_codes": ["high_severity", "optimal_conditions"]
  },
  ...
]
```

---

### Actuation Planning & Safety

#### POST /missions/{mission_id}/actuation-plan
Create actuation plan from prescription map.

**Request:**
```json
{
  "dry_run": true,
  "test_fluid_only": true,
  "require_review": true
}
```

**Response:**
```json
{
  "plan_id": "plan_123abc",
  "mission_id": "mission_abc123",
  "spray_zones": 245,
  "estimated_fluid_ml": 612.5,
  "estimated_duration_seconds": 1850,
  "safety_checks_passed": true,
  "approval_status": "pending"
}
```

#### GET /missions/{mission_id}/safety-status
Get current safety status.

**Response:**
```json
{
  "mission_id": "mission_abc123",
  "overall_state": "SAFE",
  "dry_run_enabled": true,
  "test_fluid_only_enabled": true,
  "checks": [
    {
      "check_name": "dry_run_enabled",
      "passed": true,
      "severity": "info"
    },
    {
      "check_name": "test_fluid_only",
      "passed": true,
      "severity": "info"
    },
    {
      "check_name": "human_review",
      "passed": false,
      "severity": "warning",
      "message": "Plan awaiting human approval"
    }
  ]
}
```

#### POST /missions/{mission_id}/actuation-plan/{plan_id}/approve
Approve an actuation plan for execution.

**Request:**
```json
{
  "approved_by": "operator_name",
  "notes": "Conditions are optimal"
}
```

**Response:**
```json
{
  "plan_id": "plan_123abc",
  "approval_status": "approved",
  "approved_by": "operator_name",
  "approval_timestamp": "2024-03-17T10:45:30Z"
}
```

#### POST /missions/{mission_id}/actuate
Execute approved actuation plan (if all safety checks pass).

**Response:**
```json
{
  "mission_id": "mission_abc123",
  "actuation_status": "executing",
  "zones_sprayed": 0,
  "zones_total": 245,
  "fluid_dispensed_ml": 0,
  "elapsed_seconds": 0
}
```

---

### Logging & Data Export

#### GET /missions/{mission_id}/logs
Get mission execution logs.

**Response:**
```json
{
  "mission_id": "mission_abc123",
  "log_file": "outputs/logs/mission_abc123.log",
  "recent_events": [
    {
      "timestamp": "2024-03-17T10:45:30Z",
      "level": "INFO",
      "message": "Detection complete: 247 hotspots"
    }
  ]
}
```

#### GET /missions/{mission_id}/export
Export mission data.

**Query Parameters:**
- `format`: Export format (geojson, shapefile, csv, json)
- `include`: What to include (detections, prescription, actuation, all)

**Response:**
Binary file download or JSON with download link.

---

## Error Responses

### 400 Bad Request
```json
{
  "detail": "Invalid request parameters"
}
```

### 404 Not Found
```json
{
  "detail": "Mission not found"
}
```

### 500 Internal Server Error
```json
{
  "detail": "An error occurred processing your request"
}
```

---

## Common Status Values

- **Mission Status**: pending, active, completed, failed, archived
- **Actuation Status**: idle, armed, spraying, paused, stopped, stopped_error, emergency_stop
- **Approval Status**: pending, approved, rejected, executed
- **Safety State**: SAFE, WARNING, FAULT, SHUTDOWN
