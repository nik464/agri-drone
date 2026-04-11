# AgriDrone — Setup From Scratch

> Everything you need to clone the repo and run the full system on a fresh computer.

---

## What's NOT on GitHub (Too Large or Secret)

GitHub has the full source code, configs, evaluation scripts, frontend, and documentation. The following files are **gitignored** and must be downloaded separately.

### A) Model Files (.pt) — Upload to Google Drive

| File | Size | Purpose | Required? |
|------|------|---------|-----------|
| `models/india_agri_cls.pt` | 2.83 MB | **Primary 21-class disease classifier** (API + all eval scripts) | **YES** |
| `models/india_agri_cls_21class_backup.pt` | 2.88 MB | Backup 21-class classifier (some eval scripts) | **YES** |
| `models/yolov8n-seg.pt` | 6.74 MB | Segmentation model (field/crop detection) | YES (auto-downloads if missing) |
| `models/yolo_crop_disease.pt` | 21.48 MB | YOLO crop disease detector (API fallback) | Optional |
| `models/efficientnet_b0_21class.pt` | 15.68 MB | EfficientNet baseline (training script only) | Optional |
| `models/wheat_cls_v1.pt` | 2.83 MB | Legacy wheat classifier | Optional |
| `yolov8n-cls.pt` (root) | 5.31 MB | YOLOv8 classification pretrained | Optional |
| `yolov8n.pt` (root) | 6.25 MB | YOLOv8 detection pretrained | Optional |

**Minimum required:** `india_agri_cls.pt` + `india_agri_cls_21class_backup.pt` (~6 MB total)

### B) Dataset Folders — Upload to Google Drive

| Folder | Size | Contents | Required? |
|--------|------|----------|-----------|
| `data/training/` | 5.2 GB | Train/val/test splits (4364+935+935 images, 21 classes) | **YES** (to run eval scripts) |
| `datasets/externals/PDT_datasets/` | 6.1 GB | External PDT UAV dataset (21,332 images) | Only for cross-dataset eval |
| `data/raw/` | 8.8 GB | Raw unprocessed images | NO (not used at runtime) |
| `data/wheat_raw/` + `data/wheat_annotated/` | 2.8 GB | Wheat annotation data | NO (not used at runtime) |

**Minimum required:** `data/training/` (~5.2 GB) — needed for ablation study and evaluation

### C) Environment Files

No secrets or API keys are needed. The system uses:
- `.env.example` — copy to `.env` (all defaults work out of the box)
- `dashboard/.env.example` — copy to `dashboard/.env` (just sets `VITE_API_URL=http://localhost:8000`)

### D) External Software (Not in Repo)

| Software | Purpose | Required? |
|----------|---------|-----------|
| **Ollama** + **LLaVA model** | AI chat and image analysis in the dashboard | Optional (system works without it; falls back to rule-based responses) |
| **CUDA / cuDNN** | GPU acceleration for YOLO inference | Recommended (CPU works but slower) |
| **Node.js 18+** | Frontend dashboard | Only if running the React frontend |

---

## Google Drive Folder Structure

Upload these files to a shared Google Drive folder:

```
AgriDrone_Assets/
├── models/
│   ├── india_agri_cls.pt              (2.83 MB)  ← REQUIRED
│   ├── india_agri_cls_21class_backup.pt (2.88 MB)  ← REQUIRED
│   ├── yolov8n-seg.pt                 (6.74 MB)
│   ├── yolo_crop_disease.pt           (21.48 MB)
│   └── efficientnet_b0_21class.pt     (15.68 MB)
├── data_training.zip                  (~5.2 GB)   ← REQUIRED for eval
│   └── training/
│       ├── train/  (4364 images)
│       ├── val/    (935 images)
│       └── test/   (935 images)
└── datasets_PDT.zip                   (~6.1 GB)   ← Optional
    └── PDT_datasets/
```

---

## Fresh Computer Setup — Step by Step

### 1. Clone the Repository

```bash
git clone https://github.com/Ashut0sh-mishra/agri-drone.git
cd agri-drone
```

### 2. Set Up Python Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Download Models from Google Drive

Download the model files and place them in the `models/` directory:

```bash
mkdir models

# Place these files (download from Google Drive):
# models/india_agri_cls.pt
# models/india_agri_cls_21class_backup.pt
# models/yolov8n-seg.pt  (or let Ultralytics auto-download it)
```

### 4. Download Datasets from Google Drive (For Evaluation)

```bash
# Extract data_training.zip into the data/ folder:
# data/training/train/  (4364 images, 21 class folders)
# data/training/val/    (935 images)
# data/training/test/   (935 images)

# Optional: Extract PDT dataset
# datasets/externals/PDT_datasets/
```

### 5. Set Up Environment Variables

```bash
# Copy the example env file (defaults work out of the box)
cp .env.example .env
```

### 6. Start the Backend API

```bash
cd agri-drone
python -m uvicorn src.agridrone.api.app:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`. API docs at `http://localhost:8000/docs`.

### 7. Set Up the Frontend Dashboard

```bash
cd frontend
npm install
cp ../.env.example .env   # or create with: VITE_API_URL=http://localhost:8000
npm run dev
```

The dashboard will be available at `http://localhost:5173`.

### 8. (Optional) Install Ollama + LLaVA for AI Chat

The dashboard's chat and AI analysis features use Ollama with the LLaVA model. Without Ollama, the system still works — it falls back to rule-based responses.

```bash
# Install Ollama (Windows)
winget install Ollama.Ollama

# Or download from https://ollama.com/download

# Pull the LLaVA model (~4.7 GB)
ollama pull llava

# Verify it's running
ollama list
# Should show: llava:latest
```

Ollama runs on `http://localhost:11434` by default. The AgriDrone API connects to it automatically.

### 9. (Optional) GPU Setup for Faster Inference

For NVIDIA GPU acceleration:
1. Install [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (11.8 or 12.x)
2. Install PyTorch with CUDA: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`
3. Verify: `python -c "import torch; print(torch.cuda.is_available())"`

Without a GPU, the system runs on CPU (inference takes ~50ms instead of ~15ms per image).

---

## Running the Evaluation Scripts

All evaluation scripts are in the `evaluate/` folder. They require the test dataset in `data/training/test/`.

```bash
# Ablation study (Config A/B/C comparison)
python evaluate/ablation_study.py --model-path models/india_agri_cls_21class_backup.pt

# Statistical tests (McNemar, bootstrap CIs)
python evaluate/statistical_tests.py

# Sensitivity analysis (125 weight configurations)
python evaluate/sensitivity_analysis.py

# Cross-dataset evaluation on PDT (requires PDT dataset)
python evaluate/pdt_cross_eval.py \
    --dataset-dir "datasets/externals/PDT_datasets/PDT dataset/PDT dataset" \
    --model-path models/india_agri_cls.pt

# EML analysis
python evaluate/eml_analysis.py

# Generate paper tables
python evaluate/paper_tables.py
```

---

## Quick Verification

After setup, verify everything works:

```bash
# 1. Check models load
python -c "from ultralytics import YOLO; m = YOLO('models/india_agri_cls.pt', task='classify'); print('Model loaded:', m.model.names)"

# 2. Check API starts
python -m uvicorn src.agridrone.api.app:app --host 0.0.0.0 --port 8000 &
curl http://localhost:8000/health

# 3. Run the 4-image smoke test
python evaluate/test_4_images.py

# 4. Check Ollama (optional)
curl http://localhost:11434/api/tags
```

---

## Summary

| What | Where | Size |
|------|-------|------|
| Source code + configs | GitHub repo | ~3 MB |
| Model weights (required) | Google Drive → `models/` | ~6 MB |
| Training dataset | Google Drive → `data/training/` | ~5.2 GB |
| PDT dataset (optional) | Google Drive → `datasets/externals/` | ~6.1 GB |
| Ollama + LLaVA (optional) | `ollama pull llava` | ~4.7 GB |
| Python dependencies | `pip install -r requirements.txt` | ~2 GB |
| Node.js dependencies | `cd frontend && npm install` | ~200 MB |
