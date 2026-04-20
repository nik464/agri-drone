<div align="center">

# 🌾 AgriDrone

### A Negative Result on Hand-Authored Symbolic Rules for CNN-Based Crop-Disease Classification

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple.svg)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: Research Prototype](https://img.shields.io/badge/status-research%20prototype-orange.svg)](CHANGELOG_RESEARCH_UPGRADE.md)
[![Paper](https://img.shields.io/badge/Paper-Manuscript-red.svg)](MANUSCRIPT_SUBMISSION_VERSION.md)

**Live Demo:** [agri-drone-frontend.vercel.app](https://agri-drone-frontend.vercel.app) | **API:** [ashu010-agri-drone-api.hf.space](https://ashu010-agri-drone-api.hf.space)

</div>

---

> **Project Status: Research Prototype.** This is a research codebase, not a
> production system. All reported numbers are validated on curated close-up
> leaf photographs; they are **not** a claim of field-ready drone deployment.
> See [Repository Status and Evidence Scope](#repository-status-and-evidence-scope) and
> [Known Limitations](#known-limitations) below.

AgriDrone is a research codebase that combines a YOLOv8n-cls classifier (1.44M parameters), a multi-stage symptom reasoning engine, and an expected monetary loss (EML) estimator for **21 Indian wheat and rice disease classes**. Through a three-configuration ablation study on **934** curated close-up leaf images, we show that the standalone YOLO classifier (96.15% accuracy, 15 ms) is statistically indistinguishable from the full hybrid pipeline (95.72%, 444 ms; McNemar p = 0.134) — the rule engine adds 29× latency with zero accuracy gain and inflates EML by 840%. The core message: **ablate before you complicate.**

## Key Results

| Configuration | Accuracy | Macro-F1 | MCC | Latency |
|:---|:---:|:---:|:---:|:---:|
| **Config A** — YOLO-only | **96.15%** | **0.962** | **0.960** | **15 ms** |
| **Config B** — YOLO + Rules | 95.72% | 0.957 | 0.955 | 444 ms |
| **Config C** — Rules-only | 13.38% | 0.077 | 0.096 | 392 ms |

- Test set: **934 images**, 21 classes (n=934 in manifest; 933 successfully predicted)
- McNemar's test: χ² = 2.25, *p* = 0.134 (A vs B — **not significant**)
- Override analysis: 7 total A→B prediction changes — 0 rescues, 4 corruptions, 3 neutral
- EML: ₹294.33 (Config A) vs ₹2,769.06 (Config B) — **+840% cost gap**
- No train/test leakage detected (see `LEAKAGE_INVESTIGATION_REPORT.md`)

## Repository Status and Evidence Scope

**This repository supports ONE fully executed experiment and several planned-but-unexecuted extensions.**

### Executed and validated (safe to cite)
| Experiment | Evidence files |
|---|---|
| Three-config ablation (A/B/C) on 934 images | `evaluate/results/predictions_{A,B,C}_*.csv`, `ablation_summary.json` |
| Bootstrap CIs + McNemar (10,000 replicates) | `evaluate/results/statistical_tests.json` |
| EML cost analysis (ICAR/DAE cost matrix) | `evaluate/results/eml_summary.json` |
| Override decomposition (7 overrides analyzed) | `evaluate/results/override_decomposition.json` |
| Cross-dataset PDT (672 images, 84.4% acc) | `evaluate/results/cross_dataset_PDT.json` |
| 125-config weight sensitivity | `evaluate/results/sensitivity_results.csv` |
| Robustness / noise pipeline | `evaluate/results/robustness_summary.json` |
| Leakage investigation (no leakage found) | `LEAKAGE_INVESTIGATION_REPORT.md` |

### Planned but NOT executed (infrastructure only)
| Experiment | Status |
|---|---|
| 54-cell multi-backbone matrix (`configs/matrix/large.yaml`) | Config exists; **no training runs executed** |
| 2400-cell full matrix (`configs/matrix/full.yaml`) | Dry-run stubs only (`evaluate/results/v2/matrix/`) |
| Fair EfficientNet-B0 baseline audit | Dry-run placeholder |
| Dietterich 5×2cv test | Null placeholder |
| Paper-2 45-cell matrix (`configs/matrix/paper2.yaml`) | Not started |

### Quarantined artefacts (broken/misleading)
See `evaluate/results/_quarantined/README.md` for details.

## Current Evidence-Backed Contribution

This repository currently supports a **narrow negative-result finding**: on a 934-image, 21-class curated leaf-photo test set with a single YOLOv8n-cls backbone trained at seed 42, a substantive hand-authored HSV + texture + spatial rule engine adds no measurable accuracy and increases cost by 840% under a smallholder EML model. This is a single datapoint — useful because negative results are systematically under-reported in agri-CV, but it requires multi-backbone and multi-seed validation before generalizing.

The repository also contains substantial infrastructure (matrix runner, cost model, dashboard, multiple rule engine variants) designed for a larger study that has not yet been executed.

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Layer 1: INPUT                        │
│  Drone camera (RGB) → JPEG upload → FastAPI /detect      │
└────────────────────────┬────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│                Layer 2: PERCEPTION                       │
│  YOLOv8n-cls → top-5 class probabilities                 │
│  1.44M params │ 3.4 GFLOPs │ 224×224 input               │
│  21 classes: 14 wheat + 5 rice + 2 healthy               │
└────────────────────────┬────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│                Layer 3: REASONING                        │
│  Feature Extractor → 20+ visual metrics                  │
│  Rule Engine → 6 scoring rules + conflict resolution     │
│  Spectral Indices → VARI, RGRI, GLI                      │
│  Ensemble Voter → Bayesian fusion (0.70/0.30)            │
└────────────────────────┬────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│                Layer 4: DECISION                         │
│  Confidence-based YOLO auto-win (≥0.95)                  │
│  Grad-CAM attention visualisation                        │
└────────────────────────┬────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│              Layer 5: PRESCRIPTION                       │
│  Treatment lookup │ Yield-loss estimation │ EML           │
└────────────────────────┬────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│              Layer 6: PRESENTATION                       │
│  React dashboard │ Grad-CAM heatmap │ Reasoning chain    │
│  Differential diagnosis │ JSON/CSV export                │
└─────────────────────────────────────────────────────────┘
```

## Disease Classes (21)

| Category | Classes |
|:---|:---|
| **Wheat diseases (14)** | aphid, black rust, blast, brown rust, Fusarium head blight, leaf blight, mite, powdery mildew, root rot, septoria, smut, stem fly, tan spot, yellow rust |
| **Rice diseases (5)** | bacterial blight, blast, brown spot, leaf scald, sheath blight |
| **Healthy (2)** | healthy wheat, healthy rice |

## Repository Structure

```
agri-drone/
├── src/agridrone/              # Core Python package
│   ├── api/                    #   FastAPI routes & schemas
│   ├── vision/                 #   YOLO inference, rule engine, Grad-CAM, ensemble
│   ├── core/                   #   Detector, spectral features, yield estimator
│   ├── knowledge/              #   Disease knowledge base (JSON)
│   ├── prescription/           #   Treatment recommendation rules
│   ├── geo/                    #   Geospatial referencing & grid mapping
│   ├── io/                     #   Image/sensor/telemetry loaders, exporters
│   ├── environment/            #   Environmental feature fusion
│   ├── feedback/               #   Correction aggregation & KB updates
│   ├── services/               #   LLM service, report generation
│   ├── types/                  #   Pydantic data models
│   ├── config.py               #   Application settings
│   └── logging.py              #   Loguru logging config
├── evaluate/                   # Evaluation & ablation scripts
│   ├── ablation_study.py       #   Three-config ablation (A/B/C)
│   ├── statistical_tests.py    #   Bootstrap CIs, McNemar's test
│   ├── pdt_cross_eval.py       #   Cross-dataset PDT evaluation
│   ├── sensitivity_analysis.py #   125-config weight sweep
│   ├── eml_analysis.py         #   Expected monetary loss
│   ├── test_4_images.py        #   Pipeline verification
│   └── results/                #   JSON, CSV, PNG outputs
├── scripts/                    # Utility scripts
│   ├── run_inference.py        #   Single/batch inference
│   ├── train_model.py          #   YOLOv8 training
│   └── dashboard.py            #   Web UI launcher
├── configs/                    # YAML configuration files
├── tests/                      # pytest unit & integration tests
├── dashboard/                  # React + Vite + TailwindCSS frontend
├── models/                     # Model weights (see Data Availability)
├── data/                       # Dataset splits (see Data Availability)
├── notebooks/                  # Jupyter notebooks
├── docs/                       # Additional documentation
├── MANUSCRIPT_SUBMISSION_VERSION.md # Canonical submission manuscript
├── pyproject.toml              # Package configuration
├── requirements.txt            # Python dependencies
└── .env.example                # Environment variable template
```

## Screenshots

### AgriDrone Dashboard — Live Disease Detection

<!-- <p align="center">
  <img src="docs/screenshots/dashboard_1.png" alt="Dashboard - Detection View 1" width="90%"/>
</p>
<p align="center">
  <img src="docs/screenshots/dashboard_2.png" alt="Dashboard - Detection View 2" width="90%"/>
</p>
<p align="center">
  <img src="docs/screenshots/dashboard_3.png" alt="Dashboard - Detection View 3" width="90%"/>
</p> -->

> Real-time wheat and rice disease detection with Grad-CAM explainability, field health score, confidence breakdown, AI reasoning chain, and treatment recommendations.

## Installation

### Prerequisites

- Python 3.11+
- NVIDIA GPU with CUDA support (recommended)
- Node.js 18+ (for frontend dashboard)

### Backend Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/agri-drone.git
cd agri-drone

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Copy environment config
copy .env.example .env        # Windows
# cp .env.example .env        # Linux/macOS
```

### Download Model Weights

Model weights are hosted separately due to file size. Download from the links in [Data Availability](#data-availability) and place them in `models/`:

```
models/
├── india_agri_cls_21class_backup.pt   # 21-class classifier (2.88 MB)
├── india_agri_cls.pt                  # 4-class wheat classifier (2.83 MB)
├── efficientnet_b0_21class.pt         # EfficientNet backbone (15.68 MB)
└── yolo_crop_disease.pt               # Detection model (21.48 MB)
```

### Frontend Setup (Optional)

```bash
cd dashboard
npm install
npm run dev
```

## Usage

### Start the API Server

```bash
# From project root
uvicorn src.agridrone.api.app:app --host 0.0.0.0 --port 8000 --reload
```

### Single Image Inference

```bash
python scripts/run_inference.py --image path/to/image.jpg --model models/india_agri_cls_21class_backup.pt
```

### Launch Dashboard

```bash
python scripts/dashboard.py
```

The React dashboard will be available at `http://localhost:5173` with the API at `http://localhost:8000`.

## Reproducing the Ablation Study

All evaluation scripts are in `evaluate/`. Results are written to `evaluate/results/`.

```bash
# 1. Three-configuration ablation (Config A, B, C)
python evaluate/ablation_study.py \
    --model-path models/india_agri_cls_21class_backup.pt

# 2. Statistical tests (Bootstrap CIs + McNemar's)
python evaluate/statistical_tests.py \
    --results-dir evaluate/results --n-boot 10000

# 3. Cross-dataset evaluation on PDT
python evaluate/pdt_cross_eval.py \
    --dataset-dir datasets/externals/PDT_datasets/"PDT dataset"/"PDT dataset" \
    --model-path models/india_agri_cls.pt

# 4. Sensitivity analysis (125 weight configurations)
python evaluate/sensitivity_analysis.py

# 5. Expected monetary loss analysis
python evaluate/eml_analysis.py

# 6. Pipeline verification (4-image test)
python evaluate/test_4_images.py
```

### Expected Outputs

| Script | Key Output Files |
|:---|:---|
| `ablation_study.py` | `ablation_summary.json`, `ablation_table.csv`, confusion matrices (PNG) |
| `statistical_tests.py` | `statistical_tests.json` |
| `pdt_cross_eval.py` | `cross_dataset_PDT.json`, `cross_dataset_PDT_predictions.csv` |
| `sensitivity_analysis.py` | `sensitivity_summary.json`, `sensitivity_grid.csv` |
| `eml_analysis.py` | `eml_summary.json`, `eml_comparison.csv`, `eml_bar_chart.png` |

## Dataset

### Primary Dataset (21 classes)

| Split | Images | Per class (approx.) |
|:---|:---:|:---:|
| Train | 4,364 | ~208 |
| Validation | 935 | ~45 |
| Test | 934 | ~45 |

Stratified 70/15/15 split with seed = 42.

### External Dataset: Plant Disease Treatment (PDT)

- 672 images (105 healthy LH + 567 unhealthy LL)
- Binary healthy/unhealthy classification
- Significant domain shift (close-up training → whole-field aerial)

## Data Availability

| Resource | Location | DOI |
|:---|:---|:---|
| Source code | This repository | — |
| Model weights | [Google Drive / Zenodo] | [DOI pending] |
| Primary dataset | [Google Drive / Zenodo] | [DOI pending] |
| PDT dataset | [Publicly available](https://github.com/) | See original source |
| Evaluation results | `evaluate/results/` in this repo | — |

> **Note:** Update the links above after uploading to Google Drive and/or Zenodo.

## Citation

If you use AgriDrone in your research, please cite:

```bibtex
@article{agridrone2026,
  title     = {A Negative Result on Hand-Authored Symbolic Rules for
               CNN-Based Crop-Disease Classification},
  author    = {Mishra, Ashutosh},
  year      = {2026},
  note      = {Workshop paper in preparation}
}
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for the classification backbone
- [Grad-CAM](https://github.com/jacobgil/pytorch-grad-cam) for explainability visualisations
- The Plant Disease Treatment (PDT) dataset authors for external validation data

---

<div align="center">

**[Paper](MANUSCRIPT_SUBMISSION_VERSION.md) · [Installation](#installation) · [Reproduce Results](#reproducing-the-ablation-study) · [Citation](#citation)**

</div>


## Known Limitations

- **Single backbone, single seed.** All headline results use YOLOv8n-cls at seed 42.
  Multi-backbone comparison is planned but not executed.
- **Small test set.** n = 934 images, 20–45 per class. Adequate for a workshop
  paper; insufficient for strong per-class claims.
- **Lab-quality imagery.** Curated smartphone leaf photos, not field or drone imagery.
  The "drone" in the project name describes the intended deployment, not the
  evaluation distribution.
- **No multiple-comparison correction.** 21 per-class F1 comparisons lack
  formal Holm-Bonferroni correction. The quarantined correction was broken;
  regeneration is pending.
- **Cross-dataset PDT result.** The 84.4% headline collapses to constant
  "unhealthy" at argmax (specificity = 0%). See threshold sweep in
  `evaluate/results/v2/pdt/threshold_sweep.json`.
- **EfficientNet baseline.** The 76.15% number used unequal training settings.
  The fair re-audit (`evaluate/matrix/audit_baseline.py`) is a dry-run placeholder.
- **Cost table sensitivity.** EML headline depends on chosen cost parameters.
  See `configs/economics/india_2025.yaml`.

## Future Work

1. Execute `configs/matrix/paper2.yaml` (45 cells: 5 backbones × 3 geographies × 3 folds) on Colab/Kaggle.
2. Collect n ≥ 2,000 additional test images for adequate statistical power.
3. Regenerate Holm-Bonferroni per-class McNemar from committed A/B CSVs.
4. Run fair EfficientNet-B0 audit under shared training recipe.
5. Evaluate learned-rule and LLM-rule variants against the test set.
6. Collect field-condition drone imagery for realistic deployment evaluation.
