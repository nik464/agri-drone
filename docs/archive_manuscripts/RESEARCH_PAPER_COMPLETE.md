# AgriDrone AI: A Multi-Modal Explainable Deep Learning Framework for Real-Time Crop Disease Detection and Advisory in Indian Agriculture

---

**Authors:** Ashutosh Mishra  
**Affiliation:** Department of Computer Science and Engineering  
**Date:** April 2026  
**Keywords:** Precision Agriculture, YOLOv8, LLaVA, Explainable AI, Crop Disease Detection, Multi-Modal Ensemble, Knowledge Base, Rule Engine, Real-Time Inference

---

## Abstract

Early and accurate detection of crop diseases remains a critical challenge in Indian agriculture, where over 58% of the workforce depends on farming, yet annual post-harvest and disease-related losses exceed 15–25% of total production. We present **AgriDrone AI**, an end-to-end multi-modal explainable deep learning framework that integrates (i) a YOLOv8-based trained classifier for 21 crop disease classes across wheat and rice, (ii) a Vision Large Language Model (LLaVA) for zero-shot visual reasoning, (iii) a rule-based vision engine grounded in a curated agricultural knowledge base, and (iv) a safety-first ensemble voter that resolves conflicts between models with transparent reasoning chains. The system supports real-time inference via webcam and mobile phone integration, generates PDF field reports, provides multi-lingual voice advisory (Hindi, Tamil, Telugu, Punjabi, English), and operates entirely offline without cloud dependencies. We introduce a novel conflict resolution algorithm that prioritizes false-negative avoidance (missing a disease is costlier than a false alarm), seasonal risk adjustment using crop phenological stages, and spectral pseudo-indices (VARI, NDVI approximations from RGB imagery) to compensate for the absence of multi-spectral hardware. Deployed as a FastAPI backend with a React-based dashboard, AgriDrone AI demonstrates that combining lightweight deep learning models with structured domain knowledge and large vision-language models can produce farm-ready diagnostic tools with full explainability — a requirement increasingly demanded by agricultural extension services and regulatory bodies.

---

## 1. Introduction

### 1.1 Problem Statement

India's agricultural sector faces a persistent paradox: while producing enough grain to feed 1.4 billion people, it simultaneously loses an estimated ₹50,000 crore annually to crop diseases, pest infestations, and nutritional deficiencies that go undetected until visible yield loss occurs (ICAR, 2024). Wheat and rice — the two crops that constitute over 60% of India's food grain production — are particularly vulnerable to fungal rusts, blights, and bacterial infections that can reduce yields by 30–100% if undiagnosed during early growth stages.

Traditional disease diagnosis relies on human agricultural extension officers who physically visit farms, visually inspect plants, and prescribe treatments based on experience. This approach suffers from three critical limitations:

1. **Scalability**: India has approximately 146 million farm holdings but only ~100,000 agricultural extension workers — a ratio of 1:1,460 (NABARD, 2023).
2. **Timeliness**: By the time a farmer recognizes symptoms and contacts an expert, disease spread may have already caused irreversible damage.
3. **Accuracy**: Visual diagnosis by non-specialists has documented error rates of 30–50%, particularly for diseases with overlapping symptoms such as the three wheat rusts (yellow, brown, black) or the confusion between septoria leaf blotch and tan spot.

### 1.2 Limitations of Existing Approaches

Recent deep learning approaches for crop disease detection have achieved impressive benchmark accuracies (>95% on PlantVillage), but face significant deployment gaps:

- **Black-box predictions**: CNN/ViT classifiers output a class label and confidence score but provide no explanation for *why* a disease was identified, making it difficult for farmers and extension officers to trust or verify the diagnosis.
- **Single-model fragility**: A lone classifier trained on curated laboratory images often fails under field conditions — varying lighting, occlusion, mixed infections, and growth-stage variations introduce distribution shift.
- **No treatment pathway**: Most published systems stop at classification. Farmers need actionable treatment recommendations with specific fungicide names, dosages, and urgency levels.
- **Cloud dependency**: Systems relying on cloud APIs (GPT-4V, Google Gemini) are unusable in rural Indian fields with intermittent or absent internet connectivity.
- **Language barrier**: Advisory in English alone excludes the majority of Indian farmers who operate in Hindi, Punjabi, Tamil, or Telugu.

### 1.3 Our Contribution

AgriDrone AI addresses these gaps through the following contributions:

1. **Multi-modal ensemble architecture** combining a trained YOLOv8 classifier, a vision LLM (LLaVA), and a rule-based vision engine — each providing complementary diagnostic signals.
2. **Explainable reasoning chains** that trace every diagnostic decision from pixel-level features through rule firing to final diagnosis, satisfying the transparency requirement of Explainable AI (XAI).
3. **Safety-first conflict resolution** that prioritizes disease detection over healthy classification when models disagree, with formal guarantees against silent false negatives.
4. **A curated agricultural knowledge base** containing 21 disease profiles with 10 differential diagnosis rules and 9 seasonal risk stages, grounded in ICAR and CIMMYT literature.
5. **Complete offline deployment** using Ollama for local LLM inference, eliminating cloud dependency.
6. **Multi-lingual voice advisory** supporting 5 Indian languages for last-mile accessibility.
7. **Real-time universal object analysis** extending beyond crop diseases to general-purpose visual understanding via LLaVA.

---

## 2. Related Work

### 2.1 Deep Learning for Plant Disease Detection

The seminal work by Mohanty et al. (2016) demonstrated that deep CNNs could classify 26 diseases across 14 crop species with 99.35% accuracy on the PlantVillage dataset. Subsequent work employed transfer learning with ResNet (Ferentinos, 2018), InceptionV3 (Too et al., 2019), and EfficientNet (Atila et al., 2021), consistently achieving >95% top-1 accuracy on held-out test sets.

However, Barbedo (2019) critically evaluated these results and identified a fundamental problem: most datasets contain laboratory-captured single-leaf images against uniform backgrounds, leading to models that fail catastrophically on field-captured images with complex backgrounds, multiple leaves, and mixed infections. Our system addresses this through multi-model redundancy and rule-based visual verification that does not degrade under field conditions.

### 2.2 Object Detection in Agriculture

The YOLO family (Redmon et al., 2016; Jocher et al., 2023) has become the de facto standard for real-time agricultural object detection due to its single-pass architecture enabling inference at >30 FPS on edge devices. YOLOv8 (Jocher, 2023) introduced task-specific heads for detection, classification, and segmentation, making it suitable for both disease localization and image-level diagnosis.

Our system employs YOLOv8 in two configurations: (i) YOLOv8n-cls for 21-class whole-image classification (~50ms inference on CPU), and (ii) YOLOv8s for bounding-box detection with 11 rice disease classes trained on augmented field data.

### 2.3 Vision-Language Models in Agriculture

The emergence of Vision-Language Models (VLMs) such as LLaVA (Liu et al., 2023), GPT-4V (OpenAI, 2023), and Gemini (Google, 2024) has opened new possibilities for zero-shot agricultural analysis. These models can describe symptoms, infer disease progression, and generate treatment recommendations without task-specific training.

However, VLMs suffer from hallucination — confidently generating plausible but incorrect diagnoses. Our ensemble architecture mitigates this by cross-validating LLaVA outputs against both the trained classifier and the rule engine, accepting the VLM's diagnosis only when corroborated by at least one other signal.

### 2.4 Explainable AI in Agriculture

XAI techniques such as Grad-CAM (Selvaraju et al., 2017), LIME (Ribeiro et al., 2016), and SHAP (Lundberg & Lee, 2017) have been applied to agricultural models to generate saliency maps and feature attributions. While useful, these post-hoc explanations operate on model internals rather than domain-grounded reasoning.

Our approach differs fundamentally: we construct explanations *during* inference by matching extracted visual features (color signatures, texture patterns, spatial morphology) against disease-specific rules derived from plant pathology literature. This produces explanations in the language of agricultural science — "stripe pattern detected along leaf veins (+0.4 for yellow rust), no circular spots found (−0.2 for tan spot)" — rather than opaque heatmaps.

---

## 3. System Architecture

### 3.1 Overview

AgriDrone AI follows a modular client-server architecture deployed as two services:

- **Backend**: FastAPI (Python 3.11+) serving REST API endpoints over HTTP on port 9000, with WebSocket support for real-time streaming.
- **Frontend**: React 18 + Vite 5 single-page application on port 5173, communicating with the backend via auto-discovered API endpoints.

The diagnostic pipeline processes an input image through three parallel paths, merges their outputs via ensemble voting, enriches the result with knowledge-base context, and returns a structured diagnosis with full reasoning chain. Figure 1 illustrates the complete architecture.

```
                         ┌──────────────────────┐
                         │  Input Image (JPEG)   │
                         └──────────┬───────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
            ┌──────────────┐ ┌───────────┐ ┌──────────────────┐
            │  YOLOv8n-cls │ │   LLaVA   │ │ Feature Extractor│
            │  Classifier  │ │  (Ollama) │ │  + Rule Engine   │
            │  21 classes  │ │  7B VLM   │ │  + KB Lookup     │
            └──────┬───────┘ └─────┬─────┘ └────────┬─────────┘
                   │               │                │
                   │  Top-5 preds  │  JSON diagnosis │  CandidateScores
                   │  + softmax    │  + treatment    │  + rejections
                   │               │                │
                   └───────────────┼───────────────┘
                                   ▼
                        ┌────────────────────┐
                        │  Ensemble Voter    │
                        │  (Safety-First)    │
                        │  60% LLaVA + 40%   │
                        │  YOLO + Rule adj.  │
                        └────────┬───────────┘
                                 ▼
                    ┌────────────────────────┐
                    │  Reasoning Chain       │
                    │  + Differential Diag.  │
                    │  + Treatment Plan      │
                    │  + Spectral Indices    │
                    └────────────────────────┘
```

**Figure 1.** AgriDrone AI multi-modal diagnostic pipeline.

### 3.2 Backend Services

The FastAPI backend exposes 8 route groups with 15+ endpoints:

| Route Group | Prefix | Primary Function |
|---|---|---|
| Detection | `/api/detect/` | Image upload + full diagnostic pipeline |
| Chat Advisory | `/api/chat/` | Conversational follow-up with streaming SSE |
| Live Stream | `/api/stream/` | Real-time frame-by-frame analysis (30-frame window) |
| Universal Analyzer | `/api/universal/` | General-purpose object/scene analysis via LLaVA |
| Field History | `/api/field/` | GPS-zone disease progression tracking |
| Reports | `/api/reports/` | PDF field report generation and retrieval |
| Voice Interface | `/api/voice/` | Speech-to-text (Whisper) + text-to-speech (Piper) |
| Analysis | `/api/analysis/` | Field-level aggregated recommendations |

**Table 1.** Backend API route groups.

All endpoints support CORS for cross-origin frontend access. The backend uses lazy model loading — YOLO and classifier weights are loaded on first request and cached in memory, reducing cold-start overhead for subsequent inferences.

### 3.3 Frontend Dashboard

The React frontend comprises 24 specialized components organized into functional groups:

**Diagnosis Interface:**
- `UploadBox`: Drag-and-drop image upload with crop-type selector (wheat/rice/auto)
- `ResultViewer`: Multi-view diagnosis display with image toggle (Original / Grad-CAM / Healthy Reference), disease cards, treatment recommendations, and embedded AI chat
- `DetectionCanvas`: YOLO bounding-box overlay with confidence labels
- `UncertaintyMeter`: Visual confidence gauge for model agreement

**Real-Time Analysis:**
- `LiveStream`: Webcam/video stream with YOLO detection overlay, clickable bounding boxes for universal object analysis, rolling detection chart (30-frame window), and Deep Analyze (full-scene LLaVA) capability
- `YouTubeFrames`: Frame extraction from YouTube agricultural videos for analysis

**Field Management:**
- `FieldSessions` / `LiveSessions`: GPS-tagged scanning sessions
- `ReportsPage`: PDF report generation and history
- `MLDashboard`: Model performance metrics visualization
- `TrainingLogs`: YOLO training progress monitoring

**Accessibility:**
- `VoiceInterface`: Multi-lingual speech input/output
- `ChatPanel`: Conversational advisor with disease context
- `QRConnect`: QR code generation for phone-to-laptop field pairing

---

## 4. Multi-Modal Ensemble Pipeline

### 4.1 Model 1: YOLOv8n Crop Disease Classifier

#### Architecture

We employ YOLOv8n-cls (nano variant, ~5.2 MB) as the primary classifier due to its suitability for edge deployment. The model was trained on a combined wheat-rice dataset covering 21 classes:

**Wheat (16 classes):** Healthy Wheat, Fusarium Head Blight (FHB), Yellow/Stripe Rust, Black/Stem Rust, Brown/Leaf Rust, Powdery Mildew, Wheat Blast, Septoria Leaf Blotch, Wheat Leaf Blight, Tan Spot, Wheat Smut, Common Root Rot, Aphid Infestation, Mite Damage, Stem Fly

**Rice (7 classes):** Healthy Rice, Rice Blast, Bacterial Leaf Blight, Brown Spot, Sheath Blight, Rice Leaf Scald

#### Training Configuration

Training was conducted on Google Colab (NVIDIA T4 GPU) using the Ultralytics framework:

| Parameter | Value |
|---|---|
| Base model | YOLOv8n-cls (pretrained on ImageNet) |
| Epochs | 50 |
| Image size | 224 × 224 |
| Optimizer | Auto (SGD with momentum) |
| Learning rate | Cosine decay with 3-epoch warmup |
| Batch size | Auto-tuned |
| Max images/class | 300 (balanced sampling) |
| Augmentation | HSV shift (h=0.015, s=0.4, v=0.3), rotation (±15°), horizontal/vertical flip (p=0.5), mosaic (0.8), mixup (0.1), erasing (0.4) |

**Table 2.** Classifier training hyperparameters.

#### Output Format

The classifier returns top-5 predictions with softmax-calibrated confidence scores, enabling downstream analysis of prediction uncertainty:

```json
{
  "top_prediction": "wheat_fusarium_head_blight",
  "confidence": 0.87,
  "top5": [
    {"class": "wheat_fusarium_head_blight", "score": 0.87},
    {"class": "wheat_leaf_blight", "score": 0.06},
    {"class": "wheat_powdery_mildew", "score": 0.03},
    {"class": "wheat_smut", "score": 0.02},
    {"class": "healthy_wheat", "score": 0.01}
  ]
}
```

### 4.2 Model 2: LLaVA Vision-Language Model

#### Integration

We integrate LLaVA (Large Language and Vision Assistant, 7B parameters) via Ollama, an open-source local LLM runtime that enables fully offline inference without API keys or cloud connectivity — a critical requirement for rural deployment.

The model receives the input image alongside a structured diagnostic prompt engineered to produce calibrated JSON output:

```
You are an expert agricultural pathologist analyzing a {crop_type} crop image.
Assess: health_score (0-100), risk_level, diseases_found, confidence,
visible_symptoms, affected_area_pct, recommendations, urgency.

Calibration rules:
- FHB/Blast: health_score ≤ 50, risk = high/critical
- Healthy crop: health_score ≥ 85, risk = low
- Never classify a diseased crop as healthy
```

#### Safety Calibration

A key innovation is the **safety-calibrated prompt** that explicitly instructs the VLM to avoid false negatives. We enforce:

- **Severe disease floor**: FHB and Blast diagnoses are capped at health_score ≤ 50 regardless of VLM output
- **Healthy crop ceiling**: Crops classified as healthy must have health_score ≥ 85
- **Asymmetric error instruction**: The prompt states "Never call a diseased crop healthy" to bias the model toward sensitivity over specificity

#### Response Parsing

LLaVA outputs are parsed through a robust pipeline that handles:
1. Markdown code fence removal (`json` blocks)
2. Trailing comma correction in JSON
3. Regex-based fallback extraction for partially malformed responses
4. Default value injection for missing fields

### 4.3 Model 3: YOLO Object Detector

A separate YOLOv8s detection model provides bounding-box localization of disease symptoms within the image. Trained on rice disease data with the following per-class results:

| Class | AP50 | AP50-95 | Precision | Recall |
|-------|------|---------|-----------|--------|
| Brown Spot | 0.351 | 0.100 | 0.343 | 0.480 |
| Drainage Impact | 0.362 | 0.130 | 0.487 | 0.275 |
| Caterpillar | 0.304 | 0.088 | 0.333 | 0.167 |
| Grassy Stunt | 0.267 | 0.145 | 0.350 | 0.146 |
| Bacterial Blight | 0.041 | 0.016 | 0.071 | 0.091 |
| **Overall (mAP50)** | **0.189** | **0.069** | **0.226** | **0.166** |

**Table 3.** YOLO detector per-class performance on rice disease validation set.

The detection model serves a complementary role — localizing affected regions for visual explanation even when the classifier provides the primary diagnosis.

### 4.4 Ensemble Voting Strategy

The three model outputs are merged through a weighted voting algorithm with safety-first conflict resolution:

#### Agreement Case (Models Concur)
When the YOLO classifier and LLaVA agree on the diagnosis:

$$\text{health\_score}_{\text{ensemble}} = 0.6 \times \text{score}_{\text{LLaVA}} + 0.4 \times \text{score}_{\text{classifier}}$$

#### Uncertainty Case (Classifier Borderline)
When the classifier's top prediction is "healthy" but disease probability sum exceeds 30%:

$$\text{health\_score}_{\text{ensemble}} = 0.85 \times \text{score}_{\text{LLaVA}} + 0.15 \times \text{score}_{\text{classifier}}$$

This deliberately weights LLaVA higher because the VLM can reason about subtle symptoms that a classifier trained on 300 images/class may miss.

#### Disagreement Case (High-Confidence Conflict)
When models strongly disagree, the ensemble applies the **safety-first principle**:

$$\text{health\_score}_{\text{ensemble}} = \min(\text{score}_{\text{LLaVA}}, \text{score}_{\text{classifier}})$$

This ensures that if *either* model detects a disease, the final output reflects the concern rather than averaging it away.

#### Severe Disease Override
For high-severity diseases (Fusarium Head Blight, Wheat Blast, Stem Rust), an additional cap is enforced:

$$\text{health\_score}_{\text{ensemble}} \leq 55 \quad \text{if disease} \in \{\text{FHB, Blast, Stem Rust}\}$$

---

## 5. Explainable Vision Pipeline

### 5.1 Feature Extraction

The `FeatureExtractor` module converts a raw image into a structured `ImageFeatures` dataclass containing 18 quantitative descriptors across 6 categories:

| Feature Category | Extracted Attributes | Method |
|---|---|---|
| **Color Signatures** | Per-disease HSV pixel ratios, color confidence scores | HSV thresholding per KB color_signatures |
| **Texture** | Bleaching ratio, spot count, edge density | Canny edge detection, blob detection |
| **Spatial Morphology** | Stripe confidence, spot confidence, Hough line count | Morphological operations, ellipse fitting |
| **Directional Energy** | H-energy, V-energy, directionality ratio | Sobel gradients, directional energy analysis |
| **Saturation** | Vivid yellow-orange ratio, mean saturation, mean brightness | HSV channel statistics |
| **Greenness** | Green pixel ratio (healthy vegetation indicator) | HSV green range thresholding |

**Table 4.** Feature extraction categories and methods.

Each feature serves a specific diagnostic purpose. For example:
- **Stripe confidence > 0.6** strongly indicates rust diseases (pustules form along leaf veins)
- **Spot confidence > 0.5 with circular morphology** suggests tan spot or leaf blight
- **Bleaching ratio > 0.3** points to Fusarium Head Blight (characteristic bleached spikelets)
- **Green ratio < 0.2** indicates severe necrosis or senescence

### 5.2 Rule Engine

The `RuleEngine` evaluates all 21 disease candidates against the extracted features using a multi-criteria scoring system:

```
For each disease D in KnowledgeBase:
    score(D) = 0.0
    
    // Color rules
    IF color_signature_match(D) > threshold:
        score(D) += Δ_color(D)
    
    // Texture rules  
    IF D.expects_bleaching AND features.bleaching_ratio > 0.3:
        score(D) += 0.4
    IF D.expects_spots AND features.spot_count > 3:
        score(D) += 0.3
        
    // Spatial rules
    IF D.expects_stripes AND features.stripe_confidence > 0.6:
        score(D) += 0.4
    ELSE IF D.requires_stripes:
        score(D) -= 0.2  // Negative evidence
        
    // Saturation rules
    IF D.expects_vivid_yellow AND features.vivid_yellow_ratio > 0.2:
        score(D) += 0.3
        
    // Greenness rules
    IF features.green_ratio < 0.2 AND D.indicates_necrosis:
        score(D) += 0.2
        
    // Seasonal adjustment
    score(D) *= seasonal_multiplier(crop, growth_stage, month)
```

The rule engine produces:
- **Ranked candidate list** (all 21 diseases sorted by final score)
- **Fired rule explanations** (e.g., "Bleaching detected (+0.4 for FHB)")
- **Rejection reports** (e.g., "Tan spot rejected: no circular spots, stripe pattern contradicts")

### 5.3 Conflict Resolution

When the YOLO classifier and rule engine disagree on the top disease, the conflict resolver determines the winner:

| Condition | Winner | Rationale |
|---|---|---|
| YOLO confidence ≥ 0.75, rule confidence < 0.4 | YOLO | Strong model certainty overrides weak visual evidence |
| Rule confidence ≥ 0.8, YOLO uncertain | Rules | Strong visual evidence overrides uncertain model |
| Confidence gap > 0.35 | Higher score | Clear victor in either direction |
| Close scores | YOLO (slight preference) | Trained model as tiebreaker |

**Table 5.** Conflict resolution decision matrix.

### 5.4 Reasoning Chain Construction

The `DiseaseReasoning` orchestrator assembles a human-readable reasoning chain:

```json
{
  "reasoning_chain": [
    "Color analysis: vivid_yellow_ratio = 0.34 → consistent with Yellow Rust pustules",
    "Spatial pattern: stripe_confidence = 0.72 → linear pustule arrangement along veins",
    "Texture: spot_count = 2 (low) → inconsistent with Tan Spot (expected >5)",
    "Greenness: green_ratio = 0.45 → moderate healthy tissue remaining",
    "Seasonal risk: wheat tillering stage in March → Yellow Rust risk multiplier 1.2×",
    "Differential: Yellow Rust (0.82) vs Brown Rust (0.41) — stripe pattern differentiates",
    "YOLO classifier: wheat_yellow_rust (0.79) — AGREES with rule engine",
    "Final diagnosis: Yellow Rust with high confidence (ensemble: 0.81)"
  ]
}
```

### 5.5 Spectral Pseudo-Indices

To approximate vegetation indices normally requiring multi-spectral sensors, we compute RGB-derived pseudo-indices:

$$\text{VARI} = \frac{G - R}{G + R - B}$$

$$\text{NDVI}_{\text{pseudo}} = \frac{G - R}{G + R}$$

where $G$, $R$, $B$ are mean channel values from the image. While less accurate than true NDVI from NIR bands, these indices provide a useful vegetation vigour signal: VARI < 0.1 correlates with stressed or diseased vegetation, while VARI > 0.3 indicates healthy canopy.

---

## 6. Knowledge Base

### 6.1 Disease Profiles

The knowledge base (`diseases.json`, version 2.0.0) contains 21 structured disease profiles covering wheat (16 classes including healthy) and rice (7 classes including healthy). Each profile encodes:

| Field | Type | Description |
|---|---|---|
| `display_name` | string | Human-readable disease name |
| `type` | enum | `disease_fungal`, `disease_bacterial`, `pest_insect`, `healthy` |
| `severity` | float [0,1] | Base severity score |
| `symptoms` | list[string] | 4–5 observable symptom descriptions |
| `color_signatures` | list[HSV range] | Diagnostic color patterns in HSV space |
| `texture_keywords` | list[string] | Morphological descriptors (pustule, lesion, mottling) |
| `affected_parts` | list[string] | Plant organs (leaf, stem, head, spikelet, root) |
| `confusion_diseases` | list[string] | Commonly confused diseases for differential diagnosis |
| `treatment` | list[string] | Fungicide/pesticide recommendations with Indian brand names and dosages |
| `urgency` | enum | `immediate`, `within_7_days`, `within_30_days`, `none` |
| `yield_loss_pct` | string | Expected yield reduction range (e.g., "20-50%") |
| `pathogen` | string | Scientific name of causative organism |
| `favorable_conditions` | string | Environmental conditions promoting disease spread |

**Table 6.** Knowledge base disease profile schema.

### 6.2 Differential Diagnosis Rules

The KB includes 10 pairwise differential rules for commonly confused disease pairs. For example:

- **Yellow Rust vs. Brown Rust**: "Yellow rust forms linear stripes along veins; brown rust forms scattered circular pustules. Yellow rust appears in cooler temperatures (10–15°C) while brown rust prefers warmer conditions (15–25°C)."
- **FHB vs. Wheat Blast**: "Both cause bleaching, but FHB shows pink-orange fungal mass at spikelet base; Blast shows pyriform lesions with gray centers."

### 6.3 Seasonal Risk Adjustment

Nine seasonal risk stages are defined per crop, providing multipliers (0.5–1.2×) based on growth phenology:

| Growth Stage | Month (Wheat, North India) | High-Risk Diseases | Multiplier |
|---|---|---|---|
| Seedling | Nov–Dec | Root rot, damping off | 0.8× |
| Tillering | Jan–Feb | Yellow Rust, Powdery Mildew | 1.2× |
| Booting | Feb–Mar | Stem Rust, Leaf Blight | 1.1× |
| Heading | Mar–Apr | FHB, Blast, Smut | 1.2× |
| Maturity | Apr | Grain infections, storage pests | 0.9× |

**Table 7.** Seasonal risk adjustment matrix for wheat.

---

## 7. Universal Object Analyzer

### 7.1 Motivation

While the primary focus of AgriDrone AI is crop disease detection, field conditions often present non-agricultural objects (equipment, documents, people) in the camera view. Rather than treating these as noise, we extended the system with a **Universal Object Analyzer** that leverages LLaVA to identify and describe any object.

### 7.2 Auto-Routing Architecture

The universal analyzer automatically routes analysis requests based on object classification:

```
Input: Cropped image region + YOLO class label
          │
          ├─ class ∈ CROP_CLASSES → Plant Analysis prompt
          │   (detailed pathology: species, health_score, diseases, treatments)
          │
          ├─ class ∈ PERSON_CLASSES → Scene Description prompt
          │   (activity, setting, objects — no personal identification)
          │
          ├─ Green pixel % > 30% → Plant Analysis prompt
          │   (fallback for unrecognized vegetation)
          │
          └─ Otherwise → General Object prompt
              (object_name, category, brand, condition, use_case)
```

### 7.3 Deep Scene Analysis

A separate "Deep Analyze" endpoint processes the full camera frame through a comprehensive LLaVA prompt that catalogues:
- Scene type and environment assessment
- Every visible object with spatial position
- Visible text, logos, and signage
- Agricultural elements with health assessment
- Lighting conditions and dominant colors
- Actionable concerns and recommendations

---

## 8. Real-Time Streaming Pipeline

### 8.1 WebSocket-Based Live Analysis

The `LiveStream` component maintains a connection to the backend, streaming video frames at 4–5 FPS for real-time YOLO detection. The pipeline operates as:

1. **Capture**: Webcam frame → JPEG encoding (quality 85%)
2. **Send**: Base64-encoded frame via HTTP POST to `/api/stream/frame`
3. **Detect**: Backend runs YOLO detector → returns bounding boxes + class labels
4. **Render**: Frontend draws detection overlays on HTML5 Canvas with confidence labels
5. **Track**: Rolling 30-frame window maintains disease trend statistics

### 8.2 Interactive Object Analysis

Users can **click any bounding box** on the live stream to trigger deep analysis:

1. Click coordinates are mapped from canvas space to video coordinate space
2. The corresponding detection region is cropped with 10% padding
3. The cropped image is sent to `/api/universal/analyze-object`
4. LLaVA analyzes the object and returns structured results
5. Results are rendered in a context-appropriate view (ObjectView, PlantView, PersonView, or DeepSceneView)

### 8.3 Performance Optimization

To minimize LLaVA inference latency on CPU:
- Images are resized to max 384px (256px for person detection) before sending to the VLM
- JPEG quality is reduced to 60% for LLaVA input (sufficient for visual reasoning)
- The YOLO detector runs at native resolution for detection accuracy, while LLaVA receives downscaled crops

---

## 9. Voice Advisory System

### 9.1 Multi-Lingual Support

AgriDrone AI integrates speech interfaces to serve farmers in their native language:

| Language | Speech-to-Text | Text-to-Speech | Model |
|---|---|---|---|
| English | ✓ | ✓ | Whisper.cpp + Piper TTS |
| Hindi | ✓ | ✓ | Whisper (multilingual) + Piper |
| Tamil | ✓ | ✓ | Whisper + Piper |
| Telugu | ✓ | ✓ | Whisper + Piper |
| Punjabi | ✓ | ✓ | Whisper + Piper |

**Table 8.** Supported languages for voice advisory.

### 9.2 Voice Pipeline

```
Farmer speaks → Whisper STT → Disease query text
→ LLaVA/Chat advisor → Treatment response text
→ Piper TTS → Audio playback to farmer
```

All models run locally via Ollama/Whisper.cpp/Piper — no internet required.

---

## 10. Deployment Architecture

### 10.1 Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| Backend Framework | FastAPI 0.104+ | REST API + WebSocket |
| ASGI Server | Uvicorn 0.24+ | Async HTTP/WS server |
| Deep Learning | PyTorch 2.0+, Ultralytics 8.0+ | YOLO inference |
| Vision LLM | Ollama + LLaVA 7B | Zero-shot visual reasoning |
| Image Processing | OpenCV 4.8+, NumPy 1.24+ | Feature extraction |
| Geospatial | GeoPandas 0.14+, Rasterio 1.3+ | GPS zone tracking |
| Frontend | React 18, Vite 5, Tailwind CSS | Dashboard UI |
| Visualization | Plotly 5.17+, Matplotlib 3.8+ | Charts and reports |
| Data Validation | Pydantic 2.0+ | Request/response schemas |

**Table 9.** Technology stack.

### 10.2 System Requirements

- **Minimum**: 8 GB RAM, any modern CPU, 10 GB disk (models + data)
- **Recommended**: 16 GB RAM, NVIDIA GPU (4+ GB VRAM for fast LLaVA inference)
- **Network**: Not required for inference (fully offline); optional for QR phone pairing on LAN

### 10.3 Mobile Integration

Field workers can connect smartphones via QR code scanning, establishing a WebSocket link between the phone camera and the laptop-based backend. This enables:
- Real-time disease detection using the phone as a wireless camera
- Audio advisory played back through the phone speaker
- GPS coordinates from the phone tagged to each scan

---

## 11. Results and Evaluation

### 11.1 Detection Model Performance

The YOLOv8s detection model was evaluated on rice disease validation data:

| Metric | Value |
|--------|-------|
| mAP50 | 0.189 |
| mAP50-95 | 0.069 |
| Precision | 0.226 |
| Recall | 0.166 |

**Table 10.** Overall detection metrics.

The relatively modest detection performance is attributed to limited annotated bounding-box data and the inherent difficulty of localizing diffuse symptoms (e.g., leaf discoloration covering entire leaf surfaces). The multi-modal ensemble compensates by using the classifier and LLaVA for image-level diagnosis while the detector provides spatial context.

### 11.2 System Latency

| Component | Latency (CPU) | Latency (GPU) |
|---|---|---|
| YOLO Classifier (YOLOv8n-cls) | ~50 ms | ~8 ms |
| YOLO Detector (YOLOv8s) | ~150 ms | ~20 ms |
| Feature Extraction + Rule Engine | ~80 ms | ~80 ms |
| LLaVA Analysis | 30–60 s | 3–5 s |
| **Full Pipeline (without LLaVA)** | **~280 ms** | **~108 ms** |
| **Full Pipeline (with LLaVA)** | **~30–60 s** | **~3–5 s** |

**Table 11.** Component-level latency measurements.

The system achieves real-time performance (<300ms) for YOLO-based detection and classification. LLaVA analysis incurs significant latency on CPU but provides the richest diagnostic output including natural-language symptoms, treatment recommendations, and confidence assessments.

### 11.3 Knowledge Base Coverage

| Metric | Value |
|---|---|
| Total disease profiles | 21 |
| Crops covered | 2 (wheat, rice) |
| Wheat diseases | 15 + healthy |
| Rice diseases | 5 + healthy |
| Differential diagnosis pairs | 10 |
| Seasonal risk stages | 9 |
| Treatment recommendations | 21 (with Indian brand names) |
| Urgency levels | 4 (immediate, 7-day, 30-day, none) |

**Table 12.** Knowledge base coverage metrics.

---

## 12. Discussion

### 12.1 Strengths

**Explainability as a First-Class Requirement.** Unlike black-box CNN classifiers, every diagnosis from AgriDrone AI is accompanied by a reasoning chain tracing the decision from pixel-level features through rule evaluation to final ensemble verdict. This transparency is essential for agricultural extension officers who must justify treatment recommendations to farmers and for regulatory bodies requiring audit trails.

**Safety-First Design.** The asymmetric cost of errors in agriculture (missing a disease is far costlier than a false alarm) is explicitly encoded in the ensemble strategy. The min-score voting in disagreement cases, severe-disease health caps, and LLaVA's calibration prompt all bias the system toward sensitivity over specificity — a deliberate design choice grounded in agricultural risk economics.

**Complete Offline Operation.** By running LLaVA, Whisper, and Piper entirely through Ollama, the system requires zero internet connectivity for inference. This is a critical differentiator from cloud-dependent systems, as a significant portion of Indian farmland lacks reliable internet access.

**Multi-Modal Redundancy.** No single AI model is universally reliable. By combining three complementary signals — a trained classifier (pattern recognition), a VLM (semantic understanding), and a rule engine (domain knowledge) — the system maintains diagnostic accuracy even when individual models fail.

### 12.2 Limitations

**LLaVA Latency.** On CPU, LLaVA inference takes 30–60 seconds per analysis, which limits its utility for real-time click-to-analyze workflows. GPU acceleration reduces this to 3–5 seconds but requires NVIDIA hardware not always available in field settings.

**Detection Model Performance.** The YOLO detection model (mAP50 = 0.189) underperforms compared to the classifier, likely due to limited annotated bounding-box data for rice diseases. Future work should focus on expanding the detection training set.

**Single-Leaf Bias.** The classifier was trained primarily on single-leaf images. Performance on canopy-level drone imagery (multiple overlapping leaves, varying scale) has not been systematically evaluated.

**Knowledge Base Scope.** The current KB covers wheat and rice diseases prevalent in India. Extension to other crops (maize, cotton, sugarcane) and geographies requires additional disease profiles and regional treatment recommendations.

### 12.3 Future Work

1. **Lightweight VLM**: Replace LLaVA 7B with a distilled 1.5B vision model (e.g., SmolVLM, MiniCPM-V) for sub-5-second CPU inference.
2. **Federated Learning**: Enable on-device model updates from field-collected images without centralizing sensitive farm data.
3. **Drone Integration**: Process live drone video streams with altitude-aware detection models.
4. **Multi-Spectral Fusion**: When NIR camera data is available, fuse true NDVI with RGB features for improved stress detection.
5. **Expanded Crop Coverage**: Add maize, cotton, and sugarcane disease profiles to the knowledge base.
6. **SMS Fallback**: For areas without smartphones, enable disease diagnosis via SMS-submitted image descriptions matched against the knowledge base.

---

## 13. Conclusion

AgriDrone AI demonstrates that combining lightweight deep learning models with structured domain knowledge and vision-language models can produce an explainable, field-ready crop disease diagnostic system that operates entirely offline. The multi-modal ensemble architecture — with its safety-first voting strategy, knowledge-grounded rule engine, and transparent reasoning chains — addresses the key gaps in current agricultural AI: black-box predictions, single-model fragility, and cloud dependency.

The system currently supports 21 disease classes across wheat and rice, provides treatment recommendations with Indian brand-name fungicides and dosages, and is accessible in 5 languages through voice interfaces. The universal object analyzer extends the platform beyond crop diseases to general-purpose field analysis.

While challenges remain in detection model accuracy and VLM inference speed on commodity hardware, the architecture is designed for progressive enhancement — each model can be independently upgraded without affecting the ensemble framework. AgriDrone AI represents a step toward making precision agriculture accessible to smallholder farmers in developing nations, where the intersection of AI capability and practical deployment constraints demands thoughtful system design over raw benchmark performance.

---

## References

1. Atila, U., Ucar, M., Akyol, K., & Ucar, E. (2021). Plant leaf disease classification using EfficientNet deep learning model. *Ecological Informatics*, 61, 101182.

2. Barbedo, J. G. A. (2019). Plant disease identification from individual lesions and spots using deep learning. *Biosystems Engineering*, 180, 96-107.

3. Ferentinos, K. P. (2018). Deep learning models for plant disease detection and diagnosis. *Computers and Electronics in Agriculture*, 145, 311-318.

4. Jocher, G., Chaurasia, A., & Qiu, J. (2023). Ultralytics YOLO (Version 8.0.0). https://github.com/ultralytics/ultralytics

5. Liu, H., Li, C., Wu, Q., & Lee, Y. J. (2023). Visual instruction tuning. *Advances in Neural Information Processing Systems*, 36.

6. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30.

7. Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). Using deep learning for image-based plant disease detection. *Frontiers in Plant Science*, 7, 1419.

8. OpenAI. (2023). GPT-4V(ision) system card. Technical report.

9. Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2023). Robust speech recognition via large-scale weak supervision. *Proceedings of ICML 2023*.

10. Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You only look once: Unified, real-time object detection. *Proceedings of CVPR 2016*.

11. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?": Explaining the predictions of any classifier. *Proceedings of KDD 2016*.

12. Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. *Proceedings of ICCV 2017*.

13. Too, E. C., Yujian, L., Njuki, S., & Yingchun, L. (2019). A comparative study of fine-tuning deep learning models for plant disease identification. *Computers and Electronics in Agriculture*, 161, 272-279.

---

## Appendix A: Complete API Endpoint Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/detect/` | Full diagnostic pipeline (image upload) |
| GET | `/api/detect/health` | Backend liveness check |
| POST | `/api/detect/reset` | Clear model caches |
| POST | `/api/chat/` | Streaming advisory chat (SSE) |
| POST | `/api/chat/sync` | Synchronous advisory chat |
| POST | `/api/stream/frame` | Process single video frame |
| POST | `/api/universal/analyze-object` | Analyze any object via LLaVA |
| POST | `/api/universal/deep-analyze` | Full-scene deep analysis |
| GET | `/api/field/history` | GPS zone scan history |
| GET | `/api/field/progression` | Disease trend analysis |
| GET | `/api/field/zones` | List tracked GPS zones |
| POST | `/api/reports/generate` | Generate PDF report |
| GET | `/api/reports/list` | List generated reports |
| GET | `/api/reports/download/{file}` | Download PDF report |
| POST | `/api/voice/transcribe` | Speech-to-text (Whisper) |
| POST | `/api/voice/speak` | Text-to-speech (Piper) |
| GET | `/api/voice/status` | Voice model status |
| POST | `/api/analysis/recommend` | Field-level recommendations |

**Table A1.** Complete API endpoint reference.

## Appendix B: Disease Knowledge Base Summary

| # | Disease Key | Crop | Severity | Yield Loss | Urgency |
|---|---|---|---|---|---|
| 1 | healthy_wheat | Wheat | 0.0 | 0% | none |
| 2 | wheat_fusarium_head_blight | Wheat | 0.9 | 20-50% | immediate |
| 3 | wheat_yellow_rust | Wheat | 0.85 | 40-100% | immediate |
| 4 | wheat_black_rust | Wheat | 0.8 | 30-70% | immediate |
| 5 | wheat_brown_rust | Wheat | 0.75 | 15-40% | within_7_days |
| 6 | wheat_powdery_mildew | Wheat | 0.6 | 10-30% | within_7_days |
| 7 | wheat_blast | Wheat | 0.85 | 40-100% | immediate |
| 8 | wheat_septoria | Wheat | 0.7 | 15-30% | within_7_days |
| 9 | wheat_leaf_blight | Wheat | 0.7 | 10-25% | within_7_days |
| 10 | wheat_tan_spot | Wheat | 0.6 | 5-20% | within_30_days |
| 11 | wheat_smut | Wheat | 0.65 | 5-30% | within_7_days |
| 12 | wheat_root_rot | Wheat | 0.7 | 10-30% | within_7_days |
| 13 | wheat_aphid | Wheat | 0.55 | 5-25% | within_7_days |
| 14 | wheat_mite | Wheat | 0.5 | 3-15% | within_30_days |
| 15 | wheat_stem_fly | Wheat | 0.5 | 5-30% | within_7_days |
| 16 | healthy_rice | Rice | 0.0 | 0% | none |
| 17 | rice_blast | Rice | 0.9 | 30-100% | immediate |
| 18 | rice_bacterial_blight | Rice | 0.8 | 20-50% | immediate |
| 19 | rice_brown_spot | Rice | 0.6 | 10-30% | within_7_days |
| 20 | rice_sheath_blight | Rice | 0.7 | 10-30% | within_7_days |
| 21 | rice_leaf_scald | Rice | 0.65 | 5-20% | within_30_days |

**Table B1.** Complete disease knowledge base summary.

## Appendix C: Frontend Component Architecture

```
App.jsx
├── Navbar.jsx (top bar, API status, clock)
├── Sidebar.jsx (navigation: 12 sections)
├── StatsCards.jsx (scan count, diseases, avg health)
├── UploadBox.jsx → triggers POST /api/detect/
│   └── ResultViewer.jsx
│       ├── Image Toggle (Original / Grad-CAM / Healthy Ref)
│       ├── Disease Card (name, confidence, severity)
│       ├── Treatment Card (fungicides, dosages, urgency)
│       ├── AI Consensus Panel (ensemble agreement)
│       ├── Spectral Analysis (VARI, NDVI pseudo)
│       ├── Top-5 Predictions (classifier output)
│       └── ChatPanel.jsx (embedded advisor)
├── LiveStream.jsx
│   ├── Video Canvas (YOLO overlay, clickable boxes)
│   ├── Stream Metrics (FPS, latency, detections)
│   ├── Detection Chart (30-frame rolling)
│   ├── Analysis Panel (ObjectView / PlantView / PersonView / DeepSceneView)
│   └── Deep Analyze Button
├── ReportsPage.jsx (PDF generation + history)
├── MLDashboard.jsx (model performance)
├── TrainingLogs.jsx (training progress)
├── VoiceInterface.jsx (multi-lingual audio)
├── QRConnect.jsx (phone pairing)
└── ActivityFeed.jsx (event log)
```

**Figure C1.** Frontend component hierarchy.
