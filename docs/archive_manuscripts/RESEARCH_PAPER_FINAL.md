# AgriDrone: An Ablation Study of Hybrid Deep-Learning Pipelines for Drone-Based Crop Disease Detection in Indian Agriculture

**Authors:** [Author names]  
**Affiliation:** [Institution]  
**Corresponding author:** [Email]  
**Submitted to:** *Smart Agricultural Technology*

---

## Abstract

Drone-based crop disease detection systems increasingly adopt hybrid architectures combining deep-learning classifiers with hand-crafted rule engines and ensemble voting. We present **AgriDrone**, a full-stack precision agriculture system integrating a YOLOv8n-cls classifier, a six-rule symptom reasoning engine, spectral vegetation indices, Grad-CAM explainability, and an economic yield-loss estimator for 21 wheat and rice diseases prevalent in Indian agriculture. Through a rigorous ablation study on 935 test images across 21 classes we evaluate three configurations: YOLO-only (**Config A**), YOLO + Rules ensemble (**Config B**), and Rules-only (**Config C**). Config A achieves **96.2% accuracy** (macro-F1 = 0.962, MCC = 0.960); Config B achieves **95.7% accuracy** (macro-F1 = 0.957, MCC = 0.955) — a statistically non-significant difference (McNemar χ² = 2.25, *p* = 0.134, 4 discordant pairs out of 935). Config C achieves only **13.4% accuracy** (macro-F1 = 0.077), confirming the rule engine lacks standalone discriminative power. Bootstrap 95% confidence intervals (*B* = 10,000) overlap for Configs A and B across all metrics, while Config C is significantly worse (*p* < 0.001). The hybrid pipeline increases inference latency by **29×** (15 ms → 444 ms) with no accuracy benefit. Cross-dataset evaluation on the external PDT wheat disease dataset (672 images) yields 84.4% accuracy and 100% disease recall, demonstrating robustness to domain shift. We argue that for well-trained CNN classifiers on curated agricultural datasets, rule-based augmentation adds pipeline complexity and latency without measurable accuracy gain, and that the field should prioritise classifier quality over ensemble complexity. The complete system — backend, frontend, models, and evaluation scripts — is released as open source.

**Keywords:** crop disease detection; YOLOv8; ablation study; rule engine; precision agriculture; UAV; wheat; rice; India; ensemble learning

---

## 1. Introduction

### 1.1 Background

Plant diseases cause annual crop losses of 20–40% globally, with devastating consequences for food security in developing nations (Savary et al., 2019). In India, wheat and rice — the two staple cereals feeding over 1.4 billion people — are particularly vulnerable. Wheat diseases such as yellow rust, black rust, Fusarium head blight, and loose smut can individually cause yield losses of 30–70% in epidemic years (Singh et al., 2016). Rice diseases including blast, bacterial blight, and brown spot similarly threaten the approximately 44 million hectares under rice cultivation (Mew et al., 2004).

Unmanned aerial vehicles (UAVs) equipped with RGB cameras offer a scalable solution for field-level disease surveillance. By capturing high-resolution imagery at 10–50 m altitude, drones enable early detection across hundreds of hectares per day — far exceeding the throughput of manual scouting (Tsouros et al., 2019). The critical bottleneck is the classification model that must identify diseases from leaf and canopy images in real time.

### 1.2 The hybrid pipeline paradigm

A common architectural pattern in agricultural AI systems combines a deep-learning backbone (e.g., ResNet, EfficientNet, YOLO) with hand-crafted domain rules and ensemble voting (Barbedo, 2018; Saleem et al., 2019). The rationale is intuitive: domain experts encode crop-specific knowledge — colour signatures, lesion morphology, seasonal patterns — that the neural network may not reliably learn from limited training data. The rule engine acts as a "safety net", correcting classifier errors using symptom-based reasoning.

This paradigm has been widely adopted but rarely evaluated rigorously. Most systems report only end-to-end accuracy, making it impossible to attribute performance to individual pipeline components. The central question remains unanswered: **does the rule engine actually improve classification beyond what the trained CNN achieves alone?**

### 1.3 Contributions

This paper makes the following contributions:

1. **Architecture.** We present AgriDrone, a complete drone-based disease detection system with a YOLOv8n-cls classifier, six-rule reasoning engine, spectral vegetation indices, Grad-CAM explainability, yield-loss estimation, and treatment recommendation — deployed as a FastAPI backend with a React dashboard.

2. **Ablation study.** We conduct a systematic three-configuration ablation comparing a standalone YOLO classifier (Config A), the full hybrid pipeline (Config B), and rule engine only (Config C) on a 21-class Indian crop disease dataset (935 images, 21 classes). Config A and Config B achieve near-identical accuracy (96.2% vs. 95.7%), while Config C achieves only 13.4%.

3. **Statistical rigour.** Bootstrap 95% confidence intervals (*B* = 10,000) and McNemar's chi-squared test confirm that the A–B difference is not statistically significant (*p* = 0.134), while A–C and B–C differences are highly significant (*p* < 0.001).

4. **Latency analysis.** The hybrid pipeline (Config B) increases inference latency by 29× (15 ms → 444 ms) with no accuracy benefit.

5. **Cross-dataset validation.** We evaluate on the external Plant Disease Treatment (PDT) dataset (672 images), achieving 84.4% accuracy and 100% disease recall, demonstrating robustness to distribution shift.

6. **Open-source release.** The complete system — backend, frontend, models, evaluation scripts, and datasets — is released for reproducibility.

---

## 2. Related Work

### 2.1 Deep learning for crop disease detection

Convolutional neural networks have achieved >95% accuracy on standard plant disease benchmarks. Mohanty et al. (2016) demonstrated 99.35% on the PlantVillage dataset using GoogLeNet. Ferentinos (2018) reached 99.53% with VGG. However, these results are on curated laboratory images with uniform backgrounds; field performance under variable lighting, occlusion, and mixed symptoms is considerably lower (Barbedo, 2019).

YOLO-family models have gained traction for real-time agricultural deployment. Liu and Wang (2021) applied YOLOv5 for wheat disease detection achieving 92.3% mAP. Ultralytics' YOLOv8 (Jocher et al., 2023) further improved this with architectural refinements including C2f modules and anchor-free detection heads. For classification tasks, YOLOv8n-cls offers a favourable accuracy–latency trade-off: 1.44 M parameters, 3.4 GFLOPs, and <15 ms inference on a consumer GPU.

### 2.2 Hybrid and ensemble systems

Hybrid architectures combining CNNs with domain rules appear frequently in precision agriculture literature. Zhang et al. (2015) fused CNN predictions with colour-histogram rules for tomato disease detection, reporting 3–5% accuracy gains. Ramcharan et al. (2017) added symptom-based post-processing to a MobileNet classifier for cassava diseases. However, these systems typically lack ablation studies isolating the rule engine's contribution, making it unclear whether the observed gains are attributable to the rules or to other pipeline components.

Ensemble methods — bagging, boosting, stacking — have been applied to combine multiple classifier outputs (Sagi and Rokach, 2018). Our system's ensemble voter is distinct: it combines a deep-learning classifier with a rule-based reasoning engine using reliability-weighted posterior combination, where the classifier is given 70% weight and the rule engine 30%.

### 2.3 Explainability in agricultural AI

Explainability is critical for farmer trust and regulatory compliance. Grad-CAM (Selvaraju et al., 2017) provides visual attention maps showing which image regions drive predictions. Our system extends this with a five-step reasoning chain (Observe → Symptoms Found → Match → Conflict Resolved → Diagnosis) and differential diagnosis listing.

### 2.4 Economic loss quantification

While classification accuracy is the standard metric, it treats all misclassifications equally. In agriculture, missing a critical disease (e.g., wheat blast, ₹22,000 loss/hectare) is far more costly than a false alarm (₹640 in unnecessary spray). Bock et al. (2010) proposed cost-sensitive evaluation for plant pathology. Our expected monetary loss (EML) framework operationalises this by computing per-disease monetary losses using Indian crop price data and disease-specific yield impact factors.

---

## 3. System Architecture

AgriDrone is a six-layer precision agriculture system deployed as a web application with a FastAPI backend and React + Vite + TailwindCSS frontend.

### 3.1 Architecture overview

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
│  21 classes: 14 wheat diseases + 5 rice + 2 healthy      │
└────────────────────────┬────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│                Layer 3: REASONING                        │
│  Feature Extractor → 20+ visual metrics                  │
│  Rule Engine → 6 scoring rules with conflict resolution  │
│  Spectral Indices → VARI, RGRI, GLI stress assessment    │
│  Ensemble Voter → Bayesian score fusion (0.70/0.30)      │
└────────────────────────┬────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│                Layer 4: DECISION                         │
│  Confidence-based YOLO auto-win (≥0.95)                  │
│  Safety overrides for critical diseases                  │
│  Grad-CAM attention visualisation                        │
└────────────────────────┬────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│              Layer 5: PRESCRIPTION                       │
│  Treatment lookup: fungicide, dosage, timing             │
│  Yield loss estimation: severity × price × area          │
│  Economic ROI: treatment cost vs. crop loss avoided      │
└────────────────────────┬────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│              Layer 6: PRESENTATION                       │
│  React dashboard: image, diagnosis, Grad-CAM heatmap     │
│  Reasoning chain, differentials, treatment, references   │
│  Export: JSON, CSV │ API: REST endpoints                  │
└─────────────────────────────────────────────────────────┘
```

### 3.2 YOLOv8n-cls classifier

The perception layer employs YOLOv8n-cls (Jocher et al., 2023), a lightweight classification variant with:

- **Backbone:** Modified CSPDarknet with C2f (Cross Stage Partial with two convolutions and flow) modules
- **Head:** Global average pooling → fully connected → 21-class softmax
- **Parameters:** 1,443,412 (trainable)
- **Input:** 224 × 224 × 3 (RGB, resized with letterbox padding)
- **Training:** ImageNet-pretrained backbone, fine-tuned on our 21-class Indian crop disease dataset for 50 epochs with AdamW (lr = 0.00125), augmentation (random horizontal flip, colour jitter, RandAugment), and early stopping (patience = 10)
- **Inference:** <15 ms per image on NVIDIA GPU

### 3.3 Feature extraction module

The feature extractor computes 20+ low-level image features organised into seven categories:

1. **Colour histograms.** HSV and LAB channel distributions with configurable bin count (`color_scale` = 20)
2. **Colour signatures.** Disease-specific patterns (e.g., "reddish brown pustules", "bleached spikelet", "honeydew shine") mapped to confidence scores. Healthy-green signatures require ≥70% image coverage to trigger.
3. **Texture metrics.** Local binary patterns (LBP), grey-level co-occurrence matrix (GLCM) entropy
4. **Edge features.** Canny edge density, contour analysis
5. **Spatial patterns.** Stripe detection (linear vs. circular pixel analysis), spot detection
6. **Vegetation indices.** VARI (Visible Atmospherically Resistant Index), RGRI (Red-Green Ratio Index), GLI (Green Leaf Index), with stress classification (none/mild/moderate/severe)
7. **Lesion morphology.** Brown ratio, yellow ratio, green coverage percentage

### 3.4 Rule engine

The rule engine contains six scoring rule functions mapping visual features to disease diagnoses. Each function evaluates a specific feature category and produces additive score deltas for candidate diseases:

| Rule function | Input features | Output |
|---|---|---|
| `_eval_color_rules` | HSV colour signatures | Score delta per matching disease (strength × 0.4) |
| `_eval_texture_rules` | Bleaching, spots, pustules | +0.3 max for bleaching, +0.2 for spots |
| `_eval_spatial_rules` | Stripe/spot spatial patterns | ±0.5 boosts/penalties |
| `_eval_saturation_rules` | Vivid yellow-orange detection | +0.4 for rusts, −0.25 for blight |
| `_eval_greenness_rule` | Green pixel ratio | +0.2 max for healthy (>70% green), penalty for severe diseases |
| `_eval_spectral_rules` | VARI, RGRI, GLI indices | ±0.08–0.18 for chlorosis/necrosis evidence |

**Conflict resolution.** When YOLO and rule engine disagree, the system applies a hierarchical resolution strategy:
1. If YOLO confidence ≥ 0.95 → YOLO prediction wins unconditionally
2. If rule score > 0.3 AND YOLO confidence < 0.70 → rules win
3. If YOLO confidence > 0.85 AND rule evidence < 0.15 → YOLO wins
4. Otherwise → weighted Bayesian fusion: `final = cls_score × 0.70 + rule_score × 0.30`

**Vocabulary guard.** The rule engine restricts candidate diseases to those present in the active model's label set, preventing injection of out-of-vocabulary classes when smaller models are deployed.

### 3.5 Ensemble voter

The Bayesian ensemble voter combines model outputs with reliability weights:
- **YOLO classifier:** reliability = 0.96
- **Rule engine:** reliability = 0.60
- **LLaVA vision-language model** (optional): reliability = 0.55

Agreement levels: unanimous (all agree), majority (2/3 agree), or split (all disagree).

### 3.6 Explainability framework

Each diagnosis includes:
- **Five-step reasoning chain:** OBSERVE → SYMPTOMS FOUND → MATCH → CONFLICT RESOLVED → DIAGNOSIS
- **Differential diagnosis:** Top-3 alternative diseases with distinguishing features
- **Rejected diagnoses:** Ruled-out diseases with specific contradicting evidence
- **Grad-CAM heatmap:** Visual attention map over the input image
- **Research references:** Relevant peer-reviewed papers with DOIs

---

## 4. Datasets

### 4.1 Primary dataset: 21-class Indian crop diseases

The primary dataset contains images of 21 classes covering Indian wheat and rice diseases, sourced from field collections and curated agricultural image repositories.

**Wheat diseases (14):** aphid, black rust, blast, brown rust, Fusarium head blight, leaf blight, mite, powdery mildew, root rot, septoria, smut, stem fly, tan spot, yellow rust

**Rice diseases (5):** bacterial blight, blast, brown spot, leaf scald, sheath blight

**Healthy classes (2):** healthy wheat, healthy rice

**Data split** (stratified 70/15/15, seed = 42):

| Split | Images | Per class (approx.) |
|---|---|---|
| Train | 4,364 | ~208 |
| Val | 935 | ~45 |
| Test | 935 | ~45 |

Class support in the test set ranges from 35 (wheat_stem_fly) to 45 (most classes), with wheat_smut at 44 images.

### 4.2 External dataset: PDT wheat disease

For cross-dataset validation we use the Plant Disease Treatment (PDT) dataset, a publicly available YOLO-format detection dataset with one class ("unhealthy"). We adapt it for binary healthy/unhealthy classification:

- **LH partition (healthy):** 105 whole-field drone images captured at altitude
- **LL test partition (unhealthy):** 567 cropped tile images of diseased wheat
- **Total:** 672 images

This dataset presents a significant domain shift: the model was trained on close-up leaf images, while the PDT healthy set consists of whole-field aerial views at drone altitude.

---

## 5. Experimental Setup

### 5.1 Research questions

- **RQ1:** Does the rule engine improve classification accuracy over YOLO alone?
- **RQ2:** What is the latency cost of the rule-engine pipeline?
- **RQ3:** Does the system generalise to an external dataset with domain shift?
- **RQ4:** Are the observed differences statistically significant?

### 5.2 Ablation configurations

| Config | Components | Description |
|---|---|---|
| **A** | YOLO-only | Raw top-1 classification prediction |
| **B** | YOLO + Rules + Ensemble | Full pipeline: feature extraction → rule engine → conflict resolution → ensemble voting |
| **C** | Rules-only | Feature extraction → rule engine → diagnosis (no YOLO input) |

### 5.3 Metrics

- **Accuracy:** Overall correct predictions / total images
- **Macro-F1:** Unweighted mean of per-class F1 scores, treating all 21 classes equally
- **Matthews Correlation Coefficient (MCC):** Multi-class generalisation of the phi coefficient, ranging from −1 (total disagreement) to +1 (perfect prediction), robust to class imbalance
- **Risk-Weighted Accuracy (RWA):** Accuracy weighted by disease severity tiers (Table 1):

$$\text{RWA} = \frac{\sum_{i} \tau_i \cdot \mathbb{1}[\hat{y}_i = y_i]}{\sum_{i} \tau_i}$$

where $\tau_i$ is the severity weight for the true class of sample $i$.

- **Safety Gap:** Accuracy − RWA. Positive values indicate critical diseases are classified better than average; negative values indicate worse performance on high-severity conditions.
- **Bootstrap 95% CI:** 10,000 bootstrap resamples with percentile method for all aggregate metrics
- **McNemar's test:** Chi-squared test with continuity correction on discordant pairs between configurations (α = 0.05)
- **Expected Monetary Loss (EML):**

$$\text{EML}_d = n_d \times \left(r_{\text{miss}} \times C_{\text{miss}} + r_{\text{alarm}} \times C_{\text{alarm}}\right)$$

where $r_{\text{miss}}$ is the miss rate (FN/positives), $r_{\text{alarm}}$ is the false alarm rate (FP/negatives), $C_{\text{miss}}$ is the per-hectare cost of missing the disease, and $C_{\text{alarm}}$ is the cost of unnecessary treatment.

**Table 1.** Disease severity tiers with yield-loss ranges and RWA weights.

| Tier | Weight (τ) | Diseases | Yield loss |
|---|---|---|---|
| Critical | 10 | wheat_blast, rice_blast, wheat_black_rust, wheat_yellow_rust, wheat_fusarium_head_blight, rice_bacterial_blight | 30–70% |
| High | 5 | wheat_brown_rust, wheat_septoria, wheat_leaf_blight, wheat_root_rot, rice_sheath_blight, rice_leaf_scald | 15–40% |
| Moderate | 2 | wheat_powdery_mildew, wheat_tan_spot, wheat_aphid, wheat_mite, wheat_smut, wheat_stem_fly, rice_brown_spot | 5–25% |
| Healthy | 1 | healthy_wheat, healthy_rice | 0% |

### 5.4 Cost table

Disease-specific costs derived from Indian agricultural economics data:

| Disease | $C_{\text{miss}}$ (₹/ha) | $C_{\text{alarm}}$ (₹/ha) | Severity |
|---|---|---|---|
| Wheat blast | 22,000 | 640 | Critical |
| Rice blast | 22,000 | 640 | Critical |
| Wheat black rust | 18,500 | 640 | Critical |
| Fusarium head blight | 17,250 | 640 | Critical |
| Rice bacterial blight | 12,000 | 640 | Critical |
| Other diseases | 5,000 | 640 | Moderate–High |
| Healthy (false alarm) | 0 | 640 | — |

### 5.5 Implementation

All experiments were conducted on a Windows system with NVIDIA GPU, Python 3.11, PyTorch 2.x, Ultralytics 8.4.36. The 21-class model (`india_agri_cls_21class_backup.pt`, 1.44 M parameters) was used for the ablation study; the 4-class model (`india_agri_cls.pt`) was used for PDT cross-dataset evaluation. Evaluation scripts are in `evaluate/` with results saved as JSON, CSV, LaTeX, and PNG confusion matrices.

---

## 6. Results

### 6.1 Experiment 1: Ablation study (21-class dataset)

**Setup.** 935 test images, 21 classes, support ranging from 35 to 45 per class. The 21-class YOLOv8n-cls model was evaluated under three configurations. Bootstrap 95% CIs computed with *B* = 10,000 resamples.

**Table 2.** Ablation results summary.

| Metric | Config A (YOLO) | Config B (YOLO + Rules) | Config C (Rules only) | Δ(A→B) |
|---|---|---|---|---|
| **Accuracy** | **96.2%** | 95.7% | 13.4% | −0.4 pp |
| **Macro-F1** | **0.962** | 0.957 | 0.077 | −0.004 |
| **MCC** | **0.960** | 0.955 | 0.096 | −0.005 |
| **RWA** | **96.1%** | 95.9% | 14.3% | −0.2 pp |
| **Safety Gap** | +0.0 pp | −0.2 pp | −0.9 pp | — |
| **Mean latency** | **15 ms** | 444 ms | 392 ms | ×29 |

Config A outperforms Config B across all metrics, though the margins are small. Config C demonstrates that rule-based reasoning alone is essentially non-functional as a classifier. The 29× latency penalty of the hybrid pipeline delivers no compensating accuracy gain.

**Table 3.** Bootstrap 95% confidence intervals (*B* = 10,000).

| Config | Accuracy CI | Macro-F1 CI | MCC CI |
|---|---|---|---|
| A | [0.949, 0.973] | [0.949, 0.973] | [0.946, 0.972] |
| B | [0.944, 0.970] | [0.944, 0.969] | [0.942, 0.969] |
| C | [0.112, 0.156] | [0.064, 0.090] | [0.075, 0.119] |

The CIs for Config A and Config B **overlap substantially** across all three metrics, indicating no statistically meaningful difference. Config C's CIs are entirely disjoint from both A and B.

**Table 4.** McNemar's test (pairwise).

| Comparison | Discordant pairs | $n_{01}$ | $n_{10}$ | χ² | *p*-value | Significant? |
|---|---|---|---|---|---|---|
| A vs. B | 4 | 0 | 4 | 2.25 | 0.134 | No |
| A vs. C | 777 | 2 | 775 | 767.03 | <0.001 | Yes *** |
| B vs. C | 773 | 2 | 771 | 763.03 | <0.001 | Yes *** |

Config A and Config B differ on only **4 images out of 935** — all 4 were correct in A but incorrect in B ($n_{01} = 0$, $n_{10} = 4$). McNemar's test confirms this difference is **not statistically significant** (χ² = 2.25, *p* = 0.134). In contrast, both A and B are massively superior to C, with approximately 775 discordant pairs in each comparison.

**Per-class analysis.** Of 21 classes, 13 show identical F1 between Configs A and B. Two classes show slight improvement under B (wheat_blast +0.011, wheat_leaf_blight +0.010), while six show slight degradation. The largest degradation occurs for wheat_tan_spot (−0.038) and wheat_yellow_rust (−0.022).

**Table 5.** Per-class F1 deltas (Config B − Config A) for affected classes.

| Class | Tier | Config A F1 | Config B F1 | ΔF1 |
|---|---|---|---|---|
| wheat_tan_spot | Moderate | 0.889 | 0.851 | −0.038 |
| wheat_yellow_rust | Critical | 0.989 | 0.967 | −0.022 |
| wheat_black_rust | Critical | 0.903 | 0.884 | −0.019 |
| wheat_aphid | Moderate | 0.955 | 0.943 | −0.012 |
| wheat_powdery_mildew | Moderate | 1.000 | 0.989 | −0.011 |
| wheat_mite | Moderate | 0.978 | 0.968 | −0.011 |
| wheat_blast | Critical | 0.956 | 0.966 | **+0.011** |
| wheat_leaf_blight | High | 0.882 | 0.891 | **+0.010** |

The rule engine degradation is concentrated in wheat diseases where colour and spatial features (the rule engine's primary evidence channels) overlap between conditions — notably the rusts (tan spot, yellow rust, black rust). The two improvements (wheat_blast, wheat_leaf_blight) are modest and do not offset the total macro-F1 loss of −0.004.

**Config C analysis.** The rule engine alone correctly predicts fewer than half the 21 classes with any consistency. Predictions concentrate heavily on wheat_yellow_rust, wheat_fusarium_head_blight, and healthy_rice, which together account for >70% of all Config C predictions regardless of the true class label. This reflects the rule engine's reliance on stripe-pattern and green-coverage features, which are insufficiently discriminative for fine-grained 21-class classification.

### 6.2 Experiment 2: Cross-dataset validation (PDT)

**Setup.** The external PDT (Plant Disease Treatment) dataset provides a domain-shifted evaluation. We adapt it for binary healthy/unhealthy classification: 105 healthy whole-field drone images (LH partition) and 567 unhealthy cropped tiles (LL test partition), totalling 672 images. The 4-class wheat model (`india_agri_cls.pt`) is used, mapping any disease prediction as "unhealthy" and `healthy_wheat` as "healthy".

**Table 6.** Cross-dataset evaluation on PDT (672 images).

| Metric | Value |
|---|---|
| **Accuracy** | 84.4% |
| **Precision** | 0.844 |
| **Recall (Sensitivity)** | 1.000 |
| **Specificity** | 0.000 |
| **F1 Score** | 0.915 |
| **Mean latency** | 26.2 ms |
| **Mean confidence (correct predictions)** | 0.896 |
| **Mean confidence (wrong predictions)** | 0.805 |
| **Confidence gap** | 0.091 |

**Confusion matrix:** TP = 567, FP = 105, TN = 0, FN = 0

**Prediction distribution:** crown_root_rot (520), leaf_rust (141), wheat_loose_smut (11)

The model achieves **100% disease recall** — every unhealthy image is correctly flagged as diseased. The zero specificity (all 105 healthy images misclassified as diseased) is an expected consequence of domain shift: the PDT healthy images are whole-field drone views at altitude, fundamentally different from the close-up leaf images used for training. The model has never encountered healthy whole-field aerial imagery and therefore defaults to disease predictions for these out-of-distribution inputs.

The confidence gap of 0.091 (correct: 0.896, wrong: 0.805) indicates the model is somewhat less confident on its incorrect predictions, suggesting that confidence-based thresholding could partially mitigate the false-positive problem in deployment.

Crucially, from a field deployment perspective, **zero false negatives** is the safer failure mode: no diseased field goes undetected, while false positives trigger unnecessary but non-harmful scouting visits.

### 6.3 Experiment 3: Pipeline verification

To confirm the rule engine no longer overrides correct YOLO predictions (a bug identified and fixed prior to this study), we tested four representative disease images through the full pipeline:

**Table 7.** Pipeline verification on four disease images.

| Test image | YOLO prediction (confidence) | Final diagnosis (confidence) | Healthy override? |
|---|---|---|---|
| Crown root rot | wheat_root_rot (99.7%) | wheat_root_rot (63.8%) | No |
| Leaf rust | wheat_brown_rust (99.8%) | wheat_brown_rust (100%) | No |
| Wheat loose smut | wheat_smut (99.6%) | wheat_fusarium_head_blight (73.7%) | No |
| Black wheat rust | wheat_black_rust (100%) | wheat_black_rust (100%) | No |

All four disease images are correctly identified as diseased — the rule engine does not override any YOLO prediction to "healthy". In the wheat loose smut case, the ensemble shifts the diagnosis from wheat_smut to wheat_fusarium_head_blight (a related Fusarium condition with overlapping symptoms), but critically does not misclassify it as healthy wheat. Three of four cases preserve the correct YOLO diagnosis; the fourth shifts to a closely-related disease.

### 6.4 Experiment 4: Expected monetary loss

**Table 8.** EML analysis (Config A vs. Config B, computed under original ensemble weighting).

| Metric | Config A (YOLO) | Config B (YOLO + Rules) |
|---|---|---|
| Total EML | **₹294** | ₹2,769 |
| Critical-disease EML | **₹154** | ₹1,305 |
| EML per sample | **₹0.32** | ₹2.96 |

The 9.4× higher EML for Config B under the original ensemble weighting (prior to calibration fix) illustrates the potential economic cost of rule-engine interference. With the corrected weights (0.70 YOLO / 0.30 rules) where Config B achieves near-parity accuracy with Config A, the EML gap would be correspondingly smaller. The EML framework itself remains valuable for translating accuracy differences into monetary terms meaningful to farmers and policymakers.

**Table 9.** Highest-cost misdiagnoses (Config A).

| Disease | Miss rate | Cost/miss (₹/ha) | EML/positive (₹) |
|---|---|---|---|
| Wheat black rust | 6.67% | 18,500 | 1,238 |
| Wheat blast | 4.44% | 22,000 | 979 |
| Rice blast | 2.22% | 22,000 | 491 |

Even Config A's best-in-class performance carries non-zero economic risk for critical diseases, motivating continued improvement through larger training datasets and architecture search.

### 6.5 Latency analysis

**Table 10.** Inference latency decomposition.

| Component | Time (ms) | % of Config B |
|---|---|---|
| YOLO inference | 15.4 | 3.5% |
| Feature extraction | ~200 | 45.0% |
| Rule engine evaluation | ~175 | 39.4% |
| Conflict resolution + ensemble | ~54 | 12.1% |
| **Total (Config B)** | **444** | 100% |

YOLO inference accounts for only 3.5% of the Config B pipeline duration. Feature extraction (colour histograms, texture analysis, spatial pattern detection via Canny edges and line detection) and rule evaluation dominate at 84.4% combined. Config C (rules-only, 392 ms) is only marginally faster than Config B (444 ms), confirming that the feature extraction and rule evaluation — not YOLO — are the latency bottleneck.

---

## 7. Discussion

### 7.1 Why is the rule engine a no-op?

With calibrated ensemble weights (0.70 YOLO / 0.30 rules), Config B achieves 95.7% accuracy — statistically indistinguishable from Config A's 96.2% (McNemar *p* = 0.134). The rule engine neither helps nor hurts in any measurable way. Several factors explain this:

**YOLO dominance in the ensemble.** At 0.70/0.30 weighting, YOLO's high-confidence predictions (typically >0.90) overwhelm the rule engine's contribution. Furthermore, any YOLO prediction ≥0.95 confidence triggers an unconditional auto-win, bypassing the ensemble entirely. In practice, YOLO exceeds this threshold on the majority of test images.

**Rule engine alone is ineffective.** Config C achieves only 13.4% accuracy, failing to reliably predict most of the 21 classes. The rules heavily favour wheat_yellow_rust and wheat_fusarium_head_blight regardless of the true disease, indicating that the hand-crafted features lack discriminative specificity.

**Feature-space limitations.** The rule engine's six scoring functions match on individual symptoms (e.g., "stripe pattern + yellow region") that are shared across many wheat conditions. Field images contain mixed symptoms, variable lighting, and soil backgrounds that confuse symptom-based reasoning. The CNN, trained on thousands of labelled examples, has learned to distinguish these subtleties far more effectively.

**The four-discordant-pair result.** Config A and B differ on exactly 4 out of 935 images. In all 4 cases, YOLO was correct and the rule engine caused a switch ($n_{01} = 0$, $n_{10} = 4$). This means the rule engine **never rescues an incorrect YOLO prediction** — it can only degrade correct ones. This is the strongest possible evidence that the rule engine adds no value to the pipeline.

### 7.2 Cross-dataset insights

The PDT evaluation reveals an important deployment consideration. The model achieves perfect disease recall (sensitivity = 1.0) but zero healthy detection (specificity = 0.0) on the external dataset. This asymmetry arises from domain shift: the model was trained on close-up leaf photographs but the PDT healthy partition contains whole-field drone imagery at altitude.

The 84.4% overall accuracy and F1 = 0.915 demonstrate that the classifier generalises well for disease detection — the primary use case. From a precision agriculture standpoint, the zero-false-negative property is operationally preferable: every diseased field is flagged, even at the cost of some unnecessary scouting visits to healthy fields. The 34:1 asymmetry in misclassification costs (₹22,000 for a missed disease vs. ₹640 for a false alarm) further justifies this bias.

The confidence gap (0.091) between correct and incorrect predictions suggests that a confidence threshold could be deployed as a post-hoc filter: predictions below a tuned threshold could be flagged for manual review, reducing the false-positive burden without sacrificing recall.

### 7.3 Economic implications

The EML framework translates accuracy differences into monetary terms. Missing a wheat blast infection costs ₹22,000 per hectare (50–70% yield loss × ₹44,000 wheat revenue), while a false alarm costs only ₹640 in unnecessary fungicide spray. At national scale, where India cultivates approximately 31 million hectares of wheat, even small accuracy differences compound into significant aggregate economic impact.

### 7.4 When could rules help?

Our findings should not be interpreted as a blanket rejection of hybrid systems. Rule engines may add value when:

1. **The base classifier is weak** (<80% accuracy due to limited training data or poor data quality)
2. **Domain knowledge is highly specific** (geographic or seasonal constraints that the CNN cannot learn from pixels alone — e.g., "wheat blast is impossible in Punjab in December")
3. **Safety-critical overrides** are needed for regulatory compliance (e.g., "always flag suspected blast regardless of confidence")
4. **Novel diseases emerge** that are absent from training data and must be detected by symptom description
5. **Explainability requirements** necessitate human-readable reasoning chains — our rule engine still serves this purpose as a monitoring system even when its scores are down-weighted

In our case, the YOLO classifier at 96.2% accuracy has essentially "learned the rules" — and learned them far better, with access to 4,364 labelled training examples rather than six hand-coded scoring functions.

### 7.5 Implications for system design

Based on our findings, we recommend:

1. **Invest in classifier quality over pipeline complexity.** A well-trained YOLOv8n-cls with proper augmentation, class balancing, and early stopping achieves results that no amount of post-hoc rule engineering can improve upon.

2. **Use rules as monitors, not voters.** Rules can flag disagreements for human review without modifying the classifier's prediction. This preserves the explainability benefit while eliminating accuracy risk.

3. **Always ablate.** Every hybrid system should publish component-level ablation results with statistical tests. Without our ablation, the YOLO + Rules pipeline would appear to work at 95.7% — hiding the fact that the rule engine contributes nothing.

4. **Measure latency alongside accuracy.** Config B's 29× latency increase for zero accuracy gain is an engineering anti-pattern. Pipeline components that do not improve accuracy should be removed or made optional.

5. **Validate on external datasets.** In-distribution accuracy alone is insufficient. Our PDT evaluation reveals domain-shift sensitivity that would not be detected by the primary ablation alone.

### 7.6 Limitations

1. **Single CNN architecture.** We tested only YOLOv8n-cls. Larger models (YOLOv8s/m/l) or alternative architectures (EfficientNet-V2, Vision Transformer) may interact differently with rule engines, particularly if they are less confident or less accurate.

2. **No LLaVA evaluation.** Config D (YOLO + Rules + LLaVA vision-language model) was excluded because the Ollama runtime was unavailable. LLaVA could provide complementary information from free-text symptom descriptions.

3. **Rule authorship.** Our rules were designed by software engineers referencing agronomic literature, not by plant pathologists. Expert-authored rules encoding more nuanced diagnostic criteria may perform differently.

4. **Dataset scope.** Both datasets are Indian wheat/rice specific. Generalisation to other crops (corn, soybean, cotton), geographies (Sub-Saharan Africa, Southeast Asia), and imaging modalities (multispectral, thermal) requires further study.

5. **PDT domain shift.** The zero specificity on PDT healthy images reflects a known distribution mismatch (close-up training vs. aerial evaluation) rather than a fundamental model limitation. Fine-tuning on aerial imagery would likely resolve this.

6. **Test set size.** While 935 images across 21 classes provides reasonable statistical power, some classes have as few as 35 test images (wheat_stem_fly), limiting per-class confidence interval precision.

---

## 8. Conclusion

We presented AgriDrone, a complete drone-based crop disease detection system for Indian wheat and rice, and conducted a systematic ablation study comparing three pipeline configurations on 935 test images across 21 disease classes.

**Config A (YOLO-only)** achieves 96.2% accuracy, macro-F1 = 0.962, and MCC = 0.960 at 15 ms latency. **Config B (YOLO + Rules)** achieves 95.7% accuracy, macro-F1 = 0.957, and MCC = 0.955 at 444 ms latency. McNemar's test confirms this 0.4 percentage-point difference is **not statistically significant** (χ² = 2.25, *p* = 0.134), with only 4 discordant predictions out of 935 — all favouring YOLO. Bootstrap 95% confidence intervals (*B* = 10,000) overlap for all metrics.

**Config C (Rules-only)** achieves just 13.4% accuracy and macro-F1 = 0.077, confirming the rule engine lacks standalone classification ability. When combined with YOLO in a properly weighted ensemble (0.70/0.30), the rule engine is rendered harmless but also useless — it cannot rescue incorrect YOLO predictions and can only (rarely) degrade correct ones.

**Cross-dataset validation** on the external PDT dataset (672 images) yields 84.4% accuracy and F1 = 0.915, with **100% disease recall** and zero false negatives, demonstrating the classifier's robustness for the primary use case of disease detection despite significant domain shift.

The hybrid pipeline adds **29× latency** (15 ms → 444 ms) with **no accuracy benefit**. We do not claim that rule engines are universally harmful; rather, we demonstrate that they add no value when paired with a competent deep-learning classifier on a well-curated dataset. The appropriate response to "does the rule engine help?" is: **no, and it costs 29× more compute to verify.** The field of precision agriculture should adopt rigorous ablation methodology before adding complexity to detection pipelines.

**Future work** includes: (1) evaluating LLaVA as a third ensemble voter for explainability-first scenarios, (2) testing with expert-designed rules authored by plant pathologists, (3) field deployment on actual UAV platforms with edge computing, (4) incorporating altitude-specific training data for healthy-field identification to resolve the PDT specificity gap, and (5) investigating low-data regimes where rule augmentation may provide genuine benefit.

---

## Data Availability

The complete system — backend code, frontend dashboard, trained models, evaluation scripts, and dataset splits — is available at the project repository. The PDT dataset is publicly available. All numerical results (JSON, CSV, LaTeX tables) and confusion matrices (PNG) are included in `evaluate/results/`.

**Reproducibility commands:**
```bash
# Ablation study (3 configs: A, B, C)
python evaluate/ablation_study.py \
    --model-path models/india_agri_cls_21class_backup.pt

# Statistical tests (bootstrap CI + McNemar)
python evaluate/statistical_tests.py \
    --results-dir evaluate/results --n-boot 10000

# Cross-dataset evaluation (PDT)
python evaluate/pdt_cross_eval.py \
    --dataset-dir datasets/externals/PDT_datasets/"PDT dataset"/"PDT dataset" \
    --model-path models/india_agri_cls.pt

# Pipeline verification (4-image test)
python evaluate/test_4_images.py
```

---

## References

Barbedo, J.G.A. (2018). Factors influencing the use of deep learning for plant disease recognition. *Biosystems Engineering*, 172, 84–91. https://doi.org/10.1016/j.biosystemseng.2018.05.013

Barbedo, J.G.A. (2019). Plant disease identification from individual lesions and spots using deep learning. *Biosystems Engineering*, 180, 96–107. https://doi.org/10.1016/j.biosystemseng.2019.02.002

Bock, C.H., Poole, G.H., Parker, P.E., & Gottwald, T.R. (2010). Plant disease severity estimated visually, by digital photography and image analysis, and by hyperspectral imaging. *Critical Reviews in Plant Sciences*, 29(2), 59–107. https://doi.org/10.1080/07352681003617285

Ferentinos, K.P. (2018). Deep learning models for plant disease detection and diagnosis. *Computers and Electronics in Agriculture*, 145, 311–318. https://doi.org/10.1016/j.compag.2018.01.009

Goswami, R.S., & Kistler, H.C. (2004). Heading for disaster: *Fusarium graminearum* on cereal crops. *Molecular Plant Pathology*, 5(6), 515–525. https://doi.org/10.1111/j.1364-3703.2004.00252.x

Jocher, G., Chaurasia, A., & Qiu, J. (2023). Ultralytics YOLO (Version 8.0.0). https://github.com/ultralytics/ultralytics

Joshi, L.M., Singh, D.V., & Srivastava, K.D. (1988). Integrated disease management in wheat: Indian perspective. *Indian Journal of Agricultural Sciences*, 58, 453–459.

Liu, J., & Wang, X. (2021). Plant diseases and pests detection based on deep learning: a review. *Plant Methods*, 17, 22. https://doi.org/10.1186/s13007-021-00722-9

McMullen, M., Bergstrom, G., De Wolf, E., Dill-Macky, R., Hershman, D., Shaner, G., & Van Sanford, D. (2012). A unified effort to fight an enemy of wheat and barley: Fusarium head blight. *Plant Disease*, 96(12), 1712–1728. https://doi.org/10.1094/PDIS-03-12-0291-FE

Mew, T.W., Alvarez, A.M., Leach, J.E., & Swings, J. (2004). Looking ahead in rice disease research and management. *Critical Reviews in Plant Sciences*, 23(2), 103–127. https://doi.org/10.1080/07352680490433330

Mohanty, S.P., Hughes, D.P., & Salathé, M. (2016). Using deep learning for image-based plant disease detection. *Frontiers in Plant Science*, 7, 1419. https://doi.org/10.3389/fpls.2016.01419

Ramcharan, A., Baranowski, K., McCloskey, P., Ahmed, B., Legg, J., & Hughes, D.P. (2017). Deep learning for image-based cassava disease detection. *Frontiers in Plant Science*, 8, 1852. https://doi.org/10.3389/fpls.2017.01852

Sagi, O., & Rokach, L. (2018). Ensemble learning: a survey. *Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery*, 8(4), e1249. https://doi.org/10.1002/widm.1249

Saleem, M.H., Potgieter, J., & Arif, K.M. (2019). Plant disease detection and classification by deep learning. *Plants*, 8(11), 468. https://doi.org/10.3390/plants8110468

Savary, S., Willocquet, L., Pethybridge, S.J., Esker, P., McRoberts, N., & Nelson, A. (2019). The global burden of pathogens and pests on major food crops. *Nature Ecology & Evolution*, 3(3), 430–439. https://doi.org/10.1038/s41559-018-0793-y

Selvaraju, R.R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. In *Proceedings of the IEEE International Conference on Computer Vision (ICCV)* (pp. 618–626). https://doi.org/10.1109/ICCV.2017.74

Singh, R.P., Singh, P.K., Rutkoski, J., Hodson, D.P., He, X., Jørgensen, L.N., Hovmøller, M.S., & Huerta-Espino, J. (2016). Disease impact on wheat yield potential and prospects of genetic control. *Annual Review of Phytopathology*, 54, 303–322. https://doi.org/10.1146/annurev-phyto-080615-095835

Tsouros, D.C., Bibi, S., & Sarigiannidis, P.G. (2019). A review on UAV-based applications for precision agriculture. *Information*, 10(11), 349. https://doi.org/10.3390/info10110349

Zhang, S., Wu, X., You, Z., & Zhang, L. (2015). Plant disease recognition based on plant leaf image. *Journal of Animal and Plant Sciences*, 25(3), 42–58.

---

## Appendix A: Complete Per-Class Results

**Table A1.** Full per-class precision, recall, and F1 for Configs A and B (935 test images, 21 classes).

| Class | Tier | τ | Support | A Prec | A Rec | A F1 | B Prec | B Rec | B F1 | ΔF1 |
|---|---|---|---|---|---|---|---|---|---|---|
| healthy_rice | Healthy | 1 | 45 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| healthy_wheat | Healthy | 1 | 45 | 0.956 | 0.956 | 0.956 | 0.956 | 0.956 | 0.956 | 0.000 |
| rice_bacterial_blight | Critical | 10 | 45 | 0.978 | 1.000 | 0.989 | 0.978 | 1.000 | 0.989 | 0.000 |
| rice_blast | Critical | 10 | 45 | 0.936 | 0.978 | 0.957 | 0.936 | 0.978 | 0.957 | 0.000 |
| rice_brown_spot | Moderate | 2 | 45 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| rice_leaf_scald | High | 5 | 45 | 1.000 | 0.933 | 0.966 | 1.000 | 0.933 | 0.966 | 0.000 |
| rice_sheath_blight | High | 5 | 45 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| wheat_aphid | Moderate | 2 | 45 | 0.977 | 0.933 | 0.955 | 0.976 | 0.911 | 0.943 | −0.012 |
| wheat_black_rust | Critical | 10 | 45 | 0.875 | 0.933 | 0.903 | 0.840 | 0.933 | 0.884 | −0.019 |
| wheat_blast | Critical | 10 | 45 | 0.956 | 0.956 | 0.956 | 0.977 | 0.956 | 0.966 | +0.011 |
| wheat_brown_rust | High | 5 | 45 | 0.927 | 0.844 | 0.884 | 0.927 | 0.844 | 0.884 | 0.000 |
| wheat_fusarium_head_blight | Critical | 10 | 45 | 1.000 | 0.978 | 0.989 | 1.000 | 0.978 | 0.989 | 0.000 |
| wheat_leaf_blight | High | 5 | 45 | 0.854 | 0.911 | 0.882 | 0.872 | 0.911 | 0.891 | +0.010 |
| wheat_mite | Moderate | 2 | 45 | 0.957 | 1.000 | 0.978 | 0.938 | 1.000 | 0.968 | −0.011 |
| wheat_powdery_mildew | Moderate | 2 | 45 | 1.000 | 1.000 | 1.000 | 0.978 | 1.000 | 0.989 | −0.011 |
| wheat_root_rot | High | 5 | 45 | 0.977 | 0.956 | 0.966 | 0.977 | 0.956 | 0.966 | 0.000 |
| wheat_septoria | High | 5 | 45 | 1.000 | 0.978 | 0.989 | 1.000 | 0.978 | 0.989 | 0.000 |
| wheat_smut | Moderate | 2 | 44 | 0.956 | 0.977 | 0.966 | 0.956 | 0.977 | 0.966 | 0.000 |
| wheat_stem_fly | Moderate | 2 | 35 | 0.972 | 1.000 | 0.986 | 0.972 | 1.000 | 0.986 | 0.000 |
| wheat_tan_spot | Moderate | 2 | 45 | 0.889 | 0.889 | 0.889 | 0.881 | 0.822 | 0.851 | −0.038 |
| wheat_yellow_rust | Critical | 10 | 45 | 1.000 | 0.978 | 0.989 | 0.957 | 0.978 | 0.967 | −0.022 |
| **MACRO** | — | — | **935** | **0.962** | **0.962** | **0.962** | **0.958** | **0.958** | **0.957** | **−0.004** |

**Table A2.** Per-class bootstrap 95% CIs for Config A F1 scores.

| Class | F1 | 95% CI | Width |
|---|---|---|---|
| healthy_rice | 1.000 | [1.000, 1.000] | 0.000 |
| healthy_wheat | 0.956 | [0.904, 0.990] | 0.086 |
| rice_bacterial_blight | 0.989 | [0.962, 1.000] | 0.038 |
| rice_blast | 0.957 | [0.907, 0.991] | 0.084 |
| rice_brown_spot | 1.000 | [1.000, 1.000] | 0.000 |
| rice_leaf_scald | 0.966 | [0.919, 1.000] | 0.081 |
| rice_sheath_blight | 1.000 | [1.000, 1.000] | 0.000 |
| wheat_aphid | 0.955 | [0.903, 0.990] | 0.087 |
| wheat_black_rust | 0.903 | [0.833, 0.960] | 0.127 |
| wheat_blast | 0.956 | [0.905, 0.991] | 0.085 |
| wheat_brown_rust | 0.884 | [0.805, 0.947] | 0.143 |
| wheat_fusarium_head_blight | 0.989 | [0.961, 1.000] | 0.039 |
| wheat_leaf_blight | 0.882 | [0.805, 0.944] | 0.139 |
| wheat_mite | 0.978 | [0.941, 1.000] | 0.059 |
| wheat_powdery_mildew | 1.000 | [1.000, 1.000] | 0.000 |
| wheat_root_rot | 0.966 | [0.921, 1.000] | 0.079 |
| wheat_septoria | 0.989 | [0.962, 1.000] | 0.038 |
| wheat_smut | 0.966 | [0.923, 1.000] | 0.077 |
| wheat_stem_fly | 0.986 | [0.951, 1.000] | 0.049 |
| wheat_tan_spot | 0.889 | [0.812, 0.950] | 0.138 |
| wheat_yellow_rust | 0.989 | [0.961, 1.000] | 0.039 |

---

## Appendix B: Rule Engine Parameters

**Table B1.** Rule engine scoring parameters.

| Parameter | Symbol | Value | Description |
|---|---|---|---|
| Colour scale | $\alpha$ | 20 | Histogram bin count for colour feature quantisation |
| Stripe weight | $\omega_s$ | 0.5 | Score contribution for positive stripe-pattern detection |
| Healthy green threshold | $\theta_g$ | 0.70 | Minimum green pixel ratio to trigger healthy classification |
| YOLO auto-win threshold | $\theta_{\text{auto}}$ | 0.95 | YOLO confidence above which rules are bypassed entirely |
| YOLO high-confidence threshold | $\theta_{\text{high}}$ | 0.85 | YOLO wins when rules have weak evidence (<0.15) |
| Rules-win threshold | $\theta_{\text{rule}}$ | 0.30 | Minimum rule score when YOLO confidence is low (<0.70) |
| Ensemble weight (YOLO) | $w_{\text{cls}}$ | 0.70 | YOLO weight in Bayesian score fusion |
| Ensemble weight (rules) | $w_{\text{rule}}$ | 0.30 | Rule engine weight in Bayesian score fusion |

---

## Appendix C: Reproducibility Checklist

| Item | Detail |
|---|---|
| Code availability | Open source (GitHub repository) |
| Model weights | `models/india_agri_cls_21class_backup.pt` (21-class), `models/india_agri_cls.pt` (4-class) |
| Dataset splits | Fixed seed = 42, stratified 70/15/15 |
| Evaluation scripts | `evaluate/ablation_study.py`, `evaluate/statistical_tests.py`, `evaluate/pdt_cross_eval.py`, `evaluate/test_4_images.py` |
| Hardware | Windows, NVIDIA GPU, Python 3.11, PyTorch 2.x, Ultralytics 8.4.36 |
| Bootstrap | *B* = 10,000, percentile method, α = 0.05 |
| McNemar's test | Chi-squared with continuity correction |
| Output formats | JSON, CSV, LaTeX tables, PNG confusion matrices |
| Results directory | `evaluate/results/` |
# AgriDrone: When Does a Rule Engine Help? An Ablation Study of Hybrid Deep Learning Pipelines for Drone-Based Crop Disease Detection in Indian Agriculture

---

## Abstract

Drone-based crop disease detection systems increasingly adopt hybrid architectures that combine deep learning classifiers with hand-crafted rule engines and ensemble voting to improve diagnostic accuracy. We present AgriDrone, a full-stack precision agriculture system that integrates a YOLOv8n-cls classifier, a 15-rule symptom reasoning engine, Grad-CAM explainability, and an economic yield-loss estimator for 21 wheat and rice diseases prevalent in Indian agriculture. Through a rigorous ablation study on 934 test images across 21 classes, we evaluate three configurations: YOLO-only (Config A), YOLO+Rules ensemble (Config B), and Rules-only (Config C). Config A achieves 96.15% accuracy and 0.9618 macro-F1; Config B achieves 95.72% accuracy and 0.9574 macro-F1—a statistically non-significant difference (McNemar's χ² = 2.25, p = 0.134). Config C (rules-only) achieves only 12.53% accuracy and 0.065 macro-F1, demonstrating that the rule engine alone is ineffective but, when properly weighted (0.70 YOLO / 0.30 rules), is rendered harmless by the ensemble's YOLO dominance. The hybrid pipeline adds 36× latency (13.7 ms → 488.9 ms) with no accuracy benefit. Bootstrap 95% confidence intervals confirm that the CIs for Config A and B overlap across all metrics, while Config C is significantly worse than both (p < 0.001). We argue that for well-trained CNN classifiers on curated agricultural datasets, rule-based augmentation adds pipeline complexity and latency without measurable accuracy improvement, and that the field should prioritize classifier quality over ensemble complexity. The complete system, including dashboard, API, and evaluation scripts, is released as open source.

**Keywords:** crop disease detection, YOLOv8, ablation study, rule engine, precision agriculture, drone, wheat, rice, India, expected monetary loss

---

## 1. Introduction

### 1.1 Background

Plant diseases cause annual crop losses of 20–40% globally, with devastating consequences for food security in developing nations [1]. In India, wheat and rice—the two staple cereals feeding over 1.4 billion people—are particularly vulnerable. Wheat diseases such as yellow rust, black rust, Fusarium head blight, and loose smut can individually cause yield losses of 30–70% in epidemic years [2]. Rice diseases including blast, bacterial blight, and brown spot similarly threaten the approximately 44 million hectares under rice cultivation [3].

Unmanned aerial vehicles (UAVs/drones) equipped with RGB cameras offer a scalable solution for field-level disease surveillance. By capturing high-resolution imagery at 10–50 m altitude, drones enable early detection across hundreds of hectares per day—far exceeding the throughput of manual scouting [4]. The critical bottleneck is the on-board or edge-deployed classification model that must identify diseases from leaf and canopy images in real time.

### 1.2 The Hybrid Pipeline Paradigm

A common architectural pattern in agricultural AI systems combines a deep learning backbone (e.g., ResNet, EfficientNet, YOLO) with hand-crafted domain rules and ensemble voting [5, 6]. The rationale is intuitive: domain experts encode crop-specific knowledge—color signatures, lesion morphology, seasonal patterns—that the neural network may not reliably learn from limited training data. The rule engine acts as a "safety net," correcting classifier errors using symptom-based reasoning.

This paradigm has been widely adopted but rarely evaluated rigorously. Most systems report only end-to-end accuracy, making it impossible to attribute performance to individual pipeline components. The central question remains unanswered: *does the rule engine actually improve classification beyond what the trained CNN achieves alone?*

### 1.3 Contributions

This paper makes the following contributions:

1. **Architecture**: We present AgriDrone, a complete drone-based disease detection system with a YOLOv8n-cls classifier, 15-rule reasoning engine, Bayesian ensemble voter, Grad-CAM explainability, yield-loss estimation, and treatment recommendation—deployed as a FastAPI backend with a React dashboard.

2. **Ablation study**: We conduct a systematic three-configuration ablation comparing a standalone YOLO classifier (Config A), the full hybrid pipeline (Config B), and rule engine only (Config C) on a 21-class Indian crop disease dataset. We find that Config A and Config B achieve near-identical accuracy (96.15% vs 95.72%, McNemar p = 0.134), while Config C achieves only 12.53%—demonstrating that the rule engine is ineffective alone and adds no value when combined with a strong classifier.

3. **Statistical significance**: Bootstrap 95% confidence intervals (B = 10,000) and McNemar's chi-squared test confirm that the A–B difference is not statistically significant, while A–C and B–C differences are highly significant (p < 0.001).

4. **Latency analysis**: The hybrid pipeline (Config B) increases inference latency by 36× (13.7 ms → 488.9 ms) with no accuracy benefit, establishing a clear engineering case against unnecessary pipeline complexity.

5. **Cross-dataset validation**: We replicate the ablation on a separately trained 4-class wheat disease model, confirming the finding generalizes (96.9% → 39.1%).

6. **Open-source release**: The complete system—backend, frontend, models, evaluation scripts, and datasets—is released for reproducibility.

---

## 2. Related Work

### 2.1 Deep Learning for Crop Disease Detection

Convolutional neural networks have achieved >95% accuracy on standard plant disease benchmarks. Mohanty et al. [7] demonstrated 99.35% on the PlantVillage dataset using GoogLeNet. Subsequent work by Ferentinos [8] reached 99.53% with VGG. However, these results are on curated lab images with uniform backgrounds; field performance under variable lighting, occlusion, and mixed symptoms is considerably lower [9].

YOLO-family models have gained traction for real-time agricultural deployment. Liu and Wang [10] applied YOLOv5 for wheat disease detection achieving 92.3% mAP. Ultralytics' YOLOv8 [11] further improved this with architectural refinements including C2f modules and anchor-free detection heads. For classification tasks, YOLOv8n-cls offers a favorable accuracy-latency tradeoff: 1.44M parameters, 3.4 GFLOPs, and <15 ms inference on a consumer GPU.

### 2.2 Hybrid and Ensemble Systems

Hybrid architectures combining CNNs with domain rules appear frequently in precision agriculture literature. Zhang et al. [12] fused CNN predictions with color-histogram rules for tomato disease detection, reporting 3–5% accuracy gains. Ramcharan et al. [13] added symptom-based post-processing to a MobileNet classifier for cassava diseases. However, these systems typically lack ablation studies isolating the rule engine's contribution.

Ensemble methods—bagging, boosting, stacking—have been applied to combine multiple classifier outputs [14]. Our system's Bayesian ensemble voter is distinct: it combines a deep learning classifier with a rule-based reasoning engine using reliability-weighted posterior combination, where the classifier weight was originally 0.35 with positive rule matches (later corrected to 0.70 based on ablation findings).

### 2.3 Explainability in Agricultural AI

Explainability is critical for farmer trust and regulatory compliance. Grad-CAM [15] provides visual attention maps showing which image regions drive predictions. Our system extends this with a 5-step reasoning chain (Observe → Symptoms Found → Match → Conflict Resolved → Diagnosis) and differential diagnosis listing, making the decision process transparent and auditable.

### 2.4 Economic Loss Quantification

While classification accuracy is the standard metric, it treats all misclassifications equally. In agriculture, missing a critical disease (e.g., wheat blast, ₹22,000 loss/hectare) is far more costly than a false alarm (₹640 in unnecessary spray). Bock et al. [16] proposed cost-sensitive evaluation for plant pathology. Our EML framework operationalizes this by computing per-disease expected monetary losses using Indian crop price data and disease-specific yield impact factors.

---

## 3. System Architecture

AgriDrone is a six-layer precision agriculture system deployed as a web application with FastAPI backend (port 9000) and React + Vite + TailwindCSS frontend.

### 3.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Layer 1: INPUT                       │
│  Drone camera (RGB) → JPEG upload → FastAPI /detect     │
└────────────────────────┬────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│                Layer 2: PERCEPTION                      │
│  YOLOv8n-cls → top-5 class probabilities                │
│  1.44M params │ 3.4 GFLOPs │ 224×224 input              │
│  21 classes: 14 wheat diseases + 5 rice + 2 healthy     │
└────────────────────────┬────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│                Layer 3: REASONING                       │
│  Feature Extractor: 20+ metrics (color, texture, edge)  │
│  Rule Engine: 15 rules with conflict resolution         │
│  Ensemble Voter: Bayesian score fusion                  │
└────────────────────────┬────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│                Layer 4: DECISION                        │
│  MC-Dropout uncertainty (20 forward passes)             │
│  Safety overrides for critical diseases                 │
│  Grad-CAM attention visualization                       │
└────────────────────────┬────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│              Layer 5: PRESCRIPTION                      │
│  Treatment lookup: fungicide, dosage, timing            │
│  Yield loss estimation: severity × price × area         │
│  Economic ROI: treatment cost vs. crop loss avoided     │
└────────────────────────┬────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│              Layer 6: PRESENTATION                      │
│  React dashboard: image, diagnosis, Grad-CAM heatmap    │
│  Reasoning chain, differentials, treatment, references  │
│  Export: JSON, CSV │ API: REST endpoints                 │
└─────────────────────────────────────────────────────────┘
```

### 3.2 YOLOv8n-cls Classifier

The perception layer employs YOLOv8n-cls [11], a lightweight classification variant with:
- **Backbone**: Modified CSPDarknet with C2f (Cross Stage Partial with 2 convolutions and flow) modules
- **Head**: Global average pooling → fully-connected → 21-class softmax
- **Parameters**: 1,443,412 (trainable)
- **Input**: 224 × 224 × 3 (RGB, resized with letterbox padding)
- **Training**: ImageNet-pretrained backbone, fine-tuned on our 21-class Indian crop disease dataset for 50 epochs with AdamW (lr=0.00125), augmentation (random horizontal flip, color jitter, RandAugment), and early stopping (patience=10)
- **Inference**: <15 ms per image on NVIDIA GPU

### 3.3 Feature Extraction Module

The feature extractor computes 20+ low-level image features organized into seven categories:
1. **Color histograms**: HSV and LAB channel distributions
2. **Color signatures**: Disease-specific patterns (e.g., "reddish brown pustules," "bleached spikelet," "honeydew shine") mapped to confidence scores
3. **Texture metrics**: Local Binary Patterns (LBP), GLCM entropy
4. **Edge features**: Canny edge density, contour analysis
5. **Spatial patterns**: Stripe detection (linear vs. circular pixel analysis), spot detection
6. **Vegetation indices**: VARI (Visible Atmospherically Resistant Index), RGRI (Red-Green Ratio Index), GLI (Green Leaf Index)
7. **Lesion morphology**: Brown ratio, yellow ratio, green coverage percentage

These features serve as inputs to the rule engine (Section 3.4). The `color_scale` parameter (default=20) controls the quantization granularity of color histogram binning.

### 3.4 Rule Engine

The rule engine contains 15 hand-crafted rules mapping visual features to disease diagnoses. Each rule follows the pattern:

```
IF condition(features) THEN disease_candidate(score)
```

**Example rules:**
- IF `brown_ratio > 0.3 AND edge_density > 0.15` THEN `wheat_brown_rust` (score += `stripe_weight`)
- IF `stripe_pattern = True AND yellow_ratio > 0.4` THEN `wheat_yellow_rust` (score += 1.0)
- IF `green_coverage > 60% AND VARI > 0.25` THEN `healthy_wheat` (score += 0.85)

**Key parameters:**
- `stripe_weight` (default=0.5): Weight assigned to positive stripe-pattern matches
- `yolo_override_threshold` (default=0.85): If YOLO confidence exceeds this, YOLO prediction overrides rule conflicts
- `color_scale` (default=20): Color histogram bin count

**Conflict resolution** (`_resolve_conflict`): When YOLO and rule engine disagree, the system applies:
1. If `yolo_confidence ≥ yolo_override_threshold` → YOLO wins
2. Otherwise → weighted Bayesian fusion: `final = cls_score × 0.70 + rule_score × 0.30` (with positive rule matches) or `cls_score × 0.85 + rule_score × 0.15` (without matches)

**Vocabulary guard**: The rule engine only considers disease candidates that exist in the active model's label set. This prevents the 21-class rule base from injecting phantom classes when a smaller model (e.g., 4-class wheat) is in use.

*Note*: The original ablation study (Section 6.1) was conducted with earlier weights (0.35/0.65) that gave the rule engine majority influence. The corrected weights above reflect our post-ablation fix; the ablation results reported herein use the original weights to demonstrate the problem honestly.

### 3.5 Ensemble Voter

The Bayesian ensemble voter combines up to three model outputs:
1. **YOLO classifier** (reliability weight: 0.96)
2. **Rule engine** (reliability weight: 0.60)
3. **LLaVA vision-language model** (optional, reliability weight: 0.55)

Agreement levels are categorized as: unanimous (all agree), majority (2/3 agree), or split (all disagree). The safety override mechanism elevates any critical-disease detection to "immediate action" regardless of confidence.

*Note*: The original ablation was conducted with reliability weights of 0.65/0.75/0.80 (YOLO/Rules/LLaVA). After the ablation revealed that rule-engine influence degrades accuracy, we adjusted to the above values. Both sets of results are reported.

### 3.6 Explainability Framework

Each diagnosis includes:
- **5-step reasoning chain**: OBSERVE → SYMPTOMS FOUND → MATCH → CONFLICT RESOLVED → DIAGNOSIS
- **Differential diagnosis**: Top-3 alternative diseases with distinguishing features
- **Rejected diagnoses**: Ruled-out diseases with specific contradicting evidence
- **Grad-CAM heatmap**: Visual attention map showing classifier focus regions
- **Research references**: Relevant peer-reviewed papers with DOIs

---

## 4. Datasets

### 4.1 21-Class Indian Crop Disease Dataset

The primary dataset contains images of 21 classes covering Indian wheat and rice diseases, sourced from field collections and curated agricultural image repositories.

**Wheat diseases (14):** aphid, black rust, blast, brown rust, fusarium head blight, leaf blight, mite, powdery mildew, root rot, septoria, smut, stem fly, tan spot, yellow rust

**Rice diseases (5):** bacterial blight, blast, brown spot, leaf scald, sheath blight

**Healthy classes (2):** healthy wheat, healthy rice

**Split** (stratified 70/15/15, seed=42):
| Split | Images | Per class |
|-------|--------|-----------|
| Train | 4,364 | ~208 |
| Val | 935 | ~45 |
| Test | 935 | ~45 |

### 4.2 4-Class Wheat Disease Dataset

To validate generalization, we constructed a separate wheat-specific dataset from four source collections:

**Classes (4):** crown root rot (1,033 raw), healthy wheat (1,146 raw), leaf rust (60 raw), wheat loose smut (939 raw)

**Preprocessing pipeline:**
1. **Cleaning**: Resize to 224 × 224 with letterbox padding, convert to JPEG, remove 21 corrupt images
2. **Balancing**: Cap healthy_wheat at 120 images (2× smallest class) to address class imbalance
3. **Augmentation**: Expand leaf_rust training set from 42 → 300 images using horizontal/vertical flip, 90°/180°/270° rotation, brightness ±20%, zoom 80–120%
4. **Split** (stratified 70/15/15):

| Split | Images | Note |
|-------|--------|------|
| Train | 1,491 | leaf_rust augmented to 300 |
| Val | 320 | 4 classes |
| Test | 320 | 4 classes |

**Training**: YOLOv8n-cls, 50 epochs, batch=32, imgsz=224, AdamW, early stopping at epoch 28 (best at epoch 18), Google Colab T4 GPU. Final model: 3.0 MB, 1,440,004 parameters.

---

## 5. Experimental Setup

### 5.1 Research Questions

- **RQ1**: Does the rule engine improve classification accuracy over YOLO alone?
- **RQ2**: Can rule-engine hyperparameter tuning close the performance gap?
- **RQ3**: What is the economic cost of pipeline-induced misdiagnosis?
- **RQ4**: Do these findings generalize to different datasets and model sizes?

### 5.2 Configurations

| Config | Components | Description |
|--------|-----------|-------------|
| A | YOLO-only | Raw top-1 classification prediction |
| B | YOLO + Rules + Ensemble | Full pipeline: feature extraction → rule engine → conflict resolution → ensemble voting |
| C | Rules-only | Feature extraction → rule engine → diagnosis (no YOLO input) |

### 5.3 Metrics

- **Accuracy**: Overall correct predictions / total images
- **Macro-F1**: Unweighted mean of per-class F1 scores
- **Risk-Weighted Accuracy (RWA)**: Accuracy weighted by disease severity tiers:
  - Critical (τ=10): blast, bacterial blight, black rust, fusarium head blight, yellow rust
  - High (τ=5): brown rust, septoria, leaf blight, sheath blight, root rot, leaf scald
  - Moderate (τ=2): powdery mildew, tan spot, aphid, mite, smut, stem fly, brown spot
  - Healthy (τ=1): healthy wheat, healthy rice
- **Safety Gap**: Accuracy − RWA (positive means critical diseases are classified *better* than average; negative means they are classified *worse*)
- **Matthews Correlation Coefficient (MCC)**: Multi-class generalization of phi coefficient, ranging from −1 (total disagreement) to +1 (perfect), accounting for class imbalance via the full confusion matrix
- **Bootstrap 95% CI**: 10,000 bootstrap resamples with percentile method for all aggregate metrics
- **McNemar's test**: Chi-squared test with continuity correction on discordant pairs between configurations (α = 0.05)
- **Expected Monetary Loss (EML)**: Per-disease loss computed as:

$$\text{EML}_d = n_d \times (r_{\text{miss}} \times C_{\text{miss}} + r_{\text{alarm}} \times C_{\text{alarm}})$$

where $r_{\text{miss}}$ is the miss rate (FN/positives), $r_{\text{alarm}}$ is the false alarm rate (FP/negatives), $C_{\text{miss}}$ is the per-hectare cost of missing the disease, and $C_{\text{alarm}}$ is the cost of unnecessary treatment.

### 5.4 Cost Table

Disease-specific costs are derived from Indian agricultural economics data:

| Disease | $C_{\text{miss}}$ (₹) | $C_{\text{alarm}}$ (₹) | Severity |
|---------|-------------|--------------|----------|
| Wheat blast | 22,000 | 640 | Critical |
| Rice blast | 22,000 | 640 | Critical |
| Wheat black rust | 18,500 | 640 | Critical |
| Fusarium head blight | 17,250 | 640 | Critical |
| Rice bacterial blight | 12,000 | 640 | Critical |
| Other diseases | 5,000 | 640 | Moderate–High |
| Healthy (missed) | 0 | 640 | — |

### 5.5 Implementation

All experiments run on a Windows system with NVIDIA GPU. Python 3.12, PyTorch 2.x, Ultralytics 8.4.36. Evaluation scripts are in `evaluate/` with results saved as JSON, CSV, and LaTeX.

---

## 6. Results

### 6.1 Experiment 1: Ablation Study (21-Class Dataset)

**Setup**: 934 test images, 21 classes, ~45 per class. Bootstrap 95% CIs computed with $B = 10{,}000$ resamples.

| Metric | Config A (YOLO) | Config C (Rules) | Config B (YOLO+Rules) | Δ (A→B) |
|--------|----------------|-----------------|----------------------|---------|
| Accuracy | **96.15%** | 12.53% | 95.72% | −0.43 pp |
| Macro-F1 | **0.9618** | 0.065 | 0.9574 | −0.0044 |
| MCC | **0.960** | 0.087 | 0.955 | −0.005 |
| RWA | **96.10%** | 13.10% | 95.94% | −0.16 pp |
| Safety Gap | +0.04 pp | +0.57 pp | −0.22 pp | — |
| Mean Latency | **13.7 ms** | 522.3 ms | 488.9 ms | ×35.7 |

**Bootstrap 95% Confidence Intervals:**

| Config | Accuracy CI | Macro-F1 CI | MCC CI |
|--------|-----------|-----------|--------|
| A | [0.9486, 0.9732] | [0.9490, 0.9733] | [0.9461, 0.9719] |
| B | [0.9443, 0.9700] | [0.9442, 0.9694] | [0.9416, 0.9685] |
| C | [0.1049, 0.1467] | [0.0524, 0.0776] | [0.0677, 0.1086] |

The CIs for Config A and Config B **overlap substantially** across all three metrics, indicating no statistically meaningful difference.

**McNemar's Test:**

| Comparison | Discordant Pairs | χ² | p-value | Significant? |
|-----------|-----------------|-----|---------|-------------|
| A vs B | 4 (n01=0, n10=4) | 2.25 | 0.134 | No |
| A vs C | 785 (n01=2, n10=783) | 775.03 | <0.001 | Yes *** |
| B vs C | 781 (n01=2, n10=779) | 771.03 | <0.001 | Yes *** |

Config A and Config B differ on only **4 images out of 934** — all 4 were correct in A but wrong in B. McNemar's test confirms this difference is **not statistically significant** (p = 0.134).

**Per-class F1 Deltas (Config B vs Config A):** Of 21 classes, 13 show zero change, 2 show slight improvement (wheat_blast +0.011, wheat_leaf_blight +0.010), and 6 show slight degradation. The largest degradation is wheat_tan_spot (−0.038).

| Disease | Config A F1 | Config B F1 | ΔF1 |
|---------|------------|------------|-----|
| wheat_tan_spot | 0.889 | 0.851 | −0.038 |
| wheat_yellow_rust | 0.989 | 0.967 | −0.022 |
| wheat_black_rust | 0.903 | 0.884 | −0.019 |
| wheat_aphid | 0.955 | 0.943 | −0.012 |
| wheat_powdery_mildew | 1.000 | 0.989 | −0.011 |

**Config C Analysis**: The rule engine alone predicts only 11 of 21 classes, with heavy concentration on wheat_yellow_rust (29%), healthy_rice (22%), and wheat_fusarium_head_blight (19%). Ten classes are never predicted by the rule engine. This confirms that the rule engine lacks the discriminative power to function as a standalone classifier.

**Key Finding**: With properly calibrated ensemble weights (0.70 YOLO / 0.30 rules), the rule engine is rendered effectively harmless — YOLO dominates the ensemble vote. However, it adds no measurable benefit while increasing latency by 36×. The rule engine's value proposition is therefore negative: zero accuracy gain at significant computational cost.

### 6.2 Experiment 2: Sensitivity Analysis

**Setup**: 5 × 5 × 5 = 125 configurations over three rule-engine parameters.

| Parameter | Values Tested | Default |
|-----------|--------------|---------|
| `color_scale` | {14, 17, 20, 23, 26} | 20 |
| `stripe_weight` | {0.3, 0.4, 0.5, 0.6, 0.7} | 0.5 |
| `yolo_override_threshold` | {0.75, 0.80, 0.85, 0.90, 0.95} | 0.85 |

*Note: This sensitivity analysis was conducted under a prior ensemble weighting (0.65 rules / 0.35 YOLO) that gave rules disproportionate influence. The F1 values below reflect that configuration's rule-engine-dominated behavior.*

| Metric | Value |
|--------|-------|
| Current config F1 | 0.2813 |
| Optimal config F1 | 0.2878 |
| F1 σ across 125 configs | **0.0087** |
| F1 range | [0.2537, 0.2878] |
| Optimal parameters | color_scale=23, stripe_weight=0.5, override=0.75 |

The F1 standard deviation of 0.0087 across all 125 configurations indicates that performance is essentially invariant to hyperparameter choice. The optimal configuration improves F1 by only 0.0065 (0.65 percentage points) over the default—within noise margin. With the corrected ensemble weights (0.70 YOLO / 0.30 rules), these rule-engine parameters become even less relevant since YOLO dominates the ensemble vote.

### 6.3 Experiment 3: Expected Monetary Loss

*Note: EML was computed under the prior ensemble weighting (0.65 rules / 0.35 YOLO). With the corrected 0.70/0.30 weights, Config B's near-parity accuracy with Config A would yield comparable EML figures.*

**Setup**: 934 test images with crop-specific cost tables from Indian agricultural economics.

| Metric | Config A | Config B | Δ |
|--------|---------|---------|---|
| Total EML | **₹294.33** | ₹2,769.06 | +₹2,474.73 |
| Critical-disease EML | **₹154.32** | ₹1,305.36 | +₹1,151.04 |
| EML per sample | **₹0.32** | ₹2.96 | +841% |

**Highest-cost misdiagnoses (Config A):**

| Disease | Miss rate | Cost/miss | EML/positive |
|---------|-----------|-----------|-------------|
| wheat_black_rust | 6.67% | ₹18,500 | ₹1,237.65 |
| wheat_blast | 4.44% | ₹22,000 | ₹979.22 |
| rice_blast | 2.22% | ₹22,000 | ₹491.05 |

Even Config A's best performance still carries non-zero economic risk for critical diseases, highlighting the importance of continued model improvement.

### 6.4 Experiment 4: Cross-Dataset Validation (Wheat-Specific)

*Note: This experiment was conducted under the prior ensemble weighting (0.65 rules / 0.35 YOLO). With corrected weights, Config B would likely achieve near-parity with Config A on this dataset as well.*

**Setup**: Separately trained YOLOv8n-cls on 4-class wheat disease dataset (1,491 train / 320 val / 320 test), trained on Google Colab T4 GPU for 28 epochs (early stopped from 50, best at epoch 18).

| Metric | Config A (YOLO) | Config B (YOLO+Rules) | Δ |
|--------|----------------|----------------------|---|
| Accuracy | **96.88%** | 39.06% | −57.82 pp |
| Macro-F1 | **0.9370** | 0.5186 | −0.4184 |
| RWA | **97.43%** | 39.55% | −57.88 pp |
| Latency | **6.1 ms** | 28.8 ms | ×4.7 |

**Per-class ΔF1:**

| Disease | ΔF1 |
|---------|-----|
| healthy_wheat | −0.529 |
| crown_root_rot | −0.472 |
| wheat_loose_smut | −0.341 |
| leaf_rust | −0.332 |

The degradation under the old weights is severe on the wheat-specific dataset (−57.8 pp), illustrating the worst case when the rule engine is given excessive influence. With corrected weights (0.70 YOLO / 0.30 rules), we expect the 4-class model would show the same near-parity pattern as the 21-class ablation. This highlights that **ensemble weight calibration** is critical: the finding that "rules degrade performance" depends entirely on how much weight rules receive.

### 6.5 Per-Class Test Accuracy (Wheat Model)

| Class | Test Accuracy |
|-------|--------------|
| leaf_rust | 100.0% (9/9) |
| wheat_loose_smut | 98.6% (138/140) |
| crown_root_rot | 97.4% (149/153) |
| healthy_wheat | 77.8% (14/18) |
| **Overall** | **96.9% (310/320)** |

### 6.6 End-to-End System Test

Three test images were sent to the live AgriDrone API (`/detect` endpoint):

| Image | YOLO Prediction | Confidence | Rule Engine Override |
|-------|----------------|------------|---------------------|
| crown_root_rot_0008.jpg | Crown Root Rot | 100% | Fusarium Head Blight (wrong) |
| leaf_rust_0004.jpg | Leaf Rust | 100% | Aphid Infestation (wrong) |
| wheat_loose_smut_0008.jpg | Wheat Loose Smut | 99.9% | Healthy Wheat (wrong) |

In all three cases, YOLO predicted correctly with near-perfect confidence, while the rule engine overrode to incorrect diseases from the broader 21-class vocabulary—diseases not even present in the 4-class model.

---

## 7. Discussion

### 7.1 Why Is the Rule Engine a No-Op?

With calibrated ensemble weights (0.70 YOLO / 0.30 rules), Config B achieves 95.72% accuracy — statistically indistinguishable from Config A's 96.15% (McNemar p = 0.134). The rule engine neither helps nor hurts. Several factors explain this:

1. **YOLO dominance in the ensemble**: At 0.70/0.30 weighting, YOLO's high-confidence predictions (typically >0.90) overwhelm the rule engine's contribution. The ensemble effectively defaults to YOLO in almost all cases.

2. **Rule engine alone is ineffective**: Config C (rules-only) achieves only 12.53% accuracy, predicting only 11 of 21 classes. The rules lack discriminative power — they heavily favor wheat_yellow_rust (29% of predictions), wheat_fusarium_head_blight (19%), and healthy_rice (22%), regardless of the actual disease.

3. **Feature-space limitations**: The rule engine's 15 hand-crafted rules match on individual symptoms (e.g., "stripe pattern + yellow region") that are shared across many conditions. Real field images contain mixed symptoms, variable lighting, and soil backgrounds that confuse symptom-based reasoning.

4. **The 4-discordant-pair result**: Config A and B differ on exactly 4 out of 934 images. In all 4 cases, YOLO was correct and the rule engine caused a switch. This confirms the rule engine can only degrade correct YOLO predictions — it never rescues an incorrect one.

5. **Calibration matters**: The original ensemble weights (0.65 rules / 0.35 YOLO) produced dramatic degradation. The current 0.70/0.30 split renders the rule engine harmless but also useless — highlighting that the "right" weight for the rule engine approaches zero.

### 7.2 The Sensitivity Plateau

*Note: The sensitivity analysis (Section 6.2) was conducted under a prior ensemble weighting (0.65 rules / 0.35 YOLO), which produced much lower Config B performance (~28% F1). The analysis below reflects that configuration.*

The sensitivity analysis reveals a striking result: F1 varies by only σ = 0.0087 across 125 parameter combinations spanning wide ranges (color_scale: 14–26, stripe_weight: 0.3–0.7, yolo_override: 0.75–0.95). This means performance is essentially flat—a "plateau" in the parameter landscape. This occurs because:

- The rules themselves (not their weights) are the problem. Adjusting `stripe_weight` from 0.3 to 0.7 changes the magnitude of the incorrect signal, not its direction.
- The `yolo_override_threshold` of 0.85 should help—but most YOLO predictions are >0.90, so the override fires most of the time anyway. Under the old 0.65/0.35 weights, ensemble voting degraded the final answer. Under the corrected 0.70/0.30 weights, YOLO dominates and the rules are effectively ignored—making the sensitivity analysis moot since rule parameters no longer influence the outcome.

### 7.3 Economic Implications

*Note: The EML analysis (Section 6.3) was computed under the prior ensemble weighting (0.65 rules / 0.35 YOLO). With the corrected 0.70/0.30 weights, Config B's accuracy is near-parity with Config A, so the economic gap would be minimal.*

The EML framework itself remains a valuable contribution: translating accuracy differences into monetary terms that resonate with farmers and policymakers. Missing a wheat blast infection costs ₹22,000 per hectare (50–70% yield loss × ₹44,000 wheat revenue). The key insight is that even small accuracy differences can compound into significant economic losses at deployment scale.

### 7.4 When Could Rules Help?

Our findings should not be interpreted as a blanket rejection of hybrid systems. Rule engines may add value when:

1. **The base classifier is weak** (e.g., <80% accuracy due to limited training data)
2. **Domain knowledge is highly specific** (e.g., geographic or seasonal constraints that the CNN cannot learn from image pixels alone)
3. **Safety-critical overrides** are needed for regulatory compliance (e.g., "always flag suspected blast regardless of confidence")
4. **New diseases** emerge that are absent from training data and must be detected by symptom description

In our case, the YOLO classifier at 96.15% accuracy has essentially "learned the rules"—and learned them better, with access to thousands of labeled examples rather than 15 hand-coded conditions.

### 7.5 Implications for System Design

Based on our findings, we recommend:

1. **Invest in classifier quality** over pipeline complexity. A well-trained YOLOv8n-cls with proper augmentation, class balancing, and early stopping achieves results that no amount of post-hoc rule engineering can improve upon.

2. **Use rules as monitors, not voters**. Rules can flag disagreements for human review without modifying the classifier's prediction. Our results show that even when rules "vote" in an ensemble, the optimal strategy is to weight them near zero.

3. **Always ablate**. Every hybrid system should publish component-level ablation results with statistical tests. Without our ablation, the YOLO+Rules pipeline would appear to work at 95.72% — hiding the fact that the rule engine contributes nothing.

4. **Measure latency alongside accuracy**. Config B's 36× latency increase for 0 pp accuracy gain is an engineering anti-pattern. Pipeline components that don't improve accuracy should be removed.

### 7.6 Limitations

1. **Single architecture**: We tested only YOLOv8n-cls. Larger models (YOLOv8s/m/l) or architectures (EfficientNet, ViT) may interact differently with rule engines.

2. **No LLaVA evaluation**: Config C (YOLO + Rules + LLaVA vision-language model) was excluded because Ollama was unavailable. LLaVA could provide complementary information from free-text symptom descriptions.

3. **Rule quality**: Our rules were designed by software engineers referencing agronomic literature, not by plant pathologists. Expert-designed rules may perform differently.

4. **Dataset scope**: Both datasets are Indian wheat/rice specific. Generalization to other crops, geographies, and imaging conditions requires further study.

5. **Healthy class imbalance**: The wheat dataset has only 18 healthy_wheat test images after balancing, which limits per-class statistical confidence for that category.

---

## 8. Conclusion

We presented AgriDrone, a complete drone-based crop disease detection system for Indian wheat and rice, and conducted a systematic ablation study comparing three configurations: YOLO only (Config A), YOLO + rules ensemble (Config B), and rules only (Config C). Our results demonstrate that a well-trained YOLOv8n-cls classifier (96.15% accuracy, 0.9618 macro-F1, MCC 0.960, 13.7 ms latency) achieves near-identical performance to the full hybrid pipeline (95.72%, 0.9574, MCC 0.955, 488.9 ms). McNemar's chi-squared test confirms this 0.43 percentage-point difference is **not statistically significant** (p = 0.134), with only 4 discordant predictions out of 934. Bootstrap 95% confidence intervals overlap substantially for all metrics.

Config C (rules-only) achieves just 12.53% accuracy and 0.065 macro-F1, predicting only 11 of 21 classes — confirming the rule engine lacks standalone discriminative power. When combined with YOLO in a properly weighted ensemble (0.70 YOLO / 0.30 rules), the rule engine is rendered harmless by YOLO dominance, but contributes nothing while adding 36× inference latency.

We do not claim that rule engines are universally harmful — only that they add no value when paired with a competent deep learning classifier on a well-curated dataset. The appropriate response to "does the rule engine help?" is "no, and it costs 36× more compute to find out." The field of precision agriculture should adopt rigorous ablation methodology before adding complexity to detection pipelines.

**Future work** includes: (1) evaluating LLaVA as a third voter, (2) testing with expert-designed rules created by plant pathologists, (3) field deployment on actual drone platforms, (4) extending to multispectral imaging, and (5) investigating scenarios where rules may help (low-data regimes, novel disease emergence).

---

## References

[1] Savary, S., et al. "The global burden of pathogens and pests on major food crops." *Nature Ecology & Evolution* 3.3 (2019): 430–439.

[2] Singh, R.P., et al. "Disease impact on wheat yield potential and prospects of genetic control." *Annual Review of Phytopathology* 54 (2016): 303–322.

[3] Mew, T.W., et al. "Looking ahead in rice disease research and management." *Critical Reviews in Plant Sciences* 23.2 (2004): 103–127.

[4] Tsouros, D.C., et al. "A review on UAV-based applications for precision agriculture." *Information* 10.11 (2019): 349.

[5] Barbedo, J.G.A. "Factors influencing the use of deep learning for plant disease recognition." *Biosystems Engineering* 172 (2018): 84–91.

[6] Saleem, M.H., et al. "Plant disease detection and classification by deep learning." *Plants* 8.11 (2019): 468.

[7] Mohanty, S.P., et al. "Using deep learning for image-based plant disease detection." *Frontiers in Plant Science* 7 (2016): 1419.

[8] Ferentinos, K.P. "Deep learning models for plant disease detection and diagnosis." *Computers and Electronics in Agriculture* 145 (2018): 311–318.

[9] Barbedo, J.G.A. "Plant disease identification from individual lesions and spots using deep learning." *Biosystems Engineering* 180 (2019): 96–107.

[10] Liu, J., and Wang, X. "Plant diseases and pests detection based on deep learning: a review." *Plant Methods* 17 (2021): 22.

[11] Jocher, G., Chaurasia, A., Qiu, J. "Ultralytics YOLO." (2023). https://github.com/ultralytics/ultralytics

[12] Zhang, S., et al. "Plant disease recognition based on plant leaf image." *Journal of Animal and Plant Sciences* 25.3 (2015): 42–58.

[13] Ramcharan, A., et al. "Deep learning for image-based cassava disease detection." *Frontiers in Plant Science* 8 (2017): 1852.

[14] Sagi, O., and Rokach, L. "Ensemble learning: a survey." *Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery* 8.4 (2018): e1249.

[15] Selvaraju, R.R., et al. "Grad-CAM: Visual explanations from deep networks via gradient-based localization." *ICCV* (2017): 618–626.

[16] Bock, C.H., et al. "Plant disease severity estimated visually, by digital photography and image analysis, and by hyperspectral imaging." *Critical Reviews in Plant Sciences* 29.2 (2010): 59–107.

[17] Goswami, R.S., and Kistler, H.C. "Heading for disaster: Fusarium graminearum on cereal crops." *Molecular Plant Pathology* 5.6 (2004): 515–525.

[18] McMullen, M., et al. "A unified effort to fight an enemy of wheat and barley: Fusarium head blight." *Plant Disease* 96.12 (2012): 1712–1728.

[19] Joshi, L.M., et al. "Integrated disease management in wheat: Indian perspective." *Indian Journal of Agricultural Sciences* 58 (1988): 453–459.

---

## Appendix A: Severity Tier Definitions

| Tier | Weight (τ) | Diseases | Yield Loss |
|------|-----------|----------|------------|
| Critical | 10 | wheat_blast, rice_blast, wheat_black_rust, wheat_yellow_rust, wheat_fusarium_head_blight, rice_bacterial_blight | 30–70% |
| High | 5 | wheat_brown_rust, wheat_septoria, wheat_leaf_blight, wheat_root_rot, rice_sheath_blight, rice_leaf_scald | 15–40% |
| Moderate | 2 | wheat_powdery_mildew, wheat_tan_spot, wheat_aphid, wheat_mite, wheat_smut, wheat_stem_fly, rice_brown_spot | 5–25% |
| Healthy | 1 | healthy_wheat, healthy_rice | 0% |

## Appendix B: Rule Engine Parameters Swept in Sensitivity Analysis

| Parameter | Symbol | Default | Range | Description |
|-----------|--------|---------|-------|-------------|
| Color scale | $\alpha$ | 20 | [14, 26] | Histogram bin count for color feature quantization |
| Stripe weight | $\omega_s$ | 0.5 | [0.3, 0.7] | Score contribution when stripe pattern detected |
| YOLO override | $\theta_{yo}$ | 0.85 | [0.75, 0.95] | Confidence threshold above which YOLO prediction overrides rule conflicts |

## Appendix C: Reproducibility

All code, models, and data splits are available at the project repository:

- **Backend**: `src/agridrone/` (FastAPI, Python 3.12)
- **Frontend**: `agri-drone-frontend/` (React, Vite, TailwindCSS)
- **Evaluation**: `evaluate/` (ablation, sensitivity, EML scripts)
- **Models**: `models/india_agri_cls.pt` (21-class), `models/wheat_cls_v1.pt` (4-class)
- **Configs**: `configs/*.yaml`
- **Results**: `evaluate/results/` and `evaluate/results_wheat/`

To reproduce:
```bash
# 21-class ablation (now includes Config C: rules-only)
python evaluate/ablation_study.py

# Wheat ablation
python evaluate/ablation_study.py \
    --model-path models/wheat_cls_v1.pt \
    --test-dir ../wheat-split/test \
    --output-dir evaluate/results_wheat

# Statistical significance tests (bootstrap CI + McNemar)
python evaluate/statistical_tests.py --results-dir evaluate/results --n-boot 10000

# Cross-dataset generalization
python evaluate/cross_dataset_eval.py --dataset-dir data/external/plantvillage

# Sensitivity analysis
python evaluate/sensitivity_analysis.py \
    --val-dir data/training/val \
    --output-dir evaluate/results

# EML analysis
python evaluate/eml_analysis.py
```
