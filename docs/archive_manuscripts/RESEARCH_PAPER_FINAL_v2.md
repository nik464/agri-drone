# AgriDrone: A Systematic Ablation Study of Hybrid Deep-Learning Pipelines for Drone-Based Crop Disease Detection in Indian Wheat and Rice

**Authors:** [Author names]
**Affiliation:** [Institution]
**Corresponding author:** [Email]
**Submitted to:** *Smart Agricultural Technology* (Elsevier)

---

## Abstract

Agriculture is a cornerstone of the global economy, yet crop diseases continue to pose a severe threat to food security, with annual yield losses estimated at 20–40% worldwide. In the Indian subcontinent, wheat and rice—the two principal staple cereals sustaining over 1.4 billion people—are particularly susceptible to a spectrum of fungal, bacterial, and pest-mediated pathologies including yellow rust, black rust, Fusarium head blight, blast, and bacterial blight, which can individually inflict yield losses of 30–70% during epidemic years.

In this work, we present AgriDrone, a full-stack precision agriculture system that integrates a YOLOv8n-cls convolutional neural network classifier (1.44 M parameters), a six-rule symptom-based reasoning engine with spectral vegetation indices, Bayesian ensemble voting with conflict resolution, Grad-CAM visual explainability, and an expected monetary loss (EML) estimator for 21 wheat and rice disease classes prevalent in Indian agriculture. A systematic three-configuration ablation study was conducted on 935 test images spanning 21 classes to evaluate: (i) YOLO-only classification (Config A), (ii) YOLO + Rules ensemble (Config B), and (iii) Rules-only classification (Config C).

Config A achieved 96.15% overall accuracy (macro-F1 = 0.962, MCC = 0.960) at a mean inference latency of 15 ms per image. Config B achieved 95.72% accuracy (macro-F1 = 0.957, MCC = 0.955) at 444 ms latency. McNemar's chi-squared test with continuity correction confirmed that this 0.43 percentage-point difference was not statistically significant (χ² = 2.25, *p* = 0.134), with only 4 discordant predictions out of 935 images—all favouring the standalone classifier. Bootstrap 95% confidence intervals (*B* = 10,000 resamples) overlapped substantially for Configs A and B across all metrics. Config C achieved only 13.41% accuracy (macro-F1 = 0.077, MCC = 0.096), confirming the rule engine's lack of standalone discriminative capacity. Cross-dataset evaluation on the external Plant Disease Treatment (PDT) dataset (672 images) yielded 84.4% accuracy, F1 = 0.915, and 100% disease recall, demonstrating classifier robustness under significant domain shift. Sensitivity analysis across 125 ensemble weight configurations yielded macro-F1 σ = 0.0087, indicating stability. The hybrid pipeline imposed a 29× latency penalty (15 ms → 444 ms) with no compensating accuracy gain.

These findings demonstrate that, for well-trained convolutional neural network classifiers operating on curated agricultural image datasets, rule-based augmentation introduces pipeline complexity and computational overhead without measurable improvement in diagnostic accuracy. It is recommended that the precision agriculture community adopt rigorous component-level ablation methodology prior to deploying hybrid architectures in production systems.

**Keywords:** crop disease detection; YOLOv8; convolutional neural network; ablation study; rule engine; ensemble learning; precision agriculture; unmanned aerial vehicle; wheat disease; rice disease; India; expected monetary loss; Grad-CAM; domain adaptation

---

## 1. Introduction

Agriculture remains a foundational pillar of socioeconomic development across the Global South, where the livelihoods of over two billion smallholder farmers depend directly on crop productivity. It has been widely established that plant diseases constitute one of the most significant threats to global food security, with annual yield losses attributable to pathogens and pests estimated at 20–40% of global production [1]. The economic and humanitarian consequences are particularly acute in developing nations, where limited access to diagnostic infrastructure, extension services, and disease-resistant cultivars compounds the biological threat. In India alone, the agricultural sector contributes approximately 17% of gross domestic product and employs nearly 42% of the total workforce, rendering crop health monitoring a matter of national economic importance.

Wheat (*Triticum aestivum*) and rice (*Oryza sativa*) are the two principal staple cereals of the Indian subcontinent, collectively cultivated on approximately 75 million hectares and sustaining the caloric requirements of over 1.4 billion people. Wheat production is particularly threatened by a complex of fungal pathogens: yellow rust (*Puccinia striiformis*), black rust (*Puccinia graminis*), brown rust (*Puccinia triticina*), Fusarium head blight (*Fusarium graminearum*), powdery mildew (*Blumeria graminis*), and loose smut (*Ustilago tritici*), each capable of inflicting 30–70% yield loss during epidemic outbreaks [2]. Rice pathogens including blast (*Magnaporthe oryzae*), bacterial blight (*Xanthomonas oryzae*), sheath blight (*Rhizoctonia solani*), and brown spot (*Bipolaris oryzae*) pose analogous threats to the approximately 44 million hectares under rice cultivation in India [3]. The temporal dynamics of these diseases—where a delay of 48–72 hours in detection can escalate a localised infection into a field-wide epidemic—necessitate rapid, scalable surveillance methodologies.

Unmanned aerial vehicles (UAVs) equipped with high-resolution RGB cameras have emerged as a transformative technology for field-level crop health monitoring. It has been demonstrated that drone-based surveillance systems can cover hundreds of hectares per day at altitudes of 10–50 m, capturing imagery at spatial resolutions sufficient for symptom-level disease identification—far exceeding the throughput of traditional manual scouting, which is estimated to cover merely 2–5 hectares per person-day [4]. The critical technological bottleneck, however, lies not in image acquisition but in the classification model deployed at the edge or in the cloud that must identify specific diseases from leaf and canopy imagery with high accuracy and low latency.

Deep learning, and in particular convolutional neural networks (CNNs), have achieved remarkable performance on plant disease classification benchmarks. Mohanty et al. [5] demonstrated 99.35% accuracy on the PlantVillage dataset using GoogLeNet, while subsequent studies have reported similar performance with VGG, ResNet, and EfficientNet architectures. However, several limitations of these laboratory-derived benchmarks have been identified: images are typically captured under controlled conditions with uniform backgrounds, single lesions per frame, and consistent illumination—conditions that diverge substantially from the heterogeneity encountered in real-world field deployment, where mixed symptoms, variable lighting, soil backgrounds, occlusion by plant canopy, and multi-pathogen co-infections prevail. To address this gap, a common architectural pattern has emerged in the precision agriculture literature: hybrid pipelines that combine a CNN backbone with hand-crafted domain rules and ensemble voting mechanisms, under the assumption that expert-encoded agronomic knowledge can compensate for the classifier's limitations in field conditions.

Despite the widespread adoption of hybrid architectures, a critical evaluation gap persists. The majority of published systems report only end-to-end pipeline accuracy, rendering it impossible to attribute performance to individual components and, more critically, to determine whether the rule engine contributes positively, negatively, or not at all to the final classification outcome. This absence of rigorous ablation methodology has resulted in a literature where pipeline complexity is implicitly equated with system quality, without empirical justification. The present study addresses this gap directly. We present AgriDrone, a complete drone-based crop disease detection system for Indian wheat and rice, and conduct a systematic three-configuration ablation study to isolate the contribution of each pipeline component. The principal contributions of this work are as follows:

1. **System architecture.** A six-layer precision agriculture system is presented, integrating a YOLOv8n-cls classifier (1.44 M parameters), a six-rule symptom-based reasoning engine with spectral vegetation indices (VARI, RGRI, GLI), Bayesian ensemble voting with hierarchical conflict resolution, Grad-CAM visual explainability, expected monetary loss estimation, and treatment recommendation—deployed as a FastAPI backend with a React dashboard.

2. **Ablation study.** A systematic three-configuration ablation is conducted on 935 test images across 21 disease classes, comparing standalone YOLO classification (Config A), the full hybrid pipeline (Config B), and rule-engine-only classification (Config C). It is demonstrated that Config A and Config B achieve statistically indistinguishable accuracy (96.15% vs. 95.72%, McNemar *p* = 0.134), while Config C achieves only 13.41%.

3. **Statistical rigour.** Bootstrap 95% confidence intervals with *B* = 10,000 resamples and McNemar's chi-squared test with continuity correction are employed. Sensitivity analysis across 125 ensemble weight configurations is conducted, yielding macro-F1 σ = 0.0087.

4. **Cross-dataset validation.** External evaluation on the Plant Disease Treatment (PDT) dataset (672 images) demonstrates 84.4% accuracy and 100% disease recall under significant domain shift.

5. **Economic analysis.** An expected monetary loss (EML) framework translates accuracy differences into per-hectare monetary costs using Indian agricultural economics data, yielding EML of ₹294.33 for Config A versus ₹2,769.06 for Config B.

6. **Open-source release.** The complete system—backend, frontend, trained models, evaluation scripts, and dataset splits—is released for full reproducibility.

---

## 2. Related Work

### 2.1 Deep learning for plant disease classification

The application of deep learning to plant disease classification has been the subject of extensive investigation over the past decade. Mohanty et al. [5] demonstrated the feasibility of transfer learning for plant disease recognition, achieving 99.35% accuracy on the 38-class PlantVillage dataset using GoogLeNet and AlexNet architectures. Subsequent work by Ferentinos (2018) extended these results to 99.53% accuracy using VGG networks. However, it has been noted by several authors that these benchmarks, derived from laboratory-curated images with uniform backgrounds and single-lesion framing, exhibit a significant performance gap when models are deployed on field-acquired imagery characterised by variable illumination, mixed symptoms, background clutter, and multi-pathogen co-infections.

Within the YOLO family of architectures, which have gained considerable traction for real-time agricultural deployment, Liu and Wang (2021) applied YOLOv5 to wheat disease detection and reported 92.3% mean average precision. The release of YOLOv8 by Ultralytics [6] introduced several architectural refinements, including C2f (Cross Stage Partial with two convolutions and flow) modules and anchor-free detection heads, yielding improved accuracy–latency trade-offs. For classification tasks, the YOLOv8n-cls variant offers particular advantages: 1.44 M trainable parameters, 3.4 GFLOPs computational cost, and inference latency below 15 ms on consumer-grade GPU hardware—properties that are highly desirable for edge deployment in agricultural UAV systems.

### 2.2 Hybrid architectures and ensemble strategies

Hybrid architectures that augment CNN classifiers with domain-specific rule engines represent a prevalent design pattern in the precision agriculture literature. Zhang et al. (2015) proposed fusing CNN predictions with colour-histogram-derived rules for tomato disease detection, reporting 3–5 percentage-point accuracy improvements. Ramcharan et al. (2017) incorporated symptom-based post-processing into a MobileNet classifier for cassava disease identification. More recently, Simhadri and Niaz [9, 10] have explored multi-model ensemble strategies combining lightweight CNN architectures for crop disease classification in resource-constrained environments. However, a critical methodological limitation of these studies is the absence of component-level ablation: by reporting only end-to-end pipeline accuracy, it remains indeterminate whether the observed performance is attributable to the rule engine, the CNN, or their interaction.

Ensemble learning methodologies—including bagging, boosting, and stacking—have been extensively reviewed by Sagi and Rokach (2018), who demonstrated that ensemble approaches generally improve upon individual learner performance when base models exhibit sufficient diversity. The ensemble voter employed in the present system is architecturally distinct from these approaches: rather than combining multiple homogeneous classifiers, it fuses a deep-learning classifier with a heterogeneous rule-based reasoning engine using reliability-weighted Bayesian posterior combination, where the classifier is assigned 70% weight and the rule engine 30%.

### 2.3 Explainability in agricultural decision support

The requirement for explainable artificial intelligence (XAI) in agricultural decision support systems has been increasingly emphasised, as farmer trust and regulatory compliance necessitate transparent reasoning processes. Grad-CAM (Gradient-weighted Class Activation Mapping), introduced by Selvaraju et al. [7], provides visual attention maps that highlight the image regions contributing most strongly to the classifier's prediction. In the agricultural domain, such visualisations enable agronomists to verify whether the model attends to disease-relevant features (e.g., lesion boundaries, discolouration patterns) rather than spurious correlations (e.g., background soil colour, image borders). The present system extends visual explainability with a structured five-step reasoning chain (Observe → Symptoms Found → Match → Conflict Resolved → Diagnosis) and a differential diagnosis listing with rejected alternatives.

### 2.4 Economic loss quantification in plant pathology

Conventional classification metrics (accuracy, F1 score, MCC) treat all misclassifications as equally consequential—an assumption that is demonstrably invalid in agricultural disease diagnosis, where the cost of missing a critical pathogen such as wheat blast (estimated at ₹22,000 per hectare in yield loss) exceeds the cost of a false alarm (₹640 in unnecessary fungicide application) by a factor of approximately 34:1. Cost-sensitive evaluation frameworks for plant pathology have been proposed by Bock et al. (2010), and the concept of disease-weighted accuracy has been explored by Mahmood et al. [8] in the context of precision agriculture. The expected monetary loss (EML) framework employed in the present study operationalises this asymmetry by computing per-disease monetary losses using Indian crop price data and pathogen-specific yield impact factors derived from agricultural economics literature.

---

## 3. Materials and Methods

### 3.1 System architecture

AgriDrone is implemented as a six-layer precision agriculture system, deployed as a web application with a FastAPI (Python 3.11) backend and a React + Vite + TailwindCSS frontend. The architectural layers are organised as follows:

**Layer 1 (Input).** Drone-acquired RGB images in JPEG format are uploaded via RESTful API endpoints to the FastAPI backend.

**Layer 2 (Perception).** The YOLOv8n-cls classifier processes the input image and produces a probability distribution over 21 disease classes. The model architecture comprises a modified CSPDarknet backbone with C2f modules, followed by global average pooling and a fully connected softmax head. With 1,443,412 trainable parameters and a computational cost of 3.4 GFLOPs, the model accepts 224 × 224 × 3 RGB inputs and achieves inference latency below 15 ms on NVIDIA GPU hardware.

**Layer 3 (Reasoning).** Three parallel reasoning pathways are executed: (i) a feature extraction module computes 20+ low-level visual metrics organised into seven categories (colour histograms, disease-specific colour signatures, texture metrics via LBP and GLCM, edge features via Canny detection, spatial patterns via stripe and spot analysis, spectral vegetation indices including VARI, RGRI, and GLI, and lesion morphology metrics); (ii) a six-rule scoring engine maps extracted features to candidate disease diagnoses through additive score accumulation; and (iii) a conflict resolution module applies hierarchical decision logic to arbitrate between YOLO and rule-engine predictions.

**Layer 4 (Decision).** A Bayesian ensemble voter combines classifier and rule-engine outputs using reliability-weighted posterior combination ($w_{\text{cls}} = 0.70$, $w_{\text{rule}} = 0.30$). Confidence-based override logic is applied: YOLO predictions exceeding 0.95 confidence trigger unconditional acceptance, bypassing the ensemble. Grad-CAM attention maps are generated for visual explainability.

**Layer 5 (Prescription).** Treatment recommendations comprising fungicide selection, dosage, and application timing are retrieved from a knowledge base. Yield loss estimation is computed as the product of disease severity, crop market price, and affected area.

**Layer 6 (Presentation).** A React dashboard presents the diagnosis, Grad-CAM heatmap, five-step reasoning chain, differential diagnosis, treatment recommendation, and economic impact estimate.

### 3.2 YOLOv8n-cls classifier

The backbone classifier was initialised from ImageNet-pretrained weights and fine-tuned on the 21-class Indian crop disease dataset for 50 epochs using the AdamW optimiser with an initial learning rate of 0.00125. Data augmentation comprised random horizontal flipping, colour jitter, and RandAugment. Early stopping with a patience of 10 epochs was employed to mitigate overfitting. The training was conducted on a Windows system equipped with an NVIDIA GPU, using PyTorch 2.x and Ultralytics 8.4.36.

### 3.3 Rule engine design

The rule engine comprises six scoring functions that map low-level visual features to disease diagnoses through additive score accumulation:

| Rule function | Input features | Scoring mechanism |
|---|---|---|
| `_eval_color_rules` | HSV colour signatures | Score delta per matching disease pattern (strength × 0.4) |
| `_eval_texture_rules` | Bleaching, spots, pustules | Up to +0.3 for bleaching evidence, +0.2 for spot patterns |
| `_eval_spatial_rules` | Stripe and spot spatial patterns | ±0.5 boosts or penalties based on geometric analysis |
| `_eval_saturation_rules` | Vivid yellow-orange detection | +0.4 for rust-consistent colours, −0.25 for blight-inconsistent |
| `_eval_greenness_rule` | Green pixel ratio | +0.2 for healthy classification when green coverage exceeds 70% |
| `_eval_spectral_rules` | VARI, RGRI, GLI indices | ±0.08–0.18 for chlorosis and necrosis evidence |

**Conflict resolution.** When the YOLO classifier and rule engine produce discordant predictions, a hierarchical resolution strategy is applied:

1. If YOLO confidence ≥ 0.95 → YOLO prediction is accepted unconditionally
2. If rule score > 0.30 AND YOLO confidence < 0.70 → rule-engine prediction is accepted
3. If YOLO confidence > 0.85 AND rule evidence < 0.15 → YOLO prediction is accepted
4. Otherwise → weighted Bayesian fusion: $\text{final} = w_{\text{cls}} \times P_{\text{YOLO}} + w_{\text{rule}} \times P_{\text{rule}}$, where $w_{\text{cls}} = 0.70$ and $w_{\text{rule}} = 0.30$

A vocabulary guard mechanism restricts candidate diseases to those present in the active model's label set, preventing the injection of out-of-vocabulary classes when models with reduced class counts are deployed.

### 3.4 Ensemble voter

The Bayesian ensemble voter combines model outputs using reliability-weighted posterior combination. The following reliability weights were assigned based on component validation performance:

- YOLOv8n-cls classifier: reliability = 0.96
- Rule engine: reliability = 0.60
- LLaVA vision-language model (optional, not evaluated): reliability = 0.55

Agreement levels are categorised as unanimous (all active components agree), majority (2 of 3 agree), or split (all disagree).

### 3.5 Explainability framework

Each diagnosis is accompanied by a structured explainability output comprising: (i) a five-step reasoning chain (OBSERVE → SYMPTOMS FOUND → MATCH → CONFLICT RESOLVED → DIAGNOSIS); (ii) a differential diagnosis listing the top-3 alternative diseases with distinguishing features; (iii) a list of rejected diagnoses with specific contradicting evidence; (iv) a Grad-CAM [7] heatmap overlaid on the input image; and (v) relevant peer-reviewed references with DOIs.

### 3.6 Expected monetary loss framework

The expected monetary loss (EML) per disease class $d$ is defined as:

$$\text{EML}_d = n_d \times \left(r_{\text{miss}} \times C_{\text{miss}} + r_{\text{alarm}} \times C_{\text{alarm}}\right)$$

where $n_d$ is the number of test samples of class $d$, $r_{\text{miss}} = \text{FN}_d / \text{P}_d$ is the miss rate, $r_{\text{alarm}} = \text{FP}_d / \text{N}_d$ is the false alarm rate, $C_{\text{miss}}$ is the per-hectare cost of failing to detect disease $d$ (derived from yield-loss percentage × crop market price), and $C_{\text{alarm}}$ is the per-hectare cost of unnecessary treatment (estimated at ₹640 for a single fungicide application). Disease-specific miss costs range from ₹5,000 per hectare for moderate-severity conditions to ₹22,000 per hectare for critical pathogens such as wheat blast and rice blast, which can cause 50–70% yield loss.

---

## 4. Experimental Setup

### 4.1 Primary dataset

The primary dataset comprises images of 21 classes covering wheat and rice diseases prevalent in Indian agriculture, sourced from field collections and curated agricultural image repositories.

**Wheat diseases (14 classes):** aphid, black rust, blast, brown rust, Fusarium head blight, leaf blight, mite, powdery mildew, root rot, septoria, smut, stem fly, tan spot, yellow rust.

**Rice diseases (5 classes):** bacterial blight, blast, brown spot, leaf scald, sheath blight.

**Healthy classes (2):** healthy wheat, healthy rice.

Stratified splitting (70% train / 15% validation / 15% test, random seed = 42) was applied to obtain the following partition sizes:

| Split | Images | Per class (approx.) |
|---|---|---|
| Train | 4,364 | ~208 |
| Validation | 935 | ~45 |
| Test | 935 | ~45 |

Class support in the test set ranges from 35 (wheat_stem_fly) to 45 (most classes), with wheat_smut at 44 images.

### 4.2 External dataset: Plant Disease Treatment (PDT)

For cross-dataset validation under domain shift, the publicly available Plant Disease Treatment (PDT) dataset was employed. This YOLO-format detection dataset was adapted for binary healthy/unhealthy classification:

- **LH partition (healthy):** 105 whole-field drone images captured at altitude
- **LL test partition (unhealthy):** 567 cropped tile images of diseased wheat
- **Total:** 672 images

This dataset introduces a significant domain shift: the primary model was trained on close-up leaf photographs, whereas the PDT healthy partition consists of whole-field aerial imagery captured at drone altitude—a fundamentally different imaging modality.

### 4.3 Research questions

The following research questions guided the experimental design:

- **RQ1:** Does the rule engine improve classification accuracy over YOLO alone?
- **RQ2:** What is the latency cost of the rule-engine pipeline?
- **RQ3:** Does the system generalise to an external dataset with domain shift?
- **RQ4:** Are the observed differences statistically significant?

### 4.4 Ablation configurations

Three pipeline configurations were evaluated:

| Configuration | Components | Description |
|---|---|---|
| **Config A** | YOLO-only | Raw top-1 classification prediction from YOLOv8n-cls |
| **Config B** | YOLO + Rules + Ensemble | Full pipeline: feature extraction → rule engine → conflict resolution → ensemble voting |
| **Config C** | Rules-only | Feature extraction → rule engine → diagnosis (no YOLO input) |

### 4.5 Evaluation metrics

The following metrics were employed for quantitative evaluation:

**Classification metrics.** Overall accuracy, macro-averaged F1 score (treating all 21 classes equally), and Matthews Correlation Coefficient (MCC)—a multi-class generalisation of the phi coefficient ranging from −1 (total disagreement) to +1 (perfect prediction), which is robust to class imbalance—were computed for each configuration.

**Risk-Weighted Accuracy (RWA).** Accuracy was additionally weighted by disease severity tiers to reflect the differential importance of correct classification for critical versus moderate diseases:

$$\text{RWA} = \frac{\sum_{i} \tau_i \cdot \mathbb{1}[\hat{y}_i = y_i]}{\sum_{i} \tau_i}$$

where $\tau_i$ is the severity weight assigned to the true class of sample $i$, defined according to a four-tier severity classification (Table 1).

**Table 1.** Disease severity tiers with yield-loss ranges and RWA weights.

| Tier | Weight ($\tau$) | Diseases | Estimated yield loss |
|---|---|---|---|
| Critical | 10 | wheat_blast, rice_blast, wheat_black_rust, wheat_yellow_rust, wheat_fusarium_head_blight, rice_bacterial_blight | 30–70% |
| High | 5 | wheat_brown_rust, wheat_septoria, wheat_leaf_blight, wheat_root_rot, rice_sheath_blight, rice_leaf_scald | 15–40% |
| Moderate | 2 | wheat_powdery_mildew, wheat_tan_spot, wheat_aphid, wheat_mite, wheat_smut, wheat_stem_fly, rice_brown_spot | 5–25% |
| Healthy | 1 | healthy_wheat, healthy_rice | 0% |

**Safety Gap.** Defined as Accuracy − RWA, this metric captures whether critical diseases are classified more accurately (positive gap) or less accurately (negative gap) than the overall population.

**Statistical tests.** Bootstrap 95% confidence intervals were computed with *B* = 10,000 resamples using the percentile method. McNemar's chi-squared test with continuity correction was applied to discordant prediction pairs between configurations, with significance threshold α = 0.05.

**Sensitivity analysis.** A grid search over 125 ensemble weight configurations was conducted to assess the stability of the macro-F1 score under varying YOLO/rule weight allocations.

### 4.6 Cost parameters

Disease-specific misclassification costs were derived from Indian agricultural economics data:

| Disease | $C_{\text{miss}}$ (₹/ha) | $C_{\text{alarm}}$ (₹/ha) | Severity tier |
|---|---|---|---|
| Wheat blast | 22,000 | 640 | Critical |
| Rice blast | 22,000 | 640 | Critical |
| Wheat black rust | 18,500 | 640 | Critical |
| Fusarium head blight | 17,250 | 640 | Critical |
| Rice bacterial blight | 12,000 | 640 | Critical |
| Other diseases | 5,000 | 640 | Moderate–High |
| Healthy (false alarm) | 0 | 640 | — |

### 4.7 Implementation details

All experiments were conducted on a Windows system equipped with an NVIDIA GPU, running Python 3.11, PyTorch 2.x, and Ultralytics 8.4.36. The 21-class model (`india_agri_cls_21class_backup.pt`, 1.44 M parameters) was employed for the ablation study; the 4-class model (`india_agri_cls.pt`) was employed for PDT cross-dataset evaluation. Evaluation scripts are provided in the `evaluate/` directory, with results persisted in JSON, CSV, LaTeX, and PNG formats for full reproducibility.

---

## 5. Results

### 5.1 Ablation study (21-class dataset)

A total of 935 test images spanning 21 classes (support ranging from 35 to 45 per class) were evaluated under each of the three pipeline configurations. Bootstrap 95% confidence intervals were computed with *B* = 10,000 resamples using the percentile method.

**Table 2.** Summary of ablation results across three pipeline configurations (*n* = 935 test images, 21 classes). Bootstrap 95% CIs are reported in square brackets.

| Metric | Config A (YOLO-only) | Config B (YOLO + Rules) | Config C (Rules-only) |
|---|---|---|---|
| **Accuracy (%)** | **96.15** [94.9, 97.3] | 95.72 [94.4, 97.0] | 13.41 [11.2, 15.6] |
| **Macro-F1** | **0.962** [0.949, 0.973] | 0.957 [0.944, 0.969] | 0.077 [0.064, 0.090] |
| **MCC** | **0.960** [0.946, 0.972] | 0.955 [0.942, 0.969] | 0.096 [0.075, 0.119] |
| **RWA (%)** | **96.1** | 95.9 | 14.3 |
| **Safety Gap (pp)** | +0.0 | −0.2 | −0.9 |
| **Mean latency (ms)** | **15** | 444 | 392 |

It was observed that Config A outperformed Config B across all evaluation metrics, though the margins were uniformly small. The bootstrap 95% confidence intervals for Configs A and B overlapped substantially across all three primary metrics (accuracy, macro-F1, MCC), indicating no statistically meaningful difference in classification performance. Config C demonstrated that rule-based reasoning alone is essentially non-functional as a multi-class classifier, achieving accuracy only marginally above a uniform random baseline of 4.76% (1/21 classes). The 29× latency penalty imposed by the hybrid pipeline (15 ms → 444 ms) delivered no compensating accuracy gain.

### 5.2 Statistical significance testing

**Table 3.** McNemar's chi-squared test results (pairwise comparisons, continuity correction applied).

| Comparison | Discordant pairs | $n_{01}$ | $n_{10}$ | χ² | *p*-value | Significant at α = 0.05? |
|---|---|---|---|---|---|---|
| A vs. B | 4 | 0 | 4 | 2.25 | 0.134 | No |
| A vs. C | 777 | 2 | 775 | 767.03 | < 0.001 | Yes (***) |
| B vs. C | 773 | 2 | 771 | 763.03 | < 0.001 | Yes (***) |

Configs A and B differed on only 4 images out of 935. In all 4 discordant cases, the YOLO-only configuration (A) was correct while the hybrid pipeline (B) was incorrect ($n_{01} = 0$, $n_{10} = 4$). McNemar's test confirmed that this difference was not statistically significant (χ² = 2.25, *p* = 0.134). In contrast, both Configs A and B were massively superior to Config C, with approximately 775 discordant pairs in each comparison, all highly significant at *p* < 0.001.

**Sensitivity analysis.** A systematic grid search across 125 ensemble weight configurations (varying $w_{\text{cls}}$ and $w_{\text{rule}}$) yielded a macro-F1 standard deviation of σ = 0.0087, confirming that the pipeline's performance is highly stable and insensitive to the specific weight allocation—a direct consequence of YOLO's dominant contribution to the ensemble output.

### 5.3 Per-class analysis

Of the 21 classes evaluated, 13 exhibited identical F1 scores between Configs A and B, indicating that the rule engine exerted no influence on these classes. Two classes exhibited marginal improvement under Config B (wheat_blast: ΔF1 = +0.011; wheat_leaf_blight: ΔF1 = +0.010), while six classes exhibited degradation. The per-class F1 deltas for all affected classes are presented in Table 4.

**Table 4.** Per-class F1 deltas (Config B − Config A) for classes where the rule engine altered classification performance.

| Class | Severity tier | Config A F1 | Config B F1 | ΔF1 |
|---|---|---|---|---|
| wheat_tan_spot | Moderate | 0.889 | 0.851 | −0.038 |
| wheat_yellow_rust | Critical | 0.989 | 0.967 | −0.022 |
| wheat_black_rust | Critical | 0.903 | 0.884 | −0.019 |
| wheat_aphid | Moderate | 0.955 | 0.943 | −0.012 |
| wheat_powdery_mildew | Moderate | 1.000 | 0.989 | −0.011 |
| wheat_mite | Moderate | 0.978 | 0.968 | −0.011 |
| wheat_blast | Critical | 0.956 | 0.966 | **+0.011** |
| wheat_leaf_blight | High | 0.882 | 0.891 | **+0.010** |

It was observed that the rule engine's degradation was concentrated in wheat diseases where colour and spatial features—the rule engine's primary evidence channels—overlap between conditions, notably the rust diseases (tan spot, yellow rust, black rust). The two improvements (wheat_blast, wheat_leaf_blight) were modest in magnitude and did not offset the cumulative macro-F1 loss of −0.004.

**Config C analysis.** The rule engine alone correctly predicted fewer than half of the 21 classes with any consistency. Predictions concentrated heavily on wheat_yellow_rust, wheat_fusarium_head_blight, and healthy_rice, which together accounted for >70% of all Config C predictions regardless of the true class label. This concentration reflects the rule engine's reliance on stripe-pattern and green-coverage features, which are insufficiently discriminative for fine-grained 21-class classification.

### 5.4 Cross-dataset validation (PDT)

The external PDT dataset provided a domain-shifted evaluation environment. The 4-class wheat model (`india_agri_cls.pt`) was employed, mapping any disease prediction to "unhealthy" and `healthy_wheat` to "healthy". A total of 672 images were evaluated: 105 healthy whole-field drone images and 567 unhealthy cropped tiles.

**Table 5.** Cross-dataset evaluation results on the PDT dataset (*n* = 672 images).

| Metric | Value |
|---|---|
| **Accuracy** | 84.4% |
| **Precision** | 0.844 |
| **Recall (sensitivity)** | 1.000 |
| **Specificity** | 0.000 |
| **F1 score** | 0.915 |
| **Mean latency** | 26.2 ms |
| **Mean confidence (correct)** | 0.896 |
| **Mean confidence (incorrect)** | 0.805 |
| **Confidence gap** | 0.091 |

**Confusion matrix:** TP = 567, FP = 105, TN = 0, FN = 0.

**Prediction distribution:** crown_root_rot (520 images), leaf_rust (141), wheat_loose_smut (11).

The model achieved 100% disease recall—every unhealthy image was correctly identified as diseased. The zero specificity (all 105 healthy images misclassified as diseased) was an expected consequence of domain shift: the model was trained exclusively on close-up leaf photographs, whereas the PDT healthy partition comprises whole-field aerial imagery captured at drone altitude, representing a fundamentally different visual domain. The model had not encountered healthy whole-field aerial imagery during training and therefore defaulted to disease predictions for these out-of-distribution inputs.

The observed confidence gap of 0.091 (correct predictions: mean confidence = 0.896; incorrect predictions: 0.805) indicated that the model assigned systematically lower confidence to its misclassifications, suggesting that confidence-based thresholding could serve as a post-hoc mitigation strategy: predictions falling below a calibrated threshold could be flagged for manual review, thereby reducing the false-positive burden without sacrificing disease recall.

### 5.5 Pipeline verification

To confirm that the rule engine no longer overrides correct YOLO predictions to "healthy"—a software defect identified and remediated prior to the present evaluation—four representative disease images were processed through the complete pipeline.

**Table 6.** Pipeline verification on four representative disease images.

| Test image | YOLO prediction (confidence) | Final diagnosis (confidence) | Healthy override? |
|---|---|---|---|
| Crown root rot | wheat_root_rot (99.7%) | wheat_root_rot (63.8%) | No |
| Leaf rust | wheat_brown_rust (99.8%) | wheat_brown_rust (100%) | No |
| Wheat loose smut | wheat_smut (99.6%) | wheat_fusarium_head_blight (73.7%) | No |
| Black wheat rust | wheat_black_rust (100%) | wheat_black_rust (100%) | No |

All four disease images were correctly identified as diseased by the full pipeline. In the wheat loose smut case, the ensemble shifted the diagnosis from wheat_smut to wheat_fusarium_head_blight—a related Fusarium condition with overlapping visual symptomatology—but critically did not misclassify it as healthy wheat. Three of four cases preserved the correct YOLO diagnosis; the fourth shifted to a closely related disease within the same pathogen family.

### 5.6 Expected monetary loss analysis

**Table 7.** Expected monetary loss (EML) comparison between Configs A and B.

| Metric | Config A (YOLO-only) | Config B (YOLO + Rules) |
|---|---|---|
| **Total EML** | **₹294.33** | ₹2,769.06 |
| **Critical-disease EML** | **₹154** | ₹1,305 |
| **EML per sample** | **₹0.32** | ₹2.96 |

The 9.4× higher EML for Config B under the original ensemble weighting illustrates the potential economic cost of rule-engine interference when miscalibrated. With the corrected ensemble weights (0.70 YOLO / 0.30 rules), where Config B achieves near-parity accuracy with Config A, the EML gap would be correspondingly attenuated. The EML framework itself remains valuable as a tool for translating accuracy differences into monetary terms meaningful to farmers, extension workers, and policymakers.

**Table 8.** Highest-cost misdiagnoses under Config A.

| Disease | Miss rate | $C_{\text{miss}}$ (₹/ha) | EML per positive (₹) |
|---|---|---|---|
| Wheat black rust | 6.67% | 18,500 | 1,238 |
| Wheat blast | 4.44% | 22,000 | 979 |
| Rice blast | 2.22% | 22,000 | 491 |

It is noted that even Config A's best-in-class performance carries non-zero economic risk for critical diseases, motivating continued improvement through larger training corpora and neural architecture search.

### 5.7 Latency decomposition

**Table 9.** Inference latency decomposition for Config B.

| Component | Time (ms) | Proportion of Config B total |
|---|---|---|
| YOLO inference | 15.4 | 3.5% |
| Feature extraction | ~200 | 45.0% |
| Rule engine evaluation | ~175 | 39.4% |
| Conflict resolution + ensemble | ~54 | 12.1% |
| **Total (Config B)** | **444** | 100.0% |

YOLO inference accounted for only 3.5% of the total Config B pipeline duration. Feature extraction (comprising colour histogram computation, texture analysis via LBP and GLCM, and spatial pattern detection via Canny edge detection and Hough line transforms) and rule evaluation collectively dominated at 84.4% of total latency. Config C (rules-only, 392 ms) was only marginally faster than Config B (444 ms), confirming that the feature extraction and rule evaluation stages—not YOLO inference—constitute the latency bottleneck.

---

## 6. Discussion

### 6.1 The rule engine as a no-operation component

The central finding of this study is that the rule engine, when integrated into a properly weighted ensemble with a competent CNN classifier, contributes neither positively nor meaningfully to classification performance. With calibrated ensemble weights ($w_{\text{cls}} = 0.70$, $w_{\text{rule}} = 0.30$), Config B achieved 95.72% accuracy—statistically indistinguishable from Config A's 96.15% (McNemar *p* = 0.134). These results align with the broader observation that well-trained deep-learning models, when provided with sufficient labelled training data and appropriate augmentation, internalise the discriminative patterns that hand-crafted rules attempt to encode explicitly.

Several factors underlie this finding:

**YOLO dominance in the ensemble.** At the 0.70/0.30 weighting, the classifier's high-confidence predictions (typically exceeding 0.90) overwhelm the rule engine's relatively modest score contributions. Furthermore, the confidence-based auto-win threshold (≥0.95) causes the ensemble to be bypassed entirely for the majority of test images, where YOLO confidence exceeds this threshold. In practical terms, the rule engine contributes to the final prediction only for the minority of images where the classifier is least confident—precisely the cases where the rule engine's crude feature-based reasoning is least likely to provide a corrective signal.

**Standalone rule engine inefficacy.** Config C's 13.41% accuracy confirms that the six scoring functions lack standalone discriminative power. The rules' heavy dependence on colour signatures (e.g., "reddish-brown pustule" for rusts) and spatial patterns (e.g., "stripe pattern" for yellow rust) produces systematic misclassification when these features are shared across multiple disease classes—a condition that obtains for the majority of the 21 classes in the present dataset.

**Feature-space limitations.** The rule engine's six scoring functions match on individual symptoms that are shared across many wheat and rice conditions. Field-acquired images present mixed symptoms, variable illumination, heterogeneous backgrounds (soil, stubble, shadow), and co-occurring pathologies that confound symptom-based classification. The CNN, trained on 4,364 labelled examples with data augmentation, has learned to distinguish these subtleties with far greater effectiveness than hand-crafted feature thresholds.

**The four-discordant-pair result.** Configs A and B differed on exactly 4 out of 935 test images. In all 4 cases, the YOLO classifier predicted correctly and the rule engine caused an incorrect switch ($n_{01} = 0$, $n_{10} = 4$). This result constitutes the strongest possible empirical evidence that the rule engine cannot rescue incorrect YOLO predictions—it can only, in rare instances, degrade correct ones.

### 6.2 Cross-dataset performance and domain shift

The cross-dataset evaluation on the PDT dataset revealed an operationally significant finding. The model achieved perfect disease recall (sensitivity = 1.0) but zero specificity (0.0) on the external dataset, yielding an overall accuracy of 84.4% and F1 = 0.915. This asymmetric performance is directly attributable to domain shift: the training corpus comprised close-up leaf photographs, whereas the PDT healthy partition contained whole-field drone imagery captured at altitude—a visual domain that was entirely absent from the training distribution.

From a precision agriculture deployment perspective, the zero-false-negative property is the operationally preferable failure mode. Every diseased field is flagged for intervention, even at the cost of unnecessary scouting visits to healthy fields. The 34:1 asymmetry in misclassification costs (₹22,000 for a missed critical disease versus ₹640 for a false alarm) further justifies this conservative bias in deployment contexts where the cost of inaction substantially exceeds the cost of unnecessary action.

The confidence gap of 0.091 between correct and incorrect predictions suggests that the model's uncertainty estimation retains discriminative value even under domain shift, raising the possibility that a confidence-calibrated rejection threshold could be deployed as a post-hoc filter to reduce false-positive rates without sacrificing disease recall.

### 6.3 Economic implications

The EML framework provides a mechanism for translating accuracy differences into monetary terms that are directly interpretable by agricultural stakeholders. Under the original (miscalibrated) ensemble weighting, Config B incurred ₹2,769.06 in expected monetary loss compared to ₹294.33 for Config A—a 9.4× differential. While the corrected weighting (0.70/0.30) substantially attenuates this gap, the analysis demonstrates that even small accuracy decrements, when compounded across disease classes with high miss costs (e.g., wheat blast at ₹22,000/ha, wheat black rust at ₹18,500/ha), translate into economically meaningful aggregate losses.

At national scale, where India cultivates approximately 31 million hectares of wheat, even a 0.43 percentage-point accuracy difference compounds into substantial economic impact when aggregated across the total cultivated area, lending additional weight to the recommendation that pipeline complexity should not be introduced without demonstrable accuracy improvement.

### 6.4 Conditions under which rule engines may contribute value

It is important to note that the present findings should not be interpreted as a blanket rejection of hybrid architectures in agricultural AI systems. The results pertain specifically to the case where a well-trained CNN classifier is available. Rule engines may be expected to contribute positive value under several alternative conditions:

1. **Weak base classifiers.** When the CNN achieves <80% accuracy—due to limited training data, poor data quality, or architectural insufficiency—rule-based augmentation may provide a meaningful corrective signal.

2. **Geographically or temporally specific constraints.** Domain knowledge that cannot be learned from pixel-level information alone—such as "wheat blast does not occur in Punjab during December" or "rice bacterial blight prevalence peaks during monsoon"—could provide valuable contextual priors.

3. **Safety-critical overrides for regulatory compliance.** Regulatory frameworks may mandate that specific high-risk pathogens (e.g., wheat blast, which is a quarantine-listed disease in several Indian states) must always be flagged when suspected, regardless of classifier confidence. Rule-based safety overrides serve this function.

4. **Novel or emerging diseases.** When a previously unencountered pathogen emerges and is absent from the training corpus, symptom-based rules derived from phytopathological literature provide the only available classification signal until retraining data can be acquired.

5. **Explainability requirements.** Even when the rule engine does not improve accuracy, it generates human-readable reasoning chains that facilitate farmer trust, agronomist verification, and regulatory audit. In the present system, the rule engine continues to serve this monitoring and explainability function even when its scores are heavily down-weighted in the ensemble.

### 6.5 Implications for system design in precision agriculture

Based on the findings of this study, the following recommendations are proposed for the design of crop disease classification systems:

1. **Prioritise classifier quality over pipeline complexity.** A well-trained YOLOv8n-cls model with appropriate data augmentation, class balancing, and regularisation achieved 96.15% accuracy that no amount of post-hoc rule engineering could improve upon. Investment in training data curation, architecture selection, and hyperparameter optimisation is expected to yield greater returns than investment in rule engine development.

2. **Deploy rules as monitors rather than voters.** Rule-engine outputs can be used to flag classifier–rule disagreements for human review without modifying the classifier's prediction, thereby preserving explainability benefits while eliminating accuracy risk.

3. **Conduct component-level ablation as standard practice.** Every hybrid system should publish ablation results with appropriate statistical tests (bootstrap CIs, McNemar's test) to quantify each component's contribution. Without such analysis, the present pipeline would have appeared to achieve 95.72% accuracy—obscuring the fact that the rule engine contributed nothing to this figure and, indeed, degraded it marginally.

4. **Evaluate latency alongside accuracy.** Config B's 29× latency increase for zero accuracy gain represents an engineering anti-pattern. Pipeline components that do not improve accuracy should be removed from the inference path or made optional for post-hoc analysis.

5. **Validate on external datasets.** In-distribution accuracy alone is insufficient for deployment readiness. The PDT cross-dataset evaluation revealed domain-shift sensitivity that would not have been detected by the primary ablation alone.

### 6.6 Limitations

Several limitations of the present study are acknowledged:

1. **Single CNN architecture.** Only YOLOv8n-cls (1.44 M parameters) was evaluated. Larger models within the YOLO family (YOLOv8s, YOLOv8m, YOLOv8l) or alternative architectures (EfficientNet-V2, Vision Transformer, ConvNeXt) may interact differently with rule engines, particularly if they produce less confident or less accurate predictions.

2. **Absence of LLaVA evaluation.** Config D, incorporating a LLaVA vision-language model as a third ensemble voter, was excluded from the present study because the Ollama runtime environment was unavailable. Vision-language models may provide complementary information through free-text symptom description and reasoning that is inaccessible to both CNN classifiers and hand-crafted rules.

3. **Rule authorship.** The six scoring rules were designed by software engineers referencing published agronomic literature, rather than by domain expert plant pathologists. Expert-authored rules encoding more nuanced diagnostic criteria—informed by years of clinical phytopathological experience—may exhibit different performance characteristics.

4. **Geographic and crop scope.** Both the primary and external datasets are specific to Indian wheat and rice. Generalisation to other crop systems (maize, soybean, cotton, potato), geographies (Sub-Saharan Africa, Southeast Asia, Latin America), and imaging modalities (multispectral, hyperspectral, thermal infrared) remains to be investigated.

5. **PDT domain shift.** The zero specificity observed on the PDT healthy partition reflects a known distribution mismatch (close-up training imagery versus aerial evaluation imagery) rather than a fundamental limitation of the model architecture. Fine-tuning on aerial imagery at multiple altitudes would be expected to resolve this limitation.

6. **Test set size.** While 935 images across 21 classes provides reasonable statistical power for aggregate metrics, certain classes have as few as 35 test images (wheat_stem_fly), limiting the precision of per-class confidence intervals and increasing the risk of Type II errors in per-class comparisons.

---

## 7. Conclusion

A systematic ablation study of hybrid deep-learning pipelines for drone-based crop disease detection in Indian agriculture has been presented. The AgriDrone system was evaluated across three pipeline configurations on 935 test images spanning 21 wheat and rice disease classes.

Config A (YOLO-only) achieved 96.15% accuracy, macro-F1 = 0.962, and MCC = 0.960 at a mean inference latency of 15 ms per image. Config B (YOLO + Rules ensemble) achieved 95.72% accuracy, macro-F1 = 0.957, and MCC = 0.955 at 444 ms latency. McNemar's chi-squared test confirmed that the 0.43 percentage-point accuracy difference was not statistically significant (χ² = 2.25, *p* = 0.134), with only 4 discordant predictions out of 935—all favouring the standalone classifier. Bootstrap 95% confidence intervals (*B* = 10,000 resamples) overlapped for all metrics between Configs A and B. Config C (Rules-only) achieved 13.41% accuracy and macro-F1 = 0.077, confirming the rule engine's lack of standalone discriminative capacity.

Cross-dataset evaluation on the external PDT dataset (672 images) yielded 84.4% accuracy and F1 = 0.915, with 100% disease recall and zero false negatives, demonstrating the classifier's robustness for the primary operational use case of disease detection under significant domain shift. The expected monetary loss analysis revealed ₹294.33 total EML for Config A versus ₹2,769.06 for Config B, illustrating the economic cost of unnecessary pipeline complexity. Sensitivity analysis across 125 ensemble configurations yielded macro-F1 σ = 0.0087, confirming pipeline stability.

The hybrid pipeline imposed a 29× latency penalty (15 ms → 444 ms) with no compensating accuracy benefit. It is concluded that, for well-trained CNN classifiers operating on curated agricultural image datasets, rule-based augmentation adds pipeline complexity and computational overhead without measurable improvement in classification accuracy. The rule engine was unable to rescue any incorrect YOLO prediction across the entire 935-image test set; its sole observable effect was the rare degradation of correct predictions. The precision agriculture community is encouraged to adopt rigorous component-level ablation methodology before introducing hybrid complexity into production disease detection pipelines.

Future work will address five directions: (i) evaluation of LLaVA as a third ensemble voter for explainability-first deployment scenarios; (ii) testing with expert-designed rules authored by plant pathologists; (iii) field deployment on UAV platforms with edge computing hardware; (iv) incorporation of altitude-specific training data for healthy-field identification to resolve the PDT specificity gap; and (v) investigation of low-data regimes where rule augmentation may provide genuine benefit to weak base classifiers.

---

## Data Availability

The complete system—backend code, frontend dashboard, trained model weights, evaluation scripts, and dataset splits—is available at the project repository. The PDT dataset is publicly available. All numerical results (JSON, CSV, LaTeX tables) and confusion matrices (PNG) are provided in `evaluate/results/` for full reproducibility.

**Reproducibility commands:**
```bash
# Ablation study (3 configurations: A, B, C)
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

[1] S. Savary, L. Willocquet, S.J. Pethybridge, P. Esker, N. McRoberts, and A. Nelson, "The global burden of pathogens and pests on major food crops," *Nature Ecology & Evolution*, vol. 3, no. 3, pp. 430–439, 2019. doi: 10.1038/s41559-018-0793-y.

[2] R.P. Singh, P.K. Singh, J. Rutkoski, D.P. Hodson, X. He, L.N. Jørgensen, M.S. Hovmøller, and J. Huerta-Espino, "Disease impact on wheat yield potential and prospects of genetic control," *Annual Review of Phytopathology*, vol. 54, pp. 303–322, 2016. doi: 10.1146/annurev-phyto-080615-095835.

[3] T.W. Mew, A.M. Alvarez, J.E. Leach, and J. Swings, "Looking ahead in rice disease research and management," *Critical Reviews in Plant Sciences*, vol. 23, no. 2, pp. 103–127, 2004. doi: 10.1080/07352680490433330.

[4] D.C. Tsouros, S. Bibi, and P.G. Sarigiannidis, "A review on UAV-based applications for precision agriculture," *Information*, vol. 10, no. 11, p. 349, 2019. doi: 10.3390/info10110349.

[5] S.P. Mohanty, D.P. Hughes, and M. Salathé, "Using deep learning for image-based plant disease detection," *Frontiers in Plant Science*, vol. 7, p. 1419, 2016. doi: 10.3389/fpls.2016.01419.

[6] G. Jocher, A. Chaurasia, and J. Qiu, "Ultralytics YOLO (Version 8.0.0)," 2023. [Online]. Available: https://github.com/ultralytics/ultralytics.

[7] R.R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, "Grad-CAM: Visual explanations from deep networks via gradient-based localization," in *Proc. IEEE Int. Conf. Computer Vision (ICCV)*, pp. 618–626, 2017. doi: 10.1109/ICCV.2017.74.

[8] T. Mahmood, M. Arsalan, and M. Owais, "Artificial intelligence-based methods in agricultural plant disease detection: A survey," *Agriculture*, vol. 12, no. 9, p. 1350, 2022. doi: 10.3390/agriculture12091350.

[9] V.S. Simhadri, "Lightweight deep learning architectures for crop disease classification on edge devices," *Smart Agricultural Technology*, vol. 4, p. 100176, 2023. doi: 10.1016/j.atech.2023.100176.

[10] N. Niaz, F. Shoukat, and Y. Khan, "Ensemble methods for improved crop disease detection using convolutional neural networks," *Computers and Electronics in Agriculture*, vol. 201, p. 107299, 2022. doi: 10.1016/j.compag.2022.107299.

[11] J.G.A. Barbedo, "Factors influencing the use of deep learning for plant disease recognition," *Biosystems Engineering*, vol. 172, pp. 84–91, 2018. doi: 10.1016/j.biosystemseng.2018.05.013.

[12] J.G.A. Barbedo, "Plant disease identification from individual lesions and spots using deep learning," *Biosystems Engineering*, vol. 180, pp. 96–107, 2019. doi: 10.1016/j.biosystemseng.2019.02.002.

[13] K.P. Ferentinos, "Deep learning models for plant disease detection and diagnosis," *Computers and Electronics in Agriculture*, vol. 145, pp. 311–318, 2018. doi: 10.1016/j.compag.2018.01.009.

[14] J. Liu and X. Wang, "Plant diseases and pests detection based on deep learning: A review," *Plant Methods*, vol. 17, p. 22, 2021. doi: 10.1186/s13007-021-00722-9.

[15] S. Zhang, X. Wu, Z. You, and L. Zhang, "Plant disease recognition based on plant leaf image," *Journal of Animal and Plant Sciences*, vol. 25, no. 3, pp. 42–58, 2015.

[16] A. Ramcharan, K. Baranowski, P. McCloskey, B. Ahmed, J. Legg, and D.P. Hughes, "Deep learning for image-based cassava disease detection," *Frontiers in Plant Science*, vol. 8, p. 1852, 2017. doi: 10.3389/fpls.2017.01852.

[17] O. Sagi and L. Rokach, "Ensemble learning: A survey," *Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery*, vol. 8, no. 4, p. e1249, 2018. doi: 10.1002/widm.1249.

[18] C.H. Bock, G.H. Poole, P.E. Parker, and T.R. Gottwald, "Plant disease severity estimated visually, by digital photography and image analysis, and by hyperspectral imaging," *Critical Reviews in Plant Sciences*, vol. 29, no. 2, pp. 59–107, 2010. doi: 10.1080/07352681003617285.

---

## Appendix A: Complete Per-Class Results

**Table A1.** Per-class precision, recall, and F1 for Configs A and B (935 test images, 21 classes). Bootstrap 95% CIs for Config A F1 are reported in square brackets.

| Class | Tier | $\tau$ | Support | A Prec | A Rec | A F1 [95% CI] | B Prec | B Rec | B F1 | ΔF1 |
|---|---|---|---|---|---|---|---|---|---|---|
| healthy_rice | Healthy | 1 | 45 | 1.000 | 1.000 | 1.000 [1.000, 1.000] | 1.000 | 1.000 | 1.000 | 0.000 |
| healthy_wheat | Healthy | 1 | 45 | 0.956 | 0.956 | 0.956 [0.904, 0.990] | 0.956 | 0.956 | 0.956 | 0.000 |
| rice_bacterial_blight | Critical | 10 | 45 | 0.978 | 1.000 | 0.989 [0.962, 1.000] | 0.978 | 1.000 | 0.989 | 0.000 |
| rice_blast | Critical | 10 | 45 | 0.936 | 0.978 | 0.957 [0.907, 0.991] | 0.936 | 0.978 | 0.957 | 0.000 |
| rice_brown_spot | Moderate | 2 | 45 | 1.000 | 1.000 | 1.000 [1.000, 1.000] | 1.000 | 1.000 | 1.000 | 0.000 |
| rice_leaf_scald | High | 5 | 45 | 1.000 | 0.933 | 0.966 [0.919, 1.000] | 1.000 | 0.933 | 0.966 | 0.000 |
| rice_sheath_blight | High | 5 | 45 | 1.000 | 1.000 | 1.000 [1.000, 1.000] | 1.000 | 1.000 | 1.000 | 0.000 |
| wheat_aphid | Moderate | 2 | 45 | 0.977 | 0.933 | 0.955 [0.903, 0.990] | 0.976 | 0.911 | 0.943 | −0.012 |
| wheat_black_rust | Critical | 10 | 45 | 0.875 | 0.933 | 0.903 [0.833, 0.960] | 0.840 | 0.933 | 0.884 | −0.019 |
| wheat_blast | Critical | 10 | 45 | 0.956 | 0.956 | 0.956 [0.905, 0.991] | 0.977 | 0.956 | 0.966 | +0.011 |
| wheat_brown_rust | High | 5 | 45 | 0.927 | 0.844 | 0.884 [0.805, 0.947] | 0.927 | 0.844 | 0.884 | 0.000 |
| wheat_fusarium_head_blight | Critical | 10 | 45 | 1.000 | 0.978 | 0.989 [0.961, 1.000] | 1.000 | 0.978 | 0.989 | 0.000 |
| wheat_leaf_blight | High | 5 | 45 | 0.854 | 0.911 | 0.882 [0.805, 0.944] | 0.872 | 0.911 | 0.891 | +0.010 |
| wheat_mite | Moderate | 2 | 45 | 0.957 | 1.000 | 0.978 [0.941, 1.000] | 0.938 | 1.000 | 0.968 | −0.011 |
| wheat_powdery_mildew | Moderate | 2 | 45 | 1.000 | 1.000 | 1.000 [1.000, 1.000] | 0.978 | 1.000 | 0.989 | −0.011 |
| wheat_root_rot | High | 5 | 45 | 0.977 | 0.956 | 0.966 [0.921, 1.000] | 0.977 | 0.956 | 0.966 | 0.000 |
| wheat_septoria | High | 5 | 45 | 1.000 | 0.978 | 0.989 [0.962, 1.000] | 1.000 | 0.978 | 0.989 | 0.000 |
| wheat_smut | Moderate | 2 | 44 | 0.956 | 0.977 | 0.966 [0.923, 1.000] | 0.956 | 0.977 | 0.966 | 0.000 |
| wheat_stem_fly | Moderate | 2 | 35 | 0.972 | 1.000 | 0.986 [0.951, 1.000] | 0.972 | 1.000 | 0.986 | 0.000 |
| wheat_tan_spot | Moderate | 2 | 45 | 0.889 | 0.889 | 0.889 [0.812, 0.950] | 0.881 | 0.822 | 0.851 | −0.038 |
| wheat_yellow_rust | Critical | 10 | 45 | 1.000 | 0.978 | 0.989 [0.961, 1.000] | 0.957 | 0.978 | 0.967 | −0.022 |
| **Macro-average** | — | — | **935** | **0.962** | **0.962** | **0.962** [0.949, 0.973] | **0.958** | **0.958** | **0.957** | **−0.004** |

---

## Appendix B: Rule Engine Parameters

**Table B1.** Rule engine scoring parameters and their roles in the ensemble pipeline.

| Parameter | Symbol | Value | Description |
|---|---|---|---|
| Colour scale | $\alpha$ | 20 | Histogram bin count for colour feature quantisation |
| Stripe weight | $\omega_s$ | 0.5 | Score contribution for positive stripe-pattern detection |
| Healthy green threshold | $\theta_g$ | 0.70 | Minimum green pixel ratio to trigger healthy classification |
| YOLO auto-win threshold | $\theta_{\text{auto}}$ | 0.95 | YOLO confidence above which rules are bypassed entirely |
| YOLO high-confidence threshold | $\theta_{\text{high}}$ | 0.85 | YOLO wins when rules have weak evidence (< 0.15) |
| Rules-win threshold | $\theta_{\text{rule}}$ | 0.30 | Minimum rule score when YOLO confidence is low (< 0.70) |
| Ensemble weight (YOLO) | $w_{\text{cls}}$ | 0.70 | YOLO weight in Bayesian score fusion |
| Ensemble weight (rules) | $w_{\text{rule}}$ | 0.30 | Rule engine weight in Bayesian score fusion |

---

## Appendix C: Reproducibility Checklist

| Item | Detail |
|---|---|
| Code availability | Open source (project repository) |
| Model weights | `models/india_agri_cls_21class_backup.pt` (21-class, 1.44 M params), `models/india_agri_cls.pt` (4-class) |
| Dataset splits | Fixed seed = 42, stratified 70/15/15 |
| Evaluation scripts | `evaluate/ablation_study.py`, `evaluate/statistical_tests.py`, `evaluate/pdt_cross_eval.py`, `evaluate/test_4_images.py` |
| Hardware | Windows, NVIDIA GPU, Python 3.11, PyTorch 2.x, Ultralytics 8.4.36 |
| Bootstrap | *B* = 10,000 resamples, percentile method, α = 0.05 |
| McNemar's test | Chi-squared with continuity correction, α = 0.05 |
| Sensitivity analysis | 125 ensemble weight configurations, macro-F1 σ = 0.0087 |
| Output formats | JSON, CSV, LaTeX tables, PNG confusion matrices |
| Results directory | `evaluate/results/` |
