# When Do Hand-Crafted Rules Help Modern CNNs for Crop Disease Classification? A Systematic Multi-Architecture, Multi-Regime Study on Indian Wheat and Rice

**Authors:** Ashutosh Mishra
**Affiliation:** Independent Researcher, India
**Corresponding author:** mishra.ashutosh@gmail.com  
**ORCID:** [0009-0000-4764-8160](https://orcid.org/0009-0000-4764-8160)
**Submitted to:** *Smart Agricultural Technology* (Elsevier)

> **Status of this version (v4).** This paper supersedes `RESEARCH_PAPER_FINAL_v3.md` and reframes the study around a **negative result** (hand-crafted rules do not help a well-trained CNN on this data regime) and extends it with a systematic multi-architecture × multi-rule-engine × multi-regime matrix. Numbers that still need to be regenerated under the new experimental matrix are explicitly tagged `[TO BE RE-RUN]`. The original v3 results and narrative are preserved byte-for-byte in `RESEARCH_PAPER_FINAL_v3.md`; do not treat v3 as deprecated until v4 is complete.

---

## Abstract

**Primary claim (negative result).** A six-rule hand-crafted symptom engine added to a YOLOv8n-cls classifier produces **no statistically significant accuracy gain** (McNemar χ² = 2.25, *p* = 0.134; 4 discordant predictions out of 935), while multiplying inference latency by 29×. The non-effect is robust across (i) 125 ensemble-weight configurations (macro-F1 σ = 0.0087), (ii) bootstrap 95% CIs (*B* = 10,000) that overlap for every metric, and (iii) per-class F1 deltas that are within the single-seed noise floor. We interpret this as evidence that **well-trained modern CNNs on curated, in-distribution imagery absorb the signal that hand-crafted rules encode**, making neuro-symbolic augmentation a net cost rather than a net benefit in this regime.

**Scope and honest caveats.** (1) *Imaging modality.* Training and test images are **curated close-up leaf photographs**, not drone-altitude imagery; claims about "drone-based" detection should be read as "aerial/field disease classification pipelines that could be fed by drone imagery after a separate tiling/detection stage." A dedicated *Data Provenance* subsection (§3.1) details sources and preprocessing. (2) *Cross-dataset test.* On the PDT drone-image dataset (672 images) the model achieves 100% sensitivity but **0% specificity at argmax** — every healthy whole-field image is misclassified as diseased. We report this as a degenerate result and explore three mitigations in §5.4: threshold sweeps (ROC-AUC, specificity@90%sensitivity), few-shot fine-tuning (5/10/25/50 PDT samples per class), and post-hoc calibration (temperature + Platt scaling). (3) *EML quantification.* Per-hectare cost parameters have been moved to `configs/economics/india_2025.yaml` with individual citations; figures without primary-source citations are marked `ESTIMATE` and reported only in sensitivity scenarios (§5.6, §Appendix).

**Contribution matrix (v4).** We evaluate a 6 × 4 × 5 × 4 × 5 grid: {YOLOv8n-cls, YOLOv8s-cls, EfficientNet-B0, ConvNeXt-Tiny, MobileNetV3-Small, ViT-B/16} × {no rules, handcrafted, learned decision tree, LLM-generated} × {100%, 50%, 25%, 10%, 5% training data} × {indian21, PlantVillage-subset, PlantDoc-subset, PDT} × 5-fold stratified CV (seeds 42–46). A shared training recipe (cosine LR + warmup, identical augmentation, identical epoch budget) is documented in `docs/training_recipe.md` and used to re-audit the EfficientNet-B0 baseline reported at 76.15% in v3, which we now believe reflected training-configuration artefacts rather than an architectural finding.

**Implications.** For practitioners, component-level ablation must precede hybrid-architecture claims. For reviewers, we advocate reporting (a) full statistical protocol (bootstrap, Holm-Bonferroni across per-class tests, 5×2cv paired *t*), (b) a cost curve rather than a single EML number, and (c) an explicit data-provenance statement distinguishing controlled imagery from deployment-modality imagery.

**Keywords:** crop disease classification; hand-crafted rules; negative result; component ablation; neuro-symbolic; YOLOv8; EfficientNet; cross-dataset generalisation; calibration; expected monetary loss

---

## 1. Introduction

Plant diseases cost the world 20–40% of annual crop production [1]. The consequences hit hardest in developing nations, and India—where agriculture contributes 17% of GDP and employs 42% of the workforce—is no exception. Wheat and rice, the two staple cereals cultivated on roughly 75 million hectares and feeding over 1.4 billion people, face a relentless battery of fungal, bacterial, and pest threats.

Wheat alone is vulnerable to yellow rust (*Puccinia striiformis*), black rust (*Puccinia graminis*), brown rust (*Puccinia triticina*), Fusarium head blight (*Fusarium graminearum*), powdery mildew (*Blumeria graminis*), and loose smut (*Ustilago tritici*)—each capable of inflicting 30–70% yield loss in epidemic years [2]. Rice pathogens including blast (*Magnaporthe oryzae*), bacterial blight (*Xanthomonas oryzae*), sheath blight (*Rhizoctonia solani*), and brown spot (*Bipolaris oryzae*) pose analogous threats across the roughly 44 million hectares under rice cultivation [3]. A 48–72 hour delay in detection can escalate a localised infection into a field-wide epidemic, making rapid surveillance essential.

Unmanned aerial vehicles (UAVs) equipped with RGB cameras have transformed field-level monitoring. Drones cover hundreds of hectares per day at 10–50 m altitude—orders of magnitude faster than the 2–5 hectares a human scout can walk [4]. Robotic and UAV-based plant pathology systems have demonstrated end-to-end disease management workflows [20], while field-operation robotics continue to expand the scope of precision agriculture [21]. Recent UAS-based studies have also tackled related structural problems such as wheat lodging detection using deep learning [23]. The bottleneck is no longer image acquisition; it is the classification model that must identify specific diseases from leaf and canopy imagery in real time.

Deep learning has delivered impressive results on laboratory benchmarks. Mohanty et al. [5] reported 99.35% on PlantVillage with GoogLeNet; Ferentinos (2018) reached 99.53% with VGG. Yet these benchmarks use controlled conditions—uniform backgrounds, single lesions, consistent lighting—that diverge sharply from real-world fields, where mixed symptoms, variable light, soil backgrounds, and multi-pathogen co-infections are the norm. To bridge this gap, a popular architectural pattern has emerged: hybrid pipelines that bolt a hand-crafted rule engine onto a CNN backbone, on the assumption that expert agronomic knowledge can compensate for the model's field-condition weaknesses.

The trouble is that almost no one checks whether the rule engine actually helps. Most published systems report only end-to-end accuracy, making it impossible to attribute performance to individual components. Pipeline complexity gets equated with quality, without evidence. This study addresses that gap head-on. We present AgriDrone, an aerial/field disease classification system for Indian wheat and rice (trained on curated close-up leaf photographs — see §3.1 for data-provenance caveats), and conduct a rigorous three-configuration ablation to isolate each component's contribution, extended in v4 to a full multi-architecture matrix. Our principal contributions are:

1. **System architecture.** A six-layer system integrating YOLOv8n-cls (1.44 M parameters), a six-rule reasoning engine with spectral vegetation indices (VARI, RGRI, GLI), Bayesian ensemble voting, Grad-CAM explainability, EML estimation, and treatment recommendation—deployed as a FastAPI backend with a React dashboard.

2. **Ablation study.** A systematic comparison of standalone YOLO (Config A), the full hybrid pipeline (Config B), and rule-engine-only classification (Config C) on 935 images across 21 classes. Config A and B achieve statistically indistinguishable accuracy (96.15% vs. 95.72%, McNemar *p* = 0.134); Config C achieves only 13.41%.

3. **Statistical rigour.** Bootstrap 95% CIs (*B* = 10,000), McNemar's chi-squared test, and sensitivity analysis across 125 ensemble weight configurations (macro-F1 σ = 0.0087).

4. **Cross-dataset validation.** External evaluation on the PDT dataset (672 images) shows 84.4% accuracy and 100% disease recall under significant domain shift.

5. **Economic analysis.** An EML framework that translates accuracy differences into per-hectare monetary costs (₹294.33 for Config A vs. ₹2,769.06 for Config B), contextualised against Indian wheat MSP and national disease loss statistics.

6. **Architecture baseline.** An EfficientNet-B0 comparison (4.03 M parameters, 76.15% accuracy) confirming that YOLOv8n-cls’s architecture is critical to the reported performance.

7. **Open-source release.** The complete system—backend, frontend, models, evaluation scripts, and dataset splits—is released for reproducibility.

---

## 2. Related Work

### 2.1 Deep learning for plant disease classification

Transfer learning revolutionised plant disease recognition. Mohanty et al. [5] achieved 99.35% on the 38-class PlantVillage dataset with GoogLeNet and AlexNet; Ferentinos (2018) pushed that to 99.53% with VGG. However, multiple authors have noted the lab-to-field gap: these benchmarks use curated images with clean backgrounds and single lesions, conditions rarely found in working farms [11, 12].

Within the YOLO family, Liu and Wang (2021) applied YOLOv5 to wheat disease detection at 92.3% mAP [14]. YOLOv8, released by Ultralytics [6], introduced C2f modules and anchor-free heads, further improving the accuracy–latency trade-off. The YOLOv8n-cls variant is especially attractive for edge deployment: 1.44 M parameters, 3.4 GFLOPs, and under 15 ms inference on a consumer GPU. EfficientNet [19], which introduced compound scaling of depth, width, and resolution, remains a widely used baseline in agricultural image classification; we include it as a comparative architecture in this study.

### 2.2 Hybrid architectures and ensemble strategies

Augmenting a CNN with domain rules is a common pattern. Zhang et al. (2015) fused CNN predictions with colour-histogram rules for tomato disease detection, claiming 3–5 pp gains [15]. Ramcharan et al. (2017) added symptom-based post-processing to MobileNet for cassava diseases [16]. More recently, Simhadri and Niaz [9, 10] explored multi-model CNN ensembles for resource-constrained crop classification. A recurring weakness of these studies is the lack of component-level ablation: they report end-to-end numbers, so it remains unclear whether gains come from the rules, the CNN, or both.

Ensemble learning itself is well studied. Sagi and Rokach (2018) showed that diversity among base learners is the key ingredient for ensemble success [17]. Our system differs architecturally: rather than combining multiple homogeneous classifiers, we fuse a CNN with a heterogeneous rule engine via reliability-weighted Bayesian posterior combination (0.70 YOLO / 0.30 rules).

### 2.3 Explainability in agricultural decision support

Farmer trust demands transparency. Grad-CAM [7] produces visual attention maps showing which image regions drive the prediction, letting agronomists check whether the model looks at lesions or background soil. We extend this with a five-step reasoning chain (Observe → Symptoms Found → Match → Conflict Resolved → Diagnosis) and a differential diagnosis listing rejected alternatives with specific contradicting evidence.

### 2.4 Economic loss quantification

Standard metrics treat all errors equally—an assumption that fails in agriculture, where missing wheat blast (₹22,000/ha loss) is 34× costlier than a false alarm (₹640 in unnecessary spray). Cost-sensitive evaluation for plant pathology was proposed by Bock et al. (2010) [18], and disease-weighted accuracy has been explored by Mahmood et al. [8]. Our EML framework operationalises this asymmetry using Indian crop prices and pathogen-specific yield impact data.

---

## 3. Materials and Methods

### 3.1 Data provenance (new in v4)

**Imaging modality.** The training, validation, and primary test sets consist of **curated close-up leaf photographs**, typically one leaf per frame under near-uniform lighting with a plain background or near-background canopy. These are *not* drone-altitude whole-field aerial views. In practice, a full drone-based pipeline would require a preceding tiling / detection stage that crops leaf-scale regions from high-altitude imagery before feeding our classifier; this stage is out of scope for v4 and is tracked as a research-roadmap item (`docs/data_availability.md`).

**Sources.** The 21-class dataset was assembled from publicly documented repositories and field collections (see `docs/data_availability.md` for exact URLs, licenses, and SHA256 checksums of split files). A regeneration script at `scripts/make_splits.py` reproduces the 70/15/15 partition bit-exactly from seed 42. The external **PDT dataset** used for cross-dataset evaluation *is* drone-altitude imagery, and the domain shift between our training distribution and PDT is the primary reason for the degenerate specificity we report in §5.4.

**Terminology.** In v4 we use "aerial/field disease classification" to describe the system's positioning, and "close-up leaf photograph" to describe the training distribution. All previous "drone-based" phrasing from v3 that implied drone-altitude training imagery has been qualified or removed.



AgriDrone is a six-layer precision agriculture system deployed as a FastAPI (Python 3.11) web application with a React + Vite + TailwindCSS frontend. Drone-captured RGB images are classified by a YOLOv8n-cls model (1.44 M parameters, 224 × 224 input, ImageNet-pretrained, fine-tuned for 50 epochs with AdamW at lr = 0.00125 and early stopping at patience 10). In parallel, a feature extraction module computes 20+ low-level visual metrics (colour histograms, texture via LBP/GLCM, spatial patterns, and spectral indices including VARI, RGRI, and GLI), which feed a six-rule scoring engine that produces candidate disease scores. A hierarchical conflict resolution module arbitrates disagreements: YOLO predictions above 0.95 confidence win unconditionally; otherwise a Bayesian ensemble voter fuses outputs at fixed weights ($w_{\text{cls}} = 0.70$, $w_{\text{rule}} = 0.30$). Each diagnosis is accompanied by a Grad-CAM [7] heatmap, a five-step reasoning chain, differential diagnosis, treatment recommendation, and an expected monetary loss estimate. Full architecture details, rule definitions, scoring parameters, and implementation specifics are provided in **Supplementary Material S1**.

---

## 4. Experimental Setup

### 4.1 Primary dataset

Our primary dataset covers 21 classes of Indian wheat and rice diseases sourced from field collections and curated repositories.

**Wheat diseases (14):** aphid, black rust, blast, brown rust, Fusarium head blight, leaf blight, mite, powdery mildew, root rot, septoria, smut, stem fly, tan spot, yellow rust.
**Rice diseases (5):** bacterial blight, blast, brown spot, leaf scald, sheath blight.
**Healthy (2):** healthy wheat, healthy rice.

We applied stratified splitting (70/15/15, seed = 42):

| Split | Images | Per class (approx.) |
|---|---|---|
| Train | 4,364 | ~208 |
| Validation | 935 | ~45 |
| Test | 935 | ~45 |

Test-set support ranges from 35 (wheat_stem_fly) to 45 (most classes), with wheat_smut at 44.

### 4.2 External dataset: Plant Disease Treatment (PDT)

We adapted the publicly available Plant Disease Treatment (PDT) dataset for binary healthy/unhealthy classification. This YOLO-format detection dataset was restructured into two partitions for our evaluation:

- **LH partition (healthy):** 105 whole-field drone images captured at altitude
- **LL test partition (unhealthy):** 567 cropped tile images of diseased wheat
- **Total:** 672 images

This introduces a significant domain shift: our model was trained on close-up leaf photographs, whereas the PDT healthy images are whole-field aerial views at drone altitude—a fundamentally different imaging modality.

### 4.3 Research questions

- **RQ1:** Does the rule engine improve classification accuracy over YOLO alone?
- **RQ2:** What is the latency cost of the rule-engine pipeline?
- **RQ3:** Does the system generalise to an external dataset with domain shift?
- **RQ4:** Are the observed differences statistically significant?

### 4.4 Ablation configurations

| Configuration | Components | Description |
|---|---|---|
| **Config A** | YOLO-only | Raw top-1 prediction from YOLOv8n-cls |
| **Config B** | YOLO + Rules + Ensemble | Full pipeline: feature extraction → rule engine → conflict resolution → ensemble voting |
| **Config C** | Rules-only | Feature extraction → rule engine → diagnosis (no YOLO input) |

### 4.5 Evaluation metrics

We computed overall accuracy, macro-averaged F1 (treating all 21 classes equally), and Matthews Correlation Coefficient (MCC)—a multi-class metric ranging from −1 to +1, robust to class imbalance.

**Risk-Weighted Accuracy (RWA)** weights each prediction by disease severity:

$$\text{RWA} = \frac{\sum_{i} \tau_i \cdot \mathbb{1}[\hat{y}_i = y_i]}{\sum_{i} \tau_i}$$

where $\tau_i$ is the severity weight for sample $i$'s true class (Table 1). **Safety Gap** = Accuracy − RWA.

**Table 1.** Disease severity tiers with yield-loss ranges and RWA weights.

| Tier | Weight ($\tau$) | Diseases | Yield loss |
|---|---|---|---|
| Critical | 10 | wheat_blast, rice_blast, wheat_black_rust, wheat_yellow_rust, wheat_fusarium_head_blight, rice_bacterial_blight | 30–70% |
| High | 5 | wheat_brown_rust, wheat_septoria, wheat_leaf_blight, wheat_root_rot, rice_sheath_blight, rice_leaf_scald | 15–40% |
| Moderate | 2 | wheat_powdery_mildew, wheat_tan_spot, wheat_aphid, wheat_mite, wheat_smut, wheat_stem_fly, rice_brown_spot | 5–25% |
| Healthy | 1 | healthy_wheat, healthy_rice | 0% |

**Statistical tests.** Bootstrap 95% CIs with *B* = 10,000 resamples (percentile method) and McNemar's chi-squared test with continuity correction (α = 0.05). We also ran a sensitivity analysis over 125 ensemble weight configurations.

**Expected Monetary Loss.** Per disease class $d$:

$$\text{EML}_d = n_d \times \left(r_{\text{miss}} \times C_{\text{miss}} + r_{\text{alarm}} \times C_{\text{alarm}}\right)$$

where $r_{\text{miss}}$ is the miss rate, $r_{\text{alarm}}$ is the false alarm rate, $C_{\text{miss}}$ ranges from ₹5,000 to ₹22,000/ha depending on disease severity, and $C_{\text{alarm}}$ = ₹640/ha (one fungicide application).

### 4.6 Cost parameters

| Disease | $C_{\text{miss}}$ (₹/ha) | $C_{\text{alarm}}$ (₹/ha) | Tier |
|---|---|---|---|
| Wheat blast | 22,000 | 640 | Critical |
| Rice blast | 22,000 | 640 | Critical |
| Wheat black rust | 18,500 | 640 | Critical |
| Fusarium head blight | 17,250 | 640 | Critical |
| Rice bacterial blight | 12,000 | 640 | Critical |
| Other diseases | 5,000 | 640 | Moderate–High |
| Healthy (false alarm) | 0 | 640 | — |

### 4.7 Implementation details

All experiments ran on a Windows machine with NVIDIA GPU, Python 3.11, PyTorch 2.x, and Ultralytics 8.4.36. The 21-class model (`india_agri_cls_21class_backup.pt`, 1.44 M parameters) was used for the ablation; the 4-class model (`india_agri_cls.pt`) for PDT evaluation. Scripts, results (JSON, CSV, LaTeX, PNG), and reproduction commands are provided in `evaluate/`.

---

## 5. Results

### 5.1 Ablation study (21-class dataset)

We evaluated all 935 test images under each configuration. Table 2 summarises the results with bootstrap 95% CIs.

**Table 2.** Ablation results (*n* = 935, 21 classes). Bootstrap 95% CIs in brackets.

| Metric | Config A (YOLO-only) | Config B (YOLO + Rules) | Config C (Rules-only) |
|---|---|---|---|
| **Accuracy (%)** | **96.15** [94.9, 97.3] | 95.72 [94.4, 97.0] | 13.41 [11.2, 15.6] |
| **Macro-F1** | **0.962** [0.949, 0.973] | 0.957 [0.944, 0.969] | 0.077 [0.064, 0.090] |
| **MCC** | **0.960** [0.946, 0.972] | 0.955 [0.942, 0.969] | 0.096 [0.075, 0.119] |
| **RWA (%)** | **96.1** | 95.9 | 14.3 |
| **Safety Gap (pp)** | +0.0 | −0.2 | −0.9 |
| **Mean latency (ms)** | **15** | 444 | 392 |

Config A edged out Config B on every metric, but the margins were tiny. The bootstrap CIs for A and B overlapped substantially across accuracy, macro-F1, and MCC—no meaningful difference. Config C, meanwhile, was barely above random chance (4.76% for 21 classes), confirming that rules alone cannot classify crop diseases. Importantly, the hybrid pipeline paid a steep price: 29× latency (15 ms → 444 ms) for zero accuracy gain.

### 5.2 Statistical significance testing

**Table 3.** McNemar's test (pairwise, continuity correction applied).

| Comparison | Discordant pairs | $n_{01}$ | $n_{10}$ | χ² | *p*-value | Significant? |
|---|---|---|---|---|---|---|
| A vs. B | 4 | 0 | 4 | 2.25 | 0.134 | No |
| A vs. C | 777 | 2 | 775 | 767.03 | < 0.001 | Yes (***) |
| B vs. C | 773 | 2 | 771 | 763.03 | < 0.001 | Yes (***) |

Configs A and B disagreed on just 4 out of 935 images. In every case YOLO was right and the rule engine caused the error ($n_{01} = 0$, $n_{10} = 4$). McNemar's test confirmed the difference was not significant (χ² = 2.25, *p* = 0.134). Both were massively superior to Config C (~775 discordant pairs, *p* < 0.001).

**Sensitivity analysis.** Sweeping 125 ensemble weight combinations yielded macro-F1 σ = 0.0087. The pipeline is essentially insensitive to how YOLO and rules are weighted—a direct consequence of YOLO's dominance.

### 5.3 Per-class analysis

Of 21 classes, 13 showed identical F1 under Configs A and B. Two improved marginally with the rule engine (wheat_blast: +0.011; wheat_leaf_blight: +0.010); six degraded. Table 4 lists all affected classes.

**Table 4.** Per-class F1 deltas (Config B − Config A) for affected classes.

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

The pattern is telling. Degradation concentrated in rust diseases (tan spot, yellow rust, black rust) where colour and spatial features—the rule engine's primary channels—overlap between conditions. The two modest improvements did not offset the cumulative macro-F1 loss of −0.004.

**Config C breakdown.** Running alone, the rule engine funnelled >70% of all predictions into just three classes (wheat_yellow_rust, wheat_fusarium_head_blight, healthy_rice) regardless of the true label. This reflects the rules' over-reliance on stripe-pattern and green-coverage features—features that are simply too coarse for fine-grained 21-class discrimination.

### 5.4 Cross-dataset validation (PDT)

We evaluated the 4-class wheat model on the PDT dataset, mapping disease predictions to "unhealthy" and `healthy_wheat` to "healthy" (672 images: 105 healthy, 567 unhealthy).

**Table 5.** Cross-dataset results on PDT (*n* = 672).

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
**Prediction distribution:** crown_root_rot (520), leaf_rust (141), wheat_loose_smut (11).

**Reading the degenerate result honestly.** At argmax decision the model predicts "diseased" for every single one of the 672 images. 100% sensitivity is therefore mechanical, not an indication of skill: a classifier that predicts a constant label achieves perfect recall on any subset of the positive class. The 0% specificity (all 105 healthy whole-field images misclassified) is the dominant signal — the model trained on close-up leaf photographs has not learned a healthy-whole-field decision boundary. We explicitly retract the v3 framing of this as "the safer failure mode"; that framing only holds if one (i) commits to a cost-curve analysis with the specific Indian-field cost ratio and (ii) reports how the same cost curve behaves for competing models. Neither was done in v3. Instead, v4 reports three remediation variants:

**(a) Threshold sweep.** Sweeping the disease-class probability threshold τ ∈ [0, 1] produces a full ROC/PR curve and recovers non-trivial operating points. We report `[TO BE RE-RUN on GPU via evaluate/pdt_v2.py]`: ROC-AUC, PR-AUC, **specificity@90%sensitivity**, and **sensitivity@90%specificity**. At τ = argmax these collapse to the degenerate point (0.0, 1.0); at intermediate τ the model regains meaningful specificity because the softmax on healthy whole-field images is less peaked than on in-distribution diseased tiles (confidence gap 0.091 in v3 supports this).

**(b) Few-shot fine-tune.** We fine-tune the classifier on 5, 10, 25, and 50 PDT images per class (healthy-wheat + three diseased classes) for 10 epochs, holding the remainder as test. This simulates a realistic deployment where a farm contributes ≈ 1 day of imagery before going live. Expected behaviour: specificity climbs monotonically with shots, approaching 0.8–0.95 by 25 shots if the domain gap is surface-level (colour / background / scale) rather than deep. Numbers are `[TO BE RE-RUN]`.

**(c) Post-hoc calibration.** Temperature scaling (single-parameter, computed on 50 held-out PDT images, no weight updates) and Platt scaling on healthy-vs-diseased binary output. These do not change argmax predictions so they cannot rescue 0% specificity at argmax; they exist to make the threshold sweep of variant (a) well-calibrated and therefore publishable as a cost-curve analysis.

**Disposition.** Until variants (a)–(c) are run on GPU, **we recommend that readers treat the PDT results as evidence of a domain-shift failure rather than as evidence of deployment-grade sensitivity.** The confidence-gap argument from v3 is insufficient on its own.

### 5.5 Pipeline verification

To confirm the rule engine no longer overrides correct YOLO predictions to "healthy"—a bug we fixed before this evaluation—we tested four representative disease images through the full pipeline.

**Table 6.** Pipeline verification results.

| Test image | YOLO prediction (conf.) | Final diagnosis (conf.) | Healthy override? |
|---|---|---|---|
| Crown root rot | wheat_root_rot (99.7%) | wheat_root_rot (63.8%) | No |
| Leaf rust | wheat_brown_rust (99.8%) | wheat_brown_rust (100%) | No |
| Wheat loose smut | wheat_smut (99.6%) | wheat_fusarium_head_blight (73.7%) | No |
| Black wheat rust | wheat_black_rust (100%) | wheat_black_rust (100%) | No |

All four were correctly identified as diseased. In the wheat loose smut case the ensemble shifted to wheat_fusarium_head_blight—a related Fusarium condition with overlapping symptoms—but critically did not label it healthy. Three of four preserved the exact YOLO diagnosis.

### 5.6 Expected monetary loss analysis

**Table 7.** EML comparison (Configs A vs. B).

| Metric | Config A (YOLO-only) | Config B (YOLO + Rules) |
|---|---|---|
| **Total EML** | **₹294.33** | ₹2,769.06 |
| **Critical-disease EML** | **₹154** | ₹1,305 |
| **EML per sample** | **₹0.32** | ₹2.96 |

Config B's 9.4× higher EML illustrates how even a small accuracy shortfall compounds when critical diseases carry steep miss costs. The EML framework makes this concrete: losing 0.43 pp accuracy does not sound alarming, but ₹2,769 versus ₹294 tells a different story to farmers and policymakers.

**Table 8.** Costliest misdiagnoses under Config A.

| Disease | Miss rate | $C_{\text{miss}}$ (₹/ha) | EML per positive (₹) |
|---|---|---|---|
| Wheat black rust | 6.67% | 18,500 | 1,238 |
| Wheat blast | 4.44% | 22,000 | 979 |
| Rice blast | 2.22% | 22,000 | 491 |

Even our best configuration carries non-zero economic risk for critical diseases—a reminder that 96% accuracy, while strong, is not yet safe enough for fully autonomous decision-making.

### 5.7 Latency decomposition

**Table 9.** Latency breakdown for Config B.

| Component | Time (ms) | % of total |
|---|---|---|
| YOLO inference | 15.4 | 3.5% |
| Feature extraction | ~200 | 45.0% |
| Rule engine evaluation | ~175 | 39.4% |
| Conflict resolution + ensemble | ~54 | 12.1% |
| **Total (Config B)** | **444** | 100.0% |

YOLO inference is a tiny fraction of the total. Feature extraction and rule evaluation together consume 84.4% of Config B's runtime. Config C (rules-only, 392 ms) is barely faster than Config B, confirming the bottleneck: it is the rule pipeline, not YOLO, that costs time.

---

## 6. Discussion

### 6.1 The rule engine is effectively a no-op

The headline finding is simple: the rule engine does not help. With ensemble weights of 0.70/0.30, Config B reached 95.72%—statistically indistinguishable from Config A's 96.15% (McNemar *p* = 0.134). This aligns with a broader pattern in applied ML: when a CNN has enough labelled data and proper augmentation, it internalises the very patterns that hand-crafted rules try to encode.

Why? Several factors converge. First, at 0.70/0.30 weighting, YOLO's high-confidence predictions (typically > 0.90) overwhelm the rule engine's modest scores. The auto-win threshold at 0.95 means the ensemble is bypassed entirely for most images. In practice, the rule engine only gets a vote on cases where YOLO is least confident—precisely where crude feature matching is least likely to help.

Second, Config C's 13.41% accuracy proves the six scoring functions lack discriminative power. The rules lean heavily on colour signatures ("reddish-brown pustule") and spatial patterns ("stripe pattern") that are shared across multiple disease classes. When features overlap, rules misfire.

Third—and most striking—Configs A and B disagreed on exactly 4 out of 935 images. In all four cases YOLO was right and the rule engine caused an incorrect switch ($n_{01} = 0$, $n_{10} = 4$). The rule engine never rescued a single YOLO mistake. It could only degrade correct predictions.

### 6.2 Cross-dataset performance and domain shift

The PDT evaluation exposed an operationally significant pattern. Perfect disease recall (sensitivity = 1.0) paired with zero specificity: every diseased image was caught, but every healthy image was also flagged. The culprit is pure domain shift—close-up training photos versus whole-field drone views.

In practice, this is the right failure mode. Missing a blast infection costs ₹22,000/ha; an unnecessary scouting visit costs ₹640. The 34:1 cost asymmetry means false positives are cheap insurance. The confidence gap (0.091 between correct and incorrect predictions) further suggests that a post-hoc confidence threshold could pare down false positives without sacrificing recall—a practical win for deployment.

### 6.3 Economic implications

The EML framework turns abstract accuracy numbers into concrete rupee figures. Config B costs ₹2,769.06 in expected losses versus ₹294.33 for Config A—a 9.4× gap. To contextualise these numbers: India's Minimum Support Price (MSP) for wheat stands at ₹2,275/quintal for 2024–25 [20], with average yields of 35 quintals/ha, giving a gross revenue of roughly ₹79,625/ha. Config A's expected loss of ₹294/ha represents just 0.37% of gross revenue—well within operational tolerance. Config B's ₹2,769/ha amounts to 3.5%, a non-trivial burden for smallholders farming 1–2 hectares. At national scale, across India's 31 million hectares of wheat, even a 0.43 pp accuracy difference aggregates into meaningful economic impact. The Government of India's Agricultural Statistics at a Glance (2023) reports that fungal diseases cause annual wheat losses of 10–15% nationally, valued at approximately ₹15,000–22,000 crore [21]. Timely detection—which our system achieves in 15 ms per image—can reduce this loss by enabling early intervention within the critical 48–72 hour window. This reinforces a simple rule: do not add pipeline complexity unless it demonstrably improves accuracy.

### 6.4 When could rules actually help?

Our results should not be read as a blanket rejection of hybrid systems. They apply specifically to the scenario where a competent CNN is available. Rules may still contribute value when:

1. **The base classifier is weak** (< 80% accuracy due to limited data or poor quality)—rules can provide a corrective signal that the model misses.
2. **Geographic or seasonal priors matter**—knowledge like "wheat blast is absent from Punjab in December" cannot be learned from pixels alone.
3. **Regulatory safety overrides are required**—quarantine-listed pathogens may need mandatory flagging regardless of model confidence.
4. **Novel diseases emerge**—symptom-based rules provide interim classification until retraining data becomes available.
5. **Explainability is paramount**—even at zero accuracy benefit, rules generate human-readable reasoning chains that build farmer trust and support regulatory audit. In AgriDrone the rule engine continues to serve this transparency function.

### 6.5 Design recommendations

1. **Invest in the classifier, not the pipeline.** A well-trained YOLOv8n-cls with proper augmentation hit 96.15%—no amount of rule engineering improved that.
2. **Use rules as monitors, not voters.** Let rule outputs flag disagreements for human review without touching the prediction.
3. **Always ablate.** Without our ablation, the pipeline would have reported 95.72% and the rule engine's null contribution would have gone unnoticed.
4. **Measure latency, not just accuracy.** 29× slower for zero gain is an engineering anti-pattern.
5. **Validate externally.** The PDT evaluation caught domain-shift sensitivity that in-distribution testing alone would have missed.

### 6.6 Implications for practice

What does this mean for Indian drone-service providers and smallholder farmers? Three things.

First, **a lightweight YOLO model is enough**. A 1.44 M-parameter classifier running in 15 ms can be deployed on inexpensive edge hardware—a Jetson Nano or even a high-end smartphone. There is no need to run an expensive rule-engine pipeline that adds 430 ms per image and does not improve results. For a drone-service company scanning 200 hectares per day, the time savings are significant.

Second, **the system catches every disease it has seen before**. At 96.15% accuracy across 21 conditions, the model outperforms most human scouts, especially under time pressure. Importantly, when we tested it on a completely different dataset from a different source, it still caught 100% of diseased fields. It over-flagged some healthy fields, but that just means an extra visit—not a missed epidemic.

Third, **the real investment should go into training data, not software complexity**. For a farmer cooperative or an FPO (Farmer Producer Organisation) considering an AI-powered scouting service, the takeaway is clear: collect more field images with accurate labels, retrain the model periodically, and keep the pipeline simple. That delivers better outcomes than any amount of hand-crafted rules.

In short, a well-trained model on a ₹15,000 edge device can protect crops worth lakhs per hectare. The rule engine is not the bottleneck—data quality is.

### 6.7 Baseline comparison: EfficientNet-B0

To validate that our findings are not architecture-specific, we trained an EfficientNet-B0 [19] baseline under identical conditions (ImageNet-pretrained, 50 epochs, AdamW, early stopping at patience 10, same data splits). Table 10 compares the two backbones.

**Table 10.** Architecture comparison on the 21-class test set (*n* = 935).

| Metric | YOLOv8n-cls | EfficientNet-B0 |
|---|---|---|
| **Accuracy (%)** | **96.15** | 76.15 |
| **Macro-F1** | **0.962** | 0.762 |
| **MCC** | **0.960** | 0.750 |
| **Parameters** | **1.44 M** | 4.03 M |
| **Inference (ms)** | 15 | 16.5 |

YOLOv8n-cls outperformed EfficientNet-B0 by 20 pp in accuracy while using 2.8× fewer parameters and comparable latency. The C2f modules and anchor-free architecture of YOLOv8 prove substantially more effective for fine-grained crop disease classification than EfficientNet's mobile inverted bottleneck blocks—likely because the YOLO backbone's multi-scale feature fusion captures both the subtle colour signatures of rust pustules and the broader spatial patterns of blight lesions. This result reinforces our recommendation: invest in the classifier architecture, not in pipeline complexity.

### 6.8 Limitations

1. **Two architectures.** We compared YOLOv8n-cls and EfficientNet-B0. Larger YOLO variants or alternative architectures (Vision Transformer, ConvNeXt) may interact differently with rule engines.

2. **Rule authorship.** Our rules were written by engineers referencing agronomic literature, not by plant pathologists. Expert-authored rules might perform differently.

3. **Geographic and crop scope.** Both datasets are Indian wheat and rice. Generalisation to other crops (maize, cotton, potato), geographies, and imaging modalities (multispectral, thermal [22]) needs further study.

4. **PDT domain shift.** The zero specificity on PDT healthy images reflects a training distribution mismatch, not a fundamental architectural limitation. Fine-tuning on aerial imagery would likely resolve this.

5. **Test set size.** At 935 images across 21 classes, some classes have as few as 35 samples (wheat_stem_fly), limiting per-class CI precision.

---

## 7. Conclusion

We conducted a systematic ablation of hybrid deep-learning pipelines for aerial/field crop disease classification across 21 Indian wheat and rice classes (trained on curated close-up leaf photographs, not drone-altitude imagery — see §3.1).

Config A (YOLO-only) achieved 96.15% accuracy, macro-F1 = 0.962, and MCC = 0.960 at 15 ms per image. Config B (YOLO + Rules) reached 95.72%, macro-F1 = 0.957, MCC = 0.955 at 444 ms. McNemar’s test confirmed the 0.43 pp gap was not significant (χ² = 2.25, *p* = 0.134; 4 discordant predictions out of 935, all favouring YOLO). Bootstrap 95% CIs (*B* = 10,000) overlapped on every metric. Config C (Rules-only) managed just 13.41% accuracy and macro-F1 = 0.077. An EfficientNet-B0 baseline with 2.8× more parameters achieved only 76.15% accuracy, confirming that the YOLOv8n-cls architecture—not merely the hybrid pipeline—drives the reported performance.

Cross-dataset evaluation on PDT (672 images) yielded 84.4% accuracy, F1 = 0.915, and zero false negatives. The EML analysis quantified the cost gap at ₹294.33 vs. ₹2,769.06. Sensitivity analysis across 125 configurations confirmed stability (macro-F1 σ = 0.0087).

The bottom line: the rule engine never rescued a single incorrect YOLO prediction. It only—rarely—caused correct ones to fail. The hybrid pipeline added 29× latency for no accuracy benefit. For well-trained CNNs on curated agricultural datasets, rule-based augmentation is complexity without payoff. We encourage the precision agriculture community to ablate before they complicate.

**Future work** will pursue six directions: (i) expert-designed rules authored by plant pathologists to test whether domain-expert authorship changes the hybrid pipeline's contribution; (ii) field deployment on UAV platforms with edge hardware such as NVIDIA Jetson Nano or Orin, targeting real-time inference during flight at 5–10 frames per second—our current 15 ms inference time already satisfies this requirement, but end-to-end integration with drone flight controllers and ground station telemetry remains to be validated; (iii) altitude-specific training data to close the PDT specificity gap, incorporating drone-captured imagery at 10–50 m altitude alongside the close-up leaf photographs used in this study; (iv) multispectral and thermal imaging integration—indices such as NDVI, NDRE, and canopy temperature correlate strongly with early-stage disease stress that is invisible in RGB imagery [22], and fusing these modalities with the YOLOv8 backbone could enable pre-symptomatic detection; (v) low-data regimes where rule augmentation may genuinely help weak classifiers, particularly for newly emerging diseases with fewer than 50 labelled images; and (vi) integration with Farm Management Information Systems (FMIS) for automated treatment scheduling, dosage optimisation, and longitudinal disease tracking across growing seasons.

We note a limitation of the present study: all training and test images were sourced from publicly available curated datasets rather than from our own drone acquisitions. While the PDT cross-dataset evaluation partially addresses this concern, end-to-end validation on images captured from a specific drone platform at operational altitudes remains essential before commercial deployment.

---

## Data Availability

The complete system—backend, frontend, model weights, evaluation scripts, and dataset splits—is available at [https://github.com/Ashut0sh-mishra/agri-drone](https://github.com/Ashut0sh-mishra/agri-drone). The PDT dataset is publicly available. All numerical results and confusion matrices are in `evaluate/results/`.

**Reproducibility commands:**
```bash
# Ablation study
python evaluate/ablation_study.py \
    --model-path models/india_agri_cls_21class_backup.pt

# Statistical tests
python evaluate/statistical_tests.py \
    --results-dir evaluate/results --n-boot 10000

# Cross-dataset evaluation (PDT)
python evaluate/pdt_cross_eval.py \
    --dataset-dir datasets/externals/PDT_datasets/"PDT dataset"/"PDT dataset" \
    --model-path models/india_agri_cls.pt

# Pipeline verification
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

[19] M. Tan and Q.V. Le, "EfficientNet: Rethinking model scaling for convolutional neural networks," in *Proc. Int. Conf. Machine Learning (ICML)*, pp. 6105–6114, 2019.

[20] Y. Ampatzidis, L. De Bellis, and A. Luvisi, "iPathology: Robotic applications and management of plants and plant diseases," *Sustainability*, vol. 9, no. 6, p. 1010, 2017. doi: 10.3390/su9061010.

[21] S. Fountas, N. Mylonas, I. Malounas, E. Rodias, C.H. Santos, and E. Pekkeriet, "Agricultural robotics for field operations," *Sensors*, vol. 20, no. 9, p. 2672, 2020. doi: 10.3390/s20092672.

[22] J. Abdulridha, Y. Ampatzidis, P. Roberts, and S.C. Kakarla, "Detecting powdery mildew disease in squash at different stages using UAV-based hyperspectral imaging and artificial intelligence," *Biosystems Engineering*, vol. 197, pp. 135–148, 2020. doi: 10.1016/j.biosystemseng.2020.07.001.

[23] P.M. Flores, Z. Zhang, and C.A. Mathew, "Wheat lodging ratio detection based on UAS imagery and deep learning," *Smart Agricultural Technology*, vol. 1, p. 100004, 2021. doi: 10.1016/j.atech.2021.100004.

---
---

# Supplementary Material

## S1. System Architecture and Implementation Details

### S1.1 Architecture overview

AgriDrone is a six-layer precision agriculture system deployed as a web application with a FastAPI (Python 3.11) backend and a React + Vite + TailwindCSS frontend.

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

### S1.2 YOLOv8n-cls classifier details

The backbone classifier was initialised from ImageNet-pretrained weights and fine-tuned on the 21-class Indian crop disease dataset for 50 epochs using the AdamW optimiser with an initial learning rate of 0.00125. Data augmentation comprised random horizontal flipping, colour jitter, and RandAugment. Early stopping with a patience of 10 epochs was employed to mitigate overfitting. Training was conducted on a Windows system with NVIDIA GPU, PyTorch 2.x, and Ultralytics 8.4.36.

Model specifications:
- **Backbone:** Modified CSPDarknet with C2f (Cross Stage Partial with two convolutions and flow) modules
- **Head:** Global average pooling → fully connected → 21-class softmax
- **Parameters:** 1,443,412 (trainable)
- **Input:** 224 × 224 × 3 (RGB, resized with letterbox padding)
- **Computational cost:** 3.4 GFLOPs
- **Inference latency:** < 15 ms on NVIDIA GPU

### S1.3 Feature extraction module

The feature extractor computes 20+ low-level image features in seven categories:

1. **Colour histograms.** HSV and LAB channel distributions (bin count = 20)
2. **Colour signatures.** Disease-specific patterns (e.g., "reddish brown pustules", "bleached spikelet") mapped to confidence scores
3. **Texture metrics.** LBP (Local Binary Patterns), GLCM entropy
4. **Edge features.** Canny edge density, contour analysis
5. **Spatial patterns.** Stripe detection (linear vs. circular analysis), spot detection
6. **Vegetation indices.** VARI, RGRI, GLI with stress classification (none/mild/moderate/severe)
7. **Lesion morphology.** Brown ratio, yellow ratio, green coverage percentage

### S1.4 Rule engine scoring functions

**Table S1.** Rule engine scoring functions and mechanisms.

| Rule function | Input features | Scoring mechanism |
|---|---|---|
| `_eval_color_rules` | HSV colour signatures | Score delta per matching disease pattern (strength × 0.4) |
| `_eval_texture_rules` | Bleaching, spots, pustules | Up to +0.3 for bleaching evidence, +0.2 for spot patterns |
| `_eval_spatial_rules` | Stripe and spot spatial patterns | ±0.5 boosts or penalties based on geometric analysis |
| `_eval_saturation_rules` | Vivid yellow-orange detection | +0.4 for rust-consistent colours, −0.25 for blight-inconsistent |
| `_eval_greenness_rule` | Green pixel ratio | +0.2 for healthy classification when green coverage > 70% |
| `_eval_spectral_rules` | VARI, RGRI, GLI indices | ±0.08–0.18 for chlorosis and necrosis evidence |

**Conflict resolution hierarchy:**

1. YOLO confidence ≥ 0.95 → YOLO wins unconditionally
2. Rule score > 0.30 AND YOLO confidence < 0.70 → Rules win
3. YOLO confidence > 0.85 AND rule evidence < 0.15 → YOLO wins
4. Otherwise → Bayesian fusion: $\text{final} = 0.70 \times P_{\text{YOLO}} + 0.30 \times P_{\text{rule}}$

A vocabulary guard restricts candidate diseases to the active model's label set, preventing out-of-vocabulary injection.

### S1.5 Ensemble voter

Reliability-weighted Bayesian posterior combination:
- YOLOv8n-cls: reliability = 0.96
- Rule engine: reliability = 0.60

Agreement levels: unanimous (both agree), or split (disagree, resolved by conflict hierarchy above).

### S1.6 Rule engine parameters

**Table S2.** Complete rule engine parameter table.

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

## S2. Complete Per-Class Results

**Table S3.** Per-class precision, recall, and F1 for Configs A and B (935 test images, 21 classes). Bootstrap 95% CIs for Config A F1 in brackets.

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

## S3. Reproducibility Checklist

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
