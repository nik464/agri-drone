# A Negative Result on Hand-Authored Symbolic Rules for CNN-Based Crop-Disease Classification

**Ashutosh Mishra**
Independent Researcher, India

---

## Abstract

Hybrid neuro-symbolic systems — a CNN classifier audited by a hand-authored symbolic rule engine — are a recurring proposal in precision-agriculture literature, typically justified by appeals to interpretability and farmer trust. We test this proposal on 934 stratified held-out smartphone images covering 21 rice- and wheat-disease classes and report a **negative result**. A standalone YOLOv8n-cls backbone (Config A) achieves 96.15% accuracy, macro-F1 0.9618, MCC 0.9596 (95% bootstrap CI [0.9486, 0.9732]) at 15.4 ms/image on a single T4 GPU. Adding an agronomist-informed rule layer (Config B) changes only **4 of 934 predictions** — all four in the wrong direction — yielding a paired McNemar p = 0.134 (not significant), a 29× latency tax (444 ms), and an 840% increase in expected monetary loss under a cost matrix calibrated from ICAR/DAE advisory tables. A rules-only ablation (Config C) collapses to 13.4% accuracy, confirming the rule engine is non-trivial — the issue is informational redundancy with the CNN. For clean-to-moderately-noisy smallholder imagery at current CNN accuracy ceilings, hand-authored symbolic rules add cost without adding recognition or cost-adjusted safety.

**Keywords:** negative result, crop-disease classification, neuro-symbolic ablation, YOLOv8, expected monetary loss, smallholder agriculture

---

## 1. Introduction

Precision agriculture increasingly deploys deep-learning classifiers for automated crop-disease triage from smartphone or drone imagery. A persistent theme in the literature is the proposal to augment CNN predictions with hand-authored symbolic rules — typically encoding agronomic knowledge about disease symptoms (color signatures, lesion morphology, spatial patterns) — to improve accuracy, interpretability, or farmer trust.

The implicit assumption is that symbolic domain knowledge adds value on top of what the CNN learns from data alone. We test this assumption directly with a three-configuration ablation:

- **Config A:** YOLOv8n-cls only (softmax argmax)
- **Config B:** YOLOv8n-cls + hand-authored rule engine (weighted re-ranking)
- **Config C:** Rule engine only (no CNN)

Our headline finding is negative. On a 934-image, 21-class held-out test set:

1. Config A achieves 96.15% accuracy with 10,000-iteration bootstrap 95% CIs of [0.9486, 0.9732].
2. Config B changes only 4 of 934 predictions, all adversely (McNemar p = 0.134).
3. Config B inflates expected monetary loss by 840% under an Indian smallholder cost matrix.
4. Config C achieves 13.4% accuracy — the rules are genuinely non-trivial (10× random chance on 21 classes), but informationally redundant with the CNN.

We believe this negative result is worth stating plainly. The community would benefit from seeing that, at the accuracy ceilings modern lightweight CNNs now reach on curated leaf-photo datasets, the neuro-symbolic hybrid pattern does not pay its rent.

**Contributions:**
1. A three-configuration ablation with paired statistical tests showing rules do not help a strong CNN on this task.
2. An EML cost analysis showing the hybrid pattern actively harms cost-adjusted outcomes.
3. An open-source substantive rule engine that can serve as a non-trivial baseline for future neuro-symbolic comparisons.
4. Reproducible evaluation scripts and committed prediction artefacts for independent verification.

---

## 2. Related Work

Modern crop-disease classification has advanced rapidly. Mohanty et al. (2016) demonstrated > 99% accuracy on PlantVillage lab imagery, but field performance typically degrades to ~30% (Barbedo, 2018). Hybrid approaches combining CNNs with domain-specific rules have been proposed repeatedly (e.g., Ferentinos 2018; Ramcharan 2017), but head-to-head ablations quantifying the marginal value of the rule layer are rare.

The key gap is empirical: most neuro-symbolic crop-disease papers report only the hybrid's performance, not a paired comparison against the CNN alone with statistical testing. We address this gap directly.

---

## 3. Data

### 3.1 Training Corpus

Training images were sourced from publicly available datasets including PlantVillage, Kaggle community datasets for Bangladesh rice leaf and Indian wheat leaf diseases, covering 21 classes (2 healthy + 5 rice disease + 14 wheat disease/pest). See `docs/data_availability.md` for exact dataset provenance and download instructions.

### 3.2 Held-out Evaluation Set

The test set comprises **934 images** across 21 classes, split deterministically at seed 42 with a 70/15/15 train/val/test ratio using `scripts/make_splits.py`. Class sizes range from 20 to 45 images per class.

**Augmentation leakage check:** 21.2% of test images carry `aug_*` filenames. We verified that every augmented file has a unique base ID — no sibling augmentations from the same source image exist across splits. See `LEAKAGE_INVESTIGATION_REPORT.md` for the full analysis.

**Note:** `ablation_summary.json` reports `n_test_images: 935`. The actual manifest contains 934 unique test images; one image (`smut_837.png`) failed during model inference and is absent from prediction CSVs. All results in this paper are computed from the 933 successfully predicted images (934 in manifest, 933 in prediction CSVs).

---

## 4. Method

### 4.1 Three-Configuration Ablation

We evaluate three configurations on the identical held-out test set:

| Config | CNN | Rules | Decision |
|---|---|---|---|
| A | YOLOv8n-cls (1.4M params) | Off | Softmax argmax |
| B | YOLOv8n-cls | On | Weighted re-ranking (α_cnn=0.70, α_rules=0.30 with evidence; 0.85/0.15 without) |
| C | Off | On | Rule scores only |

All three configs are evaluated on the same images in the same order.

### 4.2 Rule Engine

The rule engine (`src/agridrone/vision/rule_engine.py`) is a substantive multi-stage system, not a toy:

- **Color rules:** HSV-signature matching against per-disease profiles loaded from `src/agridrone/knowledge/diseases.json`
- **Texture rules:** Bleaching-ratio detection, spot/pustule scoring conditioned on disease symptom profiles
- **Spatial rules:** Stripe-vs-spot penalization using Hough-line directionality; stripe-contradicts-head-disease penalties
- **Conflict resolution:** `CandidateScore` aggregation with `ConflictReport` when CNN top-1 ≠ rule top-1

This is a deliberate design choice: the negative result carries more weight because the rule engine is non-trivial. Config C's 13.4% accuracy (10× random on 21 classes) confirms the rules encode genuine agronomic signal.

### 4.3 Expected Monetary Loss (EML)

We compute per-image expected monetary loss using cost parameters from ICAR and DAE advisory tables:

- `miss_cost = 12,000` INR/ha for critical diseases (blast, rust, FHB, bacterial blight)
- `miss_cost = 0` for healthy classes
- `alarm_cost = 640` INR/ha (one wasted spray round)

See `configs/economics/india_2025.yaml` for the full cost table.

### 4.4 Statistical Testing

- **Bootstrap:** 10,000-replicate percentile CIs for accuracy, macro-F1, and MCC per config
- **McNemar:** Continuity-corrected paired test at α = 0.05 on image-level correct/incorrect vectors (A vs B, A vs C, B vs C)

---

## 5. Results

### 5.1 Headline Metrics

| Config | Accuracy | 95% CI | Macro-F1 | MCC | Latency (ms) |
|---|---|---|---|---|---|
| A (YOLO) | **0.9615** | [0.9486, 0.9732] | 0.9618 | 0.9596 | 15.4 |
| B (YOLO+rules) | 0.9572 | [0.9443, 0.9700] | 0.9574 | 0.9551 | 444.4 |
| C (rules only) | 0.1338 | [0.1124, 0.1563] | 0.0771 | 0.0962 | 392.3 |

The gap between A and B is −0.43 percentage points. SE ≈ 0.006; the CIs overlap substantially.

### 5.2 Expected Monetary Loss

| Config | Total EML (₹/ha) | Critical-disease EML (₹/ha) | Δ vs A |
|---|---|---|---|
| A | 294.33 | 154.32 | — |
| B | 2,769.06 | 1,305.36 | +840.8% |

The rule layer inflates EML by 840%, driven entirely by the 4 corrupted predictions falling on high-cost disease classes.

### 5.3 Override Analysis

The rule layer changes the CNN's top-1 prediction on **7 of 934 images**:

| Category | Count | Net effect |
|---|---|---|
| Rescues (B corrects A's error) | 0 | 0 |
| Corruptions (B breaks A's correct answer) | 4 | −4 |
| Neutral (both wrong anyway) | 3 | 0 |
| **Total overrides** | **7** | **−4** |

Zero rescues from 7 overrides. The rule engine fires on images where the CNN is already confident and correct, and introduces errors.

### 5.4 McNemar Test

| Pair | n_discordant | χ² | p-value | Significant? |
|---|---|---|---|---|
| A vs B | 4 | 2.25 | 0.134 | No |
| A vs C | 777 | 767.03 | < 10⁻³⁰⁰ | Yes |
| B vs C | 773 | 763.03 | < 10⁻³⁰⁰ | Yes |

The A-vs-B comparison is underpowered with only 4 discordant pairs. A dataset of n ∈ [2,000–5,000] would be needed to detect effects of this magnitude at α = 0.05.

### 5.5 Per-Class F1

The A→B F1 deltas per class are small and dominated by noise. The largest losses:
- `wheat_tan_spot`: −0.038
- `wheat_yellow_rust`: −0.022
- `wheat_black_rust`: −0.019

No per-class A→B gain survives its 95% bootstrap CI. See `evaluate/results/` for the full per-class bootstrap CI files.

---

## 6. Threats to Validity

1. **Small test set.** n = 934 with 20–45 images per class limits detectable effect size to ≥ ~0.6 pp aggregate and ≥ ~0.06 F1 per class at α = 0.05. Smaller real effects would be invisible.

2. **Single backbone.** All results use YOLOv8n-cls (1.4M params). The finding may not transfer to weaker backbones where the CNN leaves more room for rules to add value. A multi-backbone comparison (EfficientNet-B0, MobileNetV3, ConvNeXt-Tiny) is planned but not yet executed.

3. **Lab-quality imagery.** The evaluation set consists of smartphone leaf photographs under controlled conditions, not field-deployed drone imagery. Performance under real deployment noise is untested.

4. **Single seed.** Training used seed 42 only. Multi-seed ablation is planned via `configs/matrix/paper2.yaml` (45 cells, 5 backbones × 3 geographies × 3 folds) but has not been executed at time of writing.

5. **No multiple-comparison correction.** The 21 per-class F1 comparisons in §5.5 are not formally corrected (e.g., Holm-Bonferroni). We rely on bootstrap CIs to convey uncertainty, but acknowledge this as a limitation.

6. **Split script.** `scripts/make_splits.py` does not group-split by base image ID. For the current dataset this is safe (each `aug_*` file has a unique base; see `LEAKAGE_INVESTIGATION_REPORT.md`), but the script would introduce leakage on datasets with multi-variant augmentations.

7. **Cost parameters.** EML depends on the chosen cost matrix. Different cost assumptions would change the 840% headline. The cost table is committed at `configs/economics/india_2025.yaml` for transparency.

---

## 7. Discussion

At 96% accuracy, the CNN has saturated to the point where the rule engine's agronomic signal is informationally redundant. The 4 corrupted predictions are on images where the CNN is already correct with high confidence (mean top-1 softmax > 0.87), and the rule engine's re-ranking pushes toward a different (wrong) class.

This is consistent with a broader pattern in applied AI: symbolic post-processing adds value when the base model is weak, but becomes harmful when the base model is strong. The crossover point — below which rules help — is the interesting open question for future work.

The practical implication for smallholder agriculture is clear: at current CNN accuracy levels on curated leaf-photo datasets, engineering effort is better spent on data quality, model calibration, and deployment infrastructure than on hand-authored symbolic rules.

---

## 8. Limitations and Future Work

This paper establishes a single datapoint: rules do not help YOLOv8n-cls on this specific 21-class, 934-image test set. To generalize the finding, we need:

1. **Multi-backbone ablation** across weaker backbones (EfficientNet-B0, MobileNetV3) where the CNN may leave more headroom for rules. Infrastructure exists (`evaluate/matrix/run_matrix.py`, `configs/matrix/paper2.yaml`); execution requires GPU time.

2. **Larger test set** (n ≥ 2,000–5,000) to achieve adequate power for small effect sizes.

3. **Field-condition imagery** — the current evaluation uses lab-quality photos, not real drone/field images.

4. **Learned rules** — `src/agridrone/vision/rules_learned.py` and `rules_llm.py` implement alternative rule generation strategies that have not been evaluated against the test set.

---

## 9. Reproducibility

All results in this paper are derivable from committed artefacts:

| Artefact | Path |
|---|---|
| Predictions A/B/C | `evaluate/results/predictions_{A,B,C}_*.csv` |
| Ablation summary | `evaluate/results/ablation_summary.json` |
| Bootstrap CIs + McNemar | `evaluate/results/statistical_tests.json` |
| Override decomposition | `evaluate/results/override_decomposition.json` |
| EML analysis | `evaluate/results/eml_summary.json` |
| Cost table | `configs/economics/india_2025.yaml` |
| Validation script | `tools/research_validation/validate_ablation.py` |
| Leakage investigation | `LEAKAGE_INVESTIGATION_REPORT.md` |
| Split manifest | `evaluate/data_split_manifest.json` |

Training was performed on a free-tier T4 GPU via Google Colab. The model checkpoint is at `models/yolov8n-cls.pt`.

---

## 10. Conclusion

On 934 stratified smartphone images across 21 rice- and wheat-disease classes, a hand-authored symbolic rule layer bolted onto a 96%-accurate YOLOv8n-cls backbone changes 4 predictions in the wrong direction, is not significant under paired McNemar (p = 0.134), imposes a 29× latency tax, and inflates expected monetary loss by 840% under a smallholder-calibrated cost matrix. A rules-only arm confirms the rules are genuinely non-trivial (13.4% accuracy, 10× random chance); the issue is informational redundancy with the CNN. At the accuracy ceilings modern lightweight CNNs reach on curated leaf-photo datasets, the neuro-symbolic hybrid pattern does not pay its rent.

---

## References

- Barbedo, J. G. A. (2018). Impact of dataset size and variety on the effectiveness of deep learning and transfer learning for plant disease classification. *Computers and Electronics in Agriculture*, 153, 46–53.
- Ferentinos, K. P. (2018). Deep learning models for plant disease detection and diagnosis. *Computers and Electronics in Agriculture*, 145, 311–318.
- Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). Using deep learning for image-based plant disease detection. *Frontiers in Plant Science*, 7, 1419.
- Ramcharan, A., Baranowski, K., McCloskey, P., Ahmed, B., Legg, J., & Hughes, D. P. (2017). Deep learning for image-based cassava disease detection. *Frontiers in Plant Science*, 8, 1852.
