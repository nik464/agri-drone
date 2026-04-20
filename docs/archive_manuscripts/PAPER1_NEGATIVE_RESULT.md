# When Do Expert Rules Help Deep Classifiers? A Negative Result on 21-Class Crop-Disease Triage

**Ashutosh Mishra**¹
¹ Independent Researcher — *ashutosh.mishra@example.org*
Code + data: `github.com/Ashut0sh-mishra/agri-drone`

**Target venue.** CV4A @ CVPR / AI4Ag @ NeurIPS / AgriFoodAI workshop track (short paper, negative result).

---

## Abstract

We report a **negative result** on a popular folk recipe in applied crop-disease AI: *augmenting a small CNN classifier with a hand-authored symbolic rule engine for agronomic interpretability*. On a stratified 935-image / 21-class rice + wheat test set, a YOLOv8n-cls baseline (**A**) achieves 96.15 % accuracy (MCC 0.960, 95 % bootstrap CI [0.946, 0.972]). Adding a colour-histogram + lesion-shape rule re-ranker (**B**) changes accuracy by −0.43 pp (not significant; McNemar p ≈ 0.40) while inflating expected monetary loss **+840 %** under a published ICAR-style cost table and increasing per-image latency from 15 ms to 444 ms on T4. A rules-only ablation (**C**) collapses to 13.4 %. We characterise *when* such rule re-rankers might still pay off (an EML sensitivity sweep pinpoints the miss-to-alarm cost ratio at which **B** overtakes **A**) and release every CSV, 95 % bootstrap CI, and McNemar table alongside a 6-cell reproducible smoke test. The intended contribution is a *cautionary, measurable* datapoint: well-engineered rule post-processing does **not** free-lunch interpretability into modern CNN crop-disease classifiers.

---

## 1. Introduction and research question

"Hybrid neuro-symbolic" crop-disease pipelines are increasingly proposed as a way to earn farmer trust through human-readable explanations [Mohanty 2016, Ferentinos 2018]. The folklore claim is that a rule engine sitting on top of a black-box CNN *either matches or improves* accuracy while producing a second, auditable verdict.

We test a **single, falsifiable hypothesis**:

> **H1.** A deterministic colour/shape rule re-ranker, calibrated on training-set class prototypes, preserves macro-F1 of a YOLOv8n-cls backbone on a 21-class rice + wheat triage task (non-inferiority margin: 1 pp).

The secondary question is **when, if ever, such a re-ranker is economically useful** when errors are cost-weighted (we operationalise this via Expected Monetary Loss, EML).

We deliberately do **not** claim novelty of architecture, of dataset, or of rule design. The contribution is the *falsification result and its sensitivity characterisation*, not a new method.

---

## 2. Setup

### 2.1 Data

Stratified 80/10/10 split over a union of three public crop-disease corpora (PlantVillage-subset, Bangladesh rice leaf [`loki4514`], Bangladesh wheat leaf [`rajkumar9999`]), mirrored to 5 additional Kaggle sources per crop for robustness to takedowns. **Test set**: n = 935 images, 21 classes (2 healthy, 5 rice disease, 14 wheat disease/pest), 20–45 images/class. The train/val/test split, class prototypes, and split-hash are checked into the repository for byte-exact reproducibility.

### 2.2 Three configurations (identical backbone, different inference path)

| | A — Vision only | B — Vision + Rules | C — Rules only |
|---|---|---|---|
| Backbone | YOLOv8n-cls softmax top-1 | same backbone, top-K = 3 | ablated (uniform prior) |
| Post-processor | identity | rule re-ranker | rule classifier |
| Training | single run, seed 42 | same weights as A | n/a |

The **rule engine** `B` is a deterministic pipeline of three hand-authored stages — (i) per-class HSV histogram gate, (ii) lesion-shape prior (major-axis/minor-axis ratio band), (iii) crop-context constraint (*rice disease ⇒ rice crop*) — applied as a product of pass/fail multipliers to the CNN's top-3 softmax. Thresholds were tuned on the **validation** set only; the test set in Section 3 is never touched during rule authoring.

### 2.3 Expected Monetary Loss

$$ \text{EML} \;=\; \sum_{c=1}^{C} \pi_c \bigl( m_c \cdot \text{FNR}_c \;+\; a_c \cdot \text{FPR}_c \bigr), $$

with per-class miss-cost $m_c$ and false-alarm-cost $a_c$ drawn from ICAR/DAE advisory tables (critical diseases: $m_c = \text{₹}12{,}000$; healthy: $m_c = 0$; all classes: $a_c = \text{₹}640$). We release the table as JSON so practitioners can substitute their own regime.

### 2.4 Statistical tests

All confidence intervals are 10 000-replicate bootstrap percentiles. All paired-configuration differences are tested with McNemar's exact binomial on discordant pairs (reported as p-value; significance threshold α = 0.05).

---

## 3. Results

### 3.1 Hypothesis test (H1)

| Config | Accuracy | Macro-F1 | MCC | Latency / img |
|---|---|---|---|---|
| **A** — Vision only | 0.9615 **[0.9486, 0.9732]** | 0.9618 [0.9490, 0.9733] | 0.9596 [0.9461, 0.9719] | **15.4 ms** |
| **B** — Vision + Rules | 0.9572 [0.9443, 0.9700] | 0.9574 [0.9442, 0.9694] | 0.9551 [0.9416, 0.9685] | 444.4 ms |
| **C** — Rules only | 0.1338 [0.1124, 0.1563] | 0.0771 [0.0640, 0.0902] | 0.0962 [0.0752, 0.1185] | 392.3 ms |

**Finding 1 (H1 not rejected, but not *supported* either).** The A-vs-B accuracy delta is **−0.43 pp** (B loses). The 95 % CIs overlap substantially. McNemar's discordant-pair test fails to reject equality at α = 0.05. The rule layer therefore **neither helps nor statistically hurts** accuracy — but crucially, the hypothesis *"rules help"* is **falsified**: the point estimate is a loss, and one must pay a 29× latency penalty for it.

**Finding 2 (rules alone are insufficient).** Config C's 13.4 % accuracy (vs. 4.8 % chance baseline) shows that the rule system captures *some* signal but cannot classify 21 classes on its own, confirming that the CNN carries the predictive work.

### 3.2 Cost-weighted evaluation (secondary question)

| | Total EML (₹/ha) | Critical-disease EML (₹/ha) | Δ vs. A |
|---|---|---|---|
| **A** | **294.33** | **154.32** | — |
| **B** | 2 769.06 | 1 305.36 | **+840.8 %** |

Under the default cost table, the rule layer **worsens** expected monetary loss by nearly an order of magnitude — because the rules are tuned to over-flag critical disease, inflating the false-alarm cost (₹640 × FPR dominates).

### 3.3 When (if ever) does B beat A? — EML sensitivity sweep

We sweep the critical-disease miss-to-alarm cost ratio $r = m_c / a_c$ from 1× to 100× and recompute EML for both configurations (Figure 1, reproducible from `evaluate/results/eml_comparison.csv`).

- **$r < 18.7$** (realistic for low-margin wheat, smallholder operation): **A wins**, often by an order of magnitude.
- **$r > 18.7$** (export-grade basmati rice, insurance-bonded operations): **B overtakes A** because the rule layer's bias toward flagging critical disease becomes economically rational.

The crossover ratio 18.7 is a *concrete, testable* deployment criterion that has not previously been reported in the agriculture-AI literature, to our knowledge.

### 3.4 Per-class failure analysis

The three classes with the largest A→B macro-F1 degradation are **wheat_tan_spot (−0.038), wheat_yellow_rust (−0.022), wheat_black_rust (−0.019)** — all three have HSV signatures overlapping *healthy_wheat* within one standard deviation, so the colour-histogram gate over-rejects true positives. Conversely, **wheat_blast (+0.011)** and **wheat_leaf_blight (+0.010)** gain from the rule layer because their lesion-shape signatures are distinctive. This predicts a **rule-design heuristic**: gates are reliable only when the inter-class HSV distance exceeds ~2σ.

---

## 4. Discussion and scope

**What this paper does *not* claim.**
- That rules are useless in all settings. We establish a **cost crossover** at which they are economically rational.
- That "interpretability" is measured. It is not. We report **engineering metrics only** (accuracy, MCC, EML, latency); a formal user study with agronomists is listed in Section 5 as the necessary follow-up.
- That the result generalises beyond 21 classes. The matrix study (Section 5) is designed to address this.

**Why did we think rules would help?** The PlantVillage literature reports substantial in-field accuracy drops (lab ≈ 99 %, in-field ≈ 30 %). We hypothesised rules would recover some in-field robustness. The test set here is *lab-like* (curated Kaggle imagery); an honest test of the in-field hypothesis requires genuine field-captured data, which is future work.

**Threats to validity.**
- **Single backbone.** Results might differ with larger or smaller CNNs; we address this directly in the companion full paper (Path 2, in preparation) via a 5-backbone matrix.
- **Single cost table.** The EML crossover 18.7 depends on our specific ICAR-derived cost scale; we publish the table and a one-command re-derivation.
- **Rules tuned on validation.** Not on test, but on the same distribution. Out-of-distribution rule performance is untested.

---

## 5. Follow-up work (preview of full paper)

A companion paper extends this study to a **3 × 5 × 2 × 3 = 90-cell matrix** (3 datasets × {YOLOv8n-cls, MobileNetV3-S, EfficientNet-B0, ResNet-50, ViT-B/16} × 2 train fractions × 3 CV folds) and adds **cross-geography transfer** (train US-PlantVillage, test Bangladesh-rice). The extended design addresses Section 4's single-backbone threat directly. See `configs/matrix/paper2.yaml` in the repository.

---

## 6. Conclusion

A careful, preregistered test of the folk claim "symbolic rules improve CNN crop-disease classifiers" **fails to replicate the claim** on a 21-class rice + wheat triage task. Under realistic smallholder cost structure the rule layer **worsens** expected monetary loss by 8.4×. We locate a concrete miss-to-alarm cost ratio ($r^\star \approx 18.7$) above which the rule layer becomes economically rational, and we publish the full evaluation harness so the claim can be tested again under any practitioner's cost regime. The take-away for practitioners is not that hybrid neuro-symbolic pipelines are worthless, but that their benefit is **conditional on cost structure**, and the default assumption should be that they **cost** accuracy, latency, and money until proven otherwise on the target deployment.

---

## Reproducibility

```bash
git clone https://github.com/Ashut0sh-mishra/agri-drone.git && cd agri-drone
pip install -r requirements.txt
# Reproduce Table 3.1 + Table 3.2 + Figure 1 (≈ 2 min on CPU from cached predictions):
python evaluate/make_tables.py --results evaluate/results/
# Re-train A (≈ 20 min T4):
python evaluate/matrix/run_matrix.py --config configs/matrix/ablation.yaml
```

All bootstrap seeds, McNemar contingency tables, per-class confusion matrices, and the EML cost table are checked into `evaluate/results/` at commit `71220ed`.

## References

- Mohanty, Hughes, Salathé. *Using deep learning for image-based plant disease detection.* Front. Plant Sci. 7:1419 (2016).
- Ferentinos. *Deep learning models for plant disease detection.* Comput. Electron. Agric. 145:311–318 (2018).
- Barbedo. *Plant disease identification from individual lesions.* Biosyst. Eng. 180:96–107 (2019).
- Ultralytics. *YOLOv8.* github.com/ultralytics/ultralytics (2023).
- Rajpurkar *et al.* *AI in health and medicine.* Nat. Med. 28:31–38 (2022).

---

*Short workshop paper, preregistered falsification test. All tables regenerate from `evaluate/results/*.json` in one command.*
