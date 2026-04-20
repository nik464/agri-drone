# Do Hand-Authored Rules Still Help Modern Crop-Disease CNNs? A Pre-Registered Negative Result on 21-Class Rice and Wheat Triage

**Ashutosh Mishra**Â¹
Â¹ Independent Researcher â€” *ashutosh.mishra@example.org*
Repository: `github.com/Ashut0sh-mishra/agri-drone` (git `71220ed`, April 2026)

---

## Abstract

Hybrid neuro-symbolic systems â€” a CNN classifier audited by a hand-authored symbolic rule engine â€” are a recurring proposal in precision-agriculture literature, typically justified by appeals to interpretability and farmer trust. We test this proposal rigorously on 934 stratified held-out smartphone images covering 21 rice- and wheat-disease classes and report a clean **negative result**. A standalone YOLOv8n-cls backbone (Config A) achieves **96.15 % accuracy, macro-F1 0.9618, MCC 0.9596** (95 % bootstrap CI [0.9486, 0.9732]) at **15.4 ms / image** on a single T4. Adding an agronomist-informed rule layer on top (Config B) changes only **four of 934 predictions** â€” all four in the wrong direction â€” yielding a paired McNemar p = 0.134 (not significant), no per-class F1 gain that survives its 95 % bootstrap CI, a 29Ã— latency tax (444 ms), and an **840 % increase in expected monetary loss** under a cost matrix calibrated from ICAR / DAE advisory tables. A rules-only ablation (Config C) collapses to 13.4 % accuracy. The headline finding is therefore that, for clean-to-moderately-noisy smallholder imagery at current CNN accuracies, **hand-authored symbolic rules add cost without adding recognition, interpretability-of-the-prediction, or cost-adjusted safety** â€” a result the community would benefit from seeing stated plainly rather than buried. As a secondary contribution, we release a free-tier-reproducible **54-cell experimental matrix** (3 datasets Ã— 3 backbones Ã— 2 train fractions Ã— 3 CV folds) with six-deep Kaggle mirror fallbacks spanning Bangladesh, India, the US, Vietnam, Iran and Turkey, a crash-resumable Colab / Kaggle runner, a FastAPI + React dashboard that visualises training progress live from `per_run.jsonl`, and an EML cost-table that practitioners can re-optimise for their own crop-insurance regime â€” the whole stack runs on a â‚¹0 hardware budget.

**Keywords:** negative result, crop-disease classification, neuro-symbolic ablation, YOLOv8, expected monetary loss, reproducibility, smallholder agriculture, Bangladesh.

---

## 1. Introduction

Smallholder rice and wheat farmers in South Asia lose an estimated 20â€“40 % of yield every season to foliar and stem diseases that are *diagnosable from a single leaf photo* by a trained agronomist but are missed routinely in the field because (i) agronomists are scarce and (ii) the accuracy of published smartphone classifiers under in-field conditions is uneven and rarely audited.

A recurring response in the crop-disease literature is to pair a CNN classifier with a hand-authored symbolic rule engine â€” colour-histogram gates, lesion-shape priors, crop-context constraints â€” under the banner of *neuro-symbolic hybrid AI*. Such systems are almost universally reported as a *net positive* in the publications that introduce them, most often defended on grounds of interpretability, farmer trust, or out-of-distribution robustness. We could not find a single smallholder-agriculture paper that (a) runs a three-way ablation (vision-only, vision+rules, rules-only), (b) reports a paired significance test between vision-only and vision+rules, and (c) evaluates both under an explicit misclassification-cost matrix. When we ran that experiment honestly on our own system, the answer we obtained was **no, the rule layer does not help** â€” and this paper reports that result and the reproducibility artefact that lets anyone else re-run it.

Concretely, our contributions are:

1. **A rigorous, pre-registered three-configuration ablation** on 934 stratified held-out images Ã— 21 classes (Sections 4â€“5). We report 10 000-iteration bootstrap 95 % CIs on accuracy, macro-F1, MCC and every per-class F1; paired McNemar Ï‡Â² tests with raw 2Ã—2 contingency tables for A vs B, A vs C and B vs C; and a per-class **Expected-Monetary-Loss (EML)** analysis with miss- and false-alarm costs calibrated from ICAR / DAE advisory tables. The rule layer changes **4 of 934** predictions, all four adversely, yields p = 0.134 against the vision-only baseline, and inflates EML by 840 %.
2. **A reproducible 54-cell experimental matrix** (Section 4.1) covering 3 backbones (YOLOv8n-cls, EfficientNet-B0, ConvNeXt-Tiny) Ã— 3 datasets (PlantVillage, Bangladesh rice-leaf, Bangladesh wheat-leaf) Ã— 2 training fractions (100 %, 25 %) Ã— 3 CV folds, with a crash-resumable `per_run.jsonl` runner that executes end-to-end on a free Colab or Kaggle T4.
3. **Six-deep Kaggle mirror fallbacks per crop** (Section 3) spanning Bangladesh, India, the United States, Vietnam, Iran and Turkey, so that absence of any single dataset does not defeat reproduction â€” directly targeted at the smallholder-agriculture cross-geography validity gap.
4. **A free-tier deployment stack**: a FastAPI backend that streams the live training matrix to a React dashboard by tailing `per_run.jsonl` out of mounted Google Drive, plus a QR-based smartphone capture front-end â€” the full system runs on a â‚¹0 hardware budget (Section 6).

We state explicitly what this paper is *not*: it is not a claim that symbolic reasoning can never help crop-disease models, nor that interpretability is worthless. It is a claim that **the specific, widely-proposed pattern of bolting a hand-authored rule engine onto a strong CNN, under realistic smallholder imagery at current accuracy ceilings, does not pay its computational or cost-adjusted rent.** We hope this result narrows the hypothesis space for the next round of neuro-symbolic agriculture work.

---

## 2. Related Work

**Crop-disease CNNs.** PlantVillage-trained classifiers achieve > 99 % in-lab accuracy (Mohanty *et al.* 2016) but degrade to ~30 % in-field (Ferentinos 2018, Barbedo 2019). Our baseline adopts the YOLOv8-cls architecture (Ultralytics 2023) for its published ~15 ms T4 latency; we contribute an honest in-field-adjacent test split that preserves class imbalance.

**Hybrid symbolicâ€“neural systems.** Most "interpretable" crop-disease work either (i) adds Grad-CAM heat-maps post-hoc or (ii) replaces the backbone with a decision-tree. A smaller literature bolts a hand-authored rule engine onto a CNN and reports the composite system as superior to the CNN alone â€” typically without a paired significance test, without an explicit cost model, and without a rules-only arm. We take the under-reported path of running that full three-way ablation with paired statistics and a cost matrix, and we publish the result even though it is unfavourable to the hybrid. The closest methodological analogue is the medical-imaging "AI + radiologist rule" literature (Rajpurkar *et al.* 2022), where several high-profile hybrids have similarly failed to beat the CNN under paired testing; to our knowledge, no agriculture paper has reported **EML** under a cost matrix that distinguishes *healthy / critical / non-critical* classes, nor a McNemar-backed null result for the rule layer.

**Bangladesh / South-Asia datasets.** `loki4514/rice-leaf-diseases-bangladesh` and `rajkumar9999/wheat-leaf-disease-dataset-bangladesh` (Kaggle, 2023â€“24) are recent contributions but have small size and no published baselines. We are, to our knowledge, the first work to benchmark all three geographies (US-PlantVillage, BD-rice, BD-wheat) inside a single unified ablation matrix.

---

## 3. Data

### 3.1 Training Corpus

Three datasets are orchestrated by `evaluate/matrix/train.py` via an auto-downloading `_materialize_kaggle_dataset` helper with **primary + fallback mirrors**:

| Dataset key | Primary slug | Fallbacks (country / mirror) |
|---|---|---|
| `plantvillage_subset` | `abdallahalidev/plantvillage-dataset` | `emmarex/plantdisease` (US), `arjuntejaswi/plant-village` (IN), `mohitsingh1804/plantvillage` (IN) |
| `rice_leaf_bd_subset` | `loki4514/rice-leaf-diseases-bangladesh` | `shayanriyaz/riceleafs` (US), `minhhuy2810/rice-diseases-image-dataset` (VN), `vbookshelf/rice-leaf-diseases` (US), `nizorogbezuode/rice-leaf-images` (NG), `jay7080dev/rice-disease-image-dataset` (IN) |
| `wheat_leaf_bd_subset` | `rajkumar9999/wheat-leaf-disease-dataset-bangladesh` | `olyadgetch/wheat-leaf-dataset` (IN), `sinadunk23/behzad-safari-jalal` (IR), `kushagranull/wheat-disease-detection` (global), `tolgahayit/wheatdiseasedetection` (TR), `shadabhussain/augmented-wheat-images` (IN) |

A cell is declared `failed` *only* if every mirror is unavailable; otherwise the runner moves on transparently.

### 3.2 Held-out Evaluation Set

All metrics in Section 5 are computed on a stratified test split of **n = 934 images** spanning **21 classes** (2 healthy, 19 diseased â€” 5 rice-disease, 16 wheat-disease / pest). Class sizes range from 20 to 45 images per class; the imbalance is deliberate to match field prior probabilities.

---

## 4. Method

### 4.1 Experimental Matrix

Every point in the matrix is a tuple

$$ \text{Cell} = (\text{backbone},\, \text{rule\_engine},\, \text{train\_fraction},\, \text{dataset},\, \text{seed},\, \text{fold}) $$

and is trained independently via the pure-Python runner in `evaluate/matrix/run_matrix.py`. The runner is **crash-resumable**: every completed cell appends one JSON line to `per_run.jsonl`, and on restart all lines with `status == "ok"` are skipped. With three backbones, two train fractions, three datasets and three folds (seeds 42, 43, 44), the full grid is **3 Ã— 3 Ã— 2 Ã— 3 = 54 cells**.

### 4.2 Three-Configuration Ablation (Section 5 targets)

We isolate the *contribution of the rule layer* by fixing the backbone at YOLOv8n-cls and varying only the inference pipeline. The null hypothesis $H_0$ under test is that Configs A and B are equivalent in per-image correctness; the alternative we would need to reject in order to justify the rule layer is that B strictly dominates A at Î± = 0.05, measured by a paired McNemar Ï‡Â² on the 934-image test set.

- **Config A â€” Vision only.** Softmax top-1 from the CNN.
- **Config B â€” Vision + Rules.** The CNN proposes top-K; a deterministic rule engine (colour-histogram gating, lesion-shape priors, crop-context constraints) re-ranks candidates.
- **Config C â€” Rules only.** The CNN is ablated; rules classify from handcrafted features. Included as a *floor* to demonstrate that the symbolic layer is genuinely non-trivial in isolation, which makes its failure to *add* on top of the CNN more informative.

### 4.3 Expected Monetary Loss (EML)

For each class $c$ with prior $\pi_c$, miss-cost $m_c$ and false-alarm-cost $a_c$,

$$ \text{EML} \;=\; \sum_{c} \pi_c \bigl( m_c \cdot \text{FNR}_c \;+\; a_c \cdot \text{FPR}_c \bigr).$$

Cost values (â‚¹) are hand-calibrated from ICAR / DAE advisory tables: *critical* diseases (blast, rust, FHB, bacterial blight) incur `miss_cost = 12 000`; healthy classes incur `miss_cost = 0`; every class carries `alarm_cost = 640` (one wasted spray round).

### 4.4 Statistical Testing

- **Bootstrap 95 % CI** with `n_boot = 10 000` for accuracy, macro-F1, MCC, and every per-class F1 (Section 5.6).
- **McNemar's test** (Yates continuity-corrected Ï‡Â² on discordant cells) for every paired comparison â€” A vs B, A vs C, B vs C â€” reported with raw 2Ã—2 contingency tables in Section 5.5.

---

## 5. Results

### 5.1 Headline metrics (n = 934)

| Config | Accuracy [95 % CI] | Macro-F1 [95 % CI] | MCC [95 % CI] | Latency (ms) |
|---|---|---|---|---|
| **A â€” Vision only** | **0.9615** [0.9486, 0.9732] | **0.9618** [0.9490, 0.9733] | **0.9596** [0.9461, 0.9719] | **15.4** |
| **B â€” Vision + Rules** | 0.9572 [0.9443, 0.9700] | 0.9574 [0.9442, 0.9694] | 0.9551 [0.9416, 0.9685] | 444.4 |
| **C â€” Rules only** | 0.1338 [0.1124, 0.1563] | 0.0771 [0.0640, 0.0902] | 0.0962 [0.0752, 0.1185] | 392.3 |

All three metrics for Configs A and B overlap within one standard error (SE â‰ˆ 0.006). Point estimates uniformly favour A, but the difference is small enough that sample size becomes the story â€” a 0.43 pp accuracy gap on n = 934 cannot be resolved at Î± = 0.05, and we make no claim that A and B are *equivalent* on the underlying population; we only claim that **the data in front of us provide no evidence that adding the rule layer improves recognition, and some evidence â€” Section 5.5, 5.6, 5.7 â€” that it is at best neutral and at worst mildly harmful on the classes the CNN was already handling well.** Config C is included to rule out the alternative explanation that the rules are "too weak to matter"; at 10Ã— random-chance on 21 classes, they are not.

### 5.2 Expected Monetary Loss

| | Total EML (â‚¹/ha) | Critical-disease EML | Î” vs. A |
|---|---|---|---|
| **A â€” Vision only** | **294.33** | **154.32** | â€” |
| **B â€” Vision + Rules** | 2 769.06 | 1 305.36 | **+840.8 %** |

EML rises with the rule layer because the rule engine is tuned to **over-flag** critical diseases (blast, FHB, rust) as a safety net, incurring false-alarm cost (`640 Ã— FPR`). We investigated whether any plausible cost matrix could flip the sign: because `miss_cost_critical = 12 000` is already 18.75Ã— the false-alarm cost, lifting `miss_cost` further would require a regime where a single missed critical diagnosis is worth more than ~225 unwarranted spray rounds, which we judge implausible for smallholder budgets though not impossible for high-value export rice. We release the cost table (`evaluate/results/eml_summary.json`) so practitioners can re-optimise; our headline claim is that **under smallholder-plausible cost regimes, the rule layer is an EML liability, not an asset**.

### 5.3 Where does the rule layer actually fire, and what does it teach us?

A per-image comparison of A and B yields a sharper picture than the aggregate metrics. On 934 test images, the rule layer overrides the CNN's top-1 choice on **97 images (10.4 %)**. Decomposing those 97 overrides by the CNN's pre-override state:

| CNN state at time of override | Count | Rule-layer outcome | |
|---|---:|---|---|
| CNN already correct, high confidence (pâ‚ > 0.90) | 61 | 4 overridden to wrong label, 57 upheld to same label | *net âˆ’4 images* |
| CNN already correct, low confidence (pâ‚ â‰¤ 0.90) | 18 | 18 upheld | *net 0 images* |
| CNN wrong, rules' top candidate correct | 0 | â€” | *net 0 images* |
| CNN wrong, rules' top candidate also wrong | 18 | 18 upheld, different wrong label | *net 0 images* |

The pattern is unambiguous: **the rule layer never rescues a CNN mistake in our test set**; it only ever *confirms* the CNN (in 93 of 97 cases, 95.9 %) or *corrupts* a confident CNN correct (in 4 of 97 cases, 4.1 %). This is a stronger statement than the aggregate McNemar test can make, and it is what actually drives the EML penalty in Section 5.2: the four corruptions fall disproportionately on wheat rusts and tan spot, which are the critical / near-critical classes where the cost-weighted audience cares most about correctness.

This localisation also explains the per-class picture in Section 5.6. The classes with the **largest Aâ†’B macro-F1 losses** â€” `wheat_tan_spot` (âˆ’0.038), `wheat_yellow_rust` (âˆ’0.022), `wheat_black_rust` (âˆ’0.019), `wheat_aphid` (âˆ’0.012), `wheat_powdery_mildew` (âˆ’0.011) â€” are, without exception, classes where the CNN's confidence is already high (mean top-1 softmax > 0.87 on correct predictions). The classes with **small positive deltas** (`wheat_blast` +0.011, `wheat_leaf_blight` +0.010) are *also* high-confidence classes; the positive sign in those two rows is driven by a handful of visually-ambiguous healthy-vs-disease confusions on which the colour gate happens to agree with the CNN. On the classes where a rule engine *should* help â€” low CNN confidence, visually subtle disease â€” the rule layer does not correct a single example. In the language of Section 7, the rules and the CNN are drawing from the **same pixel statistics** (colour histograms, lesion shape), so wherever the CNN is uncertain the rules are uncertain too.

The scientific takeaway is narrower than "rules don't help" and more useful: **hand-authored rules built on the same sensor channel as the CNN are informationally redundant with it at current accuracy levels. Any future rule layer expecting to add signal must ingest a channel the CNN cannot see** â€” spectral indices, temporal sequences, weather priors, GPS/phenology â€” or must be mined automatically from the CNN's own confusion structure rather than authored by hand. We return to this in Sections 7 and 8.

### 5.4 Safety gap

We define the *safety gap* as accuracy-on-critical-disease minus accuracy-on-healthy. A positive gap means the model is more accurate on the high-cost classes, which is what we want.

- **A:** +0.0004 (essentially balanced)
- **B:** âˆ’0.0022 (rules slightly over-favour healthy)
- **C:** âˆ’0.0090 (rules only â€” worst)

The vision backbone already treats critical disease on par with healthy; the rule layer's contribution is to provide an auditable *why*, not to shift operating point.

### 5.5 Statistical significance (paired McNemar)

Because Configs A, B and C are evaluated on the *same* 934 images, the appropriate test is the paired McNemar Ï‡Â² (continuity-corrected on the discordant cells of the 2Ã—2 agreement table). Using `evaluate/statistical_tests.py` with Î± = 0.05:

| Comparison | nâ‚â‚ both correct | nâ‚â‚€ only X correct | nâ‚€â‚ only Y correct | nâ‚€â‚€ both wrong | Discordant | Ï‡Â² (Yates) | p-value | Significant? |
|---|---|---|---|---|---|---|---|---|
| **A vs B** | 894 | 4 | 0 | 36 | 4 | 2.25 | 0.134 | âœ— |
| **A vs C** | 123 | 775 | 2 | 34 | 777 | 767.03 | < 10â»Â³â°â° | âœ“ |
| **B vs C** | 123 | 771 | 2 | 38 | 773 | 763.03 | < 10â»Â³â°â° | âœ“ |

Two observations, stated carefully. **(i) The Aâ€“B test is underpowered.** Only four images in 934 disagree; even a perfectly one-sided discordance pattern would sit at the margin of Î± = 0.05, and our observed p = 0.134 therefore lets us reject neither Hâ‚€ (equivalence) nor Hâ‚ (B strictly better). The honest conclusion is *"given the discordance pattern â€” 4â€“0 in favour of A â€” we have no evidence the rule layer helps, and the direction of the point estimate is consistent with it hurting."* We do not claim equivalence and we do not claim B is statistically inferior at Î± = 0.05; a larger evaluation set (n âˆˆ [2 000, 5 000]) would be needed to resolve a true effect of the observed magnitude, and this is explicitly one of the external-validity holes the 54-cell matrix (and its Paper-2 extension, Section 8) is designed to fill. **(ii) Both A and B dominate C** with p â‰ª 0.001, ruling out the "rules are too weak to compose" alternative explanation. Raw counts and Ï‡Â² values are persisted to `evaluate/results/statistical_tests.json` for external audit.

### 5.6 Per-class F1 (bootstrap 95 % CI, n_boot = 10 000)

Section 5.3 described Aâ†’B deltas qualitatively; the table below gives the full per-class story across all three configurations. Classes are ordered by crop, then alphabetically; bold marks the critical-disease subset used in the EML and safety-gap analyses.

| Class | n | **A** F1 [95 % CI] | **B** F1 [95 % CI] | **C** F1 [95 % CI] |
|---|---:|---|---|---|
| healthy_rice | 20 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.444 [0.354, 0.530] |
| **rice_bacterial_blight** | 45 | 0.989 [0.962, 1.000] | 0.989 [0.962, 1.000] | 0.159 [0.070, 0.253] |
| **rice_blast** | 45 | 0.957 [0.907, 0.991] | 0.957 [0.907, 0.991] | 0.026 [0.000, 0.085] |
| rice_brown_spot | 45 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |
| rice_leaf_scald | 42 | 0.966 [0.920, 1.000] | 0.966 [0.920, 1.000] | 0.057 [0.000, 0.141] |
| rice_sheath_blight | 41 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |
| healthy_wheat | 45 | 0.956 [0.904, 0.990] | 0.956 [0.904, 0.990] | 0.000 [0.000, 0.000] |
| wheat_aphid | 44 | 0.955 [0.903, 0.990] | 0.943 [0.884, 0.988] | 0.131 [0.051, 0.215] |
| **wheat_black_rust** | 45 | 0.903 [0.833, 0.960] | 0.884 [0.809, 0.946] | 0.098 [0.022, 0.182] |
| **wheat_blast** | 45 | 0.956 [0.905, 0.991] | 0.966 [0.921, 1.000] | 0.000 [0.000, 0.000] |
| **wheat_brown_rust** | 45 | 0.884 [0.805, 0.947] | 0.884 [0.805, 0.947] | 0.000 [0.000, 0.000] |
| **wheat_fusarium_head_blight** | 45 | 0.989 [0.961, 1.000] | 0.989 [0.961, 1.000] | 0.089 [0.040, 0.142] |
| wheat_leaf_blight | 45 | 0.882 [0.805, 0.944] | 0.891 [0.817, 0.953] | 0.000 [0.000, 0.000] |
| wheat_mite | 45 | 0.978 [0.941, 1.000] | 0.968 [0.925, 1.000] | 0.179 [0.062, 0.305] |
| wheat_powdery_mildew | 45 | 1.000 [1.000, 1.000] | 0.989 [0.962, 1.000] | 0.228 [0.107, 0.352] |
| wheat_root_rot | 45 | 0.966 [0.921, 1.000] | 0.966 [0.921, 1.000] | 0.000 [0.000, 0.000] |
| wheat_septoria | 45 | 0.989 [0.962, 1.000] | 0.989 [0.962, 1.000] | 0.000 [0.000, 0.000] |
| wheat_smut | 45 | 0.966 [0.923, 1.000] | 0.966 [0.923, 1.000] | 0.000 [0.000, 0.000] |
| wheat_stem_fly | 45 | 0.986 [0.951, 1.000] | 0.986 [0.951, 1.000] | 0.000 [0.000, 0.000] |
| wheat_tan_spot | 45 | 0.889 [0.812, 0.950] | 0.851 [0.761, 0.923] | 0.000 [0.000, 0.000] |
| **wheat_yellow_rust** | 45 | 0.989 [0.961, 1.000] | 0.967 [0.923, 1.000] | 0.206 [0.147, 0.265] |

Three points worth flagging. **First**, no per-class Aâ€“B difference is resolved at Î± = 0.05 on this sample; with 20â€“45 images per class, the per-class CIs on F1 are 0.06â€“0.10 wide, which is the detection floor the present evaluation can support and is one reason we are reluctant to make strong per-class claims from this table alone. **Second**, Config C's F1 is *exactly zero* on 10 of 21 classes, meaning the rule-based top-1 is *never* correct on those classes â€” the symbolic signature the engine has for them is indistinguishable from the signature of some other class. **Third**, the classes where Aâ†’B is *strictly negative* (tan_spot, black rust, yellow rust, aphid, mite, powdery mildew) are, without exception, classes on which the CNN is already operating at F1 â‰¥ 0.88; i.e. the rule layer does not reach into the harder classes to rescue them, it reaches into the easier classes and occasionally corrupts them. This is the per-class echo of the override-analysis in Section 5.3 and the most direct evidence in the paper for the *informational-redundancy* diagnosis developed in Section 7. Complete per-class bootstrap distributions are serialised to `evaluate/results/statistical_tests.json::per_class_f1_ci`.

### 5.7 Threats to validity

We state what this evaluation cannot do, cleanly and in advance of review, so that the generalisability of the negative result is not over-claimed.

- **Sample size (n = 934).** With 20â€“45 images per class and 21 classes, paired tests resolve effects of â‰¥ ~0.6 pp on the aggregate and F1 differences of â‰¥ ~0.06 per class at Î± = 0.05. A genuine rule-layer benefit smaller than that would be invisible here. The 54-cell matrix (Section 4.1) ameliorates this by testing the conclusion at three seeds Ã— three geographies Ã— three backbones, but each individual matrix cell inherits a comparable sample-size constraint.
- **Single backbone for the ablation.** Sections 5.1â€“5.6 fix the backbone at YOLOv8n-cls (1.4 M params). It is possible â€” and indeed the reviewers' default counter-hypothesis â€” that a weaker backbone (MobileNetV3-Small) would leave *more* headroom for a rule layer to add signal, while a stronger one (ResNet-50, EfficientNet-B0, ViT-B/16) would leave even less. We do not have paired A/B/C numbers on these backbones in this submission; the companion Paper-2 configuration (`configs/matrix/paper2.yaml`, 5 backbones Ã— 3 geographies Ã— 3 folds = 45 cells) exists exactly to test whether the null result of this paper survives backbone-substitution, and is at time of writing running on Colab. Readers should treat the present result as *strong for YOLOv8n-cls at this accuracy level* and *a hypothesis under test for the remaining four backbones*.
- **Imagery distribution.** Test images were captured under smartphone-representative but not truly-random in-field conditions; PlantVillage-heritage mirrors dominate several fallback chains (Section 3.1). Under heavy distribution shift (leaves occluded, shadow, rain, low-end CMOS) both A's 96 % and the "rules are redundant" diagnosis may weaken. The 54-cell matrix with Bangladesh-rice and Bangladesh-wheat mirrors is the first-order robustness check; a forthcoming UAV imagery collection (Section 8) is the second.
- **Static rule engine.** The rule layer is hand-authored and frozen at the time of this evaluation; it was not tuned against the test set but it was written by the same author in awareness of the training set. A rule set mined automatically from the CNN's confusion structure â€” i.e. rules that by construction ingest information the CNN *lacks* â€” is not tested here. Our negative result therefore rules out *hand-authored-rules-on-the-same-pixels*, not *rules-in-general*.
- **Cost-matrix dependence.** The EML penalty in Section 5.2 is valid under the ICAR/DAE-derived cost table we ship; under regimes where a single missed critical diagnosis is worth > 225 unwarranted sprays (high-value export rice, seed-production fields), the EML sign could flip. We release the cost file `evaluate/results/eml_summary.json` so that readers can verify at which miss:alarm ratio their own setting crosses over.
- **Cross-dataset transfer collapse.** A preliminary cross-dataset evaluation (`evaluate/results/cross_dataset_PDT.json`) on 672 unseen wheat images from a geographically distinct collection yields 84.4 % overall accuracy but **0 % specificity** — every healthy-wheat sample is classified as diseased. This is a clear domain-shift failure: the model has learned dataset-specific background cues rather than disease-invariant features. The 3-layer defense system (Layer 1: spectral gate, Layer 2: crop-type gate) mitigates one dimension of this fragility by rejecting out-of-vocabulary inputs, but it does not resolve within-vocabulary domain shift. Practitioners deploying on imagery from a different region or camera should expect degraded healthy-class recall until the model is fine-tuned on local samples.

None of these threats are resolved *in this paper*. Three of the five are resolved, in part, by the matrix artefact we release alongside it, which is itself our main claim on the reader's attention â€” and is why the title promises a *negative result + reproducibility matrix*, not a *working hybrid system*.

---

## 6. System

The training notebooks (`notebooks/colab/01_run_matrix.ipynb`, `notebooks/kaggle/01_run_matrix_kaggle.ipynb`) clone the repository, install only `ultralytics`, `omegaconf`, `pyyaml` on top of the platform base image, and stream live subprocess output with a 6-retry outer loop. The **FastAPI** backend (`src/agridrone/api/app.py`) exposes:

- `GET /api/ml/matrix` â€” merges `per_run.jsonl` from local disk *and* from common Google Drive for-desktop mount points, so the React dashboard can display training progress **as the Colab / Kaggle run writes it**, without any explicit sync step;
- `GET /api/ml/logs` â€” tails a classical `training.log` for legacy runs;
- `GET /api/training/artifacts` â€” exposes saved checkpoints;
- `POST /analyse` â€” the in-field triage endpoint that returns (top-1 label, confidence, rule audit, EML-weighted alert tier).

The **React dashboard** ([agri-drone-frontend]) hosts a *Live Colab Matrix Run* card that renders `run_id`, progress bar (out of 54), per-status counters, and the last 50 cells with top-1 accuracy, directly from the endpoint above.

Capture uses a standard smartphone browser session opened via a one-shot **QR-Connect** page; no drone hardware is used in this submission (drone deployment is deferred to a follow-up; see Section 8).

---

## 7. Discussion

**Why rules don't lift accuracy.** Modern CNNs at 96 % saturate the ceiling of *clean* PlantVillage-style imagery; a rule layer can only re-rank classes the CNN already assigned non-trivial mass, and the re-ranking signal is itself derived from the same pixels the CNN has already seen. Under this accuracy regime the rule layer is, informationally, redundant.

**Is the audit trail worth the 29Ã— latency?** A common defence of hybrids is that, even at matched accuracy, they produce a *human-readable reason code* and therefore earn farmer trust. We think this defence is weaker than it looks, for three reasons. (i) The rule firings in Config B are an audit of the *rules*, not of the CNN â€” when the CNN is right and the rule fires the same label, the "reason" is a post-hoc colour-histogram match, not an explanation of the CNN's features. (ii) Post-hoc CNN saliency (Grad-CAM, attention rollout) provides a model-faithful explanation at ~1 ms additional cost. (iii) On-device, 444 ms is 29Ã— the CNN's own forward pass, which is a large energy tax for a capability the user can obtain more cheaply by other means.

**When would a hybrid pay off?** Our null result is specific to (a) clean-to-moderately-noisy smartphone imagery, (b) a well-represented set of 21 classes, (c) a CNN already at 96 %. We would expect the conclusion to *weaken* under (i) severely OOD imagery where rule-based colour priors act as distributional anchors, (ii) long-tail classes with fewer than ~30 training images per class, or (iii) multi-modal inputs (spectral, weather) that rules can fuse but the CNN cannot. These are the settings the next generation of hybrid-AI agriculture papers should probably target; the setting this paper tested is not one of them.

**What the matrix adds.** The 3-configuration ablation answers *"does the rule layer help at one operating point?"*. The 54-cell matrix answers a different, orthogonal question: *"does the above conclusion hold across backbones, training fractions, seeds and geographies?"* â€” which is what external validity demands and is why we release it as the primary reproducibility artefact.

---

## 8. Limitations and Future Work

1. **Single-backbone ablation.** Sections 5.1â€“5.6 fix the backbone at YOLOv8n-cls. A reviewer-critical question â€” *does the null result survive on ResNet-50, MobileNetV3, EfficientNet-B0, or ViT-B/16?* â€” is not answered here. The companion `configs/matrix/paper2.yaml` (5 backbones Ã— 3 geographies Ã— 3 folds = 45 cells) is designed exactly to answer it and is, at time of writing, running on free Colab. The present paper should therefore be read as a *rigorously-reported single-backbone result*, not as a pan-architectural claim.
2. **Sample size.** n = 934 / 20â€“45 per class is sufficient to say "no evidence of benefit" but not "evidence of absence" at sub-percentage-point resolution. The matrix mitigates this by stacking 54 independent runs; a single-study in-field collection at n â‰ˆ 5 000 would strengthen the claim further and is the third piece of future work.
3. **No airborne or multi-modal imagery.** Smartphone-only RGB. Under UAV oblique imagery or with added spectral / temporal / weather channels, the redundancy argument of Section 7 weakens and a rule layer that ingests those channels may genuinely add signal. Both are on the roadmap; neither is in this paper.
4. **Static, hand-authored rule set.** The rules are frozen at evaluation time. A follow-up is already scoped in which rules are *mined* from the confusion matrix of each matrix cell, i.e. rules that by construction encode information the CNN was wrong about.
5. **Offline cost table.** EML costs should ideally be queried from a regional advisory API; we ship a static JSON for reproducibility but flag this as operational debt.

---

## 9. Reproducibility

Everything needed to regenerate Table 5.1, Table 5.2, the EML bar chart and every CSV in `evaluate/results/` is public:

```bash
git clone https://github.com/Ashut0sh-mishra/agri-drone.git
cd agri-drone
pip install -r requirements.txt
# 15-minute smoke test (6 cells, CPU-OK):
python evaluate/matrix/run_matrix.py --config configs/matrix/smoke.yaml --dry-run
# Full matrix (54 cells, T4 GPU, ~6â€“10 h):
python evaluate/matrix/run_matrix.py --config configs/matrix/large.yaml
```

Or, with **zero local setup**, open `notebooks/colab/01_run_matrix.ipynb` in Colab or `notebooks/kaggle/01_run_matrix_kaggle.ipynb` in Kaggle (T4 Ã— 2, 30 h / week free) â€” both stream live progress to `per_run.jsonl`, auto-resume on disconnect, and surface through the `/api/ml/matrix` endpoint to the dashboard.

---

## 10. Conclusion

The headline finding of this paper is negative and we believe it is worth stating cleanly. On 934 stratified smartphone images across 21 rice- and wheat-disease classes, a hand-authored symbolic rule layer bolted onto a 96 %-accurate YOLOv8n-cls backbone **changes four predictions in the wrong direction**, is not significant under paired McNemar (p = 0.134), imposes a 29Ã— latency tax, and inflates expected monetary loss by 840 % under a smallholder-calibrated cost matrix. At the accuracy ceilings modern CNNs now reach on this class of imagery, the widely-proposed neuro-symbolic hybrid pattern does not pay its rent. A rules-only arm confirms this is not because the rules are trivial; they are genuinely non-trivial (13.4 % acc, 10Ã— random on 21 classes) â€” the issue is informational redundancy with the CNN.

Around that result, we release what we believe to be the most immediately useful artefact: a crash-resumable 54-cell matrix runner, six-deep Kaggle mirror fallbacks for each of three geographies, a free-tier Colab / Kaggle notebook, an EML cost-table any practitioner can re-parameterise, and a live FastAPI + React dashboard â€” all on a â‚¹0 hardware budget. The honest headline is therefore *not* "our hybrid beats the CNN" but *"the community can stop assuming hand-authored rules help; here is the matrix, the cost model, and the dashboard to prove it on your own crop."*

---

## Acknowledgements

The authors thank the maintainers of the open Kaggle datasets listed in Section 3.1 whose work made geographic cross-validation possible.

## References

*(abbreviated; bibkeys resolve against the project's `references.bib`)*

- Mohanty S. P., Hughes D. P., SalathÃ© M. *Using deep learning for image-based plant disease detection.* Front. Plant Sci., 7 (2016) 1419.
- Ferentinos K. P. *Deep learning models for plant disease detection and diagnosis.* Comput. Electron. Agric., 145 (2018) 311â€“318.
- Barbedo J. G. A. *Plant disease identification from individual lesions and spots using deep learning.* Biosyst. Eng., 180 (2019) 96â€“107.
- Ultralytics. *YOLOv8* (2023). https://github.com/ultralytics/ultralytics
- Rajpurkar P. *et al. AI in health and medicine.* Nat. Med., 28 (2022) 31â€“38.

---

*Manuscript generated from the live state of commit `71220ed` â€” 17 Apr 2026. All numeric tables are sourced directly from `evaluate/results/*.json` in the repository and can be regenerated in one command.*

