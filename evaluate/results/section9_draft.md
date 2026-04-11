# Section 9: Experimental Results

## 9.1 Dataset and Protocol

We evaluate on a 21-class Indian crop disease dataset comprising 14 wheat
and 5 rice pathologies plus 2 healthy classes.  A stratified 70/15/15 split
yields **935** test images (~45 per class; wheat\_stem\_fly: 35).
All configurations share the same test fold (seed=42).

## 9.2 Ablation Study (Table 2)

\input{table2_ablation}

Config A (YOLOv8n-cls standalone) achieves **0.9618** macro-F1
at 13.5 ms/image.  Adding the rule engine and ensemble
voter (Config B) drops macro-F1 to **0.6083**
($\Delta$ = -0.3535).

Risk-Weighted Accuracy confirms the pattern: RWA$_A$ = 0.9610,
RWA$_B$ = 0.6689.  The safety gap (RWA $-$ Accuracy) shifts
from +0.0004 (A) to -0.0672 (B), indicating
the rule engine introduces disproportionately more errors on high-severity diseases.

**Per-class analysis.**  The three largest F1 regressions under Config B are:
- `wheat\_root\_rot`: $\Delta$F1 = -0.7663
- `healthy\_wheat`: $\Delta$F1 = -0.7334
- `wheat\_tan\_spot`: $\Delta$F1 = -0.6928

The rule engine's color and spatial heuristics over-correct confident YOLO
predictions in wheat diseases characterised by diffuse brown lesions,
where multiple KB profiles share overlapping HSV signatures.

## 9.3 Threshold Sensitivity (Table 3)

\input{table3_sensitivity}

A $5 \times 5 \times 5$ grid search over the three hardcoded rule-engine
parameters ($\alpha_{color}$, $w_{stripe}$, $\tau_{yolo}$) evaluates
125 configurations on 935 validation images.

The macro-F1 standard deviation across all configurations is
**0.0087**, with a total range of
[0.2537, 0.2878]. The current defaults
($\alpha_{color}$=20,
$w_{stripe}$=0.5,
$\tau_{yolo}$=0.85)
achieve F1=0.2813, within 0.0065
of the grid optimum (0.2878).

**Interpretation.**  The low $\sigma$ confirms that the rule engine's
performance is dominated by the structural mismatch between
heuristic rules and the learned YOLOv8 feature space, not by any
single threshold choice.  The colour scale shows the strongest
marginal effect (higher $\alpha_{color}$ $\rightarrow$ slightly better F1),
while stripe weight and YOLO override threshold have negligible impact.

## 9.4 Expected Monetary Loss (Table 4)

\input{table4_eml}

Using field-verified cost data from Indian agricultural extension reports,
we compute the Expected Monetary Loss under both configurations.

Config A achieves a total EML of **₹294**/sample,
while Config B produces **₹2,769**/sample
($\Delta$ = ₹2,475, +840.8%).

The higher EML under Config B reflects its elevated miss rates on
high-cost diseases.  For critical diseases alone, Config A incurs
₹154 vs Config B's ₹1,305.

## 9.5 Discussion

These experiments reveal a counter-intuitive finding: the handcrafted rule
engine, despite encoding agronomic domain knowledge, **degrades**
classification performance when cascaded after a well-trained YOLOv8n
classifier.  Three factors explain this:

1. **Feature-space mismatch**: The rule engine operates on raw HSV ratios
   and morphological features, while YOLOv8 learns discriminative features
   in a 512-d embedding space.  The two representations are not calibrated.

2. **Threshold sensitivity is a red herring**: The $\sigma_{F1}$ = 0.0087
   across 125 configurations shows the problem is structural, not parametric.

3. **Conflict resolution bias**: When YOLO and rules disagree, the current
   resolution logic favours rules for any rule\_score > 0.3, but many
   diseases produce spurious matches through shared colour signatures.

**Implication for C1 (Safety-Aware Ensemble).**  These results motivate
replacing the heuristic conflict resolution with a learned fusion layer
that can weight YOLO confidence against calibrated rule evidence.
The -0.0672 safety gap under Config B, versus
+0.0004 under Config A, demonstrates that naive
rule injection amplifies errors on precisely the diseases where
misclassification costs are highest.
