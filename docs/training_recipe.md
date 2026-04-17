# Shared training recipe (v1) — used by every backbone in the v4 matrix

All cells in `configs/matrix/full.yaml` are trained with **exactly the same**
recipe below. This is the fix for the suspiciously low 76.15% EfficientNet-B0
baseline reported in v3: in v3, different architectures received different
augmentation strength, LR, and epoch budgets, so architecture-vs-recipe effects
were entangled. v4 disentangles them.

> Versioning tag: `docs/training_recipe.md@v1`.
> The matrix runner writes this tag into every `per_run.jsonl` record; changing
> the recipe requires bumping the tag and updating `docs/results_schema.md`.

## 1. Optimizer and schedule

| Item | Value |
|---|---|
| Optimizer          | AdamW |
| Initial LR         | 1.25e-3 |
| Weight decay       | 1e-4 |
| LR schedule        | Cosine annealing to 1e-5 |
| Warmup             | 3 epochs, linear from 1e-6 |
| Epoch budget       | 50 (with early stopping, patience 10) |
| Label smoothing    | 0.10 |
| Batch size         | 32 |
| Gradient clip      | 1.0 (max L2 norm) |
| Mixed precision    | fp16 (autocast) on CUDA, fp32 otherwise |

## 2. Augmentation — `standard_v1`

Applied identically to every backbone (224×224 input).

```
Resize (224, 224)
RandomHorizontalFlip(p=0.5)
ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1))
ToTensor()
Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

Val/test transforms: `Resize(224, 224) → ToTensor → Normalize` only (no aug).

## 3. Pretraining

All backbones initialised from their canonical ImageNet-1k weights:

| Backbone | Checkpoint |
|---|---|
| yolov8n-cls | Ultralytics ImageNet-pretrained |
| yolov8s-cls | Ultralytics ImageNet-pretrained |
| efficientnet_b0 | `torchvision EfficientNet_B0_Weights.IMAGENET1K_V1` |
| convnext_tiny | `torchvision ConvNeXt_Tiny_Weights.IMAGENET1K_V1` |
| mobilenetv3_small | `torchvision MobileNet_V3_Small_Weights.IMAGENET1K_V1` |
| vit_b_16 | `torchvision ViT_B_16_Weights.IMAGENET1K_V1` |

The torchvision backbones replace only the final classifier head with
`Dropout(0.2) → Linear(in_features, n_classes)`.

## 4. Splits and folds

- Primary dataset: 70/15/15 stratified, seed 42 (bit-exactly reproduced by
  `scripts/make_splits.py`).
- 5-fold stratified CV uses seeds `[42, 43, 44, 45, 46]`. Each fold reuses
  the same 85% train+val pool and rotates the 15% test.

## 5. Determinism

- `torch.manual_seed`, `numpy.random.seed`, and `random.seed` all set to
  `seed` from the cell spec.
- cuDNN deterministic mode: on (`torch.backends.cudnn.deterministic = True`,
  `benchmark = False`).
- DataLoader `num_workers=0` for full determinism in CI; raise to 4 for
  GPU runs with a `generator` seeded by the cell seed.

## 6. Per-backbone fairness checks

After every training run, `evaluate/matrix/audit_baseline.py` logs:

- Convergence curves (train/val loss, train/val acc) to
  `evaluate/results/v2/baseline_audit/<run_id>/curves.png`.
- Final-epoch gap (val acc − train acc) to flag overfitting/underfitting.
- LR trace to verify the cosine schedule executed as planned.

If any backbone's final-epoch gap is > 15 pp, the cell is flagged and
excluded from the headline v4 comparison (it is still reported in the
supplementary table with a `needs_recipe_tuning` annotation).

## 7. What v4 does NOT re-run

- The original YOLOv8n-cls training used for `models/india_agri_cls_21class_backup.pt`
  is preserved and still produces 96.15% accuracy on the frozen test set. v4
  **does not** retrain this model; it only re-evaluates it under the new matrix.
  Any future retraining lives under `models/new/` (never overwriting existing
  weights) with a new filename.
