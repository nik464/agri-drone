# Data availability

This project uses four external datasets. None are redistributed with this
repository; each must be obtained from its original provider under its own
licence.

| Name | Role | URL | Licence | Local path (default) |
|---|---|---|---|---|
| PlantVillage | Train / val / test for Configs A, B, C (v3) and all matrix cells | <https://github.com/spMohanty/PlantVillage-Dataset> | CC-BY-SA-4.0 | `datasets/externals/plantvillage/` |
| PDT | Cross-dataset drone-altitude evaluation (§5.4) | <https://github.com/Kaka-Shi/PDT> | see upstream repo | `datasets/externals/PDT_datasets/` |
| UCI Rice Leaf | Rice-crop cross-dataset generalisation | <https://archive.ics.uci.edu/ml/datasets/Rice+Leaf+Diseases> | open (UCI) | `datasets/externals/riceleaf/` |
| Kaggle Rice Pest | Rice pest generalisation | <https://www.kaggle.com/datasets/shrupyag001/rice-leaf-disease-images> | Kaggle terms | `datasets/externals/ricepest/` |

## Domain caveat (important for reviewers)

All four datasets contain **curated close-up leaf photographs under
controlled or semi-controlled conditions**. None contain drone-altitude
aerial imagery of farm canopies. Section 3.1 of the v4 paper documents this
constraint explicitly. The PDT dataset is the closest approximation to
drone-altitude imagery and is used only for the cross-dataset sanity check
reported in §5.4 — not for training.

## Reproducing the exact splits

```bash
python scripts/make_splits.py \
    --input-dir datasets/externals/plantvillage/ \
    --out-dir   datasets/splits/plantvillage/ \
    --seed      42 \
    --ratios    0.7 0.15 0.15
```

The resulting `splits_manifest.json` records both the RNG seed and a
SHA-256 of the sorted file list so silent drift is detectable.

## Downloading

See `scripts/download_data.py`. The script prints URL + expected local path
for each dataset. Automatic download is intentionally not implemented,
because the upstream hosts each have their own access terms.

```bash
python scripts/download_data.py --list
python scripts/download_data.py --check
```
