# Leakage Investigation Report

**Generated:** 2026-04-20
**Scripts:** `tools/research_validation/investigate_leakage.py`, `tools/research_validation/deep_leakage_analysis.py`

## Background

The external audit flagged a potential train/test leakage risk because `scripts/make_splits.py` shuffles all files per class without grouping augmented variants (`aug_*.jpg`) by their base image ID. If multiple augmented copies of the same source image existed across splits, the model could memorize source-image features during training and exploit them during testing.

## Splitting Logic

`scripts/make_splits.py` (lines 48-59):
1. Iterates sorted class directories
2. Collects all files per class
3. Shuffles with `random.Random(seed=42)`
4. Partitions 70/15/15 into train/val/test
5. **No grouping by base image ID**

## Augmentation Filename Analysis

The `aug_*` naming convention is `aug_{augmentation_index}_{base_id}.jpg`.

### Key Finding: Each aug file has a unique base ID

| Metric | Value |
|---|---|
| Total aug base IDs across all splits | 1,395 |
| Base IDs with >1 variant | **0** |
| Base IDs with variants in different splits | **0** |

**Every `aug_*` file has a unique base ID.** There are no sibling augmentations — each augmented image is a one-to-one transformation of a unique source, not a one-to-many expansion. Therefore, the splitting script's lack of group-awareness is **irrelevant** for this dataset.

### Per-Split Augmentation Counts

| Split | aug_* files | Total files | Aug ratio |
|---|---:|---:|---|
| Train | 988 | 4,364 | 22.6% |
| Val | 209 | 935 | 22.4% |
| Test | 198 | 935 | 21.2% |

The aug ratio is uniform across splits, consistent with random assignment.

### Cross-Split Overlap

| Comparison | Overlapping base IDs |
|---|---:|
| Train ∩ Test | **0** |
| Train ∩ Val | **0** |
| Val ∩ Test | **0** |

## Leakage Risk Assessment

**Risk level: NONE**

The leakage concern was reasonable given the split script's design, but the actual data does not exhibit the vulnerability. Each augmented file maps to a unique source image, so no source-image information can leak across splits.

## Does `make_splits.py` Need Patching?

**For the current dataset: No.** The script works correctly because each aug file has a unique base.

**For future datasets: Yes.** If someone later adds multi-variant augmentations (e.g., `aug_0_100.jpg`, `aug_1_100.jpg`, `aug_2_100.jpg` all from the same source), the current script would introduce leakage. A defensive patch should group-split by base ID.

## Impact on Main Experiment

**The main ablation results (A/B/C predictions, McNemar test, EML analysis) are NOT compromised by leakage.** They can be cited as trustworthy.

## Recommendation for Manuscript

The manuscript should:
1. **Not claim there is leakage** (there isn't)
2. **Acknowledge the defensive weakness** in the limitations section: the split script lacks group-aware splitting, which is safe for this dataset but would be risky for future datasets with multi-variant augmentation
3. **Not require a re-run** of experiments

## Evidence Files

- `evaluate/data_split_manifest.json` — per-class file lists for train/val/test
- `tools/research_validation/deep_leakage_analysis.py` — reproducible overlap analysis
- `tools/research_validation/investigate_leakage.py` — filename pattern analysis
