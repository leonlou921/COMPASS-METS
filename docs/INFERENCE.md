# Inference

## Input contract

Each case supplies four geometrically aligned NIfTI files in either form:

```text
CASE-t1c.nii.gz     CASE_0000.nii.gz
CASE-t1n.nii.gz     CASE_0001.nii.gz
CASE-t2f.nii.gz     CASE_0002.nii.gz
CASE-t2w.nii.gz     CASE_0003.nii.gz
```

Do not mix naming styles within a case.

## Step 1: generate M/XL/FT probabilities

```bash
export BRATS_METS_TEST_INPUT=/path/to/test-images
export COMPASS_PROBABILITY_ROOT=/path/to/empty-probability-root
export COMPASS_WORK_ROOT=/path/to/work
export nnUNet_results=/path/to/nnUNet_results

bash scripts/08_predict_test_probabilities.sh
```

The script runs ResEncXL, ResEncM, and FT sequentially with folds 0–4,
`checkpoint_best.pth`, test-time augmentation enabled, and
`--save_probabilities`. It validates a finite four-channel NPZ for every input
case under `XL/`, `M/`, and `FT/`.

## Step 2: build final N03 UTILITY_V4 labels

```bash
export COMPASS_OUTPUT_ROOT=/path/to/empty-output
export COMPASS_LCV1_BUNDLE=/path/to/lcv1/models.joblib
export COMPASS_LCV2_BUNDLE=/path/to/lcv2/models.joblib
export COMPASS_RGV3_BUNDLE=/path/to/rgv3/ET/models.joblib
export COMPASS_UTILITY_EXISTENCE_MODEL=/path/to/existence_model.joblib
export COMPASS_UTILITY_GEOMETRY_MODEL=/path/to/geometry_model.joblib
export COMPASS_UTILITY_FEATURE_NAMES=/path/to/feature_names.json

bash scripts/09_build_n03_utility_v4.sh
```

The final chain constructs the frozen XF12 anchor, applies the N03 ET-only
parent-supported rule, and considers only disconnected ET components from the
LCv2 structured union. UTILITY_V4 uses:

- proposal threshold `0.25`;
- RGv3-ET cutoff `0.7702616034384248`;
- accept when LCv2, existence, and geometry scores are all at least `0.75`;
- reject when either utility score is below `0.50`;
- otherwise abstain.

Accepted components are added as ET without deleting anchor voxels. RC retains
priority, and the earlier global component/TC-boundary/strict-RC chain is not
rerun after the addition.

## Output contract

The output directory contains one root-level `CASE.nii.gz` per input case:

- reference shape, affine, and spacing are preserved;
- dtype is `uint8`;
- labels are restricted to `0,1,2,3,4`;
- no model probabilities or auxiliary reports are written into the output.
