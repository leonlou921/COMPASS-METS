# Data preparation

## Input layout

Obtain the official BraTS-METS data under its data-use terms and keep it
outside this repository. Each case must contain four aligned MRI modalities:

| nnU-Net channel | Modality |
|---:|---|
| `0000` | T1 contrast-enhanced (`t1c`) |
| `0001` | T1 native (`t1n`) |
| `0002` | T2 FLAIR (`t2f`) |
| `0003` | T2 weighted (`t2w`) |

Training labels use `0,1,2,3,4` for background, NETC, SNFH, ET, and RC.
Dataset501 is a region task:

- WT: `1,2,3`
- TC: `1,3`
- ET: `3`
- RC: `4`
- region reconstruction order: `2,1,3,4`

## Build Dataset501

```bash
export nnUNet_raw=/path/to/nnUNet_raw
export BRATS_METS_TRAIN_DIR=/path/to/training
export BRATS_METS_VALID_DIR=/path/to/validation
# Optional:
export BRATS_METS_CORRECTED_LABELS_DIR=/path/to/corrected-labels

bash scripts/01_prepare_dataset.sh
```

The converter in `compass_mets.data.prepare_dataset501` checks case
completeness, modality identity, duplicate IDs, NIfTI geometry, label domain,
and corrected-label replacement before writing
`Dataset501_BraTS2025MET`.

## Plan and preprocess

```bash
export nnUNet_preprocessed=/path/to/nnUNet_preprocessed
export nnUNet_results=/path/to/nnUNet_results
bash scripts/02_plan_and_preprocess.sh
```

This produces both frozen plans:

- `nnUNetResEncUNetMPlans`
- `nnUNetResEncUNetXL30GBPlans`

The preprocessing command includes nnU-Net dataset-integrity verification.
