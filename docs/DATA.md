# Data preparation

## Expected source data

Use the official BraTS-METS training and validation distributions under their
applicable data-use terms. Do not commit them to this repository.

Each case must contain the four modalities:

| nnU-Net channel | BraTS modality |
|---:|---|
| `0000` | T1 contrast-enhanced (`t1c`) |
| `0001` | T1 native (`t1n`) |
| `0002` | T2 FLAIR (`t2f`) |
| `0003` | T2 weighted (`t2w`) |

Training labels use `0/1/2/3/4` for background, NETC, SNFH, ET, and RC.
Dataset501 is configured as a region task:

- WT: labels `1,2,3`
- TC: labels `1,3`
- ET: label `3`
- RC: label `4`
- reconstruction order: `2,1,3,4`

## Convert to Dataset501

```bash
export nnUNet_raw=/path/to/nnUNet_raw

python preprocessing/prepare_dataset501.py \
  --train-dir /path/to/BraTS-training \
  --valid-dir /path/to/BraTS-validation \
  --nnunet-raw "$nnUNet_raw"
```

The converter checks case completeness, modality identity, NIfTI geometry,
label domain, duplicate IDs, and corrected-label handling before writing:

```text
Dataset501_BraTS2025MET/
  dataset.json
  imagesTr/
  labelsTr/
  imagesTs/
```

Validation cases have no public labels and are therefore written as `imagesTs`.

## Plan and preprocess

```bash
export nnUNet_preprocessed=/path/to/nnUNet_preprocessed
export nnUNet_results=/path/to/nnUNet_results
bash preprocessing/plan_and_preprocess.sh
```

The script creates the frozen M and XL plan identifiers and invokes nnU-Net's
dataset integrity check. Inspect the produced plan JSONs before training and
archive them with each experiment.
