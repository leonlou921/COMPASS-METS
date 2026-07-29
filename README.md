# COMPASS-METS

COMPASS-METS is the source release for our BraTS-METS segmentation system. It
contains the complete data-preparation, training, out-of-fold learned-gate,
three-source inference, XF12 fusion, N03 parent-supported, and final
`N03_FINAL_UTILITY_V4` postprocessing code.

The repository intentionally contains source code only. Challenge images,
labels, checkpoints, learned model files, probability maps, and predictions
must remain outside Git.

## Final method

The final system uses two primary five-fold nnU-Net models and one directly
required fine-tuned donor:

| Role | Trainer | Plans | Checkpoint |
|---|---|---|---|
| ResEncM | `nnUNetTrainer` | `nnUNetResEncUNetMPlans` | `checkpoint_best.pth` |
| ResEncXL | `nnUNetTrainer` | `nnUNetResEncUNetXL30GBPlans` | `checkpoint_best.pth` |
| FT donor | `nnUNetTrainer_ResEncM_DiceCEFocalTverskyFT` | `nnUNetResEncUNetMPlans` | `checkpoint_best.pth` |

Each source ensembles folds 0–4 and exports four region probabilities in
`WT, TC, ET, RC` order. The final chain is:

```text
ResEncXL + ResEncM + FT probabilities
  -> XF12 structured-probability/V2-strict anchor
  -> LCv1 + LCv2 component scoring
  -> N03 ET-only parent-supported additions
  -> RGv3-ET and UTILITY_V4 three-state gate
  -> add-only ET update preserving the N03 anchor and RC priority
```

The frozen machine-readable specifications are:

- [`configs/trainers/resencm.json`](configs/trainers/resencm.json)
- [`configs/trainers/resencxl.json`](configs/trainers/resencxl.json)
- [`configs/trainers/focal_tversky.json`](configs/trainers/focal_tversky.json)
- [`configs/fusion/xf12.json`](configs/fusion/xf12.json)
- [`configs/final/n03_utility_v4.json`](configs/final/n03_utility_v4.json)

## Installation

Python 3.10 is recommended. The exact upstream nnU-Net 2.6.2 source is
vendored under `third_party/nnUNet` at commit
`86606c53ef9f556d6f024a304b52a48378453641`.

```bash
conda env create -f environment.yml
conda activate compass-mets
python -m pip install -e third_party/nnUNet
python -m pip install -e .
```

Set the normal nnU-Net storage roots:

```bash
export nnUNet_raw=/path/to/nnUNet_raw
export nnUNet_preprocessed=/path/to/nnUNet_preprocessed
export nnUNet_results=/path/to/nnUNet_results
```

## Ordered workflow

Every public entry point is a Bash or Python script:

```bash
bash scripts/00_setup_environment.sh
bash scripts/01_prepare_dataset.sh
bash scripts/02_plan_and_preprocess.sh
bash scripts/03_train_resencm_5fold.sh
bash scripts/04_train_resencxl_5fold.sh
bash scripts/05_train_small_lesion_ft_5fold.sh
bash scripts/06_generate_oof_probabilities.sh
bash scripts/07_train_learned_gates.sh
bash scripts/08_predict_test_probabilities.sh
bash scripts/09_build_n03_utility_v4.sh
```

The scripts are fail-fast and require their data/model paths through
environment variables. See:

- [Data preparation](docs/DATA.md)
- [Training and learned gates](docs/TRAINING.md)
- [Inference and final postprocessing](docs/INFERENCE.md)
- [Pipeline map](docs/PIPELINE.md)
- [Configuration reference](docs/CONFIGURATION_REFERENCE.md)
- [Reproducibility boundary](docs/REPRODUCIBILITY.md)

## Verification

Source-release checks do not require challenge test predictions:

```bash
python -m pytest -q
python -m compileall -q compass_mets third_party/nnUNet/nnunetv2
python scripts/verify_release.py . --output release_audit.json
```

## License and citation

Project code is Apache-2.0. Vendored dependencies retain their upstream
licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Cite this
software using [CITATION.cff](CITATION.cff).
