# BraTS-METS 2026 MicroBT

Reproducible source release for the final
`N03_FINAL_UTILITY_V4` inference candidate used in the
BraTS-METS challenge.

The repository covers the complete method:

1. convert the four BraTS MRI modalities and labels to nnU-Net Dataset501;
2. plan and preprocess the dataset;
3. train five folds of ResEncXL, ResEncM, and the fold-matched
   ResEncM DiceCE/Focal-Tversky model;
4. export out-of-fold probabilities and train the learned case/component
   gates;
5. run the frozen XL/M/FT soft-probability pipeline;
6. construct the XF12/V2-strict anchor and the parent-supported N03 baseline;
7. score disconnected ET additions using LCv2, RGv3-ET, utility-v4 existence,
   and utility-v4 geometry-safety models;
8. preserve the anchor and RC priority while applying accepted ET additions;
9. validate flat BraTS label outputs and build the offline Docker image.

No challenge data, labels, checkpoints, learned-model binaries, probability
maps, predictions, platform scores, credentials, or Docker archive are stored
in Git. Those assets must be obtained or trained separately.

## Frozen final candidate

| Item | Value |
|---|---|
| Candidate | `N03_FINAL_UTILITY_V4` |
| Baseline | `N03_XF12_LCv3_ET_parent_supported` |
| Proposal threshold | `0.25` |
| Candidate pool | disconnected ET from the LCv2 structured union |
| RGv3 ET cutoff | `0.7702616034384248` |
| Utility acceptance | LCv2, existence, and geometry scores all `>=0.75` |
| Update policy | add-only ET; preserve N03 anchor and RC; enforce hierarchy |
| Model ensemble | five `checkpoint_best.pth` folds per XL, M, and FT |
| Runtime input | four aligned NIfTI volumes per case |
| Runtime output | one flat `CASE.nii.gz`, labels `0,1,2,3,4` |

The machine-readable specifications are
[`configs/models/final_models.json`](configs/models/final_models.json) and
[`configs/n03/final.json`](configs/n03/final.json).

## Quick start

The repository vendors the pinned upstream nnU-Net source at commit
`86606c53ef9f556d6f024a304b52a48378453641`, together with the four project
Trainer classes. Run the ordered workflow:

```bash
export nnUNet_raw=/path/to/nnUNet_raw
export nnUNet_preprocessed=/path/to/nnUNet_preprocessed
export nnUNet_results=/path/to/nnUNet_results

bash scripts/01_prepare_dataset.sh /path/to/training /path/to/validation
bash scripts/02_train_models.sh
bash scripts/03_train_learned_gates.sh /path/to/lcv1.json /path/to/lcv2.json
bash scripts/04_export_inference_assets.sh
bash scripts/05_build_final_image.sh
bash scripts/06_verify_final_image.sh /path/to/raw-input /path/to/frozen-reference.zip
bash scripts/07_release_final_image.sh
```

Steps `06` and `07` fail closed unless two fresh 179-case runs are
voxel-identical to the frozen external reference. The reference ZIP is never
copied into the image.

Run the frozen container after preparing the private assets:

```bash
docker build -f docker/Dockerfile -t brats-mets-n03:final .
docker run --rm --gpus all --shm-size=16g \
  -v /path/to/input:/input:ro \
  -v /path/to/empty-output:/output \
  brats-mets-n03:final
```

See [docs/DATA.md](docs/DATA.md), [docs/TRAINING.md](docs/TRAINING.md),
[docs/INFERENCE.md](docs/INFERENCE.md), and [docs/DOCKER.md](docs/DOCKER.md)
for the complete ordered workflow.

## Reproducibility boundary

This release freezes the algorithm and runtime contract. The published source
can reproduce the candidate when supplied with the exact hash-locked
checkpoints and learned gate bundles, pinned nnU-Net revision, and input
images. Retraining may vary numerically because of GPU kernels and stochastic
optimization; it is therefore not claimed to regenerate byte-identical
weights.

The final image manifest is written only after external equivalence succeeds.
The Docker archive and frozen prediction ZIP are excluded from Git because
they contain private weights or challenge predictions.

## License and citation

Code in this repository is released under Apache-2.0. Third-party components
retain their original licenses. Please cite the repository using
[`CITATION.cff`](CITATION.cff).
