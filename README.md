# BraTS-METS 2026 N03

Reproducible source release for the final
`N03_XF12_LCv3_ET_parent_supported` inference candidate used in the
BraTS-METS challenge.

The repository covers the complete method:

1. convert the four BraTS MRI modalities and labels to nnU-Net Dataset501;
2. plan and preprocess the dataset;
3. train five folds of ResEncXL, ResEncM, and the fold-matched
   ResEncM DiceCE/Focal-Tversky model;
4. export out-of-fold probabilities and train the learned case/component
   gates;
5. run the frozen XL/M/FT soft-probability pipeline;
6. construct the XF12 anchor and add only ET components satisfying the
   learned cutoff and ET/TC/WT parent-support rule;
7. validate flat BraTS label outputs and build the offline Docker image.

No challenge data, labels, checkpoints, learned-model binaries, probability
maps, predictions, platform scores, credentials, or Docker archive are stored
in Git. Those assets must be obtained or trained separately.

## Frozen final candidate

| Item | Value |
|---|---|
| Candidate | `N03_XF12_LCv3_ET_parent_supported` |
| Proposal threshold | `0.25` |
| ET LCv2 cutoff | `0.5497123599` |
| Parent support | at least 2 of XL/M/FT, independently for ET, TC, and WT |
| Update policy | add-only ET; preserve XF12 anchor; enforce hierarchy |
| Model ensemble | five `checkpoint_best.pth` folds per XL, M, and FT |
| Runtime input | four aligned NIfTI volumes per case |
| Runtime output | one flat `CASE.nii.gz`, labels `0,1,2,3,4` |

The machine-readable specifications are
[`configs/models/final_models.json`](configs/models/final_models.json) and
[`configs/n03/final.json`](configs/n03/final.json).

## Quick start

Prepare Dataset501:

```bash
export nnUNet_raw=/path/to/nnUNet_raw
export nnUNet_preprocessed=/path/to/nnUNet_preprocessed
export nnUNet_results=/path/to/nnUNet_results

python preprocessing/prepare_dataset501.py \
  --train-dir /path/to/BraTS-training \
  --valid-dir /path/to/BraTS-validation \
  --nnunet-raw "$nnUNet_raw"

bash preprocessing/plan_and_preprocess.sh
```

Train and validate all folds:

```bash
python training/run_nnunet.py \
  --registry configs/models/final_models.json \
  --model m --action train

python training/run_nnunet.py \
  --registry configs/models/final_models.json \
  --model ft --action train

python training/run_nnunet.py \
  --registry configs/models/final_models.json \
  --model xl --action train

for model in m ft xl; do
  python training/run_nnunet.py \
    --registry configs/models/final_models.json \
    --model "$model" --action validate
done
```

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
can reproduce the candidate when supplied with the same trained checkpoints,
learned gate bundles, nnU-Net revision, and input images. Retraining may vary
numerically because of GPU kernels and stochastic optimization; it is therefore
not claimed to regenerate byte-identical weights.

The preserved challenge Docker archive is identified in
[`provenance/frozen_image.json`](provenance/frozen_image.json). It is excluded
from Git because of its size and embedded private weights.

## License and citation

Code in this repository is released under Apache-2.0. Third-party components
retain their original licenses. Please cite the repository using
[`CITATION.cff`](CITATION.cff).
