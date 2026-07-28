# Docker build and submission

## Private build assets

The public repository intentionally excludes trained weights. Before building,
provide:

```text
assets/
  nnUNet_results/Dataset501_BraTS2025MET/
    nnUNetTrainer__nnUNetResEncUNetXL30GBPlans__3d_fullres/
    nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/
    nnUNetTrainer_ResEncM_DiceCEFocalTverskyFT__nnUNetResEncUNetMPlans__3d_fullres/
  learned_models/
    lcv1_case/models.joblib
    lcv2_component/models.joblib
  provenance/
```

Each nnU-Net model directory must contain `plans.json`, `dataset.json`, and
folds `0..4` with `checkpoint_best.pth`. Use
`n03_docker.asset_inventory.build_source_inventory` followed by
`docker/prepare_assets.py` to hash, validate, and slim the private assets.
The slimming step preserves the network tensor hash and only removes
training-only checkpoint state.

Example:

```bash
python docker/build_asset_inventory.py \
  --xl-root /path/to/XL-run \
  --m-root /path/to/M-run \
  --ft-root /path/to/FT-run \
  --lcv1-root /path/to/learned_component_gate_v1 \
  --lcv2-root /path/to/learned_component_gate_v2 \
  --source-root inference \
  --output work/source_inventory.json

python docker/prepare_assets.py \
  --inventory work/source_inventory.json \
  --output assets
```

The nnU-Net source tree used by the trained checkpoints must be staged at
`vendor/nnUNet`. It remains subject to its upstream license.

## Build

With Docker:

```bash
docker build -f docker/Dockerfile -t brats-mets-n03:final .
```

On a rootless BuildKit host:

```bash
bash scripts/build_docker_archive.sh
```

The latter exports a Docker-load-compatible TAR and adjacent SHA256 file.

## Validate before submission

1. run the source test suite;
2. verify the release scan;
3. build the image without network access at runtime;
4. run a real one-case smoke test;
5. run the complete validation set if permitted;
6. check output case count, flat layout, geometry, dtype, labels, and
   aggregate voxel count;
7. record the image digest and TAR SHA256;
8. load the TAR into a clean Docker environment and repeat the smoke test.

Commands:

```bash
python -m pytest -q
python scripts/verify_release.py .
docker load -i artifacts/brats-mets-n03-final.docker.tar
docker run --rm --gpus all --shm-size=16g \
  -v /path/to/input:/input:ro \
  -v /path/to/empty-output:/output \
  brats-mets-n03:final
```

## Challenge submission package

Submit the Docker image through the challenge's current official mechanism.
Use the public GitHub repository URL as the source-code link where requested.
Do not upload training data, hidden-test labels, historical predictions, or
credentials. Keep the local TAR backup until the challenge result and any
artifact review are complete.
