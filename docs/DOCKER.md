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
    rgv3_et/models.joblib
    utility_v4/existence_model.joblib
    utility_v4/geometry_model.joblib
    utility_v4/feature_names.json
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
  --rgv3-root /path/to/region_component_gate_v3 \
  --utility-v4-root /path/to/N03_ET_add_utility_v4 \
  --source-root inference \
  --output work/source_inventory.json

python docker/prepare_assets.py \
  --inventory work/source_inventory.json \
  --output assets
```

The nnU-Net source tree used by the trained checkpoints is pinned at
`third_party/nnUNet`. It remains subject to its upstream license.

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
5. run the complete 179-case input twice from empty output directories;
6. compare both runs to the external frozen ZIP for exact case set, arrays,
   shape, affine, spacing, dtype, labels, and aggregate voxel counts;
7. require zero changed voxels and exact run-to-run repeatability;
8. record the image digest and TAR SHA256 only after that gate passes.

Commands:

```bash
python -m pytest -q
python scripts/verify_release.py .
docker load -i artifacts/brats-mets-n03-final.docker.tar
docker run --rm --gpus all --shm-size=16g \
  -v /path/to/input:/input:ro \
  -v /path/to/empty-output:/output \
  brats-mets-n03:final

bash scripts/06_verify_final_image.sh \
  /path/to/raw-179-input \
  /path/to/external-frozen-N03_FINAL_UTILITY_V4.zip

bash scripts/07_release_final_image.sh
```

The frozen ZIP is an external oracle. It is neither copied into the Docker
build context nor used during inference. Any nonzero voxel difference blocks
release; there is no case-specific patching fallback.

## Challenge submission package

Submit the Docker image through the challenge's current official mechanism.
Use the public GitHub repository URL as the source-code link where requested.
Do not upload training data, hidden-test labels, historical predictions, or
credentials. Keep the local TAR backup until the challenge result and any
artifact review are complete.
