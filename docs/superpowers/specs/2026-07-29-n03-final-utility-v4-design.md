# N03 FINAL UTILITY V4 Release Design

## Objective

Replace the obsolete `N03_XF12_LCv3_ET_parent_supported` Docker release with
the frozen final candidate `N03_FINAL_UTILITY_V4`, while preserving a complete
public training/inference workflow and enforcing exact external equivalence
before publication.

## Frozen identity

- Canonical candidate: `N03_FINAL_UTILITY_V4`
- Evaluated alias: `N03_ET_utility_v4_gate_MRIpatch_confirm`
- Implementation alias: `N03_ET_add_utility_v4_three_state_gate`
- Frozen reference ZIP SHA256:
  `dc1b2a6e25f1569ec68240e996b8d772e943b7fb086fff34e790d8676d53c735`
- Reference contents: 179 flat, CRC-valid NIfTI files

The reference ZIP is an external test oracle. It must never be copied into the
Docker build context or image.

## Runtime algorithm

The image accepts four aligned BraTS MRI NIfTI volumes per case from read-only
`/input` and writes one flat segmentation NIfTI per case to `/output`.

1. Run five `checkpoint_best.pth` folds for XL, M, and FT.
2. Construct the XF12 structured-probability V2-strict anchor.
3. Construct the N03 ET parent-supported baseline.
4. Consider only disconnected ET components not kept by the baseline.
5. Score each component with the frozen LCv2, RGv3-ET, utility-v4 existence,
   and utility-v4 geometry-safe models.
6. Accept only when LCv2, existence, and geometry-safe probabilities are each
   at least `0.75`.
7. Apply accepted ET components add-only, preserve the N03 anchor, and enforce
   `ET subset TC subset WT`.

MRI patch verification, temporal confirmation, RC rescue/veto, RV2, EV6, a
second global V2 pass, historical probability maps, and case-specific voxel
patches are excluded.

## Reproducibility architecture

The public repository contains:

- the pinned upstream nnU-Net source at commit
  `86606c53ef9f556d6f024a304b52a48378453641`;
- the project-specific ResEncM trainers;
- dataset conversion, planning, preprocessing, five-fold training, OOF gate
  training, inference, Docker asset export, image build, and external
  verification code;
- ordered shell entrypoints matching the original remote workflow.

Private checkpoints and learned binaries remain outside Git. Asset preparation
copies them into a private build context only after validating the frozen
SHA256 inventory.

## Ordered shell workflow

The public top-level workflow is divided into independently restartable scripts:

1. `scripts/01_prepare_dataset.sh`
2. `scripts/02_train_models.sh`
3. `scripts/03_train_learned_gates.sh`
4. `scripts/04_export_inference_assets.sh`
5. `scripts/05_build_final_image.sh`
6. `scripts/06_verify_final_image.sh`
7. `scripts/07_release_final_image.sh`

Each script is fail-fast, accepts paths through environment variables or
arguments, and records hashes/manifests rather than embedding private paths in
the repository.

## Release gates

Publication is fail-closed. A Docker archive is releasable only when:

- the repository tests, compilation, shell syntax checks, privacy scan, and
  asset inventory checks pass;
- two fresh image executions produce identical outputs;
- the image output case set equals the 179-case reference set;
- every output matches the reference voxel-for-voxel;
- shape, affine, spacing, dtype, legal labels, and aggregate voxel counts match;
- the reference ZIP is absent from the build context and image;
- the Docker archive and verification reports have SHA256 manifests.

If fresh probabilities reproduce the known 188-voxel drift, the build remains
an unreleased diagnostic artifact. The verifier reports the first divergent
layer; it does not repair individual cases.

## Remote build boundary

The restored remote host supplies the original checkpoints, learned gate
assets, raw 179-case inputs, and GPU runtime. The local E-drive stores the
source worktree, final Docker archive, and provenance reports. The remote
machine must not be stopped until the Docker archive and all release evidence
are copied locally and their hashes are independently rechecked.

