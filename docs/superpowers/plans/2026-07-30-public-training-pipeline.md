# Public BraTS-METS Training Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a code-only, script-driven reproduction repository for the final ResEncM/ResEncXL lineage, its direct SmallLesionOS/focal-Tversky ablations, and the frozen XF12 → N03 ET parent-supported → UTILITY_V4 output pipeline.

**Architecture:** Vendor the exact nnU-Net source revision and keep the three retained custom trainers inside its normal trainer-discovery namespace. Put challenge-specific Python under one `brats_mets` package, keep scientific settings in versioned JSON/YAML configuration, and expose the full workflow through numbered Bash scripts. The public pipeline stops after writing final segmentation files; data, checkpoints, probabilities, hidden case lists, Docker assets, and submission archives are outside the repository.

**Tech Stack:** Python 3.10, PyTorch, nnU-Net v2.6.2 at commit `86606c53ef9f556d6f024a304b52a48378453641`, NumPy, SciPy, pandas, nibabel/SimpleITK, scikit-learn, LightGBM, Bash, pytest.

## Global Constraints

- Preserve the active image worktree at `E:\brats_challenge\worktrees\brats-mets-MicroBT-n03-utility-v4`; all edits belong only to this branch/worktree.
- Do not publish challenge images, labels, test identifiers, checkpoints, probability maps, learned binary artifacts, frozen ZIP files, or secrets.
- Do not retain Dockerfiles, container helpers, image-release checks, or submission ZIP/CRC/SHA256 workflows.
- Do not include SyntheticTumorAug or broad historical sweeps.
- Keep only ResEncM, ResEncXL, focal-Tversky, SmallLesionOS, ComponentSmallLesionOS, LCv1, LCv2, RGv3, and UTILITY_V4.
- Preserve the frozen final semantics: XF12 soft-source construction, N03 parent-supported ET add-only rescue, then UTILITY_V4 three-state gate.
- UTILITY_V4 must accept only when v2, v4-existence, and v4-geometry scores are all at least `0.75`; reject when either v4 score is below `0.50`; otherwise fall back to the RGv3 ET cutoff `0.7702616034384248`.
- UTILITY_V4 is ET add-only. It must freeze the N03 anchor and RC, maintain `ET ⊆ TC ⊆ WT`, and must not rerun global V2.
- New behavior starts with a failing test, then the smallest implementation, then a focused and full regression run.
- Commit by coherent task. Before push, inspect tracked files and history for credentials, private paths, data, weights, predictions, archives, and oversized artifacts.

---

## Task 1: Freeze the Public Package and Environment Contract

**Files:**

- Modify: `pyproject.toml`
- Create: `environment.yml`
- Create: `requirements/runtime.txt`
- Create: `requirements/training.txt`
- Create: `brats_mets/__init__.py`
- Create: `brats_mets/validation/repository.py`
- Create: `tests/repository/test_public_contract.py`
- Delete: `requirements/inference.txt`

- [ ] **Step 1: Write the failing public-contract tests**

  Test that:

  - the distribution is named `brats-mets-microbt`;
  - `brats_mets` is importable from the repository root;
  - the pinned nnU-Net provenance file names version `2.6.2` and commit `86606c53...`;
  - runtime/training requirements and `environment.yml` exist;
  - repository paths contain no Docker or submission-package entry point.

  Run:

  ```bash
  python -m pytest tests/repository/test_public_contract.py -q
  ```

  Expected: FAIL because the new package contract does not yet exist.

- [ ] **Step 2: Implement the minimum package/environment metadata**

  - Configure setuptools package discovery for `brats_mets`.
  - Keep `third_party/nnUNet` importable by `PYTHONPATH` in shell setup instead of repackaging upstream.
  - Split dependencies into reproducible runtime and training lists.
  - Add a repository validator that exposes reusable path/privacy checks without building archives.

- [ ] **Step 3: Run focused tests and metadata smoke tests**

  ```bash
  python -m pytest tests/repository/test_public_contract.py -q
  python -m compileall -q brats_mets
  python -m pip install --no-deps -e .
  python -c "import brats_mets; print(brats_mets.__version__)"
  ```

- [ ] **Step 4: Commit**

  ```bash
  git add pyproject.toml environment.yml requirements brats_mets tests/repository
  git commit -m "Define public training package contract"
  ```

## Task 2: Remove Docker and Submission-Archive Surfaces

**Files:**

- Delete: `.dockerignore`
- Delete: `docker/`
- Delete: `scripts/05_build_final_image.sh`
- Delete: `scripts/06_verify_final_image.sh`
- Delete: `scripts/07_release_final_image.sh`
- Delete: `scripts/08_package_submission.sh`
- Delete: `scripts/build_docker_archive.sh`
- Delete: `scripts/build_docker_archive_kaniko.sh`
- Delete: `verification/package_submission.py`
- Delete: `verification/run_exported_rootfs.sh`
- Delete: `verification/verify_frozen_equivalence.py`
- Delete: `tests/docker/`
- Delete: `tests/test_frozen_equivalence.py`
- Delete: `tests/test_package_submission.py`
- Delete: `tests/test_release_contract.py`
- Delete: `tests/test_release_manifest.py`
- Delete: `docs/DOCKER.md`
- Delete: `provenance/frozen_image.json`
- Modify: `.gitignore`
- Modify: `brats_mets/validation/repository.py`
- Modify: `tests/repository/test_public_contract.py`

- [ ] **Step 1: Strengthen the failing exclusion test**

  Test the tracked file list and assert that it contains no:

  - `Dockerfile`, `.dockerignore`, or path under `docker/`;
  - image build/release scripts;
  - submission packaging implementation;
  - `.zip`, `.tar`, `.nii`, `.nii.gz`, `.npz`, `.pth`, `.pt`, `.pkl`, or `.joblib` artifact.

- [ ] **Step 2: Remove obsolete files**

  Use `git rm` only on the exact paths listed above. Preserve scientific Python that currently lives under `inference/src/n03_docker` until Task 3 migrates it.

- [ ] **Step 3: Add ignore rules for private/generated artifacts**

  Cover nnU-Net raw/preprocessed/results directories, checkpoints, probability maps, NIfTI outputs, serialized learned models, archives, logs, and local environment files.

- [ ] **Step 4: Verify and commit**

  ```bash
  python -m pytest tests/repository/test_public_contract.py -q
  git ls-files | rg -i "docker|package_submission|\.zip$|\.nii(\.gz)?$|\.npz$|\.pth$|\.pt$|\.joblib$"
  git add -A
  git commit -m "Remove container and submission archive surfaces"
  ```

  Expected `rg`: no matches.

## Task 3: Consolidate the Pinned nnU-Net Trainers

**Files:**

- Create: `third_party/nnUNet/nnunetv2/training/nnUNetTrainer/variants/brats_mets/__init__.py`
- Move: `third_party/nnUNet/nnunetv2/training/nnUNetTrainer/nnUNetTrainer_ResEncM_DiceCEFocalTverskyFT.py`
- Move: `third_party/nnUNet/nnunetv2/training/nnUNetTrainer/nnUNetTrainer_ResEncM_SmallLesionOS.py`
- Move: `third_party/nnUNet/nnunetv2/training/nnUNetTrainer/nnUNetTrainer_ResEncM_ComponentSmallLesionOS.py`
- Delete: `third_party/nnUNet/nnunetv2/training/nnUNetTrainer/nnUNetTrainer_ResEncM_ComponentSmallLesionSyntheticTumorAug.py`
- Delete: `training/nnunet_trainers/`
- Modify: `third_party/nnUNet.UPSTREAM`
- Create: `THIRD_PARTY_NOTICES.md`
- Create: `tests/training/test_trainer_discovery.py`

- [ ] **Step 1: Write failing trainer discovery and lineage tests**

  Verify that nnU-Net recursive discovery resolves exactly the three retained custom class names, that each subclasses `nnUNetTrainer`, and that neither tracked paths nor Python symbols contain `SyntheticTumorAug`.

- [ ] **Step 2: Move the retained trainers to one canonical source**

  Keep class names unchanged so historical command lines remain interpretable. Remove duplicate copies. Add package exports only if nnU-Net discovery requires them.

- [ ] **Step 3: Record upstream and local modifications**

  `third_party/nnUNet.UPSTREAM` and `THIRD_PARTY_NOTICES.md` must state upstream URL, exact commit, version, Apache-2.0 license, and the local trainer additions.

- [ ] **Step 4: Verify and commit**

  ```bash
  python -m pytest tests/training/test_trainer_discovery.py -q
  python -m compileall -q third_party/nnUNet/nnunetv2
  git diff --check
  git add third_party training/nnunet_trainers THIRD_PARTY_NOTICES.md tests/training
  git commit -m "Consolidate final BraTS nnUNet trainers"
  ```

## Task 4: Migrate Dataset501 Conversion and Preprocessing

**Files:**

- Move/Rewrite: `preprocessing/prepare_dataset501.py` → `brats_mets/data/prepare_dataset501.py`
- Create: `brats_mets/data/contracts.py`
- Create: `configs/dataset501/dataset501.json`
- Create: `configs/dataset501/paths.env.example`
- Create: `scripts/common.sh`
- Create: `scripts/00_setup_environment.sh`
- Rewrite: `scripts/01_prepare_dataset.sh`
- Create: `scripts/02_plan_and_preprocess.sh`
- Delete: `preprocessing/plan_and_preprocess.sh`
- Delete empty: `preprocessing/`
- Create: `tests/data/test_dataset501_contract.py`
- Create: `tests/scripts/test_prepare_commands.py`

- [ ] **Step 1: Write failing data-contract tests**

  Cover:

  - four modalities with the historical channel order;
  - labels/regions and `regions_class_order`;
  - deterministic subject discovery and filename mapping;
  - refusal to overwrite or silently mix case sets;
  - scripts obtain all private paths from environment variables.

- [ ] **Step 2: Implement the converter and shell setup**

  - `00_setup_environment.sh` exports `PYTHONPATH` for the repo and vendored nnU-Net and validates `nnUNet_raw`, `nnUNet_preprocessed`, and `nnUNet_results`.
  - `01_prepare_dataset.sh` calls the package module with explicit source/destination variables.
  - `02_plan_and_preprocess.sh` runs nnU-Net fingerprinting/planning/preprocessing for Dataset501 and the named ResEnc M/XL configurations.
  - Default mode is fail-safe and never embeds local or remote absolute paths.

- [ ] **Step 3: Verify with synthetic fixtures**

  ```bash
  python -m pytest tests/data/test_dataset501_contract.py tests/scripts/test_prepare_commands.py -q
  bash -n scripts/00_setup_environment.sh scripts/01_prepare_dataset.sh scripts/02_plan_and_preprocess.sh
  ```

- [ ] **Step 4: Commit**

  ```bash
  git add -A
  git commit -m "Add Dataset501 preparation and preprocessing scripts"
  ```

## Task 5: Add Final Model and Key-Ablation Training Scripts

**Files:**

- Create: `configs/trainers/resencm.json`
- Create: `configs/trainers/resencxl.json`
- Create: `configs/trainers/focal_tversky.json`
- Create: `configs/trainers/small_lesion_os.json`
- Create: `configs/trainers/component_small_lesion_os.json`
- Rewrite: `training/run_nnunet.py` → `brats_mets/training/run_nnunet.py`
- Create: `scripts/03_train_resencm_5fold.sh`
- Create: `scripts/04_train_resencxl_5fold.sh`
- Create: `scripts/05_train_small_lesion_ft_5fold.sh`
- Delete: `scripts/02_train_models.sh`
- Create: `tests/training/test_training_configs.py`
- Rewrite: `tests/test_training_commands.py` → `tests/scripts/test_training_commands.py`

- [ ] **Step 1: Write failing configuration/command tests**

  Verify fixed dataset ID, plans/configuration names, trainer classes, five folds, checkpoint selection, resumability, and exact model-role separation:

  - ResEncM final primary model;
  - ResEncXL final primary model;
  - focal-Tversky FT donor;
  - SmallLesionOS and ComponentSmallLesionOS key ablations.

- [ ] **Step 2: Implement the reusable training launcher**

  Load one versioned config, validate it, emit an auditable command, and execute one fold. Numbered scripts iterate folds `0..4` in the documented order and accept a fold override for schedulers.

- [ ] **Step 3: Verify shell and dry-run commands**

  ```bash
  python -m pytest tests/training/test_training_configs.py tests/scripts/test_training_commands.py -q
  bash -n scripts/03_train_resencm_5fold.sh scripts/04_train_resencxl_5fold.sh scripts/05_train_small_lesion_ft_5fold.sh
  python -m brats_mets.training.run_nnunet --config configs/trainers/resencm.json --fold 0 --dry-run
  ```

- [ ] **Step 4: Commit**

  ```bash
  git add -A
  git commit -m "Script final model and key ablation training"
  ```

## Task 6: Migrate OOF Features and Learned Gates

**Files:**

- Move: `inference/vendor/lcv1/*.py` → `brats_mets/learned_gates/lcv1/`
- Move: `training/learned_gates/rgv3/region_gate_v3.py` → `brats_mets/learned_gates/rgv3.py`
- Move: `training/learned_gates/utility_v4/train_utility_v4.py` → `brats_mets/learned_gates/train_utility_v4.py`
- Move: `training/learned_gates/utility_v4/utility_v4.py` → `brats_mets/learned_gates/utility_v4.py`
- Create: `brats_mets/learned_gates/contracts.py`
- Create: `configs/learned_gates/lcv1.json`
- Create: `configs/learned_gates/lcv2.json`
- Create: `configs/learned_gates/rgv3.json`
- Create: `configs/learned_gates/utility_v4.json`
- Create: `scripts/06_generate_oof_probabilities.sh`
- Rewrite: `scripts/03_train_learned_gates.sh` → `scripts/07_train_learned_gates.sh`
- Delete: `training/learned_gates/`
- Delete: `configs/learned_gate/`
- Create: `tests/learned_gates/test_gate_configs.py`
- Migrate: `tests/learned_gate/*.py` → `tests/learned_gates/`

- [ ] **Step 1: Write failing learned-gate provenance tests**

  Assert five-fold OOF isolation, one-to-one component identity joins, required feature schemas, deterministic seeds, serialized-model output paths outside Git, and the fixed UTILITY_V4 thresholds/policy.

- [ ] **Step 2: Move LCv1/LCv2 without changing their numerical contracts**

  Replace path injection with package imports. Keep feature, reconstruction, and probability column names stable. Add explicit schemas at I/O boundaries.

- [ ] **Step 3: Move RGv3 and UTILITY_V4**

  Preserve the accepted family split and all frozen cutoffs. Training scripts write learned artifacts under a user-provided output root, never inside tracked paths.

- [ ] **Step 4: Implement OOF and gate-training shell stages**

  `06_generate_oof_probabilities.sh` produces the model sources consumed by the gate trainers. `07_train_learned_gates.sh` runs LCv1 → LCv2 → RGv3 → UTILITY_V4 in dependency order.

- [ ] **Step 5: Verify and commit**

  ```bash
  python -m pytest tests/learned_gates -q
  bash -n scripts/06_generate_oof_probabilities.sh scripts/07_train_learned_gates.sh
  python -m compileall -q brats_mets/learned_gates
  git add -A
  git commit -m "Migrate OOF learned gate pipeline"
  ```

## Task 7: Migrate Test Inference, XF12, N03, and UTILITY_V4

**Files:**

- Move/Rewrite: `inference/src/n03_docker/input_contract.py` → `brats_mets/inference/input_contract.py`
- Move/Rewrite: `inference/src/n03_docker/predict.py` → `brats_mets/inference/predict.py`
- Move/Rewrite: `inference/src/n03_docker/postprocess.py` → `brats_mets/postprocessing/component.py`
- Move: `inference/vendor/pipeline/mft_regionwise_pipeline.py` → `brats_mets/fusion/mft_regionwise.py`
- Move: `inference/vendor/pipeline/xlm_fixed_fusions.py` → `brats_mets/fusion/xf12.py`
- Move only needed helpers from: `inference/vendor/pipeline/build_xl_fixed_postprocess.py`
- Create: `brats_mets/postprocessing/n03_parent_supported.py`
- Create: `brats_mets/postprocessing/final_utility_v4.py`
- Create: `brats_mets/pipeline.py`
- Create: `brats_mets/cli.py`
- Create: `configs/fusion/xf12.json`
- Create: `configs/final/n03_utility_v4.json`
- Create: `scripts/08_predict_test_probabilities.sh`
- Create: `scripts/09_build_n03_utility_v4.sh`
- Delete: `inference/`
- Delete: `configs/models/`
- Delete: `configs/n03/`
- Delete: `scripts/04_export_inference_assets.sh`
- Create: `tests/inference/test_prediction_commands.py`
- Create: `tests/fusion/test_xf12.py`
- Create: `tests/postprocessing/test_n03_parent_supported.py`
- Create: `tests/postprocessing/test_final_utility_v4.py`
- Create: `tests/test_end_to_end_synthetic.py`

- [ ] **Step 1: Write failing source-selection and postprocessing tests**

  Cover:

  - M/XL/FT probability channel order and case-set equality;
  - exact XF12 regional source/weight selection;
  - N03 ET candidates must be parent-supported and disconnected additions only;
  - RC is bitwise frozen;
  - accepted ET additions propagate into TC and WT;
  - rejected/uncertain UTILITY_V4 components follow the frozen three-state rules;
  - empty component tables return a valid unchanged result;
  - no final global V2 pass occurs.

- [ ] **Step 2: Implement M/XL/FT probability inference**

  `08_predict_test_probabilities.sh` runs the three fixed five-fold ensembles with `checkpoint_best`, saves soft region probabilities and metadata, and checks only structural contracts required for the next stage.

- [ ] **Step 3: Implement the final scientific pipeline**

  `09_build_n03_utility_v4.sh` calls:

  ```text
  M + XL + FT soft probabilities
    -> XF12 regional fusion
    -> LCv3 component gate
    -> N03 ET parent-supported add-only rescue
    -> UTILITY_V4 three-state ET gate
    -> final segmentation directory
  ```

  It must require model/gate asset paths from the environment and write no archive.

- [ ] **Step 4: Verify with synthetic arrays and frozen micro-fixtures**

  ```bash
  python -m pytest \
    tests/inference \
    tests/fusion \
    tests/postprocessing \
    tests/test_end_to_end_synthetic.py -q
  bash -n scripts/08_predict_test_probabilities.sh scripts/09_build_n03_utility_v4.sh
  ```

- [ ] **Step 5: Commit**

  ```bash
  git add -A
  git commit -m "Add script-driven N03 UTILITY V4 inference chain"
  ```

## Task 8: Rewrite Public Documentation and Release Checks

**Files:**

- Rewrite: `README.md`
- Rewrite: `docs/DATA.md`
- Rewrite: `docs/TRAINING.md`
- Rewrite: `docs/INFERENCE.md`
- Rewrite: `docs/REPRODUCIBILITY.md`
- Create: `docs/PIPELINE.md`
- Create: `docs/CONFIGURATION_REFERENCE.md`
- Modify: `CITATION.cff`
- Modify: `SECURITY.md`
- Modify: `scripts/verify_release.py`
- Delete: `provenance/learned_gate_sources.json`
- Delete: `provenance/source_inventory.schema.json`
- Delete empty: `provenance/`
- Delete empty: `verification/`
- Create: `tests/repository/test_documented_commands.py`
- Create: `tests/repository/test_privacy_boundary.py`

- [ ] **Step 1: Write failing documentation/privacy tests**

  Extract documented shell commands and verify that referenced scripts/configs exist. Scan tracked text and filenames for credentials, private hostnames, private absolute paths, challenge case IDs, binary artifacts, Docker, and submission archive instructions.

- [ ] **Step 2: Rewrite docs around the public workflow**

  Explain:

  - licensed data acquisition is user responsibility;
  - environment/path setup;
  - Dataset501 conversion and preprocessing;
  - exact final/key-ablation training commands;
  - OOF and gate training;
  - final inference sequence and output directory;
  - expected compute/storage without claiming bundled weights;
  - vendored nnU-Net provenance and licenses;
  - scientific boundary between reproduced code and unpublished challenge artifacts.

- [ ] **Step 3: Replace release verifier**

  `scripts/verify_release.py` validates source/package/test/privacy/license/doc contracts only. It must not inspect 179 outputs or create/verify a submission ZIP.

- [ ] **Step 4: Verify and commit**

  ```bash
  python -m pytest tests/repository -q
  python scripts/verify_release.py
  rg -n -i "docker|submission zip|crc|sha256|deepln|funhpc|root@" README.md docs scripts brats_mets configs tests
  git diff --check
  git add -A
  git commit -m "Document the public BraTS METS pipeline"
  ```

## Task 9: Full Verification, Push, and Draft PR

**Files:**

- Modify only if verification finds defects in files already listed above.

- [ ] **Step 1: Run the complete fresh verification suite**

  ```bash
  python -m pytest -q
  python -m compileall -q brats_mets third_party/nnUNet/nnunetv2
  Get-ChildItem scripts -Filter *.sh | ForEach-Object { bash -n $_.FullName }
  python scripts/verify_release.py
  git diff --check
  git status --short
  ```

- [ ] **Step 2: Audit the exact branch diff and tracked artifacts**

  ```bash
  git diff --stat main...HEAD
  git diff --name-status main...HEAD
  git ls-files | rg -i "\.(zip|tar|nii|nii\.gz|npz|pth|pt|pkl|joblib)$|docker"
  git grep -n -I -E "root@|deepln|funhpc|BEGIN (RSA|OPENSSH) PRIVATE KEY|password"
  ```

  Expected artifact/path scans: no matches other than explanatory deny-list tests.

- [ ] **Step 3: Add a final verification commit only if needed**

  Never create an empty “verification” commit. If fixes are required, rerun the complete suite and commit them with a scoped message.

- [ ] **Step 4: Push the independent branch**

  ```bash
  git push -u origin agent/public-training-pipeline
  ```

- [ ] **Step 5: Create a draft pull request**

  Title:

  ```text
  Publish the full BraTS-METS training and N03 pipeline
  ```

  The PR body must summarize retained model lineage, vendored nnU-Net pin, Bash workflow, frozen N03/UTILITY_V4 semantics, exclusions, and exact verification evidence.

- [ ] **Step 6: Verify remote state**

  Confirm that the remote branch head equals local `HEAD`, the repository remains public, the PR is draft, and no excluded artifact is present in the remote tree. Do not claim publication complete before this check.

