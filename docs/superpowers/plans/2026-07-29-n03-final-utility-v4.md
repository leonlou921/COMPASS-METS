# N03 FINAL UTILITY V4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, externally verify, back up, and publish a complete nnU-Net-based Docker pipeline for `N03_FINAL_UTILITY_V4`.

**Architecture:** Extend the existing three-model XL/M/FT pipeline with the frozen RGv3-ET and dual utility-v4 gates, while keeping private binaries in a hash-verified asset bundle. Vendor the pinned clean nnU-Net source and expose the original staged workflow through ordered shell scripts. Treat the frozen 179-case ZIP as an image-external oracle and release only after two fresh image runs match it voxel-for-voxel.

**Tech Stack:** Python 3.10/3.11, PyTorch 2.4.1, nnU-Net v2, LightGBM, NumPy, SciPy, nibabel, Bash, Docker/BuildKit, pytest.

## Global Constraints

- Canonical candidate is `N03_FINAL_UTILITY_V4`.
- Frozen reference ZIP SHA256 is `dc1b2a6e25f1569ec68240e996b8d772e943b7fb086fff34e790d8676d53c735`.
- The reference ZIP stays outside the Docker build context and image.
- Historical test probabilities and case-specific correction patches are forbidden image assets.
- Accept disconnected ET additions only when LCv2, utility-v4 existence, and utility-v4 geometry-safe scores are each at least `0.75`.
- Preserve the N03 anchor and enforce `ET subset TC subset WT`.
- Pin upstream nnU-Net commit `86606c53ef9f556d6f024a304b52a48378453641`.
- Release requires two repeatable fresh runs and exact 179-case external equivalence.

---

### Task 1: Frozen candidate and utility-v4 inference

**Files:**
- Modify: `configs/n03/final.json`
- Create: `inference/src/n03_docker/utility_v4.py`
- Modify: `inference/src/n03_docker/postprocess.py`
- Create: `tests/docker/test_utility_v4.py`
- Modify: `tests/docker/test_postprocess.py`

**Interfaces:**
- Consumes: `learned_component_scores(...)`, `reconstruct_scored_candidate(...)`, and the current N03 label anchor.
- Produces: `three_state_decision(frame: pandas.DataFrame) -> pandas.Series` and `build_n03_final_from_features(...) -> tuple[numpy.ndarray, list[dict]]`.

- [ ] **Step 1: Write the failing behavior tests**

```python
def test_gate_accepts_only_when_all_required_scores_reach_075():
    frame = pd.DataFrame({
        "v2_component_probability": [0.75, 0.74, 0.90],
        "v4_existence_probability": [0.75, 0.90, 0.49],
        "v4_geometry_probability": [0.75, 0.90, 0.90],
    })
    assert three_state_decision(frame).tolist() == [
        "accept", "abstain", "reject"
    ]

def test_final_candidate_is_add_only_and_preserves_existing_labels():
    base = np.array([[[4, 3, 2, 1, 0]]], dtype=np.uint8)
    result = add_accepted_et(base, [(0, 0, 4)])
    assert result.tolist() == [[[[4, 3, 2, 1, 3]]]]
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `python -m pytest tests/docker/test_utility_v4.py tests/docker/test_postprocess.py -q`

Expected: import or assertion failure because utility-v4 behavior is absent.

- [ ] **Step 3: Implement the frozen gate and disconnected ET application**

```python
ACCEPT_THRESHOLD = 0.75
REJECT_THRESHOLD = 0.50

def three_state_decision(frame):
    values = frame.loc[:, SCORE_COLUMNS].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("gate scores must be finite")
    decision = np.full(len(frame), "abstain", dtype=object)
    decision[np.any(values[:, 1:] < REJECT_THRESHOLD, axis=1)] = "reject"
    decision[np.all(values >= ACCEPT_THRESHOLD, axis=1)] = "accept"
    return pd.Series(decision, index=frame.index, name="gate_decision")
```

Use `LC_V2_STRUCTURED_FILTER` reconstruction, retain only 26-connected ET
components with no overlap with the N03 ET mask, calculate RGv3-ET and both
utility-v4 scores from the same component feature row, accept only the frozen
three-score rule, and add ET without rerunning global V2.

- [ ] **Step 4: Run focused and full tests**

Run: `python -m pytest tests/docker/test_utility_v4.py tests/docker/test_postprocess.py -q`

Expected: PASS.

Run: `python -m pytest -q`

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add configs/n03/final.json inference/src/n03_docker/utility_v4.py \
  inference/src/n03_docker/postprocess.py tests/docker
git commit -m "Add frozen utility-v4 ET gate"
```

---

### Task 2: Private asset inventory and fail-closed export

**Files:**
- Modify: `inference/src/n03_docker/asset_inventory.py`
- Modify: `inference/src/n03_docker/assets.py`
- Modify: `docker/build_asset_inventory.py`
- Modify: `docker/prepare_assets.py`
- Modify: `provenance/source_inventory.schema.json`
- Create: `tests/docker/test_utility_v4_assets.py`

**Interfaces:**
- Consumes: source checkpoint paths and learned-model paths supplied through a private inventory.
- Produces: a private `assets/learned_models/{rgv3_et,utility_v4_existence,utility_v4_geometry}` tree plus `utility_v4/feature_names.json`.

- [ ] **Step 1: Write failing asset-contract tests**

```python
def test_inventory_rejects_wrong_utility_hash(valid_inventory, tmp_path):
    valid_inventory["learned_models"]["utility_v4_existence"]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="hash"):
        validate_source_inventory(valid_inventory, verify_files=True)

def test_asset_export_contains_all_four_final_gate_assets(bundle):
    assert (bundle / "learned_models/rgv3_et/models.joblib").is_file()
    assert (bundle / "learned_models/utility_v4_existence/model.joblib").is_file()
    assert (bundle / "learned_models/utility_v4_geometry/model.joblib").is_file()
    assert (bundle / "learned_models/utility_v4/feature_names.json").is_file()
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/docker/test_utility_v4_assets.py -q`

Expected: FAIL because the new asset roles are unknown.

- [ ] **Step 3: Implement exact role and hash validation**

Add frozen expected hashes:

```python
EXPECTED_UTILITY_HASHES = {
    "utility_v4_feature_names": "87b6523508688f52ad1cee6d600d7d353991f0d1451ec29ce0aed500ee07699d",
    "utility_v4_existence": "8a8c8f02ed652861b949b9a47aedaa8faff8e9904e4d9645f48a1a743ec0e3e0",
    "utility_v4_geometry": "bfd2d2be7e9349d4cde41f1e4682f87b9f0108216d69597cb80d0a6c9991f87e",
    "rgv3_et": "5fff6d9c7ef31bf4ce33bad211abe017fa1e0235d4e7f78d264c50c2c2a9fac1",
}
```

Copy only verified final bundles. Do not copy OOF tables, test probabilities,
test predictions, reference ZIPs, or per-case audits.

- [ ] **Step 4: Run focused and full tests**

Run: `python -m pytest tests/docker/test_utility_v4_assets.py tests/docker/test_assets.py tests/docker/test_remote_asset_audit.py -q`

Expected: PASS.

Run: `python -m pytest -q`

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add inference/src/n03_docker/asset_inventory.py \
  inference/src/n03_docker/assets.py docker provenance \
  tests/docker/test_utility_v4_assets.py
git commit -m "Validate utility-v4 inference assets"
```

---

### Task 3: End-to-end final runtime

**Files:**
- Modify: `inference/src/n03_docker/pipeline.py`
- Modify: `inference/src/n03_docker/cli.py`
- Modify: `docker/Dockerfile`
- Modify: `docker/requirements.lock`
- Modify: `tests/docker/test_pipeline.py`
- Modify: `tests/docker/test_docker_context.py`

**Interfaces:**
- Consumes: three fresh probability arrays and seven learned bundles loaded from `/opt/n03/assets`.
- Produces: flat `N03_FINAL_UTILITY_V4` NIfTI outputs and a run report outside `/output`.

- [ ] **Step 1: Write a failing synthetic end-to-end test**

```python
def test_pipeline_reports_final_candidate(fake_predictor, valid_input, assets, output):
    report = run_pipeline(
        input_root=valid_input,
        output_root=output,
        assets_root=assets,
        executable=str(fake_predictor),
    )
    assert report["candidate"] == "N03_FINAL_UTILITY_V4"
    assert sorted(path.name for path in output.iterdir()) == ["case-001.nii.gz"]
```

- [ ] **Step 2: Run the test and verify RED**

Run: `python -m pytest tests/docker/test_pipeline.py -q`

Expected: FAIL because the runtime reports the obsolete N03 ID or lacks new bundles.

- [ ] **Step 3: Load the frozen bundles and invoke final postprocessing**

Change `_load_learned_bundles` to return a typed bundle mapping with LCv1,
LCv2, RGv3-ET, utility-v4 existence, utility-v4 geometry, and feature names.
Update the report candidate and image labels to `N03_FINAL_UTILITY_V4`.

- [ ] **Step 4: Run focused and full tests**

Run: `python -m pytest tests/docker/test_pipeline.py tests/docker/test_docker_context.py -q`

Expected: PASS.

Run: `python -m pytest -q`

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add inference/src/n03_docker docker tests/docker
git commit -m "Run N03 final utility-v4 pipeline"
```

---

### Task 4: Pin nnU-Net source and project trainers

**Files:**
- Create: `third_party/nnUNet/**`
- Create: `third_party/nnUNet.UPSTREAM`
- Create: `training/nnunet_trainers/nnUNetTrainer_ResEncM_ComponentSmallLesionOS.py`
- Create: `training/nnunet_trainers/nnUNetTrainer_ResEncM_ComponentSmallLesionSyntheticTumorAug.py`
- Create: `training/nnunet_trainers/nnUNetTrainer_ResEncM_SmallLesionOS.py`
- Modify: `docker/Dockerfile`
- Modify: `tests/test_release_contract.py`

**Interfaces:**
- Consumes: clean upstream archive at the pinned commit and audited custom trainers.
- Produces: an inspectable source tree installed directly by Docker and reusable by training scripts.

- [ ] **Step 1: Write failing source provenance tests**

```python
def test_pinned_nnunet_source_is_present_and_licensed(repo_root):
    assert (repo_root / "third_party/nnUNet/nnunetv2").is_dir()
    assert (repo_root / "third_party/nnUNet/LICENSE").is_file()
    assert "86606c53ef9f556d6f024a304b52a48378453641" in (
        repo_root / "third_party/nnUNet.UPSTREAM"
    ).read_text()
```

- [ ] **Step 2: Run the test and verify RED**

Run: `python -m pytest tests/test_release_contract.py -q`

Expected: FAIL because `third_party/nnUNet` is absent.

- [ ] **Step 3: Import the clean pinned tree and trainers**

Create the source tree from `git archive
86606c53ef9f556d6f024a304b52a48378453641`, not from the remote dirty
working tree. Copy the four audited custom trainer source files, exclude
`__pycache__`, and make Docker install `third_party/nnUNet`.

- [ ] **Step 4: Verify imports, license, and tests**

Run: `python -m compileall -q third_party/nnUNet training`

Expected: exit 0.

Run: `python -m pytest tests/test_release_contract.py tests/test_training_commands.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add third_party training docker/Dockerfile tests
git commit -m "Vendor pinned nnU-Net and project trainers"
```

---

### Task 5: Ordered shell workflow and public documentation

**Files:**
- Create: `scripts/01_prepare_dataset.sh`
- Create: `scripts/02_train_models.sh`
- Create: `scripts/03_train_learned_gates.sh`
- Create: `scripts/04_export_inference_assets.sh`
- Create: `scripts/05_build_final_image.sh`
- Create: `scripts/06_verify_final_image.sh`
- Create: `scripts/07_release_final_image.sh`
- Modify: `README.md`
- Modify: `docs/TRAINING.md`
- Modify: `docs/INFERENCE.md`
- Modify: `docs/DOCKER.md`
- Modify: `tests/test_training_commands.py`
- Modify: `tests/test_release_contract.py`

**Interfaces:**
- Consumes: environment variables for data/results/assets/reference/archive roots.
- Produces: restartable training artifacts, Docker archive, verification JSON/CSV, and release manifest.

- [ ] **Step 1: Write failing shell behavior tests**

```python
@pytest.mark.parametrize("name", [
    "01_prepare_dataset.sh", "02_train_models.sh",
    "03_train_learned_gates.sh", "04_export_inference_assets.sh",
    "05_build_final_image.sh", "06_verify_final_image.sh",
    "07_release_final_image.sh",
])
def test_ordered_shell_entrypoints_parse(name, repo_root):
    subprocess.run(["bash", "-n", str(repo_root / "scripts" / name)], check=True)
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/test_training_commands.py tests/test_release_contract.py -q`

Expected: FAIL because the ordered scripts do not exist.

- [ ] **Step 3: Implement fail-fast shell entrypoints**

Each script starts with:

```bash
#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
```

The release script must read the verifier JSON and require:

```bash
jq -e '
  .passed == true and
  .case_count == 179 and
  .different_case_count == 0 and
  .different_voxel_count == 0 and
  .repeatability_passed == true
' "${VERIFY_JSON}" >/dev/null
```

- [ ] **Step 4: Run shell, privacy, and repository tests**

Run: `python -m pytest tests/test_training_commands.py tests/test_release_contract.py -q`

Expected: PASS.

Run: `python scripts/verify_release.py .`

Expected: `"passed": true`.

- [ ] **Step 5: Commit**

```bash
git add scripts README.md docs tests
git commit -m "Document ordered training and release workflow"
```

---

### Task 6: Image-external 179-case equivalence verifier

**Files:**
- Create: `verification/verify_frozen_equivalence.py`
- Create: `tests/verification/test_frozen_equivalence.py`
- Modify: `.dockerignore`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: two image output directories and the external frozen reference ZIP.
- Produces: JSON and CSV evidence covering case set, arrays, geometry, dtype, labels, aggregate counts, and run-to-run repeatability.

- [ ] **Step 1: Write failing verifier tests with literal NIfTI fixtures**

```python
def test_verifier_rejects_one_voxel_difference(reference_zip, run_a, run_b):
    mutate_one_voxel(run_a / "case-001.nii.gz")
    report = verify(reference_zip, run_a, run_b)
    assert report["passed"] is False
    assert report["different_case_count"] == 1
    assert report["different_voxel_count"] == 1
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/verification/test_frozen_equivalence.py -q`

Expected: import failure because the verifier does not exist.

- [ ] **Step 3: Implement independent semantic comparison**

The verifier loads each NIfTI independently, requires the exact 179-case set,
compares arrays with `numpy.array_equal`, affine with zero-tolerance equality,
spacing, shape, header dtype, legal labels `{0,1,2,3,4}`, per-label aggregate
counts, and run A versus run B. It records every differing case and voxel count
without modifying inputs.

- [ ] **Step 4: Run verifier and full tests**

Run: `python -m pytest tests/verification/test_frozen_equivalence.py -q`

Expected: PASS.

Run: `python -m pytest -q`

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add verification tests/verification .dockerignore .gitignore
git commit -m "Add external frozen-output release gate"
```

---

### Task 7: Remote build, two fresh runs, and evidence capture

**Files:**
- Generate remotely outside Git: private asset inventory and build context
- Generate remotely outside Git: Docker archive, two output directories, JSON/CSV verifier reports
- Copy locally: `docker/n03_final_utility_v4/artifacts/**`
- Copy locally: `docker/n03_final_utility_v4/provenance/**`

**Interfaces:**
- Consumes: remote original checkpoints, raw 179-case inputs, learned assets, public source worktree, and external frozen ZIP.
- Produces: a release-qualified Docker TAR or an explicit failed diagnostic build.

- [ ] **Step 1: Deploy the committed source and construct the private inventory**

Run the committed `04_export_inference_assets.sh` with remote source paths
provided only through environment variables. Verify the four fixed learned
asset hashes and 15 checkpoint tensor hashes before build.

- [ ] **Step 2: Build the image archive**

Run: `bash scripts/05_build_final_image.sh`

Expected: Docker TAR and SHA256 sidecar, no reference ZIP in context layers.

- [ ] **Step 3: Run all 179 cases twice from empty output directories**

Run: `bash scripts/06_verify_final_image.sh`

Expected: two completed fresh inference runs; no reused output or historical
test probability directory.

- [ ] **Step 4: Execute the external equivalence gate**

Run:

```bash
python verification/verify_frozen_equivalence.py \
  --reference-zip "${REFERENCE_ZIP}" \
  --run-a "${RUN_A}" \
  --run-b "${RUN_B}" \
  --json "${VERIFY_JSON}" \
  --csv "${VERIFY_CSV}"
```

Expected: `passed=true`, 179 cases, zero differing cases and voxels, exact
geometry/dtype/labels/aggregate counts, and `repeatability_passed=true`.

- [ ] **Step 5: Run release script only if verification passed**

Run: `bash scripts/07_release_final_image.sh`

Expected: release manifest binds source commit, Docker TAR SHA256, asset
manifest SHA256, reference SHA256, and verifier report SHA256.

- [ ] **Step 6: Independently copy and rehash artifacts on the E drive**

Compare remote and local TAR/report SHA256 values. Do not stop the remote host
until all values match.

---

### Task 8: Final repository verification and GitHub publication

**Files:**
- Modify: `provenance/frozen_image.json`
- Generate: `release_audit.json`

**Interfaces:**
- Consumes: release-qualified archive manifest and all committed source.
- Produces: reviewed feature branch, merged `main`, and verified public GitHub commit.

- [ ] **Step 1: Record only non-private release provenance**

Update `provenance/frozen_image.json` with candidate ID, source commit, image
archive SHA256, size, nnU-Net commit, asset-manifest SHA256, and external
verification report SHA256. Do not record remote absolute paths.

- [ ] **Step 2: Run the complete verification suite**

Run:

```bash
python -m pytest -q
python -m compileall -q inference training preprocessing verification
python scripts/verify_release.py . --output release_audit.json
git diff --check
git status --short
```

Expected: all tests PASS, compilation succeeds, audit `passed=true`, no
whitespace errors, and only intended release files are staged.

- [ ] **Step 3: Commit provenance**

```bash
git add provenance/frozen_image.json release_audit.json
git commit -m "Freeze N03 final utility-v4 release evidence"
```

- [ ] **Step 4: Complete the branch**

Use `superpowers:finishing-a-development-branch`, re-run tests, review the
diff against `main`, and merge only after the external gate has passed.

- [ ] **Step 5: Push and independently verify GitHub**

Push `main` to `https://github.com/leonlou921/brats-mets-MicroBT`, then verify
the remote default branch commit equals the local merged commit and that the
public README identifies `N03_FINAL_UTILITY_V4`.

