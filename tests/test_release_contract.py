from __future__ import annotations

import json
from pathlib import Path

from scripts.verify_release import inspect_release


def test_release_verifier_rejects_credentials_and_private_absolute_paths(
    tmp_path: Path,
) -> None:
    (tmp_path / "safe.py").write_text("print('ok')\n", encoding="utf-8")
    credential = "pass" + "word = supersecret"
    private_path = "/data/" + "coding/challenge/private"
    (tmp_path / "bad.txt").write_text(
        f"{credential}\n{private_path}\n", encoding="utf-8"
    )
    report = inspect_release(tmp_path)
    assert report["passed"] is False
    kinds = {finding["kind"] for finding in report["findings"]}
    assert "credential" in kinds
    assert "private_absolute_path" in kinds


def test_release_verifier_allows_only_pinned_nnunet_public_test_fixtures(
    tmp_path: Path,
) -> None:
    public = (
        tmp_path
        / "third_party"
        / "nnUNet"
        / "nnunetv2"
        / "tests"
        / "example_data"
    )
    public.mkdir(parents=True)
    (public / "example_ct_sm.nii.gz").write_bytes(b"upstream fixture")
    private = tmp_path / "predictions"
    private.mkdir()
    (private / "case.nii.gz").write_bytes(b"challenge prediction")

    report = inspect_release(tmp_path)
    binaries = [
        finding["path"]
        for finding in report["findings"]
        if finding["kind"] == "private_binary"
    ]
    assert binaries == [str(Path("predictions") / "case.nii.gz")]


def test_release_verifier_does_not_treat_generated_private_assets_as_source(
    tmp_path: Path,
) -> None:
    assets = tmp_path / "assets" / "nnUNet_results" / "fold_0"
    assets.mkdir(parents=True)
    (assets / "checkpoint_best.pth").write_bytes(b"private model")

    report = inspect_release(tmp_path)
    assert not [
        finding
        for finding in report["findings"]
        if finding["path"].startswith("assets")
    ]


def test_release_verifier_accepts_documented_placeholders_and_frozen_config(
    tmp_path: Path,
) -> None:
    (tmp_path / "README.md").write_text(
        "Use /path/to/data and set SSH_PASSWORD in your own shell.\n",
        encoding="utf-8",
    )
    config_root = tmp_path / "configs" / "n03"
    config_root.mkdir(parents=True)
    (config_root / "final.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "candidate_id": "N03_FINAL_UTILITY_V4",
                "baseline_candidate_id": "N03_XF12_LCv3_ET_parent_supported",
                "proposal_threshold": 0.25,
                "et_component_cutoff": 0.5497123599,
                "minimum_parent_model_support": 2,
                "parent_models": ["XL", "M", "FT"],
                "policy": "add_only_et_parent_supported",
                "preserve_anchor": True,
                "rerun_anchor_postprocess_after_addition": False,
                "utility_v4": {
                    "candidate_scope": "disconnected_et_from_lcv2_structured_union",
                    "rgv3_et_cutoff": 0.7702616034384248,
                    "accept_all_scores_gte": 0.75,
                    "reject_any_utility_score_lt": 0.5,
                    "operation": "et_add_only",
                    "preserve_rc_priority": True,
                },
            }
        ),
        encoding="utf-8",
    )
    report = inspect_release(tmp_path)
    assert report["passed"] is True
    assert report["n03_config_valid"] is True


def test_pinned_nnunet_source_is_present_and_licensed() -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "third_party" / "nnUNet"
    provenance = (root / "third_party" / "nnUNet.UPSTREAM").read_text(
        encoding="utf-8"
    )

    assert (source / "nnunetv2").is_dir()
    assert (source / "LICENSE").is_file()
    assert "86606c53ef9f556d6f024a304b52a48378453641" in provenance
    assert "https://github.com/MIC-DKFZ/nnUNet" in provenance


def test_ordered_shell_workflow_is_present() -> None:
    root = Path(__file__).resolve().parents[1]
    expected = {
        "01_prepare_dataset.sh",
        "02_train_models.sh",
        "03_train_learned_gates.sh",
        "04_export_inference_assets.sh",
        "05_build_final_image.sh",
        "06_verify_final_image.sh",
        "07_release_final_image.sh",
    }
    scripts = root / "scripts"
    assert expected.issubset({path.name for path in scripts.glob("*.sh")})

    for name in sorted(expected):
        text = (scripts / name).read_text(encoding="utf-8")
        assert text.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")

    assert "third_party/nnUNet" in (
        scripts / "05_build_final_image.sh"
    ).read_text(encoding="utf-8")
    assert "verify_frozen_equivalence.py" in (
        scripts / "06_verify_final_image.sh"
    ).read_text(encoding="utf-8")


def test_restricted_host_can_run_an_exported_rootfs_without_docker() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (
        root / "verification" / "run_exported_rootfs.sh"
    ).read_text(encoding="utf-8")
    assert script.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
    assert "N03_RUNTIME_ROOT" in script
    assert "/opt/conda/bin/nnUNetv2_predict" in script
    assert "run_pipeline" in script
    assert "N03_FINAL_UTILITY_V4" in script
    private_prefix = "/data/" + "coding/challenge"
    assert private_prefix not in script
