from __future__ import annotations

import pathlib
import re

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[2]
PINNED_NNUNET_COMMIT = "86606c53ef9f556d6f024a304b52a48378453641"


def _pyproject_text() -> str:
    return (ROOT / "pyproject.toml").read_text(encoding="utf-8")


def test_public_distribution_and_package_contract() -> None:
    pyproject = _pyproject_text()
    assert re.search(
        r'(?ms)^\[project\].*?^name\s*=\s*"compass-mets"$',
        pyproject,
    )
    assert (ROOT / "compass_mets" / "__init__.py").is_file()
    assert re.search(
        r'(?ms)^\[tool\.setuptools\.packages\.find\].*?'
        r'^include\s*=\s*\["compass_mets\*"\]$',
        pyproject,
    )


def test_environment_manifests_are_self_contained() -> None:
    assert (ROOT / "environment.yml").is_file()
    assert (ROOT / "requirements" / "runtime.txt").is_file()
    training = ROOT / "requirements" / "training.txt"
    assert training.is_file()
    for manifest in (
        ROOT / "environment.yml",
        ROOT / "requirements" / "runtime.txt",
        training,
    ):
        text = manifest.read_text(encoding="utf-8")
        assert "../docker" not in text
        assert "docker/" not in text


def test_vendored_nnunet_revision_is_exactly_pinned() -> None:
    provenance = (ROOT / "third_party" / "nnUNet.UPSTREAM").read_text(
        encoding="utf-8"
    )
    assert "Version: 2.6.2" in provenance
    assert f"Commit: {PINNED_NNUNET_COMMIT}" in provenance
    assert (
        ROOT / "third_party" / "nnUNet" / "nnunetv2" / "__init__.py"
    ).is_file()


@pytest.mark.parametrize(
    "path",
    [
        "docker",
        ".dockerignore",
        "scripts/05_build_final_image.sh",
        "scripts/06_verify_final_image.sh",
        "scripts/07_release_final_image.sh",
        "scripts/08_package_submission.sh",
        "scripts/build_docker_archive.sh",
        "scripts/build_docker_archive_kaniko.sh",
        "verification/package_submission.py",
        "verification/run_exported_rootfs.sh",
        "verification/verify_frozen_equivalence.py",
        "docs/DOCKER.md",
        "provenance/frozen_image.json",
    ],
)
def test_container_and_submission_entrypoints_are_absent(path: str) -> None:
    assert not (ROOT / path).exists()


def test_public_docs_use_compass_mets_source_pipeline_only() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert readme.startswith("# COMPASS-METS")
    assert "configs/final/n03_utility_v4.json" in readme
    assert "scripts/09_build_n03_utility_v4.sh" in readme
    combined = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
    )
    for stale in (
        "configs/models/final_models.json",
        "configs/n03/final.json",
        "preprocessing/prepare_dataset501.py",
        "training/run_nnunet.py",
        "inference/src/n03_docker",
        "brats-mets-MicroBT",
        "brats_mets",
    ):
        assert stale not in combined


def test_legacy_asset_and_release_metadata_are_absent() -> None:
    for relative in (
        "provenance/learned_gate_sources.json",
        "provenance/source_inventory.schema.json",
        "verification/release_manifest.py",
        "scripts/04_export_inference_assets.sh",
    ):
        assert not (ROOT / relative).exists()
