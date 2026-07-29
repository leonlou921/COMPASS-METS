from __future__ import annotations

from pathlib import Path

import pytest

from verification.release_manifest import build_release_manifest


def _report(*, exact: bool, structural: bool = True) -> dict:
    return {
        "candidate_id": "N03_FINAL_UTILITY_V4",
        "passed": exact,
        "structural_passed": structural,
        "verification_mode": "single_run",
        "case_count": 179,
        "candidate_case_count": 179,
        "candidate_case_set_equal": True,
        "different_case_count": 0 if exact else 3,
        "different_voxels": 0 if exact else 7,
    }


def test_release_manifest_keeps_exact_equivalence_as_default_gate(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "image.tar"
    archive.write_bytes(b"image")
    payload = build_release_manifest(_report(exact=True), archive)
    assert payload["release_accepted"] is True
    assert payload["acceptance_mode"] == "exact_frozen_equivalence"
    assert payload["different_voxels"] == 0


def test_release_manifest_requires_explicit_acceptance_for_recorded_difference(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "image.tar"
    archive.write_bytes(b"image")
    with pytest.raises(ValueError, match="exact frozen equivalence"):
        build_release_manifest(_report(exact=False), archive)

    payload = build_release_manifest(
        _report(exact=False),
        archive,
        accept_recorded_difference=True,
    )
    assert payload["release_accepted"] is True
    assert payload["acceptance_mode"] == "explicit_recorded_difference"
    assert payload["different_case_count"] == 3
    assert payload["different_voxels"] == 7
    assert payload["frozen_equivalence_passed"] is False


def test_release_manifest_never_accepts_structural_failure(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "image.tar"
    archive.write_bytes(b"image")
    with pytest.raises(ValueError, match="structural"):
        build_release_manifest(
            _report(exact=False, structural=False),
            archive,
            accept_recorded_difference=True,
        )
