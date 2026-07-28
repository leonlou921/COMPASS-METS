from __future__ import annotations

from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import pytest


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE_ROOT))

from n03_docker.input_contract import (  # noqa: E402
    MODALITIES,
    discover_cases,
    stage_nnunet_inputs,
)


def _write_image(path: Path, affine: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(
        nib.Nifti1Image(
            np.zeros((4, 5, 6), dtype=np.float32),
            np.eye(4) if affine is None else affine,
        ),
        path,
    )


def test_discovers_flat_nnunet_and_nested_named_modalities(tmp_path: Path) -> None:
    flat = tmp_path / "flat"
    for channel in range(4):
        _write_image(flat / f"BraTS-MET-12345-100_{channel:04d}.nii.gz")

    nested = tmp_path / "named" / "BraTS-MET-54321-200"
    for modality in MODALITIES:
        _write_image(nested / f"BraTS-MET-54321-200-{modality}.nii.gz")

    cases = discover_cases(tmp_path)

    assert [case.case_id for case in cases] == [
        "BraTS-MET-12345-100",
        "BraTS-MET-54321-200",
    ]
    assert set(cases[0].modalities) == set(MODALITIES)
    assert set(cases[1].modalities) == set(MODALITIES)


def test_rejects_missing_or_ambiguous_modality(tmp_path: Path) -> None:
    incomplete = tmp_path / "incomplete"
    for modality in MODALITIES[:-1]:
        _write_image(incomplete / f"case-a-{modality}.nii.gz")
    with pytest.raises(ValueError, match="case-a.*missing.*t2w"):
        discover_cases(incomplete)

    complete = tmp_path / "ambiguous"
    for modality in MODALITIES:
        _write_image(complete / f"case-b-{modality}.nii.gz")
    _write_image(complete / "duplicate" / "case-b-t1c.nii.gz")
    with pytest.raises(ValueError, match="duplicate.*case-b.*t1c"):
        discover_cases(complete)


def test_rejects_geometry_mismatch(tmp_path: Path) -> None:
    for modality in MODALITIES:
        affine = np.eye(4)
        if modality == "t2w":
            affine[0, 3] = 2.0
        _write_image(tmp_path / f"case-c-{modality}.nii.gz", affine)

    with pytest.raises(ValueError, match="case-c.*geometry.*t2w"):
        discover_cases(tmp_path)


def test_staging_uses_links_and_never_mutates_input(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    for modality in MODALITIES:
        _write_image(input_root / f"case-d-{modality}.nii.gz")
    before = {
        path: (path.stat().st_size, path.stat().st_mtime_ns)
        for path in input_root.rglob("*.nii.gz")
    }
    case = discover_cases(input_root)[0]

    staged = stage_nnunet_inputs([case], tmp_path / "work")

    assert [path.name for path in staged] == [
        f"case-d_{channel:04d}.nii.gz" for channel in range(4)
    ]
    assert all(path.is_symlink() for path in staged)
    after = {
        path: (path.stat().st_size, path.stat().st_mtime_ns)
        for path in input_root.rglob("*.nii.gz")
    }
    assert before == after
