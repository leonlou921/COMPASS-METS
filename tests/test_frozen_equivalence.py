from __future__ import annotations

import csv
import json
from pathlib import Path
import zipfile

import nibabel as nib
import numpy as np

from verification.verify_frozen_equivalence import verify_frozen_equivalence


def _write_label(
    path: Path,
    array: np.ndarray,
    *,
    affine: np.ndarray | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(
        np.asarray(array, dtype=np.uint8),
        np.eye(4) if affine is None else affine,
    )
    image.header.set_data_dtype(np.uint8)
    nib.save(image, path)


def _zip_flat(source: Path, destination: Path) -> None:
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_STORED) as archive:
        for path in sorted(source.glob("*.nii.gz")):
            archive.write(path, arcname=path.name)


def test_frozen_equivalence_accepts_exact_repeatable_outputs(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    repeat = tmp_path / "repeat"
    for index, case_id in enumerate(("case-a", "case-b")):
        label = np.zeros((3, 4, 5), dtype=np.uint8)
        label[index, 1, 2] = index + 3
        for root in (reference, candidate, repeat):
            _write_label(root / f"{case_id}.nii.gz", label)
    reference_zip = tmp_path / "reference.zip"
    _zip_flat(reference, reference_zip)

    json_output = tmp_path / "report.json"
    csv_output = tmp_path / "report.csv"
    report = verify_frozen_equivalence(
        reference_zip,
        candidate,
        repeat,
        expected_cases=2,
        json_output=json_output,
        csv_output=csv_output,
    )

    assert report["passed"] is True
    assert report["case_count"] == 2
    assert report["different_voxels"] == 0
    assert report["repeat_different_voxels"] == 0
    assert report["aggregate_voxel_counts"]["reference"] == {
        "0": 118,
        "1": 0,
        "2": 0,
        "3": 1,
        "4": 1,
    }
    assert json.loads(json_output.read_text(encoding="utf-8"))["passed"] is True
    with csv_output.open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 2


def test_frozen_equivalence_reports_voxel_geometry_and_case_set_failures(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    repeat = tmp_path / "repeat"
    base = np.zeros((2, 2, 2), dtype=np.uint8)
    _write_label(reference / "case-a.nii.gz", base)
    _write_label(reference / "case-b.nii.gz", base)
    changed = base.copy()
    changed[0, 0, 0] = 3
    _write_label(candidate / "case-a.nii.gz", changed)
    different_affine = np.eye(4)
    different_affine[0, 0] = 2
    _write_label(candidate / "case-b.nii.gz", base, affine=different_affine)
    _write_label(repeat / "case-a.nii.gz", base)
    _write_label(repeat / "case-c.nii.gz", base)
    reference_zip = tmp_path / "reference.zip"
    _zip_flat(reference, reference_zip)

    report = verify_frozen_equivalence(
        reference_zip,
        candidate,
        repeat,
        expected_cases=2,
    )

    assert report["passed"] is False
    assert report["different_voxels"] == 1
    assert report["repeat_case_set_equal"] is False
    assert report["candidate_missing_cases"] == []
    assert report["repeat_missing_cases"] == ["case-b"]
    rows = {row["case_id"]: row for row in report["cases"]}
    assert rows["case-a"]["array_equal"] is False
    assert rows["case-b"]["affine_equal"] is False
