from __future__ import annotations

import json
from pathlib import Path
import zipfile

import nibabel as nib
import numpy as np
import pytest

from verification.package_submission import package_submission


def _write_label(path: Path, label: int) -> None:
    array = np.zeros((3, 4, 5), dtype=np.uint8)
    array[1, 2, 3] = label
    image = nib.Nifti1Image(array, np.eye(4))
    image.header.set_data_dtype(np.uint8)
    nib.save(image, path)


def test_package_submission_builds_flat_crc_valid_zip_and_manifest(
    tmp_path: Path,
) -> None:
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    _write_label(predictions / "case-a.nii.gz", 3)
    _write_label(predictions / "case-b.nii.gz", 4)
    destination = tmp_path / "submission.zip"
    manifest = tmp_path / "submission.json"

    report = package_submission(
        predictions,
        destination,
        expected_cases=2,
        manifest_path=manifest,
    )

    assert report["candidate_id"] == "N03_FINAL_UTILITY_V4"
    assert report["case_count"] == 2
    assert report["flat"] is True
    assert report["crc_ok"] is True
    assert report["labels"] == [0, 3, 4]
    assert json.loads(manifest.read_text(encoding="utf-8")) == report
    with zipfile.ZipFile(destination) as archive:
        assert archive.namelist() == ["case-a.nii.gz", "case-b.nii.gz"]
        assert archive.testzip() is None


def test_package_submission_rejects_foreign_or_illegal_output(
    tmp_path: Path,
) -> None:
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    _write_label(predictions / "case-a.nii.gz", 3)
    (predictions / "audit.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="flat"):
        package_submission(
            predictions,
            tmp_path / "submission.zip",
            expected_cases=1,
        )
