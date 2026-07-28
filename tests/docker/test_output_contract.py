from __future__ import annotations

from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import pytest


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE_ROOT))

from n03_docker.output_contract import (  # noqa: E402
    publish_segmentation,
    validate_flat_output_set,
)


def _reference(path: Path) -> nib.Nifti1Image:
    affine = np.eye(4)
    affine[0, 0] = 1.2
    affine[1, 1] = 1.3
    affine[2, 2] = 1.4
    image = nib.Nifti1Image(np.zeros((5, 6, 7), dtype=np.float32), affine)
    nib.save(image, path)
    return nib.load(path)


def test_publish_is_atomic_uint8_legal_and_geometry_exact(tmp_path: Path) -> None:
    reference = _reference(tmp_path / "reference.nii.gz")
    segmentation = np.zeros(reference.shape, dtype=np.uint8)
    segmentation[1, 2, 3] = 4
    output_root = tmp_path / "output"

    destination = publish_segmentation(
        "BraTS-MET-12345-100",
        segmentation,
        reference,
        output_root,
    )

    assert destination == output_root / "BraTS-MET-12345-100.nii.gz"
    assert not list(output_root.glob("*.tmp*"))
    result = nib.load(destination)
    assert result.get_data_dtype() == np.dtype(np.uint8)
    assert np.array_equal(result.affine, reference.affine)
    assert set(np.unique(np.asarray(result.dataobj))) == {0, 4}


def test_publish_rejects_illegal_labels_without_partial_output(tmp_path: Path) -> None:
    reference = _reference(tmp_path / "reference.nii.gz")
    segmentation = np.zeros(reference.shape, dtype=np.int16)
    segmentation[0, 0, 0] = 5
    output_root = tmp_path / "output"

    with pytest.raises(ValueError, match="illegal labels.*5"):
        publish_segmentation("case-a", segmentation, reference, output_root)

    assert not output_root.exists() or not list(output_root.iterdir())


def test_validate_flat_output_set_rejects_missing_extra_and_nested(
    tmp_path: Path,
) -> None:
    reference = _reference(tmp_path / "reference.nii.gz")
    output_root = tmp_path / "output"
    publish_segmentation(
        "case-a",
        np.zeros(reference.shape, dtype=np.uint8),
        reference,
        output_root,
    )
    validate_flat_output_set(output_root, {"case-a"})

    with pytest.raises(ValueError, match="case set"):
        validate_flat_output_set(output_root, {"case-a", "case-b"})

    nested = output_root / "nested"
    nested.mkdir()
    (nested / "foreign.txt").write_text("not allowed", encoding="utf-8")
    with pytest.raises(ValueError, match="non-flat"):
        validate_flat_output_set(output_root, {"case-a"})
