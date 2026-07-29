"""Validate and atomically publish final N03 segmentations."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np


VALID_LABELS = frozenset({0, 1, 2, 3, 4})


def _validated_uint8(segmentation: np.ndarray) -> np.ndarray:
    values = np.asarray(segmentation)
    if not np.isfinite(values).all():
        raise ValueError("segmentation contains NaN or Inf")
    unique = np.unique(values)
    if not np.equal(unique, np.floor(unique)).all():
        raise ValueError(f"segmentation contains non-integer labels: {unique.tolist()}")
    labels = {int(value) for value in unique}
    illegal = sorted(labels.difference(VALID_LABELS))
    if illegal:
        raise ValueError(f"illegal labels: {illegal}")
    return values.astype(np.uint8, copy=False)


def publish_segmentation(
    case_id: str,
    segmentation: np.ndarray,
    reference: nib.Nifti1Image,
    output_root: Path,
) -> Path:
    if not case_id or "/" in case_id or "\\" in case_id:
        raise ValueError(f"invalid case_id: {case_id!r}")
    output = _validated_uint8(segmentation)
    if tuple(output.shape) != tuple(reference.shape):
        raise ValueError(
            f"output shape {output.shape} differs from reference {reference.shape}"
        )

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / f"{case_id}.nii.gz"
    temporary = output_root / f".{case_id}.tmp.nii.gz"
    if temporary.exists():
        temporary.unlink()

    header = reference.header.copy()
    header.set_data_dtype(np.uint8)
    image = nib.Nifti1Image(output, reference.affine, header)
    nib.save(image, temporary)
    written = nib.load(temporary)
    if written.get_data_dtype() != np.dtype(np.uint8):
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"{case_id}: written dtype is not uint8")
    if tuple(written.shape) != tuple(reference.shape) or not np.allclose(
        written.affine,
        reference.affine,
        rtol=0.0,
        atol=1e-5,
    ):
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"{case_id}: written geometry drifted")
    written_labels = {int(value) for value in np.unique(np.asarray(written.dataobj))}
    if not written_labels.issubset(VALID_LABELS):
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"{case_id}: written file contains illegal labels")

    os.replace(temporary, destination)
    return destination


def validate_flat_output_set(output_root: Path, expected_case_ids: Iterable[str]) -> None:
    output_root = Path(output_root)
    if not output_root.is_dir():
        raise FileNotFoundError(f"output root does not exist: {output_root}")
    nested_or_foreign = [
        path
        for path in output_root.rglob("*")
        if path.parent != output_root or (path.is_file() and not path.name.endswith(".nii.gz"))
    ]
    if nested_or_foreign:
        raise ValueError(f"non-flat or foreign output entries: {nested_or_foreign[:5]}")

    actual = {
        path.name[: -len(".nii.gz")]
        for path in output_root.glob("*.nii.gz")
        if path.is_file()
    }
    expected = set(expected_case_ids)
    if actual != expected:
        raise ValueError(
            f"output case set differs: missing={sorted(expected - actual)} "
            f"extra={sorted(actual - expected)}"
        )
