"""Discover read-only BraTS MRI inputs and stage nnU-Net-compatible names."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Mapping

import nibabel as nib
import numpy as np


MODALITIES = ("t1c", "t1n", "t2f", "t2w")
CHANNEL_TO_MODALITY = {index: name for index, name in enumerate(MODALITIES)}
CHANNEL_PATTERN = re.compile(r"^(?P<case>.+)_000(?P<channel>[0-3])\.nii\.gz$")
NAMED_PATTERN = re.compile(
    r"^(?P<case>.+?)[_-](?P<modality>t1c|t1n|t2f|t2w)\.nii\.gz$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CaseInput:
    case_id: str
    modalities: Mapping[str, Path]
    reference_path: Path


def _parse_input_name(path: Path) -> tuple[str, str] | None:
    match = CHANNEL_PATTERN.match(path.name)
    if match:
        return (
            match.group("case"),
            CHANNEL_TO_MODALITY[int(match.group("channel"))],
        )
    match = NAMED_PATTERN.match(path.name)
    if match:
        return match.group("case"), match.group("modality").lower()
    return None


def _geometry_signature(path: Path) -> tuple[tuple[int, ...], np.ndarray, tuple[float, ...]]:
    image = nib.load(path)
    return (
        tuple(int(value) for value in image.shape),
        np.asarray(image.affine, dtype=np.float64),
        tuple(float(value) for value in image.header.get_zooms()[: len(image.shape)]),
    )


def _validate_geometry(case_id: str, modalities: Mapping[str, Path]) -> None:
    reference_modality = MODALITIES[0]
    reference_shape, reference_affine, reference_zooms = _geometry_signature(
        modalities[reference_modality]
    )
    for modality in MODALITIES[1:]:
        shape, affine, zooms = _geometry_signature(modalities[modality])
        if (
            shape != reference_shape
            or zooms != reference_zooms
            or not np.allclose(affine, reference_affine, rtol=0.0, atol=1e-5)
        ):
            raise ValueError(
                f"{case_id} geometry mismatch for {modality}: "
                f"shape={shape} zooms={zooms}"
            )


def discover_cases(input_root: Path) -> list[CaseInput]:
    input_root = Path(input_root)
    if not input_root.is_dir():
        raise FileNotFoundError(f"input root is not a directory: {input_root}")

    grouped: dict[str, dict[str, Path]] = {}
    for path in sorted(input_root.rglob("*.nii.gz")):
        parsed = _parse_input_name(path)
        if parsed is None:
            continue
        case_id, modality = parsed
        case_modalities = grouped.setdefault(case_id, {})
        if modality in case_modalities:
            raise ValueError(
                f"duplicate modality for {case_id} {modality}: "
                f"{case_modalities[modality]} and {path}"
            )
        case_modalities[modality] = path.resolve()

    if not grouped:
        raise ValueError(f"no recognized four-modality NIfTI cases in {input_root}")

    result = []
    for case_id in sorted(grouped):
        modalities = grouped[case_id]
        missing = [name for name in MODALITIES if name not in modalities]
        if missing:
            raise ValueError(f"{case_id} missing modalities: {', '.join(missing)}")
        _validate_geometry(case_id, modalities)
        result.append(
            CaseInput(
                case_id=case_id,
                modalities=dict(modalities),
                reference_path=modalities["t1c"],
            )
        )
    return result


def stage_nnunet_inputs(cases: list[CaseInput], staging_root: Path) -> list[Path]:
    staging_root = Path(staging_root)
    staging_root.mkdir(parents=True, exist_ok=True)
    if any(staging_root.iterdir()):
        raise ValueError(f"nnU-Net staging root must be empty: {staging_root}")

    staged = []
    for case in cases:
        for channel, modality in CHANNEL_TO_MODALITY.items():
            destination = staging_root / f"{case.case_id}_{channel:04d}.nii.gz"
            source = Path(case.modalities[modality]).resolve()
            os.symlink(source, destination)
            staged.append(destination)
    return staged
