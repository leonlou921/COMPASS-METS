#!/usr/bin/env python3
"""Validate N03 outputs and create a flat challenge submission ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable
import zipfile

import nibabel as nib
import numpy as np


CANDIDATE_ID = "N03_FINAL_UTILITY_V4"
LEGAL_LABELS = frozenset({0, 1, 2, 3, 4})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_files(root: Path, expected_cases: int) -> tuple[list[Path], dict[str, Any]]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"prediction directory is absent: {root}")
    entries = sorted(root.rglob("*"))
    foreign = [
        path
        for path in entries
        if path.is_dir()
        or path.parent != root
        or not path.name.endswith(".nii.gz")
    ]
    if foreign:
        raise ValueError(f"output must be flat NIfTI-only: {foreign[:5]}")
    files = [path for path in entries if path.is_file()]
    if len(files) != int(expected_cases):
        raise ValueError(
            f"expected {expected_cases} output cases, found {len(files)}"
        )
    if len({path.name for path in files}) != len(files):
        raise ValueError("duplicate output names")

    labels: set[int] = set()
    aggregate = {str(label): 0 for label in sorted(LEGAL_LABELS)}
    case_ids = []
    for path in files:
        image = nib.load(path)
        array = np.asanyarray(image.dataobj)
        if array.ndim != 3:
            raise ValueError(f"{path.name}: expected a 3D label map")
        if np.dtype(image.get_data_dtype()) != np.dtype(np.uint8):
            raise ValueError(f"{path.name}: output dtype must be uint8")
        if not np.isfinite(image.affine).all():
            raise ValueError(f"{path.name}: affine contains NaN or Inf")
        spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
        if len(spacing) != 3 or any(
            not np.isfinite(value) or value <= 0 for value in spacing
        ):
            raise ValueError(f"{path.name}: invalid spacing {spacing}")
        unique = {int(value) for value in np.unique(array)}
        if not unique.issubset(LEGAL_LABELS):
            raise ValueError(f"{path.name}: illegal labels {sorted(unique)}")
        labels.update(unique)
        for label in LEGAL_LABELS:
            aggregate[str(label)] += int(np.count_nonzero(array == label))
        case_ids.append(path.name[:-7])
    return files, {
        "case_ids": case_ids,
        "labels": sorted(labels),
        "aggregate_label_voxels": aggregate,
    }


def package_submission(
    prediction_root: Path,
    destination: Path,
    *,
    expected_cases: int = 179,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Create and independently reopen a deterministic, flat ZIP."""
    prediction_root = Path(prediction_root).resolve()
    destination = Path(destination).resolve()
    files, audit = _validated_files(prediction_root, expected_cases)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists():
        temporary.unlink()
    with zipfile.ZipFile(
        temporary,
        "w",
        compression=zipfile.ZIP_STORED,
        allowZip64=True,
    ) as archive:
        for path in files:
            info = zipfile.ZipInfo(
                path.name,
                date_time=(1980, 1, 1, 0, 0, 0),
            )
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            archive.writestr(info, path.read_bytes())
    os.replace(temporary, destination)

    with zipfile.ZipFile(destination) as archive:
        infos = archive.infolist()
        crc_ok = archive.testzip() is None
    flat = all(
        not item.is_dir()
        and "/" not in item.filename
        and "\\" not in item.filename
        and item.filename.endswith(".nii.gz")
        for item in infos
    )
    if (
        len(infos) != int(expected_cases)
        or not flat
        or not crc_ok
        or {item.compress_type for item in infos} != {zipfile.ZIP_STORED}
    ):
        raise RuntimeError("submission ZIP validation failed")

    report: dict[str, Any] = {
        "candidate_id": CANDIDATE_ID,
        "prediction_root": str(prediction_root),
        "zip_path": str(destination),
        "zip_sha256": _sha256(destination),
        "zip_size_bytes": destination.stat().st_size,
        "case_count": len(files),
        "case_ids": audit["case_ids"],
        "labels": audit["labels"],
        "aggregate_label_voxels": audit["aggregate_label_voxels"],
        "flat": flat,
        "crc_ok": crc_ok,
        "compression": "stored",
    }
    if manifest_path is not None:
        manifest = Path(manifest_path)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return report


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--expected-cases", type=int, default=179)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    report = package_submission(
        args.prediction_root,
        args.destination,
        expected_cases=args.expected_cases,
        manifest_path=args.manifest,
    )
    print(
        json.dumps(
            {
                "candidate_id": report["candidate_id"],
                "case_count": report["case_count"],
                "zip_sha256": report["zip_sha256"],
                "crc_ok": report["crc_ok"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
