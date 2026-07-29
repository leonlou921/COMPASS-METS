#!/usr/bin/env python3
"""Compare fresh Docker outputs with the external frozen N03 reference ZIP."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import tempfile
from typing import Any, Iterable
import zipfile

import nibabel as nib
import numpy as np


CANDIDATE_ID = "N03_FINAL_UTILITY_V4"
LEGAL_LABELS = (0, 1, 2, 3, 4)


def _case_id(path: Path) -> str:
    if not path.name.endswith(".nii.gz"):
        raise ValueError(f"not a NIfTI gzip file: {path.name}")
    return path.name[:-7]


def _directory_cases(root: Path) -> dict[str, Path]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"prediction directory is absent: {root}")
    files = sorted(root.glob("*.nii.gz"))
    mapping = {_case_id(path): path for path in files}
    if len(mapping) != len(files):
        raise RuntimeError(f"duplicate case names in {root}")
    return mapping


def _extract_flat_reference(reference_zip: Path, destination: Path) -> dict[str, Path]:
    with zipfile.ZipFile(reference_zip) as archive:
        infos = archive.infolist()
        names = [item.filename for item in infos]
        if archive.testzip() is not None:
            raise RuntimeError("reference ZIP CRC validation failed")
        if any(
            item.is_dir()
            or "/" in item.filename
            or "\\" in item.filename
            or not item.filename.endswith(".nii.gz")
            for item in infos
        ):
            raise RuntimeError("reference ZIP must contain only flat .nii.gz files")
        if len(names) != len(set(names)):
            raise RuntimeError("reference ZIP contains duplicate entries")
        for item in infos:
            target = destination / item.filename
            with archive.open(item) as source, target.open("wb") as sink:
                while chunk := source.read(1024 * 1024):
                    sink.write(chunk)
    return _directory_cases(destination)


def _load(path: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    image = nib.load(path)
    array = np.asanyarray(image.dataobj)
    return image, array


def _spacing(image: nib.Nifti1Image) -> tuple[float, ...]:
    return tuple(float(value) for value in image.header.get_zooms()[:3])


def _dtype(image: nib.Nifti1Image) -> str:
    return np.dtype(image.get_data_dtype()).name


def _label_counts(array: np.ndarray) -> dict[str, int]:
    return {
        str(label): int(np.count_nonzero(array == label))
        for label in LEGAL_LABELS
    }


def _add_counts(total: dict[str, int], counts: dict[str, int]) -> None:
    for label, value in counts.items():
        total[label] += value


def _empty_counts() -> dict[str, int]:
    return {str(label): 0 for label in LEGAL_LABELS}


def _compare_images(
    reference_image: nib.Nifti1Image,
    reference_array: np.ndarray,
    other_image: nib.Nifti1Image,
    other_array: np.ndarray,
) -> dict[str, Any]:
    shape_equal = tuple(reference_array.shape) == tuple(other_array.shape)
    affine_equal = np.array_equal(reference_image.affine, other_image.affine)
    spacing_equal = _spacing(reference_image) == _spacing(other_image)
    dtype_equal = _dtype(reference_image) == _dtype(other_image)
    other_labels = sorted(int(value) for value in np.unique(other_array))
    labels_legal = set(other_labels).issubset(LEGAL_LABELS)
    if shape_equal:
        changed_voxels = int(np.count_nonzero(reference_array != other_array))
        array_equal = changed_voxels == 0
    else:
        changed_voxels = None
        array_equal = False
    return {
        "shape_equal": shape_equal,
        "affine_equal": affine_equal,
        "spacing_equal": spacing_equal,
        "dtype_equal": dtype_equal,
        "labels_legal": labels_legal,
        "labels": other_labels,
        "changed_voxels": changed_voxels,
        "array_equal": array_equal,
    }


def _write_csv(rows: list[dict[str, Any]], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_id",
        "candidate_present",
        "repeat_present",
        "array_equal",
        "changed_voxels",
        "shape_equal",
        "affine_equal",
        "spacing_equal",
        "dtype_equal",
        "labels_legal",
        "repeat_array_equal",
        "repeat_changed_voxels",
        "repeat_shape_equal",
        "repeat_affine_equal",
        "repeat_spacing_equal",
        "repeat_dtype_equal",
        "repeat_labels_legal",
        "candidate_repeat_array_equal",
        "candidate_repeat_changed_voxels",
    ]
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def verify_frozen_equivalence(
    reference_zip: Path,
    candidate_dir: Path,
    repeat_dir: Path,
    *,
    expected_cases: int = 179,
    json_output: Path | None = None,
    csv_output: Path | None = None,
) -> dict[str, Any]:
    reference_zip = Path(reference_zip).resolve()
    candidate_dir = Path(candidate_dir).resolve()
    repeat_dir = Path(repeat_dir).resolve()
    candidate = _directory_cases(candidate_dir)
    repeat = _directory_cases(repeat_dir)

    with tempfile.TemporaryDirectory(prefix="n03-frozen-reference-") as temporary:
        reference = _extract_flat_reference(
            reference_zip, Path(temporary)
        )
        reference_ids = set(reference)
        candidate_ids = set(candidate)
        repeat_ids = set(repeat)

        candidate_missing = sorted(reference_ids - candidate_ids)
        candidate_extra = sorted(candidate_ids - reference_ids)
        repeat_missing = sorted(reference_ids - repeat_ids)
        repeat_extra = sorted(repeat_ids - reference_ids)

        aggregate = {
            "reference": _empty_counts(),
            "candidate": _empty_counts(),
            "repeat": _empty_counts(),
        }
        rows: list[dict[str, Any]] = []
        different_voxels = 0
        repeat_different_voxels = 0
        incomparable_candidate = 0
        incomparable_repeat = 0

        for case_id in sorted(reference):
            reference_image, reference_array = _load(reference[case_id])
            reference_labels = sorted(
                int(value) for value in np.unique(reference_array)
            )
            reference_legal = set(reference_labels).issubset(LEGAL_LABELS)
            _add_counts(aggregate["reference"], _label_counts(reference_array))
            row: dict[str, Any] = {
                "case_id": case_id,
                "candidate_present": case_id in candidate,
                "repeat_present": case_id in repeat,
                "reference_labels_legal": reference_legal,
            }
            candidate_image = None
            candidate_array = None
            if case_id in candidate:
                candidate_image, candidate_array = _load(candidate[case_id])
                result = _compare_images(
                    reference_image,
                    reference_array,
                    candidate_image,
                    candidate_array,
                )
                row.update(result)
                _add_counts(
                    aggregate["candidate"], _label_counts(candidate_array)
                )
                if result["changed_voxels"] is None:
                    incomparable_candidate += 1
                else:
                    different_voxels += int(result["changed_voxels"])
            else:
                incomparable_candidate += 1

            if case_id in repeat:
                repeat_image, repeat_array = _load(repeat[case_id])
                repeat_result = _compare_images(
                    reference_image,
                    reference_array,
                    repeat_image,
                    repeat_array,
                )
                row.update(
                    {
                        f"repeat_{key}": value
                        for key, value in repeat_result.items()
                    }
                )
                _add_counts(aggregate["repeat"], _label_counts(repeat_array))
                if (
                    candidate_image is not None
                    and candidate_array is not None
                ):
                    run_result = _compare_images(
                        candidate_image,
                        candidate_array,
                        repeat_image,
                        repeat_array,
                    )
                    row["candidate_repeat_array_equal"] = run_result[
                        "array_equal"
                    ]
                    row["candidate_repeat_changed_voxels"] = run_result[
                        "changed_voxels"
                    ]
                    if run_result["changed_voxels"] is None:
                        incomparable_repeat += 1
                    else:
                        repeat_different_voxels += int(
                            run_result["changed_voxels"]
                        )
                else:
                    incomparable_repeat += 1
            else:
                incomparable_repeat += 1
            rows.append(row)

    row_gate = all(
        row.get("reference_labels_legal") is True
        and row.get("array_equal") is True
        and row.get("shape_equal") is True
        and row.get("affine_equal") is True
        and row.get("spacing_equal") is True
        and row.get("dtype_equal") is True
        and row.get("labels_legal") is True
        and row.get("repeat_array_equal") is True
        and row.get("repeat_shape_equal") is True
        and row.get("repeat_affine_equal") is True
        and row.get("repeat_spacing_equal") is True
        and row.get("repeat_dtype_equal") is True
        and row.get("repeat_labels_legal") is True
        and row.get("candidate_repeat_array_equal") is True
        for row in rows
    )
    case_count = len(reference_ids)
    candidate_case_set_equal = candidate_ids == reference_ids
    repeat_case_set_equal = repeat_ids == reference_ids
    aggregate_equal = (
        aggregate["reference"]
        == aggregate["candidate"]
        == aggregate["repeat"]
    )
    passed = (
        case_count == int(expected_cases)
        and candidate_case_set_equal
        and repeat_case_set_equal
        and different_voxels == 0
        and repeat_different_voxels == 0
        and incomparable_candidate == 0
        and incomparable_repeat == 0
        and aggregate_equal
        and row_gate
    )
    report: dict[str, Any] = {
        "candidate_id": CANDIDATE_ID,
        "passed": passed,
        "expected_case_count": int(expected_cases),
        "case_count": case_count,
        "candidate_case_count": len(candidate_ids),
        "repeat_case_count": len(repeat_ids),
        "candidate_case_set_equal": candidate_case_set_equal,
        "repeat_case_set_equal": repeat_case_set_equal,
        "candidate_missing_cases": candidate_missing,
        "candidate_extra_cases": candidate_extra,
        "repeat_missing_cases": repeat_missing,
        "repeat_extra_cases": repeat_extra,
        "different_voxels": different_voxels,
        "repeat_different_voxels": repeat_different_voxels,
        "incomparable_candidate_cases": incomparable_candidate,
        "incomparable_repeat_cases": incomparable_repeat,
        "aggregate_voxel_counts": aggregate,
        "aggregate_voxel_counts_equal": aggregate_equal,
        "cases": rows,
    }
    if json_output is not None:
        destination = Path(json_output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if csv_output is not None:
        _write_csv(rows, Path(csv_output))
    return report


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-zip", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--repeat-dir", type=Path, required=True)
    parser.add_argument("--expected-cases", type=int, default=179)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    report = verify_frozen_equivalence(
        args.reference_zip,
        args.candidate_dir,
        args.repeat_dir,
        expected_cases=args.expected_cases,
        json_output=args.json_output,
        csv_output=args.csv_output,
    )
    print(
        json.dumps(
            {
                "candidate_id": report["candidate_id"],
                "passed": report["passed"],
                "case_count": report["case_count"],
                "different_voxels": report["different_voxels"],
                "repeat_different_voxels": report[
                    "repeat_different_voxels"
                ],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
