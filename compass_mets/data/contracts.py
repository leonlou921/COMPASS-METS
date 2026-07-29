"""Frozen public contract for Dataset501_BraTS2025MET."""

from __future__ import annotations

import re
from typing import Sequence


DATASET_ID = 501
DATASET_NAME = "BraTS2025MET"
MODALITIES = ("t1c", "t1n", "t2f", "t2w")
FLAT_LABELS = {
    "background": 0,
    "NETC": 1,
    "SNFH": 2,
    "ET": 3,
    "RC": 4,
}
REGION_LABELS = {
    "background": 0,
    "WT": (1, 2, 3),
    "TC": (1, 3),
    "ET": 3,
    "RC": 4,
}
REGIONS_CLASS_ORDER = (2, 1, 3, 4)
_CASE_ID = re.compile(r"^BraTS-MET-\d{5}-\d{3}$")


def validate_case_id(case_id: str) -> str:
    if not _CASE_ID.fullmatch(case_id):
        raise ValueError(f"invalid BraTS-METS case identifier: {case_id!r}")
    return case_id


def _json_labels(labels: dict[str, int | tuple[int, ...]]) -> dict[str, int | list[int]]:
    return {
        name: list(value) if isinstance(value, tuple) else value
        for name, value in labels.items()
    }


def build_dataset_json(
    num_training: int,
    modalities: Sequence[str] = MODALITIES,
    label_mode: str = "regions",
) -> dict:
    if num_training < 0:
        raise ValueError("num_training must be non-negative")
    if tuple(modalities) != MODALITIES:
        raise ValueError(f"modality order must be {MODALITIES!r}")
    if label_mode not in {"regions", "flat"}:
        raise ValueError("label_mode must be 'regions' or 'flat'")
    labels = REGION_LABELS if label_mode == "regions" else FLAT_LABELS
    result = {
        "channel_names": {
            str(index): modality for index, modality in enumerate(modalities)
        },
        "labels": _json_labels(labels),
        "numTraining": int(num_training),
        "file_ending": ".nii.gz",
    }
    if label_mode == "regions":
        result["regions_class_order"] = list(REGIONS_CLASS_ORDER)
    return result
