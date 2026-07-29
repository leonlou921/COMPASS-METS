from __future__ import annotations

from pathlib import Path

import pytest

from brats_mets.data.contracts import (
    DATASET_ID,
    DATASET_NAME,
    MODALITIES,
    REGIONS_CLASS_ORDER,
    build_dataset_json,
    validate_case_id,
)


def test_dataset501_region_contract_is_frozen() -> None:
    assert DATASET_ID == 501
    assert DATASET_NAME == "BraTS2025MET"
    assert MODALITIES == ("t1c", "t1n", "t2f", "t2w")
    assert REGIONS_CLASS_ORDER == (2, 1, 3, 4)
    dataset = build_dataset_json(num_training=7)
    assert dataset["channel_names"] == {
        "0": "t1c",
        "1": "t1n",
        "2": "t2f",
        "3": "t2w",
    }
    assert dataset["labels"] == {
        "background": 0,
        "WT": [1, 2, 3],
        "TC": [1, 3],
        "ET": 3,
        "RC": 4,
    }
    assert dataset["regions_class_order"] == [2, 1, 3, 4]
    assert dataset["numTraining"] == 7


@pytest.mark.parametrize(
    "case_id",
    ["BraTS-MET-00001-000", "BraTS-MET-12345-999"],
)
def test_case_id_contract_accepts_expected_ids(case_id: str) -> None:
    assert validate_case_id(case_id) == case_id


@pytest.mark.parametrize(
    "case_id",
    ["", "../BraTS-MET-00001-000", "BraTS-GLI-00001-000", "case 1"],
)
def test_case_id_contract_rejects_unsafe_or_wrong_ids(case_id: str) -> None:
    with pytest.raises(ValueError):
        validate_case_id(case_id)


def test_public_paths_example_contains_names_not_private_values() -> None:
    text = (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "dataset501"
        / "paths.env.example"
    ).read_text(encoding="utf-8")
    for variable in (
        "BRATS_METS_TRAIN_DIR",
        "BRATS_METS_VALID_DIR",
        "nnUNet_raw",
        "nnUNet_preprocessed",
        "nnUNet_results",
    ):
        assert variable in text
    assert "root@" not in text
    assert "deepln" not in text
    assert "funhpc" not in text
