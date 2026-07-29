from __future__ import annotations

import json
from pathlib import Path

import pytest

from compass_mets.training.run_nnunet import (
    build_nnunet_command,
    load_training_config,
)


def _config(tmp_path: Path, model_id: str = "resencxl") -> Path:
    path = tmp_path / f"{model_id}.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_id": model_id,
                "dataset_id": 501,
                "configuration": "3d_fullres",
                "folds": [0, 1, 2, 3, 4],
                "trainer": "nnUNetTrainer",
                "plans": "nnUNetResEncUNetXL30GBPlans",
                "checkpoint": "checkpoint_best.pth",
            }
        ),
        encoding="utf-8",
    )
    return path


def test_training_command_uses_frozen_model_identity(tmp_path: Path) -> None:
    config = load_training_config(_config(tmp_path))
    command = build_nnunet_command(config, fold=3, action="train")
    assert command == [
        "nnUNetv2_train",
        "501",
        "3d_fullres",
        "3",
        "-tr",
        "nnUNetTrainer",
        "-p",
        "nnUNetResEncUNetXL30GBPlans",
        "--npz",
    ]


def test_ft_training_requires_matching_m_checkpoint(tmp_path: Path) -> None:
    config_path = _config(tmp_path, "focal_tversky")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update(
        {
            "trainer": "nnUNetTrainer_ResEncM_DiceCEFocalTverskyFT",
            "plans": "nnUNetResEncUNetMPlans",
            "pretrained_from": {
                "trainer": "nnUNetTrainer",
                "plans": "nnUNetResEncUNetMPlans",
            },
        }
    )
    config_path.write_text(json.dumps(config), encoding="utf-8")
    config = load_training_config(config_path)
    with pytest.raises(ValueError, match="pretrained checkpoint"):
        build_nnunet_command(config, fold=1, action="train")

    checkpoint = tmp_path / "fold_1" / "checkpoint_best.pth"
    command = build_nnunet_command(
        config,
        fold=1,
        action="train",
        pretrained_checkpoint=checkpoint,
    )
    assert command[-2:] == ["-pretrained_weights", str(checkpoint)]


def test_validation_command_is_best_checkpoint_oof_export(tmp_path: Path) -> None:
    config = load_training_config(_config(tmp_path, "resencm"))
    command = build_nnunet_command(config, fold=0, action="validate")
    assert command[-3:] == ["--val", "--val_best", "--npz"]


def test_ordered_public_pipeline_scripts_cover_preprocess_training_and_gates() -> None:
    root = Path(__file__).resolve().parents[2]
    preprocess = (root / "scripts" / "02_plan_and_preprocess.sh").read_text(
        encoding="utf-8"
    )
    assert "nnUNetv2_plan_and_preprocess" in preprocess
    assert "--verify_dataset_integrity" in preprocess
    for script in (
        "03_train_resencm_5fold.sh",
        "04_train_resencxl_5fold.sh",
        "05_train_small_lesion_ft_5fold.sh",
    ):
        assert (root / "scripts" / script).is_file()
