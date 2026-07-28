from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.run_nnunet import build_nnunet_command, load_model_registry


def _registry(tmp_path: Path) -> Path:
    path = tmp_path / "models.json"
    path.write_text(
        json.dumps(
            {
                "dataset_id": 501,
                "configuration": "3d_fullres",
                "models": {
                    "xl": {
                        "trainer": "nnUNetTrainer",
                        "plans": "nnUNetResEncUNetXL30GBPlans",
                    },
                    "m": {
                        "trainer": "nnUNetTrainer",
                        "plans": "nnUNetResEncUNetMPlans",
                    },
                    "ft": {
                        "trainer": "nnUNetTrainer_ResEncM_DiceCEFocalTverskyFT",
                        "plans": "nnUNetResEncUNetMPlans",
                        "pretrained_from": "m",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_training_command_uses_frozen_model_identity(tmp_path: Path) -> None:
    registry = load_model_registry(_registry(tmp_path))
    command = build_nnunet_command(registry, "xl", fold=3, action="train")
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
    registry = load_model_registry(_registry(tmp_path))
    with pytest.raises(ValueError, match="pretrained checkpoint"):
        build_nnunet_command(registry, "ft", fold=1, action="train")

    checkpoint = tmp_path / "fold_1" / "checkpoint_best.pth"
    command = build_nnunet_command(
        registry,
        "ft",
        fold=1,
        action="train",
        pretrained_checkpoint=checkpoint,
    )
    assert command[-2:] == ["-pretrained_weights", str(checkpoint)]


def test_validation_command_is_best_checkpoint_oof_export(tmp_path: Path) -> None:
    registry = load_model_registry(_registry(tmp_path))
    command = build_nnunet_command(registry, "m", fold=0, action="validate")
    assert command[-3:] == ["--val", "--val_best", "--npz"]


def test_ordered_public_pipeline_scripts_cover_preprocess_training_and_gates() -> None:
    root = Path(__file__).resolve().parents[1]
    preprocess = (root / "preprocessing" / "plan_and_preprocess.sh").read_text(
        encoding="utf-8"
    )
    gates = (root / "training" / "train_learned_gates.sh").read_text(
        encoding="utf-8"
    )
    all_steps = (root / "scripts" / "run_training_pipeline.sh").read_text(
        encoding="utf-8"
    )
    assert "nnUNetv2_plan_and_preprocess" in preprocess
    assert "--verify_dataset_integrity" in preprocess
    assert "build_features.py" in gates
    assert "train_models_v2.py" in gates
    assert "consolidate-calibration" in gates
    assert all_steps.index("--model m") < all_steps.index("--model ft")
    assert all_steps.index("--model ft") < all_steps.index("--model xl")
