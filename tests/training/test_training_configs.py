from __future__ import annotations

from pathlib import Path

from brats_mets.training.run_nnunet import load_training_config


ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "configs" / "trainers"


def test_final_and_key_ablation_configs_are_fixed() -> None:
    expected = {
        "resencm": ("nnUNetTrainer", "nnUNetResEncUNetMPlans"),
        "resencxl": ("nnUNetTrainer", "nnUNetResEncUNetXL30GBPlans"),
        "focal_tversky": (
            "nnUNetTrainer_ResEncM_DiceCEFocalTverskyFT",
            "nnUNetResEncUNetMPlans",
        ),
        "small_lesion_os": (
            "nnUNetTrainer_ResEncM_SmallLesionOS",
            "nnUNetResEncUNetMPlans",
        ),
        "component_small_lesion_os": (
            "nnUNetTrainer_ResEncM_ComponentSmallLesionOS",
            "nnUNetResEncUNetMPlans",
        ),
    }
    for model_id, (trainer, plans) in expected.items():
        config = load_training_config(CONFIG_ROOT / f"{model_id}.json")
        assert config["model_id"] == model_id
        assert config["dataset_id"] == 501
        assert config["configuration"] == "3d_fullres"
        assert config["folds"] == [0, 1, 2, 3, 4]
        assert config["trainer"] == trainer
        assert config["plans"] == plans
        assert config["checkpoint"] == "checkpoint_best.pth"


def test_only_focal_tversky_uses_fold_matched_pretraining() -> None:
    for path in CONFIG_ROOT.glob("*.json"):
        config = load_training_config(path)
        if config["model_id"] == "focal_tversky":
            assert config["pretrained_from"] == {
                "trainer": "nnUNetTrainer",
                "plans": "nnUNetResEncUNetMPlans",
            }
        else:
            assert "pretrained_from" not in config
