from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_numbered_preprocessing_scripts_use_public_package_and_environment() -> None:
    prepare = (ROOT / "scripts" / "01_prepare_dataset.sh").read_text(
        encoding="utf-8"
    )
    plan = (ROOT / "scripts" / "02_plan_and_preprocess.sh").read_text(
        encoding="utf-8"
    )
    assert "-m brats_mets.data.prepare_dataset501" in prepare
    assert "BRATS_METS_TRAIN_DIR" in prepare
    assert "BRATS_METS_VALID_DIR" in prepare
    assert "--overwrite" not in prepare
    assert "nnUNetPlannerResEncM" in plan
    assert "nnUNetPlannerResEncXL" in plan
    assert "--verify_dataset_integrity" in plan
