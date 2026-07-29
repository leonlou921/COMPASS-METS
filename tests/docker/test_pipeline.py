from __future__ import annotations

from pathlib import Path
import sys

import joblib
import nibabel as nib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from n03_docker.pipeline import (  # noqa: E402
    DEFAULT_MODEL_SPECS,
    _load_learned_bundles,
    label_zyx_to_reference_xyz,
)


def test_default_specs_freeze_exact_three_model_sources() -> None:
    assert list(DEFAULT_MODEL_SPECS) == ["XL", "M", "FT"]
    assert DEFAULT_MODEL_SPECS["XL"].trainer == "nnUNetTrainer"
    assert DEFAULT_MODEL_SPECS["XL"].plans == "nnUNetResEncUNetXL30GBPlans"
    assert DEFAULT_MODEL_SPECS["M"].trainer == "nnUNetTrainer"
    assert DEFAULT_MODEL_SPECS["M"].plans == "nnUNetResEncUNetMPlans"
    assert (
        DEFAULT_MODEL_SPECS["FT"].trainer
        == "nnUNetTrainer_ResEncM_DiceCEFocalTverskyFT"
    )
    assert DEFAULT_MODEL_SPECS["FT"].plans == "nnUNetResEncUNetMPlans"


def test_zyx_postprocess_label_is_transposed_back_to_reference_xyz() -> None:
    reference = nib.Nifti1Image(
        np.zeros((5, 4, 3), dtype=np.float32),
        np.eye(4),
    )
    label_zyx = np.zeros((3, 4, 5), dtype=np.uint8)
    label_zyx[2, 1, 4] = 3

    output_xyz = label_zyx_to_reference_xyz(label_zyx, reference)

    assert output_xyz.shape == reference.shape
    assert output_xyz[4, 1, 2] == 3


def test_zyx_conversion_rejects_wrong_shape() -> None:
    reference = nib.Nifti1Image(
        np.zeros((5, 4, 3), dtype=np.float32),
        np.eye(4),
    )
    with np.testing.assert_raises_regex(ValueError, "does not match reference"):
        label_zyx_to_reference_xyz(
            np.zeros((3, 4, 4), dtype=np.uint8),
            reference,
        )


def test_runtime_loads_every_final_gate_asset(tmp_path: Path) -> None:
    learned = tmp_path / "learned_models"
    for role in ("lcv1_case", "lcv2_component", "rgv3_et"):
        destination = learned / role / "models.joblib"
        destination.parent.mkdir(parents=True)
        joblib.dump({"role": role}, destination)
    utility = learned / "utility_v4"
    utility.mkdir()
    joblib.dump({"role": "existence"}, utility / "existence_model.joblib")
    joblib.dump({"role": "geometry"}, utility / "geometry_model.joblib")
    (utility / "feature_names.json").write_text(
        '["v2_component_probability", "v3_component_probability"]',
        encoding="utf-8",
    )

    bundles = _load_learned_bundles(tmp_path)

    assert set(bundles) == {
        "lcv1_case",
        "lcv2_component",
        "rgv3_et",
        "utility_v4_existence",
        "utility_v4_geometry",
        "utility_v4_feature_names",
    }
    assert bundles["rgv3_et"]["role"] == "rgv3_et"
    assert bundles["utility_v4_feature_names"] == [
        "v2_component_probability",
        "v3_component_probability",
    ]
