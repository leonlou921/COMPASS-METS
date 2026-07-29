from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_final_inference_scripts_and_configs_exist() -> None:
    for relative in (
        "scripts/08_predict_test_probabilities.sh",
        "scripts/09_build_n03_utility_v4.sh",
        "configs/fusion/xf12.json",
        "configs/final/n03_utility_v4.json",
    ):
        assert (ROOT / relative).is_file(), relative


def test_final_configs_freeze_xf12_n03_and_utility_v4() -> None:
    xf12 = json.loads((ROOT / "configs/fusion/xf12.json").read_text())
    final = json.loads(
        (ROOT / "configs/final/n03_utility_v4.json").read_text()
    )
    assert xf12["candidate_id"] == "XF12_XLM_structured_probability_V2_strict"
    assert final["candidate_id"] == "N03_FINAL_UTILITY_V4"
    assert final["baseline_candidate_id"] == "N03_XF12_LCv3_ET_parent_supported"
    assert final["allowed_add_regions"] == ["ET"]
    assert final["allowed_delete_regions"] == []
    assert final["global_v2_rerun"] is False
    assert final["utility_v4"]["accept_threshold"] == 0.75
    assert final["utility_v4"]["reject_threshold"] == 0.5


def test_public_inference_scripts_never_package_submission_archives() -> None:
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((ROOT / "scripts").glob("*.sh"))
    ).lower()
    assert "package_submission" not in text
    assert "zip " not in text
