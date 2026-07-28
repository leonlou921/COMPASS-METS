from __future__ import annotations

import json
from pathlib import Path

from scripts.verify_release import inspect_release


def test_release_verifier_rejects_credentials_and_private_absolute_paths(
    tmp_path: Path,
) -> None:
    (tmp_path / "safe.py").write_text("print('ok')\n", encoding="utf-8")
    credential = "pass" + "word = supersecret"
    private_path = "/data/" + "coding/challenge/private"
    (tmp_path / "bad.txt").write_text(
        f"{credential}\n{private_path}\n", encoding="utf-8"
    )
    report = inspect_release(tmp_path)
    assert report["passed"] is False
    kinds = {finding["kind"] for finding in report["findings"]}
    assert "credential" in kinds
    assert "private_absolute_path" in kinds


def test_release_verifier_accepts_documented_placeholders_and_frozen_config(
    tmp_path: Path,
) -> None:
    (tmp_path / "README.md").write_text(
        "Use /path/to/data and set SSH_PASSWORD in your own shell.\n",
        encoding="utf-8",
    )
    config_root = tmp_path / "configs" / "n03"
    config_root.mkdir(parents=True)
    (config_root / "final.json").write_text(
        json.dumps(
            {
                "candidate_id": "N03_XF12_LCv3_ET_parent_supported",
                "proposal_threshold": 0.25,
                "et_component_cutoff": 0.5497123599,
                "minimum_parent_model_support": 2,
                "parent_models": ["XL", "M", "FT"],
                "policy": "add_only_et_parent_supported",
            }
        ),
        encoding="utf-8",
    )
    report = inspect_release(tmp_path)
    assert report["passed"] is True
    assert report["n03_config_valid"] is True
