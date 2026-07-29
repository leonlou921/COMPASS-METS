from __future__ import annotations

import json
from pathlib import Path

from compass_mets.learned_gates.contracts import (
    RGV3_ET_CUTOFF,
    UTILITY_ACCEPT_THRESHOLD,
    UTILITY_REJECT_THRESHOLD,
    utility_decision,
)


ROOT = Path(__file__).resolve().parents[2]


def test_frozen_gate_configs_match_runtime_contract() -> None:
    gate_root = ROOT / "configs" / "learned_gates"
    rgv3 = json.loads((gate_root / "rgv3.json").read_text(encoding="utf-8"))
    utility = json.loads(
        (gate_root / "utility_v4.json").read_text(encoding="utf-8")
    )
    assert rgv3["folds"] == [0, 1, 2, 3, 4]
    assert rgv3["precision_floors"] == {"ET": 0.97, "TC": 0.98, "WT": 0.98}
    assert utility["candidate_pool"]["rgv3_et_cutoff"] == RGV3_ET_CUTOFF
    assert utility["decision"]["accept_threshold"] == UTILITY_ACCEPT_THRESHOLD
    assert utility["decision"]["reject_threshold"] == UTILITY_REJECT_THRESHOLD


def test_utility_v4_three_state_policy() -> None:
    assert utility_decision(0.80, 0.75, 0.90) == "accept"
    assert utility_decision(0.99, 0.49, 0.99) == "reject"
    assert utility_decision(0.99, 0.99, 0.49) == "reject"
    assert utility_decision(0.74, 0.70, 0.70) == "rgv3_fallback"


def test_learned_gate_scripts_are_package_driven_and_archive_free() -> None:
    script = (ROOT / "scripts" / "07_train_learned_gates.sh").read_text(
        encoding="utf-8"
    )
    assert "compass_mets.learned_gates.lcv1" in script
    assert "compass_mets.learned_gates.rgv3" in script
    assert "compass_mets.learned_gates.train_utility_v4" in script
    assert "submission" not in script.lower()
    assert ".zip" not in script.lower()
