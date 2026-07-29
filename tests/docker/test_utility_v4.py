from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from n03_docker.utility_v4 import (
    add_accepted_et,
    three_state_decision,
)


def test_gate_accepts_only_when_all_required_scores_reach_075() -> None:
    frame = pd.DataFrame(
        {
            "v2_component_probability": [0.75, 0.74, 0.90],
            "v4_existence_probability": [0.75, 0.90, 0.49],
            "v4_geometry_probability": [0.75, 0.90, 0.90],
        }
    )

    assert three_state_decision(frame).tolist() == [
        "accept",
        "abstain",
        "reject",
    ]


def test_gate_rejects_non_finite_scores() -> None:
    frame = pd.DataFrame(
        {
            "v2_component_probability": [0.90],
            "v4_existence_probability": [np.nan],
            "v4_geometry_probability": [0.90],
        }
    )

    with pytest.raises(ValueError, match="finite"):
        three_state_decision(frame)


def test_final_candidate_adds_et_but_preserves_rc_priority() -> None:
    base = np.asarray([[[4, 3, 2, 1, 0]]], dtype=np.uint8)

    result = add_accepted_et(
        base,
        [(0, 0, 0), (0, 0, 2), (0, 0, 4)],
    )

    assert result.tolist() == [[[4, 3, 3, 1, 3]]]
    np.testing.assert_array_equal(base, np.asarray([[[4, 3, 2, 1, 0]]]))

