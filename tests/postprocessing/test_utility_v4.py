from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from compass_mets.postprocessing.utility_v4 import (
    add_accepted_et,
    three_state_decision,
)


def test_gate_uses_frozen_three_state_thresholds() -> None:
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


def test_gate_rejects_nonfinite_scores() -> None:
    frame = pd.DataFrame(
        {
            "v2_component_probability": [0.90],
            "v4_existence_probability": [np.nan],
            "v4_geometry_probability": [0.90],
        }
    )
    with pytest.raises(ValueError, match="finite"):
        three_state_decision(frame)


def test_utility_adds_et_without_overwriting_rc() -> None:
    base = np.asarray([[[4, 3, 2, 1, 0]]], dtype=np.uint8)
    result = add_accepted_et(base, [(0, 0, 0), (0, 0, 2), (0, 0, 4)])
    assert result.tolist() == [[[4, 3, 3, 1, 3]]]
    np.testing.assert_array_equal(base, np.asarray([[[4, 3, 2, 1, 0]]]))
