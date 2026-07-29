from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
REMOTE_SCRIPTS = ROOT.parents[1] / "remote_scripts"
LCV1_SOURCE = REMOTE_SCRIPTS / "learned_component_gate_v1_20260723"
LCV3_SOURCE = REMOTE_SCRIPTS / "learned_component_gate_v3_20260726"
for path in (SRC, REMOTE_SCRIPTS, LCV1_SOURCE, LCV3_SOURCE):
    sys.path.insert(0, str(path))

from n03_docker.postprocess import (  # noqa: E402
    BASELINE_CANDIDATE_ID,
    CANDIDATE_ID,
    LCV2_CUTOFFS,
    build_n03_from_scores,
    score_maps,
    utility_candidate_records,
)
from portfolio_common import generate_union_proposals, masks_to_brats_segmentation  # noqa: E402
from portfolio_variants import (  # noqa: E402
    CHANNELS,
    VARIANTS,
    _FIXED_LCV2_CUTOFFS,
    apply_nonrouter_variant,
    reconstruct_anchor,
)


def _probabilities() -> dict[str, np.ndarray]:
    arrays = {
        model: np.full((4, 7, 7, 7), 0.01, dtype=np.float32)
        for model in ("XL", "M", "FT")
    }
    # A parent-supported ET proposal outside the conservative XF12 anchor.
    for model in ("XL", "M"):
        arrays[model][CHANNELS.index("et"), 1:3, 1:3, 1:3] = 0.9
        arrays[model][CHANNELS.index("tc"), 1:3, 1:3, 1:3] = 0.9
        arrays[model][CHANNELS.index("wt"), 1:3, 1:3, 1:3] = 0.9
    return arrays


def _proposals(arrays: dict[str, np.ndarray]):
    return {
        region.upper(): generate_union_proposals(
            arrays,
            channel=CHANNELS.index(region),
            threshold=0.25,
            spacing_zyx=(1.0, 1.0, 1.0),
        )
        for region in CHANNELS
    }


def test_frozen_candidate_and_cutoffs_match_original_portfolio() -> None:
    assert CANDIDATE_ID == "N03_FINAL_UTILITY_V4"
    assert LCV2_CUTOFFS == _FIXED_LCV2_CUTOFFS
    spec = next(
        spec
        for spec in VARIANTS
        if spec.variant_id == "N03_XF12_LCv3_ET_parent_supported"
    )
    assert spec.anchor_id == "XF12"
    assert spec.allowed_add_regions == ("ET",)
    assert spec.mechanism == "et_parent_supported"


def test_score_maps_requires_exact_proposal_identity() -> None:
    arrays = _probabilities()
    proposals = _proposals(arrays)
    rows = []
    for region, region_proposals in proposals.items():
        for proposal in region_proposals:
            rows.append(
                {
                    "region": region,
                    "proposal_id": proposal.proposal_id,
                    "score": 1.0,
                }
            )
    scores, accepted = score_maps(pd.DataFrame(rows), proposals, LCV2_CUTOFFS)
    assert set(scores) == {"ET", "RC", "TC", "WT"}
    assert accepted["ET"] == {1}

    with np.testing.assert_raises_regex(RuntimeError, "score/proposal mismatch"):
        score_maps(pd.DataFrame(rows[:-1]), proposals, LCV2_CUTOFFS)


def test_wrapper_is_voxel_identical_to_frozen_n03_primitives() -> None:
    arrays = _probabilities()
    proposals = _proposals(arrays)
    rows = [
        {
            "region": region,
            "proposal_id": proposal.proposal_id,
            "score": 1.0,
        }
        for region, region_proposals in proposals.items()
        for proposal in region_proposals
    ]
    actual, actual_audit = build_n03_from_scores(
        arrays,
        (1.0, 1.0, 1.0),
        pd.DataFrame(rows),
    )

    scores, accepted = score_maps(pd.DataFrame(rows), proposals, LCV2_CUTOFFS)
    anchor = reconstruct_anchor("XF12", arrays, (1.0, 1.0, 1.0))
    spec = next(
        spec
        for spec in VARIANTS
        if spec.variant_id == BASELINE_CANDIDATE_ID
    )
    expected_masks, expected_audit = apply_nonrouter_variant(
        spec,
        anchor,
        proposals,
        arrays,
        accepted,
        proposal_scores=scores,
    )
    expected = masks_to_brats_segmentation(expected_masks)

    np.testing.assert_array_equal(actual, expected)
    assert actual_audit == expected_audit


def test_utility_pool_selects_only_disconnected_high_score_et() -> None:
    arrays = {
        model: np.full((4, 5, 5, 5), 0.01, dtype=np.float32)
        for model in ("XL", "M", "FT")
    }
    for model in ("XL", "FT"):
        arrays[model][2, 1:3, 1:3, 1:3] = 0.9
    component_frame = pd.DataFrame(
        [
            {
                "region": "ET",
                "component_id": 1,
                "bbox_z0": 1,
                "bbox_z1": 3,
                "bbox_y0": 1,
                "bbox_y1": 3,
                "bbox_x0": 1,
                "bbox_x1": 3,
                "v2_component_probability": 0.88,
                "v3_component_probability": 0.86,
                "v4_existence_probability": 0.85,
                "v4_geometry_probability": 0.80,
                "gate_decision": "accept",
            }
        ]
    )

    records = utility_candidate_records(
        arrays,
        np.zeros((5, 5, 5), dtype=np.uint8),
        component_frame,
    )

    assert len(records) == 1
    assert records[0]["voxels"] == 8
    assert records[0]["gate_decision"] == "accept"
    assert records[0]["coordinates_zyx"][0] == (1, 1, 1)

    overlapping = np.zeros((5, 5, 5), dtype=np.uint8)
    overlapping[1, 1, 1] = 3
    assert utility_candidate_records(arrays, overlapping, component_frame) == []
