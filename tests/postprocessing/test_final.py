from __future__ import annotations

import numpy as np
import pandas as pd

from compass_mets.postprocessing.final import (
    BASELINE_CANDIDATE_ID,
    CANDIDATE_ID,
    LCV2_CUTOFFS,
    build_n03_from_scores,
    proposal_maps,
    score_maps,
)
from compass_mets.postprocessing.portfolio.portfolio_common import (
    masks_to_brats_segmentation,
)
from compass_mets.postprocessing.portfolio.portfolio_variants import (
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
    for model in ("XL", "M"):
        arrays[model][CHANNELS.index("et"), 1:3, 1:3, 1:3] = 0.9
        arrays[model][CHANNELS.index("tc"), 1:3, 1:3, 1:3] = 0.9
        arrays[model][CHANNELS.index("wt"), 1:3, 1:3, 1:3] = 0.9
    return arrays


def test_frozen_candidate_and_cutoffs_match_n03_parent_supported() -> None:
    assert CANDIDATE_ID == "N03_FINAL_UTILITY_V4"
    assert LCV2_CUTOFFS == _FIXED_LCV2_CUTOFFS
    spec = next(spec for spec in VARIANTS if spec.variant_id == BASELINE_CANDIDATE_ID)
    assert spec.anchor_id == "XF12"
    assert spec.allowed_add_regions == ("ET",)
    assert spec.allowed_delete_regions == ()
    assert spec.mechanism == "et_parent_supported"


def test_n03_wrapper_is_voxel_identical_to_frozen_primitives() -> None:
    arrays = _probabilities()
    proposals = proposal_maps(arrays, (1.0, 1.0, 1.0))
    rows = pd.DataFrame(
        [
            {
                "region": region,
                "proposal_id": proposal.proposal_id,
                "score": 1.0,
            }
            for region, region_proposals in proposals.items()
            for proposal in region_proposals
        ]
    )
    actual, actual_audit = build_n03_from_scores(
        arrays, (1.0, 1.0, 1.0), rows
    )
    scores, accepted = score_maps(rows, proposals, LCV2_CUTOFFS)
    anchor = reconstruct_anchor("XF12", arrays, (1.0, 1.0, 1.0))
    spec = next(spec for spec in VARIANTS if spec.variant_id == BASELINE_CANDIDATE_ID)
    expected_masks, expected_audit = apply_nonrouter_variant(
        spec,
        anchor,
        proposals,
        arrays,
        accepted,
        proposal_scores=scores,
    )
    np.testing.assert_array_equal(actual, masks_to_brats_segmentation(expected_masks))
    assert actual_audit == expected_audit
