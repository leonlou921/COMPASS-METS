"""Exact frozen XF12 -> N03 inference composition.

The numerical primitives are imported from the audited challenge source tree.
This module only narrows the portfolio implementation to the single approved
candidate and makes the two learned inference bundles explicit.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from features import extract_case_features
from portfolio_common import (
    generate_union_proposals,
    masks_to_brats_segmentation,
    validate_probability_alignment,
)
from portfolio_variants import (
    CHANNELS,
    VARIANTS,
    _FIXED_LCV2_CUTOFFS,
    apply_nonrouter_variant,
    reconstruct_anchor,
)
from train_models_v2 import predict_test_component_probability


CANDIDATE_ID = "N03_XF12_LCv3_ET_parent_supported"
PROPOSAL_THRESHOLD = 0.25
LCV2_CUTOFFS = dict(_FIXED_LCV2_CUTOFFS)
_REGIONS = ("ET", "RC", "TC", "WT")
_SPEC = next(spec for spec in VARIANTS if spec.variant_id == CANDIDATE_ID)


def proposal_maps(
    arrays: Mapping[str, np.ndarray],
    spacing_zyx: tuple[float, float, float],
) -> dict[str, list[Any]]:
    """Recreate the persisted LCv2 proposal universe exactly."""
    validate_probability_alignment(arrays)
    return {
        region.upper(): generate_union_proposals(
            arrays,
            channel=CHANNELS.index(region),
            threshold=PROPOSAL_THRESHOLD,
            spacing_zyx=spacing_zyx,
        )
        for region in CHANNELS
    }


def score_maps(
    score_rows: pd.DataFrame,
    proposals: Mapping[str, list[Any]],
    cutoffs: Mapping[str, float],
) -> tuple[dict[str, dict[int, float]], dict[str, set[int]]]:
    """Validate score identity and apply the frozen region-specific cutoffs."""
    scores: dict[str, dict[int, float]] = {}
    accepted: dict[str, set[int]] = {}
    for region in _REGIONS:
        selected = score_rows[
            score_rows["region"].astype(str).str.upper() == region
        ]
        if selected.duplicated("proposal_id").any():
            raise RuntimeError(f"{region}: duplicate LCv2 proposal scores")
        region_scores = {
            int(row.proposal_id): float(row.score)
            for row in selected.itertuples()
        }
        expected_ids = {
            int(proposal.proposal_id) for proposal in proposals.get(region, ())
        }
        if set(region_scores) != expected_ids:
            raise RuntimeError(
                f"{region}: LCv2 score/proposal mismatch "
                f"missing={sorted(expected_ids - set(region_scores))[:5]} "
                f"unexpected={sorted(set(region_scores) - expected_ids)[:5]}"
            )
        if not np.isfinite(list(region_scores.values())).all():
            raise ValueError(f"{region}: LCv2 scores contain non-finite values")
        scores[region] = region_scores
        accepted[region] = {
            proposal_id
            for proposal_id, score in region_scores.items()
            if score >= float(cutoffs[region])
        }
    return scores, accepted


def learned_component_scores(
    case_id: str,
    arrays: Mapping[str, np.ndarray],
    *,
    lcv1_case_bundle: Mapping[str, Any],
    lcv2_component_bundle: Mapping[str, Any],
) -> pd.DataFrame:
    """Extract test features and score them with both required final bundles."""
    validate_probability_alignment(arrays)
    empty_gt = np.zeros(np.asarray(arrays["XL"]).shape[1:], dtype=np.uint8)
    case_rows, component_rows = extract_case_features(
        str(case_id),
        -1,
        np.asarray(arrays["XL"]),
        np.asarray(arrays["M"]),
        np.asarray(arrays["FT"]),
        empty_gt,
    )
    case_frame = pd.DataFrame(case_rows)
    component_frame = pd.DataFrame(component_rows)
    probabilities, _ = predict_test_component_probability(
        dict(lcv2_component_bundle),
        dict(lcv1_case_bundle),
        case_frame,
        component_frame,
    )
    if len(probabilities) != len(component_frame):
        raise RuntimeError(f"{case_id}: LCv2 test score length mismatch")
    identity = (
        "component_id" if "component_id" in component_frame else "proposal_id"
    )
    return pd.DataFrame(
        {
            "region": component_frame["region"].astype(str),
            "proposal_id": component_frame[identity].astype(int),
            "score": np.asarray(probabilities, dtype=float),
        }
    )


def build_n03_from_scores(
    arrays: Mapping[str, np.ndarray],
    spacing_zyx: tuple[float, float, float],
    score_rows: pd.DataFrame,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Build the exact N03 label map from already-computed LCv2 scores."""
    proposals = proposal_maps(arrays, spacing_zyx)
    proposal_scores, accepted = score_maps(
        score_rows,
        proposals,
        LCV2_CUTOFFS,
    )
    anchor = reconstruct_anchor("XF12", arrays, spacing_zyx)
    masks, audit = apply_nonrouter_variant(
        _SPEC,
        anchor,
        proposals,
        arrays,
        accepted,
        proposal_scores=proposal_scores,
    )
    return masks_to_brats_segmentation(masks), audit


def build_n03(
    case_id: str,
    arrays: Mapping[str, np.ndarray],
    spacing_zyx: tuple[float, float, float],
    *,
    lcv1_case_bundle: Mapping[str, Any],
    lcv2_component_bundle: Mapping[str, Any],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Run the complete learned XF12 -> N03 postprocess for one case."""
    scores = learned_component_scores(
        case_id,
        arrays,
        lcv1_case_bundle=lcv1_case_bundle,
        lcv2_component_bundle=lcv2_component_bundle,
    )
    return build_n03_from_scores(arrays, spacing_zyx, scores)
