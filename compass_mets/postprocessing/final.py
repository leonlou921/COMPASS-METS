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
from scipy import ndimage as ndi

from compass_mets.learned_gates.lcv1.component_gate import (
    label_to_regions,
    propose_components,
)
from compass_mets.learned_gates.lcv1.features import extract_case_features
from compass_mets.postprocessing.portfolio.portfolio_common import (
    generate_union_proposals,
    masks_to_brats_segmentation,
    validate_probability_alignment,
)
from compass_mets.postprocessing.portfolio.portfolio_variants import (
    CHANNELS,
    VARIANTS,
    _FIXED_LCV2_CUTOFFS,
    apply_nonrouter_variant,
    reconstruct_anchor,
)
from compass_mets.learned_gates.lcv1.train_models_v2 import (
    _case_probability_map,
    predict_test_component_probability,
)
from compass_mets.learned_gates.lcv1.v2_component_gate import (
    LC_V2_STRUCTURED_FILTER,
    reconstruct_scored_candidate,
)
from compass_mets.learned_gates.lcv1.v2_pipeline import (
    proposal_scores_from_predictions,
)

from .utility_v4 import (
    RGV3_ET_CUTOFF,
    add_accepted_et,
    score_utility_features,
)


CANDIDATE_ID = "N03_FINAL_UTILITY_V4"
BASELINE_CANDIDATE_ID = "N03_XF12_LCv3_ET_parent_supported"
PROPOSAL_THRESHOLD = 0.25
LCV2_CUTOFFS = dict(_FIXED_LCV2_CUTOFFS)
_REGIONS = ("ET", "RC", "TC", "WT")
_SPEC = next(
    spec for spec in VARIANTS if spec.variant_id == BASELINE_CANDIDATE_ID
)


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
    component_frame = learned_component_feature_frame(
        case_id,
        arrays,
        lcv1_case_bundle=lcv1_case_bundle,
        lcv2_component_bundle=lcv2_component_bundle,
    )
    return pd.DataFrame(
        {
            "region": component_frame["region"].astype(str),
            "proposal_id": component_frame["component_id"].astype(int),
            "score": component_frame["v2_component_probability"].astype(float),
        }
    )


def learned_component_feature_frame(
    case_id: str,
    arrays: Mapping[str, np.ndarray],
    *,
    lcv1_case_bundle: Mapping[str, Any],
    lcv2_component_bundle: Mapping[str, Any],
) -> pd.DataFrame:
    """Return the exact test component features enriched with LCv1/LCv2."""
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
    case_probability = np.asarray(
        lcv1_case_bundle["case_model"]
        .predict_proba(case_frame.loc[:, lcv1_case_bundle["case_features"]])[:, 1],
        dtype=np.float64,
    )
    component_frame["case_probability_feature"] = _case_probability_map(
        component_frame,
        case_frame,
        case_probability,
    )
    component_frame["v2_component_probability"] = np.asarray(
        probabilities,
        dtype=np.float64,
    )
    return component_frame


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


def _bbox_contains_coordinates(
    row: pd.Series,
    coordinates: list[tuple[int, int, int]],
) -> bool:
    return all(
        int(row["bbox_z0"]) <= z < int(row["bbox_z1"])
        and int(row["bbox_y0"]) <= y < int(row["bbox_y1"])
        and int(row["bbox_x0"]) <= x < int(row["bbox_x1"])
        for z, y, x in coordinates
    )


def utility_candidate_records(
    arrays: Mapping[str, np.ndarray],
    base_label_zyx: np.ndarray,
    component_frame: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Describe disconnected structured-union ET components eligible for v4."""
    proposals = propose_components(arrays, threshold=PROPOSAL_THRESHOLD)
    proposal_scores = proposal_scores_from_predictions(
        component_frame,
        component_frame["v2_component_probability"].to_numpy(dtype=np.float64),
    )
    candidate_regions, candidate_scores, _ = reconstruct_scored_candidate(
        arrays,
        proposals,
        proposal_scores,
        LC_V2_STRUCTURED_FILTER,
    )
    base_et = label_to_regions(base_label_zyx)["ET"]
    labels, count = ndi.label(
        candidate_regions["ET"],
        structure=np.ones((3, 3, 3), dtype=np.uint8),
    )
    et_features = component_frame.loc[
        component_frame["region"].astype(str).str.upper().eq("ET")
    ]
    records: list[dict[str, Any]] = []
    for component_id in range(1, count + 1):
        component = labels == component_id
        if np.logical_and(component, base_et).any():
            continue
        values = np.asarray(candidate_scores["ET"][component], dtype=np.float64)
        if not len(values):
            continue
        mean_score = float(values.mean())
        peak_score = float(values.max())
        if peak_score < RGV3_ET_CUTOFF:
            continue
        coordinates = [
            tuple(int(value) for value in coordinate)
            for coordinate in np.argwhere(component)
        ]
        matches = et_features.loc[
            np.isclose(
                et_features["v2_component_probability"].to_numpy(dtype=np.float64),
                mean_score,
                rtol=0.0,
                atol=1e-7,
            )
        ]
        matches = matches.loc[
            [
                _bbox_contains_coordinates(row, coordinates)
                for _, row in matches.iterrows()
            ]
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"utility ET component {component_id}: "
                f"expected one proposal match, found {len(matches)}"
            )
        row = matches.iloc[0]
        records.append(
            {
                "component_id": component_id,
                "proposal_component_id": int(row["component_id"]),
                "coordinates_zyx": coordinates,
                "voxels": len(coordinates),
                "mean_score": mean_score,
                "peak_score": peak_score,
                "v2_component_probability": float(
                    row["v2_component_probability"]
                ),
                "v3_component_probability": float(
                    row["v3_component_probability"]
                ),
                "v4_existence_probability": float(
                    row["v4_existence_probability"]
                ),
                "v4_geometry_probability": float(
                    row["v4_geometry_probability"]
                ),
                "gate_decision": str(row["gate_decision"]),
            }
        )
    return records


def build_n03_final(
    case_id: str,
    arrays: Mapping[str, np.ndarray],
    spacing_zyx: tuple[float, float, float],
    *,
    lcv1_case_bundle: Mapping[str, Any],
    lcv2_component_bundle: Mapping[str, Any],
    rgv3_et_bundle: Mapping[str, Any],
    utility_v4_existence_model: Any,
    utility_v4_geometry_model: Any,
    utility_v4_feature_names: list[str],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Build the frozen baseline and add only utility-v4 accepted ET."""
    component_frame = learned_component_feature_frame(
        case_id,
        arrays,
        lcv1_case_bundle=lcv1_case_bundle,
        lcv2_component_bundle=lcv2_component_bundle,
    )
    score_rows = pd.DataFrame(
        {
            "region": component_frame["region"].astype(str),
            "proposal_id": component_frame["component_id"].astype(int),
            "score": component_frame["v2_component_probability"].astype(float),
        }
    )
    baseline, baseline_audit = build_n03_from_scores(
        arrays,
        spacing_zyx,
        score_rows,
    )
    utility_features = score_utility_features(
        component_frame,
        rgv3_et_bundle=rgv3_et_bundle,
        existence_model=utility_v4_existence_model,
        geometry_model=utility_v4_geometry_model,
        utility_feature_names=utility_v4_feature_names,
    )
    utility_records = utility_candidate_records(
        arrays,
        baseline,
        utility_features,
    )
    accepted_coordinates = [
        coordinate
        for record in utility_records
        if record["gate_decision"] == "accept"
        for coordinate in record["coordinates_zyx"]
    ]
    result = add_accepted_et(baseline, accepted_coordinates)
    return result, [
        *baseline_audit,
        *[
            {
                "stage": "utility_v4",
                **record,
                "decision": (
                    "add" if record["gate_decision"] == "accept" else "skip"
                ),
            }
            for record in utility_records
        ],
    ]
