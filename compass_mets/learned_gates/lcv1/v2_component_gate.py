"""Fixed probability bases and final-component reconstruction for gate v2."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy import ndimage as ndi
from scipy import special

from compass_mets.learned_gates.lcv1.component_gate import (
    REGION_NAMES,
    _validate_model_probabilities,
)
from compass_mets.learned_gates.lcv1.reconstruct_and_evaluate import (
    _filter_components,
    _mask_bbox,
)


LC_V2_STRUCTURED_FILTER = "LCv2_structured_filter_V2"
LC_V2_STRUCTURED_PROTECTED_RESCUE = "LCv2_structured_protected_rescue_V2"
LC_V2_XLFT_FILTER = "LCv2_XLFT_filter_V2"
CANDIDATES = (
    LC_V2_STRUCTURED_FILTER,
    LC_V2_STRUCTURED_PROTECTED_RESCUE,
    LC_V2_XLFT_FILTER,
)
FT_WEIGHTS = {"WT": 0.15, "TC": 0.20, "ET": 0.15, "RC": 0.30}
EPS = 1e-5


def _logit_fuse(
    arrays: tuple[np.ndarray, ...],
    weights_by_region: Mapping[str, tuple[float, ...]],
) -> np.ndarray:
    shape = arrays[0].shape
    fused = np.empty(shape, dtype=np.float32)
    temporary = np.empty(shape[1:], dtype=np.float32)
    for channel, region in enumerate(REGION_NAMES):
        weights = weights_by_region[region]
        if len(weights) != len(arrays) or not np.isclose(sum(weights), 1.0):
            raise ValueError(f"invalid {region} fusion weights: {weights}")
        fused[channel].fill(0.0)
        for array, weight in zip(arrays, weights):
            np.clip(array[channel], EPS, 1.0 - EPS, out=temporary)
            special.logit(temporary, out=temporary)
            fused[channel] += temporary * float(weight)
        special.expit(fused[channel], out=fused[channel])
    return fused


def _logit_fuse_channel(
    arrays: tuple[np.ndarray, ...],
    channel: int,
    weights: tuple[float, ...],
) -> np.ndarray:
    """Fuse one channel without materializing an unused four-channel volume."""
    if len(weights) != len(arrays) or not np.isclose(sum(weights), 1.0):
        raise ValueError(f"invalid channel fusion weights: {weights}")
    fused = np.zeros(arrays[0].shape[1:], dtype=np.float32)
    temporary = np.empty_like(fused)
    for array, weight in zip(arrays, weights):
        np.clip(array[channel], EPS, 1.0 - EPS, out=temporary)
        special.logit(temporary, out=temporary)
        fused += temporary * float(weight)
    special.expit(fused, out=fused)
    return fused


def candidate_score_maps(
    probabilities_by_model: Mapping[str, np.ndarray],
    candidate: str,
) -> dict[str, np.ndarray]:
    """Return the fixed output basis and candidate-specific V2 score maps."""
    missing = {"XL", "M", "FT"}.difference(probabilities_by_model)
    if missing:
        raise KeyError(f"missing model probabilities: {sorted(missing)}")
    _validate_model_probabilities(probabilities_by_model)
    xl = np.asarray(probabilities_by_model["XL"], dtype=np.float32)
    m = np.asarray(probabilities_by_model["M"], dtype=np.float32)
    ft = np.asarray(probabilities_by_model["FT"], dtype=np.float32)
    if candidate in {
        LC_V2_STRUCTURED_FILTER,
        LC_V2_STRUCTURED_PROTECTED_RESCUE,
    }:
        calibrated = _logit_fuse(
            (xl, ft),
            {
                region: (1.0 - FT_WEIGHTS[region], FT_WEIGHTS[region])
                for region in REGION_NAMES
            },
        )
        tc_v2 = np.maximum(calibrated[1], calibrated[2])
        basis = calibrated
        basis[0] = ft[0]
        basis[1] = ft[1]
        if candidate == LC_V2_STRUCTURED_PROTECTED_RESCUE:
            rc_v2 = basis[3].copy()
            mft_rc = _logit_fuse_channel((m, ft), 3, (0.5, 0.5))
            np.maximum(rc_v2, mft_rc, out=rc_v2)
        else:
            rc_v2 = basis[3]
    elif candidate == LC_V2_XLFT_FILTER:
        basis = _logit_fuse(
            (xl, ft), {region: (0.5, 0.5) for region in REGION_NAMES}
        )
        tc_v2 = np.maximum(basis[1], basis[2])
        rc_v2 = basis[3]
    else:
        raise ValueError(f"unknown v2 candidate: {candidate}")
    return {
        "basis": np.asarray(basis, dtype=np.float32),
        "tc_v2": np.asarray(tc_v2, dtype=np.float32),
        "rc_v2": np.asarray(rc_v2, dtype=np.float32),
    }


def _protected_local_mask(
    probabilities_by_model: Mapping[str, np.ndarray],
    channel: int,
    component: Mapping,
) -> np.ndarray | None:
    bbox = component["bbox"]
    local_mask = np.asarray(component["local_mask"], dtype=bool)
    if not local_mask.any():
        return None
    m = np.asarray(probabilities_by_model["M"][channel][bbox])
    ft = np.asarray(probabilities_by_model["FT"][channel][bbox])
    m_support = np.logical_and(m >= 0.25, local_mask)
    ft_support = np.logical_and(ft >= 0.25, local_mask)
    if (
        float(m_support.sum()) / float(local_mask.sum()) < 0.70
        or float(ft_support.sum()) / float(local_mask.sum()) < 0.70
        or not np.any(np.logical_and(m >= 0.50, local_mask))
        or not np.any(np.logical_and(ft >= 0.50, local_mask))
    ):
        return None
    support = np.logical_and(m_support, ft_support)
    seed = np.logical_and(np.logical_or(m >= 0.50, ft >= 0.50), support)
    labels, count = ndi.label(support, structure=np.ones((3, 3, 3), dtype=np.uint8))
    if not count or not seed.any():
        return None
    accepted_labels = np.unique(labels[seed])
    accepted_labels = accepted_labels[accepted_labels > 0]
    protected = np.logical_and(support, np.isin(labels, accepted_labels))
    return protected if protected.any() else None


def _enforce_scored_hierarchy(
    regions: dict[str, np.ndarray],
    scores: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    output = {region: np.asarray(regions[region], dtype=bool).copy() for region in REGION_NAMES}
    output_scores = {
        region: np.asarray(scores[region], dtype=np.float32).copy()
        for region in REGION_NAMES
    }
    output["TC"] |= output["ET"]
    np.maximum(output_scores["TC"], output_scores["ET"], out=output_scores["TC"])
    output["WT"] |= output["TC"]
    np.maximum(output_scores["WT"], output_scores["TC"], out=output_scores["WT"])
    return output, output_scores


def reconstruct_scored_candidate(
    probabilities_by_model: Mapping[str, np.ndarray],
    proposals: Mapping[str, list[Mapping]],
    proposal_scores: Mapping[tuple[str, int], float],
    candidate: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], int]:
    """Reconstruct the fixed candidate basis and retain a learned score volume."""
    full_shape, _ = _validate_model_probabilities(probabilities_by_model)
    maps = candidate_score_maps(probabilities_by_model, candidate)
    regions = {region: np.zeros(full_shape, dtype=bool) for region in REGION_NAMES}
    scores = {
        region: np.zeros(full_shape, dtype=np.float32) for region in REGION_NAMES
    }
    protected_count = 0
    for channel, region in enumerate(REGION_NAMES):
        for component in proposals.get(region, []):
            component_id = int(component["component_id"])
            bbox = component["bbox"]
            local_mask = np.asarray(component["local_mask"], dtype=bool)
            accepted = np.logical_and(maps["basis"][channel][bbox] >= 0.50, local_mask)
            protected = None
            if candidate == LC_V2_STRUCTURED_PROTECTED_RESCUE:
                protected = _protected_local_mask(
                    probabilities_by_model, channel, component
                )
                if protected is not None:
                    accepted = np.logical_or(accepted, protected)
                    protected_count += 1
            if not accepted.any():
                continue
            local_region = regions[region][bbox]
            np.logical_or(local_region, accepted, out=local_region)
            component_score = float(
                proposal_scores.get((region, component_id), 0.0)
            )
            local_scores = scores[region][bbox]
            local_scores[accepted] = np.maximum(
                local_scores[accepted], component_score
            )
    regions["TC"] |= regions["ET"]
    np.maximum(scores["TC"], scores["ET"], out=scores["TC"])
    regions["WT"] |= regions["TC"]
    np.maximum(scores["WT"], scores["TC"], out=scores["WT"])
    return regions, scores, protected_count


def filter_scored_components(
    regions: Mapping[str, np.ndarray],
    score_volumes: Mapping[str, np.ndarray],
    cutoffs: Mapping[str, float],
) -> dict[str, np.ndarray]:
    """Filter 26-connected final components by their maximum learned score."""
    output: dict[str, np.ndarray] = {}
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    for region in REGION_NAMES:
        mask = np.asarray(regions[region], dtype=bool)
        score = np.asarray(score_volumes[region], dtype=np.float32)
        if mask.shape != score.shape:
            raise ValueError(f"{region}: mask and score shapes differ")
        labels, count = ndi.label(mask, structure=structure)
        if count == 0:
            output[region] = np.zeros_like(mask)
            continue
        peaks = np.zeros(count + 1, dtype=np.float32)
        np.maximum.at(peaks, labels.ravel(), score.ravel())
        keep = peaks >= float(cutoffs[region])
        keep[0] = False
        output[region] = keep[labels]
    output["TC"] |= output["ET"]
    output["WT"] |= output["TC"]
    return output


def scored_final_component_rows(
    regions: Mapping[str, np.ndarray],
    score_volumes: Mapping[str, np.ndarray],
    gt_regions: Mapping[str, np.ndarray],
    case_id: str,
    fold: int,
    candidate: str,
) -> list[dict]:
    """Describe reconstructed 26-connected components at the calibration grain."""
    rows: list[dict] = []
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    for region in REGION_NAMES:
        mask = np.asarray(regions[region], dtype=bool)
        score = np.asarray(score_volumes[region], dtype=np.float32)
        target = np.asarray(gt_regions[region], dtype=bool)
        labels, count = ndi.label(mask, structure=structure)
        objects = ndi.find_objects(labels, max_label=count)
        for component_id, bbox in enumerate(objects, start=1):
            if bbox is None:
                continue
            local_component = labels[bbox] == component_id
            local_score = score[bbox][local_component]
            overlap = int(np.logical_and(local_component, target[bbox]).sum())
            rows.append(
                {
                    "case_id": str(case_id),
                    "fold": int(fold),
                    "candidate": str(candidate),
                    "region": region,
                    "component_id": int(component_id),
                    "score": float(local_score.max()) if local_score.size else 0.0,
                    "target": int(overlap > 0),
                    "overlap_voxels": overlap,
                    "volume_voxels": int(local_component.sum()),
                }
            )
    return rows


def apply_gate_preserving_v2(
    regions: Mapping[str, np.ndarray],
    score_maps: Mapping[str, np.ndarray],
    spacing_zyx: tuple[float, float, float],
    settings: Mapping[str, float | int],
) -> dict[str, np.ndarray]:
    """Apply TC growth and strict RC only within the learned RC candidate mask."""
    output = {
        region: np.asarray(regions[region], dtype=bool).copy()
        for region in REGION_NAMES
    }
    output["TC"] |= output["ET"]
    output["WT"] |= output["TC"]
    tc_score = np.asarray(score_maps["tc_v2"], dtype=np.float32)
    structure = ndi.generate_binary_structure(3, 1)
    budget = int(settings["tc_boundary_budget"])
    added = 0
    wt_bbox = _mask_bbox(output["WT"], pad=1)
    if wt_bbox is not None:
        local_tc = output["TC"][wt_bbox].copy()
        local_wt = output["WT"][wt_bbox]
        local_score = tc_score[wt_bbox]
        while added < budget and local_tc.any():
            frontier = (
                ndi.binary_dilation(local_tc, structure=structure)
                & local_wt
                & ~local_tc
                & (local_score >= float(settings["tc_boundary_threshold"]))
            )
            if not frontier.any():
                break
            coordinates = np.argwhere(frontier)
            coordinate = tuple(coordinates[np.argmax(local_score[frontier])])
            local_tc[coordinate] = True
            added += 1
        output["TC"][wt_bbox] = local_tc

    rc_score = np.asarray(score_maps["rc_v2"], dtype=np.float32)
    learned_rc = np.logical_and(
        output["RC"], rc_score >= float(settings["rc_threshold"])
    )
    output["RC"] = _filter_components(
        learned_rc,
        rc_score,
        spacing_zyx,
        float(settings["rc_minimum_volume_mm3"]),
        float(settings["rc_minimum_mean"]),
        float(settings["rc_minimum_peak"]),
    )
    output["TC"] |= output["ET"]
    output["WT"] |= output["TC"]
    return output
