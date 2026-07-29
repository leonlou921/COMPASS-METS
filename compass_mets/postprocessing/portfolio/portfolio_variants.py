"""Frozen anchor reconstruction and ten deterministic non-router portfolio variants."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import ndimage

from compass_mets.postprocessing.portfolio.portfolio_common import (
    LocalProposal,
    assert_anchor_protected,
    enforce_hierarchy,
    has_two_of_three_support,
    reconstruct_xl_core_shape,
    reconstruct_xlft_consensus_shape,
    validate_probability_alignment,
)


from compass_mets.fusion.build_xl_fixed_postprocess import (
    RC_STRICT,
    TC_BOUNDARY,
    apply_rc_gate,
    segmentation_to_masks,
)
from compass_mets.fusion.mft_regionwise_pipeline import (
    CHANNELS,
    DEFAULT_GATES,
    apply_tc_boundary_completion,
    component_conf_masks,
    masks_to_segmentation,
)
from compass_mets.fusion.xlm_fixed_fusions import (
    build_candidate_segmentations,
    fuse_logits_many,
)


_MODELS = ("XL", "M", "FT")
_REGIONS = ("ET", "RC", "TC", "WT")
_HIERARCHY_REGIONS = frozenset(("ET", "TC", "WT"))
_XF12_ID = "XF12_XLM_structured_probability_V2_strict"
_FIXED_LCV2_CUTOFFS = {
    "ET": 0.5497123599,
    "RC": 0.0877561048,
    "TC": 0.3819091916,
    "WT": 0.3819091916,
}


@dataclass(frozen=True)
class VariantSpec:
    variant_id: str
    priority: str
    anchor_id: str
    mechanism: str
    allowed_add_regions: tuple[str, ...]
    allowed_delete_regions: tuple[str, ...]
    use_lcv2: bool
    require_consensus: bool
    shape_mode: str


VARIANTS: tuple[VariantSpec, ...] = (
    VariantSpec(
        "N05_XF12_LCv3_exist_XLcore_shape",
        "P0",
        "XF12",
        "lcv2_add_only",
        ("ET",),
        (),
        True,
        False,
        "xl_core",
    ),
    VariantSpec(
        "N01_XF12_LCv3_addonly_consensus",
        "P0",
        "XF12",
        "lcv2_consensus_add_only",
        ("ET", "RC", "TC", "WT"),
        (),
        True,
        True,
        "proposal_local",
    ),
    VariantSpec(
        "N06_XF12_LCv3_exist_XLFTcons_shape",
        "P0",
        "XF12",
        "lcv2_add_only",
        ("ET",),
        (),
        True,
        False,
        "xlft_consensus",
    ),
    VariantSpec(
        "N03_XF12_LCv3_ET_parent_supported",
        "P0",
        "XF12",
        "et_parent_supported",
        ("ET",),
        (),
        True,
        True,
        "proposal_local",
    ),
    VariantSpec(
        "N08_XF12_LCv3_RC_veto_only",
        "P1",
        "XF12",
        "rc_veto_only",
        (),
        ("RC",),
        True,
        False,
        "none",
    ),
    VariantSpec(
        "N11_XF12_XLMFT_consensus_addonly",
        "P1",
        "XF12",
        "deterministic_consensus_add_only",
        ("ET",),
        (),
        False,
        True,
        "proposal_local",
    ),
    VariantSpec(
        "N02_XF16_LCv3_addonly_consensus",
        "P1",
        "XF16",
        "lcv2_consensus_add_only",
        ("ET", "RC", "TC", "WT"),
        (),
        True,
        True,
        "proposal_local",
    ),
    VariantSpec(
        "N04_XF16_LCv3_ET_parent_supported",
        "P2",
        "XF16",
        "et_parent_supported",
        ("ET",),
        (),
        True,
        True,
        "proposal_local",
    ),
    VariantSpec(
        "N07_XF16_LCv3_exist_XLFTcons_shape",
        "P2",
        "XF16",
        "lcv2_add_only",
        ("ET",),
        (),
        True,
        False,
        "xlft_consensus",
    ),
    VariantSpec(
        "N12_XF16_XLMFT_consensus_addonly",
        "P2",
        "XF16",
        "deterministic_consensus_add_only",
        ("ET",),
        (),
        False,
        True,
        "proposal_local",
    ),
)

_VARIANTS_BY_ID = {spec.variant_id: spec for spec in VARIANTS}


def _validate_model_probabilities(
    probabilities: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    missing = sorted(set(_MODELS).difference(probabilities))
    if missing:
        raise KeyError(f"missing model probabilities: {missing}")
    arrays = {model: np.asarray(probabilities[model]) for model in _MODELS}
    validate_probability_alignment(arrays)
    for model, array in arrays.items():
        if array.ndim != 4 or array.shape[0] != len(CHANNELS):
            raise ValueError(
                f"{model}: expected canonical {CHANNELS} probability channels, got {array.shape}"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"{model}: model probabilities contain non-finite values")
    return arrays


def _copy_region_masks(masks: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for region in _REGIONS:
        key = region if region in masks else region.lower()
        if key not in masks:
            raise KeyError(f"missing anchor region: {region}")
        output[region.lower()] = np.asarray(masks[key], dtype=bool).copy()
    shapes = {mask.shape for mask in output.values()}
    if len(shapes) != 1:
        raise ValueError(f"anchor region shapes differ: {sorted(shapes)}")
    return output


def reconstruct_anchor(
    anchor_id: str,
    probabilities: Mapping[str, np.ndarray],
    spacing_zyx: tuple[float, float, float],
) -> dict[str, np.ndarray]:
    """Rebuild XF12 or XF16 by delegating to the frozen fusion primitives."""
    if anchor_id not in {"XF12", "XF16"}:
        raise KeyError(f"unknown anchor: {anchor_id}")
    arrays = _validate_model_probabilities(probabilities)

    if anchor_id == "XF12":
        segmentation = build_candidate_segmentations(
            arrays,
            spacing_zyx,
            (_XF12_ID,),
        )[_XF12_ID]
        return _copy_region_masks(segmentation_to_masks(segmentation))

    fused = fuse_logits_many((arrays["XL"], arrays["FT"]), (0.5, 0.5))
    masks = component_conf_masks(fused, spacing_zyx, DEFAULT_GATES)
    tc_score = np.maximum(
        fused[CHANNELS.index("tc")],
        fused[CHANNELS.index("et")],
    )
    masks, _ = apply_tc_boundary_completion(masks, tc_score, TC_BOUNDARY)
    masks = apply_rc_gate(
        masks,
        fused[CHANNELS.index("rc")],
        spacing_zyx,
        **RC_STRICT,
    )
    # XF16's stored candidate is a single-label BraTS segmentation. Canonical
    # serialization writes RC last, so decode that legal label map rather than
    # returning the pre-serialization independent masks when RC overlaps WT.
    return _copy_region_masks(
        segmentation_to_masks(masks_to_segmentation(masks))
    )


def _is_model_bundle(value: Mapping[str, object]) -> bool:
    return "component_model" in value or "component_features" in value


def _region_bundle(
    models: Mapping[str, object],
    region: str,
) -> Mapping[str, object]:
    if _is_model_bundle(models):
        return models
    for key in (region, region.lower()):
        if key in models:
            bundle = models[key]
            if not isinstance(bundle, Mapping):
                raise TypeError(f"{region}: model bundle must be a mapping")
            return bundle
    raise KeyError(f"missing model bundle for region {region}")


def _validate_feature_schema(
    case_features: pd.DataFrame,
    feature_order: tuple[str, ...],
) -> None:
    if case_features.columns.duplicated().any():
        duplicates = case_features.columns[case_features.columns.duplicated()].tolist()
        raise ValueError(f"duplicate feature columns: {duplicates}")
    if len(feature_order) != len(set(feature_order)):
        raise ValueError("persisted model bundle has duplicate feature columns")
    metadata = {"case_id", "proposal_id", "component_id", "region"}
    available_features = set(case_features.columns).difference(metadata)
    expected_features = set(feature_order)
    missing = sorted(expected_features - available_features)
    extra = sorted(available_features - expected_features)
    if missing:
        raise KeyError(f"missing feature columns: {missing}")
    if extra:
        raise ValueError(f"extra feature columns: {extra}")
    values = case_features.loc[:, list(feature_order)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("LCv2 features must be finite")


def score_lcv2_proposals(
    case_features: pd.DataFrame,
    models: Mapping[str, object],
    cutoffs: Mapping[str, float],
) -> pd.DataFrame:
    """Score proposals with persisted feature order and frozen LCv2 cutoffs."""
    frame = case_features.copy()
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError(f"duplicate feature columns: {duplicates}")
    if "region" not in frame:
        raise KeyError("missing proposal identity column: region")
    id_column = "proposal_id" if "proposal_id" in frame else "component_id"
    if id_column not in frame:
        raise KeyError("missing proposal identity column: proposal_id")
    canonical_regions = frame["region"].astype(str).str.upper()
    unknown_regions = sorted(set(canonical_regions).difference(_REGIONS))
    if unknown_regions:
        raise ValueError(f"unknown proposal regions: {unknown_regions}")
    for region in sorted(set(canonical_regions)):
        if region not in cutoffs:
            raise KeyError(f"missing LCv2 cutoff for region {region}")
        supplied = float(cutoffs[region])
        fixed = _FIXED_LCV2_CUTOFFS[region]
        if not np.isclose(supplied, fixed, rtol=0.0, atol=1e-12):
            raise ValueError(
                f"{region}: cutoff {supplied} differs from frozen LCv2 cutoff {fixed}"
            )

    output = pd.DataFrame(
        {
            "proposal_id": frame[id_column].to_numpy(),
            "region": canonical_regions.to_numpy(),
            "score": np.full(len(frame), np.nan, dtype=np.float64),
        },
        index=frame.index,
    )
    for region in sorted(set(canonical_regions)):
        selected = canonical_regions.eq(region)
        bundle = _region_bundle(models, region)
        if "component_model" not in bundle or "component_features" not in bundle:
            raise KeyError(f"{region}: incomplete persisted component model bundle")
        feature_order = tuple(str(column) for column in bundle["component_features"])
        _validate_feature_schema(frame, feature_order)
        model = bundle["component_model"]
        if not hasattr(model, "predict_proba"):
            raise TypeError(f"{region}: component model lacks predict_proba")
        probabilities = np.asarray(
            model.predict_proba(frame.loc[selected, list(feature_order)]),
            dtype=np.float64,
        )
        if probabilities.ndim != 2 or probabilities.shape != (int(selected.sum()), 2):
            raise ValueError(
                f"{region}: predict_proba returned {probabilities.shape}, "
                f"expected {(int(selected.sum()), 2)}"
            )
        scores = probabilities[:, 1]
        if not np.isfinite(scores).all() or np.any((scores < 0.0) | (scores > 1.0)):
            raise ValueError(f"{region}: model scores must be finite probabilities")
        output.loc[selected, "score"] = scores
    if not np.isfinite(output["score"]).all():
        raise RuntimeError("LCv2 scoring left unscored proposals")
    output["cutoff"] = [
        float(cutoffs[region]) for region in output["region"]
    ]
    output["accepted"] = output["score"] >= output["cutoff"]
    return output.reset_index(drop=True)[
        ["proposal_id", "region", "score", "cutoff", "accepted"]
    ]


def _proposal_probability(
    proposal: LocalProposal,
    probabilities: np.ndarray,
    channel: int,
) -> np.ndarray:
    local = np.asarray(probabilities[channel][proposal.bbox])
    if local.shape != proposal.local_mask.shape:
        raise ValueError(
            f"proposal {proposal.proposal_id}: bbox shape {local.shape} "
            f"differs from local mask {proposal.local_mask.shape}"
        )
    return local


def _proposal_model_peaks(
    proposal: LocalProposal,
    region: str,
    arrays: Mapping[str, np.ndarray],
) -> dict[str, float]:
    channel = CHANNELS.index(region.lower())
    peaks: dict[str, float] = {}
    for model in _MODELS:
        persisted = proposal.model_peaks.get(model, {}).get(region.lower())
        if persisted is not None:
            peaks[model] = float(persisted)
            continue
        local = _proposal_probability(proposal, arrays[model], channel)
        values = local[np.asarray(proposal.local_mask, dtype=bool)]
        peaks[model] = float(values.max()) if values.size else float("-inf")
    return peaks


def _two_of_three_from_peaks(peaks: Mapping[str, float]) -> bool:
    return sum(float(peaks[model]) >= 0.25 for model in _MODELS) >= 2


def _proposal_support(
    proposal: LocalProposal,
    region: str,
    arrays: Mapping[str, np.ndarray],
) -> tuple[dict[str, float], bool]:
    peaks = _proposal_model_peaks(proposal, region, arrays)
    # Use the shared helper whenever Task 2 already recorded this region's
    # peaks; otherwise use the same threshold on canonical local probabilities.
    recorded = all(
        region.lower() in proposal.model_peaks.get(model, {})
        for model in _MODELS
    )
    supported = (
        has_two_of_three_support(proposal, region)
        if recorded
        else _two_of_three_from_peaks(peaks)
    )
    return peaks, supported


def _bbox_record(bbox: tuple[slice, slice, slice]) -> tuple[tuple[int | None, int | None], ...]:
    return tuple((axis.start, axis.stop) for axis in bbox)


def _base_audit_row(
    spec: VariantSpec,
    region: str,
    proposal: LocalProposal,
    peaks: Mapping[str, float],
    lcv2_accepted: bool | None,
    lcv2_score: float | None,
    support: bool,
) -> dict[str, Any]:
    return {
        "case_id": getattr(proposal, "case_id", None),
        "variant_id": spec.variant_id,
        "region": region,
        "proposal_id": int(proposal.proposal_id),
        "bbox": _bbox_record(proposal.bbox),
        "volume_mm3": float(proposal.volume_mm3),
        "xl_peak": float(peaks["XL"]),
        "m_peak": float(peaks["M"]),
        "ft_peak": float(peaks["FT"]),
        "lcv2_score": lcv2_score,
        "lcv2_cutoff": _FIXED_LCV2_CUTOFFS[region] if spec.use_lcv2 else None,
        "lcv2_accepted": lcv2_accepted,
        "support_xl": bool(peaks["XL"] >= 0.25),
        "support_m": bool(peaks["M"] >= 0.25),
        "support_ft": bool(peaks["FT"] >= 0.25),
        "support_two_of_three": bool(support),
        "support_et": None,
        "support_tc": None,
        "support_wt": None,
        "decision": "consider",
        "shape_mode": spec.shape_mode,
        "added_voxels": 0,
        "deleted_voxels": 0,
    }


def _local_shape(
    spec: VariantSpec,
    region: str,
    proposal: LocalProposal,
    arrays: Mapping[str, np.ndarray],
) -> np.ndarray:
    channel = CHANNELS.index(region.lower())
    if spec.shape_mode == "proposal_local":
        return np.asarray(proposal.local_mask, dtype=bool).copy()
    if spec.shape_mode == "xl_core":
        return reconstruct_xl_core_shape(proposal, arrays["XL"][channel])
    if spec.shape_mode == "xlft_consensus":
        return reconstruct_xlft_consensus_shape(
            proposal,
            arrays["XL"][channel],
            arrays["FT"][channel],
        )
    raise ValueError(f"{spec.variant_id}: unsupported addition shape mode {spec.shape_mode}")


def _accepted_ids(
    accepted_proposal_ids: Mapping[str, set[int]],
    region: str,
) -> set[int]:
    values = accepted_proposal_ids.get(
        region,
        accepted_proposal_ids.get(region.lower(), set()),
    )
    return {int(value) for value in values}


def _proposal_score(
    proposal_scores: Mapping[str, Mapping[int, float]],
    region: str,
    proposal_id: int,
) -> float:
    scores = proposal_scores.get(
        region,
        proposal_scores.get(region.lower()),
    )
    if scores is None:
        raise KeyError(f"missing proposal_scores for region {region}")
    if proposal_id not in scores:
        raise KeyError(
            f"missing proposal_scores entry for {region} proposal {proposal_id}"
        )
    score = float(scores[proposal_id])
    if not np.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(
            f"{region} proposal {proposal_id}: LCv2 score must be a finite probability"
        )
    return score


def _validate_score_acceptance(
    region: str,
    proposal_id: int,
    score: float,
    accepted: bool,
) -> None:
    score_accepted = score >= _FIXED_LCV2_CUTOFFS[region]
    if score_accepted != accepted:
        raise ValueError(
            f"{region} proposal {proposal_id}: accepted ID membership {accepted} "
            f"disagrees with score={score} cutoff={_FIXED_LCV2_CUTOFFS[region]}"
        )


def _proposals_for(
    proposals_by_region: Mapping[str, list[LocalProposal]],
    region: str,
) -> list[LocalProposal]:
    return list(
        proposals_by_region.get(
            region,
            proposals_by_region.get(region.lower(), []),
        )
    )


def _place_local_mask(
    full_shape: tuple[int, ...],
    proposal: LocalProposal,
    local_mask: np.ndarray,
) -> np.ndarray:
    local = np.asarray(local_mask, dtype=bool)
    if local.shape != proposal.local_mask.shape:
        raise ValueError(
            f"proposal {proposal.proposal_id}: reconstructed shape {local.shape} "
            f"differs from proposal mask {proposal.local_mask.shape}"
        )
    if np.any(local & ~np.asarray(proposal.local_mask, dtype=bool)):
        raise AssertionError(
            f"proposal {proposal.proposal_id}: reconstructed addition escaped source proposal mask"
        )
    output = np.zeros(full_shape, dtype=bool)
    output[proposal.bbox] = local
    return output


def _apply_rc_veto(
    spec: VariantSpec,
    output: dict[str, np.ndarray],
    proposals: list[LocalProposal],
    arrays: Mapping[str, np.ndarray],
    accepted_ids: set[int],
    proposal_scores: Mapping[str, Mapping[int, float]],
) -> list[dict[str, Any]]:
    labels, _ = ndimage.label(
        output["rc"],
        structure=ndimage.generate_binary_structure(3, 1),
    )
    accepted_component_ids: set[int] = set()
    for proposal in proposals:
        if proposal.proposal_id not in accepted_ids:
            continue
        local_labels = labels[proposal.bbox]
        accepted_component_ids.update(
            int(value)
            for value in np.unique(local_labels[proposal.local_mask])
            if int(value) != 0
        )

    audit: list[dict[str, Any]] = []
    for proposal in proposals:
        peaks, support = _proposal_support(proposal, "RC", arrays)
        accepted = proposal.proposal_id in accepted_ids
        score = _proposal_score(
            proposal_scores,
            "RC",
            int(proposal.proposal_id),
        )
        _validate_score_acceptance("RC", int(proposal.proposal_id), score, accepted)
        row = _base_audit_row(
            spec,
            "RC",
            proposal,
            peaks,
            accepted,
            score,
            support,
        )
        if accepted:
            row["decision"] = "keep_lcv2"
            audit.append(row)
            continue
        local_labels = labels[proposal.bbox]
        component_ids = {
            int(value)
            for value in np.unique(local_labels[proposal.local_mask])
            if int(value) != 0
        }
        component_ids.difference_update(accepted_component_ids)
        deletion = np.isin(labels, tuple(component_ids)) & output["rc"]
        row["deleted_voxels"] = int(deletion.sum())
        output["rc"][deletion] = False
        row["decision"] = "delete_rejected_component" if component_ids else "reject_no_anchor_component"
        audit.append(row)
    return audit


def apply_nonrouter_variant(
    spec: VariantSpec,
    anchor: Mapping[str, np.ndarray],
    proposals_by_region: Mapping[str, list[LocalProposal]],
    model_probabilities: Mapping[str, np.ndarray],
    accepted_proposal_ids: Mapping[str, set[int]],
    *,
    proposal_scores: Mapping[str, Mapping[int, float]] | None = None,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    """Apply one frozen non-router variant without reprocessing its anchor."""
    canonical = _VARIANTS_BY_ID.get(spec.variant_id)
    if canonical is None:
        raise KeyError(f"unknown variant: {spec.variant_id}")
    if spec != canonical:
        raise ValueError(f"variant spec differs from frozen definition: {spec.variant_id}")
    if spec.use_lcv2 and proposal_scores is None:
        raise ValueError(f"{spec.variant_id}: proposal_scores are required for LCv2 variants")
    arrays = _validate_model_probabilities(model_probabilities)
    output = _copy_region_masks(anchor)
    full_shape = next(iter(output.values())).shape
    if arrays["XL"].shape[1:] != full_shape:
        raise ValueError(
            f"anchor/probability shape mismatch: {full_shape} != {arrays['XL'].shape[1:]}"
        )

    if spec.allowed_delete_regions:
        audit = _apply_rc_veto(
            spec,
            output,
            _proposals_for(proposals_by_region, "RC"),
            arrays,
            _accepted_ids(accepted_proposal_ids, "RC"),
            proposal_scores,
        )
        output = enforce_hierarchy(output)
        return output, audit

    audit: list[dict[str, Any]] = []
    hierarchy_additions = np.zeros(full_shape, dtype=bool)
    rc_additions = np.zeros(full_shape, dtype=bool)
    for region in spec.allowed_add_regions:
        accepted_ids = _accepted_ids(accepted_proposal_ids, region)
        for proposal in _proposals_for(proposals_by_region, region):
            peaks, support = _proposal_support(proposal, region, arrays)
            accepted = proposal.proposal_id in accepted_ids if spec.use_lcv2 else None
            score = (
                _proposal_score(proposal_scores, region, int(proposal.proposal_id))
                if spec.use_lcv2
                else None
            )
            if spec.use_lcv2:
                _validate_score_acceptance(
                    region,
                    int(proposal.proposal_id),
                    score,
                    bool(accepted),
                )
            row = _base_audit_row(
                spec,
                region,
                proposal,
                peaks,
                accepted,
                score,
                support,
            )
            if spec.use_lcv2 and not accepted:
                row["decision"] = "reject_lcv2"
                audit.append(row)
                continue
            if spec.require_consensus and not support:
                row["decision"] = "reject_consensus"
                audit.append(row)
                continue
            if spec.mechanism == "et_parent_supported":
                parent_support: dict[str, bool] = {}
                for parent_region in ("ET", "TC", "WT"):
                    _, parent_support[parent_region] = _proposal_support(
                        proposal,
                        parent_region,
                        arrays,
                    )
                    row[f"support_{parent_region.lower()}"] = parent_support[parent_region]
                if not all(parent_support.values()):
                    row["decision"] = "reject_parent_support"
                    audit.append(row)
                    continue

            local_shape = _local_shape(spec, region, proposal, arrays)
            addition = _place_local_mask(full_shape, proposal, local_shape)
            if region in _HIERARCHY_REGIONS:
                # Trusted anchor RC membership wins. RC additions already accepted
                # in this variant also remain protected from later hierarchy rows.
                addition &= ~output["rc"]
                addition &= ~rc_additions
                hierarchy_additions |= addition
            else:
                # Trusted anchor/hierarchy membership wins over an RC proposal.
                addition &= ~output["wt"]
                addition &= ~hierarchy_additions
                rc_additions |= addition
            target = output[region.lower()]
            novel = addition & ~target
            target |= addition
            row["added_voxels"] = int(novel.sum())
            row["decision"] = "add" if row["added_voxels"] else "accept_no_new_voxels"
            audit.append(row)

    # This is the sole hierarchy operation in the variant engine. Frozen V2,
    # boundary completion, and RC strict filtering are intentionally not rerun.
    output = enforce_hierarchy(output)
    assert_anchor_protected(anchor, output)
    return output, audit
