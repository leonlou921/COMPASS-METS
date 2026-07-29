"""Core region and localized component operations for learned component gating."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
from scipy import ndimage as ndi


REGION_NAMES = ("WT", "TC", "ET", "RC")


def load_probabilities(path: str | Path) -> np.ndarray:
    """Load and validate a four-channel WT/TC/ET/RC probability array."""
    with np.load(path) as data:
        if "probabilities" not in data:
            raise KeyError(f"{path} does not contain 'probabilities'")
        probabilities = np.asarray(data["probabilities"], dtype=np.float32)
    if probabilities.ndim != 4 or probabilities.shape[0] != len(REGION_NAMES):
        raise ValueError(f"expected shape (4, z, y, x), got {probabilities.shape}")
    if not np.isfinite(probabilities).all():
        raise ValueError(f"non-finite probabilities in {path}")
    return probabilities


def label_to_regions(label_zyx: np.ndarray) -> dict[str, np.ndarray]:
    """Convert BraTS labels to ordered WT/TC/ET/RC boolean region masks."""
    label = np.asarray(label_zyx)
    return {
        "WT": np.isin(label, (1, 2, 3)),
        "TC": np.isin(label, (1, 3)),
        "ET": label == 3,
        "RC": label == 4,
    }


def regions_to_label(regions: Mapping[str, np.ndarray]) -> np.ndarray:
    """Convert regions to legal labels, enforcing ET inside TC inside WT."""
    missing = set(REGION_NAMES).difference(regions)
    if missing:
        raise KeyError(f"missing regions: {sorted(missing)}")
    shape = np.asarray(regions["WT"]).shape
    if any(np.asarray(regions[name]).shape != shape for name in REGION_NAMES):
        raise ValueError("all region masks must have identical shapes")

    et = np.asarray(regions["ET"], dtype=bool)
    tc = np.asarray(regions["TC"], dtype=bool) | et
    wt = np.asarray(regions["WT"], dtype=bool) | tc
    rc = np.asarray(regions["RC"], dtype=bool)

    label = np.zeros(shape, dtype=np.uint8)
    label[wt] = 2
    label[tc] = 1
    label[et] = 3
    label[rc] = 4
    return label


def _validate_model_probabilities(
    probabilities_by_model: Mapping[str, np.ndarray],
) -> tuple[tuple[int, int, int], list[np.ndarray]]:
    if not probabilities_by_model:
        raise ValueError("at least one probability source is required")
    arrays = [np.asarray(value) for value in probabilities_by_model.values()]
    expected = arrays[0].shape
    if len(expected) != 4 or expected[0] != len(REGION_NAMES):
        raise ValueError(f"expected (4, z, y, x), got {expected}")
    if any(array.shape != expected for array in arrays):
        raise ValueError("model probability arrays must have identical shapes")
    return tuple(int(x) for x in expected[1:]), arrays


def propose_components(
    probabilities_by_model: Mapping[str, np.ndarray], threshold: float = 0.25
) -> dict[str, list[dict]]:
    """Propose localized 26-connected components from the model-wise maximum."""
    full_shape, arrays = _validate_model_probabilities(probabilities_by_model)
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    proposals: dict[str, list[dict]] = {name: [] for name in REGION_NAMES}
    for channel, region in enumerate(REGION_NAMES):
        candidate = np.logical_or.reduce([array[channel] >= threshold for array in arrays])
        coordinates = np.where(candidate)
        if coordinates[0].size == 0:
            continue
        outer_bbox = tuple(
            slice(int(axis.min()), int(axis.max()) + 1) for axis in coordinates
        )
        local_candidate = candidate[outer_bbox]
        labeled, count = ndi.label(local_candidate, structure=structure)
        objects = ndi.find_objects(labeled, max_label=count)
        outer_starts = tuple(sl.start for sl in outer_bbox)
        for component_id, local_bbox in enumerate(objects, start=1):
            if local_bbox is None:
                continue
            bbox = tuple(
                slice(start + sl.start, start + sl.stop)
                for start, sl in zip(outer_starts, local_bbox)
            )
            local_mask = labeled[local_bbox] == component_id
            proposals[region].append(
                {
                    "component_id": component_id,
                    "bbox": bbox,
                    "local_mask": local_mask,
                    "voxel_count": int(local_mask.sum()),
                    "full_shape": full_shape,
                }
            )
    return proposals


def match_component(
    component: Mapping,
    gt_region: np.ndarray,
    gt_size: int | None = None,
) -> dict[str, float | int]:
    """Match a localized proposal to a same-region ground-truth mask."""
    gt = np.asarray(gt_region, dtype=bool)
    if gt.shape != tuple(component["full_shape"]):
        raise ValueError(f"GT shape {gt.shape} differs from {component['full_shape']}")
    local_mask = np.asarray(component["local_mask"], dtype=bool)
    local_gt = gt[component["bbox"]]
    overlap = int(np.logical_and(local_mask, local_gt).sum())
    component_size = int(component["voxel_count"])
    gt_size = int(gt.sum()) if gt_size is None else int(gt_size)
    union = component_size + gt_size - overlap
    return {
        "target": int(overlap > 0),
        "overlap_voxels": overlap,
        "component_precision": overlap / component_size if component_size else 0.0,
        "gt_coverage": overlap / gt_size if gt_size else 0.0,
        "iou": overlap / union if union else 0.0,
    }


def equal_logit_probability(values: np.ndarray) -> np.ndarray:
    """Fuse a model axis by an equal-weight arithmetic mean in logit space."""
    clipped = np.clip(np.asarray(values, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    mean_logit = np.mean(np.log(clipped / (1.0 - clipped)), axis=0)
    return 1.0 / (1.0 + np.exp(-mean_logit))


def _component_reconstruction_masks(
    arrays: list[np.ndarray],
    channel: int,
    component: Mapping,
) -> dict[str, np.ndarray]:
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    bbox = component["bbox"]
    local_mask = np.asarray(component["local_mask"], dtype=bool)
    local_stack = np.stack([array[channel][bbox] for array in arrays], axis=0)
    base = np.logical_and(equal_logit_probability(local_stack) >= 0.50, local_mask)
    support = np.logical_and((local_stack >= 0.25).sum(axis=0) >= 2, local_mask)
    seed = np.logical_and((local_stack >= 0.50).any(axis=0), local_mask)
    graph = np.logical_or(support, seed)
    labeled, count = ndi.label(graph, structure=structure)
    rescue = np.zeros_like(local_mask)
    if count and seed.any():
        accepted_labels = np.unique(labeled[seed])
        accepted_labels = accepted_labels[accepted_labels > 0]
        rescue = np.logical_and(support, np.isin(labeled, accepted_labels))
    return {
        "filter_only": base,
        "consensus_rescue": np.logical_or(base, rescue),
    }


def prepare_component_reconstruction_cache(
    probabilities_by_model: Mapping[str, np.ndarray],
    proposals: Mapping[str, list[Mapping]],
) -> dict[tuple[str, int], dict[str, np.ndarray]]:
    """Compute each proposal's two immutable policy masks exactly once."""
    _, arrays = _validate_model_probabilities(probabilities_by_model)
    cache: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    for channel, region in enumerate(REGION_NAMES):
        for component in proposals.get(region, []):
            component_id = int(component["component_id"])
            cache[(region, component_id)] = _component_reconstruction_masks(
                arrays, channel, component
            )
    return cache


def reconstruct_regions(
    probabilities_by_model: Mapping[str, np.ndarray],
    proposals: Mapping[str, list[Mapping]],
    keep_decisions: Mapping[tuple[str, int], bool],
    policy: str = "filter_only",
    component_cache: Mapping[tuple[str, int], Mapping[str, np.ndarray]] | None = None,
    streaming: bool = False,
) -> dict[str, np.ndarray]:
    """Reconstruct kept proposal components under one of two fixed policies."""
    if policy not in {"filter_only", "consensus_rescue"}:
        raise ValueError(f"unknown reconstruction policy: {policy}")
    full_shape, arrays = _validate_model_probabilities(probabilities_by_model)
    output = {name: np.zeros(full_shape, dtype=bool) for name in REGION_NAMES}
    if component_cache is None and not streaming:
        component_cache = prepare_component_reconstruction_cache(probabilities_by_model, proposals)
    for channel, region in enumerate(REGION_NAMES):
        for component in proposals.get(region, []):
            component_id = int(component["component_id"])
            if not bool(keep_decisions.get((region, component_id), False)):
                continue
            bbox = component["bbox"]
            if streaming:
                local_cache = _component_reconstruction_masks(arrays, channel, component)
            else:
                assert component_cache is not None
                local_cache = component_cache[(region, component_id)]
            accepted = np.asarray(local_cache[policy], dtype=bool)
            local_output = output[region][bbox]
            np.logical_or(local_output, accepted, out=local_output)
    return output
