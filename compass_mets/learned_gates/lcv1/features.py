"""Feature extraction for case-presence and component-retention models."""

from __future__ import annotations

from collections.abc import Mapping
from math import pi

import numpy as np
from scipy import ndimage as ndi
from scipy import special

from compass_mets.learned_gates.lcv1.component_gate import (
    REGION_NAMES,
    equal_logit_probability,
    label_to_regions,
    match_component,
    propose_components,
)


MODEL_NAMES = ("XL", "M", "FT")
EPS = 1e-7
MAX_DISTRIBUTION_SAMPLES = 131_072
FROZEN_COMPONENT_CONF = {
    "minimum_volume": {"WT": 10, "TC": 10, "ET": 5, "RC": 20},
    "minimum_mean": {"WT": 0.0, "TC": 0.0, "ET": 0.55, "RC": 0.65},
    "minimum_peak": {"WT": 0.0, "TC": 0.0, "ET": 0.70, "RC": 0.80},
}


def equal_logit_fusion(arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    shape = np.asarray(arrays[MODEL_NAMES[0]]).shape
    fused = np.zeros(shape, dtype=np.float32)
    for channel in range(len(REGION_NAMES)):
        candidate = np.logical_or.reduce(
            [arrays[name][channel] >= 0.25 for name in MODEL_NAMES]
        )
        if not candidate.any():
            continue
        fused_values = np.zeros(int(candidate.sum()), dtype=np.float32)
        for name in MODEL_NAMES:
            temporary = np.asarray(arrays[name][channel][candidate], dtype=np.float32)
            np.clip(temporary, 1e-5, 1.0 - 1e-5, out=temporary)
            special.logit(temporary, out=temporary)
            fused_values += temporary / len(MODEL_NAMES)
        special.expit(fused_values, out=fused_values)
        fused[channel][candidate] = fused_values
    return fused


def _deterministic_sample(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values).reshape(-1)
    if flat.size <= MAX_DISTRIBUTION_SAMPLES:
        return flat
    stride = int(np.ceil(flat.size / MAX_DISTRIBUTION_SAMPLES))
    return flat[::stride]


def fixed_component_conf_masks(
    fused_probabilities: np.ndarray,
    spacing_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0),
    settings: Mapping | None = None,
) -> dict[str, np.ndarray]:
    """Apply the frozen M component_conf gates with original 6-connectivity."""
    settings = FROZEN_COMPONENT_CONF if settings is None else settings
    masks: dict[str, np.ndarray] = {}
    structure = ndi.generate_binary_structure(3, 1)
    voxel_volume = float(np.prod(spacing_zyx))
    for channel, region in enumerate(REGION_NAMES):
        score = np.asarray(fused_probabilities[channel])
        mask = score >= 0.50
        coordinates = np.where(mask)
        if coordinates[0].size == 0:
            masks[region] = np.zeros_like(mask)
            continue
        bbox = tuple(slice(int(axis.min()), int(axis.max()) + 1) for axis in coordinates)
        local_mask = mask[bbox]
        local_score = score[bbox]
        labels, count = ndi.label(local_mask, structure=structure)
        sizes = np.bincount(labels.ravel(), minlength=count + 1)
        keep = sizes.astype(np.float64) * voxel_volume >= float(
            settings["minimum_volume"][region]
        )
        minimum_mean = float(settings["minimum_mean"][region])
        minimum_peak = float(settings["minimum_peak"][region])
        if minimum_mean:
            sums = np.bincount(labels.ravel(), weights=local_score.ravel(), minlength=count + 1)
            means = np.divide(sums, sizes, out=np.zeros_like(sums), where=sizes > 0)
            keep &= means >= minimum_mean
        if minimum_peak:
            peaks = np.zeros(count + 1, dtype=np.float32)
            np.maximum.at(peaks, labels.ravel(), local_score.ravel())
            keep &= peaks >= minimum_peak
        keep[0] = False
        masks[region] = np.zeros_like(mask)
        masks[region][bbox] = keep[labels]
    masks["TC"] |= masks["ET"]
    masks["WT"] |= masks["TC"]
    return masks


def _probability_stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(values)
    if values.size == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_peak": 0.0,
            f"{prefix}_p90": 0.0,
            f"{prefix}_p95": 0.0,
            f"{prefix}_std": 0.0,
        }
    sample = _deterministic_sample(values).astype(np.float64, copy=False)
    return {
        f"{prefix}_mean": float(values.mean(dtype=np.float64)),
        f"{prefix}_peak": float(values.max()),
        f"{prefix}_p90": float(np.quantile(sample, 0.90)),
        f"{prefix}_p95": float(np.quantile(sample, 0.95)),
        f"{prefix}_std": float(sample.std()),
    }


def _case_probability_stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    sample = _deterministic_sample(values)
    stats = _probability_stats(sample, prefix)
    stats[f"{prefix}_peak"] = float(np.asarray(values).max())
    return stats


def _binary_entropy(probability: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=np.float64), EPS, 1.0 - EPS)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def _component_geometry(local_mask: np.ndarray, bbox: tuple[slice, ...]) -> dict[str, float | int]:
    mask = np.asarray(local_mask, dtype=bool)
    volume = int(mask.sum())
    extents = np.asarray([sl.stop - sl.start for sl in bbox], dtype=np.float64)
    bbox_volume = int(np.prod(extents))
    eroded = ndi.binary_erosion(mask, structure=np.ones((3, 3, 3), dtype=bool), border_value=0)
    surface = int(np.logical_and(mask, ~eroded).sum())
    compactness = float((36.0 * pi * volume * volume) ** (1.0 / 3.0) / surface) if surface else 0.0
    starts = np.asarray([sl.start for sl in bbox], dtype=np.float64)
    local_centroid = ndi.center_of_mass(mask) if volume else (0.0, 0.0, 0.0)
    centroid = np.asarray(local_centroid, dtype=np.float64) + starts
    return {
        "volume_voxels": volume,
        "bbox_z": int(extents[0]),
        "bbox_y": int(extents[1]),
        "bbox_x": int(extents[2]),
        "bbox_volume": bbox_volume,
        "bbox_fill_fraction": volume / bbox_volume if bbox_volume else 0.0,
        "surface_voxels": surface,
        "compactness": compactness,
        "aspect_ratio": float(extents.max() / max(extents.min(), 1.0)),
        "centroid_z": float(centroid[0]),
        "centroid_y": float(centroid[1]),
        "centroid_x": float(centroid[2]),
    }


def _case_region_features(
    case_id: str,
    fold: int,
    region: str,
    channel: int,
    arrays: Mapping[str, np.ndarray],
    gt_region: np.ndarray,
    component_count: int,
    fused: np.ndarray,
) -> dict[str, float | int | str]:
    row: dict[str, float | int | str] = {
        "case_id": case_id,
        "fold": int(fold),
        "region": region,
        "region_index": channel,
        "target": int(np.asarray(gt_region, dtype=bool).any()),
        "gt_voxels": int(np.asarray(gt_region, dtype=bool).sum()),
        "candidate_component_count": int(component_count),
    }
    channel_arrays = [arrays[name][channel] for name in MODEL_NAMES]
    for name, probability in zip(MODEL_NAMES, channel_arrays):
        row.update(_case_probability_stats(probability, name))
        row[f"{name}_volume_t025"] = int((probability >= 0.25).sum())
        row[f"{name}_volume_t050"] = int((probability >= 0.50).sum())
    row.update(_case_probability_stats(fused, "fused"))
    stride = max(1, int(np.ceil(fused.size / MAX_DISTRIBUTION_SAMPLES)))
    sampled_stack = np.stack([value.reshape(-1)[::stride] for value in channel_arrays], axis=0)
    sampled_fused = fused.reshape(-1)[::stride]
    row["mean_entropy"] = float(_binary_entropy(sampled_fused).mean())
    row["model_disagreement"] = float(sampled_stack.std(axis=0).mean())
    row["model_range_peak"] = float(
        (sampled_stack.max(axis=0) - sampled_stack.min(axis=0)).max()
    )
    return row


def _component_probability_features(
    component: Mapping,
    channel: int,
    arrays: Mapping[str, np.ndarray],
    fused_probabilities: np.ndarray,
    fused_base_masks: Mapping[str, np.ndarray],
) -> dict[str, float | int]:
    bbox = component["bbox"]
    mask = np.asarray(component["local_mask"], dtype=bool)
    local_values = {name: arrays[name][channel][bbox][mask] for name in MODEL_NAMES}
    stack = np.stack([local_values[name] for name in MODEL_NAMES], axis=0)
    fused = fused_probabilities[channel][bbox][mask]
    row: dict[str, float | int] = {}
    for name in MODEL_NAMES:
        values = local_values[name]
        row.update(_probability_stats(values, name))
        row[f"{name}_voxels_t025"] = int((values >= 0.25).sum())
        row[f"{name}_voxels_t050"] = int((values >= 0.50).sum())
        row[f"{name}_fraction_t025"] = float((values >= 0.25).mean())
        row[f"{name}_fraction_t050"] = float((values >= 0.50).mean())
    row.update(_probability_stats(fused, "fused"))
    for threshold, suffix in ((0.25, "t025"), (0.50, "t050")):
        support = (stack >= threshold).sum(axis=0)
        for count in (1, 2, 3):
            row[f"support{count}_{suffix}_fraction"] = float((support >= count).mean())
    stride = max(1, int(np.ceil(fused.size / MAX_DISTRIBUTION_SAMPLES)))
    sampled_stack = stack[:, ::stride]
    sampled_fused = fused[::stride]
    row["mean_entropy"] = float(_binary_entropy(sampled_fused).mean())
    row["model_disagreement"] = float(sampled_stack.std(axis=0).mean())
    row["model_range_mean"] = float(
        (sampled_stack.max(axis=0) - sampled_stack.min(axis=0)).mean()
    )
    for enclosing_region in REGION_NAMES:
        enclosing = fused_base_masks[enclosing_region][bbox][mask]
        row[f"within_{enclosing_region}_t050_fraction"] = float(enclosing.mean())
    return row


def extract_case_features(
    case_id: str,
    fold: int,
    xl: np.ndarray,
    m: np.ndarray,
    ft: np.ndarray,
    gt_label_zyx: np.ndarray,
) -> tuple[list[dict], list[dict]]:
    """Extract deterministic finite case and localized component features."""
    arrays = {"XL": np.asarray(xl), "M": np.asarray(m), "FT": np.asarray(ft)}
    expected = arrays["XL"].shape
    if len(expected) != 4 or expected[0] != 4:
        raise ValueError(f"expected four channels, got {expected}")
    if any(value.shape != expected for value in arrays.values()):
        raise ValueError("all model probability arrays must share a shape")
    if np.asarray(gt_label_zyx).shape != expected[1:]:
        raise ValueError("GT and probabilities must share z-y-x shape")

    gt_regions = label_to_regions(gt_label_zyx)
    gt_sizes = {region: int(mask.sum()) for region, mask in gt_regions.items()}
    proposals = propose_components(arrays, threshold=0.25)
    fused_probabilities = equal_logit_fusion(arrays)
    fused_base_masks = {
        region: fused_probabilities[channel] >= 0.50
        for channel, region in enumerate(REGION_NAMES)
    }
    fixed_baseline_masks = fixed_component_conf_masks(fused_probabilities)

    case_rows: list[dict] = []
    component_rows: list[dict] = []
    rows_by_region: dict[str, list[dict]] = {region: [] for region in REGION_NAMES}
    for channel, region in enumerate(REGION_NAMES):
        case_rows.append(
            _case_region_features(
                case_id,
                fold,
                region,
                channel,
                arrays,
                gt_regions[region],
                len(proposals[region]),
                fused_probabilities[channel],
            )
        )
        for component in proposals[region]:
            row: dict[str, float | int | str] = {
                "case_id": case_id,
                "fold": int(fold),
                "region": region,
                "region_index": channel,
                "component_id": int(component["component_id"]),
            }
            bbox = component["bbox"]
            for axis, sl in zip(("z", "y", "x"), bbox):
                row[f"bbox_{axis}0"] = int(sl.start)
                row[f"bbox_{axis}1"] = int(sl.stop)
            row.update(_component_geometry(component["local_mask"], bbox))
            row.update(
                _component_probability_features(
                    component, channel, arrays, fused_probabilities, fused_base_masks
                )
            )
            row["baseline_keep"] = int(
                fixed_baseline_masks[region][bbox][component["local_mask"]].any()
            )
            row.update(match_component(component, gt_regions[region], gt_size=gt_sizes[region]))
            rows_by_region[region].append(row)

    for region in REGION_NAMES:
        rows = rows_by_region[region]
        if not rows:
            continue
        volumes = np.asarray([row["volume_voxels"] for row in rows], dtype=np.float64)
        peaks = np.asarray([row["fused_peak"] for row in rows], dtype=np.float64)
        volume_order = np.argsort(-volumes, kind="stable")
        peak_order = np.argsort(-peaks, kind="stable")
        volume_ranks = np.empty(len(rows), dtype=np.int64)
        peak_ranks = np.empty(len(rows), dtype=np.int64)
        volume_ranks[volume_order] = np.arange(1, len(rows) + 1)
        peak_ranks[peak_order] = np.arange(1, len(rows) + 1)
        centroids = np.asarray(
            [[row["centroid_z"], row["centroid_y"], row["centroid_x"]] for row in rows],
            dtype=np.float64,
        )
        largest_index = int(volume_order[0])
        for index, row in enumerate(rows):
            row["volume_rank"] = int(volume_ranks[index])
            row["peak_rank"] = int(peak_ranks[index])
            if len(rows) == 1 or index == largest_index:
                row["nearest_large_distance"] = -1.0
                row["nearest_large_distance_missing"] = 1
            else:
                row["nearest_large_distance"] = float(
                    np.linalg.norm(centroids[index] - centroids[largest_index])
                )
                row["nearest_large_distance_missing"] = 0
            component_rows.append(row)

    return case_rows, component_rows
