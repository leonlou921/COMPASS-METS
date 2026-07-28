"""Leakage-safe complete-case routers for the frozen BraTS portfolio."""

from __future__ import annotations

import importlib.util
import json
import math
import platform
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import joblib
import lightgbm
import nibabel as nib
import numpy as np
import pandas as pd
import scipy
import sklearn
from lightgbm import LGBMClassifier
from scipy import ndimage
from scipy.spatial import cKDTree


SEED = 20260726
ROUTER_PAIRS = {
    "N09_Router_XF12_vs_LCprotected_case_safe": (
        "XF12",
        "LCv2_structured_protected_rescue_V2",
    ),
    "N10_Router_XF16_vs_LCXLFT_case_safe": (
        "XF16",
        "LCv2_XLFT_filter_V2",
    ),
}

_REGIONS = ("ET", "RC", "TC", "WT")
_MODELS = ("XL", "M", "FT")
# Production uses the frozen default. Tests and constrained callers may request a
# smaller chunk, but values above the reviewed bound are rejected.
ROUTER_FEATURE_CHUNK_VOXELS = 1 << 20
# Conservative disagreement bound: three nditer float64 conversion buffers,
# float64 minimum/maximum work arrays, one spare float64 ufunc buffer, and two
# one-byte boolean/count buffers. This excludes the original probability arrays.
ROUTER_FEATURE_PEAK_SCRATCH_BYTES = ROUTER_FEATURE_CHUNK_VOXELS * (6 * 8 + 2)
ROUTER_DISTANCE_SCAN_CHUNK_VOXELS = 1 << 20
ROUTER_DISTANCE_QUERY_CHUNK_POINTS = 1 << 18
ROUTER_DISTANCE_MAX_BOUNDARY_POINTS = 1 << 20
# Explicit NumPy coordinate/query payload. cKDTree's native index is separately
# bounded by ROUTER_DISTANCE_MAX_BOUNDARY_POINTS.
ROUTER_DISTANCE_EXPLICIT_COORDINATE_SCRATCH_BYTES = (
    24 * ROUTER_DISTANCE_MAX_BOUNDARY_POINTS
    + 8 * ROUTER_DISTANCE_SCAN_CHUNK_VOXELS
    + 56 * ROUTER_DISTANCE_QUERY_CHUNK_POINTS
)
_FORBIDDEN_FEATURE_TERMS = (
    "label",
    "target",
    "dice",
    "dsc",
    "nsd",
    "hd95",
    "tp",
    "fp",
    "fn",
)
_METRIC_ALIASES = {
    "small F1": (
        "macro_small_instance_f1",
        "macro_small_f1",
        "small_instance_f1",
        "small_f1",
    ),
    "DSC": (
        "macro_lesionwise_dsc",
        "macro_lesion_wise_dsc",
        "macro_dsc",
        "lesionwise_dsc_mean",
    ),
    "NSD": (
        "macro_lesionwise_nsd",
        "macro_lesion_wise_nsd",
        "macro_nsd",
        "lesionwise_nsd_mean",
    ),
}


def _case_index(frame: pd.DataFrame, description: str) -> pd.DataFrame:
    table = frame.copy()
    if "case_id" in table.columns:
        identifiers = table.pop("case_id")
    else:
        identifiers = pd.Series(table.index, index=table.index)
    identifiers = identifiers.map(str)
    if identifiers.duplicated().any():
        duplicates = identifiers[identifiers.duplicated(keep=False)].tolist()
        raise ValueError(f"{description} has duplicate case IDs: {duplicates[:5]}")
    table.index = pd.Index(identifiers, name="case_id")
    return table


def _metric_columns(frame: pd.DataFrame, description: str) -> list[str]:
    selected: list[str] = []
    for metric, aliases in _METRIC_ALIASES.items():
        present = [column for column in aliases if column in frame.columns]
        if len(present) != 1:
            if not present:
                raise KeyError(f"{description} is missing router metric {metric}")
            raise ValueError(
                f"{description} has ambiguous router metric {metric}: {present}"
            )
        selected.append(present[0])
    values = frame.loc[:, selected].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{description} router metrics must be finite")
    return selected


def build_router_target(
    anchor_metrics: pd.DataFrame,
    candidate_metrics: pd.DataFrame,
) -> pd.Series:
    """Build a candidate-win target from three strict within-case rank units."""
    anchor = _case_index(anchor_metrics, "anchor metrics")
    candidate = _case_index(candidate_metrics, "candidate metrics")
    anchor_ids = set(anchor.index)
    candidate_ids = set(candidate.index)
    if anchor_ids != candidate_ids:
        missing = sorted(anchor_ids - candidate_ids)
        extra = sorted(candidate_ids - anchor_ids)
        raise RuntimeError(
            f"anchor and candidate case IDs differ: missing={missing[:5]} "
            f"extra={extra[:5]}"
        )
    anchor_columns = _metric_columns(anchor, "anchor metrics")
    candidate_columns = _metric_columns(candidate, "candidate metrics")
    candidate = candidate.loc[anchor.index]
    anchor_values = anchor.loc[:, anchor_columns].to_numpy(dtype=np.float64)
    candidate_values = candidate.loc[:, candidate_columns].to_numpy(dtype=np.float64)

    anchor_units = (anchor_values < candidate_values).sum(axis=1)
    candidate_units = (candidate_values < anchor_values).sum(axis=1)
    target = (candidate_units < anchor_units).astype(np.int8)
    return pd.Series(target, index=anchor.index.copy(), dtype=np.int8, name="router_choice")


def validate_router_feature_names(names: Sequence[str] | Mapping[str, Any]) -> None:
    """Reject feature names that could encode outcomes or ground truth."""
    for raw_name in names:
        name = str(raw_name).casefold()
        for forbidden in _FORBIDDEN_FEATURE_TERMS:
            if forbidden in name:
                raise ValueError(
                    f"forbidden router feature name {raw_name!r} contains {forbidden!r}"
                )


def _region_arrays(
    regions: Mapping[str, np.ndarray],
    description: str,
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for region in _REGIONS:
        key = region if region in regions else region.lower()
        if key not in regions:
            raise KeyError(f"{description} is missing region {region}")
        array = np.asarray(regions[key], dtype=bool)
        if array.ndim != 3:
            raise ValueError(f"{description} {region} must be three-dimensional")
        output[region] = array
    shapes = {array.shape for array in output.values()}
    if len(shapes) != 1:
        raise ValueError(f"{description} region shapes differ: {sorted(shapes)}")
    return output


def _component_count(mask: np.ndarray) -> int:
    return int(
        ndimage.label(
            mask,
            structure=np.ones((3, 3, 3), dtype=np.uint8),
        )[1]
    )


def _hierarchy_violation_count(regions: Mapping[str, np.ndarray]) -> int:
    et_outside_tc = np.logical_and(regions["ET"], ~regions["TC"])
    tc_outside_wt = np.logical_and(regions["TC"], ~regions["WT"])
    return int(et_outside_tc.sum() + tc_outside_wt.sum())


def _probability_arrays(
    model_probabilities: Mapping[str, np.ndarray],
    shape: tuple[int, int, int],
    chunk_voxels: int,
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for model in _MODELS:
        if model not in model_probabilities:
            raise KeyError(f"missing model probabilities: {model}")
        array = np.asarray(model_probabilities[model])
        if array.shape != (4, *shape):
            raise ValueError(
                f"{model} probabilities have shape {array.shape}, "
                f"expected {(4, *shape)}"
            )
        for chunk in _iter_float64_chunks(array, chunk_voxels):
            if not np.isfinite(chunk).all() or np.any(
                (chunk < 0.0) | (chunk > 1.0)
            ):
                raise ValueError(
                    f"{model} probabilities must be finite in [0, 1]"
                )
        output[model] = array
    return output


def _validate_probability_chunk_voxels(value: int) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) <= 0
        or int(value) > ROUTER_FEATURE_CHUNK_VOXELS
    ):
        raise ValueError(
            "probability_chunk_voxels must be an integer in "
            f"[1, {ROUTER_FEATURE_CHUNK_VOXELS}]"
        )
    return int(value)


def _iter_float64_chunks(
    array: np.ndarray,
    chunk_voxels: int,
):
    iterator = np.nditer(
        array,
        flags=("external_loop", "buffered", "zerosize_ok"),
        op_flags=("readonly",),
        op_dtypes=(np.float64,),
        order="C",
        buffersize=chunk_voxels,
        casting="unsafe",
    )
    yield from iterator


def _probability_summary(
    values: np.ndarray,
    chunk_voxels: int,
) -> tuple[float, float, float]:
    probability_sums: list[float] = []
    entropy_sums: list[float] = []
    peak = -math.inf
    count = int(values.size)
    if count == 0:
        raise ValueError("probability volumes cannot be empty")
    for chunk in _iter_float64_chunks(values, chunk_voxels):
        probability_sums.append(float(np.sum(chunk, dtype=np.float64)))
        peak = max(peak, float(np.max(chunk)))

        clipped = np.clip(chunk, 1e-12, 1.0 - 1e-12)
        entropy_term = np.log2(clipped)
        entropy_term *= clipped
        complement = np.subtract(1.0, clipped)
        np.log2(complement, out=clipped)
        complement *= clipped
        entropy_term += complement
        entropy_sums.append(-float(np.sum(entropy_term, dtype=np.float64)))
    return (
        math.fsum(probability_sums) / count,
        peak,
        math.fsum(entropy_sums) / count,
    )


def _model_disagreement_summary(
    values: Sequence[np.ndarray],
    chunk_voxels: int,
) -> tuple[float, float, float]:
    count = int(values[0].size)
    if count == 0:
        raise ValueError("probability volumes cannot be empty")
    iterator = np.nditer(
        tuple(values),
        flags=("external_loop", "buffered", "zerosize_ok"),
        op_flags=(("readonly",),) * len(values),
        op_dtypes=(np.float64,) * len(values),
        order="C",
        buffersize=chunk_voxels,
        casting="unsafe",
    )
    spread_sums: list[float] = []
    peak = -math.inf
    support_voxels = 0
    for chunks in iterator:
        minimum = np.array(chunks[0], dtype=np.float64, copy=True)
        maximum = minimum.copy()
        support_count = np.zeros(minimum.shape, dtype=np.uint8)
        for chunk in chunks:
            np.minimum(minimum, chunk, out=minimum)
            np.maximum(maximum, chunk, out=maximum)
            np.add(
                support_count,
                chunk >= 0.5,
                out=support_count,
                casting="unsafe",
            )
        np.subtract(maximum, minimum, out=maximum)
        spread_sums.append(float(np.sum(maximum, dtype=np.float64)))
        peak = max(peak, float(np.max(maximum)))
        support_voxels += int(np.count_nonzero(support_count >= 2))
    return (
        math.fsum(spread_sums) / count,
        peak,
        support_voxels / count,
    )


def _audit_column(
    audit: pd.DataFrame,
    aliases: tuple[str, ...],
    description: str,
) -> str:
    present = [column for column in aliases if column in audit.columns]
    if len(present) != 1:
        if not present:
            raise KeyError(f"proposal audit is missing {description}")
        raise ValueError(f"proposal audit has ambiguous {description}: {present}")
    return present[0]


def _maximum_addition_distance(
    anchor: Mapping[str, np.ndarray],
    candidate: Mapping[str, np.ndarray],
) -> float:
    shape = anchor[_REGIONS[0]].shape
    anchor_union = np.zeros(shape, dtype=bool)
    additions = np.zeros(shape, dtype=bool)
    scratch = np.empty(shape, dtype=bool)
    for region in _REGIONS:
        np.logical_or(anchor_union, anchor[region], out=anchor_union)
        np.logical_not(anchor[region], out=scratch)
        np.logical_and(candidate[region], scratch, out=scratch)
        np.logical_or(additions, scratch, out=additions)
    del scratch
    if not additions.any():
        return 0.0
    if not anchor_union.any():
        shape = np.asarray(additions.shape, dtype=np.float64)
        return float(np.sqrt(np.square(np.maximum(shape - 1.0, 0.0)).sum()) + 1.0)

    # Additions already covered by any anchor region have distance zero and
    # cannot increase the maximum. Removing them also makes an anchor boundary
    # sufficient for exact nearest-neighbour queries.
    additions[anchor_union] = False
    if not additions.any():
        return 0.0

    eroded = ndimage.binary_erosion(
        anchor_union,
        structure=ndimage.generate_binary_structure(3, 1),
        border_value=0,
    )
    np.logical_xor(anchor_union, eroded, out=eroded)
    boundary_coordinates = _bounded_mask_coordinates(
        eroded,
        maximum_points=ROUTER_DISTANCE_MAX_BOUNDARY_POINTS,
        description="anchor boundary",
    )
    del anchor_union, eroded
    tree = cKDTree(
        boundary_coordinates,
        compact_nodes=True,
        balanced_tree=True,
        copy_data=False,
    )
    maximum = 0.0
    for query_coordinates in _iter_mask_coordinate_chunks(additions):
        distances, indices = tree.query(
            query_coordinates,
            k=1,
            eps=0.0,
            p=2,
            workers=1,
        )
        maximum = max(maximum, float(np.max(distances)))
        del distances, indices, query_coordinates
    return maximum


def _iter_mask_coordinate_chunks(mask: np.ndarray):
    flat = mask.reshape(-1)
    plane = int(mask.shape[1] * mask.shape[2])
    width = int(mask.shape[2])
    scan_chunk = ROUTER_DISTANCE_SCAN_CHUNK_VOXELS
    coordinate_chunk = ROUTER_DISTANCE_QUERY_CHUNK_POINTS
    for start in range(0, flat.size, scan_chunk):
        flat_indices = np.flatnonzero(flat[start : start + scan_chunk])
        if not flat_indices.size:
            continue
        flat_indices += start
        for offset in range(0, flat_indices.size, coordinate_chunk):
            indices = flat_indices[offset : offset + coordinate_chunk]
            coordinates = np.empty((indices.size, 3), dtype=np.float64)
            coordinates[:, 0] = np.floor_divide(indices, plane)
            remainder = np.remainder(indices, plane)
            coordinates[:, 1] = np.floor_divide(remainder, width)
            coordinates[:, 2] = np.remainder(remainder, width)
            yield coordinates


def _bounded_mask_coordinates(
    mask: np.ndarray,
    *,
    maximum_points: int,
    description: str,
) -> np.ndarray:
    count = int(np.count_nonzero(mask))
    if count > maximum_points:
        raise MemoryError(
            f"{description} has {count} points, exceeding the "
            f"memory-safe limit {maximum_points}"
        )
    coordinates = np.empty((count, 3), dtype=np.float64)
    offset = 0
    for chunk in _iter_mask_coordinate_chunks(mask):
        next_offset = offset + len(chunk)
        coordinates[offset:next_offset] = chunk
        offset = next_offset
    if offset != count:
        raise RuntimeError(
            f"{description} coordinate extraction count mismatch: "
            f"expected {count}, got {offset}"
        )
    return coordinates


def build_router_features(
    anchor_regions: Mapping[str, np.ndarray],
    candidate_regions: Mapping[str, np.ndarray],
    model_probabilities: Mapping[str, np.ndarray],
    proposal_audit: pd.DataFrame,
    *,
    probability_chunk_voxels: int = ROUTER_FEATURE_CHUNK_VOXELS,
) -> dict[str, float]:
    """Summarize only pre-ground-truth case evidence for router inference."""
    chunk_voxels = _validate_probability_chunk_voxels(probability_chunk_voxels)
    validate_router_feature_names(proposal_audit.columns)
    anchor = _region_arrays(anchor_regions, "anchor")
    candidate = _region_arrays(candidate_regions, "candidate")
    shape = next(iter(anchor.values())).shape
    if {array.shape for array in candidate.values()} != {shape}:
        raise ValueError("anchor and candidate region shapes differ")
    probabilities = _probability_arrays(model_probabilities, shape, chunk_voxels)

    features: dict[str, float] = {}
    for region in _REGIONS:
        prefix = region.lower()
        anchor_voxels = int(anchor[region].sum())
        candidate_voxels = int(candidate[region].sum())
        anchor_components = _component_count(anchor[region])
        candidate_components = _component_count(candidate[region])
        features[f"anchor_{prefix}_voxel_count"] = float(anchor_voxels)
        features[f"candidate_{prefix}_voxel_count"] = float(candidate_voxels)
        features[f"{prefix}_voxel_count_delta"] = float(
            candidate_voxels - anchor_voxels
        )
        features[f"anchor_{prefix}_component_count"] = float(anchor_components)
        features[f"candidate_{prefix}_component_count"] = float(candidate_components)
        features[f"{prefix}_component_count_delta"] = float(
            candidate_components - anchor_components
        )

    features["anchor_hierarchy_violation_count"] = float(
        _hierarchy_violation_count(anchor)
    )
    features["candidate_hierarchy_violation_count"] = float(
        _hierarchy_violation_count(candidate)
    )
    features["maximum_addition_anchor_distance_voxels"] = _maximum_addition_distance(
        anchor, candidate
    )

    # Channel meaning is inherited from the canonical frozen fusion module.
    from portfolio_variants import CHANNELS

    for model, array in probabilities.items():
        for region in _REGIONS:
            values = array[CHANNELS.index(region.lower())]
            mean_probability, peak_probability, mean_entropy = _probability_summary(
                values,
                chunk_voxels,
            )
            prefix = f"{model.lower()}_{region.lower()}"
            features[f"{prefix}_mean_probability"] = mean_probability
            features[f"{prefix}_peak_probability"] = peak_probability
            features[f"{prefix}_mean_entropy"] = mean_entropy

    for region in _REGIONS:
        channel = CHANNELS.index(region.lower())
        mean_disagreement, peak_disagreement, support_fraction = (
            _model_disagreement_summary(
                [probabilities[model][channel] for model in _MODELS],
                chunk_voxels,
            )
        )
        prefix = region.lower()
        features[f"{prefix}_mean_model_disagreement"] = mean_disagreement
        features[f"{prefix}_peak_model_disagreement"] = peak_disagreement
        features[f"{prefix}_two_of_three_support_fraction"] = support_fraction

    audit = proposal_audit.copy()
    if audit.empty:
        features.update(
            {
                "accepted_proposal_count": 0.0,
                "rejected_proposal_count": 0.0,
                "accepted_proposal_volume_sum_mm3": 0.0,
                "rejected_proposal_volume_sum_mm3": 0.0,
                "proposal_two_of_three_support_fraction": 0.0,
            }
        )
        for quantile in (0, 25, 50, 75, 100):
            features[f"lcv2_score_q{quantile:02d}"] = 0.0
        features["lcv2_cutoff_margin_mean"] = 0.0
        features["lcv2_cutoff_margin_minimum"] = 0.0
        features["lcv2_cutoff_margin_maximum"] = 0.0
    else:
        accepted_column = _audit_column(
            audit,
            ("accepted", "lcv2_accepted"),
            "acceptance column",
        )
        volume_column = _audit_column(
            audit,
            ("physical_volume_mm3", "volume_mm3"),
            "physical volume column",
        )
        score_column = _audit_column(
            audit,
            ("lcv2_score", "score"),
            "LCv2 score column",
        )
        cutoff_column = _audit_column(
            audit,
            ("lcv2_cutoff", "cutoff"),
            "LCv2 cutoff column",
        )
        accepted = audit[accepted_column]
        if accepted.isna().any():
            raise ValueError("proposal audit acceptance values must be complete")
        accepted_values = accepted.astype(bool).to_numpy()
        volumes = audit[volume_column].to_numpy(dtype=np.float64)
        scores = audit[score_column].to_numpy(dtype=np.float64)
        cutoffs = audit[cutoff_column].to_numpy(dtype=np.float64)
        if (
            not np.isfinite(volumes).all()
            or not np.isfinite(scores).all()
            or not np.isfinite(cutoffs).all()
        ):
            raise ValueError("proposal audit summaries must be finite")
        if np.any(volumes < 0.0):
            raise ValueError("proposal physical volumes cannot be negative")
        features["accepted_proposal_count"] = float(accepted_values.sum())
        features["rejected_proposal_count"] = float((~accepted_values).sum())
        features["accepted_proposal_volume_sum_mm3"] = float(
            volumes[accepted_values].sum()
        )
        features["rejected_proposal_volume_sum_mm3"] = float(
            volumes[~accepted_values].sum()
        )
        support_columns = ("support_xl", "support_m", "support_ft")
        if not set(support_columns).issubset(audit.columns):
            raise KeyError("proposal audit is missing model support columns")
        support_count = (
            audit.loc[:, list(support_columns)].astype(bool).sum(axis=1).to_numpy()
        )
        features["proposal_two_of_three_support_fraction"] = float(
            np.mean(support_count >= 2)
        )
        for quantile in (0, 25, 50, 75, 100):
            features[f"lcv2_score_q{quantile:02d}"] = float(
                np.quantile(scores, quantile / 100.0)
            )
        margins = scores - cutoffs
        features["lcv2_cutoff_margin_mean"] = float(margins.mean())
        features["lcv2_cutoff_margin_minimum"] = float(margins.min())
        features["lcv2_cutoff_margin_maximum"] = float(margins.max())

    validate_router_feature_names(features)
    values = np.asarray(list(features.values()), dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("router features must be finite")
    return features


def _feature_frame(features: pd.DataFrame) -> pd.DataFrame:
    frame = features.copy()
    if "case_id" in frame.columns:
        identifiers = frame.pop("case_id").map(str)
        frame.index = pd.Index(identifiers, name="case_id")
    else:
        frame.index = pd.Index([str(value) for value in frame.index], name="case_id")
    if frame.index.duplicated().any():
        duplicates = frame.index[frame.index.duplicated(keep=False)].tolist()
        raise ValueError(f"duplicate feature case IDs: {duplicates[:5]}")
    validate_router_feature_names(frame.columns)
    if frame.columns.duplicated().any():
        raise ValueError("router feature columns must be unique")
    if frame.empty or not len(frame.columns):
        raise ValueError("router feature table cannot be empty")
    values = frame.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("router feature table must be finite")
    return frame


def _aligned_series(
    values: pd.Series,
    index: pd.Index,
    description: str,
) -> pd.Series:
    series = values.copy()
    series.index = pd.Index([str(value) for value in series.index], name="case_id")
    if series.index.duplicated().any():
        raise ValueError(f"{description} has duplicate case IDs")
    if set(series.index) != set(index):
        raise RuntimeError(f"{description} case IDs differ from feature case IDs")
    return series.loc[index]


def _fold_values(folds: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(folds, errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError("fold values must be finite integers")
    if np.any((numeric < 0.0) | (numeric > 4.0)):
        raise ValueError("fold values must be within 0..4")
    result = pd.Series(numeric.astype(np.int8), index=folds.index, name="fold")
    unique = sorted(int(value) for value in result.unique())
    if unique != [0, 1, 2, 3, 4]:
        raise ValueError(f"router training requires folds [0, 1, 2, 3, 4], got {unique}")
    return result


def _versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "lightgbm": lightgbm.__version__,
        "joblib": joblib.__version__,
    }


def train_crossfit_routers(
    features: pd.DataFrame,
    targets: pd.Series,
    folds: pd.Series,
    output_dir: Path,
) -> list[object]:
    """Fit and persist five binary routers, each excluding its held-out fold."""
    frame = _feature_frame(features)
    target = _aligned_series(targets, frame.index, "targets")
    fold = _fold_values(_aligned_series(folds, frame.index, "folds"))
    target_values = pd.to_numeric(target, errors="raise").to_numpy(dtype=np.float64)
    if (
        not np.isfinite(target_values).all()
        or not np.isin(target_values, (0.0, 1.0)).all()
    ):
        raise ValueError("router targets must be finite binary values")
    target = pd.Series(target_values.astype(np.int8), index=frame.index)

    split_metadata: list[dict[str, Any]] = []
    for held_out_fold in range(5):
        held_out = fold.eq(held_out_fold)
        train_ids = frame.index[~held_out].tolist()
        held_out_ids = frame.index[held_out].tolist()
        overlap = set(train_ids).intersection(held_out_ids)
        if overlap:
            raise RuntimeError(
                f"subject overlap in router fold {held_out_fold}: {sorted(overlap)[:5]}"
            )
        class_counts = target.loc[train_ids].value_counts().sort_index()
        if set(int(value) for value in class_counts.index) != {0, 1}:
            raise ValueError(
                f"router fold {held_out_fold} training split has a single class"
            )
        split_metadata.append(
            {
                "held_out_fold": held_out_fold,
                "train_ids": train_ids,
                "held_out_ids": held_out_ids,
                "class_counts": {
                    str(class_id): int(class_counts.loc[class_id])
                    for class_id in (0, 1)
                },
                "prevalence": float(target.loc[train_ids].mean()),
            }
        )

    root = Path(output_dir)
    models: list[object] = []
    feature_names = [str(column) for column in frame.columns]
    library_versions = _versions()
    for metadata in split_metadata:
        held_out_fold = int(metadata["held_out_fold"])
        train_ids = metadata["train_ids"]
        model = LGBMClassifier(
            objective="binary",
            n_estimators=64,
            learning_rate=0.05,
            max_depth=3,
            num_leaves=7,
            min_child_samples=5,
            subsample=1.0,
            colsample_bytree=1.0,
            random_state=SEED,
            n_jobs=1,
            verbosity=-1,
            deterministic=True,
            force_col_wise=True,
        )
        model.fit(frame.loc[train_ids, feature_names], target.loc[train_ids])
        model._router_held_out_fold = held_out_fold
        model._router_feature_names = tuple(feature_names)
        model._router_train_ids = tuple(train_ids)
        model._router_held_out_ids = tuple(metadata["held_out_ids"])

        persisted = {
            "model": model,
            "feature_names": feature_names,
            "train_ids": train_ids,
            "held_out_ids": metadata["held_out_ids"],
            "class_counts": metadata["class_counts"],
            "prevalence": metadata["prevalence"],
            "held_out_fold": held_out_fold,
            "seed": SEED,
            "library_versions": library_versions,
        }
        fold_dir = root / f"fold_{held_out_fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(persisted, fold_dir / "models.joblib")
        (fold_dir / "metadata.json").write_text(
            json.dumps(
                {key: value for key, value in persisted.items() if key != "model"},
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        models.append(model)
    return models


def _model_and_features(item: object) -> tuple[object, tuple[str, ...]]:
    if isinstance(item, Mapping):
        if "model" not in item or "feature_names" not in item:
            raise KeyError("persisted router bundle lacks model or feature_names")
        return item["model"], tuple(str(value) for value in item["feature_names"])
    if not hasattr(item, "_router_feature_names"):
        raise ValueError("router model lacks its exact persisted feature list")
    return item, tuple(str(value) for value in item._router_feature_names)


def crossfit_oof_probabilities(
    models: Sequence[object],
    features: pd.DataFrame,
    folds: pd.Series,
) -> pd.Series:
    """Predict every OOF row only with the model that excluded its fold."""
    if len(models) != 5:
        raise ValueError(f"expected five router models, got {len(models)}")
    frame = _feature_frame(features)
    fold = _fold_values(_aligned_series(folds, frame.index, "folds"))
    by_fold: dict[int, object] = {}
    for item in models:
        model, feature_names = _model_and_features(item)
        held_out_fold = (
            int(item["held_out_fold"])
            if isinstance(item, Mapping)
            else int(model._router_held_out_fold)
        )
        if held_out_fold in by_fold:
            raise ValueError(f"duplicate router model for fold {held_out_fold}")
        if tuple(frame.columns) != feature_names:
            raise ValueError("router model feature list differs from OOF features")
        by_fold[held_out_fold] = model
    if set(by_fold) != set(range(5)):
        raise ValueError(f"router held-out folds are incomplete: {sorted(by_fold)}")

    result = pd.Series(np.nan, index=frame.index, dtype=np.float64, name="probability")
    for held_out_fold in range(5):
        selected = fold.eq(held_out_fold)
        probability = np.asarray(
            by_fold[held_out_fold].predict_proba(frame.loc[selected])[:, 1],
            dtype=np.float64,
        )
        result.loc[selected] = probability
    if not np.isfinite(result.to_numpy()).all() or not result.between(0.0, 1.0).all():
        raise RuntimeError("incomplete or invalid cross-fit router probabilities")
    return result


def unanimous_candidate_vote(
    models: Sequence[object],
    row: pd.DataFrame,
) -> bool:
    """Select the candidate only when all five model probabilities reach 0.50."""
    if len(models) != 5:
        raise ValueError(f"unanimous routing requires five models, got {len(models)}")
    if len(row) != 1:
        raise ValueError("router vote requires exactly one feature row")
    votes: list[bool] = []
    for item in models:
        model, feature_names = _model_and_features(item)
        if list(row.columns) != list(feature_names):
            raise ValueError("router row differs from persisted feature list")
        values = row.to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("router row must be finite")
        probability = np.asarray(model.predict_proba(row), dtype=np.float64)
        if probability.shape != (1, 2) or not np.isfinite(probability).all():
            raise RuntimeError(
                f"router predict_proba returned invalid shape/value {probability.shape}"
            )
        votes.append(bool(probability[0, 1] >= 0.50))
    return all(votes)


def choose_complete_case(
    anchor: np.ndarray,
    candidate: np.ndarray,
    use_candidate: bool,
) -> np.ndarray:
    """Return an independent copy of exactly one complete segmentation."""
    anchor_array = np.asarray(anchor)
    candidate_array = np.asarray(candidate)
    if anchor_array.shape != candidate_array.shape:
        raise ValueError(
            f"anchor and candidate shapes differ: "
            f"{anchor_array.shape} != {candidate_array.shape}"
        )
    return np.array(candidate_array if use_candidate else anchor_array, copy=True)


def assert_existing_oof_metrics_match(
    rebuilt_metrics: pd.DataFrame,
    stored_metrics: pd.DataFrame,
    candidate: str,
    tolerance: float = 1e-12,
) -> None:
    """Hard-fail unless rebuilt candidate metrics reproduce every stored scalar."""
    if tolerance != 1e-12:
        raise ValueError("LCv2 OOF comparison tolerance is frozen at 1e-12")

    def selected(frame: pd.DataFrame, description: str) -> pd.DataFrame:
        table = frame.copy()
        if "candidate" in table.columns:
            table = table.loc[table["candidate"].astype(str).eq(candidate)].copy()
        if table.empty:
            raise RuntimeError(f"{description} has no rows for {candidate}")
        return _case_index(table, description)

    rebuilt = selected(rebuilt_metrics, "rebuilt metrics")
    stored = selected(stored_metrics, "stored metrics")
    if set(rebuilt.index) != set(stored.index):
        missing = sorted(set(stored.index) - set(rebuilt.index))
        extra = sorted(set(rebuilt.index) - set(stored.index))
        raise RuntimeError(
            f"{candidate}: OOF case IDs differ: missing={missing[:5]} extra={extra[:5]}"
        )
    rebuilt = rebuilt.loc[stored.index]
    scalar_columns = [
        column
        for column in stored.columns
        if column != "candidate" and pd.api.types.is_numeric_dtype(stored[column])
    ]
    missing_columns = sorted(set(scalar_columns).difference(rebuilt.columns))
    if missing_columns:
        raise RuntimeError(
            f"{candidate}: rebuilt metrics lack stored scalars {missing_columns}"
        )
    for column in scalar_columns:
        expected = stored[column].to_numpy(dtype=np.float64)
        actual = rebuilt[column].to_numpy(dtype=np.float64)
        if not np.isfinite(expected).all() or not np.isfinite(actual).all():
            raise RuntimeError(f"{candidate}: non-finite scalar metric {column}")
        mismatched = ~np.isclose(actual, expected, rtol=0.0, atol=tolerance)
        if mismatched.any():
            case_id = str(stored.index[np.flatnonzero(mismatched)[0]])
            raise RuntimeError(
                f"{candidate}: metric mismatch for {case_id} {column}: "
                f"rebuilt={actual[mismatched][0]!r} stored={expected[mismatched][0]!r}"
            )


def load_lcv2_fold_bundle(lcv2_root: Path, fold: int) -> Mapping[str, Any]:
    """Load only the LCv2 component model that excluded the requested OOF fold."""
    fold_value = int(fold)
    if fold_value not in range(5):
        raise ValueError(f"invalid LCv2 fold: {fold}")
    path = (
        Path(lcv2_root)
        / "models"
        / "lightgbm"
        / f"fold_{fold_value}"
        / "models.joblib"
    )
    if not path.is_file():
        raise FileNotFoundError(f"missing LCv2 held-out-fold model: {path}")
    bundle = joblib.load(path)
    if not isinstance(bundle, Mapping):
        raise TypeError(f"LCv2 fold bundle must be a mapping: {path}")
    if "component_model" not in bundle or "component_features" not in bundle:
        raise KeyError(f"incomplete LCv2 fold bundle: {path}")
    return bundle


def score_lcv2_fold_components(
    lcv2_root: Path,
    fold: int,
    component_features: pd.DataFrame,
) -> np.ndarray:
    """Score OOF proposals with exactly the LCv2 model excluding their fold."""
    bundle = load_lcv2_fold_bundle(lcv2_root, fold)
    feature_names = [str(value) for value in bundle["component_features"]]
    if len(feature_names) != len(set(feature_names)):
        raise ValueError("LCv2 fold bundle has duplicate feature names")
    missing = sorted(set(feature_names).difference(component_features.columns))
    if missing:
        raise KeyError(f"OOF component features are missing {missing}")
    values = component_features.loc[:, feature_names].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("OOF component features must be finite")
    probability = np.asarray(
        bundle["component_model"].predict_proba(
            component_features.loc[:, feature_names]
        ),
        dtype=np.float64,
    )
    if (
        probability.shape != (len(component_features), 2)
        or not np.isfinite(probability).all()
    ):
        raise RuntimeError("LCv2 fold model returned invalid probabilities")
    return probability[:, 1]


def rebuild_lcv2_candidate(
    probabilities_by_model: Mapping[str, np.ndarray],
    proposals: Mapping[str, list[Mapping[str, Any]]],
    proposal_scores: Mapping[tuple[str, int], float],
    candidate: str,
    cutoffs: Mapping[str, float],
    spacing_zyx: tuple[float, float, float],
    v2_final: Mapping[str, float | int],
    *,
    lcv2_source_root: str | Path,
    return_audit: bool = False,
) -> dict[str, np.ndarray] | tuple[dict[str, np.ndarray], dict[str, int]]:
    """Delegate candidate regeneration to the original frozen V2 reconstruction."""
    allowed = {
        ROUTER_PAIRS["N09_Router_XF12_vs_LCprotected_case_safe"][1],
        ROUTER_PAIRS["N10_Router_XF16_vs_LCXLFT_case_safe"][1],
    }
    if candidate not in allowed:
        raise ValueError(f"candidate is not registered for a router: {candidate}")
    v2_root = Path(lcv2_source_root)
    if str(v2_root) not in sys.path:
        sys.path.insert(0, str(v2_root))
    v2_component_path = v2_root / "v2_component_gate.py"
    if not v2_component_path.is_file():
        raise FileNotFoundError(
            f"missing canonical V2 component source: {v2_component_path}"
        )
    spec = importlib.util.spec_from_file_location(
        "_lcv3_v2_component_gate", v2_component_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load V2 component source: {v2_component_path}")
    v2_component_gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v2_component_gate)
    from component_gate import label_to_regions, regions_to_label

    regions, score_volumes, protected_count = (
        v2_component_gate.reconstruct_scored_candidate(
        probabilities_by_model,
        proposals,
        proposal_scores,
        candidate,
        )
    )
    regions = v2_component_gate.filter_scored_components(
        regions, score_volumes, cutoffs
    )
    maps = v2_component_gate.candidate_score_maps(
        probabilities_by_model, candidate
    )
    regions = v2_component_gate.apply_gate_preserving_v2(
        regions, maps, spacing_zyx, v2_final
    )
    canonical = label_to_regions(regions_to_label(regions))
    if return_audit:
        return canonical, {"protected_proposal_count": int(protected_count)}
    return canonical


def validate_test_candidate_set(
    candidate_root: Path,
    reference_root: Path,
    expected_count: int = 179,
) -> list[str]:
    """Validate the exact test set and NIfTI geometry of an existing candidate."""
    candidate_paths = {
        path.name[: -len(".nii.gz")]: path
        for path in Path(candidate_root).glob("*.nii.gz")
    }
    reference_paths = {
        path.name[: -len(".nii.gz")]: path
        for path in Path(reference_root).glob("*.nii.gz")
    }
    if len(candidate_paths) != expected_count or set(candidate_paths) != set(reference_paths):
        missing = sorted(set(reference_paths) - set(candidate_paths))
        extra = sorted(set(candidate_paths) - set(reference_paths))
        raise RuntimeError(
            f"test candidate universe differs: found={len(candidate_paths)} "
            f"expected={expected_count} missing={missing[:5]} extra={extra[:5]}"
        )
    for case_id in sorted(reference_paths):
        candidate = nib.load(candidate_paths[case_id])
        reference = nib.load(reference_paths[case_id])
        if (
            candidate.shape != reference.shape
            or not np.array_equal(candidate.affine, reference.affine)
            or tuple(candidate.header.get_zooms()[:3])
            != tuple(reference.header.get_zooms()[:3])
        ):
            raise RuntimeError(f"{case_id}: candidate geometry mismatch")
    return sorted(candidate_paths)


def _router_artifact_path(
    configured: Mapping[str, Any] | None,
    router_id: str,
    candidate: str,
    fallback: Path,
) -> Path:
    if configured is None:
        return fallback
    if router_id in configured:
        return Path(configured[router_id])
    if candidate in configured:
        return Path(configured[candidate])
    raise KeyError(
        f"router artifact mapping lacks {router_id} and candidate {candidate}"
    )


def _validate_lcv2_bundle(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing LCv2 model bundle: {path}")
    bundle = joblib.load(path)
    if not isinstance(bundle, Mapping):
        raise TypeError(f"LCv2 model bundle must be a mapping: {path}")
    if "component_model" not in bundle or "component_features" not in bundle:
        raise KeyError(f"incomplete LCv2 model bundle: {path}")
    features = [str(value) for value in bundle["component_features"]]
    if not features or len(features) != len(set(features)):
        raise ValueError(f"invalid LCv2 component feature order: {path}")


def validate_router_artifacts(
    config: Mapping[str, Any],
    *,
    mode: str = "full",
    router_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Preflight existing LCv2 inputs or strictly validate rebuilt OOF metrics."""
    if mode not in {"preflight", "full"}:
        raise ValueError(f"unsupported router artifact validation mode: {mode}")
    required = {"lcv2_root", "test_roots"}
    if missing := required.difference(config):
        raise KeyError(f"router artifact config is missing {sorted(missing)}")
    if not isinstance(config["test_roots"], Mapping) or "XL" not in config["test_roots"]:
        raise KeyError("router artifact config requires test_roots.XL")

    lcv2_root = Path(config["lcv2_root"])
    output_root = Path(config.get("output_root", "."))
    stored_metrics_path = Path(
        config.get(
            "lcv2_oof_metrics",
            lcv2_root / "metrics" / "oof_case_metrics.csv",
        )
    )
    if not stored_metrics_path.is_file():
        raise FileNotFoundError(f"missing stored LCv2 OOF metrics: {stored_metrics_path}")
    stored_metrics = pd.read_csv(stored_metrics_path)
    reference_root = Path(
        config.get("router_reference_root", config["test_roots"]["XL"])
    )
    rebuilt_mapping = config.get("router_oof_metrics")
    candidate_mapping = config.get("router_candidate_roots")
    if rebuilt_mapping is not None and not isinstance(rebuilt_mapping, Mapping):
        raise TypeError("router_oof_metrics must be a mapping")
    if candidate_mapping is not None and not isinstance(candidate_mapping, Mapping):
        raise TypeError("router_candidate_roots must be a mapping")

    expected_oof = int(config.get("expected_oof_cases", 1295))
    expected_test = int(config.get("expected_test_cases", 179))
    bundle_paths = [
        lcv2_root / "models" / "lightgbm" / f"fold_{fold}" / "models.joblib"
        for fold in range(5)
    ]
    bundle_paths.append(
        lcv2_root / "models" / "lightgbm" / "final" / "models.joblib"
    )
    for bundle_path in bundle_paths:
        _validate_lcv2_bundle(bundle_path)
    selected_router_ids = (
        tuple(ROUTER_PAIRS)
        if mode == "preflight" or router_ids is None
        else tuple(str(value) for value in router_ids)
    )
    unknown = sorted(set(selected_router_ids).difference(ROUTER_PAIRS))
    if unknown:
        raise KeyError(f"unknown router IDs for validation: {unknown}")
    audit: dict[str, Any] = {
        "router_count": len(selected_router_ids),
        "stored_oof_metrics": str(stored_metrics_path),
        "test_reference_root": str(reference_root),
        "expected_oof_cases": expected_oof,
        "expected_test_cases": expected_test,
        "mode": mode,
        "bundle_count": len(bundle_paths),
        "model_bundles": [str(path) for path in bundle_paths],
        "rebuilt_oof_status": "pending" if mode == "preflight" else "validated",
        "routers": {},
    }
    for router_id in selected_router_ids:
        _anchor, candidate = ROUTER_PAIRS[router_id]
        stored_candidate = stored_metrics.loc[
            stored_metrics["candidate"].astype(str).eq(candidate)
        ].copy()
        stored_cases = _case_index(
            stored_candidate, f"{router_id} stored LCv2 metrics"
        )
        if len(stored_cases) != expected_oof:
            raise RuntimeError(
                f"{router_id}: stored LCv2 OOF count {len(stored_cases)} "
                f"!= expected {expected_oof}"
            )
        rebuilt_path = _router_artifact_path(
            rebuilt_mapping,
            router_id,
            candidate,
            output_root / "router_metrics" / f"{router_id}.csv",
        )
        rebuilt_case_count: int | None = None
        if mode == "full":
            if not rebuilt_path.is_file():
                raise FileNotFoundError(
                    f"{router_id}: missing regenerated OOF metrics: {rebuilt_path}"
                )
            rebuilt_metrics = pd.read_csv(rebuilt_path)
            assert_existing_oof_metrics_match(
                rebuilt_metrics,
                stored_metrics,
                candidate,
                tolerance=1e-12,
            )
            rebuilt_cases = _case_index(
                rebuilt_metrics, f"{router_id} rebuilt metrics"
            )
            rebuilt_case_count = len(rebuilt_cases)
            if rebuilt_case_count != expected_oof:
                raise RuntimeError(
                    f"{router_id}: regenerated OOF count {rebuilt_case_count} "
                    f"!= expected {expected_oof}"
                )

        candidate_root = _router_artifact_path(
            candidate_mapping,
            router_id,
            candidate,
            lcv2_root / "test_predictions" / candidate,
        )
        test_case_ids = validate_test_candidate_set(
            candidate_root,
            reference_root,
            expected_count=expected_test,
        )
        audit["routers"][router_id] = {
            "anchor": ROUTER_PAIRS[router_id][0],
            "candidate": candidate,
            "rebuilt_oof_metrics": str(rebuilt_path),
            "stored_oof_case_count": len(stored_cases),
            "oof_case_count": rebuilt_case_count,
            "rebuilt_oof_status": (
                "pending" if mode == "preflight" else "validated"
            ),
            "test_candidate_root": str(candidate_root),
            "test_case_count": len(test_case_ids),
            "oof_scalar_match": None if mode == "preflight" else True,
            "test_geometry_match": True,
        }
    return audit
