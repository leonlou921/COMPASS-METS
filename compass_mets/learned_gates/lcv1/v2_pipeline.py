"""OOF calibration, evaluation, test inference, and packaging for gate v2."""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Iterable

import joblib
import nibabel as nib
import numpy as np
import pandas as pd

from compass_mets.learned_gates.lcv1.component_gate import (
    REGION_NAMES,
    label_to_regions,
    load_probabilities,
    propose_components,
    regions_to_label,
)
from compass_mets.learned_gates.lcv1.features import (
    extract_case_features,
    fixed_component_conf_masks,
)
from compass_mets.learned_gates.lcv1.reconstruct_and_evaluate import (
    _case_in_shard,
    apply_v2_final,
    evaluate_case_regions,
    summarize_case_metrics,
)
from compass_mets.learned_gates.lcv1.train_models import (
    select_fp_constrained_cutoff,
)
from compass_mets.learned_gates.lcv1.train_models_v2 import (
    predict_test_component_probability,
)
from compass_mets.learned_gates.lcv1.v2_component_gate import (
    CANDIDATES,
    LC_V2_STRUCTURED_FILTER,
    LC_V2_STRUCTURED_PROTECTED_RESCUE,
    LC_V2_XLFT_FILTER,
    apply_gate_preserving_v2,
    candidate_score_maps,
    filter_scored_components,
    reconstruct_scored_candidate,
    scored_final_component_rows,
)


ANCHOR_STRUCTURED = "anchor_structured_V2"
ANCHOR_XLFT = "anchor_XLFT_V2"


def aligned_anchor(candidate: str) -> str:
    if candidate in {
        LC_V2_STRUCTURED_FILTER,
        LC_V2_STRUCTURED_PROTECTED_RESCUE,
    }:
        return ANCHOR_STRUCTURED
    if candidate == LC_V2_XLFT_FILTER:
        return ANCHOR_XLFT
    raise ValueError(f"unknown v2 candidate: {candidate}")


def _hierarchy_safe_cutoffs(cutoffs: dict[str, float]) -> dict[str, float]:
    result = {region: float(value) for region, value in cutoffs.items()}
    result["TC"] = min(result["TC"], result["ET"])
    result["WT"] = min(result["WT"], result["TC"])
    return result


def _fit_final_component_cutoffs(
    candidate_rows: pd.DataFrame,
    anchor_rows: pd.DataFrame,
) -> dict[str, float]:
    cutoffs: dict[str, float] = {}
    for region in ("WT", "TC", "ET", "RC"):
        candidate_region = candidate_rows["region"].eq(region)
        anchor_region = anchor_rows["region"].eq(region)
        allowed_fp = int(
            np.logical_and(
                anchor_region.to_numpy(),
                anchor_rows["target"].to_numpy(dtype=np.int8) == 0,
            ).sum()
        )
        selected = select_fp_constrained_cutoff(
            candidate_rows.loc[candidate_region, "score"].to_numpy(dtype=float),
            candidate_rows.loc[candidate_region, "target"].to_numpy(dtype=np.int8),
            allowed_fp=allowed_fp,
        )
        cutoffs[region] = float(selected["threshold"])
    return _hierarchy_safe_cutoffs(cutoffs)


def calibrate_nested_cutoffs(
    candidate_rows: pd.DataFrame,
    anchor_rows: pd.DataFrame,
) -> dict:
    """Fit each held-out cutoff on the other folds, plus one all-OOF test cutoff."""
    required_candidate = {"fold", "region", "score", "target"}
    required_anchor = {"fold", "region", "target"}
    if missing := required_candidate.difference(candidate_rows.columns):
        raise KeyError(f"missing candidate calibration columns: {sorted(missing)}")
    if missing := required_anchor.difference(anchor_rows.columns):
        raise KeyError(f"missing anchor calibration columns: {sorted(missing)}")
    folds = sorted(int(value) for value in candidate_rows["fold"].unique())
    if folds != sorted(int(value) for value in anchor_rows["fold"].unique()):
        raise RuntimeError("candidate and anchor folds do not match")
    result = {"folds": {}}
    for heldout in folds:
        candidate_training = candidate_rows["fold"].astype(int) != heldout
        anchor_training = anchor_rows["fold"].astype(int) != heldout
        result["folds"][str(heldout)] = {
            "training_folds": [fold for fold in folds if fold != heldout],
            "cutoffs": _fit_final_component_cutoffs(
                candidate_rows.loc[candidate_training],
                anchor_rows.loc[anchor_training],
            ),
        }
    result["final"] = {
        "training_folds": folds,
        "cutoffs": _fit_final_component_cutoffs(candidate_rows, anchor_rows),
    }
    return result


def candidate_submission_verdict(
    anchor: Mapping,
    candidate: Mapping,
    contract: Mapping,
) -> dict:
    reasons: list[str] = []
    allowed_fp = float(anchor["false_positive_components"]) * (
        1.0 + float(contract["max_fp_relative_increase"])
    )
    if float(candidate["false_positive_components"]) > allowed_fp:
        reasons.append("false_positive_components")
    if float(candidate["macro_hd95"]) > float(anchor["macro_hd95"]) + float(
        contract["max_macro_hd95_increase"]
    ):
        reasons.append("macro_hd95")
    if float(candidate["worst_case_macro_dsc_delta"]) < float(
        contract["min_worst_case_macro_dsc_delta"]
    ):
        reasons.append("worst_case_macro_dsc_delta")
    if not (
        float(candidate["small_f1"]) > float(anchor["small_f1"])
        or int(candidate["true_positive_components"])
        > int(anchor["true_positive_components"])
    ):
        reasons.append("no_detection_improvement")
    return {
        "candidate": candidate["candidate"],
        "anchor": anchor["candidate"],
        "worth_submitting": not reasons,
        "reasons": reasons,
    }


def proposal_score_map(component_rows: pd.DataFrame) -> dict[tuple[str, int], float]:
    """Map deterministic v1 proposal IDs to leakage-safe v2 probabilities."""
    if component_rows.empty:
        return {}
    required = {"region", "component_id", "v2_component_probability"}
    if missing := required.difference(component_rows.columns):
        raise KeyError(f"missing proposal score columns: {sorted(missing)}")
    duplicated = component_rows.duplicated(["region", "component_id"], keep=False)
    if duplicated.any():
        keys = component_rows.loc[
            duplicated, ["region", "component_id"]
        ].head().to_dict("records")
        raise RuntimeError(f"duplicate proposal score IDs: {keys}")
    return {
        (str(row.region), int(row.component_id)): float(
            row.v2_component_probability
        )
        for row in component_rows.itertuples()
    }


def proposal_scores_from_predictions(
    component_frame: pd.DataFrame,
    component_probability: np.ndarray,
) -> dict[tuple[str, int], float]:
    probability = np.asarray(component_probability, dtype=float)
    if len(component_frame) != len(probability):
        raise RuntimeError(
            f"component probability length differs: "
            f"rows={len(component_frame)} probabilities={len(probability)}"
        )
    if component_frame.empty:
        return {}
    required = {"region", "component_id"}
    if missing := required.difference(component_frame.columns):
        raise KeyError(f"component features lack ID columns: {sorted(missing)}")
    scored = component_frame[["region", "component_id"]].copy()
    scored["v2_component_probability"] = probability
    return proposal_score_map(scored)


def _release_native_memory() -> None:
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (AttributeError, OSError):
        pass


def _mask_bbox(mask: np.ndarray) -> tuple[slice, slice, slice] | None:
    boolean = np.asarray(mask, dtype=bool)
    if boolean.ndim != 3:
        raise ValueError(f"expected a 3D support mask, got {boolean.shape}")
    bounds = []
    for axis in range(3):
        other_axes = tuple(index for index in range(3) if index != axis)
        occupied = np.flatnonzero(boolean.any(axis=other_axes))
        if not occupied.size:
            return None
        bounds.append(slice(int(occupied[0]), int(occupied[-1]) + 1))
    return tuple(bounds)


def _merge_bbox(
    left: tuple[slice, slice, slice] | None,
    right: tuple[slice, slice, slice] | None,
) -> tuple[slice, slice, slice] | None:
    if left is None:
        return right
    if right is None:
        return left
    return tuple(
        slice(min(a.start, b.start), max(a.stop, b.stop))
        for a, b in zip(left, right)
    )


def bbox_from_component_frame(
    component_frame: pd.DataFrame,
) -> tuple[slice, slice, slice] | None:
    if component_frame.empty:
        return None
    columns = [
        "bbox_z0",
        "bbox_z1",
        "bbox_y0",
        "bbox_y1",
        "bbox_x0",
        "bbox_x1",
    ]
    if missing := set(columns).difference(component_frame.columns):
        raise KeyError(f"component features lack bbox columns: {sorted(missing)}")
    return tuple(
        slice(
            int(component_frame[f"bbox_{axis}0"].min()),
            int(component_frame[f"bbox_{axis}1"].max()),
        )
        for axis in ("z", "y", "x")
    )


def load_cropped_probabilities(
    paths_by_model: Mapping[str, Path],
    threshold: float,
    include_mask: np.ndarray | None = None,
    known_support_bbox: tuple[slice, slice, slice] | None = None,
    support_bbox_is_complete: bool = False,
    known_full_shape: tuple[int, int, int] | None = None,
) -> tuple[
    dict[str, np.ndarray],
    tuple[slice, slice, slice],
    tuple[int, int, int],
]:
    """Load full NPZs sequentially, then retain only proposal/GT support."""
    if set(paths_by_model) != {"XL", "M", "FT"}:
        raise KeyError("cropped loading requires XL, M, and FT paths")
    bbox = _merge_bbox(
        _mask_bbox(include_mask) if include_mask is not None else None,
        known_support_bbox,
    )
    full_shape = (
        tuple(int(value) for value in known_full_shape)
        if known_full_shape is not None
        else (
            tuple(int(value) for value in include_mask.shape)
            if include_mask is not None
            else None
        )
    )
    if not support_bbox_is_complete:
        for name in ("XL", "M", "FT"):
            probabilities = load_probabilities(paths_by_model[name])
            shape = tuple(int(value) for value in probabilities.shape[1:])
            if full_shape is None:
                full_shape = shape
            elif shape != full_shape:
                raise RuntimeError(
                    f"probability shapes differ: expected={full_shape} {name}={shape}"
                )
            support = np.any(probabilities >= float(threshold), axis=0)
            bbox = _merge_bbox(bbox, _mask_bbox(support))
            del probabilities, support
            _release_native_memory()
    if full_shape is None:
        raise ValueError(
            "known_full_shape is required when probability support scanning is skipped"
        )
    if include_mask is not None and tuple(include_mask.shape) != full_shape:
        raise ValueError(
            f"include mask shape {include_mask.shape} differs from {full_shape}"
        )
    if bbox is None:
        bbox = (slice(0, 1), slice(0, 1), slice(0, 1))
    channel_bbox = (slice(None), *bbox)
    cropped = {}
    for name in ("XL", "M", "FT"):
        probabilities = load_probabilities(paths_by_model[name])
        if tuple(probabilities.shape[1:]) != full_shape:
            raise RuntimeError(
                f"probability shapes differ: expected={full_shape} "
                f"{name}={probabilities.shape[1:]}"
            )
        cropped[name] = np.asarray(
            probabilities[channel_bbox], dtype=np.float32
        ).copy()
        del probabilities
        _release_native_memory()
    return cropped, bbox, full_shape


def restore_cropped_label(
    cropped_label: np.ndarray,
    bbox: tuple[slice, slice, slice],
    full_shape: tuple[int, int, int],
) -> np.ndarray:
    restored = np.zeros(full_shape, dtype=np.asarray(cropped_label).dtype)
    if restored[bbox].shape != np.asarray(cropped_label).shape:
        raise ValueError(
            f"crop shape {np.asarray(cropped_label).shape} "
            f"differs from bbox shape {restored[bbox].shape}"
        )
    restored[bbox] = cropped_label
    return restored


def _oof_probability_path(root: Path, fold: int, case_id: str) -> Path:
    return root / f"fold_{fold}" / "validation" / f"{case_id}.npz"


def _case_folds(config: Mapping) -> dict[str, int]:
    case_predictions = pd.read_parquet(
        Path(config["v1_output_root"])
        / "oof_predictions"
        / "case_predictions.parquet",
        columns=["case_id", "fold"],
    )
    counts = case_predictions.groupby("case_id", sort=False)["fold"].nunique()
    if (counts != 1).any():
        raise RuntimeError(
            f"invalid case fold mappings: {counts[counts != 1].index.tolist()[:5]}"
        )
    return {
        str(case_id): int(group["fold"].iloc[0])
        for case_id, group in case_predictions.groupby("case_id", sort=False)
    }


def _load_oof_case(
    config: Mapping, case_id: str, fold: int
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], tuple[float, float, float]]:
    roots = {
        name: Path(config["oof_roots"][name]) for name in ("XL", "M", "FT")
    }
    paths = {
        name: _oof_probability_path(root, fold, case_id)
        for name, root in roots.items()
    }
    gt_image = nib.load(Path(config["labels_root"]) / f"{case_id}.nii.gz")
    gt_label = np.asarray(gt_image.dataobj).transpose(2, 1, 0)
    component_path = (
        Path(config["v1_output_root"])
        / "features"
        / "per_case"
        / f"{case_id}.components.parquet"
    )
    if not component_path.is_file():
        raise RuntimeError(f"{case_id}: v1 component features are missing")
    component_bbox = bbox_from_component_frame(pd.read_parquet(component_path))
    arrays, bbox, full_shape = load_cropped_probabilities(
        paths,
        threshold=float(config["proposal_threshold"]),
        include_mask=gt_label != 0,
        known_support_bbox=component_bbox,
        support_bbox_is_complete=True,
        known_full_shape=tuple(int(value) for value in gt_label.shape),
    )
    if tuple(gt_label.shape) != full_shape:
        raise RuntimeError(
            f"{case_id}: GT shape {gt_label.shape} differs from {full_shape}"
        )
    gt_regions = label_to_regions(gt_label[bbox])
    spacing = tuple(float(value) for value in gt_image.header.get_zooms()[:3][::-1])
    return arrays, gt_regions, spacing


def build_anchor_regions(
    arrays: Mapping[str, np.ndarray],
    candidate: str,
    spacing_zyx: tuple[float, float, float],
    config: Mapping,
) -> dict[str, np.ndarray]:
    """Build the exact existing V2 anchor aligned to one new candidate."""
    basis_candidate = (
        LC_V2_STRUCTURED_FILTER
        if aligned_anchor(candidate) == ANCHOR_STRUCTURED
        else LC_V2_XLFT_FILTER
    )
    maps = candidate_score_maps(arrays, basis_candidate)
    regions = fixed_component_conf_masks(
        maps["basis"],
        spacing_zyx=spacing_zyx,
        settings=config["fixed_component_conf"],
    )
    regions = apply_v2_final(
        regions,
        dict(arrays),
        spacing_zyx,
        config["v2_final"],
        fused_scores={"TC": maps["tc_v2"], "RC": maps["rc_v2"]},
    )
    return label_to_regions(regions_to_label(regions))


def _expected_proposal_keys(proposals: Mapping[str, list[Mapping]]) -> set[tuple[str, int]]:
    return {
        (region, int(component["component_id"]))
        for region in REGION_NAMES
        for component in proposals.get(region, [])
    }


CALIBRATION_COLUMNS = [
    "case_id",
    "fold",
    "candidate",
    "region",
    "component_id",
    "score",
    "target",
    "overlap_voxels",
    "volume_voxels",
]


def build_calibration_cases(
    config: Mapping,
    max_cases: int | None = None,
    shard_count: int = 1,
    shard_index: int = 0,
    case_ids: set[str] | None = None,
) -> dict[str, int]:
    """First OOF pass: checkpoint final-component score/target rows per case."""
    output_root = Path(config["output_root"])
    prediction_table = pd.read_parquet(
        output_root / "oof_predictions" / "component_predictions.parquet"
    )
    groups = {
        str(case_id): group
        for case_id, group in prediction_table.groupby("case_id", sort=False)
    }
    empty = prediction_table.iloc[0:0].copy()
    case_folds = _case_folds(config)
    checkpoint_root = output_root / "calibration" / "per_case"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    processed = 0
    for case_id, fold in sorted(case_folds.items()):
        if case_ids is not None and case_id not in case_ids:
            continue
        if not _case_in_shard(case_id, shard_count, shard_index):
            continue
        checkpoint = checkpoint_root / f"{case_id}.parquet"
        if checkpoint.is_file():
            continue
        arrays, gt_regions, spacing = _load_oof_case(config, case_id, fold)
        proposals = propose_components(
            arrays, threshold=float(config["proposal_threshold"])
        )
        component_group = groups.get(case_id, empty)
        proposal_scores = proposal_score_map(component_group)
        expected_keys = _expected_proposal_keys(proposals)
        if set(proposal_scores) != expected_keys:
            raise RuntimeError(
                f"{case_id}: proposal IDs differ from v2 predictions: "
                f"missing={sorted(expected_keys - set(proposal_scores))[:5]} "
                f"extra={sorted(set(proposal_scores) - expected_keys)[:5]}"
            )
        rows: list[dict] = []
        for candidate in CANDIDATES:
            regions, scores, _ = reconstruct_scored_candidate(
                arrays, proposals, proposal_scores, candidate
            )
            rows.extend(
                scored_final_component_rows(
                    regions,
                    scores,
                    gt_regions,
                    case_id,
                    fold,
                    candidate,
                )
            )
        for anchor, candidate in (
            (ANCHOR_STRUCTURED, LC_V2_STRUCTURED_FILTER),
            (ANCHOR_XLFT, LC_V2_XLFT_FILTER),
        ):
            regions = build_anchor_regions(arrays, candidate, spacing, config)
            anchor_scores = {
                region: np.asarray(regions[region], dtype=np.float32)
                for region in REGION_NAMES
            }
            rows.extend(
                scored_final_component_rows(
                    regions,
                    anchor_scores,
                    gt_regions,
                    case_id,
                    fold,
                    anchor,
                )
            )
        frame = pd.DataFrame(rows, columns=CALIBRATION_COLUMNS)
        temporary = checkpoint.with_name(checkpoint.name + f".tmp.{os.getpid()}")
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, checkpoint)
        processed += 1
        print(
            json.dumps(
                {
                    "event": "v2_calibration_case_complete",
                    "case_id": case_id,
                    "fold": fold,
                    "processed": processed,
                    "total": len(case_folds),
                }
            ),
            flush=True,
        )
        del arrays, gt_regions, proposals, rows, frame
        _release_native_memory()
        if max_cases is not None and processed >= max_cases:
            break
    return {
        "processed": processed,
        "completed": len(list(checkpoint_root.glob("*.parquet"))),
        "expected": len(case_folds),
    }


def consolidate_calibration(config: Mapping) -> dict:
    output_root = Path(config["output_root"])
    case_folds = _case_folds(config)
    checkpoint_root = output_root / "calibration" / "per_case"
    paths = sorted(checkpoint_root.glob("*.parquet"))
    if {path.stem for path in paths} != set(case_folds):
        missing = sorted(set(case_folds) - {path.stem for path in paths})
        raise RuntimeError(
            f"calibration universe incomplete: found={len(paths)} "
            f"expected={len(case_folds)} missing={missing[:5]}"
        )
    frames = [pd.read_parquet(path) for path in paths]
    rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=CALIBRATION_COLUMNS
    )
    calibration_root = output_root / "calibration"
    rows.to_parquet(calibration_root / "final_component_rows.parquet", index=False)
    report = {"candidates": {}, "case_count": len(case_folds)}
    for candidate in CANDIDATES:
        candidate_rows = rows[rows["candidate"].eq(candidate)]
        anchor = aligned_anchor(candidate)
        anchor_rows = rows[rows["candidate"].eq(anchor)]
        report["candidates"][candidate] = {
            "anchor": anchor,
            **calibrate_nested_cutoffs(candidate_rows, anchor_rows),
        }
    (calibration_root / "cutoffs.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def _canonical_regions(regions: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return label_to_regions(regions_to_label(regions))


def evaluate_oof_cases(
    config: Mapping,
    max_cases: int | None = None,
    shard_count: int = 1,
    shard_index: int = 0,
    case_ids: set[str] | None = None,
) -> dict[str, int]:
    output_root = Path(config["output_root"])
    prediction_table = pd.read_parquet(
        output_root / "oof_predictions" / "component_predictions.parquet"
    )
    groups = {
        str(case_id): group
        for case_id, group in prediction_table.groupby("case_id", sort=False)
    }
    empty = prediction_table.iloc[0:0].copy()
    cutoffs = json.loads(
        (output_root / "calibration" / "cutoffs.json").read_text(encoding="utf-8")
    )
    case_folds = _case_folds(config)
    checkpoint_root = output_root / "metrics" / "per_case"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    processed = 0
    for case_id, fold in sorted(case_folds.items()):
        if case_ids is not None and case_id not in case_ids:
            continue
        if not _case_in_shard(case_id, shard_count, shard_index):
            continue
        checkpoint = checkpoint_root / f"{case_id}.json"
        if checkpoint.is_file():
            continue
        arrays, gt_regions, spacing = _load_oof_case(config, case_id, fold)
        proposals = propose_components(
            arrays, threshold=float(config["proposal_threshold"])
        )
        proposal_scores = proposal_score_map(groups.get(case_id, empty))
        rows: list[dict] = []
        anchor_metrics: dict[str, dict] = {}
        for anchor, candidate in (
            (ANCHOR_STRUCTURED, LC_V2_STRUCTURED_FILTER),
            (ANCHOR_XLFT, LC_V2_XLFT_FILTER),
        ):
            regions = build_anchor_regions(arrays, candidate, spacing, config)
            row = evaluate_case_regions(
                anchor,
                regions,
                gt_regions,
                spacing,
                float(config["small_lesion_max_volume_mm3"]),
            )
            row["macro_dsc_delta"] = 0.0
            anchor_metrics[anchor] = row
            rows.append(row)
        for candidate in CANDIDATES:
            regions, score_volumes, protected_count = reconstruct_scored_candidate(
                arrays, proposals, proposal_scores, candidate
            )
            fold_cutoffs = cutoffs["candidates"][candidate]["folds"][str(fold)][
                "cutoffs"
            ]
            regions = filter_scored_components(
                regions, score_volumes, fold_cutoffs
            )
            maps = candidate_score_maps(arrays, candidate)
            regions = apply_gate_preserving_v2(
                regions, maps, spacing, config["v2_final"]
            )
            regions = _canonical_regions(regions)
            row = evaluate_case_regions(
                candidate,
                regions,
                gt_regions,
                spacing,
                float(config["small_lesion_max_volume_mm3"]),
            )
            anchor = aligned_anchor(candidate)
            row["macro_dsc_delta"] = float(row["macro_dsc"]) - float(
                anchor_metrics[anchor]["macro_dsc"]
            )
            row["protected_proposal_count"] = int(protected_count)
            rows.append(row)
        for row in rows:
            row["case_id"] = case_id
            row["fold"] = int(fold)
        temporary = checkpoint.with_name(checkpoint.name + f".tmp.{os.getpid()}")
        temporary.write_text(json.dumps(rows, sort_keys=True), encoding="utf-8")
        os.replace(temporary, checkpoint)
        processed += 1
        print(
            json.dumps(
                {
                    "event": "v2_oof_case_complete",
                    "case_id": case_id,
                    "fold": fold,
                    "processed": processed,
                    "total": len(case_folds),
                }
            ),
            flush=True,
        )
        del arrays, gt_regions, proposals, rows
        _release_native_memory()
        if max_cases is not None and processed >= max_cases:
            break
    return {
        "processed": processed,
        "completed": len(list(checkpoint_root.glob("*.json"))),
        "expected": len(case_folds),
    }


def consolidate_evaluation(config: Mapping) -> dict:
    output_root = Path(config["output_root"])
    case_folds = _case_folds(config)
    checkpoint_root = output_root / "metrics" / "per_case"
    paths = sorted(checkpoint_root.glob("*.json"))
    if {path.stem for path in paths} != set(case_folds):
        missing = sorted(set(case_folds) - {path.stem for path in paths})
        raise RuntimeError(
            f"evaluation universe incomplete: found={len(paths)} "
            f"expected={len(case_folds)} missing={missing[:5]}"
        )
    all_rows: list[dict] = []
    for path in paths:
        all_rows.extend(json.loads(path.read_text(encoding="utf-8")))
    metrics = pd.DataFrame(all_rows)
    metrics_root = output_root / "metrics"
    metrics.to_parquet(metrics_root / "oof_case_metrics.parquet", index=False)
    metrics.to_csv(metrics_root / "oof_case_metrics.csv", index=False)
    summaries = summarize_case_metrics(metrics)
    by_candidate = {row["candidate"]: row for row in summaries}
    verdicts = {}
    for candidate in CANDIDATES:
        anchor = aligned_anchor(candidate)
        verdicts[candidate] = candidate_submission_verdict(
            by_candidate[anchor],
            by_candidate[candidate],
            config["safety_contract"],
        )
    report = {
        "case_count": int(metrics["case_id"].nunique()),
        "summaries": summaries,
        "verdicts": verdicts,
    }
    (metrics_root / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_root / "submission_verdicts.json").write_text(
        json.dumps(verdicts, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def _test_case_ids(test_roots: Mapping[str, Path]) -> list[str]:
    sets = {
        name: {path.stem for path in root.glob("*.npz")}
        for name, root in test_roots.items()
    }
    if not sets or any(values != sets["XL"] for values in sets.values()):
        raise RuntimeError("XL/M/FT test case sets are not identical")
    cases = sorted(sets["XL"])
    if len(cases) != 179:
        raise RuntimeError(f"expected 179 test cases, found {len(cases)}")
    return cases


def infer_test_cases(
    config: Mapping,
    max_cases: int | None = None,
    shard_count: int = 1,
    shard_index: int = 0,
    case_ids: set[str] | None = None,
) -> dict[str, int]:
    output_root = Path(config["output_root"])
    v1_root = Path(config["v1_output_root"])
    test_roots = {
        name: Path(config["test_roots"][name]) for name in ("XL", "M", "FT")
    }
    cases = _test_case_ids(test_roots)
    v2_bundle = joblib.load(
        output_root / "models" / "lightgbm" / "final" / "models.joblib"
    )
    v1_case_bundle = joblib.load(
        v1_root / "models" / "lightgbm" / "final" / "models.joblib"
    )
    cutoffs = json.loads(
        (output_root / "calibration" / "cutoffs.json").read_text(encoding="utf-8")
    )
    prediction_roots = {
        candidate: output_root / "test_predictions" / candidate
        for candidate in CANDIDATES
    }
    for root in prediction_roots.values():
        root.mkdir(parents=True, exist_ok=True)
    feature_root = v1_root / "test_features" / "per_case"
    processed = 0
    for index, case_id in enumerate(cases, start=1):
        if case_ids is not None and case_id not in case_ids:
            continue
        if not _case_in_shard(case_id, shard_count, shard_index):
            continue
        missing_candidates = [
            candidate
            for candidate, root in prediction_roots.items()
            if not (root / f"{case_id}.nii.gz").is_file()
        ]
        if not missing_candidates:
            continue
        case_path = feature_root / f"{case_id}.case.parquet"
        component_path = feature_root / f"{case_id}.components.parquet"
        probability_paths = {
            name: test_roots[name] / f"{case_id}.npz"
            for name in ("XL", "M", "FT")
        }
        reference = nib.load(test_roots["XL"] / f"{case_id}.nii.gz")
        if case_path.is_file() and component_path.is_file():
            case_frame = pd.read_parquet(case_path)
            component_frame = pd.read_parquet(component_path)
            arrays, crop_bbox, full_shape_zyx = load_cropped_probabilities(
                probability_paths,
                threshold=float(config["proposal_threshold"]),
                known_support_bbox=bbox_from_component_frame(component_frame),
                support_bbox_is_complete=True,
                known_full_shape=tuple(
                    int(value) for value in reference.shape[::-1]
                ),
            )
        else:
            arrays = {
                name: load_probabilities(path)
                for name, path in probability_paths.items()
            }
            full_shape_zyx = tuple(int(value) for value in arrays["XL"].shape[1:])
            crop_bbox = tuple(slice(0, value) for value in full_shape_zyx)
        if full_shape_zyx != tuple(int(value) for value in reference.shape[::-1]):
            raise RuntimeError(
                f"{case_id}: probability shape {full_shape_zyx} "
                f"!= reference ZYX shape {reference.shape[::-1]}"
            )
        if not (case_path.is_file() and component_path.is_file()):
            empty_gt = np.zeros(arrays["XL"].shape[1:], dtype=np.uint8)
            case_rows, component_rows = extract_case_features(
                case_id,
                -1,
                arrays["XL"],
                arrays["M"],
                arrays["FT"],
                empty_gt,
            )
            case_frame = pd.DataFrame(case_rows)
            component_frame = pd.DataFrame(component_rows)
        component_probability, _ = predict_test_component_probability(
            v2_bundle,
            v1_case_bundle,
            case_frame,
            component_frame,
        )
        proposal_scores = proposal_scores_from_predictions(
            component_frame, component_probability
        )
        proposals = propose_components(
            arrays, threshold=float(config["proposal_threshold"])
        )
        if set(proposal_scores) != _expected_proposal_keys(proposals):
            raise RuntimeError(f"{case_id}: cached test features differ from proposals")
        spacing = tuple(float(value) for value in reference.header.get_zooms()[:3][::-1])
        for candidate in missing_candidates:
            regions, score_volumes, _ = reconstruct_scored_candidate(
                arrays, proposals, proposal_scores, candidate
            )
            regions = filter_scored_components(
                regions,
                score_volumes,
                cutoffs["candidates"][candidate]["final"]["cutoffs"],
            )
            maps = candidate_score_maps(arrays, candidate)
            regions = apply_gate_preserving_v2(
                regions, maps, spacing, config["v2_final"]
            )
            segmentation_zyx = restore_cropped_label(
                regions_to_label(regions),
                crop_bbox,
                full_shape_zyx,
            )
            segmentation_xyz = segmentation_zyx.transpose(2, 1, 0)
            if segmentation_xyz.shape != reference.shape:
                raise RuntimeError(
                    f"{case_id}: output shape {segmentation_xyz.shape} "
                    f"!= reference {reference.shape}"
                )
            header = reference.header.copy()
            header.set_data_dtype(np.uint8)
            image = nib.Nifti1Image(
                segmentation_xyz.astype(np.uint8), reference.affine, header
            )
            destination = prediction_roots[candidate] / f"{case_id}.nii.gz"
            temporary = destination.with_name(f".{case_id}.tmp.nii.gz")
            nib.save(image, temporary)
            os.replace(temporary, destination)
        processed += 1
        print(
            json.dumps(
                {
                    "event": "v2_test_case_complete",
                    "case_id": case_id,
                    "index": index,
                    "processed": processed,
                    "total": len(cases),
                    "candidates": missing_candidates,
                }
            ),
            flush=True,
        )
        del arrays, case_frame, component_frame, proposals
        _release_native_memory()
        if max_cases is not None and processed >= max_cases:
            break
    counts = {
        candidate: len(list(root.glob("*.nii.gz")))
        for candidate, root in prediction_roots.items()
    }
    return {"processed": processed, **counts}


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "build-calibration",
            "consolidate-calibration",
            "evaluate",
            "consolidate-evaluation",
            "infer",
        ),
    )
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--case-ids", nargs="+")
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.stage == "build-calibration":
        result = build_calibration_cases(
            config,
            max_cases=args.max_cases,
            shard_count=args.shard_count,
            shard_index=args.shard_index,
            case_ids=set(args.case_ids) if args.case_ids else None,
        )
    elif args.stage == "consolidate-calibration":
        result = consolidate_calibration(config)
    elif args.stage == "evaluate":
        result = evaluate_oof_cases(
            config,
            max_cases=args.max_cases,
            shard_count=args.shard_count,
            shard_index=args.shard_index,
            case_ids=set(args.case_ids) if args.case_ids else None,
        )
    elif args.stage == "consolidate-evaluation":
        result = consolidate_evaluation(config)
    elif args.stage == "infer":
        result = infer_test_cases(
            config,
            max_cases=args.max_cases,
            shard_count=args.shard_count,
            shard_index=args.shard_index,
            case_ids=set(args.case_ids) if args.case_ids else None,
        )
    else:
        result = infer_test_cases(
            config,
            max_cases=args.max_cases,
            shard_count=args.shard_count,
            shard_index=args.shard_index,
            case_ids=set(args.case_ids) if args.case_ids else None,
        )
    print(
        json.dumps(
            {"event": f"v2_{args.stage}_complete", "result": result},
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
