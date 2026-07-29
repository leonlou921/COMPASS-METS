"""OOF reconstruction, finite lesion metrics, and safety-contract selection."""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from compass_mets.learned_gates.lcv1.component_gate import (
    REGION_NAMES,
    label_to_regions,
    load_probabilities,
    propose_components,
    reconstruct_regions,
    regions_to_label,
)
from compass_mets.learned_gates.lcv1.features import (
    equal_logit_fusion,
    fixed_component_conf_masks,
)


def select_safe_candidate(
    baseline: dict,
    candidates: Iterable[dict],
    contract: dict,
) -> dict:
    """Apply hard safety gates then the approved metric tie order."""
    baseline_fp = int(baseline["false_positive_components"])
    allowed_fp = baseline_fp * (1.0 + float(contract["max_fp_relative_increase"]))
    allowed_hd95 = float(baseline["macro_hd95"]) + float(
        contract["max_macro_hd95_increase"]
    )
    minimum_worst_delta = float(contract["min_worst_case_macro_dsc_delta"])
    evaluated = []
    passing = []
    for source in candidates:
        candidate = dict(source)
        reasons = []
        if float(candidate["false_positive_components"]) > allowed_fp:
            reasons.append("false_positive_components")
        if float(candidate["macro_hd95"]) > allowed_hd95:
            reasons.append("macro_hd95")
        if float(candidate["worst_case_macro_dsc_delta"]) < minimum_worst_delta:
            reasons.append("worst_case_macro_dsc_delta")
        improves_detection = (
            float(candidate["small_f1"]) > float(baseline["small_f1"])
            or int(candidate["true_positive_components"])
            > int(baseline["true_positive_components"])
        )
        if not improves_detection:
            reasons.append("no_detection_improvement")
        record = {"candidate": candidate["candidate"], "passes": not reasons, "reasons": reasons}
        evaluated.append(record)
        if not reasons:
            passing.append(candidate)
    if passing:
        selected = max(
            passing,
            key=lambda row: (
                float(row["small_f1"]),
                float(row["all_f1"]),
                float(row["macro_dsc"]),
                -float(row["macro_hd95"]),
            ),
        )
    else:
        selected = dict(baseline)
    return {"selected": selected, "evaluated": evaluated, "contract": dict(contract)}


def _enforce_hierarchy(regions: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    output = {name: np.asarray(regions[name], dtype=bool).copy() for name in REGION_NAMES}
    output["TC"] |= output["ET"]
    output["WT"] |= output["TC"]
    return output


def _mask_bbox(mask: np.ndarray, pad: int = 0) -> tuple[slice, ...] | None:
    coordinates = np.where(mask)
    if coordinates[0].size == 0:
        return None
    lower = [max(int(axis.min()) - pad, 0) for axis in coordinates]
    upper = [min(int(axis.max()) + pad + 1, size) for axis, size in zip(coordinates, mask.shape)]
    return tuple(slice(lo, hi) for lo, hi in zip(lower, upper))


def _filter_components(
    mask: np.ndarray,
    score: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    minimum_volume: float,
    minimum_mean: float,
    minimum_peak: float,
) -> np.ndarray:
    """Exact six-connected component gate used by the frozen M V2."""
    if not np.any(mask):
        return np.asarray(mask, dtype=bool)
    bbox = _mask_bbox(mask, pad=0)
    assert bbox is not None
    local_mask = np.asarray(mask[bbox], dtype=bool)
    local_score = np.asarray(score[bbox])
    labels, count = ndi.label(local_mask, structure=ndi.generate_binary_structure(3, 1))
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    keep = sizes.astype(np.float64) * float(np.prod(spacing_zyx)) >= minimum_volume
    if minimum_mean:
        sums = np.bincount(labels.ravel(), weights=local_score.ravel(), minlength=count + 1)
        means = np.divide(sums, sizes, out=np.zeros_like(sums), where=sizes > 0)
        keep &= means >= minimum_mean
    if minimum_peak:
        peaks = np.zeros(count + 1, dtype=np.float32)
        np.maximum.at(peaks, labels.ravel(), local_score.ravel())
        keep &= peaks >= minimum_peak
    keep[0] = False
    output = np.zeros_like(mask, dtype=bool)
    output[bbox] = keep[labels]
    return output


def apply_v2_final(
    regions: dict[str, np.ndarray],
    probabilities_by_model: dict[str, np.ndarray],
    spacing_zyx: tuple[float, float, float],
    settings: dict,
    fused_scores: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Apply the frozen M t040/b20 TC growth then strict RC replacement."""
    output = _enforce_hierarchy(regions)
    if fused_scores is None:
        fused_scores = _localized_v2_scores(probabilities_by_model)
    tc_score = fused_scores["TC"]
    structure = ndi.generate_binary_structure(3, 1)
    added = 0
    budget = int(settings["tc_boundary_budget"])
    wt_bbox = _mask_bbox(output["WT"], pad=1)
    if wt_bbox is not None:
        local_tc = output["TC"][wt_bbox].copy()
        local_wt = output["WT"][wt_bbox]
        local_score = tc_score[wt_bbox]
    else:
        local_tc = local_wt = local_score = None
    while added < budget and local_tc is not None and local_tc.any():
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
    if wt_bbox is not None:
        output["TC"][wt_bbox] = local_tc
    rc_score = fused_scores["RC"]
    output["RC"] = _filter_components(
        rc_score >= float(settings["rc_threshold"]),
        rc_score,
        spacing_zyx,
        float(settings["rc_minimum_volume_mm3"]),
        float(settings["rc_minimum_mean"]),
        float(settings["rc_minimum_peak"]),
    )
    return _enforce_hierarchy(output)


def _localized_v2_scores(
    probabilities_by_model: dict[str, np.ndarray],
    fused_probabilities: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Build only threshold-relevant float32 V2 maps without a full model stack."""
    fused = (
        equal_logit_fusion(probabilities_by_model)
        if fused_probabilities is None
        else np.asarray(fused_probabilities)
    )
    return {
        "TC": np.maximum(fused[1], fused[2]),
        "RC": fused[3],
    }


def _dice(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    denominator = int(prediction.sum()) + int(target.sum())
    return 1.0 if denominator == 0 else 2.0 * int(np.logical_and(prediction, target).sum()) / denominator


def _surface(mask: np.ndarray) -> np.ndarray:
    return np.logical_and(
        mask,
        ~ndi.binary_erosion(mask, structure=ndi.generate_binary_structure(3, 1), border_value=0),
    )


def _hd95(
    prediction: np.ndarray,
    target: np.ndarray,
    spacing_zyx: tuple[float, float, float],
) -> float:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if not prediction.any() and not target.any():
        return 0.0
    if not prediction.any() or not target.any():
        return float(np.linalg.norm(np.asarray(prediction.shape) * np.asarray(spacing_zyx)))
    bbox = _mask_bbox(np.logical_or(prediction, target), pad=1)
    assert bbox is not None
    prediction_surface = _surface(prediction[bbox])
    target_surface = _surface(target[bbox])
    scale = np.asarray(spacing_zyx, dtype=np.float64)
    prediction_coordinates = np.argwhere(prediction_surface) * scale
    target_coordinates = np.argwhere(target_surface) * scale
    to_target = cKDTree(target_coordinates).query(prediction_coordinates, k=1, workers=1)[0]
    to_prediction = cKDTree(prediction_coordinates).query(target_coordinates, k=1, workers=1)[0]
    distances = np.r_[to_target, to_prediction]
    return float(np.quantile(distances, 0.95)) if distances.size else 0.0


def _detection_counts(
    prediction: np.ndarray,
    target: np.ndarray,
    small_max_voxels: float,
) -> dict[str, int]:
    union = np.logical_or(prediction, target)
    bbox = _mask_bbox(union, pad=1)
    if bbox is None:
        return {key: 0 for key in ("tp", "fp", "fn", "small_tp", "small_fp", "small_fn")}
    prediction = np.asarray(prediction[bbox], dtype=bool)
    target = np.asarray(target[bbox], dtype=bool)
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    prediction_labels, prediction_count = ndi.label(prediction, structure=structure)
    target_labels, target_count = ndi.label(target, structure=structure)
    target_sizes = np.bincount(target_labels.ravel(), minlength=target_count + 1)
    prediction_sizes = np.bincount(prediction_labels.ravel(), minlength=prediction_count + 1)
    detected_target_ids = np.unique(target_labels[prediction_labels > 0])
    detected_target_ids = detected_target_ids[detected_target_ids > 0]
    matched_prediction_ids = np.unique(prediction_labels[target_labels > 0])
    matched_prediction_ids = matched_prediction_ids[matched_prediction_ids > 0]
    target_detected = np.zeros(target_count + 1, dtype=bool)
    prediction_matched = np.zeros(prediction_count + 1, dtype=bool)
    target_detected[detected_target_ids] = True
    prediction_matched[matched_prediction_ids] = True
    target_small = target_sizes <= small_max_voxels
    prediction_small = prediction_sizes <= small_max_voxels
    target_small[0] = False
    prediction_small[0] = False
    tp = int(target_detected[1:].sum())
    fn = int(target_count - tp)
    fp = int((~prediction_matched[1:]).sum())
    small_tp = int(np.logical_and(target_detected, target_small).sum())
    small_fn = int(np.logical_and(~target_detected, target_small).sum())
    small_fp = int(np.logical_and(~prediction_matched, prediction_small).sum())
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "small_tp": small_tp,
        "small_fp": small_fp,
        "small_fn": small_fn,
    }


def evaluate_case_regions(
    candidate: str,
    predicted_regions: dict[str, np.ndarray],
    gt_regions: dict[str, np.ndarray],
    spacing_zyx: tuple[float, float, float],
    small_max_volume_mm3: float,
) -> dict[str, float | int | str]:
    predicted = _enforce_hierarchy(predicted_regions)
    voxel_volume = float(np.prod(spacing_zyx))
    small_max_voxels = small_max_volume_mm3 / voxel_volume
    row: dict[str, float | int | str] = {"candidate": candidate}
    dice_values = []
    hd_values = []
    counts = {key: 0 for key in ("tp", "fp", "fn", "small_tp", "small_fp", "small_fn")}
    for region in REGION_NAMES:
        prediction = predicted[region]
        target = gt_regions[region]
        bbox = _mask_bbox(np.logical_or(prediction, target), pad=1)
        if bbox is None:
            local_prediction = prediction[(slice(0, 1),) * 3]
            local_target = target[(slice(0, 1),) * 3]
        else:
            local_prediction = prediction[bbox]
            local_target = target[bbox]
        dice = _dice(local_prediction, local_target)
        hd95 = _hd95(local_prediction, local_target, spacing_zyx)
        row[f"{region}_dsc"] = dice
        row[f"{region}_hd95"] = hd95
        dice_values.append(dice)
        hd_values.append(hd95)
        region_counts = _detection_counts(
            local_prediction, local_target, small_max_voxels
        )
        for key, value in region_counts.items():
            counts[key] += int(value)
    row["macro_dsc"] = float(np.mean(dice_values))
    row["macro_hd95"] = float(np.mean(hd_values))
    row.update(counts)
    return row


def _f1(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * tp + fp + fn
    return 1.0 if denominator == 0 else 2.0 * tp / denominator


def summarize_case_metrics(table: pd.DataFrame) -> list[dict]:
    summaries = []
    for candidate, group in table.groupby("candidate", sort=False):
        tp, fp, fn = (int(group[key].sum()) for key in ("tp", "fp", "fn"))
        small_tp, small_fp, small_fn = (
            int(group[key].sum()) for key in ("small_tp", "small_fp", "small_fn")
        )
        summaries.append(
            {
                "candidate": candidate,
                "small_f1": _f1(small_tp, small_fp, small_fn),
                "all_f1": _f1(tp, fp, fn),
                "macro_dsc": float(group["macro_dsc"].mean()),
                "macro_hd95": float(group["macro_hd95"].mean()),
                "false_positive_components": fp,
                "true_positive_components": tp,
                "false_negative_components": fn,
                "small_true_positive_components": small_tp,
                "small_false_positive_components": small_fp,
                "small_false_negative_components": small_fn,
                "worst_case_macro_dsc_delta": float(group["macro_dsc_delta"].min()),
                "case_count": int(len(group)),
            }
        )
    return summaries


def _oof_probability_path(root: Path, fold: int, case_id: str) -> Path:
    return root / f"fold_{fold}" / "validation" / f"{case_id}.npz"


def _metrics_checkpoint(output_dir: Path, case_id: str) -> Path:
    return output_dir / f"{case_id}.json"


def _release_native_memory() -> None:
    """Return large temporary ndimage/numpy arenas between candidate reconstructions."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (AttributeError, OSError):
        pass


def _case_in_shard(case_id: str, shard_count: int, shard_index: int) -> bool:
    """Return a stable, mutually exclusive case assignment."""
    if shard_count < 1:
        raise ValueError("shard_count must be >= 1")
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must be in [0, shard_count)")
    digest = hashlib.sha256(case_id.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return value % shard_count == shard_index


def _oof_case_folds(
    case_predictions: pd.DataFrame, component_predictions: pd.DataFrame
) -> dict[str, int]:
    fold_counts = case_predictions.groupby("case_id", sort=False)["fold"].nunique()
    invalid = fold_counts[fold_counts != 1]
    if not invalid.empty:
        raise RuntimeError(f"invalid case fold mappings: {invalid.index.tolist()[:5]}")
    case_folds = {
        str(case_id): int(group["fold"].iloc[0])
        for case_id, group in case_predictions.groupby("case_id", sort=False)
    }
    component_cases = set(component_predictions["case_id"].astype(str))
    extra = sorted(component_cases - set(case_folds))
    if extra:
        raise RuntimeError(f"component predictions contain unknown cases: {extra[:5]}")
    for case_id, group in component_predictions.groupby("case_id", sort=False):
        folds = group["fold"].astype(int).unique()
        if len(folds) != 1 or int(folds[0]) != case_folds[str(case_id)]:
            raise RuntimeError(f"component fold mapping disagrees for {case_id}: {folds}")
    return case_folds


def evaluate_oof(
    config: dict,
    max_cases: int | None = None,
    shard_count: int = 1,
    shard_index: int = 0,
    checkpoint_only: bool = False,
    consolidate_only: bool = False,
) -> dict:
    if checkpoint_only and consolidate_only:
        raise ValueError("checkpoint_only and consolidate_only are mutually exclusive")
    _case_in_shard("validation", shard_count, shard_index)
    root = Path(config["output_root"])
    predictions = pd.read_parquet(root / "oof_predictions" / "component_predictions.parquet")
    case_predictions = pd.read_parquet(root / "oof_predictions" / "case_predictions.parquet")
    case_folds = _oof_case_folds(case_predictions, predictions)
    component_groups = {
        str(case_id): group for case_id, group in predictions.groupby("case_id", sort=False)
    }
    empty_components = predictions.iloc[0:0].copy()
    metric_dir = root / "metrics" / "per_case"
    metric_dir.mkdir(parents=True, exist_ok=True)
    oof_roots = {name: Path(config["oof_roots"][name]) for name in ("XL", "M", "FT")}
    labels_root = Path(config["labels_root"])
    processed = 0
    if not consolidate_only:
        for case_id, fold in sorted(case_folds.items()):
            if not _case_in_shard(case_id, shard_count, shard_index):
                continue
            checkpoint = _metrics_checkpoint(metric_dir, case_id)
            if checkpoint.is_file():
                continue
            group = component_groups.get(case_id, empty_components)
            arrays = {
                name: load_probabilities(_oof_probability_path(oof_roots[name], fold, case_id))
                for name in ("XL", "M", "FT")
            }
            gt_image = nib.load(labels_root / f"{case_id}.nii.gz")
            gt_label = np.asarray(gt_image.dataobj).transpose(2, 1, 0)
            gt_regions = label_to_regions(gt_label)
            spacing = tuple(float(x) for x in gt_image.header.get_zooms()[:3][::-1])
            proposals = propose_components(arrays, threshold=float(config["proposal_threshold"]))
            fused_probabilities = equal_logit_fusion(arrays)
            v2_scores = _localized_v2_scores(arrays, fused_probabilities)
            decisions = {}
            for algorithm in ("logistic", "lightgbm"):
                decisions[algorithm] = {
                    (row.region, int(row.component_id)): bool(getattr(row, f"{algorithm}_keep"))
                    for row in group.itertuples()
                }
            rows = []
            baseline_regions = fixed_component_conf_masks(
                fused_probabilities,
                spacing_zyx=spacing,
                settings=config["fixed_component_conf"],
            )
            baseline_regions = label_to_regions(regions_to_label(baseline_regions))
            baseline_row = evaluate_case_regions(
                "fixed_component_conf",
                baseline_regions,
                gt_regions,
                spacing,
                float(config["small_lesion_max_volume_mm3"]),
            )
            rows.append(baseline_row)
            del baseline_regions
            _release_native_memory()
            for algorithm in ("logistic", "lightgbm"):
                for policy in ("filter_only", "consensus_rescue"):
                    regions = reconstruct_regions(
                        arrays,
                        proposals,
                        decisions[algorithm],
                        policy,
                        streaming=True,
                    )
                    regions = label_to_regions(regions_to_label(regions))
                    base_name = f"{algorithm}_{policy}"
                    rows.append(
                        evaluate_case_regions(
                            base_name,
                            regions,
                            gt_regions,
                            spacing,
                            float(config["small_lesion_max_volume_mm3"]),
                        )
                    )
                    v2_regions = apply_v2_final(
                        regions, arrays, spacing, config["v2_final"], fused_scores=v2_scores
                    )
                    rows.append(
                        evaluate_case_regions(
                            f"{base_name}_V2final",
                            v2_regions,
                            gt_regions,
                            spacing,
                            float(config["small_lesion_max_volume_mm3"]),
                        )
                    )
                    del regions, v2_regions
                    _release_native_memory()
            baseline_dsc = float(baseline_row["macro_dsc"])
            for row in rows:
                row["case_id"] = case_id
                row["fold"] = fold
                row["macro_dsc_delta"] = float(row["macro_dsc"]) - baseline_dsc
            temporary = checkpoint.with_name(checkpoint.name + f".tmp.{os.getpid()}")
            temporary.write_text(json.dumps(rows, sort_keys=True), encoding="utf-8")
            os.replace(temporary, checkpoint)
            processed += 1
            print(
                json.dumps(
                    {
                        "event": "oof_case_evaluated",
                        "case_id": case_id,
                        "fold": fold,
                        "processed": processed,
                        "shard_index": shard_index,
                        "shard_count": shard_count,
                    }
                ),
                flush=True,
            )
            if max_cases is not None and processed >= max_cases:
                break

    if checkpoint_only:
        return {
            "processed": processed,
            "shard_index": shard_index,
            "shard_count": shard_count,
        }

    checkpoint_paths = sorted(metric_dir.glob("*.json"))
    expected_case_ids = set(case_folds)
    checkpoint_case_ids = {path.stem for path in checkpoint_paths}
    if max_cases is None and checkpoint_case_ids != expected_case_ids:
        missing = sorted(expected_case_ids - checkpoint_case_ids)
        extra = sorted(checkpoint_case_ids - expected_case_ids)
        raise RuntimeError(
            f"OOF metric checkpoint universe invalid: expected={len(expected_case_ids)}, "
            f"found={len(checkpoint_case_ids)}, missing={missing[:5]}, extra={extra[:5]}"
        )
    all_rows = []
    for path in checkpoint_paths:
        all_rows.extend(json.loads(path.read_text(encoding="utf-8")))
    metrics = pd.DataFrame(all_rows)
    metrics_root = root / "metrics"
    metrics.to_parquet(metrics_root / "oof_case_metrics.parquet", index=False)
    metrics.to_csv(metrics_root / "oof_case_metrics.csv", index=False)
    summaries = summarize_case_metrics(metrics)
    baseline = next(row for row in summaries if row["candidate"] == "fixed_component_conf")
    candidates = [row for row in summaries if row["candidate"] != "fixed_component_conf"]
    selection = select_safe_candidate(baseline, candidates, config["safety_contract"])
    report = {
        "case_count": int(metrics["case_id"].nunique()),
        "summaries": summaries,
        "selection": selection,
    }
    (metrics_root / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    (root / "selection.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--checkpoint-only", action="store_true")
    parser.add_argument("--consolidate-only", action="store_true")
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    report = evaluate_oof(
        config,
        max_cases=args.max_cases,
        shard_count=args.shard_count,
        shard_index=args.shard_index,
        checkpoint_only=args.checkpoint_only,
        consolidate_only=args.consolidate_only,
    )
    if args.checkpoint_only:
        event = {"event": "oof_evaluation_shard_complete", **report}
    else:
        event = {
            "event": "oof_evaluation_complete",
            "case_count": report["case_count"],
            "selected": report["selection"]["selected"]["candidate"],
        }
    print(json.dumps(event), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
