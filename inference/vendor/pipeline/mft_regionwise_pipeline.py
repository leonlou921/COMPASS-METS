#!/usr/bin/env python
"""Stream aligned M/FT/Synthetic probabilities into postprocessed labels."""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy import ndimage
from scipy import special


CHANNELS = ("wt", "tc", "et", "rc")
LABELS = {"wt": 2, "tc": 1, "et": 3, "rc": 4}
EPS = 1e-5

DEFAULT_GATES: dict[str, dict[str, float]] = {
    "thresholds": {name: 0.50 for name in CHANNELS},
    "min_volume_mm3": {"wt": 10.0, "tc": 10.0, "et": 5.0, "rc": 20.0},
    "min_mean": {"wt": 0.0, "tc": 0.0, "et": 0.55, "rc": 0.65},
    "min_peak": {"wt": 0.0, "tc": 0.0, "et": 0.70, "rc": 0.80},
}
CURRENT_FT_WEIGHTS = {"wt": 0.15, "tc": 0.20, "et": 0.15, "rc": 0.30}


def load_probabilities(path: Path) -> np.ndarray:
    with np.load(path) as data:
        key = "probabilities" if "probabilities" in data else "softmax"
        probabilities = np.asarray(data[key], dtype=np.float32)
    if probabilities.ndim != 4 or probabilities.shape[0] != 4:
        raise ValueError(f"{path}: expected WT/TC/ET/RC probabilities, got {probabilities.shape}")
    return probabilities


def fuse_region_logits(
    m_probabilities: np.ndarray,
    ft_probabilities: np.ndarray,
    ft_weights: Mapping[str, float],
) -> np.ndarray:
    if m_probabilities.shape != ft_probabilities.shape:
        raise ValueError(f"M/FT shape mismatch: {m_probabilities.shape} vs {ft_probabilities.shape}")
    fused = np.empty_like(m_probabilities, dtype=np.float32)
    temporary = np.empty_like(fused[0])
    for index, region in enumerate(CHANNELS):
        weight = float(ft_weights[region])
        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"Invalid FT weight for {region}: {weight}")
        np.clip(m_probabilities[index], EPS, 1.0 - EPS, out=fused[index])
        special.logit(fused[index], out=fused[index])
        fused[index] *= 1.0 - weight
        np.clip(ft_probabilities[index], EPS, 1.0 - EPS, out=temporary)
        special.logit(temporary, out=temporary)
        temporary *= weight
        fused[index] += temporary
        special.expit(fused[index], out=fused[index])
    return fused


def filter_components(
    mask: np.ndarray,
    probabilities: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    min_volume_mm3: float,
    min_mean: float,
    min_peak: float,
) -> np.ndarray:
    if not np.any(mask):
        return np.asarray(mask, dtype=bool)
    labels, count = ndimage.label(mask, structure=ndimage.generate_binary_structure(3, 1))
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    keep = sizes.astype(np.float64) * float(np.prod(spacing_zyx)) >= float(min_volume_mm3)
    if min_mean:
        sums = np.bincount(labels.ravel(), weights=probabilities.ravel(), minlength=count + 1)
        means = np.divide(sums, sizes, out=np.zeros_like(sums), where=sizes > 0)
        keep &= means >= float(min_mean)
    if min_peak:
        peaks = np.zeros(count + 1, dtype=np.float32)
        np.maximum.at(peaks, labels.ravel(), probabilities.ravel())
        keep &= peaks >= float(min_peak)
    keep[0] = False
    return keep[labels]


def enforce_hierarchy(masks: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    output = {name: np.asarray(masks[name], dtype=bool).copy() for name in CHANNELS}
    output["tc"] |= output["et"]
    output["wt"] |= output["tc"]
    return output


def component_conf_masks(
    probabilities: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    gates: Mapping[str, Mapping[str, float]],
) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {}
    for index, region in enumerate(CHANNELS):
        masks[region] = filter_components(
            probabilities[index] >= float(gates["thresholds"][region]),
            probabilities[index],
            spacing_zyx,
            float(gates["min_volume_mm3"][region]),
            float(gates["min_mean"][region]),
            float(gates["min_peak"][region]),
        )
    return enforce_hierarchy(masks)


def structured_probability_masks(
    fused_probabilities: np.ndarray,
    ft_probabilities: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    gates: Mapping[str, Mapping[str, float]],
) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {}
    for index, region in enumerate(CHANNELS):
        probabilities = ft_probabilities[index] if region in {"wt", "tc"} else fused_probabilities[index]
        masks[region] = filter_components(
            probabilities >= float(gates["thresholds"][region]),
            probabilities,
            spacing_zyx,
            float(gates["min_volume_mm3"][region]),
            float(gates["min_mean"][region]),
            float(gates["min_peak"][region]),
        )
    return enforce_hierarchy(masks)


def threshold_component_masks(
    probabilities: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    thresholds: Mapping[str, float],
    min_volume_mm3: Mapping[str, float],
) -> dict[str, np.ndarray]:
    gates = {
        "thresholds": thresholds,
        "min_volume_mm3": min_volume_mm3,
        "min_mean": {name: 0.0 for name in CHANNELS},
        "min_peak": {name: 0.0 for name in CHANNELS},
    }
    return component_conf_masks(probabilities, spacing_zyx, gates)


def masks_to_segmentation(masks: Mapping[str, np.ndarray]) -> np.ndarray:
    fixed = enforce_hierarchy(masks)
    segmentation = np.zeros(fixed["wt"].shape, dtype=np.uint8)
    for region in CHANNELS:
        segmentation[fixed[region]] = LABELS[region]
    return segmentation


def compose_label_hybrid(
    mft_masks: Mapping[str, np.ndarray], ft_masks: Mapping[str, np.ndarray]
) -> np.ndarray:
    masks = {
        "wt": np.asarray(ft_masks["wt"], dtype=bool),
        "tc": np.asarray(ft_masks["tc"], dtype=bool),
        "et": np.asarray(mft_masks["et"], dtype=bool),
        "rc": np.asarray(mft_masks["rc"], dtype=bool),
    }
    return masks_to_segmentation(masks)


def _connected_budget_mask(component: np.ndarray, score: np.ndarray, budget: int) -> np.ndarray:
    coordinates = np.argwhere(component)
    if len(coordinates) <= budget:
        return component.copy()
    peak_coordinate = tuple(coordinates[np.argmax(score[component])])
    selected = np.zeros_like(component, dtype=bool)
    selected[peak_coordinate] = True
    structure = ndimage.generate_binary_structure(3, 1)
    while int(selected.sum()) < budget:
        frontier = ndimage.binary_dilation(selected, structure=structure) & component & ~selected
        if not np.any(frontier):
            break
        frontier_coordinates = np.argwhere(frontier)
        next_coordinate = tuple(frontier_coordinates[np.argmax(score[frontier])])
        selected[next_coordinate] = True
    return selected


def apply_synthetic_rescue(
    base_masks: Mapping[str, np.ndarray],
    m_et: np.ndarray,
    ft_et: np.ndarray,
    synthetic_et: np.ndarray,
    fused_et: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    rescue: Mapping[str, float | int],
    synthetic_components: tuple[np.ndarray, int] | None = None,
    allowed_mask: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    output = enforce_hierarchy(base_masks)
    if synthetic_components is None:
        labels, count = ndimage.label(
            synthetic_et >= float(rescue["candidate_threshold"]),
            structure=ndimage.generate_binary_structure(3, 1),
        )
    else:
        labels, count = synthetic_components
    if count == 0:
        return output, {"rescued_components": 0, "rescued_voxels": 0}

    adjacency = int(rescue["adjacency_voxels"])
    allowed = allowed_mask
    if allowed is None:
        allowed = ndimage.binary_dilation(
            output["wt"],
            structure=ndimage.generate_binary_structure(3, 1),
            iterations=adjacency,
        )
    voxel_mm3 = float(np.prod(spacing_zyx))
    support_threshold = float(rescue["model_support_threshold"])
    eligible: list[tuple[tuple[float, float, float, float], np.ndarray]] = []
    for component_id in range(1, count + 1):
        component = labels == component_id
        volume = float(component.sum()) * voxel_mm3
        if volume <= 0.0 or volume > float(rescue["max_volume_mm3"]):
            continue
        constrained = component & allowed & ~output["et"]
        if not np.any(constrained):
            continue
        m_peak = float(m_et[component].max())
        ft_peak = float(ft_et[component].max())
        synthetic_peak = float(synthetic_et[component].max())
        fused_peak = float(fused_et[component].max())
        support_count = int(m_peak >= support_threshold) + int(ft_peak >= support_threshold) + int(
            synthetic_peak >= float(rescue["candidate_threshold"])
        )
        if support_count < 2 or fused_peak < float(rescue["fused_peak"]):
            continue
        rank_key = (fused_peak, float(support_count), synthetic_peak, -volume)
        eligible.append((rank_key, constrained))

    if not eligible:
        return output, {"rescued_components": 0, "rescued_voxels": 0}
    eligible.sort(key=lambda item: item[0], reverse=True)
    rescued_components = 0
    rescued_voxels = 0
    for _, component in eligible[: int(rescue["max_components_per_case"])]:
        selected = _connected_budget_mask(component, fused_et, int(rescue["voxel_budget"]))
        if not np.any(selected):
            continue
        output["et"] |= selected
        rescued_components += 1
        rescued_voxels += int(selected.sum())
    output = enforce_hierarchy(output)
    return output, {"rescued_components": rescued_components, "rescued_voxels": rescued_voxels}


def apply_tc_boundary_completion(
    base_masks: Mapping[str, np.ndarray],
    tc_score: np.ndarray,
    completion: Mapping[str, float | int],
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Grow an existing TC only through high-confidence voxels already inside WT."""
    output = enforce_hierarchy(base_masks)
    budget = int(completion["voxel_budget"])
    if budget <= 0 or not np.any(output["tc"]):
        return output, {"rescued_components": 0, "rescued_voxels": 0}

    threshold = float(completion["threshold"])
    structure = ndimage.generate_binary_structure(3, 1)
    added = 0
    while added < budget:
        frontier = (
            ndimage.binary_dilation(output["tc"], structure=structure)
            & output["wt"]
            & ~output["tc"]
            & (tc_score >= threshold)
        )
        if not np.any(frontier):
            break
        coordinates = np.argwhere(frontier)
        coordinate = tuple(coordinates[np.argmax(tc_score[frontier])])
        output["tc"][coordinate] = True
        added += 1
    return enforce_hierarchy(output), {"rescued_components": 0, "rescued_voxels": added}


def _copy_gates(config: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    source = config.get("gates", DEFAULT_GATES)
    return {section: {region: float(source[section][region]) for region in CHANNELS} for section in DEFAULT_GATES}


def build_segmentation(
    m_probabilities: np.ndarray,
    ft_probabilities: np.ndarray,
    synthetic_probabilities: np.ndarray | None,
    spacing_zyx: tuple[float, float, float],
    config: Mapping[str, Any],
    *,
    fused_probabilities: np.ndarray | None = None,
    base_component_masks: Mapping[str, np.ndarray] | None = None,
    synthetic_components: tuple[np.ndarray, int] | None = None,
    rescue_allowed_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    mode = str(config["mode"])
    gates = _copy_gates(config)
    weights = {name: float(config.get("ft_weights", CURRENT_FT_WEIGHTS)[name]) for name in CHANNELS}
    source = str(config.get("source", "fused"))
    needs_fused = source == "fused" or mode in {
        "label_hybrid",
        "probability_hybrid",
        "rescue",
        "tc_boundary_completion",
    }
    fused = fused_probabilities
    if needs_fused and fused is None:
        fused = fuse_region_logits(m_probabilities, ft_probabilities, weights)
    stats = {"rescued_components": 0, "rescued_voxels": 0}

    if mode == "component":
        if source == "fused" and fused is None:
            raise RuntimeError("Missing fused probabilities")
        probabilities = {"m": m_probabilities, "ft": ft_probabilities, "fused": fused}[source]
        masks = component_conf_masks(probabilities, spacing_zyx, gates)
        return masks_to_segmentation(masks), stats
    if mode == "threshold_component":
        source = str(config.get("source", "m"))
        if source == "fused" and fused is None:
            raise RuntimeError("Missing fused probabilities")
        probabilities = {"m": m_probabilities, "ft": ft_probabilities, "fused": fused}[source]
        masks = threshold_component_masks(
            probabilities,
            spacing_zyx,
            config["thresholds"],
            config["min_volume_mm3"],
        )
        return masks_to_segmentation(masks), stats
    if mode == "label_hybrid":
        assert fused is not None
        mft_masks = component_conf_masks(fused, spacing_zyx, gates)
        ft_masks = component_conf_masks(ft_probabilities, spacing_zyx, gates)
        return compose_label_hybrid(mft_masks, ft_masks), stats
    if mode == "probability_hybrid":
        assert fused is not None
        masks = structured_probability_masks(fused, ft_probabilities, spacing_zyx, gates)
        return masks_to_segmentation(masks), stats
    if mode == "tc_boundary_completion":
        assert fused is not None
        masks = structured_probability_masks(fused, ft_probabilities, spacing_zyx, gates)
        tc_score = np.maximum(
            fused[CHANNELS.index("tc")],
            fused[CHANNELS.index("et")],
        )
        masks, stats = apply_tc_boundary_completion(masks, tc_score, config["completion"])
        return masks_to_segmentation(masks), stats
    if mode == "rescue":
        if synthetic_probabilities is None:
            raise ValueError("Synthetic probabilities are required for rescue mode")
        assert fused is not None
        masks = (
            enforce_hierarchy(base_component_masks)
            if base_component_masks is not None
            else component_conf_masks(fused, spacing_zyx, gates)
        )
        masks, stats = apply_synthetic_rescue(
            masks,
            m_probabilities[CHANNELS.index("et")],
            ft_probabilities[CHANNELS.index("et")],
            synthetic_probabilities[CHANNELS.index("et")],
            fused[CHANNELS.index("et")],
            spacing_zyx,
            config["rescue"],
            synthetic_components=synthetic_components,
            allowed_mask=rescue_allowed_mask,
        )
        return masks_to_segmentation(masks), stats
    raise ValueError(f"Unknown config mode: {mode}")


@dataclass(frozen=True)
class Case:
    case_id: str
    fold: int | None
    m_npz: Path
    ft_npz: Path
    synthetic_npz: Path | None
    reference: Path


def _npz_uncompressed_bytes(path: Path) -> int:
    with zipfile.ZipFile(path) as archive:
        return sum(info.file_size for info in archive.infolist())


def partition_cases_by_memory(
    cases: list[Case],
    need_synthetic: bool,
    small_max_bytes: int,
    medium_max_bytes: int,
) -> tuple[list[Case], list[Case], list[Case]]:
    tiers: tuple[list[Case], list[Case], list[Case]] = ([], [], [])
    for case in cases:
        estimated = _npz_uncompressed_bytes(case.m_npz) + _npz_uncompressed_bytes(case.ft_npz)
        if need_synthetic:
            if case.synthetic_npz is None:
                raise ValueError(f"Missing Synthetic probabilities for {case.case_id}")
            estimated += _npz_uncompressed_bytes(case.synthetic_npz)
        tier = 0 if estimated <= small_max_bytes else 1 if estimated <= medium_max_bytes else 2
        tiers[tier].append(case)
    return tiers


def _flat_ids(root: Path) -> set[str]:
    return {path.stem for path in root.glob("*.npz")}


def collect_cases(
    case_mode: str,
    m_root: Path,
    ft_root: Path,
    synthetic_root: Path | None,
    need_synthetic: bool,
) -> list[Case]:
    cases: list[Case] = []
    expected = 1295 if case_mode == "oof" else 179
    if case_mode == "oof":
        for fold in range(5):
            m_dir = m_root / f"fold_{fold}" / "validation"
            ft_dir = ft_root / f"fold_{fold}" / "validation"
            synthetic_dir = synthetic_root / f"fold_{fold}" / "validation" if synthetic_root else None
            ids = _flat_ids(m_dir)
            if ids != _flat_ids(ft_dir):
                raise RuntimeError(f"Fold {fold}: M and FT case sets differ")
            if need_synthetic and (synthetic_dir is None or ids != _flat_ids(synthetic_dir)):
                raise RuntimeError(f"Fold {fold}: Synthetic case set is missing or differs")
            for case_id in sorted(ids):
                cases.append(
                    Case(
                        case_id,
                        fold,
                        m_dir / f"{case_id}.npz",
                        ft_dir / f"{case_id}.npz",
                        synthetic_dir / f"{case_id}.npz" if synthetic_dir else None,
                        m_dir / f"{case_id}.nii.gz",
                    )
                )
    else:
        ids = _flat_ids(m_root)
        if ids != _flat_ids(ft_root):
            raise RuntimeError("Test M and FT case sets differ")
        if need_synthetic and (synthetic_root is None or ids != _flat_ids(synthetic_root)):
            raise RuntimeError("Test Synthetic case set is missing or differs")
        for case_id in sorted(ids):
            cases.append(
                Case(
                    case_id,
                    None,
                    m_root / f"{case_id}.npz",
                    ft_root / f"{case_id}.npz",
                    synthetic_root / f"{case_id}.npz" if synthetic_root else None,
                    m_root / f"{case_id}.nii.gz",
                )
            )
    if len(cases) != expected or len({case.case_id for case in cases}) != expected:
        raise RuntimeError(f"Expected {expected} distinct {case_mode} cases, found {len(cases)}")
    for case in cases:
        if not case.reference.exists():
            raise FileNotFoundError(case.reference)
    return cases


_CONFIGS: list[dict[str, Any]] = []
_OUT_ROOT: Path | None = None
_RESUME = False


def _init_worker(configs: list[dict[str, Any]], out_root: str, resume: bool) -> None:
    global _CONFIGS, _OUT_ROOT, _RESUME
    _CONFIGS = configs
    _OUT_ROOT = Path(out_root)
    _RESUME = resume


def _process_case(case: Case) -> dict[str, int | str]:
    import SimpleITK as sitk

    assert _OUT_ROOT is not None
    outputs = {
        config["version_id"]: _OUT_ROOT / "candidates" / config["version_id"] / "predictions" / f"{case.case_id}.nii.gz"
        for config in _CONFIGS
    }
    if _RESUME and all(path.exists() for path in outputs.values()):
        return {"case_id": case.case_id, "written": 0, "rescued_components": 0, "rescued_voxels": 0}
    m = load_probabilities(case.m_npz)
    ft = load_probabilities(case.ft_npz)
    need_synthetic = any(config["mode"] == "rescue" for config in _CONFIGS)
    synthetic = load_probabilities(case.synthetic_npz) if need_synthetic and case.synthetic_npz else None
    reference = sitk.ReadImage(str(case.reference))
    spacing = tuple(float(value) for value in reference.GetSpacing())[::-1]
    totals: dict[str, int | str] = {
        "case_id": case.case_id,
        "written": 0,
        "rescued_components": 0,
        "rescued_voxels": 0,
    }
    fused_cache: dict[tuple[float, ...], np.ndarray] = {}
    base_mask_cache: dict[str, dict[str, np.ndarray]] = {}
    synthetic_component_cache: dict[float, tuple[np.ndarray, int]] = {}
    allowed_mask_cache: dict[tuple[str, int], np.ndarray] = {}
    output_by_digest: dict[bytes, Path] = {}
    for config in _CONFIGS:
        output_path = outputs[config["version_id"]]
        if _RESUME and output_path.exists():
            continue
        mode = str(config["mode"])
        source = str(config.get("source", "fused"))
        needs_fused = source == "fused" or mode in {
            "label_hybrid",
            "probability_hybrid",
            "rescue",
            "tc_boundary_completion",
        }
        weights = config.get("ft_weights", CURRENT_FT_WEIGHTS)
        weights_key = tuple(float(weights[name]) for name in CHANNELS)
        fused = None
        if needs_fused:
            fused = fused_cache.get(weights_key)
            if fused is None:
                fused = fuse_region_logits(m, ft, weights)
                fused_cache[weights_key] = fused

        base_masks = None
        synthetic_components = None
        allowed_mask = None
        if mode == "rescue":
            assert fused is not None and synthetic is not None
            gates = _copy_gates(config)
            base_key = json.dumps(
                {"weights": weights_key, "gates": gates},
                sort_keys=True,
                separators=(",", ":"),
            )
            base_masks = base_mask_cache.get(base_key)
            if base_masks is None:
                base_masks = component_conf_masks(fused, spacing, gates)
                base_mask_cache[base_key] = base_masks
            threshold = float(config["rescue"]["candidate_threshold"])
            synthetic_components = synthetic_component_cache.get(threshold)
            if synthetic_components is None:
                synthetic_components = ndimage.label(
                    synthetic[CHANNELS.index("et")] >= threshold,
                    structure=ndimage.generate_binary_structure(3, 1),
                )
                synthetic_component_cache[threshold] = synthetic_components
            adjacency = int(config["rescue"]["adjacency_voxels"])
            allowed_key = (base_key, adjacency)
            allowed_mask = allowed_mask_cache.get(allowed_key)
            if allowed_mask is None:
                allowed_mask = ndimage.binary_dilation(
                    base_masks["wt"],
                    structure=ndimage.generate_binary_structure(3, 1),
                    iterations=adjacency,
                )
                allowed_mask_cache[allowed_key] = allowed_mask

        segmentation, stats = build_segmentation(
            m,
            ft,
            synthetic,
            spacing,
            config,
            fused_probabilities=fused,
            base_component_masks=base_masks,
            synthetic_components=synthetic_components,
            rescue_allowed_mask=allowed_mask,
        )
        image = sitk.GetImageFromArray(segmentation)
        image.CopyInformation(reference)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp.nii.gz")
        try:
            digest = hashlib.blake2b(memoryview(np.ascontiguousarray(segmentation)), digest_size=16).digest()
            existing = output_by_digest.get(digest)
            if existing is not None and existing.exists():
                os.link(existing, temporary_path)
            else:
                sitk.WriteImage(image, str(temporary_path), True)
            os.replace(temporary_path, output_path)
            output_by_digest[digest] = output_path
        finally:
            temporary_path.unlink(missing_ok=True)
        totals["written"] = int(totals["written"]) + 1
        totals["rescued_components"] = int(totals["rescued_components"]) + stats["rescued_components"]
        totals["rescued_voxels"] = int(totals["rescued_voxels"]) + stats["rescued_voxels"]
    return totals


def _process_case_batch(
    cases: list[Case],
    workers: int,
    configs: list[dict[str, Any]],
    out_root: Path,
    resume: bool,
    max_tasks_per_child: int,
    totals: dict[str, int],
    tier_name: str,
) -> None:
    if not cases:
        print(f"[memory-tier] {tier_name}: 0 cases", flush=True)
        return
    print(f"[memory-tier] {tier_name}: cases={len(cases)} workers={workers}", flush=True)
    if workers == 1:
        _init_worker(configs, str(out_root), resume)
        results = map(_process_case, cases)
        for done, result in enumerate(results, 1):
            for key in totals:
                totals[key] += int(result[key])
            if done % 10 == 0 or done == len(cases):
                print(
                    f"tier={tier_name} completed={done}/{len(cases)} last={result['case_id']} totals={totals}",
                    flush=True,
                )
        return
    context = mp.get_context("fork") if hasattr(os, "fork") else mp.get_context("spawn")
    with context.Pool(
        processes=max(1, workers),
        initializer=_init_worker,
        initargs=(configs, str(out_root), resume),
        maxtasksperchild=max_tasks_per_child or None,
    ) as pool:
        for done, result in enumerate(pool.imap_unordered(_process_case, cases, chunksize=1), 1):
            for key in totals:
                totals[key] += int(result[key])
            if done % 10 == 0 or done == len(cases):
                print(
                    f"tier={tier_name} completed={done}/{len(cases)} last={result['case_id']} totals={totals}",
                    flush=True,
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-mode", choices=("oof", "test"), required=True)
    parser.add_argument("--m-root", type=Path, required=True)
    parser.add_argument("--ft-root", type=Path, required=True)
    parser.add_argument("--synthetic-root", type=Path)
    parser.add_argument("--configs-json", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--max-tasks-per-child", type=int, default=4)
    parser.add_argument("--memory-tiering", action="store_true")
    parser.add_argument("--small-max-gib", type=float, default=1.0)
    parser.add_argument("--medium-max-gib", type=float, default=2.0)
    parser.add_argument("--medium-workers", type=int, default=2)
    parser.add_argument("--large-workers", type=int, default=1)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    configs = json.loads(args.configs_json.read_text(encoding="utf-8"))
    if not isinstance(configs, list) or not configs:
        raise ValueError("configs-json must contain a non-empty list")
    version_ids = [str(config["version_id"]) for config in configs]
    if len(version_ids) != len(set(version_ids)):
        raise ValueError("Duplicate version_id in configs-json")
    need_synthetic = any(config["mode"] == "rescue" for config in configs)
    cases = collect_cases(args.case_mode, args.m_root, args.ft_root, args.synthetic_root, need_synthetic)
    if args.max_cases:
        cases = cases[: args.max_cases]
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Require num_shards >= 1 and 0 <= shard_index < num_shards")
    cases = cases[args.shard_index :: args.num_shards]
    args.out_root.mkdir(parents=True, exist_ok=True)
    for version_id in version_ids:
        (args.out_root / "candidates" / version_id / "predictions").mkdir(parents=True, exist_ok=True)
    manifest = {
        "case_mode": args.case_mode,
        "case_count": len(cases),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "sources": {"m": str(args.m_root), "ft": str(args.ft_root), "synthetic": str(args.synthetic_root)},
        "configs": configs,
        "memory_tiering": args.memory_tiering,
    }
    suffix = f"_shard{args.shard_index:02d}" if args.num_shards > 1 else ""
    (args.out_root / f"generation_manifest{suffix}.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"):
        os.environ[variable] = "1"
    totals = {"written": 0, "rescued_components": 0, "rescued_voxels": 0}
    if args.memory_tiering:
        gib = 1024**3
        tiered_cases = partition_cases_by_memory(
            cases,
            need_synthetic,
            int(args.small_max_gib * gib),
            int(args.medium_max_gib * gib),
        )
        tier_specs = (
            ("small", tiered_cases[0], args.workers),
            ("medium", tiered_cases[1], min(args.workers, args.medium_workers)),
            ("large", tiered_cases[2], min(args.workers, args.large_workers)),
        )
    else:
        tier_specs = (("all", cases, args.workers),)
    for tier_name, tier_cases, workers in tier_specs:
        _process_case_batch(
            tier_cases,
            max(1, workers),
            configs,
            args.out_root,
            args.resume,
            args.max_tasks_per_child,
            totals,
            tier_name,
        )
    (args.out_root / f"generation_summary{suffix}.json").write_text(
        json.dumps(totals, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
