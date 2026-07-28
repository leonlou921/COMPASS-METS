#!/usr/bin/env python
"""Generate the frozen XL-only BraTS MET postprocessing submission set.

The thresholds and component rules are copied from the already evaluated
ResEncM postprocessors. No threshold, volume, confidence, or boundary sweep is
performed. XL-06 is the previously specified N6-style ET consensus rescue,
using three-of-five independent XL fold probability sources.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import zipfile
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy import ndimage

try:
    from .mft_regionwise_pipeline import (
        CHANNELS,
        DEFAULT_GATES,
        apply_tc_boundary_completion,
        component_conf_masks,
        enforce_hierarchy,
        filter_components,
        masks_to_segmentation,
    )
except ImportError:  # direct remote script execution
    from mft_regionwise_pipeline import (
        CHANNELS,
        DEFAULT_GATES,
        apply_tc_boundary_completion,
        component_conf_masks,
        enforce_hierarchy,
        filter_components,
        masks_to_segmentation,
    )


CORE_VERSION_IDS = (
    "XL-00_raw_corrected",
    "XL-01_component_conf_ET_RC",
    "XL-02_highrecall_ET035_RC060_min_ET5_RC50",
    "XL-03_precision_RC075_min_RC100",
    "XL-04_component_tc_boundary_t040_b20",
    "XL-05_V2_t040b20_RCconf_strict",
)
RESCUE_VERSION_ID = "XL-06_ET_fold_consensus_rescue"

HIGH_RECALL_GATES = {
    "thresholds": {"wt": 0.50, "tc": 0.50, "et": 0.35, "rc": 0.60},
    "min_volume_mm3": {"wt": 10.0, "tc": 10.0, "et": 5.0, "rc": 50.0},
    "min_mean": {name: 0.0 for name in CHANNELS},
    "min_peak": {name: 0.0 for name in CHANNELS},
}
PRECISION_GATES = {
    "thresholds": {"wt": 0.50, "tc": 0.50, "et": 0.50, "rc": 0.75},
    "min_volume_mm3": {"wt": 10.0, "tc": 10.0, "et": 5.0, "rc": 100.0},
    "min_mean": {name: 0.0 for name in CHANNELS},
    "min_peak": {name: 0.0 for name in CHANNELS},
}
TC_BOUNDARY = {"threshold": 0.40, "voxel_budget": 20}
RC_STRICT = {
    "threshold": 0.50,
    "min_volume_mm3": 20.0,
    "min_mean": 0.70,
    "min_peak": 0.85,
}
ET_FOLD_CONSENSUS = {
    "support_threshold": 0.25,
    "candidate_threshold": 0.25,
    "min_support_count": 3,
    "max_volume_mm3": 30.0,
    "voxel_budget": 12,
    "max_components": 1,
    "extent": 1,
    "min_mean": 0.55,
    "min_peak": 0.70,
}


def load_probabilities(path: Path) -> np.ndarray:
    with np.load(path) as data:
        key = "probabilities" if "probabilities" in data else "softmax"
        probabilities = np.asarray(data[key], dtype=np.float32)
    if probabilities.ndim != 4 or probabilities.shape[0] != 4:
        raise ValueError(f"{path}: expected WT/TC/ET/RC probabilities, got {probabilities.shape}")
    if not np.isfinite(probabilities).all():
        raise ValueError(f"{path}: probabilities contain NaN or Inf")
    return probabilities


def segmentation_to_masks(segmentation: np.ndarray) -> dict[str, np.ndarray]:
    segmentation = np.asarray(segmentation)
    return {
        "wt": np.isin(segmentation, (1, 2, 3)),
        "tc": np.isin(segmentation, (1, 3)),
        "et": segmentation == 3,
        "rc": segmentation == 4,
    }


def apply_rc_gate(
    base_masks: Mapping[str, np.ndarray],
    rc_score: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    *,
    threshold: float,
    min_volume_mm3: float,
    min_mean: float,
    min_peak: float,
) -> dict[str, np.ndarray]:
    output = enforce_hierarchy(base_masks)
    output["rc"] = filter_components(
        np.asarray(rc_score) >= float(threshold),
        np.asarray(rc_score),
        spacing_zyx,
        float(min_volume_mm3),
        float(min_mean),
        float(min_peak),
    )
    return enforce_hierarchy(output)


def build_core_versions(
    probabilities: np.ndarray,
    spacing_zyx: tuple[float, float, float],
) -> dict[str, np.ndarray]:
    if probabilities.ndim != 4 or probabilities.shape[0] != 4:
        raise ValueError(f"Expected four WT/TC/ET/RC channels, got {probabilities.shape}")
    component = component_conf_masks(probabilities, spacing_zyx, DEFAULT_GATES)
    high_recall = component_conf_masks(probabilities, spacing_zyx, HIGH_RECALL_GATES)
    precision = component_conf_masks(probabilities, spacing_zyx, PRECISION_GATES)
    tc_score = np.maximum(probabilities[CHANNELS.index("tc")], probabilities[CHANNELS.index("et")])
    boundary, _ = apply_tc_boundary_completion(component, tc_score, TC_BOUNDARY)
    strict = apply_rc_gate(boundary, probabilities[CHANNELS.index("rc")], spacing_zyx, **RC_STRICT)
    return {
        CORE_VERSION_IDS[1]: masks_to_segmentation(component),
        CORE_VERSION_IDS[2]: masks_to_segmentation(high_recall),
        CORE_VERSION_IDS[3]: masks_to_segmentation(precision),
        CORE_VERSION_IDS[4]: masks_to_segmentation(boundary),
        CORE_VERSION_IDS[5]: masks_to_segmentation(strict),
    }


def _connected_budget_mask(component: np.ndarray, score: np.ndarray, budget: int) -> np.ndarray:
    selected = np.zeros_like(component, dtype=bool)
    if budget <= 0 or not np.any(component):
        return selected
    coordinates = np.argwhere(component)
    seed = tuple(coordinates[np.argmax(score[component])])
    selected[seed] = True
    structure = ndimage.generate_binary_structure(3, 1)
    while int(selected.sum()) < min(int(budget), int(component.sum())):
        frontier = ndimage.binary_dilation(selected, structure=structure) & component & ~selected
        if not np.any(frontier):
            break
        frontier_coordinates = np.argwhere(frontier)
        coordinate = tuple(frontier_coordinates[np.argmax(score[frontier])])
        selected[coordinate] = True
    return selected


def rescue_et_fold_consensus(
    base_masks: Mapping[str, np.ndarray],
    probability_sources: Sequence[np.ndarray],
    *,
    support_threshold: float,
    candidate_threshold: float,
    min_support_count: int,
    spacing_zyx: tuple[float, float, float],
    max_volume_mm3: float,
    voxel_budget: int,
    max_components: int,
    extent: int,
    min_mean: float,
    min_peak: float,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    if len(probability_sources) != 5:
        raise ValueError("XL ET fold consensus rescue requires exactly five probability sources")
    if not 1 <= int(min_support_count) <= 5:
        raise ValueError("min_support_count must be in [1, 5]")
    output = enforce_hierarchy(base_masks)
    sources = tuple(np.asarray(source, dtype=np.float32) for source in probability_sources)
    candidate = np.logical_or.reduce(tuple(source >= float(candidate_threshold) for source in sources))
    candidate &= ~output["et"]
    candidate &= ndimage.binary_dilation(
        output["wt"],
        structure=ndimage.generate_binary_structure(3, 1),
        iterations=max(1, int(extent)),
    )
    labels, count = ndimage.label(candidate, structure=ndimage.generate_binary_structure(3, 1))
    slices = ndimage.find_objects(labels, max_label=count)
    voxel_mm3 = float(np.prod(spacing_zyx))
    eligible: list[tuple[tuple[int, float, float], int, tuple[slice, ...]]] = []
    for component_id, component_slice in enumerate(slices, 1):
        if component_slice is None:
            continue
        component = labels[component_slice] == component_id
        volume = float(component.sum()) * voxel_mm3
        if volume <= 0.0 or volume > float(max_volume_mm3):
            continue
        local_sources = tuple(source[component_slice] for source in sources)
        support_count = sum(
            float(source[component].max()) >= float(support_threshold) for source in local_sources
        )
        if support_count < int(min_support_count):
            continue
        score = np.maximum.reduce(local_sources)
        peak = float(score[component].max())
        mean = float(score[component].mean())
        if peak < float(min_peak) or mean < float(min_mean):
            continue
        eligible.append(((support_count, peak, mean), component_id, component_slice))
    eligible.sort(key=lambda item: item[0], reverse=True)
    rescued_components = rescued_voxels = 0
    for _, component_id, component_slice in eligible[: int(max_components)]:
        component = labels[component_slice] == component_id
        score = np.maximum.reduce(tuple(source[component_slice] for source in sources))
        selected = _connected_budget_mask(component, score, int(voxel_budget))
        if not np.any(selected):
            continue
        output["et"][component_slice] |= selected
        rescued_components += 1
        rescued_voxels += int(selected.sum())
    return enforce_hierarchy(output), {
        "rescued_components": rescued_components,
        "rescued_voxels": rescued_voxels,
    }


def rescue_delta_statistics(
    base_segmentation: np.ndarray,
    rescued_segmentation: np.ndarray,
) -> dict[str, int]:
    """Recover XL-06 rescue counts from completed outputs during --resume."""
    base = np.asarray(base_segmentation)
    rescued = np.asarray(rescued_segmentation)
    if base.shape != rescued.shape:
        raise ValueError(f"Rescue/base shape mismatch: {rescued.shape} != {base.shape}")
    removed_et = (base == 3) & (rescued != 3)
    if np.any(removed_et):
        raise ValueError(f"Rescue output removed ET from {int(removed_et.sum())} voxels")
    added_et = (rescued == 3) & (base != 3)
    _, component_count = ndimage.label(
        added_et,
        structure=ndimage.generate_binary_structure(3, 1),
    )
    return {
        "rescued_components": int(component_count),
        "rescued_voxels": int(added_et.sum()),
    }


_ENSEMBLE_ROOT: Path | None = None
_FOLD_ROOTS: tuple[Path, ...] = ()
_OUT_ROOT: Path | None = None
_RESUME = False


def _version_path(version_id: str, case_id: str) -> Path:
    assert _OUT_ROOT is not None
    return _OUT_ROOT / "candidates" / version_id / "predictions" / f"{case_id}.nii.gz"


def _init_worker(ensemble_root: str, fold_roots: list[str], out_root: str, resume: bool) -> None:
    global _ENSEMBLE_ROOT, _FOLD_ROOTS, _OUT_ROOT, _RESUME
    _ENSEMBLE_ROOT = Path(ensemble_root)
    _FOLD_ROOTS = tuple(Path(path) for path in fold_roots)
    _OUT_ROOT = Path(out_root)
    _RESUME = bool(resume)


def _atomic_write_segmentation(segmentation: np.ndarray, reference, destination: Path) -> None:
    import SimpleITK as sitk

    image = sitk.GetImageFromArray(np.asarray(segmentation, dtype=np.uint8))
    image.CopyInformation(reference)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp.nii.gz")
    try:
        sitk.WriteImage(image, str(temporary), True)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _process_case(case_id: str) -> dict[str, int | str]:
    import SimpleITK as sitk

    assert _ENSEMBLE_ROOT is not None and _OUT_ROOT is not None
    reference_path = _ENSEMBLE_ROOT / f"{case_id}.nii.gz"
    raw_output = _version_path(CORE_VERSION_IDS[0], case_id)
    if not (_RESUME and raw_output.exists()):
        _link_or_copy(reference_path, raw_output)

    core_outputs = {version_id: _version_path(version_id, case_id) for version_id in CORE_VERSION_IDS[1:]}
    need_core = not (_RESUME and all(path.exists() for path in core_outputs.values()))
    reference = sitk.ReadImage(str(reference_path))
    if need_core:
        probabilities = load_probabilities(_ENSEMBLE_ROOT / f"{case_id}.npz")
        if probabilities.shape[1:] != sitk.GetArrayViewFromImage(reference).shape:
            raise ValueError(f"{case_id}: probability/reference shape mismatch")
        spacing_zyx = tuple(float(value) for value in reference.GetSpacing())[::-1]
        versions = build_core_versions(probabilities, spacing_zyx)
        for version_id, segmentation in versions.items():
            destination = core_outputs[version_id]
            if _RESUME and destination.exists():
                continue
            _atomic_write_segmentation(segmentation, reference, destination)

    rescued_components = rescued_voxels = 0
    if _FOLD_ROOTS:
        rescue_output = _version_path(RESCUE_VERSION_ID, case_id)
        if not (_RESUME and rescue_output.exists()):
            base_image = sitk.ReadImage(str(core_outputs[CORE_VERSION_IDS[5]]))
            base_masks = segmentation_to_masks(sitk.GetArrayFromImage(base_image))
            et_sources: list[np.ndarray] = []
            for root in _FOLD_ROOTS:
                probabilities = load_probabilities(root / f"{case_id}.npz")
                et_sources.append(probabilities[CHANNELS.index("et")].copy())
                del probabilities
            rescued, stats = rescue_et_fold_consensus(
                base_masks,
                et_sources,
                spacing_zyx=tuple(float(value) for value in reference.GetSpacing())[::-1],
                **ET_FOLD_CONSENSUS,
            )
            _atomic_write_segmentation(masks_to_segmentation(rescued), reference, rescue_output)
            rescued_components = stats["rescued_components"]
            rescued_voxels = stats["rescued_voxels"]
        else:
            base_segmentation = sitk.GetArrayFromImage(
                sitk.ReadImage(str(core_outputs[CORE_VERSION_IDS[5]]))
            )
            rescued_segmentation = sitk.GetArrayFromImage(sitk.ReadImage(str(rescue_output)))
            stats = rescue_delta_statistics(base_segmentation, rescued_segmentation)
            rescued_components = stats["rescued_components"]
            rescued_voxels = stats["rescued_voxels"]
    return {
        "case_id": case_id,
        "rescued_components": rescued_components,
        "rescued_voxels": rescued_voxels,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_versions(out_root: Path, submission_root: Path, version_ids: Sequence[str], expected: int) -> None:
    submission_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {"expected_cases": expected, "versions": {}}
    for version_id in version_ids:
        prediction_root = out_root / "candidates" / version_id / "predictions"
        files = sorted(prediction_root.glob("*.nii.gz"))
        if len(files) != expected:
            raise RuntimeError(f"{version_id}: expected {expected} predictions, found {len(files)}")
        zip_path = submission_root / f"Dataset501_BraTS2025MET_ResEncXL_{version_id}.zip"
        temporary = zip_path.with_suffix(".zip.tmp")
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in files:
                archive.write(path, arcname=path.name)
        os.replace(temporary, zip_path)
        with zipfile.ZipFile(zip_path) as archive:
            names = archive.namelist()
            bad = archive.testzip()
        if len(names) != expected or any("/" in name or not name.endswith(".nii.gz") for name in names):
            raise RuntimeError(f"{version_id}: invalid flat ZIP structure")
        if bad is not None:
            raise RuntimeError(f"{version_id}: CRC failure at {bad}")
        digest = _sha256(zip_path)
        zip_path.with_suffix(zip_path.suffix + ".sha256").write_text(
            f"{digest}  {zip_path.name}\n", encoding="utf-8"
        )
        manifest["versions"][version_id] = {
            "prediction_count": len(files),
            "zip": str(zip_path),
            "zip_entries": len(names),
            "sha256": digest,
        }
    (submission_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensemble-root", type=Path, required=True)
    parser.add_argument("--fold-root", type=Path, action="append", default=[])
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--submission-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--expected", type=int, default=179)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-cases", type=int)
    args = parser.parse_args()
    if args.fold_root and len(args.fold_root) != 5:
        parser.error("Provide either zero or exactly five --fold-root directories")
    case_ids = sorted(path.stem for path in args.ensemble_root.glob("*.npz"))
    if len(case_ids) != args.expected or len(set(case_ids)) != args.expected:
        raise RuntimeError(f"Expected {args.expected} ensemble probability maps, found {len(case_ids)}")
    if args.max_cases:
        case_ids = case_ids[: args.max_cases]
    for case_id in case_ids:
        if not (args.ensemble_root / f"{case_id}.nii.gz").is_file():
            raise FileNotFoundError(args.ensemble_root / f"{case_id}.nii.gz")
        for root in args.fold_root:
            if not (root / f"{case_id}.npz").is_file():
                raise FileNotFoundError(root / f"{case_id}.npz")

    version_ids = list(CORE_VERSION_IDS) + ([RESCUE_VERSION_ID] if args.fold_root else [])
    args.out_root.mkdir(parents=True, exist_ok=True)
    for version_id in version_ids:
        (args.out_root / "candidates" / version_id / "predictions").mkdir(parents=True, exist_ok=True)
    manifest = {
        "ensemble_root": str(args.ensemble_root),
        "fold_roots": [str(path) for path in args.fold_root],
        "case_count": len(case_ids),
        "versions": version_ids,
        "configs": {
            "component": DEFAULT_GATES,
            "high_recall": HIGH_RECALL_GATES,
            "precision": PRECISION_GATES,
            "tc_boundary": TC_BOUNDARY,
            "rc_strict": RC_STRICT,
            "et_fold_consensus": ET_FOLD_CONSENSUS,
        },
    }
    (args.out_root / "generation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"):
        os.environ[variable] = "1"
    context = mp.get_context("fork") if hasattr(os, "fork") else mp.get_context("spawn")
    rescue_totals = {"rescued_components": 0, "rescued_voxels": 0}
    with context.Pool(
        processes=max(1, args.workers),
        initializer=_init_worker,
        initargs=(str(args.ensemble_root), [str(path) for path in args.fold_root], str(args.out_root), args.resume),
        maxtasksperchild=4,
    ) as pool:
        for done, result in enumerate(pool.imap_unordered(_process_case, case_ids, chunksize=1), 1):
            for key in rescue_totals:
                rescue_totals[key] += int(result[key])
            if done % 10 == 0 or done == len(case_ids):
                print(f"completed={done}/{len(case_ids)} last={result['case_id']} rescue={rescue_totals}", flush=True)
    (args.out_root / "generation_summary.json").write_text(
        json.dumps({"processed_cases": len(case_ids), **rescue_totals}, indent=2) + "\n",
        encoding="utf-8",
    )
    if len(case_ids) == args.expected:
        package_versions(args.out_root, args.submission_root, version_ids, args.expected)


if __name__ == "__main__":
    main()
