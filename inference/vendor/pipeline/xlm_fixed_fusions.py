#!/usr/bin/env python
"""Generate frozen ResEncXL/ResEncM-family BraTS-MET probability fusions."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy import special

try:
    from .build_xl_fixed_postprocess import RC_STRICT, TC_BOUNDARY, apply_rc_gate
    from .mft_regionwise_pipeline import (
        CHANNELS,
        CURRENT_FT_WEIGHTS,
        DEFAULT_GATES,
        apply_tc_boundary_completion,
        component_conf_masks,
        compose_label_hybrid,
        masks_to_segmentation,
        structured_probability_masks,
        threshold_component_masks,
    )
except ImportError:  # direct execution with the vendor directory on PYTHONPATH
    from build_xl_fixed_postprocess import RC_STRICT, TC_BOUNDARY, apply_rc_gate
    from mft_regionwise_pipeline import (
        CHANNELS,
        CURRENT_FT_WEIGHTS,
        DEFAULT_GATES,
        apply_tc_boundary_completion,
        component_conf_masks,
        compose_label_hybrid,
        masks_to_segmentation,
        structured_probability_masks,
        threshold_component_masks,
    )


EPS = 1e-5
CANDIDATE_IDS = (
    "XF01_XL_M_equal_logit_component",
    "XF02_XL_FT_equal_logit_rawgate",
    "XF03_XL_FT_equal_logit_component",
    "XF04_XL_M_FT_equal_logit_component",
    "XF05_XL_OS_equal_logit_component",
    "XF06_XL_SYN_equal_logit_component",
    "XF07_XL_MFTcal_equal_logit_component",
    "XF08_XL_M_FT_OS_equal_logit_component",
    "XF09_XLM_structured_label",
    "XF10_XLM_structured_probability",
    "XF11_XLM_structured_probability_t040b20",
    "XF12_XLM_structured_probability_V2_strict",
    "XF13_XL_MsideStructured_equal_component",
    "XF14_XL_MsideStructured_V2_strict",
)

SIMPLE_THRESHOLDS = {region: 0.50 for region in CHANNELS}
SIMPLE_MIN_VOLUME = dict(DEFAULT_GATES["min_volume_mm3"])

CANDIDATE_SOURCES: dict[str, tuple[str, ...]] = {
    "XF01_XL_M_equal_logit_component": ("XL", "M"),
    "XF02_XL_FT_equal_logit_rawgate": ("XL", "FT"),
    "XF03_XL_FT_equal_logit_component": ("XL", "FT"),
    "XF04_XL_M_FT_equal_logit_component": ("XL", "M", "FT"),
    "XF05_XL_OS_equal_logit_component": ("XL", "OS"),
    "XF06_XL_SYN_equal_logit_component": ("XL", "SYN"),
    "XF07_XL_MFTcal_equal_logit_component": ("XL", "M", "FT"),
    "XF08_XL_M_FT_OS_equal_logit_component": ("XL", "M", "FT", "OS"),
    "XF09_XLM_structured_label": ("XL", "FT"),
    "XF10_XLM_structured_probability": ("XL", "FT"),
    "XF11_XLM_structured_probability_t040b20": ("XL", "FT"),
    "XF12_XLM_structured_probability_V2_strict": ("XL", "FT"),
    "XF13_XL_MsideStructured_equal_component": ("XL", "M", "FT"),
    "XF14_XL_MsideStructured_V2_strict": ("XL", "M", "FT"),
}

_CANDIDATE_GROUPS = (
    ("XF01_XL_M_equal_logit_component",),
    ("XF02_XL_FT_equal_logit_rawgate", "XF03_XL_FT_equal_logit_component"),
    ("XF04_XL_M_FT_equal_logit_component",),
    ("XF05_XL_OS_equal_logit_component",),
    ("XF06_XL_SYN_equal_logit_component",),
    ("XF07_XL_MFTcal_equal_logit_component",),
    ("XF08_XL_M_FT_OS_equal_logit_component",),
    (
        "XF09_XLM_structured_label",
        "XF10_XLM_structured_probability",
        "XF11_XLM_structured_probability_t040b20",
        "XF12_XLM_structured_probability_V2_strict",
    ),
    ("XF13_XL_MsideStructured_equal_component", "XF14_XL_MsideStructured_V2_strict"),
)


def parse_memory_stat(text: str) -> dict[str, int]:
    parsed: dict[str, int] = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) == 2:
            parsed[fields[0]] = int(fields[1])
    return parsed


def required_sources(candidate_ids: Sequence[str]) -> tuple[str, ...]:
    unknown = sorted(set(candidate_ids) - set(CANDIDATE_IDS))
    if unknown:
        raise ValueError(f"Unknown candidate IDs: {unknown}")
    needed = {source for candidate_id in candidate_ids for source in CANDIDATE_SOURCES[candidate_id]}
    return tuple(source for source in ("XL", "M", "FT", "OS", "SYN") if source in needed)


def candidate_groups(candidate_ids: Sequence[str]) -> tuple[tuple[str, ...], ...]:
    requested = set(candidate_ids)
    groups: list[tuple[str, ...]] = []
    for group in _CANDIDATE_GROUPS:
        selected = tuple(candidate_id for candidate_id in group if candidate_id in requested)
        if selected:
            groups.append(selected)
    flattened = {candidate_id for group in groups for candidate_id in group}
    if flattened != requested:
        raise ValueError(f"Unknown candidate IDs: {sorted(requested - flattened)}")
    return tuple(groups)


def select_shard(case_ids: Sequence[str], num_shards: int, shard_index: int) -> list[str]:
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")
    return list(case_ids[shard_index::num_shards])


def load_probabilities(path: Path) -> np.ndarray:
    with np.load(path) as archive:
        key = "probabilities" if "probabilities" in archive else "softmax"
        probabilities = np.asarray(archive[key], dtype=np.float32)
    if probabilities.ndim != 4 or probabilities.shape[0] != len(CHANNELS):
        raise ValueError(f"{path}: expected WT/TC/ET/RC probabilities, got {probabilities.shape}")
    if not np.isfinite(probabilities).all():
        raise ValueError(f"{path}: probabilities contain NaN or Inf")
    return probabilities


def validate_source_alignment(source_roots: Mapping[str, Path], expected: int) -> list[str]:
    if "XL" not in source_roots:
        raise ValueError("XL source is required for image geometry")
    case_sets: dict[str, set[str]] = {}
    for name, root in source_roots.items():
        if not root.is_dir():
            raise FileNotFoundError(root)
        case_sets[name] = {path.stem for path in root.glob("*.npz")}
    xl_ids = case_sets["XL"]
    if len(xl_ids) != expected:
        raise RuntimeError(f"XL: expected {expected} cases, found {len(xl_ids)}")
    for name, ids in case_sets.items():
        if ids != xl_ids:
            raise RuntimeError(
                f"{name}/XL case sets differ: missing={sorted(xl_ids - ids)[:5]} extra={sorted(ids - xl_ids)[:5]}"
            )
    for case_id in xl_ids:
        if not (source_roots["XL"] / f"{case_id}.nii.gz").is_file():
            raise FileNotFoundError(source_roots["XL"] / f"{case_id}.nii.gz")
    return sorted(xl_ids)


def _npz_uncompressed_bytes(path: Path) -> int:
    with zipfile.ZipFile(path) as archive:
        return sum(info.file_size for info in archive.infolist())


def estimate_peak_anon_bytes(
    current_anon_bytes: int,
    source_uncompressed_bytes: Sequence[int],
    safety_margin_bytes: int,
) -> int:
    if not source_uncompressed_bytes:
        return int(current_anon_bytes) + int(safety_margin_bytes)
    largest = max(int(value) for value in source_uncompressed_bytes)
    # Fused probability workspace, component masks, and up to four uint8 structured outputs.
    workspace = 3 * largest
    return int(current_anon_bytes) + sum(int(value) for value in source_uncompressed_bytes) + workspace + int(
        safety_margin_bytes
    )


def can_reserve_memory(
    current_anon_bytes: int,
    existing_reservation_bytes: int,
    requested_reservation_bytes: int,
    safety_margin_bytes: int,
    memory_max_bytes: int,
) -> bool:
    return (
        int(current_anon_bytes)
        + int(existing_reservation_bytes)
        + int(requested_reservation_bytes)
        + int(safety_margin_bytes)
        <= int(memory_max_bytes)
    )


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_memory(
    paths: Sequence[Path],
    memory_stat_path: Path,
    memory_max_bytes: int,
    safety_margin_bytes: int,
    poll_seconds: float,
    reservation_path: Path,
) -> dict[str, int]:
    import fcntl

    source_bytes = tuple(_npz_uncompressed_bytes(path) for path in paths)
    requested = sum(source_bytes) + (3 * max(source_bytes) if source_bytes else 0)
    reservation_path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        with reservation_path.open("a+", encoding="utf-8") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            stream.seek(0)
            try:
                records = json.loads(stream.read() or "{}")
            except json.JSONDecodeError:
                records = {}
            records = {
                key: value
                for key, value in records.items()
                if str(key).isdigit() and _pid_is_alive(int(key))
            }
            existing = sum(int(value["bytes"]) for value in records.values())
            memory = parse_memory_stat(memory_stat_path.read_text(encoding="utf-8"))
            current_anon = int(memory.get("anon", 0))
            if can_reserve_memory(
                current_anon,
                existing,
                requested,
                safety_margin_bytes,
                memory_max_bytes,
            ):
                records[str(os.getpid())] = {"bytes": requested, "created_at": time.time()}
                stream.seek(0)
                stream.truncate()
                stream.write(json.dumps(records, sort_keys=True))
                stream.flush()
                os.fsync(stream.fileno())
                return {
                    "anon_before": current_anon,
                    "estimated_peak_anon": current_anon + requested + safety_margin_bytes,
                    "source_uncompressed_bytes": sum(source_bytes),
                    "reserved_bytes": requested,
                }
        print(
            f"[memory-wait] anon={current_anon} existing_reserved={existing} "
            f"requested={requested} limit={memory_max_bytes}",
            flush=True,
        )
        time.sleep(max(1.0, poll_seconds))


def _release_memory_reservation(reservation_path: Path) -> None:
    import fcntl

    if not reservation_path.exists():
        return
    with reservation_path.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.seek(0)
        try:
            records = json.loads(stream.read() or "{}")
        except json.JSONDecodeError:
            records = {}
        records.pop(str(os.getpid()), None)
        stream.seek(0)
        stream.truncate()
        stream.write(json.dumps(records, sort_keys=True))
        stream.flush()
        os.fsync(stream.fileno())


def _weights_by_region(
    weights: Sequence[float] | Mapping[str, Sequence[float]],
    source_count: int,
) -> dict[str, tuple[float, ...]]:
    if isinstance(weights, Mapping):
        missing = set(CHANNELS) - set(weights)
        if missing:
            raise ValueError(f"Missing region weights: {sorted(missing)}")
        result = {region: tuple(float(value) for value in weights[region]) for region in CHANNELS}
    else:
        shared = tuple(float(value) for value in weights)
        result = {region: shared for region in CHANNELS}
    for region, values in result.items():
        if len(values) != source_count:
            raise ValueError(f"{region}: expected {source_count} weights, got {len(values)}")
        if any(value < 0.0 for value in values):
            raise ValueError(f"{region}: negative fusion weight")
        if not np.isclose(sum(values), 1.0, rtol=0, atol=1e-8):
            raise ValueError(f"{region}: weights must sum to one, got {sum(values)}")
    return result


def fuse_logits_many(
    probability_arrays: Sequence[np.ndarray],
    weights: Sequence[float] | Mapping[str, Sequence[float]],
) -> np.ndarray:
    """Fuse aligned region probabilities with a memory-bounded logit average."""
    if not probability_arrays:
        raise ValueError("At least one probability array is required")
    arrays = tuple(np.asarray(array, dtype=np.float32) for array in probability_arrays)
    shape = arrays[0].shape
    if len(shape) != 4 or shape[0] != len(CHANNELS):
        raise ValueError(f"Expected four region channels, got {shape}")
    for array in arrays:
        if array.shape != shape:
            raise ValueError(f"Probability shape mismatch: {array.shape} != {shape}")
        if not np.isfinite(array).all():
            raise ValueError("Probabilities contain NaN or Inf")
    region_weights = _weights_by_region(weights, len(arrays))
    fused = np.empty(shape, dtype=np.float32)
    temporary = np.empty(shape[1:], dtype=np.float32)
    for channel, region in enumerate(CHANNELS):
        fused[channel].fill(0.0)
        for array, weight in zip(arrays, region_weights[region]):
            if weight == 0.0:
                continue
            np.clip(array[channel], EPS, 1.0 - EPS, out=temporary)
            special.logit(temporary, out=temporary)
            temporary *= weight
            fused[channel] += temporary
        special.expit(fused[channel], out=fused[channel])
    return fused


def _require_sources(sources: Mapping[str, np.ndarray], names: Sequence[str]) -> tuple[np.ndarray, ...]:
    missing = [name for name in names if name not in sources]
    if missing:
        raise ValueError(f"Missing probability sources: {missing}")
    return tuple(np.asarray(sources[name], dtype=np.float32) for name in names)


def _xl_ft_calibrated_weights() -> dict[str, tuple[float, float]]:
    return {
        region: (1.0 - float(CURRENT_FT_WEIGHTS[region]), float(CURRENT_FT_WEIGHTS[region]))
        for region in CHANNELS
    }


def _xl_mftcal_equal_weights() -> dict[str, tuple[float, float, float]]:
    return {
        region: (
            0.50,
            0.50 * (1.0 - float(CURRENT_FT_WEIGHTS[region])),
            0.50 * float(CURRENT_FT_WEIGHTS[region]),
        )
        for region in CHANNELS
    }


def _xl_mside_structured_weights() -> dict[str, tuple[float, float, float]]:
    output: dict[str, tuple[float, float, float]] = {}
    for region in CHANNELS:
        if region in {"wt", "tc"}:
            output[region] = (0.50, 0.0, 0.50)
        else:
            ft_weight = float(CURRENT_FT_WEIGHTS[region])
            output[region] = (0.50, 0.50 * (1.0 - ft_weight), 0.50 * ft_weight)
    return output


def build_candidate_segmentations(
    sources: Mapping[str, np.ndarray],
    spacing_zyx: tuple[float, float, float],
    candidate_ids: Sequence[str] = CANDIDATE_IDS,
) -> dict[str, np.ndarray]:
    requested = tuple(candidate_ids)
    unknown = sorted(set(requested) - set(CANDIDATE_IDS))
    if unknown:
        raise ValueError(f"Unknown candidate IDs: {unknown}")
    if len(requested) != len(set(requested)):
        raise ValueError("Duplicate candidate IDs")
    outputs: dict[str, np.ndarray] = {}

    if "XF01_XL_M_equal_logit_component" in requested:
        fused = fuse_logits_many(_require_sources(sources, ("XL", "M")), (0.5, 0.5))
        outputs["XF01_XL_M_equal_logit_component"] = masks_to_segmentation(
            component_conf_masks(fused, spacing_zyx, DEFAULT_GATES)
        )

    equal_xl_ft_ids = {
        "XF02_XL_FT_equal_logit_rawgate",
        "XF03_XL_FT_equal_logit_component",
    }
    if equal_xl_ft_ids.intersection(requested):
        fused = fuse_logits_many(_require_sources(sources, ("XL", "FT")), (0.5, 0.5))
        if "XF02_XL_FT_equal_logit_rawgate" in requested:
            masks = threshold_component_masks(fused, spacing_zyx, SIMPLE_THRESHOLDS, SIMPLE_MIN_VOLUME)
            outputs["XF02_XL_FT_equal_logit_rawgate"] = masks_to_segmentation(masks)
        if "XF03_XL_FT_equal_logit_component" in requested:
            outputs["XF03_XL_FT_equal_logit_component"] = masks_to_segmentation(
                component_conf_masks(fused, spacing_zyx, DEFAULT_GATES)
            )

    if "XF04_XL_M_FT_equal_logit_component" in requested:
        fused = fuse_logits_many(_require_sources(sources, ("XL", "M", "FT")), (1 / 3, 1 / 3, 1 / 3))
        outputs["XF04_XL_M_FT_equal_logit_component"] = masks_to_segmentation(
            component_conf_masks(fused, spacing_zyx, DEFAULT_GATES)
        )

    if "XF05_XL_OS_equal_logit_component" in requested:
        fused = fuse_logits_many(_require_sources(sources, ("XL", "OS")), (0.5, 0.5))
        outputs["XF05_XL_OS_equal_logit_component"] = masks_to_segmentation(
            component_conf_masks(fused, spacing_zyx, DEFAULT_GATES)
        )

    if "XF06_XL_SYN_equal_logit_component" in requested:
        fused = fuse_logits_many(_require_sources(sources, ("XL", "SYN")), (0.5, 0.5))
        outputs["XF06_XL_SYN_equal_logit_component"] = masks_to_segmentation(
            component_conf_masks(fused, spacing_zyx, DEFAULT_GATES)
        )

    if "XF07_XL_MFTcal_equal_logit_component" in requested:
        fused = fuse_logits_many(
            _require_sources(sources, ("XL", "M", "FT")),
            _xl_mftcal_equal_weights(),
        )
        outputs["XF07_XL_MFTcal_equal_logit_component"] = masks_to_segmentation(
            component_conf_masks(fused, spacing_zyx, DEFAULT_GATES)
        )

    if "XF08_XL_M_FT_OS_equal_logit_component" in requested:
        fused = fuse_logits_many(
            _require_sources(sources, ("XL", "M", "FT", "OS")),
            (0.25, 0.25, 0.25, 0.25),
        )
        outputs["XF08_XL_M_FT_OS_equal_logit_component"] = masks_to_segmentation(
            component_conf_masks(fused, spacing_zyx, DEFAULT_GATES)
        )

    structured_ids = {
        "XF09_XLM_structured_label",
        "XF10_XLM_structured_probability",
        "XF11_XLM_structured_probability_t040b20",
        "XF12_XLM_structured_probability_V2_strict",
    }
    if structured_ids.intersection(requested):
        xl, ft = _require_sources(sources, ("XL", "FT"))
        fused = fuse_logits_many((xl, ft), _xl_ft_calibrated_weights())
        fused_masks = component_conf_masks(fused, spacing_zyx, DEFAULT_GATES)
        if "XF09_XLM_structured_label" in requested:
            ft_masks = component_conf_masks(ft, spacing_zyx, DEFAULT_GATES)
            outputs["XF09_XLM_structured_label"] = compose_label_hybrid(fused_masks, ft_masks)
        probability_masks = None
        if structured_ids.intersection(requested) - {"XF09_XLM_structured_label"}:
            probability_masks = structured_probability_masks(fused, ft, spacing_zyx, DEFAULT_GATES)
        if "XF10_XLM_structured_probability" in requested:
            assert probability_masks is not None
            outputs["XF10_XLM_structured_probability"] = masks_to_segmentation(probability_masks)
        if {
            "XF11_XLM_structured_probability_t040b20",
            "XF12_XLM_structured_probability_V2_strict",
        }.intersection(requested):
            assert probability_masks is not None
            tc_score = np.maximum(fused[CHANNELS.index("tc")], fused[CHANNELS.index("et")])
            boundary_masks, _ = apply_tc_boundary_completion(probability_masks, tc_score, TC_BOUNDARY)
            if "XF11_XLM_structured_probability_t040b20" in requested:
                outputs["XF11_XLM_structured_probability_t040b20"] = masks_to_segmentation(boundary_masks)
            if "XF12_XLM_structured_probability_V2_strict" in requested:
                strict_masks = apply_rc_gate(
                    boundary_masks,
                    fused[CHANNELS.index("rc")],
                    spacing_zyx,
                    **RC_STRICT,
                )
                outputs["XF12_XLM_structured_probability_V2_strict"] = masks_to_segmentation(strict_masks)

    mside_ids = {
        "XF13_XL_MsideStructured_equal_component",
        "XF14_XL_MsideStructured_V2_strict",
    }
    if mside_ids.intersection(requested):
        fused = fuse_logits_many(
            _require_sources(sources, ("XL", "M", "FT")),
            _xl_mside_structured_weights(),
        )
        masks = component_conf_masks(fused, spacing_zyx, DEFAULT_GATES)
        if "XF13_XL_MsideStructured_equal_component" in requested:
            outputs["XF13_XL_MsideStructured_equal_component"] = masks_to_segmentation(masks)
        if "XF14_XL_MsideStructured_V2_strict" in requested:
            tc_score = np.maximum(fused[CHANNELS.index("tc")], fused[CHANNELS.index("et")])
            masks, _ = apply_tc_boundary_completion(masks, tc_score, TC_BOUNDARY)
            masks = apply_rc_gate(
                masks,
                fused[CHANNELS.index("rc")],
                spacing_zyx,
                **RC_STRICT,
            )
            outputs["XF14_XL_MsideStructured_V2_strict"] = masks_to_segmentation(masks)

    if set(outputs) != set(requested):
        raise RuntimeError(f"Missing generated candidates: {sorted(set(requested) - set(outputs))}")
    return {candidate_id: outputs[candidate_id] for candidate_id in requested}


def _parse_source(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--source must be NAME=PATH")
    name, raw_path = value.split("=", 1)
    if name not in {"XL", "M", "FT", "OS", "SYN"}:
        raise argparse.ArgumentTypeError(f"Unknown source name: {name}")
    if not raw_path:
        raise argparse.ArgumentTypeError("Source path is empty")
    return name, Path(raw_path)


def _atomic_write_segmentation(segmentation: np.ndarray, reference: Any, destination: Path) -> None:
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


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _script_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _case_set_sha256(case_ids: Sequence[str]) -> str:
    payload = "\n".join(case_ids).encode("utf-8") + b"\n"
    return hashlib.sha256(payload).hexdigest()


def _update_manifest(
    out_root: Path,
    source_roots: Mapping[str, Path],
    case_ids: Sequence[str],
    requested: Sequence[str],
    memory_max_bytes: int,
    safety_margin_bytes: int,
) -> None:
    manifest_path = out_root / "generation_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {
            "expected_cases": len(case_ids),
            "case_set_sha256": _case_set_sha256(case_ids),
            "sources": {},
            "candidates": [],
            "fixed_configuration": {
                "component_gates": DEFAULT_GATES,
                "current_ft_weights": CURRENT_FT_WEIGHTS,
                "tc_boundary": TC_BOUNDARY,
                "rc_strict": RC_STRICT,
            },
        }
    if manifest.get("case_set_sha256") != _case_set_sha256(case_ids):
        raise RuntimeError("Existing output manifest uses a different case set")
    for name, root in source_roots.items():
        existing = manifest["sources"].get(name)
        if existing is not None and existing != str(root):
            raise RuntimeError(f"Existing manifest source mismatch for {name}: {existing} != {root}")
        manifest["sources"][name] = str(root)
    manifest["candidates"] = [candidate_id for candidate_id in CANDIDATE_IDS if candidate_id in set(manifest["candidates"]) | set(requested)]
    manifest["script_sha256"] = _script_sha256()
    manifest["memory_policy"] = {
        "memory_max_bytes": memory_max_bytes,
        "safety_margin_bytes": safety_margin_bytes,
        "workers": 1,
        "uses_anonymous_memory_estimate": True,
    }
    _atomic_write_json(manifest_path, manifest)


def _run_generation(args: argparse.Namespace) -> None:
    import SimpleITK as sitk

    source_pairs = [_parse_source(value) for value in args.source]
    source_roots = dict(source_pairs)
    if len(source_roots) != len(source_pairs):
        raise ValueError("Duplicate --source name")
    requested = tuple(args.candidate or CANDIDATE_IDS)
    if len(requested) != len(set(requested)):
        raise ValueError("Duplicate --candidate")
    needed = required_sources(requested)
    missing_sources = [name for name in needed if name not in source_roots]
    if missing_sources:
        raise ValueError(f"Missing --source arguments for {missing_sources}")
    source_roots = {name: source_roots[name] for name in needed}
    case_ids = validate_source_alignment(source_roots, args.expected)
    selected_case_ids = select_shard(case_ids, args.num_shards, args.shard_index)
    if args.max_cases:
        selected_case_ids = selected_case_ids[: args.max_cases]
    args.out_root.mkdir(parents=True, exist_ok=True)
    for candidate_id in requested:
        (args.out_root / "candidates" / candidate_id / "predictions").mkdir(parents=True, exist_ok=True)

    memory_max_bytes = int(args.memory_limit_gib * 1024**3)
    safety_margin_bytes = int(args.memory_safety_gib * 1024**3)
    _update_manifest(
        args.out_root,
        source_roots,
        case_ids,
        requested,
        memory_max_bytes,
        safety_margin_bytes,
    )

    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS",
    ):
        os.environ[variable] = "1"

    written = skipped = 0
    progress_path = args.out_root / "progress.jsonl"
    started = time.time()
    for index, case_id in enumerate(selected_case_ids, 1):
        output_paths = {
            candidate_id: args.out_root / "candidates" / candidate_id / "predictions" / f"{case_id}.nii.gz"
            for candidate_id in requested
        }
        missing = tuple(
            candidate_id
            for candidate_id, path in output_paths.items()
            if not (args.resume and path.is_file())
        )
        if not missing:
            skipped += len(requested)
            print(f"completed={index}/{len(selected_case_ids)} case={case_id} written=0 skipped={len(requested)}", flush=True)
            continue

        case_sources = required_sources(missing)
        paths = [source_roots[name] / f"{case_id}.npz" for name in case_sources]
        memory_record = _wait_for_memory(
            paths,
            args.memory_stat,
            memory_max_bytes,
            safety_margin_bytes,
            args.memory_poll_seconds,
            args.memory_reservations,
        )
        arrays = {name: load_probabilities(source_roots[name] / f"{case_id}.npz") for name in case_sources}
        shapes = {array.shape for array in arrays.values()}
        if len(shapes) != 1:
            raise ValueError(f"{case_id}: source probability shapes differ: {sorted(shapes)}")
        reference_path = source_roots["XL"] / f"{case_id}.nii.gz"
        reference = sitk.ReadImage(str(reference_path))
        reference_shape = tuple(int(value) for value in sitk.GetArrayViewFromImage(reference).shape)
        probability_shape = next(iter(shapes))[1:]
        if probability_shape != reference_shape:
            raise ValueError(f"{case_id}: probability/reference shape mismatch {probability_shape} != {reference_shape}")
        spacing_zyx = tuple(float(value) for value in reference.GetSpacing())[::-1]

        case_written = 0
        for group in candidate_groups(missing):
            outputs = build_candidate_segmentations(arrays, spacing_zyx, group)
            for candidate_id, segmentation in outputs.items():
                _atomic_write_segmentation(segmentation, reference, output_paths[candidate_id])
                case_written += 1
            del outputs
        written += case_written
        skipped += len(requested) - len(missing)
        progress = {
            "case_id": case_id,
            "index": index,
            "case_count": len(selected_case_ids),
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "written": case_written,
            "skipped": len(requested) - len(missing),
            "elapsed_seconds": time.time() - started,
            **memory_record,
        }
        with progress_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(progress, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(
            f"completed={index}/{len(selected_case_ids)} case={case_id} written={case_written} "
            f"skipped={len(requested) - len(missing)} estimated_peak={memory_record['estimated_peak_anon']}",
            flush=True,
        )
        del arrays
        _release_memory_reservation(args.memory_reservations)

    summary_key = hashlib.sha256("\n".join(requested).encode("utf-8")).hexdigest()[:12]
    summary = {
        "selected_case_count": len(selected_case_ids),
        "expected_case_count": len(case_ids),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "candidates": list(requested),
        "written": written,
        "skipped": skipped,
        "elapsed_seconds": time.time() - started,
    }
    _atomic_write_json(
        args.out_root / f"generation_summary_{summary_key}_shard{args.shard_index:02d}of{args.num_shards:02d}.json",
        summary,
    )
    print(json.dumps(summary, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True, help="NAME=PATH")
    parser.add_argument("--candidate", action="append", choices=CANDIDATE_IDS)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--expected", type=int, default=179)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--memory-stat", type=Path, default=Path("/sys/fs/cgroup/memory.stat"))
    parser.add_argument("--memory-limit-gib", type=float, default=28.0)
    parser.add_argument("--memory-safety-gib", type=float, default=3.0)
    parser.add_argument("--memory-poll-seconds", type=float, default=20.0)
    parser.add_argument(
        "--memory-reservations",
        type=Path,
        default=Path("/tmp/xlm_fixed_fusions_memory_reservations.json"),
    )
    args = parser.parse_args()
    _run_generation(args)


if __name__ == "__main__":
    main()
