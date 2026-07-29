"""Shared immutable configuration and case-universe validation for portfolio v3."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage


_CASE_SUFFIXES = (".nii.gz", ".npz", ".pkl")
_OOF_MODELS = ("XL", "M", "FT")
_REGIONS = ("wt", "tc", "et", "rc")
_BRATS_LABELS = {"wt": 2, "tc": 1, "et": 3, "rc": 4}


@dataclass(frozen=True)
class CaseInputs:
    case_id: str
    xl_path: Path
    m_path: Path
    ft_path: Path
    anchor_path: Path | None
    fold: int | None


@dataclass(frozen=True)
class LocalProposal:
    proposal_id: int
    bbox: tuple[slice, slice, slice]
    local_mask: np.ndarray
    volume_mm3: float
    model_peaks: dict[str, dict[str, float]]


def _region_name(channel: int) -> str:
    try:
        return _REGIONS[int(channel)]
    except (IndexError, ValueError) as error:
        raise ValueError(f"unsupported BraTS region channel: {channel}") from error


def _mask_for_region(regions: Mapping[str, np.ndarray], region: str) -> np.ndarray:
    for key in (region, region.upper()):
        if key in regions:
            return np.asarray(regions[key], dtype=bool)
    raise KeyError(f"missing region mask: {region}")


def generate_union_proposals(
    model_probabilities: Mapping[str, np.ndarray],
    channel: int,
    threshold: float,
    spacing_zyx: tuple[float, float, float],
) -> list[LocalProposal]:
    """Generate local components from the thresholded XL/M/FT union."""
    region = _region_name(channel)
    arrays = {model: np.asarray(model_probabilities[model]) for model in _OOF_MODELS}
    validate_probability_alignment(arrays)
    probability_shape = arrays["XL"].shape
    if len(probability_shape) != 4 or not 0 <= int(channel) < probability_shape[0]:
        raise ValueError(f"channel {channel} is incompatible with probability shape {probability_shape}")

    union = np.zeros(probability_shape[1:], dtype=bool)
    for probabilities in arrays.values():
        union |= probabilities[int(channel)] >= float(threshold)
    # Match the persisted LCv2 proposal universe exactly. The original
    # component_gate.propose_components uses 26-connectivity, so diagonal
    # contacts must retain the same component IDs here.
    labels, count = ndimage.label(union, structure=np.ones((3, 3, 3), dtype=np.uint8))
    objects = ndimage.find_objects(labels)
    voxel_mm3 = float(np.prod(spacing_zyx))
    proposals: list[LocalProposal] = []
    for proposal_id in range(1, count + 1):
        bbox = objects[proposal_id - 1]
        if bbox is None:
            continue
        local_labels = labels[bbox]
        local_mask = local_labels == proposal_id
        peaks = {
            model: {region: float(probabilities[int(channel)][bbox][local_mask].max())}
            for model, probabilities in arrays.items()
        }
        proposals.append(
            LocalProposal(
                proposal_id=proposal_id,
                bbox=bbox,
                local_mask=local_mask,
                volume_mm3=float(local_mask.sum()) * voxel_mm3,
                model_peaks=peaks,
            )
        )
    return proposals


def has_two_of_three_support(
    proposal: LocalProposal,
    region: str,
    threshold: float = 0.25,
) -> bool:
    """Return whether at least two source models peak above ``threshold``."""
    canonical_region = str(region).lower()
    support_count = sum(
        proposal.model_peaks.get(model, {}).get(canonical_region, float("-inf")) >= float(threshold)
        for model in _OOF_MODELS
    )
    return support_count >= 2


def _local_probability(proposal: LocalProposal, probability: np.ndarray) -> np.ndarray:
    array = np.asarray(probability)
    if array.shape == proposal.local_mask.shape:
        return array
    local = array[proposal.bbox]
    if local.shape != proposal.local_mask.shape:
        raise ValueError(
            f"proposal bbox produces shape {local.shape}, expected {proposal.local_mask.shape}"
        )
    return local


def _seeded_growth(local_mask: np.ndarray, growth_domain: np.ndarray, seed: np.ndarray) -> np.ndarray:
    if not np.any(seed):
        return np.zeros(local_mask.shape, dtype=bool)
    labels, count = ndimage.label(growth_domain, structure=ndimage.generate_binary_structure(3, 1))
    if count == 0:
        return np.zeros(local_mask.shape, dtype=bool)
    touching_ids = np.unique(labels[seed])
    touching_ids = touching_ids[touching_ids != 0]
    if touching_ids.size == 0:
        return np.zeros(local_mask.shape, dtype=bool)
    return np.isin(labels, touching_ids) & local_mask


def reconstruct_xl_core_shape(
    proposal: LocalProposal,
    xl_region_probability: np.ndarray,
) -> np.ndarray:
    """Grow XL core only through a proposal-local XL-supported domain."""
    local_xl = _local_probability(proposal, xl_region_probability)
    seed = local_xl >= 0.50
    growth_domain = proposal.local_mask & (local_xl >= 0.25)
    return _seeded_growth(proposal.local_mask, growth_domain, seed)


def reconstruct_xlft_consensus_shape(
    proposal: LocalProposal,
    xl_region_probability: np.ndarray,
    ft_region_probability: np.ndarray,
) -> np.ndarray:
    """Grow a local shape only where XL and FT agree and reach a joint seed."""
    local_xl = _local_probability(proposal, xl_region_probability)
    local_ft = _local_probability(proposal, ft_region_probability)
    seed = ((local_xl >= 0.50) & (local_ft >= 0.35)) | (
        (local_ft >= 0.50) & (local_xl >= 0.35)
    )
    growth_domain = proposal.local_mask & (local_xl >= 0.25) & (local_ft >= 0.25)
    return _seeded_growth(proposal.local_mask, growth_domain, seed)


def enforce_hierarchy(regions: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return independent RC and nested ET subset TC subset WT masks."""
    output = {region: _mask_for_region(regions, region).copy() for region in _REGIONS}
    output["tc"] |= output["et"]
    output["wt"] |= output["tc"]
    return output


def assert_anchor_protected(anchor: Mapping[str, np.ndarray], output: Mapping[str, np.ndarray]) -> None:
    """Raise if a candidate removes any voxel from an anchor region."""
    for region in ("et", "rc", "tc", "wt"):
        missing = _mask_for_region(anchor, region) & ~_mask_for_region(output, region)
        missing_count = int(missing.sum())
        if missing_count:
            raise AssertionError(
                f"anchor protection failed for {region}: missing_voxels={missing_count}"
            )


def masks_to_brats_segmentation(regions: Mapping[str, np.ndarray]) -> np.ndarray:
    """Encode hierarchy-consistent regions with the established BraTS labels."""
    fixed = enforce_hierarchy(regions)
    overlap_voxels = int((fixed["rc"] & fixed["wt"]).sum())
    if overlap_voxels:
        raise ValueError(
            f"RC/WT overlap is not representable in a BraTS label map: overlap_voxels={overlap_voxels}"
        )
    segmentation = np.zeros(fixed["wt"].shape, dtype=np.uint8)
    for region in _REGIONS:
        segmentation[fixed[region]] = _BRATS_LABELS[region]
    return segmentation


def load_config(path: Path) -> dict:
    """Read the versioned portfolio configuration as a JSON object."""
    with Path(path).open(encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"configuration must be a JSON object: {path}")
    return config


def _canonical_case_id(path: Path) -> str:
    name = path.name
    for suffix in _CASE_SUFFIXES:
        if name.endswith(suffix):
            case_id = name[: -len(suffix)]
            if case_id:
                return case_id
            break
    raise ValueError(f"unsupported case source filename: {path}")


def _case_file_index(
    root: Path, *, recursive: bool, required_suffix: str
) -> dict[str, Path]:
    if required_suffix not in _CASE_SUFFIXES:
        raise ValueError(f"unsupported required case suffix: {required_suffix}")
    if not root.is_dir():
        raise FileNotFoundError(f"case source directory is missing: {root}")
    paths = root.rglob("*") if recursive else root.iterdir()
    indexed: dict[str, Path] = {}
    for path in sorted(paths):
        if not path.is_file() or not path.name.endswith(required_suffix):
            continue
        case_id = _canonical_case_id(path)
        if case_id in indexed:
            raise RuntimeError(
                f"duplicate case source for {case_id}: {indexed[case_id]} and {path}"
            )
        indexed[case_id] = path
    return indexed


def _require_matching_case_sets(named_indexes: Mapping[str, Mapping[str, Path]]) -> set[str]:
    case_sets = {name: set(index) for name, index in named_indexes.items()}
    reference_name, reference_set = next(iter(case_sets.items()))
    mismatches = {
        name: {
            "missing": sorted(reference_set - case_set),
            "unexpected": sorted(case_set - reference_set),
        }
        for name, case_set in case_sets.items()
        if case_set != reference_set
    }
    if mismatches:
        raise RuntimeError(
            f"case sets differ from {reference_name} before intersection: {mismatches}"
        )
    return reference_set


def _require_expected_count(case_ids: set[str], expected: int, field: str) -> None:
    if len(case_ids) != int(expected):
        raise RuntimeError(
            f"{field} mismatch: found={len(case_ids)} expected={int(expected)}"
        )


def build_test_registry(config: Mapping, anchor_id: str) -> list[CaseInputs]:
    """Build an all-model, all-anchor test universe without silent intersections."""
    test_roots = config["test_roots"]
    anchor_roots = config["test_anchor_roots"]
    if anchor_id not in anchor_roots:
        raise KeyError(f"unknown test anchor: {anchor_id}")
    indexes = {
        name: _case_file_index(
            Path(test_roots[name]), recursive=False, required_suffix=".npz"
        )
        for name in _OOF_MODELS
    }
    indexes[anchor_id] = _case_file_index(
        Path(anchor_roots[anchor_id]),
        recursive=False,
        required_suffix=".nii.gz",
    )
    case_ids = _require_matching_case_sets(indexes)
    _require_expected_count(case_ids, config["expected_test_cases"], "expected_test_cases")
    return [
        CaseInputs(
            case_id=case_id,
            xl_path=indexes["XL"][case_id],
            m_path=indexes["M"][case_id],
            ft_path=indexes["FT"][case_id],
            anchor_path=indexes[anchor_id][case_id],
            fold=None,
        )
        for case_id in sorted(case_ids)
    ]


def _case_folds(config: Mapping) -> dict[str, int]:
    if "case_predictions_path" not in config:
        raise KeyError("case_predictions_path is required")
    table_path = Path(config["case_predictions_path"])
    if not table_path.is_file():
        raise FileNotFoundError(f"case-fold table is missing: {table_path}")
    case_predictions = pd.read_parquet(table_path, columns=["case_id", "fold"])
    required = {"case_id", "fold"}
    missing = required.difference(case_predictions.columns)
    if missing:
        raise KeyError(f"case-fold table lacks columns: {sorted(missing)}")
    counts = case_predictions.groupby("case_id", sort=False)["fold"].nunique()
    if (counts != 1).any():
        bad_cases = counts[counts != 1].index.tolist()[:5]
        raise RuntimeError(f"invalid case fold mappings: {bad_cases}")
    return {
        str(case_id): int(group["fold"].iloc[0])
        for case_id, group in case_predictions.groupby("case_id", sort=False)
    }


def build_oof_registry(config: Mapping) -> list[CaseInputs]:
    """Build the v1/v2-compatible OOF universe and its held-out fold mapping."""
    case_folds = _case_folds(config)
    expected_ids = set(case_folds)
    indexes = {
        name: _case_file_index(
            Path(config["oof_roots"][name]),
            recursive=True,
            required_suffix=".npz",
        )
        for name in _OOF_MODELS
    }
    for name, index in indexes.items():
        if set(index) != expected_ids:
            missing = sorted(expected_ids - set(index))
            unexpected = sorted(set(index) - expected_ids)
            raise RuntimeError(
                f"case sets differ from LCv2 case-fold mapping before intersection: "
                f"{name} missing={missing[:5]} unexpected={unexpected[:5]}"
            )
    _require_matching_case_sets(indexes)
    _require_expected_count(expected_ids, config["expected_oof_cases"], "expected_oof_cases")
    registry = []
    for case_id in sorted(expected_ids):
        fold = case_folds[case_id]
        expected_suffix = Path(f"fold_{fold}") / "validation" / f"{case_id}.npz"
        paths = {name: index[case_id] for name, index in indexes.items()}
        wrong_folds = [name for name, path in paths.items() if not str(path).endswith(str(expected_suffix))]
        if wrong_folds:
            raise RuntimeError(
                f"{case_id}: OOF files are not in the assigned fold {fold}: {wrong_folds}"
            )
        registry.append(
            CaseInputs(
                case_id=case_id,
                xl_path=paths["XL"],
                m_path=paths["M"],
                ft_path=paths["FT"],
                anchor_path=None,
                fold=fold,
            )
        )
    return registry


def load_probability(path: Path) -> np.ndarray:
    """Load a finite four-region nnU-Net probability tensor from an NPZ file."""
    with np.load(Path(path), allow_pickle=False) as archive:
        if "probabilities" not in archive:
            raise KeyError(f"{path}: missing nnU-Net NPZ key 'probabilities'")
        array = np.asarray(archive["probabilities"])
    if array.ndim != 4:
        raise ValueError(f"{path}: probabilities must have 4 dimensions, got {array.ndim}")
    if array.shape[0] < 4:
        raise ValueError(
            f"{path}: probabilities must have at least 4 channels, got {array.shape[0]}"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{path}: probabilities contain non-finite values")
    return array


def validate_probability_alignment(arrays: Mapping[str, np.ndarray]) -> None:
    """Require identical channel and spatial shapes for every probability source."""
    if not arrays:
        raise ValueError("no probability arrays were supplied")
    reference_name, reference = next(iter(arrays.items()))
    reference_shape = tuple(reference.shape)
    mismatches = {
        name: tuple(array.shape)
        for name, array in arrays.items()
        if tuple(array.shape) != reference_shape
    }
    if mismatches:
        raise ValueError(
            f"probability shapes differ from {reference_name}={reference_shape}: {mismatches}"
        )


def validate_nifti_alignment(reference: nib.Nifti1Image, candidate: nib.Nifti1Image) -> None:
    """Require matching NIfTI image dimensions, world geometry, and voxel spacing."""
    if tuple(reference.shape) != tuple(candidate.shape):
        raise ValueError(
            f"NIfTI shapes differ: reference={reference.shape} candidate={candidate.shape}"
        )
    if not np.allclose(reference.affine, candidate.affine):
        raise ValueError("NIfTI affines differ")
    reference_spacing = np.asarray(reference.header.get_zooms()[:3], dtype=float)
    candidate_spacing = np.asarray(candidate.header.get_zooms()[:3], dtype=float)
    if not np.allclose(reference_spacing, candidate_spacing):
        raise ValueError(
            "NIfTI spacing differs: "
            f"reference={tuple(reference_spacing)} candidate={tuple(candidate_spacing)}"
        )
