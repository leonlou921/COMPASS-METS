"""Build final N03 UTILITY_V4 segmentations from M/XL/FT probabilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import nibabel as nib
import numpy as np

from compass_mets.inference.input_contract import discover_cases
from compass_mets.inference.output_contract import (
    publish_segmentation,
    validate_flat_output_set,
)
from compass_mets.inference.predict import (
    MODEL_SEQUENCE,
    _probability_array,
    validate_probability_directory,
)


def label_zyx_to_reference_xyz(
    label_zyx: np.ndarray,
    reference: nib.Nifti1Image,
) -> np.ndarray:
    """Convert the postprocess array contract back to NIfTI voxel order."""
    label = np.asarray(label_zyx)
    expected_zyx = tuple(int(value) for value in reference.shape[::-1])
    if tuple(label.shape) != expected_zyx:
        raise ValueError(
            f"ZYX label shape {label.shape} does not match reference "
            f"XYZ shape {reference.shape}"
        )
    return label.transpose(2, 1, 0)


def _load_learned_bundles(
    *,
    lcv1_bundle: Path,
    lcv2_bundle: Path,
    rgv3_bundle: Path,
    utility_existence_model: Path,
    utility_geometry_model: Path,
    utility_feature_names: Path,
) -> dict[str, Any]:
    feature_names = json.loads(
        Path(utility_feature_names).read_text(encoding="utf-8")
    )
    if (
        not isinstance(feature_names, list)
        or not feature_names
        or not all(isinstance(value, str) and value for value in feature_names)
    ):
        raise ValueError("utility-v4 feature names must be a non-empty string list")
    return {
        "lcv1_case": joblib.load(lcv1_bundle),
        "lcv2_component": joblib.load(lcv2_bundle),
        "rgv3_et": joblib.load(rgv3_bundle),
        "utility_v4_existence": joblib.load(utility_existence_model),
        "utility_v4_geometry": joblib.load(utility_geometry_model),
        "utility_v4_feature_names": feature_names,
    }


def run_pipeline(
    *,
    input_root: Path,
    output_root: Path,
    probability_root: Path,
    lcv1_bundle: Path,
    lcv2_bundle: Path,
    rgv3_bundle: Path,
    utility_existence_model: Path,
    utility_geometry_model: Path,
    utility_feature_names: Path,
) -> dict[str, Any]:
    """Apply the frozen final chain and publish a segmentation directory."""
    from compass_mets.postprocessing.final import build_n03_final

    cases = discover_cases(Path(input_root))
    case_ids = {case.case_id for case in cases}
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    if any(output_root.iterdir()):
        raise ValueError(f"output root must be empty before inference: {output_root}")

    bundles = _load_learned_bundles(
        lcv1_bundle=lcv1_bundle,
        lcv2_bundle=lcv2_bundle,
        rgv3_bundle=rgv3_bundle,
        utility_existence_model=utility_existence_model,
        utility_geometry_model=utility_geometry_model,
        utility_feature_names=utility_feature_names,
    )
    probability_roots = {
        role: Path(probability_root) / role for role in MODEL_SEQUENCE
    }
    for role in MODEL_SEQUENCE:
        validate_probability_directory(probability_roots[role], case_ids)

    audits = {}
    for case in cases:
        arrays = {
            role: _probability_array(
                probability_roots[role] / f"{case.case_id}.npz"
            )
            for role in MODEL_SEQUENCE
        }
        reference = nib.load(case.reference_path)
        spacing_zyx = tuple(
            float(value) for value in reference.header.get_zooms()[:3][::-1]
        )
        label_zyx, audit = build_n03_final(
            case.case_id,
            arrays,
            spacing_zyx,
            lcv1_case_bundle=bundles["lcv1_case"],
            lcv2_component_bundle=bundles["lcv2_component"],
            rgv3_et_bundle=bundles["rgv3_et"],
            utility_v4_existence_model=bundles["utility_v4_existence"],
            utility_v4_geometry_model=bundles["utility_v4_geometry"],
            utility_v4_feature_names=bundles["utility_v4_feature_names"],
        )
        label_xyz = label_zyx_to_reference_xyz(label_zyx, reference)
        publish_segmentation(case.case_id, label_xyz, reference, output_root)
        audits[case.case_id] = {
            "proposal_rows": len(audit),
            "added_rows": sum(row.get("decision") == "add" for row in audit),
        }
        del arrays, label_zyx, label_xyz

    validate_flat_output_set(output_root, case_ids)
    report = {
        "candidate": "N03_FINAL_UTILITY_V4",
        "case_count": len(cases),
        "model_sequence": list(MODEL_SEQUENCE),
        "audits": audits,
    }
    return report
