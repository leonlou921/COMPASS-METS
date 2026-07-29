"""End-to-end `/input` to `/output` N03 inference pipeline."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
from typing import Any

import joblib
import nibabel as nib
import numpy as np

from .input_contract import discover_cases, stage_nnunet_inputs
from .output_contract import publish_segmentation, validate_flat_output_set
from .predict import (
    MODEL_SEQUENCE,
    ModelInferenceSpec,
    _probability_array,
    run_prediction_sources,
    validate_probability_directory,
)


DEFAULT_MODEL_SPECS = {
    "XL": ModelInferenceSpec(
        role="XL",
        trainer="nnUNetTrainer",
        plans="nnUNetResEncUNetXL30GBPlans",
    ),
    "M": ModelInferenceSpec(
        role="M",
        trainer="nnUNetTrainer",
        plans="nnUNetResEncUNetMPlans",
    ),
    "FT": ModelInferenceSpec(
        role="FT",
        trainer="nnUNetTrainer_ResEncM_DiceCEFocalTverskyFT",
        plans="nnUNetResEncUNetMPlans",
    ),
}


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


def _load_learned_bundles(assets_root: Path) -> dict[str, Any]:
    learned = Path(assets_root) / "learned_models"
    utility = learned / "utility_v4"
    feature_names = json.loads(
        (utility / "feature_names.json").read_text(encoding="utf-8")
    )
    if (
        not isinstance(feature_names, list)
        or not feature_names
        or not all(isinstance(value, str) and value for value in feature_names)
    ):
        raise ValueError("utility-v4 feature names must be a non-empty string list")
    return {
        "lcv1_case": joblib.load(learned / "lcv1_case" / "models.joblib"),
        "lcv2_component": joblib.load(
            learned / "lcv2_component" / "models.joblib"
        ),
        "rgv3_et": joblib.load(learned / "rgv3_et" / "models.joblib"),
        "utility_v4_existence": joblib.load(
            utility / "existence_model.joblib"
        ),
        "utility_v4_geometry": joblib.load(
            utility / "geometry_model.joblib"
        ),
        "utility_v4_feature_names": feature_names,
    }


def run_pipeline(
    *,
    input_root: Path,
    output_root: Path,
    assets_root: Path,
    work_parent: Path = Path("/tmp"),
    executable: str = "nnUNetv2_predict",
) -> dict[str, Any]:
    """Run the three frozen ensembles sequentially and publish only N03."""
    from .postprocess import build_n03_final

    cases = discover_cases(Path(input_root))
    case_ids = {case.case_id for case in cases}
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    if any(output_root.iterdir()):
        raise ValueError(f"output root must be empty before inference: {output_root}")

    assets_root = Path(assets_root)
    nnunet_results = assets_root / "nnUNet_results"
    bundles = _load_learned_bundles(assets_root)
    Path(work_parent).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="n03-", dir=work_parent) as temporary:
        work_root = Path(temporary)
        staging = work_root / "input"
        stage_nnunet_inputs(cases, staging)
        probability_roots = run_prediction_sources(
            DEFAULT_MODEL_SPECS,
            input_root=staging,
            work_root=work_root / "probabilities",
            nnunet_results=nnunet_results,
            executable=executable,
        )
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
                "added_rows": sum(
                    row.get("decision") == "add" for row in audit
                ),
            }
            del arrays, label_zyx, label_xyz

    validate_flat_output_set(output_root, case_ids)
    report = {
        "candidate": "N03_FINAL_UTILITY_V4",
        "case_count": len(cases),
        "model_sequence": list(MODEL_SEQUENCE),
        "audits": audits,
    }
    # The report stays outside /output so the submission directory remains flat.
    report_path = Path(work_parent) / "n03_last_run.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def remove_work_tree(path: Path) -> None:
    """Explicit helper for test/build scripts; never accepts a broad root."""
    path = Path(path).resolve()
    if path.name.startswith("n03-") and path.parent != path:
        shutil.rmtree(path)
    else:
        raise ValueError(f"refusing to remove non-N03 work tree: {path}")
