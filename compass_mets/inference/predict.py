"""Run the three frozen nnU-Net probability sources sequentially."""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Mapping

import numpy as np

from compass_mets.inference.input_contract import (
    discover_cases,
    stage_nnunet_inputs,
)


MODEL_SEQUENCE = ("XL", "M", "FT")
DATASET = "Dataset501_BraTS2025MET"
CONFIGURATION = "3d_fullres"
CHECKPOINT = "checkpoint_best.pth"
FOLDS = ("0", "1", "2", "3", "4")


@dataclass(frozen=True)
class ModelInferenceSpec:
    role: str
    trainer: str
    plans: str

    def __post_init__(self) -> None:
        if self.role not in MODEL_SEQUENCE:
            raise ValueError(f"unsupported model role: {self.role}")


def build_predict_command(
    *,
    executable: str,
    input_root: Path,
    output_root: Path,
    spec: ModelInferenceSpec,
    preprocessing_workers: int,
    export_workers: int,
) -> list[str]:
    if preprocessing_workers < 1 or export_workers < 1:
        raise ValueError("nnU-Net worker counts must be positive")
    return [
        executable,
        "-i",
        str(input_root),
        "-o",
        str(output_root),
        "-d",
        DATASET,
        "-c",
        CONFIGURATION,
        "-tr",
        spec.trainer,
        "-p",
        spec.plans,
        "-f",
        *FOLDS,
        "-chk",
        CHECKPOINT,
        "-npp",
        str(preprocessing_workers),
        "-nps",
        str(export_workers),
        "--save_probabilities",
        "--disable_progress_bar",
    ]


def run_prediction_sources(
    specs: Mapping[str, ModelInferenceSpec],
    *,
    input_root: Path,
    work_root: Path,
    nnunet_results: Path,
    executable: str = "nnUNetv2_predict",
    preprocessing_workers: int = 1,
    export_workers: int = 1,
) -> dict[str, Path]:
    if set(specs) != set(MODEL_SEQUENCE):
        raise ValueError(f"prediction specs must be exactly {MODEL_SEQUENCE}")
    work_root = Path(work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["nnUNet_results"] = str(Path(nnunet_results))
    outputs = {}
    for role in MODEL_SEQUENCE:
        output_root = work_root / role
        output_root.mkdir(parents=True, exist_ok=True)
        command = build_predict_command(
            executable=executable,
            input_root=Path(input_root),
            output_root=output_root,
            spec=specs[role],
            preprocessing_workers=preprocessing_workers,
            export_workers=export_workers,
        )
        subprocess.run(
            command,
            check=True,
            env=environment,
        )
        outputs[role] = output_root
    return outputs


def _probability_array(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if "probabilities" in archive.files:
            value = archive["probabilities"]
        elif len(archive.files) == 1:
            value = archive[archive.files[0]]
        else:
            raise ValueError(
                f"{path.name} has ambiguous probability keys: {archive.files}"
            )
    return np.asarray(value)


def validate_probability_directory(
    root: Path,
    expected_case_ids: set[str],
) -> dict[str, int]:
    root = Path(root)
    files = sorted(root.glob("*.npz"))
    actual_case_ids = {path.stem for path in files}
    if actual_case_ids != set(expected_case_ids):
        raise ValueError(
            f"probability case set differs: "
            f"missing={sorted(set(expected_case_ids) - actual_case_ids)} "
            f"extra={sorted(actual_case_ids - set(expected_case_ids))}"
        )
    shapes = set()
    for path in files:
        probabilities = _probability_array(path)
        if probabilities.ndim != 4 or probabilities.shape[0] != 4:
            raise ValueError(
                f"{path.stem} probability shape must be (4, Z, Y, X), "
                f"got {probabilities.shape}"
            )
        if not np.isfinite(probabilities).all():
            raise ValueError(f"{path.stem} probabilities are not finite (NaN/Inf)")
        shapes.add(tuple(int(value) for value in probabilities.shape))
    return {
        "case_count": len(files),
        "channel_count": 4,
        "distinct_shapes": len(shapes),
    }


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


def predict_from_input(
    input_root: Path,
    output_root: Path,
    nnunet_results: Path,
    work_root: Path,
    executable: str = "nnUNetv2_predict",
) -> dict[str, object]:
    cases = discover_cases(input_root)
    case_ids = {case.case_id for case in cases}
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    if any(output_root.iterdir()):
        raise ValueError(f"probability output must be empty: {output_root}")
    Path(work_root).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="compass-", dir=work_root) as temporary:
        staging = Path(temporary) / "nnunet_input"
        stage_nnunet_inputs(cases, staging)
        roots = run_prediction_sources(
            DEFAULT_MODEL_SPECS,
            input_root=staging,
            work_root=output_root,
            nnunet_results=nnunet_results,
            executable=executable,
        )
    audits = {
        role: validate_probability_directory(roots[role], case_ids)
        for role in MODEL_SEQUENCE
    }
    return {"case_count": len(cases), "sources": audits}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nnunet-results", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--executable", default="nnUNetv2_predict")
    args = parser.parse_args()
    report = predict_from_input(
        args.input,
        args.output,
        args.nnunet_results,
        args.work,
        executable=args.executable,
    )
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
