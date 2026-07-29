from __future__ import annotations

from pathlib import Path
import subprocess

import numpy as np
import pytest

from compass_mets.inference.predict import (
    DEFAULT_MODEL_SPECS,
    MODEL_SEQUENCE,
    ModelInferenceSpec,
    build_predict_command,
    run_prediction_sources,
    validate_probability_directory,
)


def _spec(role: str) -> ModelInferenceSpec:
    return ModelInferenceSpec(role=role, trainer=f"trainer-{role}", plans=f"plans-{role}")


def test_final_model_specs_are_frozen() -> None:
    assert DEFAULT_MODEL_SPECS["XL"].plans == "nnUNetResEncUNetXL30GBPlans"
    assert DEFAULT_MODEL_SPECS["M"].plans == "nnUNetResEncUNetMPlans"
    assert (
        DEFAULT_MODEL_SPECS["FT"].trainer
        == "nnUNetTrainer_ResEncM_DiceCEFocalTverskyFT"
    )


def test_predict_command_uses_best_fivefold_probabilities(tmp_path: Path) -> None:
    command = build_predict_command(
        executable="nnUNetv2_predict",
        input_root=tmp_path / "input",
        output_root=tmp_path / "output",
        spec=_spec("XL"),
        preprocessing_workers=1,
        export_workers=1,
    )
    assert command[command.index("-f") + 1 : command.index("-chk")] == [
        "0",
        "1",
        "2",
        "3",
        "4",
    ]
    assert command[command.index("-chk") + 1] == "checkpoint_best.pth"
    assert "--save_probabilities" in command
    assert "--disable_tta" not in command


def test_sources_run_strictly_xl_then_m_then_ft(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    run_prediction_sources(
        {role: _spec(role) for role in MODEL_SEQUENCE},
        input_root=tmp_path / "input",
        work_root=tmp_path / "work",
        nnunet_results=tmp_path / "models",
    )
    assert [command[command.index("-tr") + 1] for command, _ in calls] == [
        "trainer-XL",
        "trainer-M",
        "trainer-FT",
    ]
    assert all(kwargs["check"] is True for _, kwargs in calls)


def test_probability_directory_requires_complete_finite_four_channels(
    tmp_path: Path,
) -> None:
    root = tmp_path / "probabilities"
    root.mkdir()
    for case_id in ("case-a", "case-b"):
        np.savez_compressed(
            root / f"{case_id}.npz",
            probabilities=np.zeros((4, 2, 3, 4), dtype=np.float32),
        )
    assert validate_probability_directory(root, {"case-a", "case-b"})[
        "case_count"
    ] == 2
    values = np.zeros((4, 2, 3, 4), dtype=np.float32)
    values[0, 0, 0, 0] = np.nan
    np.savez_compressed(root / "case-b.npz", probabilities=values)
    with pytest.raises(ValueError, match="case-b.*finite"):
        validate_probability_directory(root, {"case-a", "case-b"})
