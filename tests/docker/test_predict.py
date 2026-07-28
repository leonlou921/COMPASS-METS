from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE_ROOT))

from n03_docker.predict import (  # noqa: E402
    MODEL_SEQUENCE,
    ModelInferenceSpec,
    build_predict_command,
    run_prediction_sources,
    validate_probability_directory,
)


def _spec(role: str) -> ModelInferenceSpec:
    return ModelInferenceSpec(
        role=role,
        trainer=f"trainer-{role}",
        plans=f"plans-{role}",
    )


def test_predict_command_freezes_best_fivefold_probability_inference(
    tmp_path: Path,
) -> None:
    command = build_predict_command(
        executable="nnUNetv2_predict",
        input_root=tmp_path / "input",
        output_root=tmp_path / "output",
        spec=_spec("XL"),
        preprocessing_workers=1,
        export_workers=1,
    )

    assert command[:1] == ["nnUNetv2_predict"]
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
    assert command[command.index("-tr") + 1] == "trainer-XL"
    assert command[command.index("-p") + 1] == "plans-XL"


def test_sources_run_strictly_xl_then_m_then_ft(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    specs = {role: _spec(role) for role in MODEL_SEQUENCE}

    run_prediction_sources(
        specs,
        input_root=tmp_path / "input",
        work_root=tmp_path / "work",
        nnunet_results=tmp_path / "models",
    )

    assert [command[command.index("-tr") + 1] for command, _ in calls] == [
        "trainer-XL",
        "trainer-M",
        "trainer-FT",
    ]
    assert all(call[1]["check"] is True for call in calls)


def test_source_failure_stops_later_models(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[command.index("-tr") + 1] == "trainer-M":
            raise subprocess.CalledProcessError(2, command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    specs = {role: _spec(role) for role in MODEL_SEQUENCE}

    with pytest.raises(subprocess.CalledProcessError):
        run_prediction_sources(
            specs,
            input_root=tmp_path / "input",
            work_root=tmp_path / "work",
            nnunet_results=tmp_path / "models",
        )

    assert len(calls) == 2


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

    report = validate_probability_directory(root, {"case-a", "case-b"})
    assert report["case_count"] == 2
    assert report["channel_count"] == 4

    values = np.zeros((4, 2, 3, 4), dtype=np.float32)
    values[0, 0, 0, 0] = np.nan
    np.savez_compressed(root / "case-b.npz", probabilities=values)
    with pytest.raises(ValueError, match="case-b.*NaN|case-b.*finite"):
        validate_probability_directory(root, {"case-a", "case-b"})
