from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import pytest
import torch


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE_ROOT))

from n03_docker.assets import (  # noqa: E402
    REQUIRED_INFERENCE_CHECKPOINT_KEYS,
    prepare_asset_bundle,
    slim_checkpoint,
    state_dict_tensor_hash,
)


def _checkpoint() -> dict:
    return {
        "network_weights": {
            "layer.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4),
            "layer.bias": torch.tensor([1.0, 2.0, 3.0]),
        },
        "trainer_name": "nnUNetTrainer",
        "init_args": {"configuration": "3d_fullres"},
        "inference_allowed_mirroring_axes": (0, 1, 2),
        "optimizer_state": {
            "state": {"large": torch.ones(1024, dtype=torch.float32)}
        },
        "logging": {"loss": [1.0, 0.5]},
        "current_epoch": 10,
    }


def test_slim_checkpoint_keeps_only_inference_fields_and_identical_weights(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pth"
    destination = tmp_path / "destination.pth"
    torch.save(_checkpoint(), source)
    before = torch.load(source, map_location="cpu", weights_only=False)

    report = slim_checkpoint(source, destination)

    after = torch.load(destination, map_location="cpu", weights_only=False)
    assert set(after) == set(REQUIRED_INFERENCE_CHECKPOINT_KEYS)
    assert state_dict_tensor_hash(after["network_weights"]) == state_dict_tensor_hash(
        before["network_weights"]
    )
    for key, tensor in before["network_weights"].items():
        assert torch.equal(after["network_weights"][key], tensor)
    assert report["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert report["destination_sha256"] == hashlib.sha256(
        destination.read_bytes()
    ).hexdigest()


def test_slim_checkpoint_rejects_missing_predictor_field(tmp_path: Path) -> None:
    checkpoint = _checkpoint()
    checkpoint.pop("trainer_name")
    source = tmp_path / "source.pth"
    torch.save(checkpoint, source)

    with pytest.raises(KeyError, match="trainer_name"):
        slim_checkpoint(source, tmp_path / "destination.pth")


def test_slim_checkpoint_is_atomic_on_failure(tmp_path: Path) -> None:
    source = tmp_path / "not-a-checkpoint.pth"
    source.write_bytes(b"broken")
    destination = tmp_path / "destination.pth"

    with pytest.raises(Exception):
        slim_checkpoint(source, destination)

    assert not destination.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_prepare_asset_bundle_preserves_nnunet_layout_and_two_learned_models(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    models = {}
    for role, folder in (
        ("XL", "nnUNetTrainer__XLPlans__3d_fullres"),
        ("M", "nnUNetTrainer__MPlans__3d_fullres"),
        ("FT", "FineTuneTrainer__MPlans__3d_fullres"),
    ):
        root = source_root / folder
        root.mkdir()
        metadata = []
        for name in ("plans.json", "dataset.json"):
            path = root / name
            path.write_text("{}", encoding="utf-8")
            metadata.append(
                {
                    "path": path.as_posix(),
                    "role": f"{role}_{name}",
                    "size_bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        checkpoints = []
        for fold in range(5):
            checkpoint = root / f"fold_{fold}" / "checkpoint_best.pth"
            checkpoint.parent.mkdir()
            torch.save(_checkpoint(), checkpoint)
            checkpoints.append(
                {
                    "path": checkpoint.as_posix(),
                    "role": f"{role}_checkpoint",
                    "fold": fold,
                    "checkpoint_name": "checkpoint_best.pth",
                    "size_bytes": checkpoint.stat().st_size,
                    "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                }
            )
        models[role] = {
            "root": root.as_posix(),
            "metadata": metadata,
            "checkpoints": checkpoints,
        }

    learned_models = {}
    for role in ("lcv1_case", "lcv2_component"):
        path = source_root / role / "models.joblib"
        path.parent.mkdir()
        path.write_bytes(role.encode("ascii"))
        learned_models[role] = {
            "path": path.as_posix(),
            "role": f"{role}_final",
            "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    inventory = {
        "candidate": "N03_XF12_LCv3_ET_parent_supported",
        "models": models,
        "learned_models": learned_models,
        "assets": [],
    }
    destination = tmp_path / "bundle"
    report = prepare_asset_bundle(inventory, destination)

    dataset_root = destination / "nnUNet_results" / "Dataset501_BraTS2025MET"
    assert {
        path.name for path in dataset_root.iterdir() if path.is_dir()
    } == {Path(model["root"]).name for model in models.values()}
    for model in models.values():
        model_root = dataset_root / Path(model["root"]).name
        assert (model_root / "plans.json").is_file()
        assert (model_root / "dataset.json").is_file()
        for fold in range(5):
            checkpoint = torch.load(
                model_root / f"fold_{fold}" / "checkpoint_best.pth",
                map_location="cpu",
                weights_only=False,
            )
            assert set(checkpoint) == set(REQUIRED_INFERENCE_CHECKPOINT_KEYS)
    for role in learned_models:
        assert (destination / "learned_models" / role / "models.joblib").is_file()
    assert report["checkpoint_count"] == 15
    assert report["learned_model_count"] == 2
