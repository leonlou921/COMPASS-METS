"""Prepare inference-only checkpoint assets without changing network weights."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping

import torch

from .asset_inventory import validate_source_inventory


REQUIRED_INFERENCE_CHECKPOINT_KEYS = (
    "network_weights",
    "trainer_name",
    "init_args",
    "inference_allowed_mirroring_axes",
)
LEARNED_ASSET_DESTINATIONS = {
    "lcv1_case": Path("learned_models/lcv1_case/models.joblib"),
    "lcv2_component": Path("learned_models/lcv2_component/models.joblib"),
    "rgv3_et": Path("learned_models/rgv3_et/models.joblib"),
    "utility_v4_existence": Path(
        "learned_models/utility_v4/existence_model.joblib"
    ),
    "utility_v4_geometry": Path(
        "learned_models/utility_v4/geometry_model.joblib"
    ),
    "utility_v4_feature_names": Path(
        "learned_models/utility_v4/feature_names.json"
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_dict_tensor_hash(state_dict: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        tensor = state_dict[key]
        if not torch.is_tensor(tensor):
            raise TypeError(f"network_weights[{key!r}] is not a tensor")
        value = tensor.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(",".join(str(part) for part in value.shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        return torch.load(
            str(path),
            map_location="cpu",
            mmap=True,
            weights_only=False,
        )
    except TypeError:
        return torch.load(path, map_location="cpu", weights_only=False)


def slim_checkpoint(source: Path, destination: Path) -> dict[str, Any]:
    source = Path(source)
    destination = Path(destination)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        checkpoint = _load_checkpoint(source)
        missing = [
            key for key in REQUIRED_INFERENCE_CHECKPOINT_KEYS if key not in checkpoint
        ]
        if missing:
            raise KeyError(f"checkpoint is missing inference keys: {missing}")
        network_hash = state_dict_tensor_hash(checkpoint["network_weights"])
        slim = {
            key: checkpoint[key] for key in REQUIRED_INFERENCE_CHECKPOINT_KEYS
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(slim, temporary)
        verified = _load_checkpoint(temporary)
        if set(verified) != set(REQUIRED_INFERENCE_CHECKPOINT_KEYS):
            raise RuntimeError("slim checkpoint has unexpected keys")
        verified_hash = state_dict_tensor_hash(verified["network_weights"])
        if verified_hash != network_hash:
            raise RuntimeError("network tensor hash changed during checkpoint slimming")
        os.replace(temporary, destination)
        return {
            "source": source.resolve().as_posix(),
            "destination": destination.resolve().as_posix(),
            "source_size_bytes": source.stat().st_size,
            "destination_size_bytes": destination.stat().st_size,
            "source_sha256": sha256_file(source),
            "destination_sha256": sha256_file(destination),
            "network_tensor_sha256": network_hash,
            "network_tensor_count": len(checkpoint["network_weights"]),
            "preserved_keys": list(REQUIRED_INFERENCE_CHECKPOINT_KEYS),
        }
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _copy_verified(source: Path, destination: Path, expected_sha256: str) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    actual_hash = sha256_file(destination)
    if actual_hash != expected_sha256:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"copied asset hash changed: {source}")
    return {
        "source": source.resolve().as_posix(),
        "destination": destination.resolve().as_posix(),
        "sha256": actual_hash,
        "size_bytes": destination.stat().st_size,
    }


def prepare_asset_bundle(
    inventory: Mapping[str, Any],
    destination: Path,
    *,
    require_frozen_hashes: bool = True,
) -> dict[str, Any]:
    """Create the inference-only nnU-Net and learned-model asset tree."""
    validate_source_inventory(
        inventory,
        verify_files=True,
        require_frozen_hashes=require_frozen_hashes,
    )
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"asset destination already exists: {destination}")
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"asset temporary directory already exists: {temporary}")
    temporary.mkdir(parents=True)
    try:
        dataset_root = (
            temporary / "nnUNet_results" / "Dataset501_BraTS2025MET"
        )
        copied_metadata = []
        checkpoint_reports = []
        for role in ("XL", "M", "FT"):
            model = inventory["models"][role]
            target_model = dataset_root / Path(model["root"]).name
            for row in model["metadata"]:
                source = Path(row["path"])
                copied_metadata.append(
                    _copy_verified(
                        source,
                        target_model / source.name,
                        row["sha256"],
                    )
                )
            for row in model["checkpoints"]:
                checkpoint_reports.append(
                    slim_checkpoint(
                        Path(row["path"]),
                        target_model
                        / f"fold_{int(row['fold'])}"
                        / "checkpoint_best.pth",
                    )
                )

        learned_reports = []
        for role, relative_destination in LEARNED_ASSET_DESTINATIONS.items():
            row = inventory["learned_models"][role]
            learned_reports.append(
                _copy_verified(
                    Path(row["path"]),
                    temporary / relative_destination,
                    row["sha256"],
                )
            )

        report = {
            "candidate": inventory["candidate"],
            "checkpoint_count": len(checkpoint_reports),
            "learned_model_count": len(learned_reports),
            "metadata_count": len(copied_metadata),
            "checkpoints": checkpoint_reports,
            "learned_models": learned_reports,
            "metadata": copied_metadata,
        }
        provenance = temporary / "provenance"
        provenance.mkdir()
        (provenance / "asset_bundle_manifest.json").write_text(
            json.dumps(report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, destination)
        return report
    except BaseException:
        if temporary.is_dir():
            shutil.rmtree(temporary)
        raise
