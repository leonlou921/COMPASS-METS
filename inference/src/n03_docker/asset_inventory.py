"""Build and validate the immutable N03 inference asset inventory."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping


MODEL_ROLES = frozenset({"XL", "M", "FT"})
LEARNED_MODEL_ROLES = frozenset({"lcv1_case", "lcv2_component"})
EXPECTED_FOLDS = tuple(range(5))
FORBIDDEN_ASSET_MARKERS = (
    "/labelstr/",
    "/labels/",
    "/oof_",
    "/oof/",
    "/validation/",
    "/test_predictions/",
    "/predictions/",
    "historical_prediction",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path, role: str, **extra: Any) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "role": role,
        "path": path.resolve().as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        **extra,
    }


def build_source_inventory(
    model_roots: Mapping[str, Path],
    learned_model_roots: Mapping[str, Path],
    source_root: Path,
) -> dict[str, Any]:
    roles = set(model_roots)
    if roles != MODEL_ROLES:
        missing = sorted(MODEL_ROLES.difference(roles))
        extra = sorted(roles.difference(MODEL_ROLES))
        raise ValueError(f"model roles differ: missing={missing} extra={extra}")

    models: dict[str, Any] = {}
    for role in sorted(MODEL_ROLES):
        root = Path(model_roots[role])
        metadata = [
            _file_record(root / "plans.json", f"{role}_plans"),
            _file_record(root / "dataset.json", f"{role}_dataset"),
        ]
        checkpoints = []
        for fold in EXPECTED_FOLDS:
            checkpoint = root / f"fold_{fold}" / "checkpoint_best.pth"
            if not checkpoint.is_file():
                raise FileNotFoundError(
                    f"{role} fold_{fold} checkpoint_best.pth is missing: {checkpoint}"
                )
            checkpoints.append(
                _file_record(
                    checkpoint,
                    f"{role}_checkpoint",
                    fold=fold,
                    checkpoint_name=checkpoint.name,
                )
            )
        models[role] = {
            "root": root.resolve().as_posix(),
            "metadata": metadata,
            "checkpoints": checkpoints,
        }

    learned_roles = set(learned_model_roots)
    if learned_roles != LEARNED_MODEL_ROLES:
        missing = sorted(LEARNED_MODEL_ROLES.difference(learned_roles))
        extra = sorted(learned_roles.difference(LEARNED_MODEL_ROLES))
        raise ValueError(
            f"learned model roles differ: missing={missing} extra={extra}"
        )
    learned_models = {}
    for role in sorted(LEARNED_MODEL_ROLES):
        final_model = (
            Path(learned_model_roots[role])
            / "models"
            / "lightgbm"
            / "final"
            / "models.joblib"
        )
        learned_models[role] = _file_record(final_model, f"{role}_final")
    source_assets = []
    for path in sorted(Path(source_root).rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        source_assets.append(_file_record(path, "source_code"))

    inventory = {
        "schema_version": 1,
        "candidate": "N03_XF12_LCv3_ET_parent_supported",
        "models": models,
        "learned_models": learned_models,
        "assets": source_assets,
    }
    validate_source_inventory(inventory)
    return inventory


def _asset_rows(inventory: Mapping[str, Any]):
    for role in sorted(MODEL_ROLES):
        model = inventory["models"][role]
        yield from model["metadata"]
        yield from model["checkpoints"]
    yield from inventory["learned_models"].values()
    yield from inventory["assets"]


def validate_source_inventory(
    inventory: Mapping[str, Any],
    *,
    verify_files: bool = False,
) -> None:
    if inventory.get("candidate") != "N03_XF12_LCv3_ET_parent_supported":
        raise ValueError("inventory candidate is not frozen N03")
    if set(inventory.get("models", {})) != MODEL_ROLES:
        raise ValueError("inventory must contain exactly XL, M, and FT")

    for role in sorted(MODEL_ROLES):
        model = inventory["models"][role]
        checkpoints = model.get("checkpoints", [])
        if [row.get("fold") for row in checkpoints] != list(EXPECTED_FOLDS):
            raise ValueError(f"{role} must contain folds 0-4 in order")
        if any(row.get("checkpoint_name") != "checkpoint_best.pth" for row in checkpoints):
            raise ValueError(f"{role} contains a non-best checkpoint")
        if len(model.get("metadata", [])) != 2:
            raise ValueError(f"{role} must contain plans.json and dataset.json")

    if set(inventory.get("learned_models", {})) != LEARNED_MODEL_ROLES:
        raise ValueError(
            "inventory must contain exactly the LCv1 case and LCv2 component bundles"
        )

    for row in _asset_rows(inventory):
        text = f"{row.get('role', '')} {row.get('path', '')}".replace("\\", "/").lower()
        if any(marker in text for marker in FORBIDDEN_ASSET_MARKERS):
            raise ValueError(f"forbidden inference asset: {row.get('path')}")
        if not isinstance(row.get("size_bytes"), int) or row["size_bytes"] < 0:
            raise ValueError(f"invalid size_bytes for {row.get('path')}")
        digest = row.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"invalid sha256 for {row.get('path')}")
        if verify_files:
            path = Path(row["path"])
            if not path.is_file():
                raise FileNotFoundError(path)
            actual_size = path.stat().st_size
            if actual_size != row["size_bytes"]:
                raise ValueError(
                    f"size_bytes drift for {path}: {row['size_bytes']} != {actual_size}"
                )
            actual_hash = sha256_file(path)
            if actual_hash != digest:
                raise ValueError(f"sha256 drift for {path}")
