"""Build and validate the immutable N03 inference asset inventory."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping


MODEL_ROLES = frozenset({"XL", "M", "FT"})
LEARNED_MODEL_RELATIVE_PATHS = {
    "lcv1_case": Path("models/lightgbm/final/models.joblib"),
    "lcv2_component": Path("models/lightgbm/final/models.joblib"),
    "rgv3_et": Path("models/ET/final/models.joblib"),
    "utility_v4_existence": Path(
        "models/existence_target/final/model.joblib"
    ),
    "utility_v4_geometry": Path(
        "models/geometry_safe_target/final/model.joblib"
    ),
    "utility_v4_feature_names": Path("feature_names.json"),
}
LEARNED_MODEL_ROLES = frozenset(LEARNED_MODEL_RELATIVE_PATHS)
EXPECTED_FINAL_LEARNED_HASHES = {
    "rgv3_et": "5fff6d9c7ef31bf4ce33bad211abe017fa1e0235d4e7f78d264c50c2c2a9fac1",
    "utility_v4_existence": "8a8c8f02ed652861b949b9a47aedaa8faff8e9904e4d9645f48a1a743ec0e3e0",
    "utility_v4_geometry": "bfd2d2be7e9349d4cde41f1e4682f87b9f0108216d69597cb80d0a6c9991f87e",
    "utility_v4_feature_names": "87b6523508688f52ad1cee6d600d7d353991f0d1451ec29ce0aed500ee07699d",
}
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
    *,
    enforce_frozen_hashes: bool = True,
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
        final_model = Path(learned_model_roots[role]) / LEARNED_MODEL_RELATIVE_PATHS[
            role
        ]
        learned_models[role] = _file_record(final_model, f"{role}_final")
    source_assets = []
    for path in sorted(Path(source_root).rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        source_assets.append(_file_record(path, "source_code"))

    inventory = {
        "schema_version": 2,
        "candidate": "N03_FINAL_UTILITY_V4",
        "models": models,
        "learned_models": learned_models,
        "assets": source_assets,
        "frozen_asset_hashes_verified": bool(enforce_frozen_hashes),
    }
    validate_source_inventory(
        inventory,
        require_frozen_hashes=enforce_frozen_hashes,
    )
    return inventory


def _asset_rows(inventory: Mapping[str, Any]):
    for role in sorted(inventory.get("models", {})):
        model = inventory["models"][role]
        yield from model["metadata"]
        yield from model["checkpoints"]
    yield from inventory["learned_models"].values()
    yield from inventory["assets"]


def validate_source_inventory(
    inventory: Mapping[str, Any],
    *,
    verify_files: bool = False,
    require_model_assets: bool = True,
    require_frozen_hashes: bool = False,
) -> None:
    if inventory.get("candidate") != "N03_FINAL_UTILITY_V4":
        raise ValueError("inventory candidate is not N03_FINAL_UTILITY_V4")
    if require_model_assets and set(inventory.get("models", {})) != MODEL_ROLES:
        raise ValueError("inventory must contain exactly XL, M, and FT")

    for role in sorted(inventory.get("models", {})):
        if role not in MODEL_ROLES:
            raise ValueError(f"unexpected model role: {role}")
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
            "inventory must contain the LCv1, LCv2, RGv3-ET, and utility-v4 assets"
        )
    if require_frozen_hashes:
        if inventory.get("frozen_asset_hashes_verified") is not True:
            raise ValueError("inventory does not assert frozen asset hashes")
        for role, expected in EXPECTED_FINAL_LEARNED_HASHES.items():
            actual = inventory["learned_models"][role].get("sha256")
            if actual != expected:
                raise ValueError(
                    f"{role} sha256 differs from frozen value: "
                    f"{actual} != {expected}"
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
