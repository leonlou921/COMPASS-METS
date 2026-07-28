from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE_ROOT))

from n03_docker.asset_inventory import (  # noqa: E402
    MODEL_ROLES,
    build_source_inventory,
    validate_source_inventory,
)


def _make_model_root(root: Path, role: str) -> Path:
    model_root = root / role
    model_root.mkdir()
    (model_root / "plans.json").write_text(
        json.dumps({"configurations": {"3d_fullres": {"patch_size": [8, 8, 8]}}}),
        encoding="utf-8",
    )
    (model_root / "dataset.json").write_text(
        json.dumps({"channel_names": {"0": "t1c", "1": "t1n", "2": "t2f", "3": "t2w"}}),
        encoding="utf-8",
    )
    for fold in range(5):
        fold_root = model_root / f"fold_{fold}"
        fold_root.mkdir()
        (fold_root / "checkpoint_best.pth").write_bytes(
            f"{role}-fold-{fold}".encode("ascii")
        )
    return model_root


def _complete_fixture(
    tmp_path: Path,
) -> tuple[dict[str, Path], dict[str, Path], Path]:
    model_roots = {
        role: _make_model_root(tmp_path, role) for role in sorted(MODEL_ROLES)
    }
    learned_model_roots = {}
    for role, payload in (
        ("lcv1_case", b"frozen-lcv1-case"),
        ("lcv2_component", b"frozen-lcv2-component"),
    ):
        root = tmp_path / role
        final = root / "models" / "lightgbm" / "final"
        final.mkdir(parents=True)
        (final / "models.joblib").write_bytes(payload)
        learned_model_roots[role] = root
    source = tmp_path / "source"
    source.mkdir()
    (source / "pipeline.py").write_text("N03 = True\n", encoding="utf-8")
    return model_roots, learned_model_roots, source


def test_inventory_requires_three_roles_and_five_best_folds(tmp_path: Path) -> None:
    model_roots, learned_model_roots, source_root = _complete_fixture(tmp_path)
    inventory = build_source_inventory(model_roots, learned_model_roots, source_root)
    validate_source_inventory(inventory)

    assert set(inventory["models"]) == MODEL_ROLES
    assert set(inventory["learned_models"]) == {
        "lcv1_case",
        "lcv2_component",
    }
    for role in MODEL_ROLES:
        checkpoints = inventory["models"][role]["checkpoints"]
        assert [row["fold"] for row in checkpoints] == list(range(5))
        assert {Path(row["path"]).name for row in checkpoints} == {
            "checkpoint_best.pth"
        }
        assert all(len(row["sha256"]) == 64 for row in checkpoints)


def test_inventory_rejects_missing_fold(tmp_path: Path) -> None:
    model_roots, learned_model_roots, source_root = _complete_fixture(tmp_path)
    (model_roots["XL"] / "fold_4" / "checkpoint_best.pth").unlink()

    with pytest.raises(FileNotFoundError, match="XL.*fold_4.*checkpoint_best"):
        build_source_inventory(model_roots, learned_model_roots, source_root)


def test_inventory_rejects_forbidden_training_or_test_artifacts(
    tmp_path: Path,
) -> None:
    model_roots, learned_model_roots, source_root = _complete_fixture(tmp_path)
    inventory = build_source_inventory(model_roots, learned_model_roots, source_root)
    inventory["assets"].append(
        {
            "role": "historical_prediction",
            "path": "/data/test_predictions/case.npz",
            "size_bytes": 1,
            "sha256": "0" * 64,
        }
    )

    with pytest.raises(ValueError, match="forbidden inference asset"):
        validate_source_inventory(inventory)


def test_inventory_detects_size_or_hash_drift(tmp_path: Path) -> None:
    model_roots, learned_model_roots, source_root = _complete_fixture(tmp_path)
    inventory = build_source_inventory(model_roots, learned_model_roots, source_root)
    row = inventory["models"]["M"]["checkpoints"][2]
    row["size_bytes"] += 1

    with pytest.raises(ValueError, match="size_bytes"):
        validate_source_inventory(inventory, verify_files=True)


def test_inventory_contains_only_two_final_learned_bundles(tmp_path: Path) -> None:
    model_roots, learned_model_roots, source_root = _complete_fixture(tmp_path)
    for root in learned_model_roots.values():
        fold_model = root / "models" / "lightgbm" / "fold_0"
        fold_model.mkdir()
        (fold_model / "models.joblib").write_bytes(b"crossfit-not-for-inference")

    inventory = build_source_inventory(model_roots, learned_model_roots, source_root)

    for role, root in learned_model_roots.items():
        row = inventory["learned_models"][role]
        assert Path(row["path"]).as_posix() == (
            root / "models" / "lightgbm" / "final" / "models.joblib"
        ).as_posix()


def test_inventory_rejects_missing_lcv1_case_bundle(tmp_path: Path) -> None:
    model_roots, learned_model_roots, source_root = _complete_fixture(tmp_path)
    (
        learned_model_roots["lcv1_case"]
        / "models"
        / "lightgbm"
        / "final"
        / "models.joblib"
    ).unlink()

    with pytest.raises(FileNotFoundError, match="models.joblib"):
        build_source_inventory(model_roots, learned_model_roots, source_root)
