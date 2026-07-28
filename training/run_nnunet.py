#!/usr/bin/env python3
"""Run the frozen XL, M, or FT nnU-Net fold commands."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
from typing import Any


def load_model_registry(path: Path) -> dict[str, Any]:
    registry = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"dataset_id", "configuration", "models"}
    missing = required.difference(registry)
    if missing:
        raise ValueError(f"model registry lacks {sorted(missing)}")
    for key, model in registry["models"].items():
        if not {"trainer", "plans"}.issubset(model):
            raise ValueError(f"model {key!r} lacks trainer or plans")
    return registry


def build_nnunet_command(
    registry: dict[str, Any],
    model_key: str,
    fold: int,
    action: str,
    pretrained_checkpoint: Path | None = None,
) -> list[str]:
    if model_key not in registry["models"]:
        raise ValueError(f"unknown model {model_key!r}")
    if fold not in range(5):
        raise ValueError("fold must be in 0..4")
    if action not in {"train", "validate"}:
        raise ValueError("action must be train or validate")

    model = registry["models"][model_key]
    command = [
        "nnUNetv2_train",
        str(registry["dataset_id"]),
        str(registry["configuration"]),
        str(fold),
        "-tr",
        str(model["trainer"]),
        "-p",
        str(model["plans"]),
    ]
    if action == "validate":
        return command + ["--val", "--val_best", "--npz"]
    command.append("--npz")
    if model.get("pretrained_from"):
        if pretrained_checkpoint is None:
            raise ValueError(
                f"{model_key} training requires a pretrained checkpoint"
            )
        command += ["-pretrained_weights", str(pretrained_checkpoint)]
    return command


def _pretrained_checkpoint(
    registry: dict[str, Any], model_key: str, fold: int, results_root: Path
) -> Path | None:
    model = registry["models"][model_key]
    parent_key = model.get("pretrained_from")
    if not parent_key:
        return None
    parent = registry["models"][parent_key]
    run_name = (
        f"{parent['trainer']}__{parent['plans']}__"
        f"{registry['configuration']}"
    )
    return (
        results_root
        / f"Dataset{int(registry['dataset_id']):03d}_BraTS2025MET"
        / run_name
        / f"fold_{fold}"
        / "checkpoint_best.pth"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--model", choices=("xl", "m", "ft"), required=True)
    parser.add_argument("--action", choices=("train", "validate"), required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(os.environ.get("nnUNet_results", "nnUNet_results")),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    registry = load_model_registry(args.registry)
    for fold in args.folds:
        checkpoint = _pretrained_checkpoint(
            registry, args.model, fold, args.results_root
        )
        if checkpoint is not None and not checkpoint.is_file():
            raise FileNotFoundError(
                f"missing fold-matched M checkpoint for FT: {checkpoint}"
            )
        command = build_nnunet_command(
            registry,
            args.model,
            fold,
            args.action,
            pretrained_checkpoint=checkpoint,
        )
        print(json.dumps({"fold": fold, "command": command}), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
