"""Run one fixed nnU-Net training or best-checkpoint validation fold."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
from typing import Any


REQUIRED_KEYS = {
    "schema_version",
    "model_id",
    "dataset_id",
    "configuration",
    "folds",
    "trainer",
    "plans",
    "checkpoint",
}


def load_training_config(path: Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = REQUIRED_KEYS.difference(config)
    if missing:
        raise ValueError(f"training config lacks {sorted(missing)}")
    if config["schema_version"] != 1:
        raise ValueError("unsupported training config schema")
    if config["dataset_id"] != 501:
        raise ValueError("this release only supports Dataset501")
    if config["folds"] != [0, 1, 2, 3, 4]:
        raise ValueError("training folds must be exactly [0, 1, 2, 3, 4]")
    if config["checkpoint"] != "checkpoint_best.pth":
        raise ValueError("checkpoint policy must use checkpoint_best.pth")
    parent = config.get("pretrained_from")
    if parent is not None and set(parent) != {"trainer", "plans"}:
        raise ValueError("pretrained_from must define trainer and plans")
    return config


def build_nnunet_command(
    config: dict[str, Any],
    fold: int,
    action: str,
    pretrained_checkpoint: Path | None = None,
    continue_training: bool = False,
) -> list[str]:
    if fold not in config["folds"]:
        raise ValueError(f"fold must be one of {config['folds']}")
    if action not in {"train", "validate"}:
        raise ValueError("action must be train or validate")
    command = [
        "nnUNetv2_train",
        str(config["dataset_id"]),
        str(config["configuration"]),
        str(fold),
        "-tr",
        str(config["trainer"]),
        "-p",
        str(config["plans"]),
    ]
    if action == "validate":
        return command + ["--val", "--val_best", "--npz"]
    command.append("--npz")
    if continue_training:
        command.append("--c")
    if config.get("pretrained_from"):
        if pretrained_checkpoint is None:
            raise ValueError("training requires a fold-matched pretrained checkpoint")
        command += ["-pretrained_weights", str(pretrained_checkpoint)]
    return command


def pretrained_checkpoint(
    config: dict[str, Any], fold: int, results_root: Path
) -> Path | None:
    parent = config.get("pretrained_from")
    if parent is None:
        return None
    run_name = (
        f"{parent['trainer']}__{parent['plans']}__{config['configuration']}"
    )
    return (
        results_root
        / "Dataset501_BraTS2025MET"
        / run_name
        / f"fold_{fold}"
        / "checkpoint_best.pth"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument(
        "--action", choices=("train", "validate"), default="train"
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(os.environ.get("nnUNet_results", "nnUNet_results")),
    )
    parser.add_argument("--continue-training", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_training_config(args.config)
    checkpoint = (
        pretrained_checkpoint(config, args.fold, args.results_root)
        if args.action == "train"
        else None
    )
    if checkpoint is not None and not args.dry_run and not checkpoint.is_file():
        raise FileNotFoundError(f"missing fold-matched checkpoint: {checkpoint}")
    command = build_nnunet_command(
        config,
        fold=args.fold,
        action=args.action,
        pretrained_checkpoint=checkpoint,
        continue_training=args.continue_training,
    )
    print(
        json.dumps(
            {
                "model_id": config["model_id"],
                "fold": args.fold,
                "command": command,
            }
        ),
        flush=True,
    )
    if not args.dry_run:
        subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
