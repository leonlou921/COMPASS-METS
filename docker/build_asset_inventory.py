#!/usr/bin/env python3
"""Hash and validate the private checkpoints needed by the N03 image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from n03_docker.asset_inventory import build_source_inventory


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xl-root", type=Path, required=True)
    parser.add_argument("--m-root", type=Path, required=True)
    parser.add_argument("--ft-root", type=Path, required=True)
    parser.add_argument("--lcv1-root", type=Path, required=True)
    parser.add_argument("--lcv2-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    inventory = build_source_inventory(
        {"XL": args.xl_root, "M": args.m_root, "FT": args.ft_root},
        {"lcv1_case": args.lcv1_root, "lcv2_component": args.lcv2_root},
        args.source_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "event": "asset_inventory_complete",
                "output": str(args.output),
                "checkpoint_count": 15,
                "learned_model_count": 2,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
