#!/usr/bin/env python3
"""Create an inference-only asset bundle from a verified inventory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "inference" / "src")
)

from n03_docker.assets import prepare_asset_bundle  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    report = prepare_asset_bundle(inventory, args.output)
    print(
        json.dumps(
            {
                "event": "asset_bundle_complete",
                "checkpoint_count": report["checkpoint_count"],
                "learned_model_count": report["learned_model_count"],
                "output": str(args.output),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
