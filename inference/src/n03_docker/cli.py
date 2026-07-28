"""Container entrypoint for the frozen N03 candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import run_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("/input"))
    parser.add_argument("--output", type=Path, default=Path("/output"))
    parser.add_argument("--assets", type=Path, default=Path("/opt/n03/assets"))
    parser.add_argument("--work", type=Path, default=Path("/tmp"))
    args = parser.parse_args()
    report = run_pipeline(
        input_root=args.input,
        output_root=args.output,
        assets_root=args.assets,
        work_parent=args.work,
    )
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
