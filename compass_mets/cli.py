"""Build final COMPASS-METS N03 UTILITY_V4 segmentations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from compass_mets.pipeline import run_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probabilities", type=Path, required=True)
    parser.add_argument("--lcv1-bundle", type=Path, required=True)
    parser.add_argument("--lcv2-bundle", type=Path, required=True)
    parser.add_argument("--rgv3-bundle", type=Path, required=True)
    parser.add_argument("--utility-existence-model", type=Path, required=True)
    parser.add_argument("--utility-geometry-model", type=Path, required=True)
    parser.add_argument("--utility-feature-names", type=Path, required=True)
    args = parser.parse_args()
    report = run_pipeline(
        input_root=args.input,
        output_root=args.output,
        probability_root=args.probabilities,
        lcv1_bundle=args.lcv1_bundle,
        lcv2_bundle=args.lcv2_bundle,
        rgv3_bundle=args.rgv3_bundle,
        utility_existence_model=args.utility_existence_model,
        utility_geometry_model=args.utility_geometry_model,
        utility_feature_names=args.utility_feature_names,
    )
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
