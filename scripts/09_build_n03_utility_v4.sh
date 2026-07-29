#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=common.sh
source "${ROOT}/scripts/common.sh"

require_env \
  BRATS_METS_TEST_INPUT \
  COMPASS_PROBABILITY_ROOT \
  COMPASS_OUTPUT_ROOT \
  COMPASS_LCV1_BUNDLE \
  COMPASS_LCV2_BUNDLE \
  COMPASS_RGV3_BUNDLE \
  COMPASS_UTILITY_EXISTENCE_MODEL \
  COMPASS_UTILITY_GEOMETRY_MODEL \
  COMPASS_UTILITY_FEATURE_NAMES

"${PYTHON}" -m compass_mets.cli \
  --input "${BRATS_METS_TEST_INPUT}" \
  --output "${COMPASS_OUTPUT_ROOT}" \
  --probabilities "${COMPASS_PROBABILITY_ROOT}" \
  --lcv1-bundle "${COMPASS_LCV1_BUNDLE}" \
  --lcv2-bundle "${COMPASS_LCV2_BUNDLE}" \
  --rgv3-bundle "${COMPASS_RGV3_BUNDLE}" \
  --utility-existence-model "${COMPASS_UTILITY_EXISTENCE_MODEL}" \
  --utility-geometry-model "${COMPASS_UTILITY_GEOMETRY_MODEL}" \
  --utility-feature-names "${COMPASS_UTILITY_FEATURE_NAMES}"
