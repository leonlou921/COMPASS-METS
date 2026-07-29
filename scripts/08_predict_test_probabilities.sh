#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=common.sh
source "${ROOT}/scripts/common.sh"

require_env \
  BRATS_METS_TEST_INPUT \
  COMPASS_PROBABILITY_ROOT \
  COMPASS_WORK_ROOT \
  nnUNet_results

"${PYTHON}" -m compass_mets.inference.predict \
  --input "${BRATS_METS_TEST_INPUT}" \
  --output "${COMPASS_PROBABILITY_ROOT}" \
  --nnunet-results "${nnUNet_results}" \
  --work "${COMPASS_WORK_ROOT}"
