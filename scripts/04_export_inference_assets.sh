#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
INVENTORY="${INVENTORY:-${ROOT}/work/source_inventory.json}"
ASSET_OUTPUT="${ASSET_OUTPUT:-${ROOT}/assets}"

if [[ "${1:-}" == "--help" ]]; then
  echo "usage: set XL_MODEL_ROOT M_MODEL_ROOT FT_MODEL_ROOT LCV1_ROOT LCV2_ROOT"
  echo "       RGV3_ROOT UTILITY_V4_ROOT, then run $0"
  exit 0
fi
[[ "$#" -eq 0 ]] || { echo "unexpected arguments" >&2; exit 2; }

: "${XL_MODEL_ROOT:?set XL_MODEL_ROOT}"
: "${M_MODEL_ROOT:?set M_MODEL_ROOT}"
: "${FT_MODEL_ROOT:?set FT_MODEL_ROOT}"
: "${LCV1_ROOT:?set LCV1_ROOT}"
: "${LCV2_ROOT:?set LCV2_ROOT}"
: "${RGV3_ROOT:?set RGV3_ROOT}"
: "${UTILITY_V4_ROOT:?set UTILITY_V4_ROOT}"

[[ ! -e "${ASSET_OUTPUT}" ]] || {
  echo "refusing to overwrite existing asset output: ${ASSET_OUTPUT}" >&2
  exit 1
}
mkdir -p "$(dirname "${INVENTORY}")"
export PYTHONPATH="${ROOT}/inference/src:${PYTHONPATH:-}"

"${PYTHON}" "${ROOT}/docker/build_asset_inventory.py" \
  --xl-root "${XL_MODEL_ROOT}" \
  --m-root "${M_MODEL_ROOT}" \
  --ft-root "${FT_MODEL_ROOT}" \
  --lcv1-root "${LCV1_ROOT}" \
  --lcv2-root "${LCV2_ROOT}" \
  --rgv3-root "${RGV3_ROOT}" \
  --utility-v4-root "${UTILITY_V4_ROOT}" \
  --source-root "${ROOT}" \
  --output "${INVENTORY}"

"${PYTHON}" "${ROOT}/docker/prepare_assets.py" \
  --inventory "${INVENTORY}" \
  --output "${ASSET_OUTPUT}"
