#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"

if [[ "${1:-}" == "--help" || "$#" -ne 2 ]]; then
  echo "usage: $0 LCV1_CONFIG LCV2_CONFIG"
  echo "also requires the RGV3_* and UTILITY_* paths documented in docs/TRAINING.md"
  [[ "${1:-}" == "--help" ]] && exit 0
  exit 2
fi

bash "${ROOT}/training/train_learned_gates.sh" "$1" "$2"

: "${RGV3_COMPONENT_FEATURES:?set RGV3_COMPONENT_FEATURES}"
: "${RGV3_CASE_PREDICTIONS:?set RGV3_CASE_PREDICTIONS}"
: "${RGV3_V2_PREDICTIONS:?set RGV3_V2_PREDICTIONS}"
: "${RGV3_OUTPUT_ROOT:?set RGV3_OUTPUT_ROOT}"
: "${UTILITY_FEATURES:?set UTILITY_FEATURES}"
: "${UTILITY_V2_PREDICTIONS:?set UTILITY_V2_PREDICTIONS}"
: "${UTILITY_OUTPUT_ROOT:?set UTILITY_OUTPUT_ROOT}"

"${PYTHON}" "${ROOT}/training/learned_gates/rgv3/region_gate_v3.py" \
  --component-features "${RGV3_COMPONENT_FEATURES}" \
  --case-predictions "${RGV3_CASE_PREDICTIONS}" \
  --v2-predictions "${RGV3_V2_PREDICTIONS}" \
  --output-root "${RGV3_OUTPUT_ROOT}"

"${PYTHON}" "${ROOT}/training/learned_gates/utility_v4/train_utility_v4.py" \
  --features "${UTILITY_FEATURES}" \
  --v2 "${UTILITY_V2_PREDICTIONS}" \
  --v3 "${RGV3_OUTPUT_ROOT}/oof_predictions/component_predictions.parquet" \
  --output "${UTILITY_OUTPUT_ROOT}"
