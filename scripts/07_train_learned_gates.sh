#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=common.sh
source "${ROOT}/scripts/common.sh"

require_env \
  LCV1_CONFIG LCV2_CONFIG \
  RGV3_COMPONENT_FEATURES RGV3_CASE_PREDICTIONS RGV3_V2_PREDICTIONS \
  RGV3_OUTPUT_ROOT \
  UTILITY_FEATURES UTILITY_V2_PREDICTIONS UTILITY_OUTPUT_ROOT

SHARDS="${SHARDS:-1}"
for ((shard=0; shard<SHARDS; shard++)); do
  "${PYTHON}" -m compass_mets.learned_gates.lcv1.build_features \
    --config "${LCV1_CONFIG}" --shard-count "${SHARDS}" --shard-index "${shard}"
done
"${PYTHON}" -m compass_mets.learned_gates.lcv1.build_features \
  --config "${LCV1_CONFIG}" --consolidate
"${PYTHON}" -m compass_mets.learned_gates.lcv1.run_pipeline \
  --config "${LCV1_CONFIG}" --stage after-features

"${PYTHON}" -m compass_mets.learned_gates.lcv1.train_models_v2 \
  --config "${LCV2_CONFIG}"
for ((shard=0; shard<SHARDS; shard++)); do
  "${PYTHON}" -m compass_mets.learned_gates.lcv1.v2_pipeline \
    --config "${LCV2_CONFIG}" --stage build-calibration \
    --shard-count "${SHARDS}" --shard-index "${shard}"
done
"${PYTHON}" -m compass_mets.learned_gates.lcv1.v2_pipeline \
  --config "${LCV2_CONFIG}" --stage consolidate-calibration
for ((shard=0; shard<SHARDS; shard++)); do
  "${PYTHON}" -m compass_mets.learned_gates.lcv1.v2_pipeline \
    --config "${LCV2_CONFIG}" --stage evaluate \
    --shard-count "${SHARDS}" --shard-index "${shard}"
done
"${PYTHON}" -m compass_mets.learned_gates.lcv1.v2_pipeline \
  --config "${LCV2_CONFIG}" --stage consolidate-evaluation

"${PYTHON}" -m compass_mets.learned_gates.rgv3 \
  --component-features "${RGV3_COMPONENT_FEATURES}" \
  --case-predictions "${RGV3_CASE_PREDICTIONS}" \
  --v2-predictions "${RGV3_V2_PREDICTIONS}" \
  --output-root "${RGV3_OUTPUT_ROOT}"

"${PYTHON}" -m compass_mets.learned_gates.train_utility_v4 \
  --features "${UTILITY_FEATURES}" \
  --v2 "${UTILITY_V2_PREDICTIONS}" \
  --v3 "${RGV3_OUTPUT_ROOT}/oof_predictions/component_predictions.parquet" \
  --output "${UTILITY_OUTPUT_ROOT}"
