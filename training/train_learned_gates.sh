#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH=$(readlink -f "${BASH_SOURCE[0]}")
SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")
ROOT=$(dirname "${SCRIPT_DIR}")
PYTHON="${PYTHON:-python}"
V1_CONFIG="${1:?usage: train_learned_gates.sh V1_CONFIG V2_CONFIG}"
V2_CONFIG="${2:?usage: train_learned_gates.sh V1_CONFIG V2_CONFIG}"
SHARDS="${SHARDS:-1}"
LCV1="${ROOT}/inference/vendor/lcv1"

export PYTHONPATH="${LCV1}:${PYTHONPATH:-}"

LAST_SHARD=$(expr "${SHARDS}" - 1)
for shard in $(seq 0 "${LAST_SHARD}"); do
  "${PYTHON}" "${LCV1}/build_features.py" \
    --config "${V1_CONFIG}" \
    --shard-count "${SHARDS}" \
    --shard-index "${shard}"
done
"${PYTHON}" "${LCV1}/build_features.py" \
  --config "${V1_CONFIG}" \
  --consolidate
"${PYTHON}" "${LCV1}/run_pipeline.py" \
  --config "${V1_CONFIG}" \
  --stage after-features

"${PYTHON}" "${LCV1}/train_models_v2.py" --config "${V2_CONFIG}"

for shard in $(seq 0 "${LAST_SHARD}"); do
  "${PYTHON}" "${LCV1}/v2_pipeline.py" \
    --config "${V2_CONFIG}" \
    --stage build-calibration \
    --shard-count "${SHARDS}" \
    --shard-index "${shard}"
done
"${PYTHON}" "${LCV1}/v2_pipeline.py" \
  --config "${V2_CONFIG}" \
  --stage consolidate-calibration

for shard in $(seq 0 "${LAST_SHARD}"); do
  "${PYTHON}" "${LCV1}/v2_pipeline.py" \
    --config "${V2_CONFIG}" \
    --stage evaluate \
    --shard-count "${SHARDS}" \
    --shard-index "${shard}"
done
"${PYTHON}" "${LCV1}/v2_pipeline.py" \
  --config "${V2_CONFIG}" \
  --stage consolidate-evaluation
