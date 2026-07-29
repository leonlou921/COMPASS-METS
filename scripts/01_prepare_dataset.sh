#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"

if [[ "${1:-}" == "--help" || "$#" -ne 2 ]]; then
  echo "usage: $0 TRAIN_CASE_ROOT VALIDATION_CASE_ROOT"
  echo "requires: nnUNet_raw, nnUNet_preprocessed, nnUNet_results"
  [[ "${1:-}" == "--help" ]] && exit 0
  exit 2
fi

: "${nnUNet_raw:?set nnUNet_raw}"
: "${nnUNet_preprocessed:?set nnUNet_preprocessed}"
: "${nnUNet_results:?set nnUNet_results}"
export PYTHONPATH="${ROOT}/third_party/nnUNet:${PYTHONPATH:-}"

"${PYTHON}" "${ROOT}/preprocessing/prepare_dataset501.py" \
  --train-dir "$1" \
  --valid-dir "$2" \
  --nnunet-raw "${nnUNet_raw}"

bash "${ROOT}/preprocessing/plan_and_preprocess.sh"
