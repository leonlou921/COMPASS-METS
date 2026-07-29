#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
REGISTRY="${ROOT}/configs/models/final_models.json"
DRY_RUN=()

if [[ "${1:-}" == "--help" ]]; then
  echo "usage: $0 [--dry-run]"
  echo "trains M -> fold-matched FT -> XL, then exports best-fold OOF NPZ"
  exit 0
fi
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=(--dry-run)
  shift
fi
[[ "$#" -eq 0 ]] || { echo "unexpected arguments" >&2; exit 2; }

: "${nnUNet_raw:?set nnUNet_raw}"
: "${nnUNet_preprocessed:?set nnUNet_preprocessed}"
: "${nnUNet_results:?set nnUNet_results}"
export PYTHONPATH="${ROOT}/third_party/nnUNet:${PYTHONPATH:-}"

for model in m ft xl; do
  "${PYTHON}" "${ROOT}/training/run_nnunet.py" \
    --registry "${REGISTRY}" \
    --model "${model}" \
    --action train \
    "${DRY_RUN[@]}"
done

for model in m ft xl; do
  "${PYTHON}" "${ROOT}/training/run_nnunet.py" \
    --registry "${REGISTRY}" \
    --model "${model}" \
    --action validate \
    "${DRY_RUN[@]}"
done
