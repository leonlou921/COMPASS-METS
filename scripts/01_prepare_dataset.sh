#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=common.sh
source "${ROOT}/scripts/common.sh"

if [[ "${1:-}" == "--help" ]]; then
  echo "usage: BRATS_METS_TRAIN_DIR=... BRATS_METS_VALID_DIR=... $0"
  exit 0
fi
[[ "$#" -eq 0 ]] || { echo "unexpected arguments" >&2; exit 2; }

require_env BRATS_METS_TRAIN_DIR BRATS_METS_VALID_DIR nnUNet_raw

CORRECTED_ARGS=()
if [[ -n "${BRATS_METS_CORRECTED_LABELS_DIR:-}" ]]; then
  CORRECTED_ARGS=(--corrected-labels-dir "${BRATS_METS_CORRECTED_LABELS_DIR}")
fi

"${PYTHON}" -m brats_mets.data.prepare_dataset501 \
  --train-dir "${BRATS_METS_TRAIN_DIR}" \
  --valid-dir "${BRATS_METS_VALID_DIR}" \
  --nnunet-raw "${nnUNet_raw}" \
  "${CORRECTED_ARGS[@]}"
