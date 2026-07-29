#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=common.sh
source "${ROOT}/scripts/common.sh"
require_env nnUNet_raw nnUNet_preprocessed nnUNet_results

FOLDS=("$@")
[[ ${#FOLDS[@]} -gt 0 ]] || FOLDS=(0 1 2 3 4)
EXTRA=()
[[ "${DRY_RUN:-0}" == 1 ]] && EXTRA+=(--dry-run)
[[ "${CONTINUE_TRAINING:-0}" == 1 ]] && EXTRA+=(--continue-training)
for fold in "${FOLDS[@]}"; do
  "${PYTHON}" -m compass_mets.training.run_nnunet \
    --config "${ROOT}/configs/trainers/resencm.json" \
    --fold "${fold}" --action train "${EXTRA[@]}"
done
