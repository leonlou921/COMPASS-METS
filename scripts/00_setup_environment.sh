#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=common.sh
source "${ROOT}/scripts/common.sh"

require_env nnUNet_raw nnUNet_preprocessed nnUNet_results
for path_name in nnUNet_raw nnUNet_preprocessed nnUNet_results; do
  mkdir -p "${!path_name}"
done

"${PYTHON}" -c "import brats_mets; import nnunetv2; print('environment OK')"
