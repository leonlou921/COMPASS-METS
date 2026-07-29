#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"

if [[ "${1:-}" == "--help" ]]; then
  echo "usage: $0"
  echo "builds the offline N03_FINAL_UTILITY_V4 Docker archive with rootless BuildKit"
  exit 0
fi
[[ "$#" -eq 0 ]] || { echo "unexpected arguments" >&2; exit 2; }

test -d "${ROOT}/third_party/nnUNet"
test -f "${ROOT}/third_party/nnUNet/LICENSE"
test -d "${ROOT}/assets"
if find "${ROOT}/assets" -type f \
  \( -name '*.nii' -o -name '*.nii.gz' -o -name '*.npz' -o \
     -name '*.npy' -o -name '*.zip' -o -name '*.tar' \) \
  -print -quit | grep -q .; then
  echo "assets contain a forbidden prediction, data, or archive file" >&2
  exit 1
fi
"${PYTHON}" "${ROOT}/scripts/verify_release.py" "${ROOT}"

bash "${ROOT}/scripts/build_docker_archive.sh" "${ROOT}"
