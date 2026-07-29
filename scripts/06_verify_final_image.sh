#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
IMAGE_NAME="${IMAGE_NAME:-brats-mets-n03:final}"
ARCHIVE="${ARCHIVE:-${ROOT}/artifacts/brats-mets-n03-final.docker.tar}"
VERIFY_ROOT="${VERIFY_ROOT:-${ROOT}/work/frozen-equivalence}"

if [[ "${1:-}" == "--help" || "$#" -ne 2 ]]; then
  echo "usage: $0 RAW_INPUT_DIR FROZEN_REFERENCE_ZIP"
  echo "the reference ZIP remains outside the image and is used only as an oracle"
  [[ "${1:-}" == "--help" ]] && exit 0
  exit 2
fi

INPUT_DIR="$(readlink -f "$1")"
REFERENCE_ZIP="$(readlink -f "$2")"
test -d "${INPUT_DIR}"
test -f "${REFERENCE_ZIP}"
test -f "${ARCHIVE}"

mkdir -p "${VERIFY_ROOT}"
for run in run-a run-b; do
  destination="${VERIFY_ROOT}/${run}"
  [[ ! -e "${destination}" ]] || {
    echo "verification output already exists: ${destination}" >&2
    exit 1
  }
  mkdir -p "${destination}"
done

docker load -i "${ARCHIVE}"
for run in run-a run-b; do
  docker run --rm --gpus all --shm-size=16g \
    -v "${INPUT_DIR}:/input:ro" \
    -v "${VERIFY_ROOT}/${run}:/output" \
    "${IMAGE_NAME}"
done

"${PYTHON}" "${ROOT}/verification/verify_frozen_equivalence.py" \
  --reference-zip "${REFERENCE_ZIP}" \
  --candidate-dir "${VERIFY_ROOT}/run-a" \
  --repeat-dir "${VERIFY_ROOT}/run-b" \
  --json-output "${VERIFY_ROOT}/frozen_equivalence.json" \
  --csv-output "${VERIFY_ROOT}/frozen_equivalence.csv"
