#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"

if [[ "${1:-}" == "--help" || "$#" -lt 1 || "$#" -gt 3 ]]; then
  echo "usage: $0 PREDICTION_DIR [SUBMISSION_ZIP] [MANIFEST_JSON]"
  echo "validates 179 outputs and creates a flat CRC-checked submission ZIP"
  [[ "${1:-}" == "--help" ]] && exit 0
  exit 2
fi

PREDICTION_DIR="$1"
SUBMISSION_ZIP="${2:-${ROOT}/artifacts/N03_FINAL_UTILITY_V4_submission.zip}"
MANIFEST_JSON="${3:-${SUBMISSION_ZIP%.zip}.json}"

"${PYTHON}" "${ROOT}/verification/package_submission.py" \
  --prediction-root "${PREDICTION_DIR}" \
  --destination "${SUBMISSION_ZIP}" \
  --expected-cases 179 \
  --manifest "${MANIFEST_JSON}"
