#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/nnUNet:${PYTHONPATH:-}"

require_env() {
  local name
  for name in "$@"; do
    if [[ -z "${!name:-}" ]]; then
      echo "required environment variable is unset: ${name}" >&2
      return 2
    fi
  done
}
