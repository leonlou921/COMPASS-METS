#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
ARCHIVE="${ARCHIVE:-${ROOT}/artifacts/brats-mets-n03-final.docker.tar}"
VERIFY_REPORT="${VERIFY_REPORT:-${ROOT}/work/frozen-equivalence/frozen_equivalence.json}"
RELEASE_MANIFEST="${RELEASE_MANIFEST:-${ROOT}/artifacts/release_manifest.json}"

if [[ "${1:-}" == "--help" ]]; then
  echo "usage: $0 [--accept-recorded-difference]"
  echo "exact equivalence is the default; the explicit flag records an approved"
  echo "nonzero voxel difference only after all 179 structural checks pass"
  exit 0
fi
accept_recorded_difference=false
if [[ "${1:-}" == "--accept-recorded-difference" ]]; then
  accept_recorded_difference=true
  shift
fi
[[ "$#" -eq 0 ]] || { echo "unexpected arguments" >&2; exit 2; }
test -f "${ARCHIVE}"
test -f "${VERIFY_REPORT}"

"${PYTHON}" - \
  "${VERIFY_REPORT}" "${ARCHIVE}" "${RELEASE_MANIFEST}" \
  "${accept_recorded_difference}" "${ROOT}" <<'PY'
from __future__ import annotations
import json
from pathlib import Path
import sys

report_path, archive_path, output_path = map(Path, sys.argv[1:4])
accept_recorded_difference = sys.argv[4].lower() == "true"
root = Path(sys.argv[5])
sys.path.insert(0, str(root))

from verification.release_manifest import build_release_manifest

report = json.loads(report_path.read_text(encoding="utf-8"))
payload = build_release_manifest(
    report,
    archive_path,
    accept_recorded_difference=accept_recorded_difference,
)
payload["frozen_equivalence_report"] = report_path.name
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, sort_keys=True))
PY
