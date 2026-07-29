#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
ARCHIVE="${ARCHIVE:-${ROOT}/artifacts/brats-mets-n03-final.docker.tar}"
VERIFY_REPORT="${VERIFY_REPORT:-${ROOT}/work/frozen-equivalence/frozen_equivalence.json}"
RELEASE_MANIFEST="${RELEASE_MANIFEST:-${ROOT}/artifacts/release_manifest.json}"

if [[ "${1:-}" == "--help" ]]; then
  echo "usage: $0"
  echo "fails closed unless the external 179-case frozen equivalence gate passed"
  exit 0
fi
[[ "$#" -eq 0 ]] || { echo "unexpected arguments" >&2; exit 2; }
test -f "${ARCHIVE}"
test -f "${VERIFY_REPORT}"

"${PYTHON}" - "${VERIFY_REPORT}" "${ARCHIVE}" "${RELEASE_MANIFEST}" <<'PY'
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys

report_path, archive_path, output_path = map(Path, sys.argv[1:])
report = json.loads(report_path.read_text(encoding="utf-8"))
if report.get("candidate_id") != "N03_FINAL_UTILITY_V4":
    raise SystemExit("verification candidate is not N03_FINAL_UTILITY_V4")
if report.get("passed") is not True:
    raise SystemExit("frozen equivalence gate did not pass")
if report.get("case_count") != 179 or report.get("different_voxels") != 0:
    raise SystemExit("release requires 179 cases and zero differing voxels")

digest = hashlib.sha256()
with archive_path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
payload = {
    "candidate_id": "N03_FINAL_UTILITY_V4",
    "archive": archive_path.name,
    "archive_sha256": digest.hexdigest(),
    "archive_size_bytes": archive_path.stat().st_size,
    "frozen_equivalence_report": report_path.name,
    "frozen_equivalence_passed": True,
    "case_count": 179,
    "different_voxels": 0,
}
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, sort_keys=True))
PY
