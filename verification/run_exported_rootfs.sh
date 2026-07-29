#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "$#" -ne 4 ]]; then
  echo "usage: $0 EXPORTED_ROOTFS RAW_INPUT_DIR EMPTY_OUTPUT_DIR WORK_DIR"
  echo "runs N03_FINAL_UTILITY_V4 from an exported image rootfs without Docker"
  [[ "${1:-}" == "--help" ]] && exit 0
  exit 2
fi

N03_RUNTIME_ROOT="$(readlink -f "$1")"
INPUT_ROOT="$(readlink -f "$2")"
OUTPUT_ROOT="$(readlink -m "$3")"
WORK_ROOT="$(readlink -m "$4")"

test -x "${N03_RUNTIME_ROOT}/opt/conda/bin/python"
test -f "${N03_RUNTIME_ROOT}/opt/conda/bin/nnUNetv2_predict"
test -d "${N03_RUNTIME_ROOT}/opt/n03/assets"
test -d "${INPUT_ROOT}"

mkdir -p "${OUTPUT_ROOT}" "${WORK_ROOT}"
if find "${OUTPUT_ROOT}" -mindepth 1 -print -quit | grep -q .; then
  echo "output directory must be empty: ${OUTPUT_ROOT}" >&2
  exit 1
fi

RUNNER="${WORK_ROOT}/run_nnunet_predict_from_exported_rootfs.sh"
cat >"${RUNNER}" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
: "${N03_RUNTIME_ROOT:?N03_RUNTIME_ROOT is required}"
exec \
  "${N03_RUNTIME_ROOT}/opt/conda/bin/python" \
  "${N03_RUNTIME_ROOT}/opt/conda/bin/nnUNetv2_predict" \
  "$@"
EOF
chmod 700 "${RUNNER}"

export N03_RUNTIME_ROOT
export PYTHONPATH="${N03_RUNTIME_ROOT}/opt/n03/app:${N03_RUNTIME_ROOT}/opt/n03/vendor/lcv1:${N03_RUNTIME_ROOT}/opt/n03/vendor/portfolio:${N03_RUNTIME_ROOT}/opt/n03/vendor/pipeline"
export nnUNet_raw="${N03_RUNTIME_ROOT}/opt/n03/empty/nnUNet_raw"
export nnUNet_preprocessed="${N03_RUNTIME_ROOT}/opt/n03/empty/nnUNet_preprocessed"
export nnUNet_results="${N03_RUNTIME_ROOT}/opt/n03/assets/nnUNet_results"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export PYTHONUNBUFFERED=1

"${N03_RUNTIME_ROOT}/opt/conda/bin/python" - \
  "${INPUT_ROOT}" "${OUTPUT_ROOT}" "${WORK_ROOT}" "${RUNNER}" <<'PY'
from pathlib import Path
import json
import sys
import time

from n03_docker.pipeline import run_pipeline

input_root, output_root, work_root, runner = map(Path, sys.argv[1:])
started = time.time()
report = run_pipeline(
    input_root=input_root,
    output_root=output_root,
    assets_root=Path(__import__("os").environ["N03_RUNTIME_ROOT"]) / "opt/n03/assets",
    work_parent=work_root,
    executable=str(runner),
)
if report.get("candidate") != "N03_FINAL_UTILITY_V4":
    raise RuntimeError(f"unexpected candidate: {report.get('candidate')}")
report["elapsed_seconds"] = time.time() - started
destination = work_root / "exported_rootfs_run.json"
destination.write_text(
    json.dumps(report, indent=2, sort_keys=True),
    encoding="utf-8",
)
print(json.dumps(report, sort_keys=True), flush=True)
PY
