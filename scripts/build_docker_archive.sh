#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BUILDKIT_VERSION="${BUILDKIT_VERSION:-v0.31.2}"
BK="${ROOT}/tools/buildkit-${BUILDKIT_VERSION}/bin"
STATE="${ROOT}/work/buildkit-build"
SOCKET="${STATE}/buildkitd.sock"
ARTIFACTS="${ROOT}/artifacts"
IMAGE_NAME="${IMAGE_NAME:-brats-mets-n03:final}"
ARCHIVE="${ARTIFACTS}/brats-mets-n03-final.docker.tar"

test -x "${BK}/buildkitd"
test -x "${BK}/buildctl"
test -d "${ROOT}/assets"
test -d "${ROOT}/vendor/nnUNet"

mkdir -p "${STATE}/state" "${ARTIFACTS}" "${ROOT}/logs"
rm -f "${SOCKET}"

env PATH="${BK}:${PATH}" \
  unshare --propagation unchanged -Urm \
  "${BK}/buildkitd" \
    --rootless \
    --root "${STATE}/state" \
    --addr "unix://${SOCKET}" \
    --oci-worker-snapshotter=native \
    --oci-worker-no-process-sandbox \
    >"${ROOT}/logs/buildkitd.log" 2>&1 &
daemon_pid=$!
cleanup() {
  kill "${daemon_pid}" 2>/dev/null || true
  wait "${daemon_pid}" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 60); do
  if "${BK}/buildctl" --addr "unix://${SOCKET}" debug workers >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
"${BK}/buildctl" --addr "unix://${SOCKET}" debug workers >/dev/null

rm -f "${ARCHIVE}"
"${BK}/buildctl" \
  --addr "unix://${SOCKET}" \
  build \
  --progress=plain \
  --frontend dockerfile.v0 \
  --local context="${ROOT}" \
  --local dockerfile="${ROOT}/docker" \
  --opt filename=Dockerfile \
  --opt platform=linux/amd64 \
  --output "type=docker,name=${IMAGE_NAME},dest=${ARCHIVE}"

sha256sum "${ARCHIVE}" >"${ARCHIVE}.sha256"
stat -c '{"archive":"%n","size_bytes":%s}' "${ARCHIVE}"
cat "${ARCHIVE}.sha256"
