#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ARTIFACTS="${ROOT}/artifacts"
LOGS="${ROOT}/logs"
WORK="${ROOT}/work/kaniko-build"
KANIKO_ROOT="${WORK}/rootfs"
CONTEXT="${KANIKO_ROOT}/kaniko/context"
ARCHIVE="${ARTIFACTS}/brats-mets-n03-final.docker.tar"

: "${KANIKO_ROOTFS_TAR:?set KANIKO_ROOTFS_TAR to the Kaniko debug rootfs tar}"
: "${REGISTRY_BINARY:?set REGISTRY_BINARY to the registry executable}"
: "${REGISTRY_CONFIG:?set REGISTRY_CONFIG to the registry configuration}"
: "${CRANE:?set CRANE to the crane executable}"
: "${BASE_IMAGE:?set BASE_IMAGE to the pinned base image reference}"

IMAGE_REF="${IMAGE_REF:-127.0.0.1:5000/n03/final-utility-v4:local}"
CACHE_REPO="${CACHE_REPO:-127.0.0.1:5000/n03/cache-final-utility-v4}"
GOMEMLIMIT="${GOMEMLIMIT:-12GiB}"
GOGC="${GOGC:-20}"

test -f "${KANIKO_ROOTFS_TAR}"
test -x "${REGISTRY_BINARY}"
test -f "${REGISTRY_CONFIG}"
test -x "${CRANE}"
test -d "${ROOT}/assets"
test -d "${ROOT}/third_party/nnUNet"

case "${KANIKO_ROOT}" in
  "${ROOT}"/work/kaniko-build/rootfs) ;;
  *)
    echo "unsafe Kaniko work directory: ${KANIKO_ROOT}" >&2
    exit 1
    ;;
esac

mkdir -p "${WORK}" "${ARTIFACTS}" "${LOGS}"
if [[ -e "${KANIKO_ROOT}" ]]; then
  rm -rf --one-file-system "${KANIKO_ROOT}"
fi
mkdir -p "${KANIKO_ROOT}"
tar -xf "${KANIKO_ROOTFS_TAR}" -C "${KANIKO_ROOT}"
test -x "${KANIKO_ROOT}/kaniko/executor"

# Kaniko is run in a chroot on restricted hosts where user namespaces and
# mounts are unavailable. It only needs read-only proc metadata and devices.
mkdir -p "${KANIKO_ROOT}/proc/self" "${KANIKO_ROOT}/dev/pts" "${KANIKO_ROOT}/dev/shm"
cp /proc/self/mountinfo "${KANIKO_ROOT}/proc/self/mountinfo"
cp /proc/self/cgroup "${KANIKO_ROOT}/proc/self/cgroup"
cp /proc/mounts "${KANIKO_ROOT}/proc/mounts"
cp /proc/meminfo "${KANIKO_ROOT}/proc/meminfo"
for device in "null 1 3" "zero 1 5" "random 1 8" "urandom 1 9"; do
  read -r name major minor <<<"${device}"
  [[ -e "${KANIKO_ROOT}/dev/${name}" ]] ||
    mknod -m 666 "${KANIKO_ROOT}/dev/${name}" c "${major}" "${minor}"
done
cp --remove-destination /etc/resolv.conf "${KANIKO_ROOT}/etc/resolv.conf"
cp --remove-destination /etc/hosts "${KANIKO_ROOT}/etc/hosts"

mkdir -p "${CONTEXT}"
for directory in docker inference third_party assets provenance; do
  test -e "${ROOT}/${directory}"
  cp -al "${ROOT}/${directory}" "${CONTEXT}/${directory}"
done
cp -a "${ROOT}/.dockerignore" "${CONTEXT}/.dockerignore"

"${REGISTRY_BINARY}" serve "${REGISTRY_CONFIG}" \
  >"${LOGS}/local-registry.log" 2>&1 &
registry_pid=$!
cleanup() {
  kill "${registry_pid}" 2>/dev/null || true
  wait "${registry_pid}" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 60); do
  if "${CRANE}" digest --insecure "${BASE_IMAGE}" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
"${CRANE}" digest --insecure "${BASE_IMAGE}" >/dev/null

env GOMEMLIMIT="${GOMEMLIMIT}" GOGC="${GOGC}" \
  chroot "${KANIKO_ROOT}" /kaniko/executor \
    --force \
    --dockerfile=/kaniko/context/docker/Dockerfile \
    --context=dir:///kaniko/context \
    --build-arg="BASE_IMAGE=${BASE_IMAGE}" \
    --destination="${IMAGE_REF}" \
    --cache=true \
    --cache-repo="${CACHE_REPO}" \
    --cache-copy-layers=true \
    --cache-run-layers=true \
    --compressed-caching=false \
    --snapshot-mode=redo \
    --cleanup \
    --insecure \
    --skip-tls-verify \
    --verbosity=info \
    2>&1 | tee "${LOGS}/kaniko-build.log"

"${CRANE}" validate --insecure --remote "${IMAGE_REF}"
digest="$("${CRANE}" digest --insecure "${IMAGE_REF}")"
rm -f "${ARCHIVE}" "${ARCHIVE}.sha256"
"${CRANE}" pull --insecure --format=legacy "${IMAGE_REF}" "${ARCHIVE}"
sha256sum "${ARCHIVE}" >"${ARCHIVE}.sha256"
printf '{"image":"%s","digest":"%s"}\n' "${IMAGE_REF}" "${digest}" \
  >"${ARTIFACTS}/image_reference.json"

stat -c '{"archive":"%n","size_bytes":%s}' "${ARCHIVE}"
cat "${ARCHIVE}.sha256"
cat "${ARTIFACTS}/image_reference.json"
