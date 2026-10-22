#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
bash "$ROOT/docker_run.sh" "${ASCEND_IMAGE:-}" "${ASCEND_CONTAINER:-ascend-ops-dev-npu${ASCEND_DEVICE_ID:-6}}"
bash "$ROOT/scripts/bash/container_exec.sh" bash op-src/build.sh "${1:-/workspace/.cache/manual-build}"
