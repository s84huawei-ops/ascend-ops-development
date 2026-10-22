#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
CONFIG=$(realpath -- "${1:?Usage: run.sh CONFIG.json}")
[[ $CONFIG == "$ROOT/"* ]] || { echo 'Configuration must be inside this repository' >&2; exit 2; }
DEVICE=$(python3 "$ROOT/scripts/python/experiment.py" "$CONFIG" --device-only)
export ASCEND_DEVICE_ID=$DEVICE
export ASCEND_CONTAINER=${ASCEND_CONTAINER:-ascend-ops-dev-npu${DEVICE}}
bash "$ROOT/docker_run.sh" "${ASCEND_IMAGE:-}" "$ASCEND_CONTAINER"
bash "$ROOT/scripts/bash/container_exec.sh" python3 scripts/python/experiment.py "/workspace/${CONFIG#"$ROOT/"}"
