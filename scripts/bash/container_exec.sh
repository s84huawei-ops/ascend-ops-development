#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
NAME=${ASCEND_CONTAINER:-ascend-ops-dev-npu${ASCEND_DEVICE_ID:-6}}
[[ $# -gt 0 ]] || { echo "Usage: $0 COMMAND [ARG...]" >&2; exit 2; }
IMAGE_ID=$(docker inspect -f '{{.Image}}' "$NAME")
IMAGE_REF=$(docker inspect -f '{{.Config.Image}}' "$NAME")
PHYSICAL_DEVICE=$(docker inspect -f '{{index .Config.Labels "ascend-ops.device"}}' "$NAME")
docker exec -i -w /workspace -e "ASCEND_IMAGE_ID=$IMAGE_ID" -e "ASCEND_IMAGE_REFERENCE=$IMAGE_REF" \
    -e "ASCEND_PHYSICAL_DEVICE_ID=$PHYSICAL_DEVICE" \
    "$NAME" bash /workspace/scripts/bash/env.sh "$@"
