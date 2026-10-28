#!/usr/bin/env bash
# Adapted from the original docker_run.sh in the server workspace.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
IMAGE=${1:-quay.io/ascend/vllm-ascend:v0.23.0rc1-a3}
DEVICE=${ASCEND_DEVICE_ID:-6}
NAME=${2:-ascend-ops-dev-npu${DEVICE}}
[[ $DEVICE =~ ^[0-9]+$ ]] || { echo 'ASCEND_DEVICE_ID must be an integer' >&2; exit 2; }
[[ -c /dev/davinci${DEVICE} ]] || { echo "Missing /dev/davinci${DEVICE}" >&2; exit 2; }
if docker container inspect "$NAME" >/dev/null 2>&1; then
    [[ $(docker inspect -f '{{index .Config.Labels "ascend-ops.schema"}}' "$NAME") == 1 ]] || {
        echo 'Container settings are outdated; choose a new name or recreate this framework container' >&2; exit 2;
    }
    [[ $(docker inspect -f '{{index .Config.Labels "ascend-ops.workspace"}}' "$NAME") == "$ROOT" ]] || {
        echo "Container $NAME belongs to a different workspace" >&2; exit 2;
    }
    [[ $(docker inspect -f '{{index .Config.Labels "ascend-ops.device"}}' "$NAME") == "$DEVICE" ]] || {
        echo 'Container device differs; choose a new container name' >&2; exit 2;
    }
    [[ $(docker inspect -f '{{.Image}}' "$NAME") == "$(docker image inspect -f '{{.Id}}' "$IMAGE")" ]] || {
        echo 'Container image differs; choose a new container name' >&2; exit 2;
    }
    [[ $(docker inspect -f '{{.State.Running}}' "$NAME") == true ]] || docker start "$NAME"
    echo "$NAME"
    exit 0
fi
args=(run --name "$NAME" -d --shm-size=2g --cap-add=SYS_ADMIN --security-opt seccomp=unconfined
    --label "ascend-ops.workspace=$ROOT" --label "ascend-ops.device=$DEVICE" --label ascend-ops.schema=1
    --device "/dev/davinci${DEVICE}" --device /dev/davinci_manager
    --device /dev/hisi_hdc --device /dev/devmm_svm
    -e "ASCEND_PHYSICAL_DEVICE_ID=$DEVICE" -e ASCEND_DEVICE_ID=0 -e TZ=Europe/Warsaw
    -w /workspace -v "$ROOT:/workspace"
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro
    -v /usr/local/dcmi:/usr/local/dcmi:ro
    -v /usr/share/zoneinfo:/usr/share/zoneinfo:ro
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro
    --entrypoint /bin/bash)
docker "${args[@]}" "$IMAGE" -lc 'exec sleep infinity'
echo "$NAME"
