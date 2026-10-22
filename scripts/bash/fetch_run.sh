#!/usr/bin/env bash
set -euo pipefail
RUN=${1:?Usage: fetch_run.sh MMDD-HHMM-NAME [DESTINATION]}
[[ $RUN =~ ^[0-9]{4}-[0-9]{4}-[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || { echo 'Invalid experiment directory name' >&2; exit 2; }
DEST=${2:-./runs}
mkdir -p -- "$DEST"
[[ ! -e $DEST/$RUN ]] || { echo 'Destination already exists' >&2; exit 2; }
scp -r "${ASCEND_SSH_HOST:-huawei-blue-npu}:/home/s84435910/docker_workspace/ascend-ops-development/runs/$RUN" "$DEST/"
