#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
BUILD=${1:?Usage: operator_exec.sh BUILD_DIRECTORY COMMAND [ARG...]}
shift
[[ $# -gt 0 ]] || { echo 'A command is required' >&2; exit 2; }
bash "$ROOT/scripts/bash/container_exec.sh" python3 scripts/python/activation.py "$BUILD" "$@"
