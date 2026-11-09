#!/usr/bin/env bash
# Run inside the container containing the configured CANN / torch_npu environment.
set -euo pipefail

# Run from the project directory containing op-src/. This is trusted Bash config.
CONFIG_FILE="$PWD/op-src/builder.env"
[[ -f "$CONFIG_FILE" ]] || {
    echo "Missing configuration: $CONFIG_FILE" >&2; exit 1;
}

# Require both paths in builder.env, not inherited from the shell.
unset SRC_DIR DST_DIR
source "$CONFIG_FILE"
[[ -n "${SRC_DIR:-}" ]] || { echo "SRC_DIR is missing or empty in $CONFIG_FILE" >&2; exit 1; }
[[ -n "${DST_DIR:-}" ]] || { echo "DST_DIR is missing or empty in $CONFIG_FILE" >&2; exit 1; }

# Relative source paths are relative to op-src; destination must be absolute.
[[ "$SRC_DIR" == /* ]] || SRC_DIR="$(dirname -- "$CONFIG_FILE")/$SRC_DIR"
[[ -d "$SRC_DIR" ]] || { echo "Source directory not found: $SRC_DIR" >&2; exit 1; }
SRC_DIR=$(cd -- "$SRC_DIR" && pwd -P)
[[ "$DST_DIR" == /* ]] || { echo "DST_DIR must be an absolute path" >&2; exit 1; }

PYTHON=${PYTHON:-python3}
export MAX_JOBS=${MAX_JOBS:-8}

[[ -d "$DST_DIR" ]] || { echo "Operator directory not found: $DST_DIR" >&2; exit 1; }
DST_DIR=$(cd -- "$DST_DIR" && pwd -P)

# Reject copying a directory into itself, its parent, or its child.
if [[ "$SRC_DIR" == "$DST_DIR" || "$SRC_DIR" == "$DST_DIR/"* || "$DST_DIR" == "$SRC_DIR/"* ]]; then
    echo "SRC_DIR and DST_DIR must be separate, non-overlapping directories" >&2
    exit 1
fi
[[ -d "$SRC_DIR/op_host" && -d "$SRC_DIR/op_kernel" ]] || {
    echo "SRC_DIR must contain the complete operator: op_host and op_kernel" >&2; exit 1;
}

# Expected structure: <repository>/csrc/attention/<operator>
VLLM_ROOT=$(cd -- "$DST_DIR/../../.." && pwd -P)
[[ "$DST_DIR" == "$VLLM_ROOT/csrc/attention/$(basename -- "$DST_DIR")" ]] || {
    echo "Expected DST_DIR under <vllm-ascend>/csrc/attention/" >&2; exit 1;
}
[[ -f "$VLLM_ROOT/setup.py" && -f "$VLLM_ROOT/csrc/build_aclnn.sh" ]] || {
    echo "Missing vLLM-Ascend setup.py or csrc/build_aclnn.sh in $VLLM_ROOT" >&2; exit 1;
}
command -v "$PYTHON" >/dev/null

# Copy directory CONTENTS (including hidden files), not a nested SRC_DIR directory.
# Existing matching files are overwritten; destination-only files are retained.
printf 'Copying %s -> %s\n' "$SRC_DIR" "$DST_DIR"
cp -a -- "$SRC_DIR/." "$DST_DIR/"
[[ -d "$DST_DIR/op_host" && -d "$DST_DIR/op_kernel" ]] || {
    echo "Copied operator is missing op_host or op_kernel" >&2; exit 1;
}

cd -- "$VLLM_ROOT"
export COMPILE_CUSTOM_KERNELS=1

printf 'Operator sources: %s\nRepository: %s\n' "$DST_DIR" "$VLLM_ROOT"
printf 'SOC_VERSION: %s\nMAX_JOBS: %s\n' "${SOC_VERSION:-not set; handled by local setup.py}" "$MAX_JOBS"

BACKUP_DIR=$(mktemp -d "$PWD/build-backup.XXXXXX")

for dir in build output build_out; do
    if [[ -e "$dir" ]]; then
        mv -- "$dir" "$BACKUP_DIR/"
    fi
done

echo "Previous artifacts: $BACKUP_DIR"

set -o pipefail

# The project's build_ext invokes build_aclnn.sh to build/install custom ops,
# then builds the PyTorch extension. Keep the environment's existing dependencies.
# This rebuilds the configured custom-op set, not only the selected operator.
# "$PYTHON" -m pip install -v --no-build-isolation --no-deps -e .
bash "${VLLM_ROOT}/csrc/build.sh" \
  --pkg \
  --ops="${OP_NAME}" \
  --soc="${SOC_ARG}"

printf '\nBuild/install finished. Restart model workers before using the new version.\n'
printf 'Verify that the build log includes lightning_indexer for your target SoC.\n'
