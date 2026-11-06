#!/usr/bin/env bash
# Build the vllm-ascend custom operators on Ascend.
# Run inside the container containing the configured CANN / torch_npu
# environment (e.g. li-sfa-debug).
set -euo pipefail

SOC_OVERRIDE=""
PACKAGE_OPS=""
PACKAGE_VENDOR=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --soc|--package-ops|--vendor)
            [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || { echo "Missing value for $1" >&2; exit 2; }
            case "$1" in
                --soc) SOC_OVERRIDE=$2 ;;
                --package-ops) PACKAGE_OPS=$2 ;;
                --vendor) PACKAGE_VENDOR=$2 ;;
            esac
            shift 2 ;;
        --help)
            echo 'Usage: build_operator_fixed.sh [--soc SOC] [--package-ops "op;dependency;..."] [--vendor NAME]'
            echo 'An explicit package list requires a vendor and installs to _cann_ops_NAME.'
            exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
if [[ -n "$PACKAGE_OPS" || -n "$PACKAGE_VENDOR" ]]; then
    [[ -n "$PACKAGE_OPS" && "$PACKAGE_VENDOR" =~ ^[A-Za-z][A-Za-z0-9_]*$ ]] || {
        echo 'Specify both --package-ops and a valid --vendor name' >&2; exit 2;
    }
fi

# Run from the project directory containing op-src/. This is trusted Bash config.
CONFIG_FILE="$PWD/op-src/builder.env"
[[ -f "$CONFIG_FILE" ]] || {
    echo "Missing configuration: $CONFIG_FILE" >&2; exit 1;
}

# Require every key in builder.env, not inherited from the shell.
unset SRC_DIR DST_DIR OP_NAME SOC_ARG
source "$CONFIG_FILE"
[[ -n "${SRC_DIR:-}" ]] || { echo "SRC_DIR is missing or empty in $CONFIG_FILE" >&2; exit 1; }
[[ -n "${DST_DIR:-}" ]] || { echo "DST_DIR is missing or empty in $CONFIG_FILE" >&2; exit 1; }
[[ -n "${OP_NAME:-}" ]] || { echo "OP_NAME is missing or empty in $CONFIG_FILE" >&2; exit 1; }
[[ -n "${SOC_ARG:-}" ]] || { echo "SOC_ARG is missing or empty in $CONFIG_FILE" >&2; exit 1; }
if [[ -n "$SOC_OVERRIDE" ]]; then
    SOC_ARG=$SOC_OVERRIDE
fi
if [[ -n "$PACKAGE_OPS" && ";$PACKAGE_OPS;" != *";$OP_NAME;"* ]]; then
    echo "The explicit package list must include OP_NAME '$OP_NAME'" >&2; exit 2
fi

# Relative source paths are relative to op-src; destination must be absolute.
[[ "$SRC_DIR" == /* ]] || SRC_DIR="$(dirname -- "$CONFIG_FILE")/$SRC_DIR"
[[ -d "$SRC_DIR" ]] || { echo "Source directory not found: $SRC_DIR" >&2; exit 1; }
SRC_DIR=$(cd -- "$SRC_DIR" && pwd -P)
[[ "$DST_DIR" == /* ]] || { echo "DST_DIR must be an absolute path" >&2; exit 1; }

PYTHON=${PYTHON:-python3}
export MAX_JOBS=${MAX_JOBS:-8}

[[ -d "$(dirname -- "$DST_DIR")" ]] || { echo "Operator parent directory not found: $DST_DIR" >&2; exit 1; }
DST_PARENT=$(cd -- "$(dirname -- "$DST_DIR")" && pwd -P)
DST_DIR="$DST_PARENT/$(basename -- "$DST_DIR")"

# Reject copying a directory into itself, its parent, or its child.
if [[ "$SRC_DIR" == "$DST_DIR" || "$SRC_DIR" == "$DST_DIR/"* || "$DST_DIR" == "$SRC_DIR/"* ]]; then
    echo "SRC_DIR and DST_DIR must be separate, non-overlapping directories" >&2
    exit 1
fi
[[ -d "$SRC_DIR/op_host" && -d "$SRC_DIR/op_kernel" ]] || {
    echo "SRC_DIR must contain the complete operator: op_host and op_kernel" >&2; exit 1;
}

# Expected structure: <repository>/csrc/attention/<operator>
VLLM_ROOT=$(cd -- "$DST_PARENT/../.." && pwd -P)
[[ "$DST_DIR" == "$VLLM_ROOT/csrc/attention/$(basename -- "$DST_DIR")" ]] || {
    echo "Expected DST_DIR under <vllm-ascend>/csrc/attention/" >&2; exit 1;
}
[[ "$(basename -- "$DST_DIR")" == "$OP_NAME" ]] || {
    echo "OP_NAME '$OP_NAME' does not match operator directory '$(basename -- "$DST_DIR")'" >&2; exit 1;
}
[[ -f "$VLLM_ROOT/setup.py" && -f "$VLLM_ROOT/csrc/build_aclnn.sh" ]] || {
    echo "Missing vLLM-Ascend setup.py or csrc/build_aclnn.sh in $VLLM_ROOT" >&2; exit 1;
}
command -v "$PYTHON" >/dev/null

# Copy directory CONTENTS (including hidden files), not a nested SRC_DIR directory.
# Existing matching files are overwritten; destination-only files are retained.
printf 'Copying %s -> %s\n' "$SRC_DIR" "$DST_DIR"
mkdir -p -- "$DST_DIR"
cp -a -- "$SRC_DIR/." "$DST_DIR/"
[[ -d "$DST_DIR/op_host" && -d "$DST_DIR/op_kernel" ]] || {
    echo "Copied operator is missing op_host or op_kernel" >&2; exit 1;
}

cd -- "$VLLM_ROOT"
export COMPILE_CUSTOM_KERNELS=1

printf 'Operator sources: %s\nRepository: %s\n' "$DST_DIR" "$VLLM_ROOT"

# Preserve previous build artifacts. build.sh works under
# <vllm-ascend>/csrc/{build,output,build_out} (not the repository root!).
# build_aclnn.sh recreates output/build_out itself; move any leftovers aside so
# a stale state cannot leak into the new build. csrc/build (the CMake/Ninja
# cache) is left in place so CMake can regenerate incrementally.
BACKUP_DIR=$(mktemp -d "$PWD/build-backup.XXXXXX")
for dir in output build_out; do
    if [[ -e "csrc/$dir" ]]; then
        mv -- "csrc/$dir" "$BACKUP_DIR/"
    fi
done
echo "Previous artifacts: $BACKUP_DIR"

# Build the custom operators.
#
# NOTE: vllm-ascend's CMake (cmake/symbol.cmake:gen_cust_aicpu_kernel_symbol)
# references the AICPU metadata targets of the ENTIRE custom-op set. Building a
# single operator with  --ops=<op>  therefore fails during the CMake generate
# step, e.g.:
#
#   CMake Error at cmake/symbol.cmake:253 (add_custom_command):
#     Error evaluating generator expression:
#       $<TARGET_OBJECTS:vllm_quant_lightning_indexer_metadata_obj>
#     Objects of target "vllm_quant_lightning_indexer_metadata_obj" referenced
#     but no such target exists.
#
# So build the full custom-op set for the target SoC through the project's own
# entry point (csrc/build_aclnn.sh), exactly as setup.py's build_ext does. That
# script also installs the produced .run package into
# vllm_ascend/_cann_ops_custom. If the operator's op_api/adapter signature
# changed, additionally rebuild the PyTorch extension separately, e.g.:
#   "$PYTHON" -m pip install -v --no-build-isolation --no-deps -e .
BUILD_ACLNN="${VLLM_ROOT}/csrc/build_aclnn.sh"
SOC_VERSION="${SOC_ARG:-${SOC_VERSION:-}}"
case "$SOC_VERSION" in
    ascend310*|ascend910b*|ascend910_93*|ascend950*)
        ;;
    *)
        echo "SOC_ARG '$SOC_VERSION' is not supported by csrc/build_aclnn.sh" >&2
        exit 1
        ;;
esac

# Force a recompile of the target operator's device kernels.
#
# The kernel-compilation rules emitted by cmake/symbol.cmake are CUSTOM_COMMAND
# rules with only ORDER-ONLY dependencies (no explicit inputs). Ninja therefore
# treats a kernel as up-to-date as long as its *.done marker exists, and will
# NOT recompile the device kernel when op_kernel sources change: the operator
# silently keeps running the previously built binary. Since csrc/build (the
# CMake/Ninja cache) is preserved across builds, remove the stale markers and
# kernel outputs for the target operator so the freshly copied sources are
# actually recompiled.
BUILD_DIR="$VLLM_ROOT/csrc/build/binary/${SOC_VERSION}"
if [[ -d "$BUILD_DIR" ]]; then
    for marker in "$BUILD_DIR"/gen/"${OP_NAME}"_"${SOC_VERSION}"_*.done; do
        if [[ -e "$marker" ]]; then
            rm -f -- "$marker"
            echo "Removed stale kernel marker: $marker"
        fi
    done
    if [[ -d "$BUILD_DIR/bin/$OP_NAME" ]]; then
        rm -rf -- "$BUILD_DIR/bin/$OP_NAME"
        echo "Removed stale kernel outputs: $BUILD_DIR/bin/$OP_NAME"
    fi
    src_copy_marker="$BUILD_DIR/src/$OP_NAME/${OP_NAME}_${SOC_VERSION}_src_copy.done"
    if [[ -e "$src_copy_marker" ]]; then
        rm -f -- "$src_copy_marker"
        echo "Removed stale source-copy marker: $src_copy_marker"
    fi
fi

printf 'Building custom ops for SoC %s\n' "$SOC_VERSION"
if [[ -n "$PACKAGE_OPS" ]]; then
    # Build the explicitly selected operator and metadata dependencies into an
    # independent vendor package. Preserve the existing installed custom ops.
    if [[ -d "$VLLM_ROOT/csrc/build" ]]; then
        mv -- "$VLLM_ROOT/csrc/build" "$BACKUP_DIR/build"
    fi
    (
        cd -- "$VLLM_ROOT/csrc"
        printf 'Explicit package: ops=%s vendor=%s soc=%s\n' "$PACKAGE_OPS" "$PACKAGE_VENDOR" "$SOC_VERSION"
        bash build.sh --pkg "--ops=$PACKAGE_OPS" "--soc=$SOC_VERSION" \
            "--vendor_name=$PACKAGE_VENDOR" "-j$MAX_JOBS" -O3
        shopt -s nullglob
        installers=(./build/cann-ops-transformer*.run)
        [[ ${#installers[@]} -eq 1 ]] || { echo 'Expected exactly one package installer' >&2; exit 1; }
        install_dir="$VLLM_ROOT/vllm_ascend/_cann_ops_$PACKAGE_VENDOR"
        bash "${installers[0]}" "--install-path=$install_dir"
        printf 'Installed explicit package: %s\n' "$install_dir"
    )
else
    bash "$BUILD_ACLNN" "$VLLM_ROOT" "$SOC_VERSION"
fi

printf '\nBuild/install finished. Restart model workers before using the new version.\n'
printf 'Verify that the build log includes %s for your target SoC.\n' "$OP_NAME"
