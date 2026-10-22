#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
REF=${1:?Usage: switch_operator.sh TAG_OR_BRANCH_OR_SHA [--fetch]}
[[ $REF != -* ]] || { echo 'Invalid revision' >&2; exit 2; }
if [[ ! -e $ROOT/op-src/.git ]]; then git -C "$ROOT" submodule update --init -- op-src; fi
[[ -z $(git -C "$ROOT/op-src" status --porcelain) ]] || { echo 'Operator has local changes; refusing to overwrite' >&2; exit 2; }
if [[ ${2:-} == --fetch ]]; then git -C "$ROOT/op-src" fetch --tags origin; fi
SHA=$(git -C "$ROOT/op-src" rev-parse --verify "${REF}^{commit}")
git -C "$ROOT/op-src" checkout --detach "$SHA"
git -C "$ROOT" add -- op-src
echo "Operator pinned to $SHA. Commit the staged submodule pointer in the framework repository."
