"""Transfer clean, committed framework/operator versions over one SSH connection.

Uses Git bundles: no GitHub token is copied to the server. Remote changes are
never discarded. The server framework repository must already have a commit.
"""
import argparse
import io
from pathlib import Path
import shlex
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[2]
IMPORT = r'''#!/usr/bin/env bash
set -euo pipefail
upload=$1
root=$2
cd "$root"
git rev-parse --verify HEAD >/dev/null || { echo 'Initialize the server clone with git pull first' >&2; exit 2; }
[[ -z $(git status --porcelain) ]] || { echo 'Server framework has local changes; refusing to overwrite' >&2; exit 2; }
if [[ -e op-src/.git ]]; then
    [[ -z $(git -C op-src status --porcelain) ]] || { echo 'Server operator has local changes' >&2; exit 2; }
fi
git fetch "$upload/framework.bundle" HEAD
git merge --ff-only FETCH_HEAD
git submodule init -- op-src
if [[ ! -e op-src/.git ]]; then
    mkdir -p op-src
    git -C op-src init
fi
if ! git -C op-src remote get-url origin >/dev/null 2>&1; then
    git -C op-src remote add origin "$(git config submodule.op-src.url)"
fi
expected=$(git rev-parse HEAD:op-src)
git -C op-src fetch "$upload/operator.bundle" HEAD 'refs/tags/*:refs/tags/*'
git -C op-src checkout --detach "$expected"
[[ $(git -C op-src rev-parse HEAD) == "$expected" ]] || { echo 'Operator does not match framework pin' >&2; exit 2; }
git submodule absorbgitdirs -- op-src
git status --short --branch
'''


def git(directory, *argv):
    return subprocess.check_output(["git", "-c", f"safe.directory={directory}", "-C", str(directory), *argv], text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="huawei-blue-npu")
    parser.add_argument("--directory", default="/home/s84435910/docker_workspace/ascend-ops-development")
    args = parser.parse_args()
    if args.host.startswith("-") or "\n" in args.host or not args.directory.startswith("/"):
        parser.error("Expected an SSH host alias and an absolute server directory")
    for directory in (ROOT, ROOT / "op-src"):
        if git(directory, "status", "--porcelain"):
            parser.error(f"Commit changes in {directory} before syncing")
    if git(ROOT, "rev-parse", "HEAD:op-src") != git(ROOT / "op-src", "rev-parse", "HEAD"):
        parser.error("Framework commit does not pin the checked-out operator")
    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        for directory, name in ((ROOT, "framework"), (ROOT / "op-src", "operator")):
            revisions = ["HEAD", "--tags"] if name == "operator" else ["HEAD"]
            git(directory, "bundle", "create", str(temporary / f"{name}.bundle"), *revisions)
        payload = temporary / "upload.tar.gz"
        with tarfile.open(payload, "w:gz") as archive:
            for name in ("framework.bundle", "operator.bundle"):
                archive.add(temporary / name, arcname=name)
            script = IMPORT.encode()
            info = tarfile.TarInfo("import.sh")
            info.size = len(script)
            archive.addfile(info, io.BytesIO(script))
        root = shlex.quote(args.directory)
        remote = (f"mkdir -p {root}/.cache && upload=$(mktemp -d {root}/.cache/sync.XXXXXX) && "
                  f'tar -xzf - -C "$upload" && bash "$upload/import.sh" "$upload" {root}')
        with payload.open("rb") as source:
            subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", args.host, remote],
                           stdin=source, check=True)


if __name__ == "__main__":
    main()
