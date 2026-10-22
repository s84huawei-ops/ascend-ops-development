"""Activate a build in one process without modifying the container installation."""
import json
import os
from pathlib import Path
import sys


def environment(build_info, base=None):
    result = dict(os.environ if base is None else base)
    result["PYTHONPATH"] = build_info["pythonpath"] + os.pathsep + result.get("PYTHONPATH", "")
    result["ASCEND_CUSTOM_OPP_PATH"] = build_info["opp"]
    libdirs = sorted({str(path.parent) for path in Path(build_info["opp"]).rglob("*.so")})
    result["LD_LIBRARY_PATH"] = os.pathsep.join(libdirs) + os.pathsep + result.get("LD_LIBRARY_PATH", "")
    return result


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit("Usage: activation.py BUILD_DIRECTORY COMMAND [ARG...]")
    info = json.loads((Path(sys.argv[1]) / "build.json").read_text())
    os.execvpe(sys.argv[2], sys.argv[2:], environment(info))
