"""Read installed versions and profiler help without launching a workload."""
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def capture(argv):
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=30)
        return {"argv": argv, "returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"argv": argv, "error": str(exc)}


def collect():
    versions = {}
    for package in ("torch", "torch-npu", "vllm", "vllm-ascend"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "python": sys.version, "architecture": platform.machine(), "versions": versions,
        "environment": {key: os.environ.get(key) for key in (
            "ASCEND_HOME_PATH", "ASCEND_RT_VISIBLE_DEVICES", "ASCEND_PHYSICAL_DEVICE_ID", "ASCEND_DEVICE_ID", "ASCEND_CUSTOM_OPP_PATH", "TZ")},
        "tools": {key: shutil.which(key) for key in ("bisheng", "cmake", "msopgen", "msprof", "npu-smi")},
        "msprof_help": capture(["msprof", "--help"]),
        "msprof_op_help": capture(["msprof", "op", "--help"]),
        "simulator_help": capture(["msprof", "op", "simulator", "--help"]),
        "npu_smi": capture(["npu-smi", "info"]),
    }


if __name__ == "__main__":
    report = collect()
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(json.dumps(report, indent=2) + "\n")
        print(sys.argv[1])
    else:
        print(json.dumps(report, indent=2))
