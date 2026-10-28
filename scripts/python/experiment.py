"""Validate, build, and run a reproducible Ascend experiment inside its container."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = {
    "name": "exp", "timezone": "Europe/Warsaw", "device": 6, "soc": "Ascend910B2",
    "operator": {"module": "ascend_op_demo", "callable": "add", "ref": None},
    "data": {"shape": [8, 2048], "dtype": "float16", "seed": 17, "rtol": 1e-3, "atol": 1e-5,
             "extra_test_shapes": [[1], [17], [16387]]},
    "benchmark": {"warmup": 5, "iterations": 50},
    "build": {"force": False, "jobs": 2},
    "profiling": {"backends": [], "required": True, "kernel": "AddCustom", "launch_count": 1,
                  "timeout_s": 300, "simulator_shape": [16], "simulator_mode": "binary",
                  "metrics": {"msprof": None, "msprof-op": None, "simulator": None}},
    "allow_dirty": False,
}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def timezone_database_available():
    """Return True when the IANA timezone database can be found locally."""
    import importlib.util
    return Path("/usr/share/zoneinfo").is_dir() or importlib.util.find_spec("tzdata") is not None


def resolve_timezone(name):
    """Resolve an IANA timezone; fall back to UTC when the local database is missing."""
    try:
        return ZoneInfo(name)
    except TypeError:
        raise
    except ZoneInfoNotFoundError:
        if timezone_database_available():
            raise
        return timezone.utc


def configuration(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    def merge(default, supplied):
        if not isinstance(supplied, dict):
            raise ValueError("Configuration sections must be JSON objects")
        unknown = supplied.keys() - default.keys()
        if unknown:
            raise ValueError("Unknown configuration keys: " + ", ".join(sorted(unknown)))
        return {key: merge(value, supplied.get(key, {})) if isinstance(value, dict)
                else supplied.get(key, value) for key, value in default.items()}
    cfg = merge(DEFAULT, raw)
    if not isinstance(cfg["name"], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", cfg["name"]):
        raise ValueError("name must be 1–64 ASCII letters, digits, dots, underscores or hyphens; start with a letter/digit")
    try:
        resolve_timezone(cfg["timezone"])
    except (ZoneInfoNotFoundError, TypeError):
        raise ValueError("Timezone is unavailable; check its name or install tzdata on Windows") from None
    def integer(value, low, high):
        return type(value) is int and low <= value <= high
    if not integer(cfg["device"], 0, 63) or cfg["soc"] != "Ascend910B2":
        raise ValueError("device must be an integer in [0,63]; demo supports Ascend910B2")
    for shape in [cfg["data"]["shape"], cfg["profiling"]["simulator_shape"]] + cfg["data"]["extra_test_shapes"]:
        if not isinstance(shape, list) or not shape or not all(integer(n, 1, 2**31-1) for n in shape):
            raise ValueError("Shapes must be nonempty lists of positive integers")
        size = 1
        for n in shape:
            size *= n
        if size > 2**32 - 1:
            raise ValueError("Demo tensors must have at most UINT32_MAX elements")
    if cfg["data"]["dtype"] != "float16" or not integer(cfg["data"]["seed"], 0, 2**63-1):
        raise ValueError("Demo requires float16 and a nonnegative integer seed")
    for field in ("rtol", "atol"):
        value = cfg["data"][field]
        if type(value) not in (int, float) or not 0 <= value <= 1:
            raise ValueError(field + " must be a finite number in [0,1]")
    for section, key, low, high in (("benchmark", "warmup", 0, 100000), ("benchmark", "iterations", 1, 100000),
                                    ("build", "jobs", 1, 64), ("profiling", "launch_count", 1, 5000),
                                    ("profiling", "timeout_s", 1, 172800)):
        if not integer(cfg[section][key], low, high):
            raise ValueError(f"{section}.{key} must be an integer in [{low},{high}]")
    for value in (cfg["allow_dirty"], cfg["build"]["force"], cfg["profiling"]["required"]):
        if type(value) is not bool:
            raise ValueError("Boolean settings require JSON true or false")
    for value in (cfg["operator"]["module"], cfg["operator"]["callable"]):
        if not isinstance(value, str) or not all(part.isidentifier() for part in value.split(".")):
            raise ValueError("Operator module/callable must be Python identifiers")
    backends = cfg["profiling"]["backends"]
    if cfg["profiling"]["simulator_mode"] not in ("binary", "python"):
        raise ValueError("profiling.simulator_mode must be binary or python")
    for value in cfg["profiling"]["metrics"].values():
        if value is not None and (not isinstance(value, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_,]*", value)):
            raise ValueError("Profiler metrics must be null or comma-separated metric names")
    if not isinstance(backends, list) or any(b not in ("msprof", "msprof-op", "simulator") for b in backends) or len(set(backends)) != len(backends):
        raise ValueError("profiling.backends must contain unique msprof/msprof-op/simulator values")
    if not isinstance(cfg["profiling"]["kernel"], str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", cfg["profiling"]["kernel"]):
        raise ValueError("Invalid kernel filter")
    if cfg["operator"]["ref"] is not None and (not isinstance(cfg["operator"]["ref"], str) or cfg["operator"]["ref"].startswith("-")):
        raise ValueError("operator.ref must be null or a Git revision")
    return cfg


def git_state(path):
    def git(*argv):
        return subprocess.check_output(["git", "-c", f"safe.directory={path}", "-C", str(path), *argv], text=True).strip()
    if Path(git("rev-parse", "--show-toplevel")).resolve() != path.resolve():
        raise RuntimeError(f"{path} must be its own Git repository; initialize the submodule")
    files = git("ls-files", "--cached", "--others", "--exclude-standard").splitlines()
    hashes = {name: hashlib.sha256((path / name).read_bytes()).hexdigest()
              for name in files if (path / name).is_file()}
    return {"sha": git("rev-parse", "HEAD"), "status": git("status", "--porcelain"),
            "diff": git("diff", "HEAD", "--", "."), "path": str(path), "files_sha256": hashes}


def snapshot(state, destination):
    with tarfile.open(destination, "w:gz") as archive:
        for name in state["files_sha256"]:
            archive.add(Path(state["path"]) / name, arcname=name, recursive=False)


def command(argv, log, env=None, timeout=600, cwd=None):
    started = time.perf_counter()
    with log.open("w") as output:
        output.write(json.dumps({"argv": argv}) + "\n")
        output.flush()
        proc = subprocess.Popen(argv, cwd=cwd or ROOT, stdout=output, stderr=subprocess.STDOUT,
                                env=env, start_new_session=True)
        try:
            code = proc.wait(timeout=timeout)
        except BaseException:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            raise
    return {"argv": argv, "returncode": code, "process_wall_s": time.perf_counter() - started,
            "log": str(log.relative_to(ROOT))}


@contextmanager
def device_lock(device):
    import fcntl
    (ROOT / ".locks").mkdir(exist_ok=True)
    with (ROOT / ".locks" / f"npu-{device}.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"NPU {device} is reserved by another experiment")
        yield


def ensure_idle(device, report):
    smi = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=30)
    report.write_text(smi.stdout + smi.stderr)
    if smi.returncode:
        raise RuntimeError("Cannot verify NPU availability; see npu-smi.log")
    if re.search(r"^\|\s*" + str(device) + r"\s+\d+\s+\d+\s+", smi.stdout, re.MULTILINE):
        raise RuntimeError(f"NPU {device} has a running process")


def profile_reports(folder, kernel):
    import csv
    reports, rows = [], []
    needle = kernel.lower().replace("_", "")
    for path in sorted(folder.rglob("*.csv")):
        if not path.stat().st_size:
            continue
        matched = needle in str(path.relative_to(folder)).lower().replace("_", "")
        with path.open(errors="replace", newline="") as source:
            for index, row in enumerate(csv.DictReader(source)):
                if index >= 100000:
                    break
                if matched or needle in json.dumps(row).lower().replace("_", ""):
                    if any(re.search(r"time|duration|cycle|latency", str(key), re.IGNORECASE)
                           and re.fullmatch(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", str(value).strip())
                           for key, value in row.items()):
                        reports.append(str(path.relative_to(folder)))
                        if len(rows) < 30:
                            rows.append({"report": str(path.relative_to(folder)), "values": row})
    return {"matched_reports": sorted(set(reports)), "sample_rows": rows}


def execute(cfg, run):
    from capabilities import collect
    status = {"status": "running", "stages": {}}
    state_file = run / "status.json"
    write_json(state_file, status)
    def record(stage, result):
        if "returncode" in result:
            result.setdefault("status", "ok" if result["returncode"] == 0 else "failed")
        status["stages"][stage] = result
        write_json(state_file, status)
    try:
        with device_lock(cfg["device"]):
            ensure_idle(cfg["device"], run / "logs/npu-smi-before.log")
            capabilities = collect()
            write_json(run / "capabilities.json", capabilities)
            if os.environ.get("ASCEND_PHYSICAL_DEVICE_ID") != str(cfg["device"]):
                raise RuntimeError("Container device mapping differs from config; create/select the correct container")
            framework, operator = git_state(ROOT), git_state(ROOT / "op-src")
            write_json(run / "sources.json", {"framework": framework, "operator": operator})
            snapshot(framework, run / "framework-sources.tar.gz")
            snapshot(operator, run / "operator-sources.tar.gz")
            if not cfg["allow_dirty"] and (framework["status"] or operator["status"]):
                raise RuntimeError("Sources are dirty; commit changes or explicitly set allow_dirty=true")
            if cfg["operator"]["ref"] is not None:
                expected = subprocess.check_output(["git", "-C", str(ROOT / "op-src"), "rev-parse", cfg["operator"]["ref"] + "^{commit}"], text=True).strip()
                if operator["sha"] != expected:
                    raise RuntimeError("operator.ref differs from the checked-out submodule; use switch_operator.sh first")
            key = hashlib.sha256(json.dumps({"sources": operator, "versions": capabilities["versions"],
                "cann": os.environ.get("ASCEND_HOME_PATH"), "image": os.environ.get("ASCEND_IMAGE_ID"),
                "soc": cfg["soc"]}, sort_keys=True).encode()).hexdigest()[:20]
            build = ROOT / ".cache/builds" / key
            # Serialize shared builds even if experiments use different NPUs.
            import fcntl
            with (ROOT / ".locks/build.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                if cfg["build"]["force"] or not (build / "build.json").is_file():
                    (build / "build.json").unlink(missing_ok=True)
                    environment = dict(os.environ, MAX_JOBS=str(cfg["build"]["jobs"]))
                    result = command(["bash", str(ROOT / "op-src/build.sh"), str(build)], run / "logs/build.log", environment, 1800)
                    record("build", result)
                    if result["returncode"]:
                        raise RuntimeError("Operator compilation/installation failed; see logs/build.log")
                else:
                    record("build", {"status": "cached", "directory": str(build)})
            build_info = json.loads((build / "build.json").read_text())
            packages_dir = run / "packages"
            packages_dir.mkdir()
            for package in build_info["packages"]:
                source = Path(package["path"])
                if hashlib.sha256(source.read_bytes()).hexdigest() != package["sha256"]:
                    raise RuntimeError("Cached package checksum differs; rebuild with build.force=true")
                shutil.copyfile(source, packages_dir / source.name)
                package["archived_as"] = str((packages_dir / source.name).relative_to(run))
            for binary in Path(build_info["opp"]).rglob("*.o"):
                destination = run / "kernel-binaries" / binary.relative_to(build_info["opp"])
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(binary, destination)
                metadata = binary.with_suffix(".json")
                if metadata.exists():
                    shutil.copyfile(metadata, destination.with_suffix(".json"))
            write_json(run / "build.json", build_info)
            from activation import environment
            env = environment(build_info)
            def workload(phase, output):
                return [sys.executable, str(ROOT / "scripts/python/workload.py"), "--config", str(run / "config.resolved.json"),
                        "--phase", phase, "--output", str(output)]
            for phase in ("correctness", "benchmark"):
                ensure_idle(cfg["device"], run / f"logs/npu-smi-before-{phase}.log")
                result = command(workload(phase, run / phase / "results.json"), run / f"logs/{phase}.log", env)
                record(phase, result)
                if result["returncode"]:
                    raise RuntimeError(f"{phase} failed; see logs/{phase}.log")
                if phase == "benchmark":
                    write_json(run / "benchmark/process.json", result)
            for backend in cfg["profiling"]["backends"]:
                folder = run / "profiling" / backend
                folder.mkdir()
                if not shutil.which("msprof"):
                    record(backend, {"status": "unsupported", "reason": "msprof is not installed"})
                    continue
                help_key = {"msprof": "msprof_help", "msprof-op": "msprof_op_help", "simulator": "simulator_help"}[backend]
                if capabilities[help_key].get("returncode", 1) != 0:
                    record(backend, {"status": "unsupported", "reason": "Installed profiler does not provide this mode"})
                    continue
                try:
                    ensure_idle(cfg["device"], run / f"logs/npu-smi-before-{backend}.log")
                except RuntimeError as exc:
                    record(backend, {"status": "blocked", "reason": str(exc)})
                    continue
                argv = ["msprof"]
                profile_env = env
                binary_simulation = backend == "simulator" and cfg["profiling"]["simulator_mode"] == "binary"
                if binary_simulation:
                    launch_json = folder / "launch.json"
                    prepare = workload("simulator-config", launch_json) + ["--build-info", str(run / "build.json")]
                    preparation = command(prepare, run / "logs/simulator-prepare.log", env)
                    if preparation["returncode"]:
                        record(backend, {"status": "unsupported" if preparation["returncode"] == 3 else "failed",
                                         "reason": "Operator binary simulator preparation failed", "preparation": preparation})
                        continue
                    simulator_lib = Path(os.environ["ASCEND_HOME_PATH"]) / "tools/simulator" / cfg["soc"] / "lib"
                    if not simulator_lib.is_dir():
                        record(backend, {"status": "unsupported", "reason": "Simulator libraries for the configured SoC are missing"})
                        continue
                    profile_env = dict(env, LD_LIBRARY_PATH=str(simulator_lib) + os.pathsep + env.get("LD_LIBRARY_PATH", ""))
                if backend != "msprof":
                    argv += ["op"]
                    if backend == "simulator":
                        argv += ["simulator"]
                        if not binary_simulation:
                            argv += ["--soc-version=" + cfg["soc"]]
                    if binary_simulation:
                        argv += ["--config=" + str(launch_json)]
                    else:
                        argv += ["--kernel-name=" + cfg["profiling"]["kernel"], "--launch-count=" + str(cfg["profiling"]["launch_count"])]
                argv += ["--output=" + str(folder / "raw")]
                metrics = cfg["profiling"]["metrics"][backend]
                if metrics is not None:
                    argv += ["--aic-metrics=" + metrics]
                if not binary_simulation:
                    argv += workload(backend, folder / "workload.json")
                try:
                    result = command(argv, run / f"logs/{backend}.log", profile_env, cfg["profiling"]["timeout_s"], cwd=folder)
                    if backend == "simulator":
                        result["entry_point"] = cfg["profiling"]["simulator_mode"]
                    parsed = profile_reports(folder / "raw", cfg["profiling"]["kernel"])
                    write_json(folder / "summary.json", parsed)
                    log_text = (run / f"logs/{backend}.log").read_text(errors="replace")
                    child_failed = re.search(r"Child process exited with status\s+(?!0\b)\d+", log_text)
                    result["status"] = "ok" if result["returncode"] == 0 and parsed["matched_reports"] and not child_failed else "failed"
                    if result["status"] == "failed":
                        result["reason"] = child_failed.group() if child_failed else "Profiler failed or produced no numeric CSV rows for the custom kernel"
                    record(backend, result)
                except subprocess.TimeoutExpired:
                    record(backend, {"status": "failed", "reason": "Profiler timed out; process group terminated"})
                except Exception as exc:
                    record(backend, {"status": "failed", "reason": str(exc)})
            failed = [b for b in cfg["profiling"]["backends"] if status["stages"][b]["status"] != "ok"]
            status["status"] = ("failed" if cfg["profiling"]["required"] else "partial") if failed else "ok"
    except (Exception, KeyboardInterrupt) as exc:
        status["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        status["error"] = str(exc) or type(exc).__name__
    finally:
        status["finished_at"] = datetime.now(resolve_timezone(cfg["timezone"])).isoformat()
        write_json(state_file, status)
    print(json.dumps({"run": str(run), **status}, indent=2))
    return 0 if status["status"] in ("ok", "partial") else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--device-only", action="store_true", help="Validate config and print physical device for the host wrapper")
    args = parser.parse_args()
    try:
        cfg = configuration(args.config)
    except (ValueError, OSError, TypeError) as exc:
        # Even an invalid config gets its own diagnostic artifact, before any NPU work.
        try:
            diagnostic_timezone = resolve_timezone(DEFAULT["timezone"])
        except ZoneInfoNotFoundError:
            diagnostic_timezone = timezone.utc
        timestamp = datetime.now(diagnostic_timezone).strftime("%m%d-%H%M")
        failed_run = ROOT / "runs" / f"{timestamp}-exp"
        failed_run.parent.mkdir(exist_ok=True)
        failed_run.mkdir()
        if args.config.is_file():
            shutil.copyfile(args.config, failed_run / "config.original.json")
        write_json(failed_run / "status.json", {"status": "failed", "stage": "configuration", "error": str(exc),
                                              "timezone": str(diagnostic_timezone)})
        print(f"Invalid configuration; details saved to {failed_run}", file=sys.stderr)
        return 2
    if args.device_only:
        print(cfg["device"])
        return 0
    timestamp = datetime.now(resolve_timezone(cfg["timezone"])).strftime("%m%d-%H%M")
    run = ROOT / "runs" / f"{timestamp}-{cfg['name']}"
    run.parent.mkdir(exist_ok=True)
    run.mkdir()  # Atomic; never overwrite an experiment with the same name.
    for subdir in ("logs", "correctness", "benchmark", "profiling"):
        (run / subdir).mkdir()
    shutil.copyfile(args.config, run / "config.original.json")
    write_json(run / "config.resolved.json", cfg)
    write_json(run / "manifest.json", {"created_at": datetime.now(resolve_timezone(cfg["timezone"])).isoformat(),
        "physical_npu": cfg["device"], "logical_npu": 0, "soc": cfg["soc"],
        "image_id": os.environ.get("ASCEND_IMAGE_ID"), "image_reference": os.environ.get("ASCEND_IMAGE_REFERENCE")})
    def interrupt(signum, frame):
        raise KeyboardInterrupt("Received signal " + str(signum))
    signal.signal(signal.SIGTERM, interrupt)
    return execute(cfg, run)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)
