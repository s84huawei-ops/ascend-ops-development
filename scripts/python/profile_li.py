#!/usr/bin/env python3
"""Standalone Lightning Indexer benchmark, on-board and simulator profiler.

Requires a working torch + torch_npu environment inside the container.
For -impl ops/ops-quant, vllm_ascend and its compiled custom ops are required.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import traceback
from datetime import datetime
from typing import Any

PROFILE_MARKER = "LI_PROFILE"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark or profile Lightning Indexer on Ascend NPU")
    parser.add_argument("--mode", choices=("simple", "detailed", "simulator"), default="simple")
    parser.add_argument("--soc-version", help="Target SoC for msprof op simulator, e.g. Ascend910_93")
    parser.add_argument("--simulator-lib", type=Path, help="Directory containing simulator runtime libraries (prepended to LD_LIBRARY_PATH)")
    parser.add_argument("-impl", "--impl", choices=("ops", "npu", "ops-quant"), default="ops")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--q_len", "--q-len", type=int, default=16)
    parser.add_argument("--kv_len", "--kv-len", type=int, default=32768)
    parser.add_argument("--num_heads", "--num-heads", type=int, default=64)
    parser.add_argument("--head_dim", "--head-dim", type=int, default=128)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--block_size", "--block-size", type=int, default=128)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validate", action="store_true", help="Validate output shape and index bounds outside timing")
    parser.add_argument("--aic-metrics", default="Default", help="msprof op --aic-metrics value (detailed only)")
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    # Internal worker interface. Parent creates the run directory and invokes workers.
    parser.add_argument("--_worker", choices=("benchmark", "profile"), help=argparse.SUPPRESS)
    parser.add_argument("--_run-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    validate_config(args, parser)
    return args


def validate_config(a: argparse.Namespace, parser: argparse.ArgumentParser | None = None) -> None:
    def invalid(message: str) -> None:
        if parser is not None:
            parser.error(message)
        raise ValueError(message)

    for name in ("iterations", "repeats", "batch", "q_len", "kv_len", "num_heads", "head_dim", "topk", "block_size"):
        if getattr(a, name) < 1:
            invalid(f"--{name} must be positive")
    if a.warmup < 0 or a.device < 0 or a.seed < 0:
        invalid("--warmup, --device and --seed cannot be negative")
    if a.q_len > a.kv_len:
        invalid("--q_len cannot exceed --kv_len in the chosen causal suffix-of-KV scenario")
    if a.topk > a.kv_len:
        invalid("--topk cannot exceed --kv_len")
    if a.topk > 2048:
        invalid("--topk above 2048 is not supported by the ops-quant implementation")
    if a.num_heads > 64:
        invalid("--num_heads above 64 is not supported by the Lightning Indexer API")
    if a.head_dim != 128:
        invalid("--head_dim must be 128 for the currently supported Lightning Indexer API")
    if a.block_size % 16:
        invalid("--block_size must be a multiple of 16 for PA_BSND")
    if a.block_size > 1024:
        invalid("--block_size cannot exceed 1024 for PA_BSND")
    if not a.aic_metrics.strip():
        invalid("--aic-metrics must not be empty")
    if a.mode == "simulator" and not a.soc_version:
        invalid("--soc-version is required when --mode simulator")
    if a.mode == "simulator" and a.device != 0:
        invalid("simulator supports only --device 0")
    if a.simulator_lib is not None and not a.simulator_lib.is_dir():
        invalid(f"--simulator-lib directory does not exist: {a.simulator_lib}")
    if a._worker and not a._run_dir:
        invalid("internal --_worker needs --_run-dir")


def config_dict(a: argparse.Namespace) -> dict[str, Any]:
    return {
        "mode": a.mode, "impl": a.impl, "warmup": a.warmup,
        "soc_version": a.soc_version,
        "simulator_lib": str(a.simulator_lib) if a.simulator_lib else None,
        "iterations": a.iterations, "repeats": a.repeats,
        "batch": a.batch, "q_len": a.q_len, "kv_len": a.kv_len,
        "num_heads": a.num_heads, "head_dim": a.head_dim,
        "topk": a.topk, "block_size": a.block_size,
        "dtype": a.dtype, "device": a.device, "seed": a.seed,
        "validate": a.validate, "aic_metrics": a.aic_metrics,
        "layout_query": "TND", "layout_key": "PA_BSND", "sparse_mode": 3,
        "quant_dtype": "int8" if a.impl == "ops-quant" else None,
        "quant_scale_dtype": "fp16" if a.impl == "ops-quant" else None,
        "data_model": "synthetic causal prefill, query at the end of KV context",
        "profile_warmup": 0 if a.mode == "simulator" else a.warmup,
    }


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
        f.write("\n")
    temp.replace(path)


def new_run_directory(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    base = f"profile-{datetime.now().strftime('%m%d-%H%M')}"
    for suffix in range(1, 10000):
        candidate = root / (base if suffix == 1 else f"{base}-{suffix}")
        try:
            candidate.mkdir()
            (candidate / "logs").mkdir()
            return candidate.resolve()
        except FileExistsError:
            continue
    raise RuntimeError("Cannot create unique profile directory")


def worker_cli(a: argparse.Namespace, run_dir: Path, worker: str) -> list[str]:
    options = [
        "--mode", a.mode, "--impl", a.impl,
        "--warmup", str(a.warmup), "--iterations", str(a.iterations),
        "--repeats", str(a.repeats), "--batch", str(a.batch),
        "--q_len", str(a.q_len), "--kv_len", str(a.kv_len),
        "--num_heads", str(a.num_heads), "--head_dim", str(a.head_dim),
        "--topk", str(a.topk), "--block_size", str(a.block_size),
        "--dtype", a.dtype, "--device", str(a.device), "--seed", str(a.seed),
        "--aic-metrics", a.aic_metrics, "--_worker", worker,
        "--_run-dir", str(run_dir),
    ]
    if a.validate:
        options.append("--validate")
    if a.soc_version:
        options.extend(["--soc-version", a.soc_version])
    if a.simulator_lib:
        options.extend(["--simulator-lib", str(a.simulator_lib)])
    return [sys.executable, str(Path(__file__).resolve()), *options]


def run_logged(cmd: list[str], log_path: Path, env: dict[str, str] | None = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write("Command: " + subprocess.list2cmdline(cmd) + "\n\n")
        log.flush()
        try:
            completed = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False, env=env)
            log.write(f"\nReturn code: {completed.returncode}\n")
            return completed.returncode
        except Exception:
            traceback.print_exc(file=log)
            return 127


def summary_text(run_dir: Path, result: dict[str, Any]) -> str:
    cfg = result["config"]
    lines = [
        "Lightning Indexer — Ascend NPU",
        f"Run: {run_dir}",
        f"Status: {result['status']}",
        f"Mode / implementation: {cfg['mode']} / {cfg['impl']}",
        f"Shape: B={cfg['batch']}, Q={cfg['q_len']}, KV={cfg['kv_len']}, H={cfg['num_heads']}, D={cfg['head_dim']}, topk={cfg['topk']}",
        f"Layout: {cfg['layout_query']} + {cfg['layout_key']}; block_size={cfg['block_size']}",
        f"Timing: warmup={cfg['warmup']}, iterations={cfg['iterations']}, repeats={cfg['repeats']}",
    ]
    bench = result.get("benchmark")
    if bench:
        stats = bench["stats_ms"]
        lines += [
            "", "NPU event benchmark (milliseconds):",
            f"  mean={stats['mean']:.6f}, median={stats['median']:.6f}, min={stats['min']:.6f}, max={stats['max']:.6f}, std={stats['stdev']:.6f}",
            f"  samples: {bench['samples_ms']}",
        ]
    if cfg["mode"] == "simulator":
        lines += ["", "Simulator: no device latency benchmark (CPU simulation only)."]
    if result.get("profiling"):
        lines += ["", "msprof:"]
        for key, val in result["profiling"].items():
            lines.append(f"  {key}: {val}")
    if result.get("error"):
        lines += ["", f"Error: {result['error']}"]
    lines += ["", "Logs: logs/benchmark.log (simple/detailed), logs/msprof.log (detailed), or logs/simulator.log (simulator)"]
    return "\n".join(lines) + "\n"


def save_result(run_dir: Path, result: dict[str, Any]) -> None:
    write_json(run_dir / "results.json", result)
    (run_dir / "summary.txt").write_text(summary_text(run_dir, result), encoding="utf-8")


def simulator_env(a: argparse.Namespace) -> tuple[dict[str, str], str | None]:
    """Best-effort simulator library discovery; does not require an NPU."""
    env = os.environ.copy()
    assert a.soc_version is not None
    candidates: list[Path] = []
    if a.simulator_lib:
        candidates.append(a.simulator_lib)
    if env.get("ASCEND_HOME_PATH"):
        candidates.append(Path(env["ASCEND_HOME_PATH"]) / "tools" / "simulator" / a.soc_version / "lib")
    for base in (
        "/usr/local/Ascend/ascend-toolkit/latest",
        "/usr/local/Ascend/ascend_toolkit/latest",
        "/usr/local/Ascend/cann",
        "/usr/local/Ascend/cann-9.0.1",
    ):
        candidates.append(Path(base) / "tools" / "simulator" / a.soc_version / "lib")
    lib = next((candidate.resolve() for candidate in candidates if candidate.is_dir()), None)
    if lib:
        env["LD_LIBRARY_PATH"] = str(lib) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    return env, str(lib) if lib else None


def driver(a: argparse.Namespace) -> int:
    run_dir = new_run_directory(a.output_root)
    cfg = config_dict(a)
    cfg["created_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    cfg["python"] = sys.version.split()[0]
    cfg["script"] = str(Path(__file__).resolve())
    write_json(run_dir / "config.json", cfg)
    result: dict[str, Any] = {"status": "running", "config": config_dict(a)}
    save_result(run_dir, result)
    print(f"Run directory: {run_dir}", flush=True)

    try:
        if a.mode != "simulator":
            benchmark_cmd = worker_cli(a, run_dir, "benchmark")
            code = run_logged(benchmark_cmd, run_dir / "logs" / "benchmark.log")
            if code != 0:
                raise RuntimeError(f"Benchmark worker failed (exit {code}); see logs/benchmark.log")
            benchmark_json = run_dir / "benchmark.json"
            if not benchmark_json.is_file():
                raise RuntimeError("Benchmark worker did not write benchmark.json; see logs/benchmark.log")
            payload = json.loads(benchmark_json.read_text(encoding="utf-8"))
            result["benchmark"] = payload["benchmark"]
            result["environment"] = payload["environment"]
            if "validation" in payload:
                result["validation"] = payload["validation"]
            save_result(run_dir, result)
            print(f"Mean NPU latency: {result['benchmark']['stats_ms']['mean']:.6f} ms", flush=True)

        if a.mode in ("detailed", "simulator"):
            msprof = shutil.which("msprof")
            if not msprof:
                raise RuntimeError("msprof executable not found on PATH; see logs and verify CANN environment")
            profile_root = run_dir / "msprof"
            profile_root.mkdir()
            is_simulator = a.mode == "simulator"
            log_path = run_dir / "logs" / ("simulator.log" if is_simulator else "msprof.log")
            env, detected_lib = simulator_env(a) if is_simulator else (None, None)
            msprof_cmd = [msprof, "op"]
            if is_simulator:
                msprof_cmd += ["simulator", f"--soc-version={a.soc_version}"]
            msprof_cmd += [
                "--kernel-name=LightningIndexer",
                "--launch-count=1",
                f"--output={profile_root}",
            ]
            # Simulator has a different metrics vocabulary; do not pass the
            # hardware-only Default set unless the user explicitly overrides it.
            if not is_simulator or a.aic_metrics != "Default":
                msprof_cmd.append(f"--aic-metrics={a.aic_metrics}")
            msprof_cmd += worker_cli(a, run_dir, "profile")
            if is_simulator:
                print(f"Simulator libraries: {detected_lib or '(not autodetected; check CANN setup)'}", flush=True)
            code = run_logged(msprof_cmd, log_path, env=env)
            bins = sorted(
                p.relative_to(run_dir).as_posix()
                for p in profile_root.rglob("visualize_data.bin")
                if p.is_file() and p.stat().st_size > 0
            )
            traces = sorted(
                p.relative_to(run_dir).as_posix()
                for p in profile_root.rglob("trace.json") if p.is_file()
            )
            result["profiling"] = {
                "method": "msprof op simulator" if is_simulator else "msprof op",
                "msprof_return_code": code,
                "soc_version": a.soc_version if is_simulator else None,
                "simulator_lib": detected_lib,
                "metric_set": (a.aic_metrics if a.aic_metrics != "Default" else "simulator default") if is_simulator else a.aic_metrics,
                "visualize_data_files": bins,
                "trace_json_files": traces,
                "data_directory": "msprof",
                "log_file": log_path.relative_to(run_dir).as_posix(),
            }
            save_result(run_dir, result)
            if code != 0:
                raise RuntimeError(f"{'msprof op simulator' if is_simulator else 'msprof op'} failed (exit {code}); see {log_path.relative_to(run_dir)}")
            if not bins:
                raise RuntimeError(f"msprof finished but nonempty visualize_data.bin is missing; see {log_path.relative_to(run_dir)}")
            print("MindStudio visualize_data.bin:", *bins, sep="\n  ", flush=True)

        result["status"] = "success"
        save_result(run_dir, result)
        return 0
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = str(exc)
        save_result(run_dir, result)
        (run_dir / "logs" / "driver_error.log").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
        print(f"FAILED: {exc}\nLogs and partial results: {run_dir}", file=sys.stderr, flush=True)
        return 1


def load_operator(torch: Any, torch_npu: Any, impl: str) -> Any:
    if impl == "npu":
        fn = getattr(torch_npu, "npu_lightning_indexer", None)
        if fn is None:
            raise RuntimeError("torch_npu.npu_lightning_indexer is unavailable in this torch_npu build")
        return fn

    from vllm_ascend.utils import enable_custom_op
    if not enable_custom_op():
        raise RuntimeError(
            "vllm_ascend custom operators are disabled/unavailable. "
            "Check compiled extension, CANN paths, and vllm_ascend installation."
        )
    name = "npu_lightning_indexer" if impl == "ops" else "npu_lightning_indexer_quant"
    try:
        return getattr(torch.ops._C_ascend, name)
    except AttributeError as exc:
        raise RuntimeError(
            f"torch.ops._C_ascend.{name} not registered; this vllm-ascend version "
            "may have removed/renamed the operator; check logs and installed schemas"
        ) from exc


def prepare_inputs(torch: Any, a: argparse.Namespace) -> tuple[dict[str, Any], dict[str, list[int]]]:
    dev = torch.device(f"npu:{a.device}")
    t = a.batch * a.q_len
    pages_per_request = math.ceil(a.kv_len / a.block_size)
    num_blocks = a.batch * pages_per_request
    key_shape = (num_blocks, a.block_size, 1, a.head_dim)
    query_shape = (t, a.num_heads, a.head_dim)
    weights_shape = (t, a.num_heads)
    if a.impl == "ops-quant":
        q = torch.randint(-64, 64, query_shape, device=dev, dtype=torch.int8)
        k = torch.randint(-64, 64, key_shape, device=dev, dtype=torch.int8)
        w = torch.randn(weights_shape, device=dev, dtype=torch.float16)
        # Per-query-head and per-key-token scales, as expected by this custom op.
        qs = torch.full((t, a.num_heads), 1.0 / 64.0, device=dev, dtype=torch.float16)
        ks = torch.full((num_blocks, a.block_size, 1), 1.0 / 64.0,
                        device=dev, dtype=torch.float16)
    else:
        dtype = torch.bfloat16 if a.dtype == "bf16" else torch.float16
        q = torch.randn(query_shape, device=dev, dtype=dtype)
        k = torch.randn(key_shape, device=dev, dtype=dtype)
        w = torch.randn(weights_shape, device=dev, dtype=dtype)
        qs = ks = None

    # All pages are valid, disjoint, and allocated. No aliasing across requests.
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=dev).reshape(a.batch, pages_per_request)
    # TND query lengths are cumulative; paged key lengths are per request.
    q_lens = torch.arange(a.q_len, t + 1, a.q_len, dtype=torch.int32, device=dev)
    k_lens = torch.full((a.batch,), a.kv_len, dtype=torch.int32, device=dev)
    data = dict(query=q, key=k, weights=w, actual_seq_lengths_query=q_lens,
                actual_seq_lengths_key=k_lens, block_table=block_table,
                layout_query="TND", layout_key="PA_BSND",
                sparse_count=a.topk, sparse_mode=3)
    if a.impl == "ops-quant":
        data.update(query_dequant_scale=qs, key_dequant_scale=ks,
                    query_quant_mode=0, key_quant_mode=0)
    return data, {"query": list(query_shape), "key": list(key_shape),
                  "weights": list(weights_shape), "block_table": [a.batch, pages_per_request]}


def validate_output(torch: Any, output: Any, a: argparse.Namespace) -> dict[str, Any]:
    indices = output[0] if isinstance(output, (tuple, list)) else output
    if not isinstance(indices, torch.Tensor):
        raise TypeError(f"Expected index Tensor, got {type(indices)}")
    shape = list(indices.shape)
    expected = [a.batch * a.q_len, 1, a.topk]
    if shape != expected:
        raise ValueError(f"Unexpected output index shape {shape}, expected {expected} (TND)")
    if indices.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"Unexpected indices dtype {indices.dtype}; expected int32 or int64")
    # Index tensors can contain -1 as a sentinel for padded/invalid entries.
    indices_cpu = indices.detach().cpu().reshape(a.batch, a.q_len, a.topk)
    valid = (indices_cpu >= -1) & (indices_cpu < a.kv_len)
    if not bool(valid.all().item()):
        raise ValueError("Output contains an index outside [-1, kv_len - 1]")
    # Causal suffix constraint: query token j can see at most KV prefix
    # kv_len - q_len + j + 1; repeated for all batch elements.
    for j in range(a.q_len):
        max_allowed = a.kv_len - a.q_len + j
        row = indices_cpu[:, j, :]
        if not bool((row <= max_allowed).all().item()):
            raise ValueError(f"Causal mask violation at query position {j}")
    return {"status": "passed", "indices_shape": shape, "indices_dtype": str(indices.dtype)}


def environment_info(torch: Any, torch_npu: Any, a: argparse.Namespace) -> dict[str, Any]:
    data: dict[str, Any] = {
        "python": sys.version.split()[0],
        "torch": str(getattr(torch, "__version__", "unknown")),
        "torch_npu": str(getattr(torch_npu, "__version__", "unknown")),
        "torch_cann": str(getattr(getattr(torch, "version", None), "cann", "unknown")),
        "ascend_home_path": os.getenv("ASCEND_HOME_PATH"),
        "ascend_rt_visible_devices": os.getenv("ASCEND_RT_VISIBLE_DEVICES"),
    }
    try:
        data["vllm_ascend"] = importlib.metadata.version("vllm-ascend")
    except importlib.metadata.PackageNotFoundError:
        data["vllm_ascend"] = None
    try:
        data["npu_name"] = str(torch.npu.get_device_name(a.device))
    except Exception as exc:
        data["npu_name_error"] = str(exc)
    try:
        data["npu_count"] = int(torch.npu.device_count())
    except Exception:
        pass
    return data


def worker(a: argparse.Namespace) -> int:
    assert a._run_dir is not None
    # torch imports are deliberately restricted to the short-lived workers.
    import torch
    import torch_npu

    print(f"Worker={a._worker}, impl={a.impl}, device={a.device}", flush=True)
    # In simulator mode the CPU-backed simulator runtime can stand in for the
    # actual device, so a physical NPU availability probe is inappropriate.
    # set_device / allocations may still fail if simulator libraries are not
    # installed or the operator is not compatible with simulation.
    if a.mode != "simulator" and not torch.npu.is_available():
        raise RuntimeError("torch.npu.is_available() is false: no NPU accessible in this container")
    torch.npu.set_device(a.device)
    torch.manual_seed(a.seed)
    if hasattr(torch.npu, "manual_seed"):
        torch.npu.manual_seed(a.seed)
    op = load_operator(torch, torch_npu, a.impl)
    data, shapes = prepare_inputs(torch, a)
    fn = lambda: op(**data)
    torch.npu.synchronize()

    if a._worker == "profile":
        try:
            import mstx
        except ImportError as exc:
            raise RuntimeError(
                "Cannot import 'mstx'; required for targeting the Lightning Indexer "
                "with msprof op --mstx=on. Check CANN profiling environment."
            ) from exc
        profile_warmup = 0 if a.mode == "simulator" or a.mode == "detailed" else a.warmup
        if a.mode == "simulator":
            print("Simulator: warmup disabled; profile a single operator invocation", flush=True)
        for _ in range(profile_warmup):
            fn()
        torch.npu.synchronize()
        print(f"Starting MSTX range {PROFILE_MARKER}", flush=True)
        output = fn()
        torch.npu.synchronize()
        if a.validate:
            print("Validation: " + json.dumps(validate_output(torch, output, a)), flush=True)
        print(f"Finished MSTX range {PROFILE_MARKER}", flush=True)
        return 0

    samples: list[list[float]] = []
    last_output = None
    for repeat in range(a.repeats):
        for _ in range(a.warmup):
            fn()
        torch.npu.synchronize()
        one_repeat: list[float] = []
        for _ in range(a.iterations):
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            start.record()
            last_output = fn()
            end.record()
            end.synchronize()
            value = float(start.elapsed_time(end))
            if not math.isfinite(value) or value < 0:
                raise RuntimeError(f"Invalid event timing: {value} ms")
            one_repeat.append(value)
        samples.append(one_repeat)
        print(f"repeat {repeat + 1}: {one_repeat} ms", flush=True)

    flat = [value for row in samples for value in row]
    stats = {
        "mean": statistics.mean(flat), "median": statistics.median(flat),
        "min": min(flat), "max": max(flat),
        "stdev": statistics.stdev(flat) if len(flat) > 1 else 0.0,
    }
    payload: dict[str, Any] = {
        "benchmark": {"time_unit": "ms", "method": "torch.npu.Event",
                      "samples_ms": samples, "stats_ms": stats,
                      "input_shapes": shapes},
        "environment": environment_info(torch, torch_npu, a),
    }
    if a.validate:
        # Run validation outside the measured section.
        payload["validation"] = validate_output(torch, last_output, a)
    write_json(a._run_dir / "benchmark.json", payload)
    print("Mean: %.6f ms" % stats["mean"], flush=True)
    return 0


def main() -> int:
    args = parse_args()
    if args._worker:
        try:
            return worker(args)
        except Exception:
            traceback.print_exc()
            return 1
    return driver(args)


if __name__ == "__main__":
    sys.exit(main())
