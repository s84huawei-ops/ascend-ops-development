"""Run an operator adapter in a fresh process for each experiment stage."""
import argparse
import importlib
import json
from pathlib import Path
import statistics
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", choices=["correctness", "benchmark", "msprof", "msprof-op", "simulator", "simulator-config"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--build-info")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    import torch
    import torch_npu
    adapter = importlib.import_module(cfg["operator"]["module"])
    if args.phase == "simulator-config":
        hook = getattr(adapter, "prepare_simulator", None)
        if hook is None:
            raise SystemExit(3)
        build_info = json.loads(Path(args.build_info).read_text())
        launch = hook(cfg, Path(args.output).parent, build_info)
        Path(args.output).write_text(json.dumps(launch, indent=2) + "\n")
        return
    torch.npu.set_device(0)  # Physical selection belongs to the container, not the adapter.
    torch.npu.config.allow_internal_format = False
    operation = getattr(adapter, cfg["operator"]["callable"])
    reference = getattr(adapter, "reference")
    data = cfg["data"]
    generator = torch.Generator().manual_seed(data["seed"])

    def inputs(shape):
        x = torch.randn(shape, generator=generator, dtype=torch.float16)
        y = torch.randn(shape, generator=generator, dtype=torch.float16)
        return x, y, x.to("npu:0"), y.to("npu:0")

    if args.phase == "correctness":
        results = []
        for shape in [data["shape"]] + data["extra_test_shapes"]:
            a, b, x, y = inputs(shape)
            actual = operation(x, y)
            torch.npu.synchronize()
            assert actual.device == x.device, "Output is not on the selected NPU"
            cpu = actual.cpu()
            expected = reference(a, b)
            torch.testing.assert_close(cpu, expected, rtol=data["rtol"], atol=data["atol"])
            results.append({"shape": shape, "dtype": str(cpu.dtype), "device": str(actual.device),
                            "max_abs_error": float((cpu.float() - expected.float()).abs().max())})
        result = {"status": "ok", "cases": results}
    else:
        shape = cfg["profiling"]["simulator_shape"] if args.phase == "simulator" else data["shape"]
        _, _, x, y = inputs(shape)
        warmup = cfg["benchmark"]["warmup"] if args.phase == "benchmark" else (0 if args.phase == "simulator" else 1)
        iterations = cfg["benchmark"]["iterations"] if args.phase == "benchmark" else cfg["profiling"]["launch_count"]
        for _ in range(warmup):
            output = operation(x, y)
        torch.npu.synchronize()
        events = [(torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True))
                  for _ in range(iterations)] if args.phase == "benchmark" else None
        started = time.perf_counter()
        for index in range(iterations):
            if events:
                events[index][0].record()
            output = operation(x, y)
            if events:
                events[index][1].record()
        torch.npu.synchronize()
        measured = time.perf_counter() - started
        result = {"status": "ok", "phase": args.phase, "shape": shape, "warmup": warmup,
                  "iterations": iterations, "benchmark_wall_s": measured}
        if events:
            samples = [start.elapsed_time(end) for start, end in events]
            ordered = sorted(samples)
            result.update({"device_region_ms": samples, "device_region_median_ms": statistics.median(samples),
                           "device_region_mean_ms": statistics.mean(samples),
                           "device_region_p95_ms": ordered[min(len(ordered)-1, int(len(ordered)*0.95))],
                           "timing_scope": "Stream regions around calls, including gaps; not kernel-only latency"})
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
