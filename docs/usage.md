# Using the Ascend operator framework

## Prerequisites and repositories

The server workspace is `/home/s84435910/docker_workspace/ascend-ops-development`,
accessible through `ssh huawei-blue-npu`. The host needs Docker, Python 3.9+,
Git, the installed Ascend driver, and the required image already present.
The framework uses a single Ascend910B2. Device 6 is the default physical NPU.

The `op-src` submodule points to the private `s84huawei-ops/ascend-op-demo`
repository. Use an account with access when cloning or updating it:

```bash
git clone --recurse-submodules https://github.com/s84huawei-ops/ascend-ops-development.git
# For an existing clone:
git submodule update --init --recursive
```

The demo implements FP16 addition through a custom Ascend C kernel, generated
ACLNN entry points, and a standalone PyTorch C++ adapter. It supports nonempty
contiguous inputs of equal shape with zero storage offsets. It has no broadcasting,
autograd, or torch.compile support. It is intended to validate the development
workflow, not to compete with the optimized built-in addition operator.

## Running an experiment

On the server, from the framework repository:

```bash
bash scripts/bash/run.sh configs/demo.json
bash scripts/bash/run.sh configs/msprof.json
bash scripts/bash/run.sh configs/simulator.json
bash scripts/bash/run.sh configs/all.json
```

The wrapper validates configuration, selects the physical NPU, creates or reuses
the dedicated container, and starts the runner inside it. The runner reserves
the NPU with a filesystem lock, checks `npu-smi`, captures sources/environment,
builds and registers the operator, checks correctness, runs the benchmark, and
then runs each requested profiler in a separate process.

`all.json` requests `msprof`, detailed on-device `msprof op`, and
`msprof op simulator`. A profiler is successful only if it exits successfully
and produces numeric timing/cycle CSV rows identifying the custom operator.
The tool existing on PATH is not sufficient. There is no silent fallback from
simulation to physical execution.

Simulation has two explicit entry modes. The default `simulator_mode=binary`
profiles the same installed `.o` using an operator-provided launch configuration,
CPU-generated inputs, and the exact tiling layout. This measures the kernel
outside PyTorch; correctness and benchmarks through torch_npu are separate stages.
`configs/simulator-python.json` selects the Python path instead. With the tested
CANN 9.0.1 / torch_npu 2.10.0.post2 stack, that path produces traces but the child
process crashes during teardown (status 139), so it is reported as failed.
No exit-status override is used to present it as successful.

The container exposes only the selected `/dev/davinciN` plus management devices.
The driver maps that single device to logical `npu:0`; physical selection is
recorded separately in `ASCEND_PHYSICAL_DEVICE_ID`. Do not set
`ASCEND_RT_VISIBLE_DEVICES` to the physical ID: it would refer to an already
remapped logical device. `env.sh` clears this mask. The host timezone database
is mounted read-only so Python can use `Europe/Warsaw` even though the image
does not include its own timezone database.

The container name defaults to `ascend-ops-dev-npu6`. `ASCEND_CONTAINER` can
select another name. Image, workspace, device, and container settings are checked
before reuse. Outdated containers are not silently replaced. Recreate only the
framework's own container or choose a new name. Builds and results remain on
the host workspace when a container is removed.

The filesystem lock coordinates this framework's runs in one workspace. It
does not reserve hardware against unrelated users; `npu-smi` is checked as well.
A busy device or a held lock causes failure with a retained experiment status.
Select and commit sources before starting an experiment, and leave them unchanged
until it finishes. The run snapshots record the source state at the start.

## Configuration

Configurations are JSON, with strict key validation. Omitted fields use the
defaults in `scripts/python/experiment.py`. A complete example:

```json
{
  "name": "add-comparison",
  "timezone": "Europe/Warsaw",
  "device": 6,
  "soc": "Ascend910B2",
  "operator": {"module": "ascend_op_demo", "callable": "add", "ref": null},
  "data": {
    "shape": [8, 2048], "dtype": "float16", "seed": 17,
    "rtol": 0.001, "atol": 0.00001,
    "extra_test_shapes": [[1], [17], [16387]]
  },
  "benchmark": {"warmup": 5, "iterations": 50},
  "build": {"force": false, "jobs": 2},
  "profiling": {
    "backends": ["msprof", "msprof-op", "simulator"],
    "required": true, "kernel": "AddCustom", "launch_count": 1,
    "timeout_s": 300, "simulator_shape": [16], "simulator_mode": "binary"
  },
  "allow_dirty": false
}
```

`name` defaults to `exp` and must be a safe path segment. Experiment names use
`MMDD-HHMM-name` in the selected timezone. Creating an existing directory fails
without overwriting it; choose a different name or wait until the next minute.

`operator.ref` is an optional expected tag/branch/SHA, not an instruction to
change sources during a run. It must resolve to the checked-out operator SHA.
`allow_dirty=false` requires clean framework and operator repositories. An
explicit development run can set it to true; source hashes, diffs, and snapshots
are still saved. Invalid configuration also gets a diagnostic run directory
using the default name `exp`.

The simulator uses its own small input shape because simulation can be much
slower than device execution. Its report is not a measurement of the benchmark
shape unless those shapes are explicitly made identical. `required=false`
allows a `partial` result when a requested profiler fails or is unsupported;
the status of each backend remains visible. With `required=true`, the experiment
fails if any requested backend fails, times out, or is unsupported.

`kernel` is a case-sensitive profiler prefix: the generated kernel uses
`AddCustom_<hash>_0`, so the default is `AddCustom`, not the C++ source function's
`add_custom`. In binary mode the operator hook supplies the exact symbol from
the compiled metadata, plus the binary, input files, workspace, and tiling.

Optional `profiling.metrics` selects vendor metric groups independently, for
example `{"msprof": "PipeUtilization", "msprof-op": "Default", "simulator": "PipeUtilization"}`.
Omit a key or use null to retain that tool's defaults. Allowed names and
combinations depend on the installed version; its complete help is retained in
`capabilities.json`. Invalid combinations remain failed runs with their logs.

## Versions, builds, and custom programs

Commit and push operator changes in its own repository, then record the new
submodule pointer in the framework:

```bash
git -C op-src add .
git -C op-src commit -m "Improve the operator"
git -C op-src push origin HEAD
git add op-src
git commit -m "Pin the new operator version"
git push
```

Switch to an existing operator tag, branch, or SHA:

```bash
bash scripts/bash/switch_operator.sh v0.1.0 --fetch
git commit -m "Pin operator v0.1.0"
```

Switching refuses to overwrite operator changes and stages the new submodule
pointer. The optional `--fetch` updates refs/tags first; omit it for local refs.

Build without running a benchmark, or execute your own program with the built
operator registered through its adapter:

```bash
bash scripts/bash/build_operator.sh /workspace/.cache/manual-build
bash scripts/bash/operator_exec.sh /workspace/.cache/manual-build python3 my_program.py
```

In the program, import `torch`, `torch_npu`, and `ascend_op_demo`, then call
`torch.ops.ascend_demo.add(x, y)` or `ascend_op_demo.add(x, y)` on NPU tensors.
Activation applies only to that process; it does not modify the image's global
torch installation or system operator registry.

The experiment build cache lives in `.cache/builds/`. Keys include operator
revision, actual source hashes/diffs, installed package versions, SoC, image ID,
and toolkit location. Source changes select a new build. `build.force=true`
rebuilds the selected cache entry under a shared build lock. Keep the build
prefix in place while using its adapter; artifacts contain build-specific paths.

## Synchronizing with the server

The normal Git workflow is commit/push locally, then pull the framework and
initialize/update its submodule on the server. The server then needs read access
to the private operator repository.

Alternatively, transfer the same committed versions through SSH without storing
GitHub credentials on the server. From the local repository, including PowerShell:

```powershell
python scripts/python/sync_server.py
```

This streams two Git bundles over one SSH connection, fast-forwards the server
framework, and checks out the exact pinned operator commit. Both local repositories
and the server clone must be clean; it refuses to discard remote changes or
overwrite divergent history. The server clone must already have an initial commit.
This synchronization transfers sources, not experiment results or build caches.

Download an experiment using Git Bash or a Linux shell:

```bash
bash scripts/bash/fetch_run.sh 1022-1430-add-fp16 ./downloaded-runs
```

Or use `scp -r` from PowerShell with the exact server run directory. Existing
download directories are not overwritten by the helper.

## Results and timing interpretation

Each run retains original/resolved configuration, manifest, capability report,
both Git SHAs and actual source hashes/diffs, source archives, build/package
metadata, `.run`/`.whl` packages, kernel binaries, stage logs, correctness results,
benchmark samples, and raw/processed profiler output. Failures preserve their
status and all artifacts already produced.

`benchmark/process.json` reports total benchmark process wall time, including
Python startup/imports, input setup, warmup, measurement, and writing results.
`benchmark/results.json` separately reports the synchronized measurement loop
and per-invocation NPU event regions. Event regions may contain dispatch gaps;
they are not kernel-only times. Profiler durations and simulator cycles are
reported separately in each backend's raw reports and `summary.json`, preserving
the vendor's column names and units. Do not compare simulator CPU runtime with
physical NPU latency.

## CPU-only checks

On Linux, the standard library is sufficient. On Windows, create a local virtual
environment and install the timezone database:

```powershell
python -m venv .cache/venv
.cache/venv/Scripts/python.exe -m pip install -r requirements-dev.txt
.cache/venv/Scripts/python.exe -m unittest discover -s scripts/python -p test_experiment.py -v
```

Linux additionally exercises the actual file lock. These checks do not replace
building and profiling on the NPU server.
