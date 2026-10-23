# Using the Ascend operator framework

The main workflow assumes that you edit sources and run commands directly on
the NPU server. All Bash examples run there, from the framework repository,
unless stated otherwise. Notes labelled **If you work locally** describe the
additional transfer steps for a local checkout or Windows analysis tools.

## Prerequisites and repositories

The server workspace is `/home/s84435910/docker_workspace/ascend-ops-development`,
accessible through `ssh huawei-blue-npu`. The host needs Docker, Python 3.9+,
Git, the installed Ascend driver, and the required image already present.
The framework uses a single Ascend910B2. Device 6 is the default physical NPU.

The `op-src` submodule points to the private `s84huawei-ops/ascend-op-demo`
repository. Use an account with access when cloning or updating it:

```bash
# For a new server checkout, run from /home/s84435910/docker_workspace:
git clone --recurse-submodules https://github.com/s84huawei-ops/ascend-ops-development.git
# For an existing clone:
cd /home/s84435910/docker_workspace/ascend-ops-development
git submodule update --init --recursive
```

The demo implements FP16 addition through a custom Ascend C kernel, generated
ACLNN entry points, and a standalone PyTorch C++ adapter. It supports nonempty
contiguous inputs of equal shape with zero storage offsets. It has no broadcasting,
autograd, or torch.compile support. It is intended to validate the development
workflow, not to compete with the optimized built-in addition operator.

## Typical workflow after changing the operator

### 1. Edit the operator on a branch

Start in the server workspace:

```bash
cd /home/s84435910/docker_workspace/ascend-ops-development
git -C op-src status
# Once for a new task; choose your own unused branch name:
git -C op-src switch -c operator-change
```

Submodule initialization and version switching can leave `op-src` with a detached
HEAD. Create a branch before developing changes, or keep using your existing
development branch. Edit the kernel, host code, build script, or adapter inside
`op-src`; framework scripts and experiment configurations belong to the parent
repository.

### 2. Prepare the experiment configuration

Choose a starting configuration:

| Configuration | Purpose |
| --- | --- |
| `configs/demo.json` | Build, correctness checks, and an unprofiled benchmark |
| `configs/msprof.json` | The same stages plus application-level NPU profiling |
| `configs/simulator.json` | The same stages plus binary kernel simulation |
| `configs/all.json` | The same stages plus msprof, msprof op, and binary simulation |

For a named profiling experiment, copy a starting configuration and edit it:

```bash
cp configs/all.json configs/add-my-change.json
```

Set `name` to a descriptive value such as `add-my-change`. Check `device`,
input shape, warmup, iterations, and requested profilers. Keep these settings
consistent with the baseline when comparing operator versions. If you rename
the kernel, update `profiling.kernel` as well. An explicit `operator.ref` must
resolve to the selected operator commit; omit it or use null when no additional
revision check is needed. See [Configuration](#configuration) for all options.

### 3. Commit the operator and experiment definition

The operator and framework have independent histories. Commit the operator first,
then record its submodule pointer and the experiment configuration in the framework:

```bash
git -C op-src add .
git -C op-src commit -m "Change the demo operator"
git add op-src configs/add-my-change.json
git commit -m "Pin the operator and define its experiment"
git -C op-src status --short
git status --short
```

Both status commands should print nothing. Normal configurations require clean
sources. Commits are sufficient to run on the same server; pushing is useful for
backup and sharing. When publishing, push the operator branch first so the parent
repository never points at an unavailable commit:

```bash
# Optional publication from the development branch created above:
git -C op-src push -u origin HEAD
git push
```

<details>
<summary>If you work locally: transfer committed sources before running</summary>

Perform steps 1-3 in your local checkout. Then, from that checkout, transfer the
same committed versions to the server. The SSH-bundle method works from PowerShell
and does not require GitHub credentials on the server:

```powershell
python scripts/python/sync_server.py
```

It fast-forwards the server framework and checks out its exact operator pin.
Both local repositories and the server checkout must be clean; divergent history
and remote changes are not overwritten. The server clone must already have an
initial commit. Results and build caches are not transferred.

Alternatively, push both repositories as described above, then run on the server:

```bash
cd /home/s84435910/docker_workspace/ascend-ops-development
git pull --ff-only
git submodule update --init --recursive
```

The GitHub route requires server-side read access to the private operator
repository. After either transfer method, continue with step 4 on the server.

</details>

### 4. Run the experiment

```bash
bash scripts/bash/run.sh configs/add-my-change.json
```

The wrapper validates configuration, selects the physical NPU, creates or reuses
the dedicated container, and starts the runner inside it. The runner checks device
availability, captures sources/environment, builds and activates the operator,
checks correctness, runs the unprofiled benchmark, and launches each requested
profiler separately. No manual package installation or registration is required.

Source changes select a new build cache entry automatically. Use `build.force`
only when you need to rebuild the selected entry. Leave sources unchanged until
the experiment finishes; source snapshots describe their state at the start.

The runner prints the run directory and final status. It uses
`runs/MMDD-HHMM-name`, with Europe/Warsaw as the default timezone. A same-minute
name collision is rejected without overwriting earlier results.

### 5. Inspect the results

Use the directory printed by the runner; the following is a completed example:

```bash
RUN=runs/1022-1349-add-all
cat "$RUN/status.json"
cat "$RUN/benchmark/process.json"
cat "$RUN/benchmark/results.json"
```

Start with `status.json`: `ok` means all requested stages passed; `partial` means
an explicitly optional profiler did not succeed. Failed/interrupted runs
retain their status and existing artifacts. Inspect the stage log in `logs/` before
interpreting timings. Individual profiling stages can also be blocked by a busy NPU.

Profiler output lives in `profiling/<backend>/raw/`; stages that reach report
parsing also write `summary.json`. If output is missing, check the stage status/log.
Read [Results and timing interpretation](#results-and-timing-interpretation) before
comparing values, then follow [Visualization in MindStudio Insight](#visualization-in-mindstudio-insight)
to locate and open `visualize_data.bin`.

### 6. Compare with a baseline

Keep the completed run directory. Repeat the experiment for another operator
revision using the [version-switching helper](#versions-builds-and-custom-programs),
commit the new parent pointer, and use a distinct experiment name. Preserve input
shape/dtype, seed, warmup, iterations, profiler settings, NPU and image/toolchain
when assessing the effect of a source change.

Compare like-for-like timing scopes and retain sample counts. The simulator's
input shape is configured separately. A failed or missing measurement is
unavailable, not zero. The [Markdown reporting plan](#markdown-reporting-planned)
describes the intended per-experiment and selected-experiment tables.

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

Switch to an existing operator tag, branch, or SHA:

```bash
bash scripts/bash/switch_operator.sh v0.1.0 --fetch
git commit -m "Pin operator v0.1.0"
```

Switching refuses to overwrite operator changes and stages the new submodule
pointer. The optional `--fetch` updates refs/tags first; omit it for local refs.
The selected revision is checked out detached. Before making further changes,
switch back to your development branch or create a new one. Committing the parent
pointer is sufficient for a server experiment; publication is optional.

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

## Container and profiling behavior

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

The container exposes only the selected `/dev/davinciN` plus management devices.
The driver maps that single device to logical `npu:0`; physical selection is
recorded separately in `ASCEND_PHYSICAL_DEVICE_ID`. `env.sh` clears
`ASCEND_RT_VISIBLE_DEVICES`: a physical ID would refer to an already remapped
logical device. The host timezone database is mounted read-only for Europe/Warsaw.

The container name defaults to `ascend-ops-dev-npu6`. `ASCEND_CONTAINER` can
select another name. Image, workspace, device, and container settings are checked
before reuse. Outdated containers are not silently replaced. Recreate only the
framework's own container or choose a new name. Builds and results remain on
the host workspace when a container is removed.

The filesystem lock coordinates runs in one workspace. It does not reserve
hardware against unrelated users; `npu-smi` is checked before stages as well.
A busy device or a held lock causes failure with retained diagnostics.

## Results and timing interpretation

Each run retains original/resolved configuration, manifest, capability report,
both Git SHAs and actual source hashes/diffs, source archives, build/package
metadata, `.run`/`.whl` packages, kernel binaries, stage logs, correctness results,
benchmark samples, and raw/processed profiler output. Failures preserve their
status and all artifacts already produced.

Use the following mapping when reading or assembling a timing table:

| Measurement | Source | Unit and scope |
| --- | --- | --- |
| Complete benchmark process | `benchmark/process.json`: `process_wall_s` | Seconds; startup/imports, input preparation, warmup, measurement, and writing results |
| Measured benchmark loop | `benchmark/results.json`: `benchmark_wall_s` | Seconds; the synchronized loop after warmup, including dispatch and event recording |
| NPU event regions | `benchmark/results.json`: `device_region_ms`, median/mean/p95 fields | Milliseconds per call; stream regions may contain dispatch gaps |
| Kernel tasks on the NPU | msprof `op_summary_*.csv` and `op_statistic_*.csv` | Microseconds; task durations or aggregates, with invocation count |
| Selected kernel task | msprof op `OpBasicInfo.csv`: `Task Duration(us)` | Microseconds; task duration in a separate profiling execution |
| Per-core activity | msprof op `PipeUtilization.csv`, `Memory*.csv`, and related files | Microseconds, cycles, ratios, bandwidth; inspect each column's unit and core ID |
| Simulated execution | Simulator `*_code_exe.csv`, `*_instr_exe.csv` and other raw reports | Model time/cycles for the configured simulator shape; not physical-device latency |

The `process_wall_s` field in a profiler stage measures collection and export
cost, rather than operator latency. `api_statistic_*.csv` contains CPU API times;
an `aclnnAddCustom` row is not a kernel execution measurement. Profiling can include
warmup calls, so retain the invocation count and inspect individual rows before
comparing averages. `benchmark.warmup` and `benchmark.iterations` configure the
unprofiled benchmark; they do not define the profiler's collection window.

Stream regions, task durations, per-core activity and simulator results have
different boundaries. Concurrent pipeline activity must not be summed as if it
were sequential. Preserve these distinctions when comparing operator versions.

### Reading summary.json

Each requested backend writes `profiling/<backend>/summary.json`. It contains:

| Field | Meaning |
| --- | --- |
| `matched_reports` | Sorted unique CSV paths containing at least one matching row, relative to that backend's `raw/` directory |
| `sample_rows` | Up to 30 matching rows across all CSV files; each has `report` and the original column/value mapping in `values` |

The parser scans nonempty CSVs recursively in sorted path order, examining up to
100,000 data rows per file. A row qualifies when both conditions hold:

1. The configured kernel text occurs in the relative file path or serialized
   row. This comparison ignores case and underscores.
2. At least one column whose name contains `time`, `duration`, `cycle` or `latency`
   has a numeric value, including scientific notation.

Only the first 30 qualifying rows are saved as samples; scanning continues to
discover other matching reports. Values remain CSV strings with their original
column names. Rows are neither ranked by duration nor aggregated.

This is a diagnostic sample, not a complete performance summary. It can include
CPU API rows, and it can omit per-core CSVs such as `PipeUtilization.csv` when
neither the path nor rows contain the kernel name. Read the full raw report for
those metrics. The profiler's own kernel-name filter is case-sensitive even
though this summary matching is case-insensitive.

## Visualization in MindStudio Insight

For device metrics and simulation together, run `configs/all.json` or an
experiment configuration derived from it, following the workflow above.
`configs/simulator.json` requests simulation alone in addition to the standard
build/correctness/benchmark stages. No manual invocation of the simulator is
needed for the demo.

After confirming the requested profiler stages succeeded, locate their files on
the server using the run directory printed by the runner:

```bash
RUN=runs/1022-1349-add-all
find "$RUN/profiling" -type f -name visualize_data.bin
```

The validated default configuration produced both of these layouts:

```text
profiling/msprof-op/raw/OPPROF_.../visualize_data.bin
profiling/simulator/raw/OPPROF_.../simulator/visualize_data.bin
```

The NPU file contains device profiling metrics; the simulator file contains
simulation pipeline and instruction information. Available views depend on
the collected metric groups and tool version. General `msprof` uses its own
CSV/timeline output and does not produce either of these two operator-view files.
See the official [device profiler guide](https://github.com/Ascend/msopprof/blob/master/docs/en/user_guide/msopprof_user_guide.md)
and [simulator guide](https://github.com/Ascend/msopprof/blob/master/docs/en/user_guide/msopprof_simulator_user_guide.md).

Import the chosen file into MindStudio Insight where you run the GUI. Keep the
full experiment directory, including raw reports, source archives and kernel
binaries, so these remain available when investigating or mapping source code.
CSV metrics and `summary.json` can be inspected directly on the server.

<details>
<summary>If you work locally: download results and open them on Windows</summary>

From local PowerShell, substitute the exact run name. This example refuses an
existing download directory, preserving previous results:

```powershell
$runName = "1022-1349-add-all"
$destination = Join-Path ".\downloaded-runs" $runName
if (Test-Path -LiteralPath $destination) {
    throw "Download destination already exists: $destination"
}
New-Item -ItemType Directory -Force -Path ".\downloaded-runs" | Out-Null
scp -r "huawei-blue-npu:/home/s84435910/docker_workspace/ascend-ops-development/runs/$runName" ".\downloaded-runs"
if ($LASTEXITCODE -ne 0) { throw "Result download failed" }
Get-ChildItem -LiteralPath $destination -Recurse -Filter visualize_data.bin
```

Open the selected local `visualize_data.bin` in MindStudio Insight on Windows.
Preserving its backend directory distinguishes the NPU and simulator files,
which have the same filename.

If you use Git Bash or a local Linux shell, the existing helper downloads a full
run and also refuses an existing destination:

```bash
bash scripts/bash/fetch_run.sh 1022-1349-add-all ./downloaded-runs
```

</details>

## Markdown reporting (planned)

The current runner writes JSON and raw profiler reports. Automatic Markdown
generation remains an implementation task; the intended outputs are:

- `report.md` inside each experiment, containing revision/configuration/status,
  timing tables with units and sample counts, links to backend summaries/raw
  reports, and links to available `visualize_data.bin` files.
- A separate Markdown comparison for explicitly selected experiments, with
  like-for-like timings and speedup relative to the first selected baseline.
  Differences in data shape, environment or measurement settings must be visible.

Reports should read benchmark JSON and the appropriate full profiler CSVs rather
than aggregate the 30 diagnostic sample rows. CPU, NPU task, per-core and simulator
measurements need separate labels. Failed, absent or unsupported measurements
should have a reason rather than a zero value. Generating a comparison should
reuse retained artifacts without rerunning experiments.

## CPU-only checks

On the server, the standard library is sufficient:

```bash
python3 -m unittest discover -s scripts/python -p test_experiment.py -v
```

> **If you work locally on Windows:** create a local virtual environment and
> install the timezone database before running the CPU-only checks:

```powershell
python -m venv .cache/venv
.cache/venv/Scripts/python.exe -m pip install -r requirements-dev.txt
.cache/venv/Scripts/python.exe -m unittest discover -s scripts/python -p test_experiment.py -v
```

Linux additionally exercises the actual file lock. These checks do not replace
building and profiling on the NPU server.
