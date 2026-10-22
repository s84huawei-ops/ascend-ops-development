# Ascend operator profiling backends

Research date: 2026-10-22. Scope: map and plan; no compilation, operator execution, profiling, or NPU initialization. The recommendations below are not additional user decisions.

## Findings and limits of certainty

I recommend separate runs: a benchmark without a profiler, on-device `msprof`, and `msprof op simulator`. On-device `msprof op` can be an additional diagnostic option. Each run should have its own status and artifacts. Simulation analyzes instructions; its CPU runtime is not the operator's NPU execution time.

**The simulator can wrap Python/PyTorch. A separate C++ program is not a general requirement.** The official guide includes a PyTorch scenario and `msprof op simulator ... python3 test_add.py`; it also describes separate kernel launch and ACLNN variants. For kernel launch, the example requires a separate binary variant linked with simulator libraries, while the ACLNN example allows a shared executable. Do not apply the requirements of one invocation method to the others. The documentation warns that a kernel filter does not eliminate the cost of simulating the program's other operators. [Official msOpProf scenarios](https://github.com/Ascend/msopprof/blob/master/docs/en/user_guide/msopprof_usage.md).

This confirms a tool capability, **not successful execution of the future demo** through the specific torch_npu adapter on this installation. The contents of a fresh container and operation of both backends remain to be checked. The [local inventory](../environment-inventory.md) confirms only inspection of `msprof op` help and the presence of simulator directories in an already used container.

The vLLM-Ascend table for `v0.23.0rc1` specifies CANN `9.0.1`, torch `2.10.0`, and torch_npu `2.10.0.post2`, consistent with the observed environment. This does not certify every profiler feature. [vLLM-Ascend compatibility matrix](https://docs.vllm.ai/projects/ascend/en/v0.23.0rc1/community/versioning_policy.html#release-compatibility-matrix).

The `Ascend/msopprof` guides on `master` are updated independently of the image. During implementation, record the tool version/identifier and full `--help` output from the container. A feature described in newer documentation cannot automatically be treated as supported by CANN 9.0.1. The tool is part of CANN and can also be updated separately. [msOpProf installation and versioning](https://github.com/Ascend/msopprof/blob/master/docs/en/install_guide/msopprof_install_guide.md).

## What to measure

| Measurement | Planned framework definition | Interpretation |
| --- | --- | --- |
| `process_wall_s` | Parent process monotonic clock: child start → completion and receipt of result | Entire benchmark: Python startup, imports, data preparation, warmup, computation, validation; excludes building the operator |
| `benchmark_wall_s` | Monotonic clock inside the child, with NPU synchronization before and after the measurement loop | The specified loop, including Python/dispatch overhead and waiting for the device; excludes warmup and result checking |
| `device_region_ms` | NPU events with timing enabled on the correct stream, synchronizing the final event | Region time on the stream timeline; may include gaps between kernels and does not guarantee kernel-only time |
| `kernel_duration_us` | Read the relevant `msprof` report rows, including name/op type/task/stream and invocation count | Kernel task time observed by the profiler in a separate run |
| `profiler_process_wall_s` | Profiler process start → completion of collection and export | Total profiling cost; do not publish as the performance baseline |
| `simulator_process_wall_s` | Simulation start → completion and export | CPU/simulation cost; not execution time on a physical Ascend910B2 |
| `simulated_cycles` / `simulated_time_us` | Simulator report fields for a core/instruction | Instruction execution model, a separate metric domain |

`torch_npu.npu.Event(enable_timing=True)` provides `record`, `elapsed_time` in milliseconds, and `synchronize`; timing is disabled by default. The event stream must belong to the same device. This forms the basis of the proposed region measurements. [Event source and contract in v2.10.0](https://github.com/Ascend/pytorch/blob/v2.10.0/torch_npu/npu/streams.py).

Methodology recommendations: warm up the operator outside the measurement window, prepare inputs in advance, use fixed seeds/shapes/dtypes, and record the iteration count. Report samples, median and percentiles, and sum/mean with invocation count. Summing overlapping task durations does not give total timeline duration. For a small Add confined to one stream, these views can be compared later. Do not subtract an arbitrary "profiler overhead" from measurements; run the baseline and profilers separately with identical SHAs and data.

## `msprof` on a physical NPU

`msprof [options] <app>` and `--application=<app>` are documented. Application arguments follow its name; the official documentation prefers this form. [General msprof command, CANN 9.0.0](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/devaids/Profiling/atlasprofiling_16_0010.html).

The following is an **integration sketch**, based on documented syntax. `scripts/python/profile_case.py` does not exist yet; flags must be checked against the tool's help in a fresh container. The configuration selects the device before torch_npu initialization, following the container isolation plan.

```bash
msprof --output="$RUN_DIR/msprof" \
  --ascendcl=on --runtime-api=on --task-time=on \
  --ai-core=on --aic-metrics=PipeUtilization \
  python3 scripts/python/profile_case.py --config "$RUN_DIR/config.yaml"
```

`--task-time` covers operator submission and execution time; data appears in reports including `op_summary`, `op_statistic`, and `task_time`. `PipeUtilization` covers the duration and share of compute and transfer instructions. `--task-time=l0` omits some information and reduces overhead relative to l1/on; it is an option for later validation, not a default replacement. [msprof collection and metrics, CANN 9.0.0](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/devaids/Profiling/atlasprofiling_16_0011.html).

Plan: preserve the profiler's entire output directory and produce a framework summary in addition. Result filtering must identify the actual demo kernel. Command success or the presence of an arbitrary `Add` operator is insufficient: the profiler may record input initialization, copies, or a different library.

An optional detailed on-device variant, also a **sketch**:

```bash
msprof op --kernel-name=add_custom --launch-count=1 \
  --warm-up=5 --aic-metrics=Default --output="$RUN_DIR/msprof-op" \
  python3 scripts/python/profile_case.py --config "$RUN_DIR/config.yaml"
```

`msprof op` supports kernel selection, launch count, and its own warmup; without a filter, it collects the first operator by default. It may use replay: `application` mode reruns the entire application. Check Roofline, Occupancy, Source, and TimelineDetail separately, including their allowed combinations on A2. Do not plan to enable all of them automatically. The guide advises against compiling with `-O0`. [On-device msOpProf guide](https://github.com/Ascend/msopprof/blob/master/docs/en/user_guide/msopprof_user_guide.md).

Detailed reports contain per-core data, including `aic_time(us)`, `aiv_time(us)`, cycles, Vector/Cube shares, and transfer sizes. A core's time after task assignment is not automatically the entire operator's duration. [msOpProf data description](https://github.com/Ascend/msopprof/blob/master/docs/en/user_guide/msopprof_performance_data.md).

## Simulator and artifact requirements

Python-mode **sketch**; `add_custom` is a planned name, and `Ascend910B2` requires confirmation of compatibility with the build and local simulator:

```bash
msprof op simulator --soc-version=Ascend910B2 \
  --kernel-name=add_custom --launch-count=1 \
  --output="$RUN_DIR/simulator" \
  python3 scripts/python/profile_case.py --config "$RUN_DIR/config.yaml"
```

The simulator library for the appropriate SoC is required; for A2/A3, documentation specifies `${INSTALL_DIR}/tools/simulator/<soc-version>/lib`. Mapping code to instructions and the stack requires `-g` debug information and preservation of the corresponding sources. The `--config` mode is a separate path: JSON describes launching a binary kernel `.o`; it is not the framework's experiment YAML. `--export` parses an existing dump; line mapping also requires the kernel binary. Documentation does not support printing device variables through Python `print` during simulation. [Simulator guide and options](https://github.com/Ascend/msopprof/blob/master/docs/en/user_guide/msopprof_simulator_user_guide.md).

The JSON path requires reconstructing the launch contract: `kernel_name`, `kernel_path`, `ca` mode, the appropriate `magic`, input/output parameters, and workspace and tiling data if used. An `.o` alone is insufficient for a correct experiment. This is a potential diagnostic option after establishing the exact operator ABI, without testing torch_npu registration. [msOpProf JSON schema and parameters](https://github.com/Ascend/msopprof/blob/master/docs/en/user_guide/extended_functions.md#json-configuration-file-description).

Older CANN 8.0.RC3 documentation required logical card 0 for msProf simulator and warned against changing visible cards; it also explicitly showed `msprof op simulator python a.py`. This is a version-specific warning, **not evidence of an identical restriction in CANN 9.0.1**. Do not automatically copy physical ID 6 into the simulator script or assume a need to occupy physical card 0. Verify the actual mode of the loaded libraries and absence of launches on a shared card before declaring the backend available. [CANN 8.0.RC3, section 8.1.2, document page 170](https://www.hiascend.com/doc_center/source/zh/canncommercial/80RC3/devaids/opdev/optool/CANN%E5%95%86%E7%94%A8%E7%89%88%208.0.RC3%20%E7%AE%97%E5%AD%90%E5%BC%80%E5%8F%91%E5%B7%A5%E5%85%B7%E6%8C%87%E5%8D%97%2001.pdf).

The simulator generates a raw `dump`, per-core results, `*_code_exe.csv`, `*_instr_exe.csv`, `trace.json`, and `visualize_data.bin` for MindStudio Insight. CSV fields include cycles and `running_time(us)`. Also preserve `.o`, symbols, compilation options, and sources for the given SHA; do not aggregate instruction times from different pipelines as operator latency. [Simulator data description](https://github.com/Ascend/msopprof/blob/master/docs/en/user_guide/msopprof_simulator_performance_data.md).

The demo therefore needs two separate elements: a deployed kernel/host API and a registered PyTorch adapter that actually invokes it. Installing the operator package alone does not automatically create `torch_npu.npu_add_custom`. An older official example shows OpPlugin registration and ACLNN invocation. I do not recommend rebuilding all of torch_npu using the PyTorch 2.1.0 example without verification. The target adapter for 2.10.0 should be established by the demo research. [Official custom operator integration example](https://www.hiascend.com/document/detail/zh/canncommercial/80RC1/developmentguide/opdevg/Ascendcopdevg/atlas_ascendc_10_0038.html).

## Statuses and acceptance gates — recommendation

Each backend: `not_requested`, `pending_validation`, `ok`, `unsupported`, `blocked`, `failed`; a missing numeric result is `null` with a reason, never zero pretending to be a result.

- `unsupported`: a documented lack of mode/SoC/metric support, confirmed by capability checks; record the specific limitation and evidence.
- `blocked`: missing files, a busy device, missing permissions, or a reservation wait timeout. This does not mean the hardware lacks support.
- `failed`: the task started, but the tool/application/export errored, timed out, or did not produce the expected kernel/artifact. Do not change this to `unsupported` without diagnosis.
- `ok`: correctness confirmed in the appropriate stage, process and parser completed successfully, nonempty artifacts identify the correct operator, and required metrics have units and provenance.

An experiment whose required backend is `failed`, `blocked`, or `unsupported` cannot be a complete success. Allowing partial results must be explicit in the configuration; the report shows each run's status. Do not silently run on the NPU as a substitute for unavailable simulation.

## Validation plan for a fresh container — later implementation

1. Confirm the image digest, CANN/torch/torch_npu/toolchain versions, help and version of each tool, Ascend910B2 library directory, and loaded environment. Save a capability report; do not overwrite the image's installations to hide differences.
2. Verify how the physical NPU selected in configuration maps to the in-container ID, availability, and framework lock. A local lock does not protect against other processes: additionally check the device before starting and record its state. Do not allow concurrent profilers on the same NPU.
3. Build the demo from the agreed SHA, separating build/performance and build/debug artifacts while retaining the same optimization. Check correctness against a PyTorch reference and actual invocation of the uniquely identifiable kernel on the correct device.
4. Run the benchmark without a profiler: warmup, synchronized wall clock, and optional events on the same stream. Verify invocation count and exclusion of validation and data generation from the measurement window.
5. Run minimal `msprof` separately, confirm the correct kernel and timing, and preserve raw output/exports/logs. Only then check additional metrics and optional `msprof op`.
6. Validate simulator configuration without disrupting shared physical NPU0. Start with very small inputs and one invocation; generate data on the CPU, avoiding unnecessary device operations. Check the instruction report, code/symbols, and kernel identity. If the Python path does not work in this release, diagnose it and evaluate a separate ACLNN/kernel harness; its success must not be presented as successful torch_npu integration.
7. Check negative cases: incorrect kernel filter, missing symbols, unavailable metric, missing libraries, application failure, busy NPU, simulator timeout, and failed export. Verify statuses and preservation of logs on failure as well.

Each test should preserve the configuration, resolved configuration, framework/submodule SHAs, dirty state, SoC, physical/logical IDs, image digest, tool versions, full argv and exit code, timings, logs, and artifact manifest under `runs/<MMDD-HHMM>-<name>/`. Directory creation must detect a collision in the same minute with the same name; the behavior remains to be settled in the framework plan.
