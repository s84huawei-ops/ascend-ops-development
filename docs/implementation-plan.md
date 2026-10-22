# Ascend framework implementation plan

Status: implemented and validated on the target server, following the user's authorization to move beyond planning on 2026-10-22. The original decisions and dependencies are recorded in the [wayfinder map](../.scratch/ascend-framework/map.md). The [usage guide](usage.md) describes the implemented interface; [validation results](validation.md) distinguish actual checks from remaining limitations.

## Goal and agreed scope

The framework should support a repeatable cycle: select an operator version → build → install/register → check correctness through `torch_npu` → benchmark without a profiler → profile with the selected backend → save all results.

Deployment will take place in `s84huawei-ops/ascend-ops-development`, on the server accessible through `ssh huawei-blue-npu`, under `/home/s84435910/docker_workspace`. All operator build and execution stages run inside a container based on `quay.io/ascend/vllm-ascend:v0.23.0rc1`, created using the existing `docker_run.sh` as a starting point.

The user confirmed:

- A demo operator in the separate repository `s84huawei-ops/ascend-op-demo`, included as the `op-src` submodule.
- Local editing and version transfer through Git. Results remain on the server and can be downloaded.
- The `scripts/python` directory, correcting the typo in the original description.
- One NPU selected in the configuration, with an availability check. The initial candidate is NPU 6; that device has not been reserved.
- Separate measurements of total benchmark time and operator execution on the NPU, plus profiling artifacts.
- Two selectable backends: a physical NPU with `msprof`, and simulation through `msprof op simulator`.

## Project structure

```text
ascend-ops-development/
  docker_run.sh          # adapted version of the existing server script
  op-src/                # submodule: independent operator history
  docs/                  # usage, plan, environment, research, and decisions
  scripts/
    bash/                # container, versions, build, execution, result download
    python/              # configuration, adapter, correctness, benchmark, results
  configs/               # explicit experiment configurations
  runs/                  # results excluded from Git
  .scratch/
    ascend-framework/    # local map and separate decision tickets
```

Kernel sources, the operator definition, build system, simulator ABI/data preparation, and PyTorch binding belong in the independent `op-src` repository. The framework implements container orchestration, configuration, experiment stages, and result storage. The operator repository has been created as a private repository.

## Findings from completed research

The [integration research](research/torch-npu-integration.md) recommends AddCustom in Ascend C, invoked through ACLNN and a standalone PyTorch C++ adapter: an operator `.run` package and an adapter `.whl`. The proposed interface is `torch.ops.ascend_demo.add(x, y)`, after importing `torch_npu` and the adapter. The initial scope is eager forward with two contiguous FP16 tensors of the same shape, without broadcasting. Pinning a specific example, handling tails, and ensuring ABI compatibility with the actual stack remain deployment gates; rebuilding all of `torch_npu` is not planned.

The [profiling research](research/profiling-backends.md) confirms through documentation that `msprof op simulator` can wrap a Python/PyTorch program. A separate C++ harness need not be assumed in advance; compatibility with the custom adapter and installed release must be tested. Preserve the corresponding kernel symbols and sources, use a small simulation case, and report simulator runtime separately. On-device `msprof op` remains an option for detailed diagnostics alongside the required general `msprof`.

Isolation consideration: selecting physical NPU6 for the benchmark does not automatically determine the simulator device/runtime. Restrictions in older releases concerning logical device 0 must be verified for CANN 9.0.1 before implementing the simulation launcher. Do not use shared physical NPU0 as a default workaround.

Both reports cite official sources and distinguish recommendations from successful tests. Implementation has since verified the custom ACLNN operator through torch_npu, including unaligned tails, and generated valid msprof and msprof op reports. The Python simulator path produces data but crashes during process teardown in this stack. Binary simulation of the identical compiled kernel passes and is the default. See the usage guide for the explicit simulator entry modes.

## Implementation sequence

| Step | Outcome | Prerequisite / decision |
| --- | --- | --- |
| 1. Versioning | Demo repository created, first operator commit, submodule pinned to a SHA, tag/branch/SHA switching script, and synchronization instructions | [Operator versioning](../.scratch/ascend-framework/issues/03-operator-versioning.md), integration research findings |
| 2. Dedicated container | Adapted `docker_run.sh`, correct repository and result mounts, selected NPU, and an actual toolchain report from a fresh container | [Container contract](../.scratch/ascend-framework/issues/04-container-contract.md) |
| 3. Minimal operator | Demo built, registered, and actually invoked through `torch_npu`; a test comparing its output with a reference | Integration research; test in the target container |
| 4. Runner | Validated configuration; build/register/test/benchmark; atomic run directory creation, logs, and metadata even on failure | [Experiment contract](../.scratch/ascend-framework/issues/05-experiment-contract.md) |
| 5. Profiling | Separate device and simulator runs; preserved raw reports and an explanation of their meaning | Profiling research; working minimal operator |
| 6. Reproduction and instructions | Switch operator version, run a second experiment, reproduce the first from recorded SHAs, and document usage | [Rollout and acceptance criteria](../.scratch/ascend-framework/issues/06-rollout-and-acceptance.md) |

The user subsequently authorized full implementation. The repository, submodule, scripts, and dedicated container have been created; acceptance is checked against the criteria below.

## Versions and transferring work

The user-approved workflow is local editing, committing/pushing operator sources, recording their SHA in the framework repository, committing/pushing the framework, and fetching both versions on the server. A version specified by a tag or branch must be resolved to a SHA and recorded in the results. An experiment should not silently fetch the latest branch without recording the change.

Implemented contract: version switching refuses to overwrite local changes and stages the selected SHA in the framework. Experiments require clean sources unless `allow_dirty=true` explicitly enables development mode. Both source archives and hashes/diffs are retained. Operator versions and actual source contents select distinct cache entries; the switching script does not delete local changes or results.

## Separating measurements

Implemented result definitions:

| Result | Meaning |
| --- | --- |
| Total benchmark time | Wall-clock time of the benchmark process without an active profiler; the report explicitly states whether data preparation, imports, and warmup are included |
| Measured section time | Time of the loop after warmup, with NPU synchronization at measurement boundaries |
| NPU invocation region time | Event measurements on a stream, with an explicit scope; may include gaps between kernels; statistics and sample count |
| msprof metrics | A separate execution with the NPU profiler; kernel times, available counters, and raw reports |
| Simulator metrics | A separate execution of the kernel model; cycles and model metrics, with the SoC and limitations identified |

Profiler results do not replace total benchmark time measured without profiling. Correctness precedes measurement; warmup, iteration count, shapes, types, and data seed come from the saved configuration. Stream events surround individual calls; synchronized loop time and complete subprocess time are recorded separately.

## Experiment directory

Required name: `runs/<MMDD-HHMM>-<name>`, with `name=exp` by default. The configurable timezone defaults to Europe/Warsaw. Container time and run names use this default consistently.

Example layout:

```text
runs/1022-1430-add-fp16/
  config.original.json
  config.resolved.json
  manifest.json
  status.json
  logs/
  correctness/
  benchmark/
  profiling/
    msprof/
    simulator/
```

The implementation uses strictly validated JSON. The manifest, sources.json, capability report, and stage logs together identify both Git SHAs, dirty state, image reference/ID, device, SoC, driver/toolchain/package versions, selected environment variables, stages, and executed commands. The entire process environment is not copied.

Directories are created before compilation and tests. Failures retain configuration, logs, and final status. Atomic creation refuses collisions without modifying previous results; choose the next minute or a different `name`. Raw reports remain available alongside processed results. Each run also retains source archives, build packages, and kernel binaries.

## Acceptance criteria

| Test | Evidence required for acceptance |
| --- | --- |
| Fresh container | Recorded versions and image; access to the specified NPU; no dependency on modifications in someone else's running container |
| Build and registration | Exit code and log; identification of the artifact and sources; loading in a new process after environment setup |
| Invocation through torch_npu | Custom operator output from the NPU compared with a reference for the agreed shapes/types, including a size that is not a multiple of the block size |
| Benchmark | Repeatable inputs, warmup, synchronization, sample count, and separate timing scopes recorded in the results |
| msprof on the NPU | A nonempty report containing the custom kernel and meaningful events; the tool's exit code alone is insufficient |
| Simulator | Confirmed execution of the custom kernel for the Ascend910B2 model and a report; the presence of a binary or simulator directory is insufficient |
| Version switching | Two different operator SHAs with their corresponding builds and manifests; reproduction of the earlier version |
| Failure scenarios | Clear errors for a busy NPU, invalid configuration, missing tool, and directory collision; preserved logs and status |
| Concurrency | A second framework run does not simultaneously use the same NPU reserved by the framework; an availability check is not treated as a guarantee of exclusive access against other processes |

The `unsupported` mode must be explicitly distinguished from success and execution failure. If the simulator proves unsupported in the target environment, implementation must not claim that the two-backend requirement passed acceptance; scope or integration will need a further decision.

## Implementation handoff

The implementation follows the agreed local-edit/Git/server workflow, using one selectable physical NPU, the pinned image, and isolated operator installation prefixes. The initial planning-only scope was superseded by the user's request to implement. Runtime findings and remaining limitations must be reported explicitly rather than inferred from tool availability.

The user committed and published the English planning documentation on `main` before implementation. Separate research branches were not created. Operator sources are maintained in the private submodule repository; results and build caches are excluded from Git.
