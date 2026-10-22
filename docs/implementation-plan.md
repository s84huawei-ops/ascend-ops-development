# Ascend framework implementation plan

Status: draft plan; no implementation yet. User agreements were recorded on 2026-10-22. The canonical record of decisions and dependencies is the [wayfinder map](../.scratch/ascend-framework/map.md); this document organizes the future steps and acceptance criteria. Proposals that require a decision are marked and have their own tickets. Research findings do not replace tests on the server.

## Goal and agreed scope

The framework should support a repeatable cycle: select an operator version → build → install/register → check correctness through `torch_npu` → benchmark without a profiler → profile with the selected backend → save all results.

Deployment will take place in `s84huawei-ops/ascend-ops-development`, on the server accessible through `ssh huawei-blue-npu`, under `/home/s84435910/docker_workspace`. All operator build and execution stages run inside a container based on `quay.io/ascend/vllm-ascend:v0.23.0rc1`, created using the existing `docker_run.sh` as a starting point.

The user confirmed:

- A demo operator in a separate, future repository, `s84huawei-ops/ascend-op-demo`, included as the `op-src` submodule.
- Local editing and version transfer through Git. Results remain on the server and can be downloaded.
- The `scripts/python` directory, correcting the typo in the original description.
- One NPU selected in the configuration, with an availability check. The initial candidate is NPU 6; that device has not been reserved.
- Separate measurements of total benchmark time and operator execution on the NPU, plus profiling artifacts.
- Two selectable backends: a physical NPU with `msprof`, and simulation through `msprof op simulator`.

## Proposed target structure

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

This describes responsibilities rather than a list of implemented files. Kernel sources, the operator definition, build system, and PyTorch binding belong in `op-src`; the general experiment workflow and result storage belong in the framework. The integration research and experiment contract ticket will establish the exact boundary.

## Findings from completed research

The [integration research](research/torch-npu-integration.md) recommends AddCustom in Ascend C, invoked through ACLNN and a standalone PyTorch C++ adapter: an operator `.run` package and an adapter `.whl`. The proposed interface is `torch.ops.ascend_demo.add(x, y)`, after importing `torch_npu` and the adapter. The initial scope is eager forward with two contiguous FP16 tensors of the same shape, without broadcasting. Pinning a specific example, handling tails, and ensuring ABI compatibility with the actual stack remain deployment gates; rebuilding all of `torch_npu` is not planned.

The [profiling research](research/profiling-backends.md) confirms through documentation that `msprof op simulator` can wrap a Python/PyTorch program. A separate C++ harness need not be assumed in advance; compatibility with the custom adapter and installed release must be tested. Preserve the corresponding kernel symbols and sources, use a small simulation case, and report simulator runtime separately. On-device `msprof op` remains an option for detailed diagnostics alongside the required general `msprof`.

Isolation consideration: selecting physical NPU6 for the benchmark does not automatically determine the simulator device/runtime. Restrictions in older releases concerning logical device 0 must be verified for CANN 9.0.1 before implementing the simulation launcher. Do not use shared physical NPU0 as a default workaround.

Both reports cite official sources and distinguish recommendations from successful tests. Neither backend nor the demo has been executed yet. Research is complete; contract tickets remain open.

## Future implementation sequence

| Step | Outcome | Prerequisite / decision |
| --- | --- | --- |
| 1. Versioning | Demo repository created, first operator commit, submodule pinned to a SHA, tag/branch/SHA switching script, and synchronization instructions | [Operator versioning](../.scratch/ascend-framework/issues/03-operator-versioning.md), integration research findings |
| 2. Dedicated container | Adapted `docker_run.sh`, correct repository and result mounts, selected NPU, and an actual toolchain report from a fresh container | [Container contract](../.scratch/ascend-framework/issues/04-container-contract.md) |
| 3. Minimal operator | Demo built, registered, and actually invoked through `torch_npu`; a test comparing its output with a reference | Integration research; test in the target container |
| 4. Runner | Validated configuration; build/register/test/benchmark; atomic run directory creation, logs, and metadata even on failure | [Experiment contract](../.scratch/ascend-framework/issues/05-experiment-contract.md) |
| 5. Profiling | Separate device and simulator runs; preserved raw reports and an explanation of their meaning | Profiling research; working minimal operator |
| 6. Reproduction and instructions | Switch operator version, run a second experiment, reproduce the first from recorded SHAs, and document usage | [Rollout and acceptance criteria](../.scratch/ascend-framework/issues/06-rollout-and-acceptance.md) |

These steps describe implementation after the decisions are settled. This session does not create the operator repository, submodule, scripts, or container.

## Versions and transferring work

The user-approved workflow is local editing, committing/pushing operator sources, recording their SHA in the framework repository, committing/pushing the framework, and fetching both versions on the server. A version specified by a tag or branch must be resolved to a SHA and recorded in the results. An experiment should not silently fetch the latest branch without recording the change.

Proposal to resolve: version switching refuses to overwrite local changes; running a measured experiment requires clean sources or an explicit development mode that records their diff. An artifact built for a previous operator version must not be silently reused after a version switch. The switching script does not delete local changes or results.

## Separating measurements

Proposed result definitions, to be approved in the experiment contract:

| Result | Meaning |
| --- | --- |
| Total benchmark time | Wall-clock time of the benchmark process without an active profiler; the report explicitly states whether data preparation, imports, and warmup are included |
| Measured section time | Time of the loop after warmup, with NPU synchronization at measurement boundaries |
| NPU invocation region time | Event measurements on a stream, with an explicit scope; may include gaps between kernels; statistics and sample count |
| msprof metrics | A separate execution with the NPU profiler; kernel times, available counters, and raw reports |
| Simulator metrics | A separate execution of the kernel model; cycles and model metrics, with the SoC and limitations identified |

Profiler results do not replace total benchmark time measured without profiling. Check correctness before measurement; warmup, iteration count, shapes, types, and data seed should come from the saved configuration. The exact event measurement method and statistics require clarification after choosing how the operator is invoked.

## Proposed experiment directory

Required name: `runs/<MMDD-HHMM>-<name>`, with `name=exp` by default. The timezone must be explicitly agreed; the host script currently selects Shanghai, while the user works in Europe/Warsaw.

Example layout, subject to approval:

```text
runs/1022-1430-add-fp16/
  config.original.yaml
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

The configuration format (YAML/JSON), filenames, and schema are not yet settled. The `manifest` should identify both Git SHAs, dirty state, image tag/digest/ID, device, SoC, driver/CANN/torch/torch_npu versions, environment variables needed for reproduction, stages, and executed commands. Do not copy the entire process environment, which may contain tokens.

Create the directory before compilation and tests. On failure, preserve the configuration, log, and final status. Proposal that preserves the exact naming format: if atomic directory creation encounters a collision, refuse to overwrite and require the next minute or a different `name`; the experiment contract ticket owns the final choice. Raw profiler reports remain available alongside processed results.

## Acceptance criteria for future implementation

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

## Next step in the map

Both independent research tasks are complete. The first decision to resolve together is [operator versioning](../.scratch/ascend-framework/issues/03-operator-versioning.md), followed by the container and experiment contracts. Under `wayfinder`, this session creates the map and completes research; it does not manually resolve further tickets.

The documents remain local working files, with no commits on `main`. Publication of the two reports and inventory on the proposed `research/...` branches, required by the `wayfinder` workflow, did not occur: automatic approval review required explicit authorization to send environment information to the remote repository. No PR was created and no changes were merged. Publication approval is not needed to read the plan or continue working on the local map.
