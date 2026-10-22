# Ascend operator development framework — decision map

Label: wayfinder:map

## Destination

An agreed implementation plan for the framework in `ascend-ops-development`: a versioned demo operator in the `op-src` submodule, building, registration, and testing through `torch_npu`, and repeatable benchmarks with NPU and simulator profiling. The initial planning scope was superseded by the user's explicit request to implement. The framework, private demo repository, dedicated container, and profiling pipeline are now implemented.

## Notes

- Tracker: local Markdown following `issue-tracker-local.md` from the `setup-matt-pocock-skills` skill. The map is an index; questions and answers live in separate tickets. `/setup-matt-pocock-skills` can configure the tracker later; it is not a prerequisite for this map.
- Skills: `wayfinder`, `grilling`, `domain-modeling`; `research` for research tickets.
- User agreements from 2026-10-22: plan before implementation; no existing operator sources; a simple demo operator; a separate `s84huawei-ops/ascend-op-demo` repository as `op-src` (created during implementation).
- Local editing → save in Git → fetch identical versions on the server → build and run the experiment in the container. Results remain on the server and can be downloaded.
- Image: `quay.io/ascend/vllm-ascend:v0.23.0rc1`. Starting point: the existing server-side `docker_run.sh`. Server: `ssh huawei-blue-npu`, workspace `/home/s84435910/docker_workspace`.
- One NPU per experiment, selected in the configuration; initial proposal: device 6. Check availability before starting; by default, exit with a clear message if busy. A busy device fails immediately with retained diagnostics.
- Directories: `op-src`, `docs`, `scripts/bash`, `scripts/python`, `configs`, `runs`. The correction from `pyhon` to `python` was accepted.
- Separate measurements of total benchmark time and NPU operator invocations, plus raw and processed profiler artifacts. The profiling backend must be configurable.
- Experiment: `runs/<MMDD-HHMM>-<name>`, default `name=exp`, a configuration copy, logs, results, and reproduction metadata. Atomic directory creation refuses collisions without overwriting earlier results.
- Environment evidence: [Server inventory](../../docs/environment-inventory.md). Glossary: [Project terminology](../../docs/GLOSSARY.md).
- Read open tickets from the `issues/` directory; `Blocked by` defines dependencies, while `Status: claimed` and `Assignee` reserve work. Take tickets in numerical order among open, unassigned tickets with no unresolved blockers.
- Historical planning note: research reports were initially local. Publication on the proposed `research/ascend-torch-npu-integration` and `research/ascend-profiling-backends` branches did not occur: automatic approval review rejected pushing environment data without explicit approval for the specific files and destination. Publication approval is separate from research completion and further planning. The user subsequently committed and published the English planning documents on main. This earlier publication restriction did not prevent authorized implementation.

## Decisions and implementation evidence

<!-- Index of closed tickets only; detailed answers live in the tickets. -->

- [How should we build an Ascend C demo invoked through torch_npu?](issues/01-torch-npu-integration.md): research recommends AddCustom/ACLNN and a standalone adapter; ABI compatibility and execution require a deployment test.
- [What results and limitations do msprof and msprof op simulator have?](issues/02-profiling-backends.md): documentation confirms the simulator's Python path; the report separates measurements and describes the required validations.

- [Operator versions](issues/03-operator-versioning.md): private independent repository, immutable parent pin, clean-source checks, explicit switching, and Git/SSH-bundle transfer.
- [Container contract](issues/04-container-contract.md): pinned image, physical NPU6 mapped to logical npu:0, isolated installation, workspace locks and occupancy checks.
- [Experiment contract](issues/05-experiment-contract.md): strict JSON, Warsaw timestamps, retained provenance and packages, separate timing scopes, explicit profiler statuses.
- [Acceptance](issues/06-rollout-and-acceptance.md): tests and evidence are recorded in [validation](../../docs/validation.md).

## Remaining extension work

- Extending adapters to complex operators: multiple inputs/outputs, dynamic shapes, different types, tolerances, and data preparation. The scope will become clearer after choosing the demo and adapter contract.
- Cross-experiment comparisons and report processing: available columns and metrics depend on actual profiler exports. Do not promise identical device and simulator results.
- Python simulator teardown crashes in the installed stack. Binary simulation passes and is the default; the Python entry remains a diagnostic option, explicitly reported as failed when it crashes.

## Out of scope

- Optimizing a specific target operator, integrating with vLLM, and multi-device benchmarks.
- Modifying other users' existing containers, processes, or host driver/CANN installations.
