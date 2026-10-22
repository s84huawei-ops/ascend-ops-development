# Ascend operator development framework — decision map

Label: wayfinder:map

## Destination

An agreed implementation plan for the framework in `ascend-ops-development`: a versioned demo operator in the `op-src` submodule, building, registration, and testing through `torch_npu`, and repeatable benchmarks with NPU and simulator profiling. This session creates the map and plan without implementing the framework or creating a container.

## Notes

- Tracker: local Markdown following `issue-tracker-local.md` from the `setup-matt-pocock-skills` skill. The map is an index; questions and answers live in separate tickets. `/setup-matt-pocock-skills` can configure the tracker later; it is not a prerequisite for this map.
- Skills: `wayfinder`, `grilling`, `domain-modeling`; `research` for research tickets.
- User agreements from 2026-10-22: plan before implementation; no existing operator sources; a simple demo operator; a separate `s84huawei-ops/ascend-op-demo` repository as `op-src` (currently only planned).
- Local editing → save in Git → fetch identical versions on the server → build and run the experiment in the container. Results remain on the server and can be downloaded.
- Image: `quay.io/ascend/vllm-ascend:v0.23.0rc1`. Starting point: the existing server-side `docker_run.sh`. Server: `ssh huawei-blue-npu`, workspace `/home/s84435910/docker_workspace`.
- One NPU per experiment, selected in the configuration; initial proposal: device 6. Check availability before starting; by default, exit with a clear message if busy. Optional waiting remains to be settled in the container ticket.
- Directories: `op-src`, `docs`, `scripts/bash`, `scripts/python`, `configs`, `runs`. The correction from `pyhon` to `python` was accepted.
- Separate measurements of total benchmark time and NPU operator invocations, plus raw and processed profiler artifacts. The profiling backend must be configurable.
- Experiment: `runs/<MMDD-HHMM>-<name>`, default `name=exp`, a configuration copy, logs, results, and reproduction metadata. Handling name collisions remains an open decision.
- Environment evidence: [Server inventory](../../docs/environment-inventory.md). Glossary: [Project terminology](../../docs/GLOSSARY.md).
- Read open tickets from the `issues/` directory; `Blocked by` defines dependencies, while `Status: claimed` and `Assignee` reserve work. Take tickets in numerical order among open, unassigned tickets with no unresolved blockers.
- Research reports are local. Publication on the proposed `research/ascend-torch-npu-integration` and `research/ascend-profiling-backends` branches did not occur: automatic approval review rejected pushing environment data without explicit approval for the specific files and destination. Publication approval is separate from research completion and further planning. No PR or commits were created in the working repository.

## Decisions so far

<!-- Index of closed tickets only; detailed answers live in the tickets. -->

- [How should we build an Ascend C demo invoked through torch_npu?](issues/01-torch-npu-integration.md): research recommends AddCustom/ACLNN and a standalone adapter; ABI compatibility and execution require a deployment test.
- [What results and limitations do msprof and msprof op simulator have?](issues/02-profiling-backends.md): documentation confirms the simulator's Python path; the report separates measurements and describes the required validations.

## Not yet specified

- Extending adapters to complex operators: multiple inputs/outputs, dynamic shapes, different types, tolerances, and data preparation. The scope will become clearer after choosing the demo and adapter contract.
- Cross-experiment comparisons and report processing: available columns and metrics depend on actual profiler exports. Do not promise identical device and simulator results.
- Actual limitations of a fresh container and the simulator: the read-only inventory does not establish successful compilation, registration, or profiling. Undocumented discrepancies will require separate tickets after initial implementation.

## Out of scope

- Implementation, creating the operator repository, changing the remote `docker_run.sh`, and starting containers or experiments during this session.
- Optimizing a specific target operator, integrating with vLLM, and multi-device benchmarks.
- Modifying other users' existing containers, processes, or host driver/CANN installations.
