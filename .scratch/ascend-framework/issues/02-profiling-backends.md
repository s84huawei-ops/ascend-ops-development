# What results and limitations do msprof and msprof op simulator have?

Type: research
Label: wayfinder:research
Status: resolved
Assignee: research_profiling
Parent: ../map.md
Blocked by: none

## Question

How should we design two profiling backends, a physical NPU with msprof and simulation through msprof op simulator, so they work with a custom operator invoked through torch_npu and do not mix benchmark time with profiler overhead?

Check official commands, CANN/SoC/kernel file and symbol requirements, how to trigger the custom operator, reports/exports, and meaningful metrics. Verify whether the simulator can wrap a Python script or needs a different harness; do not assume the modes are interchangeable. Missing support should produce an explicit status rather than a false success. Provide a test plan to establish actual operation, without performing profiling in this session.

## Evidence to produce

One file, `docs/research/profiling-backends.md`, with links to primary sources and a recommendation. Research branch: `research/ascend-profiling-backends`, without a PR or merge. Record the answer and a link to the file under `## Answer` after research.


## Answer

Research completed: [docs/research/profiling-backends.md](../../../docs/research/profiling-backends.md).

Official sources confirm that `msprof op simulator` can wrap a Python/PyTorch script; a separate kernel harness is not a general requirement. The specific torch_npu adapter path and installed tool capabilities require validation in a fresh container. The report proposes separate baseline, msprof, and simulator runs, separate timing metrics, unsupported/blocked/failed statuses, and acceptance tests. It describes older documentation's warning about logical device 0 and the lack of grounds for automatically transferring physical NPU6 into the simulator configuration. Integration commands are marked as sketches; no profiling, compilation, or NPU initialization was performed.

The report was saved in the working tree, without checkout/commit/push in this shared repository with no commits.

## Comments

2026-10-22: research completed locally. The attempt to publish the report and inventory on the proposed `research/ascend-profiling-backends` branch was rejected by automatic approval review because of environment information and the lack of explicit authorization for this payload/destination. The command was not executed. Publication awaits user approval; it is not required to close the substantive research.
