# What implementation sequence and tests will demonstrate that the framework works?

Type: grilling
Label: wayfinder:grilling
Status: open
Assignee: unassigned
Parent: ../map.md
Blocked by: 05

## Question

Once the contracts are settled, how should implementation be divided into verifiable steps, and what exactly must pass on the server before the framework is considered ready?

Plan creation of the demo repository and submodule, fresh container setup, compilation and registration, value tests through torch_npu, a benchmark without profiling, an msprof profile on the NPU, a simulator profile, switching the operator revision, and reproducing an experiment. Cover invalid configuration, a busy NPU, a missing tool/mode, and interrupted execution. Identify stages that were actually verified; the presence of a binary or files does not prove successful profiling.

The outcome is an agreed plan and acceptance criteria; implementation begins as a separate scope of work.
