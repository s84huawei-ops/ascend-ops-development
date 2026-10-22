# What implementation sequence and tests will demonstrate that the framework works?

Type: grilling
Label: wayfinder:grilling
Status: resolved
Assignee: root
Parent: ../map.md
Blocked by: 05

## Question

Once the contracts are settled, how should implementation be divided into verifiable steps, and what exactly must pass on the server before the framework is considered ready?

Plan creation of the demo repository and submodule, fresh container setup, compilation and registration, value tests through torch_npu, a benchmark without profiling, an msprof profile on the NPU, a simulator profile, switching the operator revision, and reproducing an experiment. Cover invalid configuration, a busy NPU, a missing tool/mode, and interrupted execution. Identify stages that were actually verified; the presence of a binary or files does not prove successful profiling.

The outcome is an agreed plan and acceptance criteria; implementation begins as a separate scope of work.

## Implemented answer

The user authorized implementation after the planning phase. The implementation sequence has been completed: independent demo sources and submodule; dedicated pinned container; custom ACLNN registration and torch_npu adapter; runner and retained artifacts; on-device and binary simulation profiles; two tagged source variants and version switching; usage and validation documentation. [Validation results](../../../docs/validation.md) record actual experiments and distinguish checks from limits. The optional Python simulator entry remains a documented stack limitation, while the default binary simulator provides the required working simulation backend.
