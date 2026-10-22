# What configuration, adapter, and result contract defines one experiment?

Type: grilling
Label: wayfinder:grilling
Status: open
Assignee: unassigned
Parent: ../map.md
Blocked by: 01, 02, 03, 04

## Question

How should a single experiment be described and reproduced, from compilation to saving results, while keeping support for future operators straightforward?

Settle the configuration format and validation, operator adapter interface, build/register/correctness/benchmark/profile stages, build caching and rebuilding, warmup/iterations, NPU synchronization, measurement separation, profiler backend, exit codes, and statuses for unsupported modes.

Following research: account for separate performance artifacts and simulator artifacts with symbols, preserving optimization settings and checksums of `.run`/`.whl` packages. NPU events measure a region on a stream; do not automatically call this the kernel's execution time. Reports must verify the identity of the custom kernel.

Preserve `runs/<MMDD-HHMM>-<name>` with `name=exp` as the default. Resolve same-minute collisions and names unsafe for paths; never overwrite an earlier run. Settle storage of original and resolved configurations, framework and submodule SHAs, dirty status/diff, environment, tool versions, commands, stage logs, correctness results, timings, and raw profiler exports. Failed runs must also have a complete experiment directory.
