# What configuration, adapter, and result contract defines one experiment?

Type: grilling
Label: wayfinder:grilling
Status: resolved
Assignee: root
Parent: ../map.md
Blocked by: 01, 02, 03, 04

## Question

How should a single experiment be described and reproduced, from compilation to saving results, while keeping support for future operators straightforward?

Settle the configuration format and validation, operator adapter interface, build/register/correctness/benchmark/profile stages, build caching and rebuilding, warmup/iterations, NPU synchronization, measurement separation, profiler backend, exit codes, and statuses for unsupported modes.

Following research: account for separate performance artifacts and simulator artifacts with symbols, preserving optimization settings and checksums of `.run`/`.whl` packages. NPU events measure a region on a stream; do not automatically call this the kernel's execution time. Reports must verify the identity of the custom kernel.

Preserve `runs/<MMDD-HHMM>-<name>` with `name=exp` as the default. Resolve same-minute collisions and names unsafe for paths; never overwrite an earlier run. Settle storage of original and resolved configurations, framework and submodule SHAs, dirty status/diff, environment, tool versions, commands, stage logs, correctness results, timings, and raw profiler exports. Failed runs must also have a complete experiment directory.

## Implemented answer

Strict JSON configurations select operator expectations, shapes, seed, tolerances, warmup/iterations, build options, profiler modes and metrics. Atomic `runs/MMDD-HHMM-name` creation refuses collisions. Default name is exp. Each run retains original/resolved config, source snapshots/SHAs/diffs/hashes, environment capabilities, package checksums, binaries, logs and results. The runner builds and activates the independent operator, validates CPU-reference results, benchmarks without profiling, and launches each profiler separately. Complete process time, synchronized loop time, stream event regions and vendor metrics remain distinct. Numeric timing/cycle reports must identify the custom kernel. Required profiler failures fail the run; optional failures produce partial status. Binary simulation uses the operator's explicit ABI/data hook; the diagnostic Python mode is not presented as passing when it crashes.
