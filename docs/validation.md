# Server validation

Validated on 2026-10-22 in the dedicated `ascend-ops-dev-npu6` container, exposing
physical Ascend910B2 device 6 as logical `npu:0`. Other users' containers and
host installations were not modified.

## Stack and operator versions

- Image: `quay.io/ascend/vllm-ascend:v0.23.0rc1`, pinned to digest
  `sha256:9fd154211d2faee7735cfe40785f863603e42ccd53bfb4bca82dc98c9446ea0a`.
- Python 3.12.13, torch 2.10.0+cpu, torch-npu 2.10.0.post2, CANN 9.0.1.
- Private operator repository: `s84huawei-ops/ascend-op-demo`.
- `v0.1.0`: `c353c260ffc89c29928702cf678c854cb74fc132`, 1024-element tiles.
- `v0.1.1`: `5de8044` (full SHA retained in each run's sources.json), 2048-element tiles.

Both versions expose the same FP16 addition API through the compiled PyTorch
adapter and generated ACLNN operator package. The tile variants demonstrate
version selection and cache separation; these checks are not an optimization study.

## Experiment evidence

Directories below are relative to the server framework repository. Each retains
the configurations, status, logs, source snapshots, package checksums, correctness
results, benchmark measurements, and requested raw/processed profiler reports.

| Run | Operator | Result |
| --- | --- | --- |
| `runs/1022-1313-verify-v010` | v0.1.0 | Fresh build, torch_npu correctness, benchmark, msprof, msprof op, binary simulator: all passed |
| `runs/1022-1321-verify-v011` | v0.1.1 | Independent fresh build and the same complete pipeline: all passed |
| `runs/1022-1326-reproduce-v010` | v0.1.0, restored with switch_operator.sh | Reused the original version's cache; correctness and benchmark passed |

These implementation-time runs explicitly set `allow_dirty=true` to retain
framework changes while developing. They still capture actual files, hashes,
diffs, and both repository revisions. Normal example configurations require
clean committed sources.

Both full runs checked `[8, 2048]`, `[1]`, `[17]`, and `[16387]` against a CPU
reference, including partial tiles and unaligned transfers. Maximum absolute
error was zero. Every successful profiling backend produced numeric timing/cycle
CSV rows identifying AddCustom; a successful tool exit alone was not accepted.

For v0.1.1, compilation/installation took about 131.49 seconds. The unprofiled
benchmark process took 14.33 seconds, including startup/imports, preparation and
warmup. Its measured loop of 20 calls after 3 warmups took 4.013 milliseconds;
the median stream region was 0.12461 milliseconds. Stream regions include
dispatch gaps and must not be reported as kernel-only time. The default example
uses 5 warmups and 50 iterations; validation overrides are preserved in the runs.

## Runner checks

All 9 CPU tests passed inside the target Linux container. They cover configuration
validation/defaults, busy-device failure artifacts, interruption status,
numeric profiler report identification, directory collision protection, invalid
configuration artifacts, actual filesystem lock exclusion, and workload timeout
termination with retained logs. On Windows, 7 tests passed and the two Linux
process/lock checks were skipped. Bash syntax was checked on the server.

The busy-device test injects an occupancy error; it does not start a competing
NPU workload. Checks against the actual device occurred before experiment stages.
Filesystem locking coordinates this workspace only and cannot reserve hardware
against unrelated users.

## Known limits

The optional Python simulator entry produced kernel data but its child crashed
during teardown with status 139 in this stack. Earlier diagnostic runs, including
`runs/1022-1301-verify-all-3`, preserve that failure. It is reported as failed even
when the msprof parent returns zero.

The default binary simulator entry passes. It uses the same compiled kernel
binary, an explicit operator-specific launch ABI, and CPU-generated FP16 data.
It profiles the kernel model outside PyTorch. Separate physical-device tests
establish torch_npu adapter correctness. Simulation uses `[16]` by default;
its timings are not benchmark results for `[8, 2048]`.

Selectable vendor metric groups are forwarded to the chosen profiler; only its
default metric groups were validated. Missing tools/modes are reported as
unsupported rather than successful, but intentionally removing the installed
toolchain was not part of validation.

See [usage](usage.md) for running, switching versions, transferring committed
sources through Git/SSH bundles, and interpreting the retained results.
