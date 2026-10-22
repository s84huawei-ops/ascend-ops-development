# Ascend operator development framework

Develop a versioned Ascend C operator through `torch_npu`, with containerized
builds, correctness tests, benchmarks, and real-device/simulator profiles.

Start with the [usage guide](docs/usage.md). The demo is an independent private
repository at [ascend-op-demo](https://github.com/s84huawei-ops/ascend-op-demo),
pinned as the `op-src` submodule. Access to that repository is required when cloning.

See [server validation](docs/validation.md) for tested versions, experiment evidence,
and the limitation of the optional Python simulator entry.

```bash
git submodule update --init --recursive
bash scripts/bash/run.sh configs/demo.json
bash scripts/bash/run.sh configs/msprof.json
bash scripts/bash/run.sh configs/simulator.json
```

Run those Bash commands on the NPU server. All operator builds and workloads run
inside `quay.io/ascend/vllm-ascend:v0.23.0rc1`, pinned to its verified digest.
Each experiment produces `runs/MMDD-HHMM-name/` with configuration, source
snapshots, logs, build packages, measurements, and profiler output.
