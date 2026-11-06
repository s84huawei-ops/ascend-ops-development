# Lightning Indexer source optimization

The delivery branch is `stan/li-prefill16k-opt` in both `ascend-ops-development`
and its `op-src` submodule (`ascend-op-demo`). The deliverable is source code in
`op-src/lightning_indexer`, built and tested in `stan_ascend_study` on Ascend910B2.
Ascend 910C execution is reserved for the deployment server.

## Results

The standalone test compares the raw operator at commit
`1a2983cb8b5d3ea07838adcc58a7f9b05ea0d7b4` with source-built candidates. Each
experiment uses three independent processes/seeds, 12 warmups, 8 retained
samples per round, and 2 alternating baseline/candidate rounds. Timings are
NPU events around identical direct ELF submission paths, excluding host setup.
Reported aggregate times are medians of the three process-level medians.

| Source variant | Baseline | Candidate | Speedup | Latency reduction |
| --- | --- | --- | --- | --- |
| BF16 weight cache | 17.94 ms | 15.57 ms | 1.15x | 13.2% |
| Weight cache and paired top-k merges | 17.95 ms | 14.91 ms | 1.20x | 16.9% |

The final candidate's run medians were 14.875, 14.914, and 15.010 ms. The
baseline medians were 17.940, 17.951, and 17.957 ms. See
[`summary.json`](benchmarks/li-prefill16k-20261106/summary.json) for full precision.

The pinned initialized-weight block benchmark also passed exact indices and
hidden-state checks for both blocks:

| Measurement in the candidate run | Native | Source candidate |
| --- | --- | --- |
| Indexer, profiler median in one block | 17.97 ms | 14.45 ms |
| Complete one-block execution, NPU events | 176.81 ms | 173.50 ms |
| Complete two-block execution, NPU events | 351.85 ms | 345.71 ms |

A separate raw-source block run matched the native indexer at approximately
17.97 ms. The standalone and block measurements use different input fixtures
and timing methods; compare the paired lanes within each measurement.
The specialized study `row_ring4` remains a separate reference, previously
measured at approximately 11.63 ms. This delivery improves the general operator
source while retaining its existing host tiling and hardware-derived core count.

## Changes and correctness

`arch22/lightning_indexer_weight_cache.h` caches BF16 weights in the existing
512-byte broadcast buffer. `arch22/lightning_indexer_paired_merge.h` combines
two sorted key tiles with the retained top-k, adding 16 KiB of UB. The fast path
requires 64 heads, an eight-query tile, and a 512-key tile; pairing additionally
requires top-2048. Other shapes/dtypes keep the original calculation path.
Accumulation order, tie ordering, workspace sizes, and the arch35 path are preserved.

Validation completed on NPU 5 with CANN 9.0.0, Python 3.11.15,
PyTorch 2.10.0+cpu, and torch-npu 2.10.0:

- Thirteen exact index/value cases cover 16k prefill, odd tails, split-core
  boundaries, zero-weight ties, BSND, TND, paged KV, two TND batches, small
  queries, 32 heads, top-1024, FP16, and FP32 weights.
  The final repeat poisons indices/values before every checked launch, clears
  scratch between lanes, and checks the candidate again after timing. All
  thirteen cases and the poisoned-output ACLNN package check passed.
- Both initialized-weight blocks preserve exact indices and hidden states.
- The isolated ACLNN package was built/installed and matched a native reference
  produced in a separate process without custom-op activation.
- The vLLM Torch extension was rebuilt and its public tensor-only indexer
  matched the same isolated native reference.
- Missing explicit indexer activation is rejected, preventing an accidental
  measurement of the original implementation after wiring the adapter.
- The same kernel sources cross-compiled for `Ascend910_9391` with CANN 9.0.0.
  This is a compilation check, not a 910C correctness or performance result.

No full-model/UCM throughput result is claimed. The block test uses deterministic
initialized BF16 weights, batch 1, 16,384 tokens, 64 indexer heads, and top-2048.

## Pull the delivery

From the framework checkout, when creating the local delivery branch:

```bash
git fetch origin
git switch --track origin/stan/li-prefill16k-opt
git submodule update --init --recursive
```

For subsequent updates on that branch:

```bash
git pull --ff-only
git submodule update --init --recursive
```

The submodule needs access to the `ascend-op-demo` remote. The NPU development
server used Git bundles containing the same pushed commits because its private
submodule credentials were unavailable. The original server checkout and its
modified `builder.env` were preserved; development used the independent checkout
`/home/s84435910/docker_workspace/stan-li-prefill16k`, mounted at
`/workspace/stan-li-prefill16k` inside the container.

## Build on the deployment server

Run inside the target container from the framework checkout root. Ensure
`op-src/builder.env` points `DST_DIR` at the local vLLM source tree. The example
uses the 910C package family and concrete `ascend910_9391` SKU; select the
concrete SKU supported by the deployment server and its CANN installation.

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export MAX_JOBS=8
bash scripts/bash/build_operator_fixed.sh \
  --soc ascend910_93 \
  --package-ops 'lightning_indexer;quant_lightning_indexer_metadata;sparse_attn_sharedkv_metadata' \
  --vendor stan \
  --torch-adapter --torch-soc ascend910_9391
```

For the tested 910B2 environment, use `--soc ascend910b` and
`--torch-soc ascend910b2`. Package families and concrete compiler targets are
different: `ascend910b` is valid for the package selector but selects an older
architecture in the legacy CANN CMake toolchain. The Torch wiring tool requires
a concrete target and rejects that ambiguous value.

The explicit package list includes this operator and the metadata dependencies
required by this vLLM version's CMake. It installs a separate `stan_transformer`
vendor package without replacing `_cann_ops_custom`. The default build list in
this vLLM selects `lightning_indexer_vllm`; the framework rejects that default
when it would omit the source operator being changed.

`--torch-adapter` updates vLLM's include and registration to use the source
operator while retaining its ten-argument, tensor-returning public schema. It
preserves backups and rebuilds the Torch extension using a saved setup snapshot
that skips the already completed ACLNN package build. The standalone
`wire_li_torch_adapter.py` helper can rebuild just this extension after copying
the current adapter into the configured vLLM operator directory.

## Activate and validate

Preserve vLLM's existing library search path. Activate just the indexer's vendor
registry and explicit API library in every benchmark/model worker environment:

```bash
LI_VENDOR_DIR=/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_stan/vendors/stan_transformer
export ASCEND_CUSTOM_OPP_PATH="$LI_VENDOR_DIR${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
export LIGHTNING_INDEXER_OPAPI_LIBRARY="$LI_VENDOR_DIR/op_api/lib/libcust_opapi.so"
export TASK_QUEUE_ENABLE=0
```

Adapt the vLLM root in `LI_VENDOR_DIR` to the deployment server. The adapter
routes only `aclnnLightningIndexer` and its workspace-size API to this absolute
library path. Other vLLM custom APIs continue using their original resolver.
Restart model workers after installing a new library.

Check NPU availability before choosing `--device`. Then validate both interfaces
against a native reference in a separate clean process:

```bash
python scripts/python/validate_li_package.py \
  --vendor-dir "$LI_VENDOR_DIR" --backend aclnn \
  --device 0 --sequence 16384 --seed 20261106 --output runs/li-aclnn-check

python scripts/python/validate_li_package.py \
  --vendor-dir "$LI_VENDOR_DIR" --backend vllm \
  --device 0 --sequence 16384 --seed 20261106 --output runs/li-vllm-check
```

Each output directory must be new. Success produces `custom.json` with
`exact_indices: true` and `complete: true`. Keep the same activation variables
when launching `ucm-bench`. The 910C deployment should be rebuilt from source;
the 910B2 ELF and shared libraries are development artifacts.

## Evidence and experiment files

Tracked JSON reports and profiler tables are in
[`benchmarks/li-prefill16k-20261106`](benchmarks/li-prefill16k-20261106/).
Complete build logs, generated sources, device ELFs, profiles, and validation
fixtures remain on the development host at
`/docker/li-reproduction/luka-20261106/ascendc-study/stan-li-prefill16k`.
The corresponding container path starts with
`/home/giraffe/ascendc-study/stan-li-prefill16k`.

`build_li_kernel.py` produces experimental device ELFs through an explicit CANN
recipe and records source hashes, compiler version, flags, and target.
`validate_li_source.py` compares two such ELFs through the study's independent
adapter. `benchmark_li_source_layer.py` changes only the accepted variant labels
of the hash-pinned study benchmark and supplies the source-built ELF.
These experiment helpers complement the installed-package and vLLM validation.
