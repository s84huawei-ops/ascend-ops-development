# Lightning Indexer follow-up, 2026-11-09

Development branches: `li-op-torch-npu` in this repository and
`li-op/torch-npu` in `op-src`. The baseline is operator commit `6bd8092`,
which already includes BF16 weight caching and paired top-k merges.

## Source changes

`arch22/lightning_indexer_paired_merge.h` tracks the retained element count
per query row. Early merges no longer process a full top-2048 list of padding.
Copies grow from 512/1024 elements to at most 2048; the untouched output tail
keeps its initialized negative-infinity values and invalid indices. The
original ordering of incoming, pending, and retained lists preserves ties.
The count resets together with pending state when the query tile changes.

The change applies to the existing paired-merge fast path. Multiplication
and reduction order, host tiling, global workspace, TPipe buffer allocation,
and arch35 are unchanged. The implementation does not require extra UB.

A second experiment queued the next score group into the alternate buffer
before scaling the current group. It passed correctness but was not retained:
direct comparisons showed mixed per-round wins and losses. Its patch and
measurements remain under `prefetch/` for reproducibility.

## Standalone results

Ascend910B2, NPU 7, CANN 9.0.0; batch 1, 16,384 queries and keys, 64 heads,
BF16 inputs/weights and top-2048. Each experiment uses three independent
processes with seeds 20261110–20261112. Each process uses 12 warmups,
8 retained samples per lane per round, and two rounds with alternating lane
order. Both lanes submit source-built ELFs through the same direct adapter,
workspace and stream. Timing uses NPU events and excludes host setup.

| Experiment | Previous implementation | Candidate | Latency reduction |
| --- | ---: | ---: | ---: |
| Retained count only (selected) | 14.9465 ms | 14.0706 ms | 5.86% |
| Retained count and score prefetch (not selected) | 15.0543 ms | 14.1186 ms | 6.22% |

These are medians of the three process medians. The rows come from separate
experiments; compare paired lanes within a row rather than absolute times
between rows. The selected candidate run medians are 14.2377, 13.9974, and
14.0706 ms. Reports and exact samples are under
[`benchmarks/li-torch-npu-20261109`](benchmarks/li-torch-npu-20261109).

The separate `prefetch/versus-retained*.json` reports directly compare the
two candidates, using 12 warmups, 16 samples and four alternating rounds.
Here `baseline` means retained-count-only, rather than commit `6bd8092`.
The process-level speedups for prefetch were 1.0085x, 1.0036x, and 0.9990x;
this small, inconsistent benefit did not justify keeping the extra scheduling
logic. `prefetch/experiment.patch` reproduces that experiment atop the selected
implementation at `op-src` commit `5aeb864`.

## Transformer block check

The selected ELF also passed exact index and hidden-state comparisons in both
initialized-weight blocks of the pinned study benchmark on NPU 7.

| Measurement | Native operator | Selected source kernel |
| --- | ---: | ---: |
| Indexer inside one block, profiler median | 17.9613 ms | 13.1462 ms |
| One complete block, NPU events | 175.1211 ms | 171.2072 ms |
| Two complete blocks, NPU events | 351.8776 ms | 344.1254 ms |

This comparison uses the native operator as its control, rather than the
previous optimized commit used in the standalone experiment. The block uses
different initialized weights and profiler timing; its absolute indexer times
should not be compared directly with the standalone times. These are block
measurements, not full-model throughput. Raw results are in `layer-results.json`.

## Correctness and scope

The selected candidate passed 15 exact index/value cases against the installed
native operator. Outputs are poisoned before each correctness launch and
workspace is cleared between checked lanes. The candidate is checked again
after timing. Cases cover odd tails, split-core boundaries, zero-weight ties,
BSND, TND, two TND batches, paged KV, small queries, 32 heads, FP16, FP32
weights, and top-1024/top-2048/top-4096. The direct candidate comparison also
checks retained-count-only against all 15 cases.

The stored build recipes record compiler options and source hashes. The
final identity is `identity.json`; its hashes match all 16 delivered kernel
source files. `retained-recipe.json` describes its build. The `prefetch/`
identity and recipe describe the rejected experiment.

This follow-up does not claim a new installed ACLNN/vLLM package validation,
910C execution, or full-model throughput result. Rebuild the package and
Torch adapter using `scripts/bash/build_operator_fixed.sh` to activate this
source in deployment; see [the integration guide](STAN_LI_OPTIMIZATION.md).

## Reproduce

Use `scripts/python/build_li_kernel.py` with the installed CANN metadata
listed in `retained-recipe.json`, a fresh output directory and
`--soc Ascend910B2`. Build baseline commit `6bd8092` separately, or use its
hash-verified identity from the earlier paired-merge study.

With the study's `exercises/07_common` adapter available, run:

```bash
export TASK_QUEUE_ENABLE=0
LD_PRELOAD="$COMMON/build/libstudy_direct.so" \
python3 scripts/python/validate_li_source.py \
  --baseline "$BASELINE/identity.json" --candidate "$CANDIDATE/identity.json" \
  --study-common "$COMMON" --device 7 --seed 20261110 \
  --warmups 12 --repeats 8 --rounds 2 --matrix --output "$RESULT/run-1.json"
```

Select an idle device first. Repeat in fresh processes with seeds 20261111
and 20261112 and new result paths. Paths above are parameters for the local
environment; exact server paths are recorded in the reports and recipes.
