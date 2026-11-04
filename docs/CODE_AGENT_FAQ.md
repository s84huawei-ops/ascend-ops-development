## Why my operator does not compile into the previous one?
**Root cause found**

`build_operator_fixed.sh` preserved `csrc/build` (the Ninja cache) for incremental builds. But the kernel-compilation rules emitted by `cmake/symbol.cmake` are `CUSTOM_COMMAND`s with **only order-only deps and no explicit inputs**, so Ninja treats a kernel as up-to-date as long as its `*.done` marker exists. The markers and `.o` binaries were from **Jul 19**, so your **Nov 3** `op_kernel` edits were silently skipped — the runtime kept using the original kernel, which is why your code seemed to "become a new custom operator" (it was never actually compiled in).

**Changes made**

1. `scripts/bash/build_operator_fixed.sh` — before invoking `build_aclnn.sh`, it now removes the stale `*.done` kernel markers, the `bin/<op>/` outputs, and the `src_copy.done` marker for the target op under `csrc/build/binary/<soc>/`, forcing Ninja to recompile the device kernels.
2. `op-src/lightning_indexer/op_kernel/arch22/lightning_indexer_ab.h` — fixed a compile error that blocked the build (3 occurrences of `AscendC::GetTPipePtr() → GetTPipePtr()`, which is a global function, not in the AscendC namespace).

Note: `ascend910_93` is a **C220 AICore**, so it compiles **arch22** — exactly the arch your A/B experiment targets.
