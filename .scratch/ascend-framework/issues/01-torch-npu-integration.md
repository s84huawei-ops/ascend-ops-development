# How should we build an Ascend C demo invoked through torch_npu?

Type: research
Label: wayfinder:research
Status: resolved
Assignee: /root/research_torch_npu
Parent: ../map.md
Blocked by: none

## Question

Which officially supported simple demo operator and PyTorch/torch_npu integration pattern would best validate building, installation/registration, and invocation of a custom Ascend C kernel on Ascend 910B2 in the required image?

Identify a minimal example (preferably Add), the file boundary between the framework and future `ascend-op-demo`, package format, invocation interface, CANN/toolchain/ABI requirements, and testing against a PyTorch reference. Distinguish source-backed facts from proposals and items requiring tests in a fresh container. Do not implement or execute the kernel.

## Evidence to produce

One file, `docs/research/torch-npu-integration.md`, with links to primary sources and a recommendation. Research branch: `research/ascend-torch-npu-integration`, without a PR or merge. Record the answer and a link to the file under `## Answer` after research.

## Answer

Research recommendation: AddCustom in Ascend C through ACLNN and a standalone PyTorch C++ extension; a separate CANN `.run` package and `.whl` adapter, with a proposed call through `torch.ops.ascend_demo.add`. All operator sources, tiling, and the adapter belong in `ascend-op-demo`; the framework handles orchestration. The official quickstart for exactly v0.23.0rc1 confirms CANN 9.0.1 / torch 2.10.0 / torch-npu 2.10.0.post2. The adapter example and ABI require a later test in a fresh container. No implementation or NPU invocations were performed.

Report: [torch-npu-integration.md](../../../docs/research/torch-npu-integration.md). The researcher did not perform checkout/commit/push in the shared repository.

## Comments

2026-10-22: research completed locally. The attempt to publish the report and inventory on the proposed `research/ascend-torch-npu-integration` branch was rejected by automatic approval review because of environment information and the lack of explicit authorization for this payload/destination. The command was not executed. Publication awaits user approval; it is not required to close the substantive research.
