# Ascend C demo and torch_npu integration

Research date: 2026-10-22. Scope: documentation and planning, without compilation, installation, or NPU invocation. Server state comes from the [inventory](../environment-inventory.md); inspecting an existing container does not test a fresh image.

## Research recommendation

I propose an **AddCustom demo through ACLNN and a standalone PyTorch C++ extension**, packaged separately as a CANN `.run` operator package and a Python `.whl` adapter. This will validate the full required chain: sources → build → installation → registration → invocation through torch_npu. This is a planning recommendation, not a decision already agreed with the user or a confirmed execution test.

The official Ascend `1_add_frameworklaunch` example includes the AddCustom project, ACLNN invocation, and a C++ extension; its script generates the project through msOpGen, builds `custom_opp_<os>_<arch>.run`, and installs it in OPP. [Ascend samples](https://gitee.com/ascend/samples/blob/master/operator/ascendc/0_introduction/1_add_frameworklaunch/README.md)

## What the sources confirm

| Topic | Finding |
| --- | --- |
| Required image versions | The **v0.23.0rc1** documentation specifies CANN and NNAL `9.0.1`, torch `2.10.0`, torch-npu `2.10.0.post2`, and Python `>=3.10,<3.13`; it shows the exact `quay.io/ascend/vllm-ascend:v0.23.0rc1` image for A2. This is the appropriate reference for our container. [Quickstart for this release](https://docs.vllm.ai/projects/ascend/en/v0.23.0rc1/quick_start.html#requirements) |
| Standalone adapter | The official C++ extensions guide shows the `TORCH_LIBRARY(myops, ...)` schema, output allocation, and `EXEC_NPU_CMD(aclnnAddCustom, self, other, result)`. This does not require adding the operator to the full torch_npu source tree. [TorchNPU guide: C++ adaptation](https://www.hiascend.com/document/detail/zh/Pytorch/730/modthirdparty/modparts/thirdpart_0005.html) |
| Adapter build | The C++ extension example builds a wheel through `python3 setup.py build bdist_wheel`, installs it, and runs eager tests; it presents a `torch.compile` test separately. The example predates our stack, so it is an integration pattern, not proof of binary compatibility with 2.10. [Official invocation example](https://www.hiascend.com/document/detail/zh/Pytorch/720/ptmoddevg/Frameworkfeatures/featuresguide_00027.html) |
| Alternative | CANN 9.0.X documents direct kernel launching from PyTorch through torch.library or Pybind. The example gets the current stream through `c10_npu::getCurrentNPUStream()`. Pybind alone does not provide dispatcher/graph tracing integration. [CANN: PyTorch](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_0101.html) |
| NPU registration | The official example registers the schema and implementation through `TORCH_LIBRARY_IMPL(..., PrivateUse1, ...)`, and Python invokes `torch.ops.<namespace>.<name>`. The kernel launch variant is packaged as a wheel. [TorchNPU: kernel and registration](https://www.hiascend.com/document/detail/zh/Pytorch/730/ptmoddevg/Frameworkfeatures/docs/zh/framework_feature_guide_pytorch/adaptation_description_single.md) |
| Package and toolchain | CANN 9.0.X provides CMake `find_package(ASC ...)`, `op_host`, `op_kernel`, and `npu_op_package`; the `RUN` type creates an installation package. `ENABLE_BINARY_PACKAGE` controls inclusion of kernel binaries. CMake presets require CMake >=3.19. [CANN 9.0.X: compilation](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_0068.html) |

The description on the specific package's PyPI page still includes an installation command for `2.10.0rc2` and a CANN 8.5.0 table. Do not infer compatibility of our stack from it; the versioned vLLM Ascend quickstart is more precise here. [Package description for 2.10.0.post2](https://pypi.org/project/torch-npu/2.10.0.post2/)

## Proposed contract and repository boundary

In `ascend-op-demo`, included as `op-src`, I propose storing **everything specific to the operator**: `op_host`, `op_kernel`, tiling, CMake/presets, the C++ adapter, Python package `ascend_op_demo`, schema definition, ACLNN helper, reference tests, and a manifest describing build commands. If we reuse example code, preserve its licenses, attribute its source, and record the exact upstream commit; do not copy a moving `master` on every experiment.

The framework should handle container, Git/submodule, build/installation, test, benchmark, and profiler orchestration, configurations, and artifact collection. It should invoke the operator contract without hardcoding AddCustom in profiling logic. Preserve both packages, or their unambiguous locations and SHA256 checksums, the operator commit, and full logs in the experiment directory.

Proposed API: import `torch`, `torch_npu`, and `ascend_op_demo`, then call `torch.ops.ascend_demo.add(x, y) -> Tensor`. Initial version: eager forward, two contiguous FP16 tensors on the same device with the same shape, an output of the same shape and dtype, and no broadcasting. Validation should reject unsupported cases before entering the kernel. Start with a fixed `[8, 2048]` case, then add small, large, and non-divisible sizes after implementing correct tail handling. Autograd and torch.compile can follow in later stages.

I propose installing `.run` into a separate prefix for each build and performing each test in a new process. Documentation describes `--install-path=<path>` and activation through `source <path>/vendors/<vendor_name>/bin/set_env.bash`, setting `ASCEND_CUSTOM_OPP_PATH`. Check the exact behavior of the 9.0.1 installer through its `--help`. [Official instructions for installation into a specified directory](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/opdevg/Ascendcopdevg/atlas_ascendc_10_00004.html)

## ABI and tests required before deployment

PyTorch provides compiler compatibility checks in `torch.utils.cpp_extension`. The `NpuExtension` helper adds torch_npu headers and libraries, but the inspected `master` code does not establish an identical implementation in our wheel. [PyTorch 2.10 source](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/utils/cpp_extension.py), [Ascend NpuExtension source](https://github.com/Ascend/pytorch/blob/master/torch_npu/utils/cpp_extension.py)

I propose not forcing `_GLIBCXX_USE_CXX11_ABI=0` or `1` from historical examples. Match the adapter ABI to the actual installed torch and separately to CANN; successful compilation and library loading are the evidence, rather than passing a compiler version check alone.

Planned execution gates that this research **does not satisfy**:

1. In a fresh container, record actual versions, CANN paths, host compiler and bisheng, CANN/torch_npu headers, `torch.__config__.show()`, and torch ABI. Verify toolkit/development tool completeness and driver/firmware compatibility with CANN 9.0.1.
2. Set the target SoC to `Ascend910B2`; confirm that the installed toolchain and pinned example accept it. Build inside the image on the aarch64 server rather than transferring Windows binaries.
3. Build `.run` with binaries, install into an isolated prefix, and inspect generated ACLNN, tiling, and binaries. Build the adapter wheel against the actual installed torch/torch_npu without automatically changing those packages.
4. In a new process, verify imports, schema, NPU dispatch, and actual execution of the custom AddCustom. Confirm kernel identity in the profile; correct addition alone does not prove use of the custom implementation.
5. With a fixed seed, compare the result transferred to CPU with `torch.add` on the same CPU inputs; also check output dtype, shape, and device. Proposed FP16 tolerances: `rtol=1e-3`, `atol=1e-5`, explicitly recorded in the configuration; these match the default `torch.testing.assert_close` tolerances in 2.10. [PyTorch 2.10 testing](https://docs.pytorch.org/docs/2.10/testing.html#torch.testing.assert_close)

The largest unknown is porting the specific ACLNN example/helper to torch_npu 2.10.0.post2 and CANN 9.0.1. If adaptation is needed, maintain it in `ascend-op-demo` and record its scope. The kernel launch alternative remains a separate option; do not implement both paths in the first milestone.
