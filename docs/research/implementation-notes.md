# AddCustom implementation notes

Research date: 2026-10-22. Target: the aarch64 image `quay.io/ascend/vllm-ascend:v0.23.0rc1`, CANN 9.0.1, torch 2.10.0, torch_npu 2.10.0.post2, Ascend910B2. This note records source research; successful compilation and execution must be demonstrated separately in the target container.

## Recommended implementation route

Keep the planned ACLNN operator package (`.run`) plus a standalone TorchNPU adapter (`.whl`). Generate the build skeleton using the installed toolkit and keep the operator implementation under version control. Avoid rebuilding torch_npu merely to add this demo: the official standalone C++ extension example already demonstrates the wheel workflow. [Official C++ extension invocation instructions](https://www.hiascend.com/document/detail/zh/Pytorch/700/modthirdparty/modparts/thirdpart_0015.html)

If the installed operator API helper cannot be consumed as a public extension header, directly link the generated ACLNN library and wrap the two-phase API in a TorchNPU `OpCommand::RunOpApiV2` callback. This is an implementation recommendation combining the documented ACLNN API and the current TorchNPU queue pattern, not a pretested compatibility claim.

## Generate and compile the operator package

The official current msOpGen quick start uses:

```bash
msopgen gen -i AddCustom.json -c ai_core-ascend910B2 -lan cpp -out generated
```

The JSON declares the operator type, required `x`/`y` inputs and `z` output, `float16`, and `ND`. The generated kernel and tiling bodies are empty. Check `msopgen gen --help` in the target toolkit before adding the historical `-f aclnn` flag: template generations differ. The CLI's SoC spelling should be checked against the installed tool, rather than lowercasing configuration values globally. [Official toolchain quick start](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/devaids/optool/docs/en/quick_start/op_tool_quick_start.md)

Minimal maintained source files are the operator specification, `op_host/add_custom.cpp`, `op_host/add_custom_tiling.h`, and `op_kernel/add_custom.cpp`, together with the generated or maintained build configuration. CANN 9.0 documents `find_package(ASC .../compiler/tikcpp/ascendc_kernel_cmake)`, `npu_op_package(... TYPE RUN ...)`, host code generation through `npu_op_code_gen`, host libraries through `npu_op_library`, and kernel building through `npu_op_kernel_library`. The host package contains ACLNN, tiling, and optionally graph prototype libraries; the kernel target references the tiling library. Set `ENABLE_BINARY_PACKAGE=True` and choose an explicit vendor name. `ASCEND_COMPUTE_UNIT` is a separate build setting; inspect what the local generated project sets for 910B2. The documented RUN build sequence is:

```bash
cmake -S . -B build_out --preset=default
cmake --build build_out --target binary -j4
cmake --build build_out --target package -j4
```

Presets need CMake 3.19 or newer. Preserve the generated project in build artifacts so that the exact toolkit template used remains inspectable. [CANN 9.0 build extension reference](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_00035.html)

## Host registration and tiling

The official AddCustom host implementation uses `register/op_def_registry.h`, an `OpDef` subclass, required FP16 ND inputs/output, shape inference by copying input shape, and dtype inference by copying input dtype. It registers using `OP_ADD(AddCustom)`. Its AICore configuration includes `ascend910b`.

Its tiling callback obtains the element count through `context->GetInputShape(0)->GetOriginShape().GetShapeSize()`, sets block dimension 8 and tile count 8, serializes with `SaveToBuffer`, sets the raw tiling data size, and requests one workspace entry containing zero. Preserve this shape/tiling relationship when changing names; the API name derives from the registered operator type. [Official host implementation](https://gitee.com/ascend/samples/blob/master/operator/ascendc/0_introduction/1_add_frameworklaunch/AddCustom/op_host/add_custom.cpp)

The tiling header includes `register/tilingdata_base.h`. Its structure uses `BEGIN_TILING_DATA_DEF(TilingData)`, two `uint32_t` fields named `totalLength` and `tileNum`, `END_TILING_DATA_DEF`, and `REGISTER_TILING_DATA_CLASS(AddCustom, TilingData)`. [Versioned official tiling header](https://gitee.com/ascend/samples/blob/v1.0-8.1.RC1.alpha001/operator/ascendc/0_introduction/1_add_frameworklaunch/AddCustom/op_host/add_custom_tiling.h)

## Device kernel limitations

The sample kernel includes `kernel_operator.h`, obtains tiling through `GET_TILING_DATA`, and launches an `extern "C" __global__ __aicore__` function receiving x, y, z, workspace, and tiling. It divides each block's data into eight tiles and two buffers, uses `TQue`, `GlobalTensor`, `DataCopy`, and `AscendC::Add`, with generated `DTYPE_X/Y/Z` types. It does not implement remainder handling. [Official kernel implementation](https://gitee.com/ascend/samples/blob/master/operator/ascendc/0_introduction/1_add_frameworklaunch/AddCustom/op_kernel/add_custom.cpp)

Consequently, the first demo should explicitly reject unsupported shapes instead of silently dropping a tail. A conservative initial contract is contiguous FP16, equal shapes/devices, nonzero element count divisible by 2048, with a bounded tile size appropriate to available UB. The 2048 divisor follows from 8 blocks × 8 tiles × 2 buffers × 16 FP16 elements per 32-byte transfer block. This is a derived implementation constraint, not the general limitation of Ascend C. Start with `[8, 2048]`; arbitrary sizes require a revised tiling/kernel implementation.

## ACLNN adapter and libraries

After compilation/installation, inspect `vendors/<vendor>/op_api/include/aclnn_add_custom.h` and `op_api/lib/libcust_opapi.so`. The generated interfaces are two-phase: `aclnnAddCustomGetWorkspaceSize(x, y, out, &bytes, &executor)` followed by `aclnnAddCustom(workspace, bytes, executor, stream)`. The official direct invocation example links `cust_opapi`, `nnopbase`, and `acl_rt`, using the CANN and vendor include/library directories. Use the actual libraries shipped with 9.0.1 if their compatibility names differ. [Official single-operator calling reference](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_0070.html)

For a direct adapter, construct ACL ND tensor descriptors from the validated contiguous ATen tensors, allocate output using ATen, use the input device guard/current stream, and retain descriptor/workspace/tensor lifetimes through execution. Errors need to propagate to Python. Do not call `aclInit`, reset the device, or finalize ACL inside each PyTorch operator call; TorchNPU owns that runtime. These are adapter design requirements inferred from embedding the two-phase API in TorchNPU, not instructions to copy the standalone ACL application's initialization sequence.

Register the schema and NPU implementation using `TORCH_LIBRARY` and `TORCH_LIBRARY_IMPL(..., PrivateUse1, ...)`; expose only the intended public API, such as `torch.ops.ascend_demo.add`. TorchNPU's `NpuExtension` adds the installed torch_npu/Torch include paths and libraries `c10`, `torch`, `torch_cpu`, `torch_python`, and `torch_npu`; extra ACLNN paths/libraries must be supplied by the operator package. [NpuExtension implementation](https://raw.githubusercontent.com/Ascend/pytorch/master/torch_npu/utils/cpp_extension.py)

## EXEC_NPU_CMD is not a standalone header

The official pattern `EXEC_NPU_CMD(aclnnAddCustom, x, y, result)` handles conversion, workspace discovery, runtime execution and queueing. However, the current `op_plugin/utils/op_api_common.h` includes `op_api_common_base.h`, calls shared functions, and references external globals. Copying only that header into a standalone extension is insufficient. Inspect the installed package's exported headers/symbols or vendor a complete compatible helper with its license and upstream revision. Do not substitute a moving `master` helper without checking the installed 2.10.0.post2 APIs. [Current helper source](https://raw.githubusercontent.com/Ascend/op-plugin/master/op_plugin/utils/op_api_common.h)

Its implementation resolves custom library paths from `ASCEND_CUSTOM_OPP_PATH`, appending `/op_api/lib/`; it also supports default vendors in `ASCEND_OPP_PATH/vendors/config.ini`. Some values are initialized globally, so activate the chosen OPP prefix before importing TorchNPU. Always launch a new Python process after switching versions. [Helper library discovery implementation](https://raw.githubusercontent.com/Ascend/op-plugin/master/op_plugin/utils/op_api_common.cpp)

That source's base-format tensor conversion passes view dimensions, strides, storage offset, a one-dimensional storage element count (`storage().nbytes()/itemsize()`), and the storage base pointer to `aclCreateTensor`. Do not combine a pointer already advanced to the view with the original nonzero storage offset. A demo can reject nonzero offsets and non-ND storage until it implements the full conversion. The declarations and exported `GetOpApiFuncAddr` are in the separate base header. [Conversion implementation](https://raw.githubusercontent.com/Ascend/op-plugin/master/op_plugin/utils/op_api_common.cpp), [base helper declarations](https://raw.githubusercontent.com/Ascend/op-plugin/master/op_plugin/utils/op_api_common_base.h)

## Direct launch fallback

The current official standalone extension example builds `.asc` files with `bisheng -x asc` and `--npu-arch=dav-2201` for the 910B architecture, and obtains the ABI from `torch._C._GLIBCXX_USE_CXX11_ABI`. It recommends `getCurrentNPUStream().stream(false)` with a kernel-launch lambda submitted through `OpCommand::RunOpApiV2`, preserving TorchNPU task queue ordering. Raw launch on a stream without accounting for pending queued TorchNPU work can break ordering. [Official extension example](https://github.com/Ascend/op-plugin/tree/master/examples/cpp_extension), [build script](https://raw.githubusercontent.com/Ascend/op-plugin/master/examples/cpp_extension/setup.py)

This is a fallback when ACLNN packaging blocks progress; it provides a wheel but does not by itself satisfy the planned `.run` package lifecycle. The current example uses int32, so its datatype and size assumptions must be changed explicitly for the FP16 demo. A successful direct-launch test must not be reported as validation of the ACLNN package path.

## Required evidence from the target container

Record installed compiler/tool versions and actual generated build flags; successful `.run` build and isolated installation; wheel build and clean import; dispatcher registration; a numerical comparison against a CPU reference; and a trace identifying the custom kernel. Keep source revision, generated sources, package hashes, full logs and profiler artifacts with the experiment. These are acceptance requirements for this project; source research alone does not establish them.

## Implementation follow-up in the required image

The installed CANN 9.0.1 generator uses a plain `AddCustomTilingData` struct in
`op_kernel/add_custom_tiling.h`, `GetTilingData<T>()` on the host, and
`REGISTER_TILING_DEFAULT` in the kernel. The implementation follows this generated
contract rather than the older sample's serialized tiling macros.

The working standalone adapter imports `BuildExtension` from PyTorch's
`torch.utils.cpp_extension` and `NpuExtension` from `torch_npu.utils.cpp_extension`.
In the installed 2.10.0.post2 headers the callback is submitted as
`OpCommand::RunOpApiV2("AddCustom", handler)`. Linking `cust_opapi`, `nnopbase`,
and `ascendcl` compiled and loaded successfully in the required image.

The implemented kernel uses DataCopyPad and handles arbitrary nonempty tails;
the older sample's divisibility-by-2048 restriction does not apply to this demo.
Actual NPU comparisons for `[8, 2048]`, `[1]`, `[17]`, and `[16387]` all reported
zero maximum absolute error. This verifies building, loading, and executing the
custom ACLNN path; profiler validation is tracked separately.

Two environment details were established by direct tests: a container exposing
only physical `/dev/davinci6` reports one logical device, index 0, and the image
has no IANA timezone database. The launcher therefore leaves runtime visibility
unmasked, records the physical ID separately, and mounts the host timezone
database read-only. The build copies its generator specification into a directory
without group/other write permissions before invoking msopgen.
