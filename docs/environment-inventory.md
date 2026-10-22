# Environment inventory

Observed: 2026-10-22, during the planning session. No server files were changed, no containers were started/stopped, and no NPU initialization, compilation, or profiling was performed. The evidence comes from SSH commands rather than vendor claims.

## Repositories and server

- Local: `D:\Praca\Huawei\ascend-ops-develop\ascend-ops-development`, `main` with no commits.
- Remote: `ssh huawei-blue-npu`, `/home/s84435910/docker_workspace/ascend-ops-development`, also with no commits; the only working file is the untracked `docker_run.sh`.
- Origin in both copies: `https://github.com/s84huawei-ops/ascend-ops-development.git`.
- No operator sources or `AGENTS.md` instructions were found in the inspected workspace.
- Host: `aarch64`, Docker `18.09.0`.
- NPUs: 8 × Ascend 910B2, 64 GB each; version reported by `npu-smi`: `25.0.rc1.1`.
- At observation time, devices 0–5 had running processes; 6 and 7 showed no processes but reported approximately 3.4 GB of occupied HBM. This is neither a reservation nor evidence of exclusive access to device 6.

## Required image

- The tag is available locally: `quay.io/ascend/vllm-ascend:v0.23.0rc1`.
- Image ID: `13315b656180`; reported size: 17.2 GB.
- Digest: `quay.io/ascend/vllm-ascend@sha256:9fd154211d2faee7735cfe40785f863603e42ccd53bfb4bca82dc98c9446ea0a`.

## Existing docker_run.sh

The script accepts two arguments: image and container name (its validation message incorrectly mentions one argument). It invokes `docker run -it -d --net=host --shm-size=1024g --privileged=true`, `--entrypoint=bash`, and `-w /home`.

It mounts the management devices `/dev/davinci_manager`, `/dev/hisi_hdc`, and `/dev/devmm_svm`; it does not explicitly select a particular `/dev/davinciN`. It mounts the Ascend driver, dcmi, npu-smi, openmpi, `/usr/local/sbin`, `/data`, `/docker`, and all of `/home/s84435910/docker_workspace` at `/home`. It sets local time through `/usr/share/zoneinfo/Asia/Shanghai:/etc/localtime`.

Future implementation must settle paths, workdir, NPU visibility, profiler permissions, timezone, artifact ownership, and safe container restarts. We do not assume that all existing mounts and privileges are necessary.

## Observations from an already running container

The container `research_vllm_ascend_023_external_workspace` is based on the required image, with other workspaces mounted and editable installations. Do not modify it. The observed versions **do not establish the contents of a fresh container**:

| Component | Observed |
| --- | --- |
| Python | 3.12.13 |
| torch | 2.10.0+cpu |
| torch_npu | 2.10.0.post2 |
| vllm | 0.23.0+empty, editable |
| vllm_ascend | 0.23.0rc1, editable |
| CMake | 4.4.0 |
| CANN | `/usr/local/Ascend/cann-9.0.1` |
| Compiler | `bisheng` available |

`msprof op --help` works and describes kernel selection, warmup, launch count, replay mode, and metrics including Roofline, Occupancy, Source, and TimelineDetail. Simulator directories for Ascend910B2 are installed. Compilation, operator invocation, on-device `msprof`, and execution of `msprof op simulator` **were not tested**.

## Evidence and limitations

Read-only checks included `git status/remote/log`, `docker images/ps/inspect`, `npu-smi info`, the contents of `docker_run.sh`, and tool version/help output in the existing container. Before implementation, recheck device availability and the versions and tools in a fresh, dedicated container.
