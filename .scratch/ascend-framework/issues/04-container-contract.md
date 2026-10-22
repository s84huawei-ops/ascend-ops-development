# How should the container be isolated and a device selected on the shared server?

Type: grilling
Label: wayfinder:grilling
Status: resolved
Assignee: root
Parent: ../map.md
Blocked by: 01, 02

## Question

How should the existing `docker_run.sh` be adapted to the repository, required image version, and one selectable NPU, while retaining the access required for compilation and both profilers?

Settle: mounts and workdir, container name and lifecycle, in-container installations, image pinning by tag+digest, physical/logical NPU IDs, availability checks and locking for concurrent experiments, behavior for an existing container, result permissions, experiment naming timezone, and actual profiler permission requirements. Validation in a fresh container will be an implementation acceptance condition, not an established fact from the current inventory.

Do not modify other users' containers or the host. Inputs: `docs/environment-inventory.md` and both research reports.

Profiling research finding: resolve device mapping for simulation separately. Older releases described restrictions on logical device 0; this is not a confirmed CANN 9.0.1 limitation. Assume neither automatic use of physical NPU6 nor a requirement to occupy physical NPU0. Validation must not interfere with other users' workloads.

## Implemented answer

The adapted launcher pins the requested tag plus its verified digest, mounts the repository at `/workspace`, and creates a dedicated `ascend-ops-dev-npu6` container. Only physical device 6 and management devices are exposed; the runtime maps the selected NPU to logical 0. Driver and timezone data are read-only mounts. Schema/workspace/device/image labels are checked before reuse. Packages install under build-specific prefixes, not into global torch or toolkit directories. Per-device filesystem locks and occupancy checks reject concurrent/busy runs. Default timezone is Europe/Warsaw. On-device and binary simulator profiling both ran successfully with the documented container capabilities. Other users' workloads were not changed.
