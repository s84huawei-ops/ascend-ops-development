# How should the container be isolated and a device selected on the shared server?

Type: grilling
Label: wayfinder:grilling
Status: open
Assignee: unassigned
Parent: ../map.md
Blocked by: 01, 02

## Question

How should the existing `docker_run.sh` be adapted to the repository, required image version, and one selectable NPU, while retaining the access required for compilation and both profilers?

Settle: mounts and workdir, container name and lifecycle, in-container installations, image pinning by tag+digest, physical/logical NPU IDs, availability checks and locking for concurrent experiments, behavior for an existing container, result permissions, experiment naming timezone, and actual profiler permission requirements. Validation in a fresh container will be an implementation acceptance condition, not an established fact from the current inventory.

Do not modify other users' containers or the host. Inputs: `docs/environment-inventory.md` and both research reports.

Profiling research finding: resolve device mapping for simulation separately. Older releases described restrictions on logical device 0; this is not a confirmed CANN 9.0.1 limitation. Assume neither automatic use of physical NPU6 nor a requirement to occupy physical NPU0. Validation must not interfere with other users' workloads.
