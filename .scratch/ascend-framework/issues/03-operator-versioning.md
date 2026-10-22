# How should operator versions be pinned and transferred between the workstation and server?

Type: grilling
Label: wayfinder:grilling
Status: resolved
Assignee: root
Parent: ../map.md
Blocked by: none

## Question

What versioning and synchronization contract should connect the framework with the `op-src` submodule from the planned `s84huawei-ops/ascend-op-demo` repository?

Decide: tag/branch/SHA switching and recording the SHA in the parent repository; dirty trees and uncommitted changes; transferring commits to the server; independent framework and operator versions; behavior when a commit is unavailable; and safe management of build artifacts after source changes. Describe the expected script behavior without implementing it.

The user approved a separate operator repository and a Git-based workflow. Creating the repository and establishing access remain future implementation tasks.

## Implemented answer

The operator lives in the private `s84huawei-ops/ascend-op-demo` repository. `op-src` is a real Git submodule with a relative GitHub URL and an immutable parent pin. `switch_operator.sh` resolves tags, branches, or SHAs, refuses dirty sources, optionally fetches, checks out detached, and stages the parent pointer. Missing revisions fail without replacing sources. Experiments verify any expected `operator.ref`; they never silently update it. Build keys include source content and revision. `sync_server.py` transfers clean committed versions as Git bundles over SSH, avoiding GitHub credentials on the server. See [usage](../../../docs/usage.md).
