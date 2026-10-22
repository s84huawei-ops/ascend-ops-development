# How should operator versions be pinned and transferred between the workstation and server?

Type: grilling
Label: wayfinder:grilling
Status: open
Assignee: unassigned
Parent: ../map.md
Blocked by: none

## Question

What versioning and synchronization contract should connect the framework with the `op-src` submodule from the planned `s84huawei-ops/ascend-op-demo` repository?

Decide: tag/branch/SHA switching and recording the SHA in the parent repository; dirty trees and uncommitted changes; transferring commits to the server; independent framework and operator versions; behavior when a commit is unavailable; and safe management of build artifacts after source changes. Describe the expected script behavior without implementing it.

The user approved a separate operator repository and a Git-based workflow. Creating the repository and establishing access remain future implementation tasks.
