# Resident resource admission and inventory

Fixed implementation **08a349ce** passes two new resident builds, eight CPU checks,
six standalone SDK processes and four Python native processes. The
[machine record](resident-resources-20261112.json) retains all commands, manifests,
process results, runtime admission and launch scripts. No performance measurement
was performed at this revision.

The session now appends service stops, admits known tensor storage and planned
workspace **before** CANN arena allocation/control recording, then checks again
after lowering. CPU refusal witnesses distinguish stop storage, planned workspace,
post-lowering additions, exact limits and a program without services. Existing
actual-device session checks retain constructor/input refusal and unchanged state.

Public `resource_usage()` and Python `manifest()["resource_usage"]` expose actual
retained programs, workers, streams, notifications, CUDA boundary events, service
nesting, windows and storage. Counts come from current owned handles; they are
neither vendor task counts nor peak usage. A default hardware ceiling was not inferred.

| Evidence | Result and scope |
| --- | --- |
| CPU | 8 passed: admission sequence, public metadata, capability dispatch and optimizer inventory |
| Python-owned / standalone CANN builds | Both passed; immutable 08a349ce resident with hash-verified unchanged 72e1d22c core |
| SDK services | Repeated/empty windows, three-level nesting, siblings and shared workspace passed; actual 6 programs, 12 streams, 10 notifications, 0 CUDA events, 16,777,472-byte workspace |
| SDK session | Independent formula/VJP, two-window BPTT, three AdamW steps and exact raw new-owner checkpoint passed |
| SDK public qry/sum | Both three-step/independent-root consumers and each fresh-process restore passed; live forward/reverse inventory and release counts asserted |
| Python native public qry/sum | Both installed three-step consumers and each fresh-process restore passed |

The SDK public checks observe two retained windows before backward, one reverse
coordinator afterwards, and zero retained programs/streams/notifications after
step. Programs/resources are checked separately from numerical results and canonical
None/zero parameter roots. Python manifest inventory is a current snapshot; users
must query after advance/backward and before detach/step to see retained work.

Two independent background.slice builds used CPU0–159/160–319, each 512 GiB and
2,400 seconds. Device qualification used separate CPU24–27/28–31 pools, 256 GiB,
1,200-second limits and a 300-second per-process bound. Cooperative idle admission
selected physical NPU11 for native and NPU5 for SDK; logical `npu:0` was used by
consumers. The separate 72e1d22c cost runs used other CPU/device resources; these
checks make no timing/exclusivity claim. Published Python CANN9.2 and standalone
LibTorch-NPU2.12/CANN9.2 stacks were kept separate.

From a new checkout, follow [development](../development.md), build this fixed
resident source against its matching core, run `test_resident_block_resources`,
then the recorded services/session/public qry/sum commands. Each public archive
has a separate `--restore-check` process. Replace historical site/output paths;
the machine record contains exact inputs and source/build identities.

CUDA resources were updated in source. The CUDA program translation unit was
not compiled on this machine and no CUDA device ran; CUDA tasks keep separate
qualification. Partial-submit failure handling, overall timeout semantics,
shared KV history and larger program-resource limits remain outside this change.
See [session contract](../resident-block-session.md) and [ROADMAP](../ROADMAP.md).
