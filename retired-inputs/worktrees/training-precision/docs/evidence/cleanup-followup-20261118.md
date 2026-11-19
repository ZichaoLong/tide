# Authorized residual cleanup — 2026-11-18

Source before this cleanup: `d211032804b087fd901295e87a0668168f603b05`.
The user confirmed that other projects no longer depend on this foundation and
authorized removal of the specifically reviewed remnants. This supplements the
[earlier governance retirement](governance-cleanup-20261118.md).
[Machine-readable evidence](cleanup-followup-20261118.json) retains exact paths,
source identities, small provenance records, deletion results and protected items.

| Removed | Allocated bytes before removal |
|---|---:|
| Two downstream-owned historical foundation source/build copies | 97,492,992 |
| Their two pip wheel-cache leaves | 888,832 |
| Old model aliases and small experiment provenance directory | 53,248 |
| Ten empty temporary directory trees | 0 |
| Total: 15 explicit roots | **98,435,072 (93.875 MiB)** |

The source copies matched all 2,947 / 2,934 tracked blobs at commits
`74e7bbd62341d2ccd382194a2c52ad4fea033665` and
`85d161d5c9148eac2b164f89a0ba4204b4331898` respectively. Both commits remain
ancestors of the current branch. There was no changed, extra or missing tracked
source; generated build/package caches were discarded. The five small external
provenance files are retained in the JSON record before their originals were removed.

A scan of 112 owned processes found no command, mapped file, environment, working
directory or open-file reference to these targets. Exact paths, directory identities
and ownership were checked before deletion; symbolic links were unlinked without
following their public model targets.

The first removal attempt encountered owner-read-only directories in one frozen
snapshot. All pre-delete identities still matched and no target had been removed.
Owner write was enabled on its 79 directories, and the explicit removal was retried
successfully. Both the failed attempt and recovery remain recorded.

All 15 roots are now absent. The six public model target directories, four user
untracked documents, downstream HEAD and tracked status are unchanged. The main
and downstream Git worktrees remain registered; converting the downstream into
an independent Git repository was outside this cleanup. Other project artifacts,
shared toolchains, upstream review environments and unrelated caches were preserved.

Current model/config/test/math files are unchanged. No model or performance tests
were rerun for deletion of these unused copies. The fixed mathematics hash check,
protected-file checks, deletion verification and current document links were checked.
Future startup needs neither the retired source directories nor the removed aliases.
