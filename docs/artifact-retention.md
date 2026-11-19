# Project artifact lifecycle

The current worktree contains active source, fixed mathematics, useful contracts,
small reviewed evidence and curated lessons. Retired implementation cycles and
necessary unique historical source are preserved in Git, not in standing duplicate
checkouts. See [project scope](project-scope.md) and [history](history.md).

## Before retirement

Inventory actual project-owned paths, allocated disk/tmpfs bytes, Git worktrees,
user modifications, running services/processes and downstream installation paths.
Check quota/grace/hard limits as well as filesystem free space. Follow symlinks only
to identify dependencies; do not delete external targets by following a link.

Retain source/dirty patches and unique untracked source, configuration, source/input
identities, small results, terminal job facts, and minimal unresolved reproducers.
Raw tensors, repetitive frozen sources, large profiler timelines, binaries and
rebuildable environments do not need permanent online storage. A former absolute
path or content hash is an identity, not an availability promise.

Use an explicit dry-run inventory with preservation and ownership decisions. Remove
registered experimental worktrees through Git after unique source is preserved and
HEAD reachability is checked. Protect the main/downstream worktrees and dependencies,
public software, reference repositories, unrelated caches and all user changes.
Unclear ownership or a live dependency is a per-path exception, not permission to
remove it or a reason to stop independent cleanup.

## Records and rebuilding

Preserve historical failures as failures; a stale running record is not converted
into a successful job because no process remains. Save actual terminal properties
before releasing retired project-only service records. Never stop another project.
Do not import old queues or replay old experiments merely to reconstruct context.

Small preservation bundles may be committed in a temporary historical commit and
removed from the final tree in the next commit. Report what bytes were retained,
what can be regenerated and what was deliberately retired. Do not move hundreds of
GiB elsewhere and count that as released storage, or place raw model data in Git.

After cleanup, [development](development.md) rebuilds a private environment from the
published stack. Verify a clean installed smoke without retired absolute paths.
Record project allocated bytes and actual exceptions; filesystem free-space changes
may also reflect other users. Temporary validation output is retired after the
reviewed result is committed. Git history and user-owned files remain intact.

## Current lossless record archives

The [storage finalization report](evidence/storage-finalization-20261119.md) owns the
current availability decisions. Large raw JSON records may be retained as `.json.gz`;
the [archive index](evidence/storage-archive-index-20261119.json) records original and
compressed SHA256, lengths and logical paths. `scripts/materialize_record.py` restores
exact bytes to an explicit new path, checks both identities, bounds decompression and
refuses overwrite. Originals are unlinked only after complete decompression verification
and fsynced archive/index publication. A failure remains the same original failure.

Git retains the index, reviewed summaries and compressed retired-payload **metadata**;
the archived JSON content remains in local artifacts. Restoring JSON to a new location
does not restore its transitive dependencies or rewrite embedded original absolute paths.
Tools may require related records at their original logical paths. Explicitly retired
tensor/trace payloads cannot be reconstructed by this helper. Historical exact replay is
not an ongoing replay promise after its anchor/update inputs have been retired.

Downstream installation and normal execution require the package/source and runtime,
not experiment artifacts. This does not authorize wholesale removal of `artifacts`:
registered worktrees, unique source and useful witnesses remain protected there.
