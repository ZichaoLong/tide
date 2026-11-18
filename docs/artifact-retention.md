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
