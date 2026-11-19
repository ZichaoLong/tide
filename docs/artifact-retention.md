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

## Current compact storage

The [compact repository report](evidence/compact-repository-20261119.md) owns current
availability. The user explicitly requested the whole checkout, common Git and private
environment below 200 MB, including retirement of the remaining diagnostic payloads.
Original artifacts and experimental checkouts are no longer online. Original branch
refs, exact tracked trees and unique ignored source/small records remain in Git history;
reviewed failures and qualifications remain available without storing model payloads.

The [current archive index](evidence/archive-index.json) points to 47 byte-preserved
`docs/evidence/*.json.gz` files stored in Git. `scripts/materialize_record.py` checks both
SHA256 identities and length, bounds decompression, stages/fsyncs output and refuses
overwrite. Restoring reviewed JSON does not reconstruct retired transitive inputs or
rewrite embedded absolute paths. Historical artifact archive indices retain identities,
not a present availability promise. Future investigations need new bounded inputs.

Downstream installation and normal execution use the package/source and runtime.
Protect the main/downstream worktrees and common Git, public environments, reference
repositories and user changes. The current cleanup retained all four untracked user
documents byte-for-byte and did not touch concurrently created downstream files.
