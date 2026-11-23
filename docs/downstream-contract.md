# Downstream collaboration for Tide instances

Current protocol: `tide-instances-collaboration-v2`. This replaces the old foundation
CLI/plugin protocol for new instance consumers. Existing requests and dependency
locks retain their original adopted revision; rewriting this page does not migrate
or accept any downstream installation.

Install a fixed `tide-instances` source/wheel in the consumer's private environment.
Start from [instances](instances.md), [development](development.md), and the checkout
`examples/train.py` mechanism example (the wheel installs `tide`). The public entry is `tide`, not a benchmark
script. Keep the source commit, package/wheel identity, math revision, complete model
configuration, actual device/dtype and environment with the experiment dependency lock.

| Responsibility | Owner |
|---|---|
| Input data, task head/loss, recipe, model quality | Experiment repository |
| Selected model semantics, modules, gradients and state mechanisms | This repository |
| Full application checkpoint (data/RNG/scheduler as needed) | Consumer, composing the model checkpoint API |
| Mathematical changes | Explicit upstream proposal and a newly adopted fixed version |
| General performance experience | Curated here, with applicable scope and measured evidence |

## Development branches

The foundation and the learning lab deliberately maintain two branches of
`ZichaoLong/tide`. Work on the owning project's existing branch across threads.

| Project | Worktree and branch |
|---|---|
| Model operators, gradients and state | `graph-execution-foundation` |
| Experiment orchestration, training recipes and consumer acceptance | `tide-learning-lab` |

Commit authorized changes directly on the corresponding branch. When editing
files in both worktrees, commit each project's changes on its branch. Threads, experiments and
delegated tasks use those branches without creating additional task branches.

When cleaning up an old task branch, check its worktree use and whether its tip
is reachable from a retained branch. If it has unique commits, preserve the tip
and its history in a Git bundle and verify that they can be recovered before
deleting the local reference.

## Requests and delivery

A request stays in its originating repository with a stable ID, fixed dependency,
actual configuration, observed problem, available evidence/reproducer, and acceptance
conditions. Unknowns can be stated; a full performance matrix is not required to ask.
Accepted foundation work enters ROADMAP and current activity STATUS. Delivery gives
implementation/evidence commits, API/state changes, exact verified scope and a
consumer reproduction command. Each consumer records its own acceptance before
changing its lock. Upstream delivery never implies downstream acceptance.

### Following a fix across repositories

| Stage | What the owning repository records |
|---|---|
| Foundation delivery | Link the consumer's request to the adopted math, fixed implementation commit, affected public behavior, independent witness and a runnable regression. Record the source, devices and precision actually checked. |
| Consumer acceptance | Bind the originating diagnosis and the delivered implementation to an acceptance record. Check the quantities affected, including gradients, a real subsequent update or restored state when applicable. Record the fixture, oracle, tolerances and remaining deviations. |
| Batch adoption | The consumer binds its acceptance and fixed source to the new plan or launch. Its handoff records the adopted version and the first observed result. A later batch identifies the version it actually uses. |

The worked example connects a failed step, the foundation's closed-form regression,
CPU and Ascend acceptance, and the next official batch. Sibling checkouts named
`graph-execution-foundation` and `tide-learning-lab` resolve the relative links.
The case names the fixed commit for standalone source lookup.
[r24 consumer case](../../tide-learning-lab/docs/requests/pilot-r3-r24-quad.md) · [Foundation numerical regressions](numerical-failure-modes.md)

### Read-only status

Each checkout ships `python scripts/status.py` and its own observation adapter.
`--json` emits `tide-worktree-status-v1`. The shared fields give the repository,
branch, revision, worktree changes, UTC sampling interval, current handoff title
and its text. `runtime` gives the observation kind, summary, sources and next
read-only check. Each owner supplies the evidence appropriate to its work.
When changing the common command, compare its Git blob across both branches and
exercise human and JSON output in separate clean checkouts.

| Observation kind | Evidence |
|---|---|
| `live` | The consumer bound its scheduler to this host's active service, process start identity and cgroup. Driver observations have their own availability. |
| `snapshot_only` | Local records exist. The foundation lists each job receipt's recorded state, source, elapsed time and exit code. The consumer reports its scheduler and any host checks it could perform. |
| `unavailable` | The required local records are missing or unreadable in this checkout. Consult its dated handoff for the last reviewed sample. Establishing that a service stopped requires an observation of that service. |

Keep the current handoff in each repository's `docs/STATUS.md`, its outstanding work
in `docs/ROADMAP.md` and durable evidence at stable paths. Runtime pointers, data,
checkpoints, and private installations are recovered separately by their recorded
identity. Select the current thread's scope for edits, branch commits, pushes and
experiment launches. Each operation uses its own authorization.

Freeze installations used by active experiments. Upgrade in a separate installation,
check affected inference/gradients/updates/state and then adopt explicitly. Source
identity, exact checkpoint data, tolerant numeric replay and bitwise optimizer replay
are separate claims. Public API redesign is allowed; expensive compatibility is
replaced by a documented migration when appropriate.

Repository access or a request file does not grant cross-repository write/push or
message-sending authority. Preserve downstream worktrees, installations and model
paths during foundation cleanup. Old protocol details are retrievable from their
adopted commits only when historical investigation is requested.
