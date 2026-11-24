# Working on Tide instances

## Re-entry

Run `git status --short --branch` and start with the user's task in this thread.
Commit foundation work directly on `graph-execution-foundation`. The learning lab
maintains `tide-learning-lab` in its own worktree. Use each project's existing
branch for every authorized task, including delegated work. Do not create task branches.
Check worktree use and preserve unique commits when retiring an old task branch.
[Development branches](docs/downstream-contract.md#development-branches)

| Task | Entry point |
|---|---|
| Check status | Run `python scripts/status.py` or add `--json`. Its sampling time and local job receipts accompany the recorded handoff. The consumer reports its own live campaigns. [Current handoff](docs/STATUS.md) |
| Continue the same work | Recover the accepted scope, implementation revision, and available verification or job records. Follow the outstanding item and any bounds of that job. [Handoff](docs/STATUS.md) · [Backlog](docs/ROADMAP.md) · [Execution](docs/execution-flows.md) |
| Locate and change an implementation | Map the mathematical binding to `src/tide`, a small independent case and its test. Run `python scripts/check_upstream.py` when checking the fixed mathematics. [Semantic map](docs/semantic-map.md) · [Architecture](docs/architecture.md) · [Numerical regressions](docs/numerical-failure-modes.md) |
| Start a new cycle | Use this thread's adopted sources, bindings, goals and authority. Select the applicable tests once the semantics are known. [Product scope](docs/project-scope.md) · [Verification](docs/verification.md) |

`docs/upstream.json` pins the adopted `math/` snapshot. Adopt a new revision only for a user task.

## Product and implementation

The instance product route can continue indefinitely. The general graph foundation
is a separate future route, activated explicitly; it adds no current API, graph or
backend obligations. The active FC/HB source is `src/tide`. This implementation's
FP32/BF16 qualification is recorded in STATUS and its source-scoped evidence.

Semantic correctness comes first, performance second. Interfaces serve tensor flow,
clear ownership and replaceable implementations; instance/backend specialization
and API redesign are allowed. Task heads, data, losses and training recipes belong
to downstream experiments. See [semantics](docs/semantics.md) before changing behavior.

Retired cycles live in Git. Look up their implementation/history only when the user
explicitly requests historical investigation, using [history](docs/history.md).
Curated [performance lessons](docs/performance-lessons.md) are available directly;
they are observations with applicability, not inherited task requirements.

## Work and maintenance

Use the relevant repository skill: develop-tide-foundation, optimize-tide-execution,
or maintain-tide-foundation. Their names are discovery identifiers, not a mandate to
build the long-term foundation. User instructions override repository guidance.
Choose serial tools, parallel calls or sub-agents by benefit; separate edits/job
inputs, coordinate resources, review results and integrate one coherent change.

Work on the authorized branch and preserve user changes/untracked files. Reference
repositories and public toolchains remain read-only unless explicitly included in
the task. Downstream delivery and downstream acceptance are separate facts; follow
[the current collaboration contract](docs/downstream-contract.md).

## Checks, execution and records

Use independent small formulas, meaningful gradient/update/continuation checks and
useful candidate combinations. [Verification](docs/verification.md) defines scope;
[precision](docs/precision.md) separates numerical policy from qualification.
Four jointly matched Transformer comparisons follow [benchmark contract](docs/benchmark-contract.md).
Record actual dtype, operator/device, timing boundary, workload and memory.

Long work uses frozen inputs, detached `background.slice` services, finite resource
and time bounds, and durable logs per [execution](docs/execution-flows.md).
Available CPUs/memory/idle NPUs may be used fully with coordination. Shared NPU use
requires current-conversation authorization, not a repository default.

For formal submissions through the account scheduler, use the installed
`use-runq` skill and `~/.local/bin/runq submit --help`. Declare host CPU threads,
host RAM for each allowed backend and HBM per assigned NPU. Developers can test
directly on a shared physical card with sufficient free memory using
`use-local-ascend`. Measurements and project metadata are optional queue inputs.

Keep STATUS as the current handoff and ROADMAP as the backlog. Update handoffs via
fsynced staging and atomic replacement (the current `scripts/job.py:atomic` helper
handles JSON records; use the same staged protocol for text).
Commit coherent checked changes; fix implementation identity before qualification
and record evidence separately. Push when the user task authorizes it.
Retire project-only artifacts after dependency/dirty-state review according to
[artifact retention](docs/artifact-retention.md); protect downstream installations.
