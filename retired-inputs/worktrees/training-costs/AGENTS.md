# Working on Tide instances

## Re-entry

1. Run `git status --short --branch` and `python scripts/status.py`.
2. Read [STATUS](docs/STATUS.md), [project scope](docs/project-scope.md), and
   [ROADMAP](docs/ROADMAP.md). Scope is the enduring agreement; status is fact;
   roadmap is this cycle's plan, not a declaration of implemented capability.
3. Use [architecture](docs/architecture.md) and [semantic map](docs/semantic-map.md)
   to locate current code and math. Run `python scripts/check_upstream.py`.
   `docs/upstream.json` pins `math/`; adopt a new revision only for a user task.

## Product and implementation

The instance product route can continue indefinitely. The general graph foundation
is a separate future route, activated explicitly; it adds no current API, graph or
backend obligations. The active FC/HB source is `src/tide`. The next cycle work
includes FP32/BF16; actual precision qualification is in STATUS and evidence.

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

Keep STATUS as the current handoff and ROADMAP as the backlog. Update handoffs via
fsynced staging and atomic replacement (the current `scripts/job.py:atomic` helper
handles JSON records; use the same staged protocol for text).
Commit coherent checked changes; fix implementation identity before qualification
and record evidence separately. Push when the user task authorizes it.
Retire project-only artifacts after dependency/dirty-state review according to
[artifact retention](docs/artifact-retention.md); protect downstream installations.
