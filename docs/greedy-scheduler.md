# Online greedy region blocks

`schedule=greedy` / native `tide::Greedy` is an optional host scheduler for
general positive-delay graphs. It uses the existing local region-block
contracts and an independent scalar schedule for verification. It does not
replace `streaming`, the DAG `frontier`, or the separate
[resident device scheduler](resident-library.md). Accelerator tensor placement alone does not device-place this loop.

At a sealed finite window boundary, all external inputs and incoming pending
messages are known. Retain actual unprocessed fibers, keyed by sample/node/time.
Every not-yet-published event must descend from one of these actual seeds.
Project wires onto region owners and retain the minimum physical delay for each
region pair *for closure accounting only*. Physical wires and message identities
remain separate in computation, transport and observations.

For each sample, multi-source shortest paths from the earliest actual queued
event of each region give lower bounds on unfinished region events. A region
fiber at time `t` is complete if `t` is strictly below every incoming source's
unfinished-event bound plus its wire delay. Select its complete chronological
prefix, batch the legal state/Full work using the existing region contract,
publish actual emitted messages and repeat. Selection and numerical values are
never computed during this bound calculation. Future-event sets are not expanded.

Correctness: any additional arrival before the computed bound would imply an
unfinished source event before its shortest-path lower bound. Region history
and node state remain ordered; no Full from another block of the same stage can
add a message to a selected fiber. Positive delays expose at least the globally
earliest actual pending event, so each nonempty stage makes progress. Idle clock
gaps do not require one iteration per tick. `stop` saturation uses int64 arithmetic
without floating conversion, including times above 2^55.

This is maximal relative to this conservative region closure certificate and
the existing region-block contract, not every possible finer P/S/U/F contract.
Feedback can split batches or reduce them to single steps; that is an observable
algorithm outcome. Changing input values may alter selection, actual pending
messages, subsequent bounds and batch shapes. Topology-specific cases are absent.

`max_events` limits simultaneously materialized fibers, including right-boundary
pending work, not cumulative events or logical clock span. It is an explicit
structural refusal limit, not yet the complete payload/activation memory budget
or safe-chunking policy required by execution-flows.md. No events are dropped.
This host algorithm retains per-atom bookkeeping. Optional
[packed eager transport](execution-placement.md#packed-eager-transport) and
[device-owned progression](resident-library.md) have separate contracts and
verification; their delivery does not make this host loop device-resident.

Counters report `greedy_stages`, `greedy_relaxed_edges`, `max_live_fibers`,
`max_greedy_frames`, actual candidates/selected events/edges and the existing
state/Full batching/fallback counts. Complete-window and profiling claims require
separate measured evidence. Development tests are in `tests/test_greedy.py`.

Clean CPU regression and six NPU FP32 fixture results are recorded in
[the 2026-09-30 evidence](evidence/online-greedy-20260930.md). These qualify
host scheduling only; they do not establish device-resident scheduling.

## Default component windows

The public host prefill default now resolves to `window`: GraphRuntime's
`schedule=auto` and weighted `prefill_schedule=auto`. Explicit `greedy` retains
the original closure policy; explicit streaming and resident are unchanged.
This is one general PDG algorithm, selected by dependencies and local contracts,
not FC/HB/LH names, graph class, parameter count or observed reference routes.

`window_plan.Components` and native `ComponentPlan` independently form strongly
connected components of the **region** graph, then traverse their condensation
DAG. A node DAG can still have a cyclic region dependency. Only static structure
is cached; inputs, fibers, state and selection remain private to the current run.
Once all predecessor components have drained actual work below `stop`, a singleton
component without a self edge has complete fibers for the whole sealed window.
One region block can therefore contain all its samples and logical times.
Incoming messages after `stop` stay pending; positive delays make them unable to
affect earlier work. Cyclic components reuse the closure certificate above on
their internal edges, after all outside predecessors have completed.

This is a topological dependency proof, not a numerical prepass. Non-aligned
path reconvergence cannot miss a late predecessor, same-rank independent regions
remain independent, and duplicate physical edges are never collapsed in delivery.
State/selection history still follows chronological sample-local order. Modules
without an exact sequence contract use their existing causal fallback inside the
block. `prefill=False` also retains scalar local progression.

Both implementations index actual fibers by component, avoiding repeated scans of
every other component. Native construction and Python's immutable graph cache
reuse the analysis across windows. Analysis is O(V+E) apart from deterministic
adjacency sorting; memory includes the graph and all live fibers. Iterative DFS
avoids a host recursion-depth limit. `max_events` includes future pending fibers
across **all** components, not just the component being visited. Capacity failure
does not truncate the return. This is not yet an automatic byte-budget splitter.

`window_components`, `window_cyclic_components`, `window_causal_blocks` supplement
the existing actual batch and fallback counters. A region block is not evidence
that all its internal operators or backward submissions have been combined.
Independent components currently run in a deterministic order; no claim is made
of maximal hardware overlap or a universally fastest ordering.

Weighted execution records with this policy use `tide-weighted-execution-v3`,
record requested and effective schedules, and change execution identity only.
Explicit greedy can reproduce previous v1/v2 identities; missing policy in an old
resolved record means greedy. New unresolved requests default to auto/window.
Graph v13/checkpoint v5 and the fixed mathematical functions are unchanged.
Changing a policy is a new execution choice, not a promise of bitwise continuation
across different floating-point evaluation orders.

Independent scalar checks: `test_component_window`,
`test_component_window_contracts` (general ports, history and state adoption),
`test_component_window_weighted` (five memories, ragged B×L, SGD and exact disk
resume), and `test_greedy_library` (public three-family training/checkpoints).
Policy/identity checks live in `test_component_window_policy` and
`test_experiment_placement`. Qualification and measured scale remain separate.
