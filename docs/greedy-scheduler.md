# Online greedy region blocks

`schedule=greedy` / native `tide::Greedy` is an optional host scheduler for
general positive-delay graphs. It uses the existing local region-block
contracts and an independent scalar schedule for verification. It does not
replace `streaming`, the older DAG `frontier`, or the future resident NPU
scheduler. Accelerator tensor placement alone does not device-place this loop.

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
The current host transport still has per-atom bookkeeping; packed accelerator
transport and fully device-owned progression remain separate unfinished work.

Counters report `greedy_stages`, `greedy_relaxed_edges`, `max_live_fibers`,
`max_greedy_frames`, actual candidates/selected events/edges and the existing
state/Full batching/fallback counts. Complete-window and profiling claims require
separate measured evidence. Development tests are in `tests/test_greedy.py`.

Clean CPU regression and six NPU FP32 fixture results are recorded in
[the 2026-09-30 evidence](evidence/online-greedy-20260930.md). These qualify
host scheduling only; they do not establish device-resident scheduling.
