# Experimental content-driven device flow

`tools/device_online/ContentFlow` connects the device queue, readiness, selection,
state and delivery stages into a single submitted forward loop. Implementation
and qualification status belong to [STATUS](STATUS.md) and [ROADMAP F4](ROADMAP.md).
Its finite module scope does not close the complete execution-flow contract.

The accepted profile uses existing semantics: sum Aggregate with physical source
scales; identity or EMA state; content-mode linear Read; count-v1 or positive-v1
selection; adopt-v1 Next with optional selected clear; identity broadcast Full.
It supports observe-all and active-only state adoption. Inputs are arbitrary
legal sealed-window values; topology can contain unequal positive delays,
parallel physical edges, feedback and disconnected components. Input-origin
projection, other modules/region programs and nontrivial state clocks explicitly
fail capability validation. This first version accepts FP32 inference with an
explicit no-grad scope. It has no VJP or optimizer contract.

Content Read does not depend on the proposed state, and identity Full returns
content. Consequently the loop can compute actual packed content and scores,
select complete region-time frames, then execute each node's ordered state
sequence with the actual active bits. This preserves active-only adoption and
clear dependencies between successive actions. It is not a license to perform
the same reordering for proposal/old Read or arbitrary Full/state contracts.

Only static graph/source tables and buffers are constructed on the host.
Within a call, NPU tasks certify readiness, pack real fibers, compute content
and Read, select frames, propose states, generate actual edge/output messages,
preflight capacities, commit and decide whether to iterate. No numerical CPU
route prepass, potential-event expansion or per-event host decision is used.
Initial and external inputs are uploaded at API boundaries. Internal pending
messages, state and histories remain in persistent device buffers across calls.

All queue, output, history, state and diagnostic-log proposals finish
before any owner commits. A refusal preserves the current stage's old owners.
A call can already have committed earlier stages; an execution failure poisons
that object. Restore a previous complete cut into a new object. Malformed input
is rejected before submission and can be corrected on the same object.

`ContentLimits` separately bounds pending/input capacity, per-stage arrivals,
per-call outputs, debug records and iteration count. Exact int64 timestamps and
counters remain separate from FP32 values. Queue space is reclaimed after each
successful stage; debug records are cleared at call boundaries. An approximate
conservative buffer budget rejects oversized declared dimensions before main
workspace allocation. This is not yet the requested model/KV/training byte
budget or aggressive-safe chunking implementation.

Returned CPU `Result` values are diagnostic boundary materialization. State,
pending, outputs, emitted messages, numeric event snapshots, activity and source
contributions come from device records. Per-event history maps are reconstructed
from recorded activity and the initial history only for presentation; the final
history is independently downloaded from persistent device storage. Neither
presentation path feeds the next device loop. Restore accepts the returned
complete-cut continuation and a separately declared scheduling policy.

The initial content, state, output and journal kernels use scalar AIV loops over
packed buffers. They establish a forward semantic integration path, not optimized
compute throughput. Profiling and vectorized numerical kernels are required
before performance recommendations. Trace storage, CPU comparison and result
materialization must be separated from future steady-state throughput timing.
FP16, other module contracts, public Python/native packaging, peer progress and
training remain independent delivery requirements.
