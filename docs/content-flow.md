# Experimental content-driven device flow

`tools/device_online/ContentFlow` connects the device queue, readiness, selection,
state and delivery stages into a single submitted forward loop. Implementation
and qualification status belong to [STATUS](STATUS.md) and [ROADMAP F4](ROADMAP.md).
Its finite module scope does not close the complete execution-flow contract.

The accepted profile uses existing semantics: sum Aggregate with physical source
scales; identity or EMA state; content-mode linear Read; count-v1 or positive-v1
selection; adopt-v1 Next with optional selected clear; identity or tanh broadcast
Full (`content + tanh(comparison @ weight + bias)`).
It supports observe-all and active-only state adoption. Inputs are arbitrary
legal sealed-window values; topology can contain unequal positive delays,
parallel physical edges, feedback and disconnected components. Input-origin
projection, other modules/region programs and nontrivial state clocks explicitly
fail capability validation. This first version accepts FP32 inference with an
explicit no-grad scope. It has no VJP or optimizer contract.

Content Read does not depend on the proposed state. Consequently the loop can
compute actual packed content and scores, select complete region-time frames,
then execute each node's ordered state sequence with the actual active bits.
Each action saves its comparison before selected clear; Full consumes that
snapshot, never the cleared next state. This preserves active-only adoption and
clear dependencies between successive actions. It is not a license to perform
the same reordering for proposal/old Read or arbitrary Full/state contracts.

`PackedFull` scans the actual selected actions on device. Identity Full returns
content; selected tanh actions are gathered into bounded physical chunks for
FP32 batch matmul, bias, tanh and content addition, then bulk index-copy back.
The chunk cursor and subsequent chunk decisions stay on device. Padding gathers
zero sentinel inputs/parameters and writes distinct scratch destinations, so an
inactive owner's NaN parameters never enter the arithmetic and padding writes
cannot collide. CANN batch matmul explicitly uses KEEP_DTYPE, with no implicit
HF32/FP16 enablement. This is hard inference only; no new emit-mode or VJP claim.

Static graph/source tables, immutable parameters and buffers are prepared on the host.
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
workspace allocation. `full_chunk_rows` is a separate upper bound on physical
Full rows; a local parameter/scratch estimate can reduce it, and inability to fit
one row explicitly fails. This does not bound all model/KV/training memory or
replace the requested calibrated conservative/aggressive-safe planner. Actual
`full_chunks` and effective `full_chunk_rows` are reported per call.

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

The clean identity-Full qualification remains scoped to source `4d2f09e`
([evidence](evidence/content-loop-20260930.md)). The selected matrix Full increment
has directed development checks: 24 component cases at widths 1/7/33 and chunk
limits 1/4, plus 160 window/continuation comparisons against independent CPU
Streaming and Greedy. The latter cover identity/tanh Full, feedback, unequal
delays, parallel edges, active-only adoption, selected clear and schedule changes.
The component placement trace has 410 AIV and 16 AI Core tasks, with no recorded
AiCPU task or host-fallback diagnostic. It includes setup and CPU assertions,
not steady-state timing. Clean qualification of this increment is pending;
exact frozen source, jobs and reports are recorded in [STATUS](STATUS.md).
