# Experimental content-driven device flow

`tools/device_online/ContentFlow` connects the device queue, readiness, selection,
state and delivery stages into a single submitted forward loop. Implementation
and qualification status belong to [STATUS](STATUS.md) and [ROADMAP F4](ROADMAP.md).
Its finite module scope does not close the complete execution-flow contract.

The accepted profile uses existing semantics: sum Aggregate with physical source
scales; identity, EMA or Add-repeat state; content/old/proposal linear Read; count-v1 or positive-v1
selection; adopt-v1 Next with optional selected clear; identity or tanh broadcast
Full (`content + tanh(comparison @ weight + bias)`).
It supports observe-all and active-only state adoption. Inputs are arbitrary
legal sealed-window values; topology can contain unequal positive delays,
parallel physical edges, feedback and disconnected components. Input-origin
projection and other modules/region programs explicitly
fail capability validation. This first version accepts FP32 inference with an
explicit no-grad scope. It has no VJP or optimizer contract.

Content Read does not depend on the proposed state. Consequently the loop can
compute actual packed content and scores, select complete region-time frames,
then execute each node's ordered state sequence with the actual active bits.
Each action saves its comparison before selected clear; Full consumes that
snapshot, never the cleared next state. This preserves active-only adoption and
clear dependencies between successive actions.

Old/proposal Read uses ordered device scratch state. A region with observe-all,
no selected clear and the supported comparison-identity Next can prepare its
whole certified state sequence before selection. Otherwise the device ready
packer retains only the earliest complete region-time frame for that sample and
region; it preserves every candidate in that frame. After the actual selection
and state commit, the next iteration decides the following frame. Other regions
keep their legal time batches. This is a module-contract fallback, independent
of fixture topology or input values; it uses no advance numerical route trace.
Regions can choose different Read modes in one graph. Identity nodes keep an
exact zero descriptor.

Periodic `StateClock` policies are evaluated on the device using int64 division
and remainder. Upd sees local ticks; stored last-adopted timestamps, Read, history,
message coordinates and continuation cuts remain global. Every candidate validates
its event phase and previous state phase before commit; an off-phase event fails
with refusal9. Idle states are also validated when restoring the continuation.
Add's repeat-work budget counts local ticks, so reserved phases never add decay.

`state_read_single_frame_regions` records the static contract restriction;
`max_causal_node_time_batch` and `max_state_read_node_time_batch` report actual
observed batches. The former is at most one. These counters distinguish legal
state time batches from causal fallback; they are not throughput measurements.
This capability does not extend to arbitrary custom state/Read/Next programs.

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

`advance_device(inputs, stop)` returns borrowed, read-only device output buffers
and device counters. Their storage is reused by the next advance; consumers must
copy anything they retain beyond that boundary. State, history and pending
payloads are never downloaded to prepare the following window. CPU metadata
retains only the graph identity, batch size, complete cut and external-input
ledger. Inputs are still validated and uploaded at the host boundary. A call
waits for completion and checks the device error flag; it is not an asynchronous
host interface, nor a claim that the whole API has only one synchronization.

`snapshot()` explicitly materializes an independent complete-cut CPU continuation.
`result()` materializes the latest window's outputs and continuation, plus its
diagnostics when enabled. The existing `advance()` combines device advance with
result export. Failed execution poisons the owner and refuses snapshots, results
and further execution; malformed pre-submission input remains retryable.

`ContentLimits.diagnostics` defaults to true for equivalence checking. With false,
the captured program omits event/fiber/contribution/Full journals and the emitted
message log; `trace=0` is then legal. Persistent state, selection history, output
buffers and exact int64 event counts remain. A lean CPU result has no event trace
or message log and cannot support comparisons of those missing diagnostics.

Diagnostic state, outputs, emitted messages, activity and contributions come from
device records. Per-event history maps are reconstructed from recorded activity
and a device-captured window-start history only for presentation; final history
is independently downloaded from persistent device storage. This works even when
several advances precede the first result export. Neither presentation path feeds
the next device loop. Restore accepts a complete-cut continuation and a separately
declared scheduling policy.

`ContentLimits.vectorized_aggregate` selects the packed sum implementation. Its
device metadata preflight validates complete offsets, physical source ownership,
logical-source uniqueness and atom/fiber coordinates before any numerical write.
Independent AIV blocks then process disjoint (fiber, payload-tile) ranges using
vector multiply/add and exact-length transfers. Tiles hold up to256 FP32 elements;
one logical message group keeps its original stable accumulation order. Absent
atoms/fibers never enter arithmetic, including poisoned unused storage. No atom
count, source index or chunk decision is downloaded for host dispatch. The scalar
device implementation remains selectable for comparison.

`ContentLimits.vectorized_state` selects vector state updates (default) or the
scalar device implementation. Clock/selection metadata is validated first. Payload
tiles have a single writer per owner/width range, with sequential time updates
inside the kernel; independent owners and width tiles execute in parallel. The
pre-clear comparison is preserved for Full. Add uses the literal tick recurrence
from [lazy-add.md](lazy-add.md), including negative/zero retention, and its own
int64 last-adopted clock. The configurable positive `max_repeat_ticks` bounds
work per candidate and explicitly refuses excess work; it never truncates decay.

Directed development passed72 scalar/vector cases,144 input-changing replays and
nine metadata refusals, plus the existing640 content and384 window cases. Its837-task
trace is entirely AIV. Small single-message cases can be slower; the scalar switch
is retained and no complete-flow speed claim follows. Exact clean qualification
and any later measured recommendation belong to STATUS/evidence.

This vector path currently implements sum Aggregate inference only. It explicitly
rejects autograd; its presence does not certify training, FP16 or other Aggregate
contracts. Read (including state proposal preparation), output, journal and
state metadata kernels still use scalar AIV loops.
Measured placement and task costs, then complete-flow timing, determine whether
an implementation is beneficial at a given scale. Trace storage, CPU comparison
and result materialization stay separate from steady-state throughput timing.
FP16, other module contracts, public Python/native packaging, peer progress and
training remain independent delivery requirements.

The window-interface gate on clean `4e45072` ([evidence](evidence/device-window-20260930.md))
passes384 windows across four topologies,
two input/state variants, both schedules, all three linear Read modes and both
diagnostic settings. It checks three successive advances without downloading
state/history/pending, delayed complete-observable export, isolation after mutating
an exported CPU snapshot, and refusal of snapshots/re-entry after a failed window.
The separate lean trace covers192 windows plus one expected failure, with193 model
submissions and193 boundary waits. It records26378 AIV and360 AI Core tasks, no
journal task, no AiCPU task and no host-fallback diagnostic. It still includes
setup, explicit verification exports and CPU assertions;6044 ordinary stream
synchronization API calls are also recorded. All16 component cells pass on that immutable source. The trace is placement
evidence, not throughput.

The clean identity-Full qualification remains scoped to source `4d2f09e`
([evidence](evidence/content-loop-20260930.md)). The selected matrix Full increment
is qualified on clean `5bf61e3` ([evidence](evidence/selected-full-20260930.md)):
24 component cases at widths 1/7/33 and chunk
limits 1/4, plus 160 window/continuation comparisons against independent CPU
Streaming and Greedy. The latter cover identity/tanh Full, feedback, unequal
delays, parallel edges, active-only adoption, selected clear and schedule changes.
The component placement trace has 410 AIV and 16 AI Core tasks, with no recorded
AiCPU task or host-fallback diagnostic. It includes setup and CPU assertions,
not steady-state timing. Qualification covers all 14 component cells; exact
source, jobs and reports are in the evidence manifest and [STATUS](STATUS.md).

The later old/proposal/mixed Read increment is qualified on clean `06db0c2` ([evidence](evidence/runtime-read-20260930.md)) with coverage of
640 windows, 3252 events and 2012 actual emitted messages against independent CPU
Streaming/Greedy. There are 66 multi-time windows, including 38 with state-Read
time batches, and causal-region batches never exceed one time per node. Ready
packing also passes FP32/FP16 checks; the numerical flow remains FP32 only.
The clean trace records 914 state-Read AIV tasks and exactly one model
submission/boundary wait for each executed window. No AiCPU task or host-fallback
diagnostic was found. The trace also includes setup and diagnostic stream synchronizations; it is not
throughput evidence. Earlier immutable evidence remains scoped to its sources.
