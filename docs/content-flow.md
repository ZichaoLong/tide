# Experimental content-driven device flow

`tools/device_online/ContentFlow` connects the device queue, readiness, selection,
state and delivery stages into a single submitted forward loop. Implementation
and qualification status belong to [STATUS](STATUS.md) and [ROADMAP F4](ROADMAP.md).
Its finite module scope does not close the complete execution-flow contract.

The accepted profile uses existing semantics: sum Aggregate with physical source
scales; identity, EMA, Add-repeat, event-GQA or LH fiber attention state; content/old/proposal linear or FP32-norm Read; count-v1 or positive-v1
selection; adopt-v1 Next with optional selected clear; identity, tanh, SwiGLU or LH
Full with HARD broadcast or slot-affine emission. Tanh is `content + tanh(comparison @ weight + bias)`; the nine LH Full profiles
apply their declared activation/normalization to comparison without that residual.
It supports observe-all and active-only state adoption. Inputs are arbitrary
legal sealed-window values; topology can contain unequal positive delays,
parallel physical edges, feedback and disconnected components. Input-origin
projection and phase-restricted slot emission are supported; other unavailable
modules/region programs explicitly fail capability validation. This first version accepts FP32 inference with an
explicit no-grad scope. It has no VJP or optimizer contract.

The bounded `lh-fiber-attention-*-repeat-v1` adapter keeps persistent key/value/
log-bias tensors and int64 lengths on the device. Static tables contain only actual
attention owners. Device planning orders the real source-weighted message rows by
local slot; it never substitutes Aggregate summary content for those rows. QKV and
output projections are bounded batch matmuls. Device-generated gather indices pack
each query with its complete proposed cache; head layouts, log biases, softmax and
value products execute as CANN tensor operations. Every current query sees all old
keys and all keys in its fiber. Physical query chunking does not add a triangular
mask or change the global softmax denominator. The full-key path pads the key axis
to the declared cache capacity; the separate tiled path described below bounds
its physical key scratch without changing the retained cache.

The five existing [post-attention pooling profiles](fiber-pooling.md) share this
QKV/cache path: sum, mean, linear, active-softmax and all-softmax. Coefficients
apply only to the completed query output rows, before the output projection and
its once-per-event bias. Mean counts present logical sources; active-softmax
normalizes only their logits, while all-softmax includes absent logical slots.
Physical aliases do not enlarge that logical domain. A zero-valued source or
zero linear coefficient still contributes its full Q/K/V row. Negative linear
weights are supported. Aggregate and its observable contributions remain unchanged.

Softmax pooling packs only actual events of the requested profile in a device
loop. Logits and coefficients use bounded buffers; absent slots use negative
infinity, while independent padding rows have their own finite denominator.
Actual local slots determine gathering and coefficient placement on-device.
Pooling reservations are included before choosing the effective attention chunk.
Sum-only graphs omit the additional pooling stage. These inference paths use
the same parameter/profile validation as the independent core reference.

For observe-all fiber regions without selected clear, the ready planner may expose
the entire certified node-time prefix. This is decided online from actual messages
and topology closure. Each owner's KV arena appends every real message once;
each event retains its own prefix length and bias row. One vector task per owner/
cache tile follows that owner's time sequence and repeats the declared subtraction
for each exact local tick. Newly appended biases start at zero. The scratch bias
rows are included in the shared budget before selecting physical chunks.

QKV, queries, pooling and projections process the actual packed rows in bounded
device loops. A query sees all keys in its own fiber plus earlier fibers, never a
later fiber; neither query chunk boundaries nor key tiles alter that visibility.
Only the last event's cache is committed per owner. Optional journals retain every
intermediate old/proposed cache and bias, including before a selected clear.
Selected-only adoption or selected clear keeps complete single-frame fallback,
including with content Read. This capability restriction does not reject a legal
topology/input; other regions keep their legal prefixes. The event adapter's
separate compact sliding-window contract is described below.
The proposal is computed before Read/selection, then adopted only under observe-all
or actual selection. Selected clear empties the persistent cache while preserving
the pre-clear comparison. Bias decay repeats subtraction for each exact local tick;
empty old caches do no decay work. Cache row count and observation count are distinct.

`kv_rows` bounds each attention owner's simultaneously retained cache, including its
complete next fiber; it never evicts or drops messages. Exceeding it fails with code11
before a live-state commit. `attention_chunk_rows` bounds physical message/query/
output and softmax-pooling event rows; the shared byte budget can reduce it. All
module minima are reserved before physical chunks grow, using the same footprint
formulas as their constructors. `attention_chunks` includes
executed pooling chunks as well as QKV/query/output chunks. The effective
row limit and `attention_kv_peak`/`attention_kv_capacity` expose actual work/capacity.
`kv_trace_rows` separately bounds optional old/proposed cache diagnostics per window.
A single stage that exceeds its diagnostic buffer fails with code12; cumulative
journal capacity retains the shared journal refusal. Lean execution omits these
copies. Explicit snapshots include exact key/value/log-bias slots; the authoritative
device cache survives windows that skip host exports. Initial capacity, work-limit
and runtime refusals preserve the existing restore-after-failure contract.
FP16 and resident backward remain separate capabilities and are rejected by this
forward-only adapter.

The separate `memory="attention"` adapter implements the `event-gqa-v1` semantics.
It consumes the Aggregate summary and appends
one KV observation per actual event. Static packing groups share `(query_heads,
kv_heads)` geometry and retain compact `[length,kv_heads,head_width]` caches.
Device metadata chooses actual query chunks, maps consecutive query-head groups
to their KV head, and retains the declared sliding-window tail before appending
the current row. `window=0` has no semantic eviction; a positive window includes
the current observation. Idle logical-time gaps never consume window slots.
The declared `kv_rows` capacity never silently shortens that window.

Event and fiber attention may coexist. Their actual proposal rows overwrite
their own ready indices; scratch arrays are not added together because unrelated
rows can retain values from earlier device iterations. Each event group preserves
selected-only adoption, clear, pre-clear comparison and continuation independently.
Compact snapshots
and optional journals expose the two event slots, without a fiber log-bias slot.
`kv_trace_rows` bounds diagnostics separately per static event head group; their
combined reservations are included in the workspace estimate. Group minima are
reserved before fiber chunks expand. `event_attention_chunks`, the effective chunk
limit, KV peak and capacity are reported separately from the fiber counters.
These capabilities require their own completed qualification evidence; they do
not follow from the earlier fiber-only reports.

### Event node-time batches

When a region observes all candidate states and its event-attention nodes do not
clear selected states, the device can prepare a complete certified time prefix
before selection. The ready planner chooses that prefix online from actual queued
messages and topology closure. One device loop packs QKV for the actual node-time
events; a second loop packs queries and output projections. Their physical chunks
may differ from logical time batches. A time batch may cross many physical chunks.

The stage retains immutable original KV arenas and one compact new KV row per
actual event. Query metadata selects the appropriate causal prefix and optional
sliding window through gather indices. Later projected rows are present in scratch
but cannot enter an earlier query's denominator. Cache windows can slide across
multiple new rows even when the stage contains more events than `kv_rows`; the
bound applies to each logical retained cache. The queue separately bounds stage
events, and the shared memory estimate includes their compact QKV scratch.
There is no per-event copy of the complete cache. Only the final adopted tail is
copied back per owner after all preflights succeed. Diagnostics reconstruct every
old/proposed cache from the same immutable stage before commit.

Selected-only adoption and selected clear retain complete single-frame fallback;
fiber nodes sharing the region obey the same adoption/clear restrictions. These are
module capabilities, independent of fixture topology and input values. Other
regions retain legal time batches, and every legal positive-delay topology still
works, including feedback. `max_node_time_batch` and `device_stages` distinguish
the actual schedule; `event_attention_chunks` counts both executed projection
and query/output chunks. Neither counter is a throughput claim. Full-key and
tiled-key paths share the same prefix/window addressing and compact continuation.

### Shared forward allocation budget

`workspace_bytes` bounds the planned forward buffers and retained CANN operator
workspaces together. CPU validation prepares the static profile before uploading
device payloads. The planner reserves topology/queue/state/journal estimates and
the minima for emission, SwiGLU, LH Full, fiber attention, event attention and tanh
Full before allocating larger physical chunks. Unused module allowances become
available to later modules. An impossible minimum is refused before these uploads;
it does not trigger repeated OOM attempts or alter logical capacities.

`chunk_policy=conservative|aggressive` retains respectively25% or10% of surplus
after mandatory minima and a4096-byte operator minimum. A further share is reserved
for CANN workspaces; unused chunk allowances also remain available to them.
Aggressive allocation therefore permits larger chunks within the same declared
budget. Queue/output capacities also cap the requested physical Full/emission rows.
The program queries each CANN workspace requirement and checks it against the
allowance before allocation. Numerical tasks execute in order on one stream and
reuse a single arena sized to the largest requirement; workspace reservations do
not sum the disjoint operator lifetimes. Refusal poisons construction: an incomplete program cannot
be finished or executed. These policies affect physical allocation, never topology,
actual selection, dtype, KV visibility, time or checkpoint semantics.

Boundary statistics expose requested/usable budget, planned buffers, CANN workspace
allowance/use, remaining planned headroom and effective chunk policy/row limits.
`retained_tensor_bytes` counts unique tensor storages referenced by the program;
aliases and views are counted once. It is not a peak-memory measurement. The memory
gate separately records TorchNPU allocator peaks, including construction, for its
finite mixed-module fixtures. Allocator reservation can include cached unused blocks.
Neither counter observes all vendor/driver internal allocation. The plan currently
excludes caller-owned device tensors, training and communication buffers; it is not
a free-HBM guarantee or the complete training-memory planner required by F4/F6.

### Physical key tiles

`attention_key_rows` is an upper bound on gathered key rows per query, independent
of `kv_rows` and of the logical visibility/window. When the effective key bound
covers `kv_rows`, the existing complete-key CANN softmax remains selectable.
Otherwise both fiber and event adapters use a shared device loop. AIV metadata
selects actual key intervals and compact GQA head indices; CANN batch matmuls
compute scores and weighted values. AIV vector work carries the running maximum,
denominator and unnormalized weighted sum across tiles, rescaling previous sums
when a new maximum appears. The final division uses the complete denominator.
Short owners and padding queries may have no keys in a later tile; these rows
contribute zero without evaluating an empty softmax or adding dummy denominator
terms. All keys of the current fiber remain visible to every query of that fiber.

The vector reduction adapter limits physical key tiles to256 rows. Requested
limits are upper bounds; the budget may further reduce the key/query size before
allocation. A cache plus one query/key row that still exceeds the budget is
explicitly rejected. This is scratch planning, not eviction, a numerical trace
prepass or a complete whole-model/training memory planner. Chunk policy never
changes KV capacity, dtype, cache persistence or optimizer boundaries.

`attention_key_rows` and `event_attention_key_rows` report effective limits.
Their respective `key_tiles`, `tiled_score_entries` and `tiled_padding_entries`
counters are int64 device counts reset per window. They count executed key-tile
calls and real/padded query-head-key entries only on the tiled path; zero tiled
counts on the full-key path do not mean attention was skipped. The existing
query/QKV/output chunk counters retain their meaning. Counters are exported at
the boundary and never drive execution. Changing physical key sizes on restore
does not alter the checkpoint state contract. Tiled floating reductions require
their own parity and placement qualification; no throughput claim is implied.

For `InputOrigin`, a static edge table declares the visible port and int64 position
stride. Device metadata preflight refuses an off-lattice position with code10
before any numerical work or transaction commit. Stable sorting of actual fiber
metadata creates Aggregate's projected source order. Scalar and vector sum share
that permutation, while contribution storage, physical receive scales, queue
coordinates, pending messages and emitted edge identities stay physical. Explicit
source exports apply the same view at the observation boundary; they never feed
candidate execution. Equal projected keys retain physical order and remain distinct
logical sources. A logical-source collision still fails with code2.

The `swiglu` Full profile preserves the existing formula
`content + (silu(comparison @ ffn_gate) * (comparison @ ffn_up)) @ ffn_down`.
The device packs only actual selected SwiGLU actions, gathers that owner's three
matrices and runs bounded batched matrix multiplication. Other Full profiles pass
through this stage unchanged. Parameter storage packs only declared SwiGLU owners;
unused graph nodes do not require dummy SwiGLU matrices. Padding uses independent
zero values/matrices and distinct scratch destinations. `full_chunk_rows` bounds
this profile too; `swiglu_full_chunk_rows` reports its effective limit, and
`full_chunks` includes its actual chunks. Its reserved workspace is deducted before
other Full planners choose limits. This remains inference only; no SwiGLU resident
VJP or training qualification is implied.

Emission uses static local-slot bindings, periods and physical edge/output scales.
The device preflights actual selected actions, decides each slot's presence from
exact int64 logical time, and packs only present slots. Empty phases mean all;
-1 means always, -2 means never; identity boundary adapters remain unconditional.
`slot_affine` computes `Full @ emit_w_slot + emit_b_slot`, then applies the physical
send scale. Unprojected Full and unscaled emitted-slot values have independent
diagnostic journals, including when a physical scale is zero. Missing slots create
no messages; a present numerical zero still does. No emission value is computed on
the host or reconstructed by dividing a message by its scale.

`emission_chunk_rows` bounds gathered affine matrices and projected rows; the
remaining workspace budget can reduce it. Actual device metadata chooses each
chunk and subsequent iteration. Padding reads independent zero sentinels and has
unique scratch destinations. Absent/inactive poisoned parameters never enter the
numerical operators. Broadcast uses gathered Full values without matrix work.
Physical message placement/scaling uses batched gather/multiply for both internal
edges and outputs. The complete stage is checked for arrival/output capacity and
actual int64 delay overflow before numerical emission; absent edges cannot cause
an arrival overflow. Emission storage is bounded by arrivals+outputs. Diagnostic
`trace` also bounds the number of recorded present slots, independently of events.
These stage limits remain explicit refusal boundaries, not the future complete
memory planner. `emission_chunks` and effective `emission_chunk_rows` are reported.

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

The explicit `norm-fp32-v1` Read uses FP32 sum-of-squares and square root of
the visible content/old/proposal vector. Each node declares its Read kind;
linear and norm Read may coexist. No descriptor is sent to the host for selection. This is
the FP32 profile from [read-programs.md](read-programs.md), not a replacement for
`norm-fp64-v1`; the latter remains rejected by this flow. Nonfinite scores refuse
the window. Rounded ties use the existing stable selector; precision changes
can change routes. This addition does not supply a resident norm VJP.

`ContentLimits.vectorized_read` defaults to true and retains the scalar device
path as an explicit alternative. Both run the same clock/work-limit preflight.
Vector Read assigns independent (owner,width tile) ranges to AIV blocks and uses
vector multiply/add/reduction on up to256 elements. Each tile follows the owner's
time sequence in local storage; no per-event state preparation runs on the host.
One final device task combines tile partials and takes a synchronized vector sqrt
for norm rows. The selector validates finite scores before committing any state.
Padded/absent rows and identity-node Read do not load unused parameters into the
arithmetic. This changes floating reduction order, so parity checks include exact
route/history agreement as well as the declared FP32 tensor tolerances.

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

`PackedLhFull` groups actual selected actions by their graph-declared Full contract.
Each group uses the same device chunk planner; actual counts and repeat decisions
stay on NPU. Batched CANN ReLU/SiLU, RMSNorm (epsilon1e-7) and LayerNorm
(epsilon1e-5) implement the nine [LH profiles](lh-full.md), including norm-only
Full. Per-owner normalization weight/bias rows are gathered and applied after
unit-affine normalization. Only real selected comparisons enter computation;
padding reads an independent zero sentinel and writes unique scratch destinations.
Each node can have a different profile, and identity/tanh nodes can coexist.
`full_chunks` counts both tanh and LH chunks; `full_chunk_rows` and
`lh_full_chunk_rows` record their distinct effective limits. Both share the forward
allocation plan above, which does not cover training memory. Slot-affine signaling is
handled by the separate emission stage; resident VJPs remain pending.

LH component precision checks use an independent FP64 activation/norm formula on
the exact FP32 input as well as the existing CPU implementation. Low-variance
LayerNorm amplifies input/mean rounding; the CPU FP32 result can itself differ
from FP64 by more than the usual absolute tolerance. For these isolated component
checks, both CPU and NPU must satisfy a declared engineering error budget:
`1e-6 + 1e-5*abs(y64) + 4*u*(1+ceil(log2(width)))*max(abs(x64))/s*abs(weight)*(1+abs(z64))`,
where `u` is FP32 epsilon, `s` is the normalization denominator, `z64` is the
normalized activation and `y64` includes learned affine values. This is a
finite-fixture estimate, not a universal error theorem. Reports retain the number
of ordinary CPU/NPU tolerance misses and each implementation's FP64 error.
Well-conditioned component rows and complete graph windows still require the
ordinary rtol1e-5/atol1e-6 comparison; all routing/discrete observables stay exact.
No runtime dtype or normalization formula changes are hidden by this test policy.

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
the captured program omits event/fiber/contribution/Full/emission/KV journals and the emitted
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
contracts. Read's scalar alternative, Read metadata/final partial combination,
emission metadata, journal and state metadata kernels still use scalar AIV loops.
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
