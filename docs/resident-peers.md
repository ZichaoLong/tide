# Device-loop peer packets

`tools/device_online/peer_exchange.h` is an internal CANN transport primitive.
Its build and hardware qualification are recorded in [STATUS](STATUS.md).
It does not yet expose a multi-device graph or training owner. The historical
[finite capture transport](bounded-scheduler.md) has a separate scope.

A packet consists of fixed-capacity contiguous tensor fields on two different
NPUs. Fields may carry payloads,exact int64 coordinates/counts,bool presence or
int32 control. FP32/FP16 payloads move without conversion. Offsets into contiguous
storage are valid;empty/noncontiguous/mismatched fields,overlapping destinations
and an exceeded byte budget are refused before creating notifications.

The source program records readiness and waits for consumption. The destination
program waits and resets readiness,pulls every field on its stream,and records
consumption. The source resets that acknowledgement before it may overwrite or
reuse its storage. A direction uses two logical notifications and their imported
handles. The same pair can be revisited in device control flow without allocating
a notification per event. Packet fields are copied in a fixed construction-time
loop;there is no host loop over runtime events.

Both program sides must contain matching protocol calls in the same order.
A device-generated command may terminate a service loop;the sender must send
that command even when a window contains no work. The receiver must acknowledge
the terminal packet and exit without executing payload computation. Capacity
splitting and graph semantics belong to the caller,not the byte transport.

Top-level code finishes input writes,submits one program per device,and then waits
for all complete-window boundaries. `CannProgram::submit()` and `wait()` stay on
the constructing thread,whose runtime contexts own the models;cross-thread use
is refused before submission. All peers are submitted before waiting for any
peer. No host inspects intermediate scalars or controls iteration counts. Peer wait and boundary
timeouts are bounded execution failures,not proof of completion. A failed graph
owner must not return a completed result.

Programs retain shared ownership of the packet,notifications and both endpoint
buffers through their commands. Explicit packet close refuses while a program
retains those commands. `CannProgram` drains before clearing them;an unconfirmed
drain quarantines its retained resources until process exit. Notification cleanup
failure likewise quarantines the packet instead of invalidating possibly live
handles. Caller code must retain normal endpoint tensor ownership and follow
the same no-mutation rule as other resident borrowed views.

The standalone `--checks peer` gate explicitly requires two visible logical
NPUs. It is excluded from the default single-device gate. Its independent CPU
state evolution covers data-dependent loop counts,empty windows,bounded work and
continuation,keys above2^55,bool masks and FP32/FP16 values. This is a transport
prerequisite;complete graph observables,VJPs,optimizer/continuation and throughput
need their own multi-device qualification under [the execution contract](execution-flows.md).

## Internal remote Full inference

`ContentFlow(..., coordinator, limits, full_device)` is an internal experimental
inference entry point. Its public single-device client defaults remain unchanged.
The coordinator owns online readiness,selection,state,attention caches,emission
and event queues. The peer owns and executes the declared identity/tanh/LH/SwiGLU
Full banks. This is a first graph integration,not general owner/parameter/state
sharding or distributed training. Current qualification is in STATUS.

Actual selected packed actions,content,comparison and sticky error travel in a
fixed request packet. The peer runs the same packed Full implementations and
returns values,error and chunk counts. The coordinator's compiled program then
continues emission and queue commits. Empty and failed windows still send a
device terminal packet so the peer service exits. Both models are submitted
before either window wait;there is no host dispatch between graph stages.

Packet endpoint buffers are included in the conservative total memory estimate.
The remaining operator workspace is divided between both serial arenas;exports
report their sum and the peer's fixed packet size. Full capacity is transferred,
including unused rows;only actual selected actions are computed. This first
integration does not claim minimized communication or parallel stage overlap.
Actual work,packing and throughput remain measurement questions.

No Full reverse tape is exposed for remote placement. A caller requesting it is
refused explicitly;single-device tapes/optimizer publication keep their existing
behavior. Cross-device VJPs and master publication need a separate implementation
and independent qualification before this placement can support training.

## Compact Full shards

The internal `ContentFlow(..., FullPlacement)` overload assigns every node to
exactly one explicitly selected logical device. `place_full(..., "memory" |
"locality")` constructs a deterministic topology-only plan. Locality refinement
counts physical parallel edges separately, accepts strict cut reductions and
does not exceed the initial peak parameter estimate; balanced swaps allow equal
size nodes to move. An explicit owner vector is also accepted. Invalid owners,
duplicate devices and empty shards are refused before device allocation.

Each device receives only its assigned Full bank rows. The coordinator's device
planner packs actual selected actions in stable order, maps global node IDs to
local bank rows and emits gather/scatter indices. It sends all nonempty peer
requests before local Full work or any peer response wait. Responses scatter to
distinct original rows; unused rows have private scratch destinations. There is
no whole-model compute-and-mask path and no per-event host packing. Independent
peer stages can overlap; actual overlap and speedup require profiling/measurement.

An empty shard stage sends no work request; every peer still receives the window
terminal command. Per-shard errors merge without clearing an earlier error. Packet
fields retain fixed capacity and include padding. Exports report selected/capacity
rows, parameter bytes per shard, actual physical chunk limits and summed program
workspace; the total tensor admission includes both packet endpoints and scratch.
The same complete continuation can be recreated with another valid Full placement.

This is Full parameter placement and computation. Read, state/KV, readiness,
selection and queues remain coordinator-owned. It is not complete model sharding,
distributed training, a public multi-device client or a throughput qualification.
Current development/qualification status is recorded in STATUS.

## Compact Full reverse and retained windows

`ContentFlow::sharded_reverse_tape()` explicitly borrows the actual coordinator
journals and the compact Full banks on their forward owners. Ordinary
`reverse_tape()` still refuses sharded placement. The coordinator view has no
Full kind bank and the single-device reverse API rejects it without the explicit
sharded executor, preventing an accidental identity-Full backward.

`retain_sharded_reverse_tape` admits the sum of coordinator records and owner
banks before copying. Each numerical bank stays on its NPU; static node IDs and
graph structure are host metadata. Retention preserves parameter versions and
allows backward after forward close or subsequent window overwrites. Existing
window bridges connect pending messages, state and attention caches on device.
State/KV and their reverse computation remain on the coordinator.

`append_sharded_graph_vjp` uses the ordinary graph reverse stage loop with a
Full-stage executor. One device packing kernel stably collects actual connected
rows, translates global node IDs and creates gather/scatter indices. Fixed-size
packets carry the compact journal, FP32 cotangents, exact counts and connection
bits. Every nonempty remote request precedes local Full reverse and response
waits. Peers run the existing FP32/FP16 Full VJPs and accumulate per-node parameter
partials on their own NPU. Only content/comparison adjoints, connections, errors
and chunk counts return per stage. Parameter matrices do not round-trip per stage.
Padding has distinct scratch destinations and cannot create gradient connections.

Each retained window has its own peer service programs. Submit all programs on
the constructing thread before any boundary wait. Empty/error windows still
terminate every service via a device command. Replay resets per-window partials
and work counters; capacity and malformed-tape errors remain explicit. Parameter
partials are physical Full-node contributions, not independently trainable copies
of shared parameters. Canonical owner reduction and atomic optimizer publication
are separate phases described below.

The standalone `--checks peer-sharded-vjp` gate requires two visible NPUs and is
excluded from default single-device checks. Its checker also accepts
`--full-shards=1|2|3|4` and `--full-placement=memory|locality`. It reuses independent
CPU FP32/FP64 retained-graph oracles, both payload precisions, None/zero roots,
normalized Aggregate, mixed Full, control modes and attention cache fixtures.
The current checker downloads completed canonical device gradients for assertions;
no CPU sum feeds a candidate. Earlier evidence retains its original partial-only
scope. Passing source-specific evidence is recorded separately in STATUS/ROADMAP.

## Canonical owners and complete internal training steps

`sharded_parameter_sources` maps the actual retained reverse results to the
registry's TensorImpl owners, in reverse-window and alias order. It reads static
shapes/identities only. `ShardedParameterReduce` places canonical owners by
stable largest-first FP32 storage cost, independently of Full-node placement.
Each contribution ordinal groups independent owners by device pair. A CANN
loop packs, transfers and consumes a bounded FP32 packet, reusing its storage
until that group is complete. Only then can the next contribution ordinal
accumulate into an owner. The receiver therefore preserves declared floating
addition order, without atomics or grouping all contributions by source card.
None payloads are not read; connected zero remains an observable connection.

Static gather/publication descriptors contain addresses of retained tensors on
the same NPU. They are not checkpoints and never directly name remote memory.
Cross-device copies use `PeerExchange` and a common device-pair order. Each
canonical gradient/master is stored once; aliases on other forward owners do
not acquire independent optimizer states. Groups are validated globally before
partitioning, including aliases and explicit empty subsets. Distinct TensorImpl
owners sharing storage are refused across the entire optimizer, not just within
one partition. Empty registries and devices without trainable owners are valid.

`DeviceOptimizer` separates proposals from commit. Every owner first checks
its update, slots, int64 counters and payload representability. Device error
consensus then broadcasts one decision before any master, slot or counter can
change. After consensus the numerical kernel recomputes from frozen inputs;
no full parameter/slot proposal banks remain. Counters commit after all value
tiles finish, preserving SGD first-use momentum. A numerical refusal updates
no card. This is not a distributed recovery
protocol for hardware/runtime failure; a failed runtime must be discarded.

`append_publish` packs only masters used by each receiving card and publishes all
its aliases after the common decision. The explicit `sharded_parameter_banks`
view includes compact Full rows, coordinator state/Read/scales, strided event
Q/K/V and same-fiber parameters. HARD Read aliases are updated even when their
own Read use has no VJP. FP32 masters/slots persist; FP16 banks receive rounded
values, and FP32 normalization banks receive that rounded value widened again.
Publication is gated by the same error, including None/zero and half-overflow
refusals. No CPU parameter export/reconstruction participates between updates.
Publication also reuses bounded packets and writes directly to local alias
views, including strided Q/K/V columns and incomplete final packets. Each
packet endpoint holds at most 64 MiB of FP32 values and shrinks to fit its
share of the tensor budget. Descriptors, flags and both endpoints are admitted
together; this is not a total per-card HBM admission policy. Canonical gradient
outputs are stored once. Optimizer identity checks keep owner metadata without
retaining the original gradient bank; training layout geometry shares master
storage instead of keeping a dummy FP32 gradient allocation.

Public reverse statistics include `canonical_stream_reserved_bytes` (both
endpoints and metadata allowance) and `canonical_stream_chunks` (planned device
packet iterations). They exclude caller-owned tensors, operator workspaces and
vendor allocations, and do not claim measured peaks or active numerical work.
The separate `--checks peer-owner-stream` gate exercises odd packet capacities,
strided publication/FP16 rounding, poisoned None payloads, connected zero,
sticky errors, replay and order-sensitive canonical accumulation on two NPUs.

All tensor groups have explicit admission budgets; per-program operator arenas
have separate bounds. Exceeding capacity fails before large buffer allocation.
Replay replaces previous partials and error decisions rather than accumulating
stale gradients. Host construction/submission and complete-boundary reporting
are allowed; no numerical result controls a host branch inside these programs.

`--checks peer-sharded-optimizer peer-sharded-training` requires two visible
NPUs, is excluded from default single-device checks, and tests FP32/FP16.
Executables additionally accept `--full-shards=1|2|3|4`; the training checker
accepts generic memory/locality Full placement. The optimizer gate tests global
finite/overflow rejection, shared-owner groups, None/zero and empty partitions.
The training gate executes four updates, each retaining four real windows,
against independent CPU FP32/FP64 references and their own continuations. It
checks gradient/slot/counter parity, exact master-to-bank publication, and that
a refused update changes no live bank. Profiling smoke is a separate subset,
not throughput evidence.

This closes an internal Full-sharded training mechanism once qualified. Online
forward scheduling and state/KV still reside on the coordinator. Whole-model
state/KV placement, public multi-device training/checkpoint clients and medium/
full-size consumer throughput remain separate work under the execution contract.

## Compact state, Read and KV owners

`ContentFlow(..., ModelPlacement{full,state})` adds an internal forward placement.
Each plan names explicit logical devices and a static owner for every node. Full
and state ownership can differ. The coordinator retains queues, region history,
global selection, Aggregate and emission; state values/clocks, Read/Upd banks
and persistent event/fiber KV live only on their assigned owners. Static source
identity metadata is replicated; there is no complete coordinator state/KV bank.

`StateKernelProfile` is an explicit kernel view of an already validated graph.
Its compact node IDs follow global node order. It is never compiled as a Graph:
physical ports/parallel edges, logical source slots and region identity retain
their global meaning. `state_shard_initial` projects only the common initial
state; candidate events, routes and selection never come from the CPU reference.

The device program packs whole ready fibers and their atoms, preserving their
order and exact int64 coordinates. Numerical payloads use batched gather;
padding has separate zero source/discard rows and cannot overwrite real rows.
Exceeding capacity refuses the proposal instead of splitting an attention group
or publishing a partial fiber. No host loop places individual runtime messages.

Each remote owner follows three device-controlled phases:

1. Receive the current packed work; propose attention/Read and return scores.
2. Receive the global region selection; propose adoption/clear and return
   comparison values plus event observables, restored to global row identities.
3. Receive the common decision after queue/emission/journal preflights. Commit
   state and KV only on success, acknowledge completion, then await more work.

All peers are submitted before the host waits for the complete window. Stop
commands terminate empty and refused windows too. Snapshots explicitly export
owner states/caches at a complete boundary and may restore with another layout;
exports never feed normal window progression. KV stage copies stay on their
owners. Current communication is bounded capacity-sized packets, so padding
and per-phase communication costs remain visible optimization targets.

`peer-state-flow`, `peer-state-control-flow` and `peer-state-transaction` are
explicit two-NPU gates, both FP32 and FP16. They cover independent CPU forward
comparison, repartitioned continuation, empty/error termination, exact int64 and
physical-edge packing, whole-fiber capacity rejection, and unchanged state/KV
bytes after a downstream refusal. A profile smoke is separate from throughput.
Distributed attention counters sum work; chunk/key rows and KV peaks report the
maximum across owners. State tensors and peer packets have separate byte counts.

This placement still refuses the old monolithic reverse/state-view interfaces.
The separate compact reverse integration below has its own development and
qualification scope; forward evidence alone does not certify its training.
Graph/checkpoint identities and the CPU reference are unchanged.

## Compact state/cache reverse and internal training

The implementation adds `StateOwnerTape` fragments to `ShardedReverseTape`.
They contain local parameters and actual KV records, without a replicated global
state bank. `StateReverseLayout` is a static kernel view, never a locally compiled
Graph. Retention clones numerical records on their original devices and preserves
parameter versions. Grouped fiber gathers have shape-only admission before their
allocation; retained tape bytes are checked separately before cloning.

Reverse preparation packs actual coordinator event/source journals by owner on
device, preserving stable order, global physical source identity and row maps.
Per-stage packets carry cotangents, explicit None flags and visible-state carry.
Owners execute state, event/fiber KV and Read VJPs, retaining cache adjoints and
parameter partials locally. Returned content/source adjoints join the coordinator
routing and physical-scale reduction. Fiber source derivatives add to direct
Aggregate derivatives; neither contribution replaces the other.

HST/SOFTP Emit and softmax derivatives use the complete global region frame.
Only the resulting per-event score derivative moves to the Read owner. Partitioning
nodes must not partition a region's normalization or invent gradient connections.
FP16 keeps actual forward payloads and FP32 cotangents/accumulators.

Retained windows explicitly bridge local cache adjoints in reverse order. Their
owner/group layout must match; a layout change inside a retained backward is
refused. Complete-cut forward snapshots may still restore with a different
layout. Empty/error windows send stop and completion packets. Repeat execution
resets accumulators and cache seeds; no host event polling drives backward.

The retained composition uses `CannSequence`: one coordinator program per window,
joined in reverse-window order by reusable local device notifications. It submits
all programs and peer services before waiting at the complete backward boundary.
This bounds each persistent stream's static task list without truncating gradients
or returning intermediate values to the CPU. The sequence has an explicit program
capacity; individual programs and graph buffers still have finite vendor/tensor
limits. Logical event queues and static runtime task storage are different limits.
On the local CANN9 stack, adding the vendor HUGE flag did not remove a demonstrated
three-owner retained-program task-buffer failure; no such fallback is claimed.

`sharded_parameter_sources` maps local state/Read/event/fiber partials into the
existing canonical alias reducer. `sharded_parameter_banks` publishes back into
actual forward storage on each owner, including grouped event projections and
original fiber banks. The existing all-device atomic optimizer decision governs
these publications as well as Full banks. No CPU gradient or parameter copy is
used to drive an update.

The retained VJP and complete training checkers accept `--state-shards`, with
Full and state owners deliberately different when more than one device is used.
They compare independent CPU FP32/FP64 trajectories, cache/input/state gradients,
None/zero roots, actual aliases, optimizer state, exact master publication,
continued windows, retained snapshots after forward close, and repeated backward.
Current implemented/verified status belongs to STATUS and source-specific evidence.
Public multi-device training/checkpoint clients and throughput remain separate
F4–F7 obligations; this internal interface is not that public delivery.
