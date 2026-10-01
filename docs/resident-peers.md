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
of shared parameters. Shared-owner reduction and atomic multi-device optimizer
publication are subsequent, separately qualified responsibilities.

The standalone `--checks peer-sharded-vjp` gate requires two visible NPUs and is
excluded from default single-device checks. Its checker also accepts
`--full-shards=1|2|3|4` and `--full-placement=memory|locality`. It reuses independent
CPU FP32/FP64 retained-graph oracles, both payload precisions, None/zero roots,
normalized Aggregate, mixed Full, control modes and attention cache fixtures.
The comparison adapter downloads completed partials and sums aliases only for
assertions; these CPU values never feed the candidate or an optimizer. This gate
therefore verifies retained graph VJPs, not device alias reduction, multi-device
training steps, a public training client or throughput. Passing source-specific
evidence is recorded separately in STATUS/ROADMAP.
