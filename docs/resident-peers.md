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
