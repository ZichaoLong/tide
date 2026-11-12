# Resident complete-region KV graph blocks

This explicit concrete executor composes the pinned core-5 Aggregate → KV Upd →
Read/Describe → complete-region allocation → selected Full → physical messages.
See [mathematical mapping](semantic-map.md), [sequence](resident-block-sequence.md)
and [block dispatch](resident-block-execution.md). It does not redefine graph or
checkpoint semantics.

`append_block_settle_flow` accepts aligned Settle `ports` graphs with complete
region observation, adopt state, broadcast emission and the original logical
clock. Origins, clearing/identity overrides, alternate Next/emission and partial
region observations are rejected. Each region binds concrete KV and regional
configurations and shape-compatible private state, independent of generic node
function-name strings. Edges and ports use unit transport of `(c,G)`; the Model
adapter must verify actual transport parameters before selecting this executor.
These conditions are an explicit specialization domain, not restrictions on the
general PDG/TimedDAG/Settle engines.

Input payload/presence is `[B,L,input_port,d+1]`/`[B,L,input_port]`; positions are
original int64 `[B,L]`. Per-region raw initial and bound Read0 state are separate.
Physical atoms retain edge/port identity even if parameter classes coincide.
Each device-chosen task evaluates a complete region across B×L, including missing
fibers; empty fibers keep private state. Device choose/commit drives the next
region. The coordinator records one call per region to a bounded numerical
program. CANN request/completion notifications keep these calls device-controlled;
CUDA uses the same phases in its conditional graph. Forward and reverse have
separate regional programs, while the reverse coordinator still consumes the
actual forward journal. No host-selected real trajectory or per-token dispatch
is consumed. `TaskServices` supplies only program lifetime and ordered calls,
not another graph schedule or the earlier WeightedFlow journal.

The caller owns `DeviceProgram` and `TaskServices`: append graph/roots, append
the service stops, finish workers before the coordinator, submit workers before
the coordinator, then wait for both before publishing or reuse. Tensor budgets
count unique storage across all participating programs. A notification-serialized
group shares one maximum vendor workspace; independent retained windows own
separate arenas. Device execution failure retains the normal quarantine policy.
`TaskServices::statistics()` distinguishes retained program owners, explicit
streams, CANN notifications and CUDA boundary events; CUDA does not create worker
programs for these phases. The [session inventory](resident-block-session.md#storage-admission)
aggregates the actual retained groups and checks planned storage before recording.

Outputs contain actual node, physical-edge and output-port payload, presence and
int64 time columns. Saved local tapes expose Aggregate, both Mem channels,
Describe, controls, Full and old/proposal/final raw fields. The reverse consumes
the actual forward journal and accumulates each physical use and canonical shared
parameter. Independent root masks distinguish unused and connected-zero values.
The derivative is first order; current payload/adjoint arithmetic is FP32.

The raw sequence tape currently materializes per-frame state snapshots. Explicit
byte budgets reject excess storage; this is not a shared-prefix-bank memory claim.
Owner publication, optimizer, retained windows, checkpoint and public automatic
lowering are separate layers, with qualification recorded separately.

Implementation: `tools/device_online/block_settle_flow{,_vjp}.cpp`,
`block_kv_local{,_vjp}.cpp`, and `block_settle_parameters.cpp`. Physical kernels
are in `ascendc/block_settle_flow_math.h` and `block_flow_reverse_math.h`.
`tests/test_resident_block_flow_layout.py` checks physical identities, missing/zero
messages, huge clocks, actual journals and root connectivity through a CPU harness.
`block_settle_flow_check.cpp` checks device execution against the independent
scalar-time `block_settle_formula.cpp`: nonempty initial state, three Read
snapshots, duplicate edges/ports, shared parameters and ten independent root sets.

Reproduce with `build_device_control.py --checks resident-block-settle-flow`,
then `verify_device_control.py --checks resident-block-settle-flow` on an allocated
FP32 device. CPU harness, build, device execution and scale are distinct evidence.
No component result alone certifies FC16 performance or complete resident delivery.
