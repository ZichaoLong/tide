# Resident Settle block training owner

`BlockSettleSession` composes the [actual block graph](resident-block-flow.md)
with retained windows, independent first-order roots, canonical parameter owners,
AdamW/SGD and exact raw checkpoints. It is a new owner, not a wrapper around the
older `WeightedFlow` or its `TaskServices` journal. Implementation and device
qualification remain separate; this contract does not transfer earlier owner
or FC16 qualifications to the new path.

## Boundary and lifecycle

The constructor binds an aligned Settle ports graph, homogeneous KV geometry per
region, complete numerical-role-to-canonical-name mappings, CPU FP32 parameters,
private raw initial rows, a separate fixed Read0 initial binding and an explicit
accelerator. Canonical identity comes from `ParameterRegistry`, including aliases;
reshape views of a bank row never define new parameter identities. Equal values
with distinct owners remain distinct. No-grad is required for all owner calls.

`advance(positions, payload, present)` accepts int64 `[B,L]` original positions,
FP32 `[B,L,input_ports,d+1]` physical `(c,G)` payloads and bool presence. Both B and
L flow through the numerical block. Positions must increase independently within
and across each sample's declared windows. Missing messages and present zero
content remain separate. Inputs are frozen; actual state is adopted only after
successful device choose/numerics/commit completion. A failed device execution
poisons the owner and requires restoration of a previously saved cut.

`backward(roots)` receives all retained windows in forward order. It reverses each
actual device journal in reverse window order, transposes physical atom routing,
and carries the next window's raw initial adjoints into the previous window's
raw final roots. Independent old/proposal/final, Read memory/totals, Describe,
controls, Full, received content and physical edge/port roots retain their own
connection masks. Fixed Read0 initial contributions are summed across all windows.
Unused and connected-zero contributions remain distinct. Source inputs and
private states stay private; all shared parameter uses reduce once into canonical
owners before the optimizer boundary.

`step()` evaluates finite optimizer proposals, commits slots/masters, publishes
every actual role and advances the parameter generation. Detached state survives;
old tapes are consumed. `detach()` preserves state while discarding outstanding
tapes and gradients. A checkpoint requires a detached complete window boundary.
The construction, backward and optimizer partitions and a complete retained-tensor
bound are explicit. The optimizer partition is conservatively reserved as well
as its visible fields; workspace has its own bound. Retained-window capacity is
simultaneous storage capacity, not cumulative logical-time capacity.

## Checkpoint identity and exactness

The CPU record contains full raw floating and int64 buffers, fixed Read0 raw
bindings, independent sample positions, canonical aliases/named values, packed
masters, every optimizer slot/counter/correction, generation and next token.
Its identity includes graph/ranks/stride, all KV/regional options, source and node
owner tables, role mappings, shapes and alias partitions. Restore validates this
identity and CPU tensor layouts before exposing a fresh owner. Effective state
projections are never substituted for actual raw state.

The standalone archive functions serialize these actual fields only. They use a
fsynced temporary archive and atomic rename. Loading an archive does not execute
a reference recurrence. Runtime device/driver errors during restoration fail
construction; callers never receive a partly restored owner.
Save/load have an explicit positive int64 `max_bytes` (default 4 GiB). Save
rejects excessive tensor bytes before writing, then checks actual archive bytes
before atomic replacement; load checks file size before deserialization. This
is a file budget, not a bound on every temporary codec allocation. Refusal keeps
an existing checkpoint intact.

## Navigation and checks

- Owner and guards: `tools/device_online/block_settle_session{,_layout}.cpp`.
- Cross-window raw transpose, canonical reduction and optimizer publication:
  `block_settle_session_training.cpp`.
- Exact archive: `block_settle_checkpoint.cpp`.
- Independent scalar causal formula, including linked-window autograd:
  `block_settle_formula.cpp`.
- Actual standalone consumer: `block_settle_session_check.cpp`, three AdamW
  steps, two retained windows per step, shared owners and exact fresh-owner
  restore. A first process writes `--checkpoint-path PATH`; a fresh process adds
  `--restore-check` to compare its actual final record to `PATH.expected`.

The current concrete owner is FP32 and single device. General PDG lowering,
other local-memory families, cross-device ownership and public Model/default
selection are separate integration boundaries, not implied by this owner.
