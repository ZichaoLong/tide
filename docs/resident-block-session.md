# Resident Settle block training owner

`BlockSettleSession` composes the [actual block graph](resident-block-flow.md)
with retained windows, independent first-order roots, canonical parameter owners,
AdamW/SGD and exact raw checkpoints. It is a new owner, not a wrapper around the
older `WeightedFlow` or its `TaskServices` journal. Implementation and device
qualification remain separate; this contract does not transfer earlier owner
or FC16 qualifications to the new path.
The [fixed a3c report](evidence/resident-block-public-20261112.md) records the
Python-owned and standalone SDK small-NPU sum/qry training and fresh-process
restore checks, alongside separate FC16/LH176 scale capacity refusals.

## Boundary and lifecycle

The constructor binds an aligned Settle ports graph, homogeneous KV geometry per
region, complete numerical-role-to-canonical-name mappings, CPU FP32 parameters,
private raw initial rows, a separate fixed Read0 initial binding and an explicit
accelerator. Canonical identity comes from `ParameterRegistry`, including aliases;
reshape views of a bank row never define new parameter identities. Equal values
with distinct owners remain distinct. Source functions retain original A/B,
embedding and optional low-rank factors; effective matrices are computed on the
device and the VJP returns every original factor contribution. Different source
hidden widths/ranks use declared zero-padded banks and prefix-view publication;
zero padding never becomes a new parameter owner. No-grad is required for all owner calls.

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
The construction, backward and optimizer partitions and total retained storage
are explicit; their actual accounting is described below. Retained-window
capacity is simultaneous storage capacity, not cumulative logical-time capacity.

## Storage admission

The canonical layout contains names, alias identities, offsets and element count
only. It no longer allocates an unused FP32 parameter-sized zero bank or applies
an arbitrary `optimizer_bytes / 8` child limit. `DeviceOptimizer` allocates its
actual master, algorithm-dependent slots, live/next counters and corrections,
owner/group/tile tables and finite-error cache lines. Its exact tensor inventory
is checked before allocation and against the actual retained storages afterwards.

- `forward_bytes` bounds new retained tensor storage of one whole `advance`,
  including copied input tensors and complete-cut metadata. Existing owner state
  and earlier windows are borrowed and counted once in the total.
- `backward_bytes` bounds new retained storage of the complete multi-window
  reverse, canonical gradient and fixed-initial adjoints, including its retained
  roots. It is not a separate full allowance for every reverse subcall.
- `optimizer_bytes` bounds the persistent optimizer inventory plus the current
  step's newly retained publication/control tensors. The gradient already belongs
  to the reverse. Publication reuses the master and actual role banks.
- `total_bytes` bounds the union of the owner's live tensor storages, all retained
  programs and the current program, plus every retained program group's vendor
  workspace high-water mark. Each forward/reverse coordinator and its device-called
  regional programs share one arena because notification handshakes serialize all
  numerical users. Different retained windows still own different arenas.
  The union uses StorageImpl identity: views and borrowed optimizer
  slots count once; distinct equal-valued tensors count separately. No whole unused
  partition is subtracted from the total. Each group also enforces `workspace_bytes`.

All hidden optimizer tensors and region owner-coordinate tensors participate in
the total. Admission is repeated after control-program finalization, which may
create additional CUDA branch tables, and before numerical execution. Constructor
storage is checked before returning an owner. Input-copy lower bounds reject before
allocating a new window. A capacity refusal before execution does not change state.
`storage_usage()` reports actual owner tensor/workspace/optimizer bytes from host
shape/ownership metadata; it does not read device routing or numerical state.

These bounds concern storage retained by the internal session. CPU model copies,
checkpoint codec temporaries, allocator rounding, runtime-internal resources and
client-held copies/derived observation masks are separate. They are not an HBM
peak guarantee. In particular, adequate byte budgets do not certify CANN task-buffer
capacity. Old fixed evidence keeps its original refusal; changing the allocation
model does not itself certify LH176/full-model admission or performance.

`optimizer-storage-cpu` independently enumerates algorithm slots, empty sentinels,
storage aliases, overflow and exact-limit refusal. Its LH176 element/owner witness
uses a worst-case tile bound, without constructing or executing a model. The device
optimizer checker verifies the exact inventory against real tensors; the session
checker verifies exact initial limits, below-limit refusal and unchanged state on
an input-capacity rejection. Device build, actual execution and complete training
remain separate qualification facts.

## Checkpoint identity and exactness

Schema 2 contains full raw floating and int64 buffers, fixed Read0 raw
bindings, independent sample positions, actual node-presence bits, last regional
query times and input occurrence/time ledgers, canonical aliases/named values,
packed masters, every optimizer slot/counter/correction, generation and next token.
The device derives history from actual candidate presence (including zero mass),
and ledgers from actual physical external-port presence. This metadata is staged
and committed with the raw window; it is not inferred from nonzero numerical values.
Its identity includes graph/ranks/stride, all KV/regional options, source and node
owner tables, role mappings, shapes, trainable flags and alias partitions. Restore validates this
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

- Owner and guards: `tools/device_online/block_settle_session{,_layout}.cpp`;
  allocation/retained-workspace accounting: `block_settle_session_budget.cpp`.
- Cross-window raw transpose, canonical reduction and optimizer publication:
  `block_settle_session_training.cpp`.
- Exact archive: `block_settle_checkpoint.cpp`.
- Independent scalar causal formula, including linked-window autograd:
  `block_settle_formula.cpp`.
- Actual standalone consumer: `block_settle_session_check.cpp`, three AdamW
  steps, two retained windows per step, shared owners and exact fresh-owner
  restore. A first process writes `--checkpoint-path PATH`; a fresh process adds
  `--restore-check` to compare its actual final record to `PATH.expected`.

## Public Model and Python entry

The installed C++ `tide/resident_blocks.h` exposes `BlockResidentSession` with an
actual `SettleGraph`, `Model`, generic `Continuation`, capacities and device.
`block_settle_model.cpp` verifies installed weighted functions, unit transport,
complete-region observation and exact rank/delay alignment before packing roles.
It preserves canonical scalar/head shapes, source classes and private states.
No reference trajectory is evaluated during lowering or decoding.

The Python `WeightedBlockResidentTraining` uses this same C++ owner through the
matching `_tide_weighted_resident` plugin. `WeightedResidentTraining` selects it
by default when given an actual `SettleGraph` certificate and supported local
bindings. Ordinary Graphs and statically incompatible local/placement domains
retain the general owner. Capacity, malformed binding and numerical errors from
an admitted block owner fail explicitly; they never silently retry another
executor. This dispatch does not infer an aligned certificate for arbitrary DAGs.

The block API deliberately takes `advance(positions, payload, present)` tensors.
It publishes `output.strength/content`, `edge.strength/content` and
`region.R.received/controls/Full/message.strength/message.content/Mem.short/Mem.long/`
`total.short/total.long/Describe.NAME/old.FIELD/proposal.FIELD/final.FIELD`.
Separate presence/valid masks and actual raw int64 metadata accompany the fields.
Each retained token accepts independent named roots with matching tensor shapes.
Returned storage is borrowed and version-guarded, including derived validity and
presence masks; callers clone before mutation. Default optimizer groups contain
only the original model's trainable canonical owners.

The public complete-cut API requires the last original position to be equal
across samples. Intermediate positions and physical-port presence may differ.
`continuation()` requires detach/step and decodes actual raw buffers, state
presence, regional query history and input occurrence/time ledgers. The internal
owner can retain independent per-sample cuts, but these cannot be represented as
one generic Continuation without an explicit common seal.

`checkpoint()` returns the new schema-2 block record; `save()`/the standalone
archive helpers use the same atomic LibTorch archive. This is an explicit codec
migration, not the old Python dictionary/legacy resident checkpoint format.
Legacy queue/stage capacity keys are replaced by the block cache, forward,
backward, optimizer, total, workspace and retained-window limits; unknown keys
are rejected. The mathematical graph identity and eager checkpoint v5 remain.

Source navigation: `block_resident_owner.cpp` owns public lifecycle,
`block_resident_checkpoint.cpp` owns public/raw conversion,
`block_settle_observations.cpp` owns named observations/roots, and
`block_resident_bindings.cpp` plus `python/tidegraph/weighted/resident_blocks.py`
are client adapters. `block_settle_model_check.cpp` independently checks original
parameter values, aliases, source padding and nonempty import. The standalone
`block_resident_check.cpp` uses an independent Streaming schedule, nonzero source
factors with different hidden/rank geometry, duplicate physical edges, all input
and trainable parameter roots, three two-window AdamW steps and a fresh process.
`tests/test_resident_block_public_dispatch.py` checks static default selection
and refuses fallback after actual owner failure.

The current concrete owner is KV sum/qry, FP32 and single device, with homogeneous
KV geometry/alpha per region and homogeneous factor presence per source group.
Different regions may use different KV read functions. Sum lowering omits the
Read query/key/value roles entirely, while keeping Full Attention roles. The
typed Memory tape and the corresponding saved-storage/VJP dispatch preserve
the lack of sum-Memory dependence on current content; Describe/Full dependencies
are separate. Model checks exercise both read functions without synthetic owners.
The named observation fields and schema-2 archive stay unchanged; `memory_read`
already participates in identity, so sum and qry checkpoints are not interchangeable.
General PDG lowering, other local-memory families, cross-device ownership,
shared-prefix storage and FC16 scale/performance require their own implementation
and evidence. Public source, build, actual device consumer and default experiment
routing are separate delivery facts; none follows from the v1 internal owner.

## Standalone benchmark entry

`tools/weighted_benchmark/resident_policy.cpp` selects `auto|general|block-v2`
from the actual Settle certificate and installed mathematical bindings before
constructing an owner. New CLI invocations default to `--resident-executor auto`;
missing selectors in historical fixture records mean general. Explicit general
keeps the earlier checkpoint request identity. The block path currently requires
FP32, one accelerator, prefill/tensor inputs, compatible KV sum/qry groups and
`--return-mode outputs|metrics --validation none`. Complete event records,
explicit record input, streaming and statically unsupported modules retain
general under auto; an explicit block request rejects them. Runtime errors never
retry another owner. Standalone block does not implement the public complete
event `Result` decoder or inherit the general owner's complete-record tests.

`--resident-limits JSON` explicitly retains the historical general capacities;
`--resident-block-limits JSON` instead names `cache`, `forward_bytes`,
`backward_bytes`, `optimizer_bytes`, `total_bytes`, `workspace_bytes`, `windows`
and `timeout_ms`. The files are mutually exclusive. The report records requested
and effective owner, refusal/fallback reason, capacities and checkpoint codec.
An early resume check compares the request; exact effective owner/capacities are
checked before restore. Unspecified block budgets reuse the saved effective
budgets, preserving raw capacity shape. Insufficient capacity fails explicitly.

`block_inputs.cpp` packs only actual fixture inputs once. Timed steps shift
int64 positions, upload the whole B×L×P boundary, evaluate device windows and
seed whole-column roots for `scale * sum_present((c*G)^2)`. Both inference return
modes compute this task scalar; legacy metrics-only inference still reports no
task loss. `task_loss_computed` makes that cost distinction explicit. Output
records expand only actual physical output fields under no-grad; metrics avoid
that expansion. Work counts come from actual presence/fiber metadata. No
reference trajectory supplies batches, results or raw state.

`resident_block.cpp` uses the public session for SGD, detach, complete-cut raw
decode and the schema-2 archive. Inference currently also allocates the session's
manual tape and optimizer storage; this is recorded, not an optimized separate
inference owner. The existing benchmark timer includes input transfer, task,
VJP/optimizer when training, requested output decode and synchronization;
checkpoint I/O remains separately timed. Minimal-return runs are measurements,
not complete-record qualification. `weighted-block-adapter-check` independently
checks CPU policy, missing/zero/duplicate inputs and task VJP; actual standalone
device execution and fresh-process archive restoration remain separate checks.
