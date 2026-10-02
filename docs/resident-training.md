# Explicit resident training owner

`tide::ResidentTrainingSession` is an optional installed C++ API in
`tide/resident_training.h`, linked with `tide::resident`. Its implementation and
qualification status are recorded in [STATUS](STATUS.md); this contract does not
by itself certify a build, Python client or throughput. The supported adjoint is
currently single-NPU FP32/FP16 HARD/HST/SOFTP, built-in Aggregate, phase-aware broadcast,
identity/EMA/Add-repeat/event/fiber-attention state and identity/tanh/LH/SwiGLU Full. Other adjoints fail at
construction. The wider [execution contract](execution-flows.md) remains required.
The normalized Aggregate implementation and its separate qualification status
are described in [its VJP contract](resident-aggregate-vjp.md).
[Emit/control/Read adjoints](resident-control-vjp.md) retain complete candidate
frames, including unselected Read connections and connected-zero HST paths.
Mode and `zeta` are execution options; HARD remains the default.
[Event attention/KV adjoints](resident-event-vjp.md) add Q/K/V/O parameters,
separate key/value roots and initial cache gradients, including retained-window
links. [Same-fiber attention adjoints](resident-fiber-vjp.md) add all five pooling
profiles, repeated-tick decay and separate log-bias roots/initial gradients.

## Lifecycle and consumers

Construct from a graph, model and complete continuation, optimizer kind/groups,
explicit logical device and training limits. Construction preserves TensorImpl
alias identity while freezing parameters on CPU and packing them onto the NPU.
All aliases of an owner whose tensor requires gradients enter the registry.
An omitted group list selects these owners; explicit groups retain the portable
named optimizer's meaning, including an empty group's selecting no parameters.
Distinct TensorImpl owners sharing storage are refused. Caller parameter updates
invalidate the session. The owner updates its own device parameters; it does not
silently copy them into the caller's original model.

All owner methods require explicit no-grad. This is a first-order VJP interface,
and construction,use and close belong to the same host thread/CANN contexts,
not an eager autograd node. A consumer computes its head/loss and supplies
cotangents for the outputs, pending messages, final state and attention caches of retained windows.
It may use its own autograd on detached output views; it must not mutate the
owner’s output storage. No particular head, loss or convergence task is required.

`forward.diagnostics` controls optional latest-window trace/messages in `result()`.
Python uses the runtime's `trace` option. Training records the actual VJP journals
even when these exports are disabled; positive trace/cache journal capacities
remain required. Disabling exports omits only the diagnostic message queue,
weighted-contribution journal and prior-history/source-scale snapshots. Outputs,
continuation, retained state/cache, reverse links and optimizer behavior remain
available. Diagnostic recording can change after complete-cut checkpoint restore.
The public C++ limits and checkpoint layouts are unchanged. Qualification of this
separation is recorded in [training-record evidence](evidence/resident-training-records-20261002.md);
existing evidence retains its source scope.

1. `advance(inputs, stop, seal)` performs independent online execution and saves
   its actual device tape. Windows carry session, sequence and parameter-generation
   tokens. Outputs, pending payloads, state values/presence and grouped KV
   values/lengths/presence remain on NPU.
2. `backward(roots)` accepts exactly one cotangent record per retained window in
   forward order. Both value and connection tensors of each root pair are supplied,
   or neither. Missing pairs mean None; connected zero remains connected. Root
   shape/device/dtype and absent-coordinate validation happens before execution.
   Root values are FP32 and masks are bool for both FP32 and FP16 forward
   payloads. Missing values allocate FP32 zero roots; half cotangents are refused.
   All reverse event decisions, pending-boundary links and shared-owner reductions
   use the candidate's own device records. No CPU reference supplies routes.
3. `step()` requires completed backward. Device SGD/AdamW proposals, finite gate,
   commit and parameter publication precede the next forward. A successful step
   consumes gradients and explicitly ends this differentiation generation, even
   when all gradients are None. A step cannot cross outstanding forward tapes.
4. `detach()` explicitly discards tapes or unapplied gradients, preserving numeric
   state/history/pending and current parameters. It invalidates their root tokens.

Advancing with unapplied gradients and repeated backward of consumed windows
are refused. Step establishes the explicit truncation boundary; differentiation
through optimizer updates and higher-order gradients are outside this profile.
Repeated backward of the same retained windows is not offered by this owner.

An optional `accumulate(max_bytes=128*1024*1024)` consumes the latest backward
into a separate FP32 device bank without updating parameters or optimizer slots.
It explicitly detaches state/history/pending/KV at that boundary and allows the
next forward at the same parameter generation. Call it after **every** backward
of an accumulated update, including the last one, then call `step()` once.
`accumulated_batches` counts these consumed backward groups; successful step or
explicit `detach()` resets the count. Step refuses outstanding tapes and an
unaccumulated final backward. Checkpoint refuses any accumulated gradients.

Accumulation sums canonical gradients and ORs their connection flags; None and
connected zero retain their optimizer meanings. It performs no averaging or loss
scaling. Consumers must normalize losses for the intended complete logical batch.
The first accumulation copies the public backward exports into private storage.
Later accumulations reuse only that private numeric bank; old connection flags
remain read-only until all device tiles finish, with separate new output flags.
All owners are preflighted and built before any execution. Admission still
conservatively charges the old and replacement banks plus their tensor metadata,
summed across devices, against `max_bytes`. Current backward exports
and CANN program arenas retain their separate budgets; keeping caller exports
alive also keeps their storage alive. Capacity refusal is retryable. Nonfinite
connected gradients are rejected by the existing all-owner optimizer transaction;
`detach()` explicitly discards a rejected accumulation.

This is an explicit truncated-gradient policy, **not** a substitute for retaining
cross-window gradients: leave connected windows within one `backward()` group.
It does not change batch size or switch independent sample continuations.
Resident physical sample slicing still requires that separate capability. Both
the single-device and compact multi-device owners expose this optional method;
the checkpoint format and default backward/step behavior are unchanged.

With the aggressive physical chunk policy, training retains only the valid
prefixes of event, source, emission and attention journals. Compaction runs at
the completed-window boundary; dynamic output extents synchronize there without
exporting numerical records. One unused row remains for an empty journal. Row
positions, aliases, stable order and cross-window links stay unchanged. Pending,
output and KV buffers keep their complete layouts, including disconnected roots.
This is retention of the candidate's own forward, not a reference execution or
offline scheduling prepass. Conservative policy keeps dense journal copies.

Before advancing, the owner still reserves the next window's dense bound. After
retention it charges the actual retained bytes, so subsequent windows can reuse
the released allowance. `retained_window_bytes` remains the dense per-window
bound excluding shared Full/emission/attention banks; `retained_full_bytes`,
`retained_projection_bytes` and `retained_attention_bytes` report their
once-per-backward-group snapshots. Full sharing is
[qualified on cleanf360489](evidence/resident-full-snapshots-20261002.md);
the complete-consumer estimate still conservatively charges Full banks per window.
`retained_dense_bytes` reports the entire
dense envelope and `retained_bytes` the stored total. `retained_compact_journals`
records the policy. Temporary masks/indices need metadata workspace proportional
to declared journal capacities, outside the retained tape/state budget. The
complete consumer planner charges this separately and still uses dense bounds;
this optimization alone does not admit a previously refused full-size run.
Compact retained journals are [qualified on fixed source](evidence/resident-retained-journals-20261002.md),
including independent VJPs, multi-update/continuation checks and allocator calibration.
Attention snapshot reuse is [qualified on clean38858d0](evidence/resident-attention-snapshots-20261002.md),
including independent multi-update/restore checks and same-lease allocator calibration.
Its implementation/lifetime contract is in [retained windows](resident-retained.md). A retained-byte decrease
alone does not establish a complete allocator-peak decrease or a throughput gain.

`snapshot_device(max_bytes=...)` saves a detached numerical continuation in opaque
NPU buffers; `restore_device(saved)` switches to it on the **same live owner**.
The saved buffers contain values/presence/clocks, complete event/fiber KV including
lengths/bias, selection history, pending identities/values/masks and input ledgers.
Default saving copies whole buffers on their original devices. Optional
`snapshot_device(max_bytes=..., compact=True)` (C++ overload `(max_bytes, true)`)
batches device row selection and gathers only valid pending rows and complete KV
prefixes, with shared int64 physical-row indices. Numerically zero rows remain
present. Dense groups are retained if indices would cost more than the saving.
Restore uses unique-index bulk scatter and resets unused padding/sentinels.

Compaction uses dynamic-shaped nonzero at this explicit detached boundary; its
output extent requires synchronization. No per-event host loop or CPU numerical
state/row-index/length-vector export is used. Input-ledger/cut metadata stays on
the host. These save/restore calls and their synchronization must be included in
consumer timings; online scheduling and selection remain device-owned.

Save/restore refuse retained windows and an unconsumed backward. They allow an
existing accumulator and preserve its parameter gradients, generation and
optimizer. Handles saved before an update may be restored afterward: this
retains that stream's numerical state while using the current shared parameters,
with an explicit gradient boundary. It does not restore parameters or advance the
optimizer. Restoring clears latest-window diagnostics; the next forward records
its own events. The independent stream supplies its own next input positions.

Each handle reports `cut`, `batch_size`, `tensor_bytes` and `device_bytes`
(logical NPU index to saved bytes); its capacity check
bounds that snapshot's saved tensors (including shared packed indices) across
owner devices before copying payloads. Optional `device_budgets={index: bytes}`
(the third C++ overload argument) additionally bounds each device. A nonempty
mapping must cover every used device; zero bytes explicitly admits no storage.
All device limits are checked before any saved payload copy. Empty/omitted
mapping retains the total-only API. The limits do not include other handles:
callers subtract their simultaneously live saved bytes before saving a new one.
Compact row selection uses additional
metadata workspace proportional to declared queue/KV capacity; this workspace
is outside `max_bytes`, as is vendor allocator overhead. Other
live handles and the active flow still require storage. Handle copies share
immutable buffers; destroying the last copy releases them. Foreign/empty handles
are refused before mutation. Device-copy failure poisons the owner. Handles are
not serializable checkpoints and cannot be transferred to a recreated session.
Compact saving changes only the opaque handle layout. Live KV, retained tapes and
logical capacity are unchanged. Consumers must account for every live handle,
packing workspace and the active owner, and preserve the logical batch/loss/update
boundary when composing this API with accumulation. No automatic capacity search
or full-size fit follows from a smaller snapshot.

Parameter gradients use canonical names, alias sets, packed offsets and separate
connection flags. Offset -1 identifies an owner without a differentiable use in
this profile. Boundary gradients retain all six physical coordinates and flags
for actual incoming leaves. Later windows also expose their incoming pending
adjoints for inspection; these are already connected to earlier tapes internally.
Initial-state gradients bind only states present at the generation's initial cut;
automatic zero initializers do not become caller leaves.
`initial_cache` follows the same rule for key/value and fiber log-bias leaves, with independent
connection flags and actual initial lengths. Cache groups and owner order are
defined in [the event VJP contract](resident-event-vjp.md).

FP16 forward parameters, state, messages and caches retain their actual half
values. Reverse uses those recorded values and the declared cast VJPs, with FP32
cotangents, accumulation and SGD/AdamW masters/slots. Each successful update
publishes the master's half-rounded value into all forward parameter banks;
sub-half increments remain in the master across updates and checkpoint resume.
Loss scaling is not implicit: a consumer supplies FP32 cotangents for the
objective it intends to differentiate. Half autograd leaves in a consumer head
would already round its returned cotangents; use FP32 loss leaves as below.

Returned tensors are read-only consumer views. Keeping exports beyond their
consumption holds device storage and belongs to the consumer's memory budget.
They do not grant permission to mutate saved tapes, parameters or gradients.
The owner is not safe for concurrent calls or concurrent consumer mutation.

## Budgets and failures

Forward limits continue to govern queues, journals, workspaces and physical
chunks. Training additionally bounds retained-window count and aggregate tape
bytes, reverse tensor bytes, optimizer tensors and CANN reverse/update workspace.
Each tape currently copies parameter banks. Shapes admit a complete saved window
before advance. Reverse components divide the total declared budget into disjoint
per-window allocations. Event/fiber attention now shrinks physical owner/query/key
batches inside those reservations before allocation; full logical fibers and
normalization stay intact. A budget too small for one complete owner rejects
before executing a partial VJP. `gradients.statistics` reports effective physical
maxima and estimates, as detailed in [online-consumers.md](online-consumers.md).
Budgets measure declared tensor/workspace footprints, excluding allocator and
vendor runtime overhead. They do not promise immunity to external device pressure.

Invalid seals/inputs, root tokens/layouts and admission failures are retryable
without advancing live numerical state. Exhausting retained capacity requires
backward or explicit detach, never silent truncation. Device forward/reverse or
runtime failures poison the owner; recover a prior checkpoint in a new session.
Optimizer refusal20 (nonfinite) or21 (counter violation) commits no live parameter
or slot, keeps the current gradients and returns `applied=false`. The consumer
can explicitly detach to skip that update. No hidden loss scaling is applied.

## Complete-cut checkpoints

`checkpoint()` exports a detached CPU `ResidentTrainingCheckpoint` only with no
outstanding tapes or gradients. It includes the actual updated parameters,
all aliases/trainable names, complete continuation and input ledger, optimizer
kind/groups, packed offsets, moments/momentum/AMSGrad slots, int64 counters,
Adam bias corrections, generation, next window sequence and explicit Emit mode/zeta. Unused parameter
owners retain their frozen initial values; updated values always come from NPU.

The checkpoint constructor validates schema, ownership, graph/continuation,
options and tensor layouts, including agreement between named and packed
parameters. It copies into a new owner. The original owner and caller model are
not modified, and consumer changes to an exported checkpoint cannot alter the
live owner. Changing a compatible schedule or memory limits on reconstruction
does not change checkpoint identity. This format is an explicit in-memory
boundary. The Python client serializes it as described below; C++ consumers may
provide their own codec. RNG, loss heads and consumer data cursors remain consumer-owned.

Named parameters retain the configured payload dtype; packed parameters and
floating optimizer slots are FP32. Restore requires matching named shapes/dtypes
and `master.to(payload_dtype) == named_parameter`, rather than discarding the
master's low bits. Nonfinite masters, invalid slots/counters and masters that
cannot publish a finite half value are rejected. The existing v1 record is
self-describing through its tensor dtypes; changing the runtime payload dtype
is not an implicit checkpoint conversion.

## Python-owned client and disk boundary

Build the optional backend against a matching Python-owned NPU core; its plugin
must not link the standalone NPU SDK. `GraphRuntime.training_session(...)` is a
client of this C++/CANN owner. It does not certify an independent pure-PyTorch
device scheduler. The normal `session()` retains its inference contract.

`ResidentTrainingLimits` configures retained windows, byte budgets and reverse
chunk rows. The runtime's resident limits still configure forward work. Training
always records the journals needed for VJP, even with public `trace=False`; this
is part of training cost. Unsupported placement, dtype, mode and adjoints refuse.

```python
with torch.no_grad(), runtime.training_session(
    batch_size=2, optimizer="adamw", limits=ResidentTrainingLimits(windows=4)
) as session:
    window = session.advance_device(external, stop=20, sealed_until=20)
    with torch.enable_grad():
        leaf = window.outputs.values.detach().float().requires_grad_(True)
        visible = torch.where(window.outputs.valid[:, None], leaf, torch.zeros_like(leaf))
        loss = visible.square().sum()  # Consumer example, not a graph semantic.
        bar, = torch.autograd.grad(loss, (leaf,))
    gradients = session.backward([session.cotangents(window, outputs=bar.detach())])
    update = session.step()
    if not update.applied:
        session.detach()  # Explicitly skip a refused update.
    session.save("new-training-checkpoint.pt")
```

Multiple windows are retained by collecting one cotangent record per window and
passing them in forward order to `backward`. Supplying a value without a mask to
`cotangents` uses that window's actual presence mask. Omitted value/mask pairs
mean None. Returned parameter and boundary gradients remain packed on NPU.
The optional `cache` argument supplies one dictionary per `window.cache` group;
each dictionary may contain `key`, `value`, fiber `log_bias` and their separate connection masks.
Omitting `cache` disconnects all cache roots. Padding is outside the logical
cache: losses must select the prefix described by each owner's `lengths`.
The consumer owns any separate head parameters and optimizer. The original
`runtime.model` is still the construction template; inspect updated graph weights
through the explicit checkpoint, not the template's stale values.

`checkpoint()` returns a CPU dictionary. `save(path)` uses existing atomic,
no-overwrite publication. `runtime.training_session(batch_size, checkpoint=path)`
creates a new owner with restored weights, slots/counters/corrections and graph
continuation. A dictionary may be supplied instead of a path. Resume is exclusive
with initialization arguments `continuation`, `optimizer` and `groups`; limits
may change. It does not mutate an existing session or caller model. Schema
`tide-resident-training-v1` is distinct from eager `tide-continuation-v5` and the
standalone named-parameter format. Loading uses `weights_only=True`, strict int64
metadata and alias/layout checks; no resume through unconsumed tapes is claimed.
Mode/zeta must match on restore. Legacy v1 records without these additive fields
mean HARD with zeta1; the HARD packed-owner layout is preserved.

Qualification remains indexed by STATUS/ROADMAP, including independent processes.
Python and C++ clients, disk restoration and training throughput are separately
verified; code or a successful build does not substitute for those target checks.


## Multiple logical devices

The optional [sharded training owner](resident-sharded-training.md) uses the same
public session and checkpoint schema with an explicit logical placement. Its
qualification is separate from earlier single-device evidence. Empty placement
preserves the original single-device path; sharded windows/roots expose local
state and KV instead of a dense coordinator copy.
