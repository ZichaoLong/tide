# Explicit resident training owner

`tide::ResidentTrainingSession` is an optional installed C++ API in
`tide/resident_training.h`, linked with `tide::resident`. Its implementation and
qualification status are recorded in [STATUS](STATUS.md); this contract does not
by itself certify a build, Python client or throughput. The supported adjoint is
currently single-NPU FP32 HARD/HST/SOFTP, built-in Aggregate, phase-aware broadcast,
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
not an eager autograd node. A consumer computes its head/loss and supplies
cotangents for the outputs, pending messages, final state and attention caches of retained windows.
It may use its own autograd on detached output views; it must not mutate the
owner's output storage. No particular head, loss or convergence task is required.

1. `advance(inputs, stop, seal)` performs independent online execution and saves
   its actual device tape. Windows carry session, sequence and parameter-generation
   tokens. Outputs, pending payloads, state values/presence and grouped KV
   values/lengths/presence remain on NPU.
2. `backward(roots)` accepts exactly one cotangent record per retained window in
   forward order. Both value and connection tensors of each root pair are supplied,
   or neither. Missing pairs mean None; connected zero remains connected. Root
   shape/device/dtype and absent-coordinate validation happens before execution.
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
Repeated retained backward within one generation is not offered by this owner.

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
per-window allocations; a small budget rejects before executing a partial VJP.
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
        leaf = window.outputs.values.detach().requires_grad_(True)
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
