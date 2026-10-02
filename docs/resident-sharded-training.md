# Public sharded resident training

This extends `tide::ResidentTrainingSession` and its Python client. It reuses the
online scheduler, compact state/Full owners, retained VJP and canonical atomic
optimizer described in [resident-training.md](resident-training.md) and
[execution-flows.md](execution-flows.md). Public C++/Python-owned FP32/FP16 training and portable repartition are qualified
on clean source 0d7c45e ([evidence](evidence/public-sharded-training-20261002.md)).
Model-scale throughput remains separately pending.

## Construction and ownership

C++ sets `ResidentTrainingLimits::placement` with logical `devices`, `policy`
(`memory` or `locality`), and optional `full_owners`/`state_owners`. Each explicit
owner map has one index into `devices` per execution node. `devices[0]` is the
explicit session coordinator; duplicates, invalid maps and insufficient budgets
are refused. Empty maps request the generic topology/parameter planner. Empty
devices select the original single-device implementation. Public struct layouts
changed; rebuild C++ consumers and bindings with the matching library. A one-device explicit
list exercises the sharded representation without inter-device transfers.

Python uses `ResidentPlacement` in `runtime.training_session(..., placement=...)`.
Physical IDs belong to launcher visibility, never this configuration. `placement`
returns the resolved node maps; the session's `manifest()` records requested and
resolved placement, memory limits and actual device count. The runtime manifest
alone describes its default single-device inference session.

For models larger than one NPU, construct `GraphRuntime(..., device="npu:0",
model_device="cpu", ...)`. This explicit option leaves initial model tensors on
CPU, including Settle boundary buffers. The native owner builds persistent banks
on the selected NPUs directly. It does not initialize a complete model on the
coordinator and then repartition. Caller parameter identity/version guards remain;
updates belong to the native banks. Host eager sessions require their normal model
storage device. A CPU parameter source performs no numerical event prepass.

## Consumer loss and roots

A sharded `ResidentTrainingWindow` retains output and pending packets on the
coordinator and `states` in device-list order. Each state entry contains global
`nodes`, local `values[sample,local_node,width]`, `present`, and local cache groups
with global node IDs. Legacy dense `state_values`, `state_present`, and `cache`
are undefined/empty in this mode. All returned tensors are read-only to the
consumer and retain their storage after session close.

`ResidentCotangents::states` mirrors this owner order; an omitted list disconnects
all state/cache roots. Each provided entry has `final`, `final_connected`, and
optional cache cotangents. Values are detached FP32, connection masks bool, on
the device of the corresponding forward output. Values and masks must be paired;
connections to absent state/messages are invalid. Connected zero, disconnected
poison and padding retain their existing meanings. No inference event lookup is
needed to construct roots.

```python
from tidegraph import ResidentPlacement

with torch.no_grad(), runtime.training_session(
    batch_size, optimizer="adamw",
    placement=ResidentPlacement(devices=("npu:0", "npu:1")),
) as session:
    window = session.advance_device(inputs, stop=stop, sealed_until=stop)
    # The consumer owns its loss/head. Compute FP32 cotangents on each owner
    # under torch.enable_grad(), using presence masks to exclude padding.
    roots = session.cotangents(window, outputs=output_cotangent,
        states=[dict(final=x) for x in state_cotangents])
    gradients = session.backward([roots])
    status = session.step()
    if not status.applied:
        session.detach()  # Explicitly discard a refused update before continuing.
```

Returned gradients use `parameter_shards` (canonical names/aliases, offsets,
FP32 values and connections on the canonical parameter owner), `initial_shards`
(local state/cache gradients), and ordinary physical boundary records. Legacy
dense parameter/initial fields are undefined in this mode. Canonical owners can
differ from Full/state owners and alias contributions are reduced exactly once.
Backward and step are separate calls; step consumes the completed reduction.

`ResidentGradients::statistics` exposes the reverse attention reservations also
recorded by the [public consumers](online-consumers.md). Physical owner/query/key
caps can shrink to fit their tensor budget while preserving complete fibers and
global attention normalization. These estimates exclude the separately bounded
vendor workspace and do not certify a total model-memory budget.

Canonical contribution and parameter-publication groups reuse per-device FP32
packet arenas. Each phase plans its maximum send and receive capacities before
allocating either buffer; endpoints remain capped at 64 MiB and shrink to the
existing tensor budget. Send and receive storage stay separate. The existing
device notification waits until the remote copy completes before the sender can
overwrite a packet; the receiver consumes its packet on the same stream before
the next receive. Global pair/ordinal order, local accumulation order, connection
flags, partial tails and sticky errors are unchanged. No host per-packet loop or
new numerical prepass participates in execution.

Reduction and publication have independent phase arenas. Per-group descriptors,
flags and notifications remain distinct; sharing those mutable controls would
need a separate proof. `canonical_stream_reserved_bytes` charges unique packet
storage plus every group's metadata for one backward reduction, while
`canonical_stream_chunks` still counts planned packet iterations. Consumers take
the maximum of byte reservations across sample groups and sum iteration counts.
These are planning counters, separate from observed allocator peaks. Full
gradients, master values and optimizer slots are not replaced by packet arenas.

With `chunk_policy=aggressive`, each completed reverse window reduces its
parameter contributions before the preceding window executes. The ordered device
coordinator appends this reduction directly to the window program; peer start
packets precede all writes, and an end-of-reduction consensus confirms that all
owners have consumed their inputs. Only then can the preceding window reset and
reuse physical projection-weight/bias/connection adjoints. The canonical output
bank and numerical packet arenas are shared across these reductions as well.
Event/fiber Attention parameter adjoints use the same completion boundary and
state-owner init packet to reuse their numeric banks and connection masks.
Their node mapping, parameter offsets, device, dtype, shape and storage aliases
are validated before reuse. Cache key/value/log-bias boundary gradients, state
and message cotangents remain independent; they are needed to bridge windows.
Per-owner floating additions remain reverse-window then registry-alias; aliases
are not first combined into a window sum. Nonfinite/disconnected handling and
sticky errors retain the same contract. Full, state, cache and message adjoints
remain per-window, including the gradients needed by continuation bridges.

`streamed_parameter_windows` counts windows reduced this way;
`reused_projection_gradient_bytes` counts duplicate physical projection-adjoint
storage avoided within each backward group. `reused_attention_gradient_bytes`
reports the corresponding event/fiber parameter-adjoint reuse. Both are storage
accounting counters, not measured allocator savings. The conservative policy retains the separate
final reduction. The legacy single-device representation does not use this
optimization; an explicit one-device placement uses the sharded representation.
Consumer admission estimates remain conservative and unchanged. Shared arena
bytes are counted once plus each reduction's distinct metadata. No new public
option, host event loop or CPU numerical prepass is introduced.

Forward state/KV values remain owner-local. State root cotangents may use bounded
coordinator scratch for reverse routing; this is not a forward state replica.
Host loops traverse static owner/module groups and retained windows, not actual
events/messages. Online decisions and reverse event links are computed on device.
A device notification chain connects retained reverse programs, with a host wait
at the complete backward boundary. Construction/validation and loss/head work
remain explicit boundaries and must be counted appropriately in performance.

## Updates, continuation and portable checkpoints

The optional [gradient accumulation API](resident-training.md) keeps one FP32
bank per canonical owner device. Alias reductions precede accumulation; summation
and connection OR remain on those devices. Only `step()` runs the common finite
agreement and optimizer/publication transaction, once for the accumulated update.
This explicitly detaches between backward groups and does not switch sample state.

All owners propose an update, reach device-wide finite/error agreement, then
commit masters/slots/counters and publish aliases to the actual forward banks.
A finite/overflow refusal returns `applied=False` and does not partially commit.
Successful update detaches the retained graph while preserving numerical state,
KV, history, pending packets and input ledger. Window tokens are session/generation
specific; stale, foreign and out-of-order roots are refused. Capacity refusal
never silently drops retained windows or truncates gradients.

At an explicit detached complete cut, `checkpoint()` exports the existing CPU
schema 1 (Python `tide-resident-training-v1`) in global canonical order. It contains
all parameter aliases, FP32 masters/slots, counters and continuation. Physical
placement is not part of the portable record. Restore validates the complete
record and can select a new owner count, policy, node maps or the legacy single
owner. A consumer separately owns RNG, data cursor and head/optimizer state.
Checkpoint export/import is not part of ordinary resident event progression.

The directed C++ gate is `tide-resident-sharded-session-check --device=npu:0
--dtype=float32 --devices=2 --resume-devices=3` (also `float16`); resume count 0
selects the legacy single owner. Standard component verification/profile entry
points expose `resident-sharded-training`. Python tests exercise the public
client, scalar CPU oracle, loss cotangents and fresh-process disk restoration.
