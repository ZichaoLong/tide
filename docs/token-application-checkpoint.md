# Two-clock application value checkpoint

Implemented and [qualified](evidence/token-checkpoint-coordinates.md) on CPU FP64/FP32. This is a
separate **tide-token-application-v1** format. Single-graph checkpoint v5 retains
its keys, value semantics and API. Shared value codecs live in checkpoint_values.py.
No generic executor or LH training behavior changes.

## Application boundary

TokenApplication names two owners, body/readout, their exact graph identities,
L body ticks per token, a chosen body output port and the common Emit mode/zeta
for both engines. Callers execute that declared policy; loading a different
mode/coefficient fails identity validation. The consumer has exactly
L external ports, one per body phase. Both models use the same CPU width/dtype;
no implicit projection/cast occurs. Registered heads belong to their model.
Cross-graph Parameter identity and every alias are part of the ownership contract.
The application is a persistence/validation object, not a new scheduler.

TokenState contains a global complete cut c, both complete continuations and the
unfinished body-output buffer. With D=L+1, body.cut is floor(c/D)*L+min(c%D,L)
and readout.cut is floor(c/D). They have the same positive batch size. Buffer
rows have unique (batch, body_time, selected_output_port) coordinates, finite
width-matching payloads and readout.cut*L <= body_time < body.cut. Immediately
after readout this interval is empty. Sparse/missing sample-phase rows are legal;
numerical zero is a present row. Buffer order is preserved, not reconstructed.

Both input occurrence ledgers are stored verbatim and validated by their graphs.
They are never inferred from token index or last send time. The caller provides
every unconsumed output; without the execution trace, structural checkpoint
validation cannot prove that a valid row was not omitted by the caller.

## Save, load and update

Construct TokenApplication(body_graph, body_model, readout_graph, readout_model,
layers, output_port=0, mode="hard", zeta=1.0). Its models ModuleDict owns the same Parameter objects;
use a deduplicated stable optimizer order and reconstruct the same aliases before
load. app.save(path, state, optimizer) writes one atomic exclusive file containing
application identity/policy, named model values/aliases, optimizer ownership/state,
both graph continuations and the partial-window payloads. The caller supplies a
quiescent complete cut; save is not synchronization with an advancing engine or
concurrent optimizer update.

app.load(path, optimizer) validates identity, alias topology, all shared weight
values and tensor metadata before touching live owners. It loads weights on a
temporary copy and validates **both** continuations, clocks and the buffer under
those saved weights, then preflights optimizer ownership/state on a temporary
optimizer. Only after all checks pass does it restore live model/optimizer values.
The temporary copies cost checkpoint-time memory. Ordinary built-in models and
optimizers are the supported transactional-preflight scope; arbitrary hooks with
external side effects retain the limitation of checkpoint-ownership.md.

Save/load and TokenState.detach() truncate both continuations and all buffer
payloads. An in-memory state by itself does not detach. The loaded value state
starts a new autograd segment. Gradient buffers, RNG, external data cursor and
sampling/generation policy and module train/eval settings are not stored; the
caller reconstructs the same programs, zeroes gradients and supplies
the same future sealed inputs. No interoperability with the separate v5 single-
PDG file is implied. Standalone C++ optimizer/file ownership remains separate.

## Qualified validation scope

Cross-graph send/receive and memory parameters are shared explicitly in fixtures.
Compare uninterrupted Python two-clock updates with twice-restored Python or
native serial-node/parallel-node engines, including native batch packing, using
full readout traces and occurrence ledgers. Test HARD/SOFTP/HST, clear, Add and
all-softmax, FP64/FP32, SGD and explicit-epsilon AdamW. Corrupt the second
continuation, buffer, alias values and optimizer ownership/state and require
rejection before either live model or optimizer changes.
