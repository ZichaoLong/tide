# SettleGraph encoding and independent anchors

`SettleGraph(graph, ranks)` requires a distinct positive integer rank per region
and edge delay equal to target rank minus source rank. `stride=max(rank)+2`;
the output boundary rank is `max(rank)+1`. Each (sample,position) input is sent
through a stateless source adapter at `stride*position`; the output adapter
aggregates source-tagged terminal values. Adapters have no trainable parameters.
Body-local input/output slots are preserved explicitly across boundary-to-edge
conversion; see `local-ports.md`. Projection re-sorts restored fiber tags when
external and internal inputs meet at one body node.

The direct Python executor settles a whole sequence by region rank, using exact
state/Full blocks. Its schedule does not call the TimedDAG planner. The native
SettleGraph execution path compiles the same encoding and runs the C++ frontier;
its ordinary dependency rules automatically form region sequence blocks. The standalone
C++ frontend in `tide/settle.h` constructs and validates this encoding independently
of Python; its binding is only a client/test adapter.
Native streaming provides a separate schedule for this encoding.

`project` removes boundary coordinates, restores external source tags, and
compares original event tensors, routes, states, histories, internal messages,
outputs and VJPs. Projection requires a complete position boundary, where no
messages remain in flight. Keep encoded continuation between native windows;
do not discard partially processed boundary state/messages. Logical time remains
the encoded time, and external position remains the original token position.

Independent topology anchors: Python/C++ `self_loop` for PositiveDelayGraph and
`chain` for TimedDAG; Python `settle_chain` for SettleGraph. They reject incompatible
graphs and own their propagation loops. They may share local formulas, and the
native specializations share the local region-block evaluator; neither invokes
a generic scheduler. Analytical cases separately check operator formulas.

The native standalone executable `tidegraph-smoke` runs a real delayed graph and
input VJP without Python. Its CLI runtime was adapted from the local
develop-portable-torch C++ asset; only CPU FP64/FP32 are accepted by this target.

## Standalone native frontend

`SettleGraph(Graph, region_ranks)` compiles the body and builds the encoded graph.
Ranks are distinct positive int64 values with room for output rank and stride;
all edges must match the strictly increasing rank difference. Empty boundaries,
bad owners/ports/domains and overflow fail. `embed_model` preserves the body
TensorImpl owners and aliases, including shared input/output scales; only frozen
identity-adapter values are added. Original model/kernel handles are not mutated.
No native constructor invokes Python or imports an already encoded Python graph.

```cpp
#include <tide/settle.h>
// body and model are ordinary C++ tide::Graph / tide::Model values.
tide::SettleGraph spec(body, region_ranks);
tide::Continuation q;
q.identity = spec.graph().identity;
q.batch_size = batch;
auto encoded_initial = spec.embed_initial(q);
tide::Options options;
options.packed = true;
tide::SettleExecutor executor(spec, model, options);
auto encoded_result = executor.run(encoded_initial, values); // [B,T,D]
auto body_result = spec.project(encoded_result);
// Continue with encoded_result.continuation, including both boundary namespaces.
```

Build without Python headers/bindings using normal LibTorch CMake discovery:
`cmake -S . -B BUILD -DCMAKE_PREFIX_PATH=LIBTORCH_PREFIX
-DTIDE_PYTHON_BINDINGS=OFF`; then
`cmake --build BUILD --target tidegraph-settle-check --parallel 2`.
`tidegraph-settle-check --device=cpu --dtype=float64` (also float32) constructs,
encodes and runs a two-layer graph and checks a literal independent recurrence,
initial-state/input/shared-parameter VJPs, unused Read gradients, actual sequence
batches, chunk continuation, owners and invalid ranks/coordinates. The ordinary
Python CPU gate also runs this executable in both dtypes.

## Exact mapping and limits

| Observable | Mapping |
| --- | --- |
| clock/event | body node v at stride*p+rank(v); input boundary at stride*p; output at stride*p+output_rank |
| nodes/regions/edges | body IDs are unchanged prefixes; two singleton boundary nodes/regions and boundary edges appended |
| ports/source identity | body local slots and logical source domains unchanged; input edge origin restores external port/position; parallel edges retain IDs |
| owner names | body nodes/regions/edge scales keep names; input/output scale handles move to appended agg_scale positions; this is an explicit owner-name map, not interchangeable checkpoint identities |
| state/slots/history | retain body prefix; adapters removed only for observation; no Tensor detach or mutation |
| messages/pending | internal edge messages keep IDs; boundary traffic removed at complete position cut; partial-cut pending must remain encoded |
| occurrence ledger | replicate each actual encoded input occurrence to each declared body input port, shift time by its rank; never infer occurrence counts from cut |
| outputs | source-tagged body terminal outputs are summed at the boundary, same order as direct SettleGraph |
| VJP/continuation | first-order roots map through functional tensor handles; preserve all slots/aliases/None vs zero; serialized continuation remains an explicit gradient boundary |

The C++ dense convenience entry seals a whole position window. Its returned
continuation remains encoded. `Frontier(spec.encoded_graph(), spec.embed_model(model), options)`
or `Streaming(...)` also accepts manually supplied aligned encoded inputs and
supports cuts within a position; projection rejects those cuts until complete.
`embed_initial` accepts only cut0 without pending/ledger. Body projection is an
observation, not a general inverse capable of reconstructing boundary state.
Missing sample ledgers stay missing, and present-zero sources remain present.
The TimedDAG-to-PDG inclusion changes no clocks, values or identities: a compiled
acyclic positive-delay graph can run through either frontier or streaming.

These constructors instantiate the rank-aligned, broadcast-input/summed-output
profile already used by the Python frontend. They do not prove arbitrary
SettleGraph encodings, arbitrary custom modules or arbitrary model equivalence.
Local program sequence/clear/Next capabilities still control legal prefill.
