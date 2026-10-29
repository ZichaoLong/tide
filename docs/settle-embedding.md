# SettleGraph: general ports and explicit boundary bindings

Authority: core-5 Settle §2–6, §9–11 at the revision locked in [upstream.json](upstream.json).
All coordinates are signed int64. Equal ranks are legal for independent regions;
every physical edge must increase rank with delay equal to the difference.
Parameters, states, descriptors and Full slots retain their chosen local contracts.

## General ports (`binding="ports"`)

Python `SettleGraph(graph, ranks, binding="ports", time_stride=D)` and C++
`SettleGraph(graph, ranks, "ports", D)` encode §2.4. D exceeds every region rank;
the default is `max(ranks)+1`. Graph nodes/edges/regions/ports and parameter owners
remain exactly the same in `embed`/`embed_model`; §10 adds no adapter nodes.
Python `port_external(values, positions, port=..., batch=..., start_occurrence=...)`
and the C++ counterpart map a `[occurrences,width]` tensor and strictly increasing
global positions to external rows. For occurrence k, `position=start_occurrence+k`
is the per-port ordinal while `time=D*positions[k]+rank(target)` is logical time.
Different ports can have different schedules, including gaps, and can attach to
interior nodes. Empty input slices do not create events or increment ledgers.

Python `settle_ports.run_ports` owns a region-major schedule. Strict ranks prove
that all possible incoming edges are settled before a region is processed;
sealed inputs provide the external completeness proof. The planner inspects
actual fibers, never scans the gap between sparse times, and does not pre-run a
reference trace. With `prefill=True` it uses exact local region/state/Full blocks;
with `False` it processes scalar frames. Module-specific fallback counters remain
visible. C++ `SettleExecutor::run_ports(q, rows, stop, sealed_until)` executes the
same specification through independently implemented streaming/frontier/greedy
schedulers; it is an embedding frontend, not an independent specialization.

Every logical cut is valid. `project` is the identity decoding for this binding,
including private state/history, per-port ledgers, messages and pending. Alignment
is checked for external rows, pending send/arrival times, state/history timestamps
and ledger times. `decode_slots(spec, result, start_position, stop_position)`
materializes named `(batch, global_position, edge/port)` families for a declared
complete-position interval: present values retain the actual Tensor handle,
missing records become `None` (mathematical ⊥), including inactive/no-input slots.
It never calls a reference formula. This slot view is not a checkpoint format;
checkpoint v5 preserves the full continuation, including interior-cut pending.

| Definition / conditions | Code and independent acceptance |
| --- | --- |
| LegalRegionRank/LegalTimeCode, §2.3–2.4 | Python/C++ `SettleGraph` constructors, `port_external`, `validate_alignment`; `test_time_code_validation_and_sparse_int64` rejects wrong rank/delay/clock and checks times above 2^54 |
| Named edge/port values and absence, §3/§6 | `settle_ports.decode_slots`; `test_named_slots_literal_parallel_paths` uses literal 31x+88y+3z coefficients, duplicate edges, interior inputs, separate outputs, zero versus absent and independent input VJPs |
| Identical record embedding, §10 | Same graph/model and tensor owners; `test_every_cut_records_independent_roots_and_continuation` compares Python time/region and native orders, all cuts, output/state/pending roots, None/zero gradients |
| Stateful composition, §9; first-order training | `test_shared_optimizer_and_checkpoint_with_pending`: three AdamW steps, shared parameters/private states, exact pending restore, resumed step |
| Standalone C++ | `cpp/test/settle_ports.cpp::check_settle_ports` checks literal parallel-path coefficients/gradients, absent root, same-rank nodes, port occurrence versus global position and every cut with all three native schedulers |

Float tolerances are CPU FP64 1e-10/1e-8, FP32 1e-6/1e-5; discrete records and
integer metadata are exact. HST in the scheduling tests is the existing declared
profile, not the new weighted-message allocation derivative. Port payloads here
still use existing local modules; structured weighted payloads have a separate
implementation milestone. Current run status and source-qualified evidence are
listed in STATUS and the evidence index; source availability is not a device claim.

## Broadcast input and summed output (`binding="broadcast_sum"`)

This is the old default convenience API and the explicit core-5 §6.4 binding.
Its time stride defaults to max(rank)+2, with output rank max(rank)+1. A larger
explicit stride is allowed and does not alter the output rank. The following
adapter/projection details apply only to this binding.

`SettleGraph(graph, ranks)` requires a positive integer rank per region
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

Independent topology anchors: Python/C++ `self_loop` and multi-node `ring` for
PositiveDelayGraph, `chain` and `diamond` for TimedDAG; Python `settle_chain` and
`settle_layered` for SettleGraph. They reject incompatible
graphs and own their propagation loops. They may share local formulas, and the
native specializations share the local region-block evaluator; neither invokes
a generic scheduler. Analytical cases separately check operator formulas.

The native standalone executable `tidegraph-smoke` runs a real delayed graph and
input VJP without Python. Its CLI runtime was adapted from the local
develop-portable-torch C++ asset; only CPU FP64/FP32 are accepted by this target.

## Standalone native frontend

`SettleGraph(Graph, region_ranks)` compiles the body and builds the encoded graph.
Ranks are positive int64 values with room for output rank and stride;
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
`cmake --build BUILD --target tidegraph-settle-check --parallel "$(python scripts/build_resources.py)"`.
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
profile already used by the Python frontend. General ports use the separate identity binding above. Neither binding proves arbitrary custom
modules or arbitrary model equivalence.
Local program sequence/clear/Next capabilities still control legal prefill.

## Fixed-topology specialization domains

The multi-node ring requires ordered singleton regions, edges i→(i+1)%N with
strictly positive delays, input at0 and output atN−1. It owns a logical-time loop
and propagates across each ring edge, including feedback and pending messages.
The diamond uses edges0→1,0→2,1→3,2→3 in that identity order and accepts either
four singleton regions or one shared middle region. Its fixed dependency layers
finish predecessors before constructing complete region/time fibers; middle
nodes compete under the actual Region program. Unequal positive edge delays
and ragged input clocks are legal. No generic scheduler or planner is called.

The Python layered specialization requires fully connected adjacent region
layers, one input per first-layer node and one output per last-layer node. It
uses scalar complete-frame formulas and a fixed layer/time/batch loop, with
independent output-boundary grouping. It does not call generic SettleGraph or
encoded frontier. Shared local modules, canonical sorting and emission mapping
do not replace that independent schedule. Wrong topologies fail explicitly.
