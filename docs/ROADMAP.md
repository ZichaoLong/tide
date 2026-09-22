# Execution roadmap

Status vocabulary: planned / implementing / implemented / verified. A verified
milestone must link a report with exact source and commands. Partial coverage
stays explicit; later milestones may refine earlier interfaces.

| Milestone | Deliverable and acceptance | Status |
| --- | --- | --- |
| M0 | Re-entry, architecture, semantic lock, status, evidence and cleanup rules | implemented; durable records/re-entry [verified](evidence/durable-records.md) |
| M1 | Independent Python + C++ sealed streaming; cycles/delays/regions; sparse CSR/CSC; FP64/FP32 trace, VJP, continuation | verified for ema-ffn-v1; [evidence](evidence/m1-streaming.md) |
| M2 | Native serial/node-parallel + batch packing; sparse allocation/work counters; differential and thread/grad-mode tests | verified for ema-ffn-v1; [evidence](evidence/m1-streaming.md); performance pending |
| M3 | TimedDAG validation and frontier contracts; Python + native; actual time batching; region quotient cycles; independent DAG specialization | verified for ema-ffn-v1; [frontier](evidence/m3-frontier.md), [specialization](evidence/m4-settle-specialized.md) |
| M4 | SettleGraph executor + encoding; Python/native generic; independent Python specialization; embedded trace and backward correspondence | verified for ema-ffn-v1; [evidence](evidence/m4-settle-specialized.md) |
| M5 | Packed attention/GQA/window, linear attention, DeltaRule, SSM, FFN/SwiGLU; step/block equivalence; source-aware Agg and HARD/HST/SOFTP Emit | SSM/SwiGLU [verified](evidence/m5a-state-programs.md); Linear/Delta [verified](evidence/m5b-matrix-memory.md); event GQA/window [verified](evidence/m5c-attention.md); broader programs pending |
| M6 | Training roots, sharing, optimizer state, checkpoint/truncation and replay contracts; serial/parallel/packed/specialized validation matrix | initial profiles and isolated roots [verified](evidence/isolated-autograd.md); [prior evidence](evidence/m6-training-contracts.md); named Python ownership [verified](evidence/checkpoint-ownership.md); standalone C++ named ownership + SGD/AdamW parity [verified](evidence/cpp-optimizer-ownership.md); bounded two-clock/single-PDG training and single-graph resume [verified](evidence/single-graph-training.md); two-clock bundle and strict coordinates [verified](evidence/token-checkpoint-coordinates.md); broader modules/objectives pending |
| M7 | LH inference adapter using original C++; exact clock/readout/decay mapping; numerical qualification without changes to LH | [Selector](evidence/lh-selector.md), [Add](evidence/lh-add.md), [Full](evidence/lh-full.md), [same-fiber sum attention](evidence/lh-attention.md), [packing/CROSSBATCH](evidence/fiber-packing.md), [post-attention pooling](evidence/fiber-pooling.md), [token-window Pronounce](evidence/pronounce.md) and bounded [whole-model two-clock adapter](evidence/lh-iocortex.md) verified; bounded [single-PDG inference map](evidence/lh-single-graph.md) verified; composite checkpoint [verified](evidence/token-checkpoint-coordinates.md); wider configurations pending |
| M8 | Scale/performance qualification, sparse graph/activation workloads and retained evidence | first bounded EMA inference [pilot retained](evidence/m8-streaming-pilot.md); user-supplied 8.8B/8.5B [LH/Tide reference scales](lh-scale-benchmark.md) planned; other profiles, prefill and training pending |

## Dependencies and acceptance details

M1 establishes graph identity, canonical atom order, explicit input seals and
complete-cut continuation. M2 optimizes only after comparison with M1. M3/M4
share local contracts but use independent schedules. M5 can progress alongside
M3 after stable state interfaces; it must not claim arbitrary open-weight model
compatibility from only representative equations. M6 begins with M1 tests and
expands per feature. M7 begins with a fresh audit of the dirty LH sources and
read-only adapter build; discuss genuine semantic incompatibilities if found.

Required comparisons: positive-delay generic vs ring specialization; TimedDAG
streaming vs frontier vs chain/diamond specialization; SettleGraph direct vs
encoded TimedDAG vs layered specialization; Python vs native; native serial vs
node parallel; unbatched vs packed batch; step vs full/chunk prefill. Check
outputs, event trace, state/history, messages, inputs/parameter/state VJPs and
optimizer updates. Unavailable cells must fail or remain explicitly planned.

Performance requires separate evidence: touched nodes/edges, visits/allocations,
batch lengths, sequence block sizes, Full/Upd call counts, wall time and peak
memory. No speed claim follows from fewer calls or correct numerical results.

Owned native cursor and explicit snapshots are [qualified](evidence/native-cursor.md),
avoiding whole-state work per streaming cut. Remaining runtime work includes
allocation and structured Delta chunk optimization. Joint EMA/SSM batch/sequence
scans are [qualified](evidence/m5d-memory-packing.md).
Preserve simple paths as comparison anchors.
Stable local port layouts, native flat inverse indexes and SettleGraph remapping
are [qualified](evidence/local-ports.md). Logical source domains for exclusive
physical phase aliases are [qualified](evidence/source-domains.md). Local state
clocks are [qualified](evidence/state-clocks.md) (`state-clocks.md`).
Extensible Full/Emit programs and sparse
per-slot emissions are [qualified](evidence/full-programs.md). Source-aware
Aggregate profiles and extension seams are [qualified](evidence/aggregate-programs.md).
Graph-owned source-origin views for tag-sensitive custom programs under boundary
embedding are [qualified](evidence/source-origins.md). Complete-content propagation
is [qualified](evidence/content-programs.md). Independent Read and three region
modes are [qualified](evidence/read-programs.md). Full Next requests and prefill
capability gates are [qualified](evidence/next-programs.md). Region programs,
typed history, tensor controls and checkpoint v4 are
[qualified](evidence/region-programs.md) (`region-programs.md`). LH selection, explicit FP64 descriptor policy and the original-selector oracle
are [qualified](evidence/lh-selector.md) (`lh-selector.md`). The
tick-repeat Add/physical decode gate is [qualified](evidence/lh-add.md)
(`lazy-add.md`). LH activation/norm/signaling is [qualified](evidence/lh-full.md)
(`lh-full.md`). Same-fiber sum attention's scalar baseline is
[qualified](evidence/lh-attention.md) (`fiber-attention.md`); actual batch/sequence
packing and CROSSBATCH are [qualified](evidence/fiber-packing.md) (`fiber-packing.md`).
Post-attention Confluence is [qualified](evidence/fiber-pooling.md) (`fiber-pooling.md`).
Token-clock/readout is [qualified](evidence/pronounce.md), and actual IOCortexNet
two-clock inference is [qualified](evidence/lh-iocortex.md). Remaining single-PDG
and composite-checkpoint obligations: `lh-iocortex-plan.md`. Training comparison
will use Tide two-clock versus single-PDG semantics, with isolated output/state/
pending roots, HARD/SOFTP/HST VJPs, parameter aliases, optimizer steps and explicit
truncation of partial-window buffers. LH training is not an authority. The single-PDG map
must preserve occurrence ledgers when phases are absent, not infer them from time.
The bounded single-PDG oracle is [qualified](evidence/lh-single-graph.md)
(`lh-single-graph.md`). Its readout projection explicitly forgets the adapter-only ledger;
complete readout continuation equivalence requires additional occurrence state.

M8 first measurement covers owned native cursor versus the functional streaming
anchor: graph/model/engine construction, advance-only execution, explicit
snapshot/identity copies, fixed touched work under growing dormant topology,
serial/node-parallel and batch packing. Record raw repetition distributions,
work counters, process RSS, shared weights and exact workload/state reset policy.
Python graph identity JSON/SHA256 conversion and native canonical identity-string
copies are separate full-structure costs. Touched region history still copies
and validates its full maps; large single-region history needs its own workload.

M8 includes local experiments around two [large LH/Tide references](lh-scale-benchmark.md):
8.8B/width2048/batch512/nominal 1/32 and 8.5B/width128/batch512/nominal 1/64,
on a common 56-core CPU budget. These sizes/times are references, not strict
acceptance targets; the user authorized local exploratory runs. Recover static nodes/hubs, four-block topology,
actual parameter owners and selector semantics before importing. Generalize the
small oracle's fixed fixture into a reusable weight-preserving importer and
separate timed harness; retain exact-inference and comparable-scale lanes.
Historical times are user-reported amortized ms/sample-token; grad-stage and
historical dtype remain uncertain. Qualify nograd-forward and grad-forward
separately from backward/optimizer work. Use explicit FP32 for initial
reconstruction, bounded scale ramps and same-host comparisons, with source,
cache-age, NUMA, memory and work records. No historical timing is a speed gate.

M8 implementation observation: `ProjectionEmit` currently launches a matmul per
output slot, while original LH uses one large Linear per CSR row. Evaluate an
inference packing/cache or a separately specified single-matrix profile. Packing
independent Parameters during training can turn an unused Parameter's None VJP
into connected zero; preserve isolated-root connectivity. Any weight cache must
detect updates or require immutable weights. This is not measured speed evidence.

Isolated-root training exposed a packed autograd connectivity defect after the
903-test qualification. Local semantic replay corrects the tested boundary;
see [qualification](evidence/isolated-autograd.md) and `packed-autograd.md`.
Optimized packed backward must preserve
this contract before replacing the replay baseline; its training overhead is
part of M8 performance qualification.

Atomic exclusive value-checkpoint publication is
[qualified](evidence/checkpoint-io.md) with Linux CPU fault injection, no-overwrite
races, retry and resumed native training. Composite application checkpoints are [qualified](evidence/token-checkpoint-coordinates.md);
standalone C++ optimizer ownership and update parity are [qualified](evidence/cpp-optimizer-ownership.md);
native value serialization/resume is [qualified](evidence/cpp-native-checkpoint.md)
as the independent `TIDENCK1` schema.

Integer-coordinate validation at graph/execution/native-adapter and checkpoint
boundaries is [qualified](evidence/token-checkpoint-coordinates.md). Malformed
Python records must fail before mutation; cursor advance must not scan retained state.

## Next bounded increment

The standalone C++ owner registry, independent LibTorch SGD/AdamW updates and
the `TIDENCK1` native value format are implemented. The next M6 interface
increment is to audit and then implement the standalone C++ SettleGraph
construction/encoding frontend required for fully independent C++ use. Keep
graph construction, embedding, codec and publication layers separate; current
native execution still consumes the ordinary encoded `Graph` produced by the
Python frontend. The native checkpoint format remains independent of Python
files and graph continuation.

The M8 large-LH workstream first needs an allocation-free graph/parameter audit
and a reusable importer/timing entry point following `lh-scale-benchmark.md`;
the two fixed targets are not runnable by changing the small oracle constants.
Expand the other M8 fixed workloads beyond the EMA pilot: Attention/SSM streaming,
frontier/SettleGraph prefill and measured backward/replay costs. Declare batch,
sequence length, width, active/dormant topology, sharing/reset policy and exact
parity anchors before timing. No parameter sweep or scale claim follows from
the existing small pilot. Broader model-specific imports and Delta chunk
optimization remain independent items above.
