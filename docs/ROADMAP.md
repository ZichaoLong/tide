# Execution roadmap

Status vocabulary: planned / implementing / implemented / verified. A verified
milestone must link a report with exact source and commands. Partial coverage
stays explicit; later milestones may refine earlier interfaces.

| Milestone | Deliverable and acceptance | Status |
| --- | --- | --- |
| M0 | Re-entry, architecture, semantic lock, status, evidence and cleanup rules | implemented |
| M1 | Independent Python + C++ sealed streaming; cycles/delays/regions; sparse CSR/CSC; FP64/FP32 trace, VJP, continuation | verified for ema-ffn-v1; [evidence](evidence/m1-streaming.md) |
| M2 | Native serial/node-parallel + batch packing; sparse allocation/work counters; differential and thread/grad-mode tests | verified for ema-ffn-v1; [evidence](evidence/m1-streaming.md); performance pending |
| M3 | TimedDAG validation and frontier contracts; Python + native; actual time batching; region quotient cycles; independent DAG specialization | verified for ema-ffn-v1; [frontier](evidence/m3-frontier.md), [specialization](evidence/m4-settle-specialized.md) |
| M4 | SettleGraph executor + encoding; Python/native generic; independent Python specialization; embedded trace and backward correspondence | verified for ema-ffn-v1; [evidence](evidence/m4-settle-specialized.md) |
| M5 | Packed attention/GQA/window, linear attention, DeltaRule, SSM, FFN/SwiGLU; step/block equivalence; source-aware Agg and HARD/HST/SOFTP Emit | SSM/SwiGLU [verified](evidence/m5a-state-programs.md); Linear/Delta [verified](evidence/m5b-matrix-memory.md); event GQA/window [verified](evidence/m5c-attention.md); broader programs pending |
| M6 | Training roots, sharing, optimizer state, checkpoint/truncation and replay contracts; serial/parallel/packed/specialized validation matrix | initial profiles and isolated roots [verified](evidence/isolated-autograd.md); [prior evidence](evidence/m6-training-contracts.md); broader modules/objectives pending |
| M7 | LH inference adapter using original C++; exact clock/readout/decay mapping; numerical qualification without changes to LH | planned |
| M8 | Scale/performance qualification, sparse graph/activation workloads and retained evidence | planned |

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
are [qualified](evidence/local-ports.md). Extensible Full/Emit programs and sparse
per-slot emissions are [qualified](evidence/full-programs.md). Source-aware
Aggregate profiles and extension seams are [qualified](evidence/aggregate-programs.md).
Graph-owned source-origin views for tag-sensitive custom programs under boundary
embedding are [qualified](evidence/source-origins.md). Complete-content propagation
is [qualified](evidence/content-programs.md); Read/Next are the next program
gates (`next-read-plan.md`).

Isolated-root training exposed a packed autograd connectivity defect after the
903-test qualification. Local semantic replay corrects the tested boundary;
see [qualification](evidence/isolated-autograd.md) and `packed-autograd.md`.
Optimized packed backward must preserve
this contract before replacing the replay baseline; its training overhead is
part of M8 performance qualification.
