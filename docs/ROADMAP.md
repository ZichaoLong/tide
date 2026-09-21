# Execution roadmap

Status vocabulary: planned / implementing / implemented / verified. A verified
milestone must link a report with exact source and commands. Partial coverage
stays explicit; later milestones may refine earlier interfaces.

| Milestone | Deliverable and acceptance | Status |
| --- | --- | --- |
| M0 | Re-entry, architecture, semantic lock, status, evidence and cleanup rules | implemented |
| M1 | Independent Python + C++ sealed streaming; cycles/delays/regions; sparse CSR/CSC; FP64/FP32 trace, VJP, continuation | implementing |
| M2 | Native serial/node-parallel + batch packing; sparse allocation/work counters; differential and thread/grad-mode tests | planned |
| M3 | TimedDAG validation and frontier contracts; Python + native; actual time batching; region quotient cycles; independent DAG specialization | planned |
| M4 | SettleGraph executor + encoding; Python/native generic; independent Python specialization; embedded trace and backward correspondence | planned |
| M5 | Packed attention/GQA/window, linear attention, DeltaRule, SSM, FFN/SwiGLU; step/block equivalence; source-aware Agg and HARD/HST/SOFTP Emit | planned |
| M6 | Training roots, sharing, optimizer state, checkpoint/truncation and replay contracts; serial/parallel/packed/specialized validation matrix | planned |
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
