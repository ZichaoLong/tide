# Bounded device scheduler qualification

Clean source `f8648f766e6ded55eeb3eb886cb94bd109c55a7d` passed all84
independent CPU/NPU qualification cells. Two fresh standalone consumer builds
each passed four CTests. This qualifies the finite configurations below;
full-size performance and longer-window capacity are separate assessments.
The [adjacent JSON](bounded-scheduler-qualification-20260929.json) records exact
commands, source/build/topology hashes, tolerances, gate inventories and audits.
Raw files resolve through `artifacts/device-scheduler-JOB`.

The runtime implementation is ff251745; f8648f7 changes only isolated-VJP
validation and its documentation. The public C++ core hash remains
`5e342902e64abbc384099c3903317955e2eac2c9bfa599cd1a9ddf9eb9fbf439`.
Its existing public CPU/accelerator qualification remains separate; this report
does not claim a new full public regression run.

## Completed matrix

Both payload dtypes use FP32 Read/controls and FP32 optimizer masters/slots.
CPU uses eager execution; NPU cells cover eager and captured replay.

| Backend | Topology and dimensions | FP32 cells | FP16 cells |
| --- | --- | ---: | ---: |
| CPU | Tiny, D8/B2/V17/T3 | 4 | 4 |
| CPU | Actual465-node/4418-wire graph, D8/B1/V17/T3 | 4 | 4 |
| NPU1 | Tiny, D8/B2/V17/T3 | 8 | 8 |
| NPU2 | Tiny, D8/B2/V17/T3; includes peer primitive | 9 | 9 |
| NPU2 | Actual465-node/4418-wire graph, D8/B1/V17/T3; includes peer primitive | 9 | 9 |
| NPU8 | Tiny, D8/B2/V17/T3 | 8 | 8 |

Each family includes Add/Attention forward and complete training. The oracle is
an independent scalar CPU schedule with matching payload dtype. It compares
states, histories, physical-message identities, absence masks, selection,
cache contents, logits and isolated VJPs. Integer/discrete observations and
None versus connected-zero gradients remain exact. Training checks three SGD
and three AdamW updates, persistent owner state, gradients, FP32 masters/slots
and counters, with changing input trials. Optimizer guards check disconnected
owners, connected zero, nonfinite values and the finite update limit.

FP32 remains atol1e-6/rtol1e-5. FP16 uses explicitly recorded atol0.004/rtol0.02
on the tiny graph and atol0.02/rtol0.02 on the actual topology. These are
matching-dtype numerical envelopes, not promises of cross-dtype route identity
or training convergence. Earlier narrower FP16 failures remain retained.

The peer primitive changes strided/scalar payloads, bool masks and int64 values
above2^55 across replay trials. Its analytic VJP includes unused and
connected-zero owners. Multi-device training cuts physical communication edges
and propagates their cotangents explicitly; device-local differentiation still
uses Torch autograd. This is one process sharding a model, not DDP/HCCL training.

## What is resident

The [finite contract](../bounded-scheduler.md) fixes topology, shapes and an
empty initial graph state. Host construction expands the possible event window.
Device tensors carry presence, candidate/selection masks, exact int64 histories,
cache visibility and continuation decisions. Captured replay includes forward,
explicit first-order VJP and optimizer updates, with no per-event host tensor
scalar/index decision. The host still constructs/captures the program, supplies
inputs, launches each device graph and exports results/status at boundaries.

Peer transfers clone immutable source buffers, record source readiness through
IPC Notify, then wait/reset and pull on the destination consumer stream. Buffers
remain owned throughout the window. Ordinary cross-model autograd Events were
unsupported on this stack; the retained explicit bridge VJP avoids them.
Exact int64 sorting may execute on device AiCPU. AiCPU is on the accelerator;
residence does not imply every operation executes on AiCore.

The finite native replay path is verified on aarch64 Ascend910_9392/A3,
Torch/TorchNPU2.10 and CANN9.0 with TASK_QUEUE_ENABLE=0 and real standalone
loader closure. Dynamic topology, arbitrary imported state, checkpoints,
unbounded queues, HST, higher-order AD, other stacks and CUDA replay are outside
this qualification. Eight-device evidence uses the tiny graph; the actual
topology has independent CPU/two-device evidence. No full-size numerical or
12-token capacity claim follows from this table.

## Numerical investigation

Qualification02 retained two isolated-VJP failures: CPU actual-topology FP16
Add (maximum difference0.033325) and NPU2 actual-topology FP32 Add
(approximately1.6e-5). The original checker used `1.4*a` as the candidate
cotangent but differentiated `0.7*sum(b*b)` on the reference, introducing
different rounded upstream values. Sharing that cotangent fixed the CPU FP16
failure at the unchanged envelope. The radial squared-norm direction remained
sensitive to cancellation in normalized outputs.

`norm-probe21` reproduced the FP32 difference in the first-node SiLU/RMSNorm
primitive without scheduling or peer transfers. Against CPU FP64, maximum
errors were2.48616e-5 for CPU FP32 and1.71172e-5 for NPU FP32; the CPU/NPU
difference was1.56164e-5. Native and composite normalization agreed within each
backend. The runtime was not changed to imitate CPU rounding.

The v2 checker instead uses two common, output-independent cotangents made from
exact binary fractions, plus connected zero. Complete training objectives and
their VJPs remain separate and unchanged. FP32 tolerances were not widened.
Development diagnostics are retained but are not substitutes for the fresh
qualified03 builds and all84 terminal cells above.

## Record audit

All24 tracked wrapper smokes passed, covering CPU eager and NPU eager/replay,
both dtypes, both models and forward/training. Their timings establish entry
point and record operation only. The local Trackio0.35.0 projection used
best-effort recording; every stored step and metric matched its raw JSONL.
No dashboard was launched. The API has no per-event durable acknowledgement;
the database comparison supplies separate evidence.

`reports/audit-qualified03-final01/audit.json` verifies source, binary, core,
topology, terminal lifecycle, portable records and Trackio data. Its28 run
records include the24 successful smokes and four failed two-device12-token
capacity probes. Those probes failed with ACL207009 Notify resource exhaustion
and zero measured observations; a successful record audit does not change their
failure status. Capacity/performance follow-ups remain in STATUS/ROADMAP.

Fresh compilation and small gates used CPUs318/319 while full-size CPU timing
used separate affinities. They still overlapped in time and shared host memory
resources. Heavy throughput jobs use the task timing lock; this is not a claim
that every kind of task or external workload was isolated.
