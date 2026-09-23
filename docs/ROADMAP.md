# Graph execution foundation acceptance roadmap

This is the only backlog and stage acceptance index. STATUS owns the current
handoff; semantics owns the contract. The six-stage acceptance scope was frozen
on 2026-09-23. Historical M0–M8 evidence remains indexed by architecture.md and
Git history; its finite profiles do not certify the broader acceptance version.
Status: verified = cited completed evidence; implemented = code without the full
required gate; planned = required work remains. No submitted/running job passes.

## Six implementation classes

All required cells target CPU FP64/FP32, inference and first-order training.
Generic/specialized schedules must be independent; local kernels may be shared.
The final gate covers values, identities, every state slot, histories, Aggregate
contributions, Read/Next/comparison/Full/Emit, routes, pending, occurrence ledger,
isolated roots and input/parameter/initial-state VJPs, None/zero connectivity,
shared/unused owners, optimizer updates and continuation. HST uses its declared
surrogate VJP. Empty/ragged/parallel-edge/reset/delay/chunk regressions remain.

| Class | Actual implementation and existing gate | Acceptance gap / next unit |
| --- | --- | --- |
| PDG generic | `reference.py`, `cpp/src/stream.cpp`, cursor; CSR/CSC, feedback, lazy state, packed and node workers; [streaming](evidence/m1-streaming.md), [cursor](evidence/native-cursor.md), [isolated roots](evidence/isolated-autograd.md), [transport](evidence/packed-transport.md) | retain/regress; S3 option matrix, S5 performance |
| PDG specialized | `specialized.py`, `cpp/src/specialized.cpp`: independent self-loop propagation; `test_specialized.py`, `test_isolated_schedules.py` | ring verified; [stage2 gate](evidence/foundation-stage2.md) |
| TimedDAG generic | `frontier.py`, native planner/frontier/block; streaming is restricted PDG; [frontier](evidence/m3-frontier.md), Attention/SSM packed sequences | S3 applicable transport/scheduler options; S5 unified entry |
| TimedDAG specialized | independent Python/native singleton-region chain; [specializations](evidence/m4-settle-specialized.md) | diamond/paired-region verified; [stage2 gate](evidence/foundation-stage2.md) |
| SettleGraph generic | `settle.py`: independent Python region-major, Python encoding then native frontier/streaming; [encoding](evidence/m4-settle-specialized.md), [ports](evidence/local-ports.md), [origins](evidence/source-origins.md) | native `settle.h` [verified](evidence/native-settle-frontend.md) for rank-aligned profile |
| SettleGraph specialized | independent Python `settle_chain`, analytic formulas; Attention/SSM isolated roots | layered scalar schedule verified; [stage2 gate](evidence/foundation-stage2.md) |

## Stage gates and required units

| Stage/unit | Required delivery and verification | Status |
| --- | --- | --- |
| S1 | audit implementation/module/training/option matrices; freeze benchmarks, resources and stops; remove superseded tuning priority | verified by [scope audit](evidence/foundation-scope-audit.md) |
| S2.1 | standalone C++ SettleGraph spec, structural encoding, model alias mapping, inputs and complete-boundary projection; no Python dependency; standalone FP32/64 forward/VJP, negative validation and Python/native parity | verified: [native frontend](evidence/native-settle-frontend.md); 230 directed tests and clean C++-only FP64/FP32 gate |
| S2.2 | independent ring and diamond Python/C++ schedules, Python layered SettleGraph; full trace/isolated VJP/initial state/cuts and representative modules | verified; 7167-test [clean stage2 gate](evidence/foundation-stage2.md) |
| S2.3 | explicit Settle → TimedDAG → PDG clock, node/region/edge/source/port, state/history/message/pending/ledger mapping and cut restrictions; source-aware roots | verified for declared profile; [stage2 gate](evidence/foundation-stage2.md), contract in settle-embedding.md |
| S3.1 | actual module step/batch/sequence capabilities and counters, Attention/GQA/window, distinct same-fiber, Linear/Delta/Gated Delta/SSM, FFN/SwiGLU, Agg/Emit | representative kernels verified; existing `delta` is gated, ungated DeltaRule still required; [capability table](execution-capabilities.md); audit fallback reasons and extend new schedules |
| S3.2 | migrate packed_sources and batch_next/reset to legal frontier/encoded Settle blocks; compact/region parallel/deferred-release applicability with explicit errors; single-option, interaction and historical failure tests | implemented; 1098 directed CPU tests passed; stage3 clean gate pending |
| S3.3 | small deterministic model-style adapter composing layout, norm, explicit position/RoPE/mask/cache; CPU formula, chunk and VJP anchors | planned; RoPE/model adapter absent from current representative modules |
| S3 gate | nontrivial time batches and counted causal fallbacks; independent simple path, node parallel, packed/unpacked, default/option parity; no backward-speed claim from scalar replay | existing Attention/SSM batches verified; complete after S3.1–3.3 |
| S4.1 | six-class isolated training roots, shared/unused owners, None vs connected zero, multi-step SGD/momentum/AdamW + decoupled decay, eps=1e-5; detached/chunk/partial buffers | broad existing coverage; extend ring/diamond/layered/native Settle |
| S4.2 | interrupt/save/new-process restore/continue vs uninterrupted trajectory for promised single-graph, two-clock application and native value scopes | current in-process gates verified; audit/add explicit new-process integration |
| S4 gate | transactional malformed-input rejection and publication failure; graph/model/owner/alias/optimizer/group identity | [Python values](evidence/checkpoint-ownership.md), [publication](evidence/checkpoint-io.md), [native format](evidence/cpp-native-checkpoint.md), [application](evidence/token-checkpoint-coordinates.md) verified; retain, do not rebuild formats |
| S5.1 | unified smoke/non-smoke entry and fixed suite below; explicit nograd-forward/grad-forward/backward/optimizer/train-step implementations and bounds | existing portable LH/PDG runners are inference only; extension required |
| S5.2 | three graph families measured; both large shape presets counted and evaluated under resource preflight, timeout and process reaping; report achieved size and failures | wide PDG/LH evidence retained; narrow PDG timeout retained; DAG/Settle measurements pending |
| S5.3 | bounded optional tuning, if justified; all defaults evidence-based, at least three independent repeats for gain claims; export/rebuild/smoke from new directory | no new candidate selected; existing negative findings retained |
| S6 | freeze clean implementation commit, independent read-only worktree, complete CPU gates + standalone/relocation, source/build/binary/result/exit audit; evidence commit and final matrix | planned, after all required gates |

S6 finishes this acceptance version: all required correctness cells pass, fixed
performance assessment is honestly closed, no task live job remains, clean local
commit/status is explicit. No automatic extra platform/model/tuning work follows.

## Frozen performance suite

Machine-readable definitions: `benchmarks/foundation-v1.json`. Twelve logical
medium/small configurations only. Shapes/workloads cannot be multiplied as
variants; baseline/option/schedule variants keep the same logical workload.
P01/P02 deliberately form one fixed-touched-work topology-size comparison.
Use smoke D16/B4/V257/6 steps/warmup2 for portable LH/PDG; legal small chain,
diamond and layered equivalents for the other graphs. Smoke has no speed claim.

| ID | Workload, width/batch/sequence, static body nodes | Main comparison |
| --- | --- | --- |
| P01 | sparse PDG ring, D128/B8/T128/N128, 4 touched nodes | Python/native, scalar/packed |
| P02 | same touched subgraph plus dormant nodes, D128/B8/T128/N8192 | compare P01, sparse allocation and topology cost |
| P03 | source transport and region/node work, D512/B8/T128/N32, same-fiber attention | serial/workers, Agg/Emit/Next and opt-ins |
| T01 | TimedDAG diamond, D128/B8/T128/N4, event attention | streaming/frontier/independent diamond |
| T02 | TimedDAG four regions, D128/B32/T128/N16, SSM | legal prefill/causal fallback |
| S01 | Settle four layers, D128/B8/T128/N8, event GQA/window128 | generic/encoded/specialized/prefill |
| S02 | Settle chain, D512/B8/T128/N3, SSM/SwiGLU | step/packed sequence |
| A01 | TimedDAG chain, D128/B8/T2048/N2, ragged event GQA | long cache, padding, causal mask |
| A02 | TimedDAG diamond, D128/B8/T512/N4, same-fiber attention | exact/single, CSR/layout/cache |
| M01 | TimedDAG chain, D128/B8/T128/N3, Linear attention | step/chunk/sequence |
| M02 | TimedDAG chain, D128/B8/T128/N3, Gated Delta | step/chunk and counted fallback |
| TR01 | legal diamond shared by three families, D128/B8/T128/N4, SSM/SwiGLU, window16 | grad-forward, backward/replay, optimizer, complete step |

Large presets are limited to wide D2048/B512/nominal1:32 and narrow
D128/B512/nominal1:64. Wide reference is actual ~17.27B Attention, LH 224 leaves
+8 hubs per cortex; inet/onet states counted separately. Narrow target ~8–9B
and tens of thousands of nodes requires exact owner/work counts and staged
resource estimates before allocation. DAG/Settle use legal ranked topologies;
report parameter/active/KV/matrix-work differences. Existing wide evidence and
narrow failure are reusable; no mechanical repeat without a new question.

Every record includes workload/profile, shape/dtype, actual activation, owner
counts, touched nodes/edges, logical Agg/Upd/Next/Full, actual packing/kernels,
padding/matrix work, reset/warmup/detach/window, CPU/thread pools/NUMA, wall
latency/throughput/distribution, RSS/cgroup, correctness anchor, source/build/
binary hashes and terminal status. ms/sample-token divides measured wall time
by batch and measured effective steps; also publish batch-step latency.

## Resource and exploration limits

Recompute from affinity ∩ cpuset and all ancestor CPU quotas; use approximately
half effective CPUs in aggregate. Memory bound is half min(host total, current
MemAvailable, cgroup limits), across the whole task. Include parameters, grads,
optimizer, KV and transient/build costs before large allocation; measure actual
RSS/cgroup (RLIMIT_AS alone is not memory accounting). Stage-one observation:
320 usable logical/physical CPUs, no finite quota, budget160; memory budget about
755 GiB at audit, dynamic. Eight NUMA nodes; choose affinity from discovered IDs.
Correctness uses ATen/OpenMP/BLAS1, build2. Record effective pools, not just env.
No heavy job overlaps formal timing; large variants run sequentially. Durable
background.slice/Nice10 jobs have bounded timeouts and complete child reaping.

At most four new performance bottlenecks, two main candidates each. Currently
zero selected. Small correctness/cost probe first, at least three independent
repeats plus spread for speed claims. No benefit/regression/instability closes
an experiment; keep conservative defaults. Implementation/correctness work is
not capped by those exploration counts. Existing exact/single, fiber and packed
transport trials are closed evidence, not an invitation to restart wide search.

## Explicit extensions after acceptance

Optional Ascend starts with real single-card CPU-parity operators/forward/VJP,
maximum eight available cards; separate Python TorchNPU and C++ qualification.
CUDA, x86 execution on a different host, arbitrary open-weight imports, other
architectures, higher-order AD, full controller/RNG/data-cursor resume, arbitrary
LH configs and general proof for all graphs/modules are outside this gate.
Stable node locality and batched persistent KV remain optional bounded candidates,
not blockers or existing capabilities. LH original C++ is inference-only authority
for the bounded exact mapping; scale comparisons permit independent weights.
