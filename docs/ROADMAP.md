# Graph execution foundation acceptance roadmap

This is the only backlog and stage acceptance index. STATUS owns the current
handoff; semantics owns the contract. The six-stage acceptance scope was frozen
on 2026-09-23 and is now complete; [final qualification](evidence/foundation-final.md)
is the delivery report. No unit of that acceptance version remains pending;
the separately authorized cross-family extension below is now active. Historical
M0–M8 evidence remains indexed by architecture.md and Git history; its finite profiles do not certify the broader acceptance version.
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

| Class | Actual implementation and existing gate | Accepted scope / evidence |
| --- | --- | --- |
| PDG generic | `reference.py`, `cpp/src/stream.cpp`, cursor; CSR/CSC, feedback, lazy state, packed and node workers; [streaming](evidence/m1-streaming.md), [cursor](evidence/native-cursor.md), [isolated roots](evidence/isolated-autograd.md), [transport](evidence/packed-transport.md) | verified: [final7741-test gate](evidence/foundation-final.md), options and bounded performance |
| PDG specialized | `specialized.py`, `cpp/src/specialized.cpp`: independent self-loop/ring propagation; `test_specialized.py`, `test_isolated_schedules.py` | ring verified; [stage2 gate](evidence/foundation-stage2.md) |
| TimedDAG generic | `frontier.py`, native planner/frontier/block; streaming is restricted PDG; [frontier](evidence/m3-frontier.md), Attention/SSM packed sequences | verified: migrated options, real prefill, [unified entry](foundation-benchmarks.md) and final gate |
| TimedDAG specialized | independent Python/native chain/diamond, including shared middle region; [specializations](evidence/m4-settle-specialized.md) | diamond/paired-region verified; [stage2 gate](evidence/foundation-stage2.md) |
| SettleGraph generic | `settle.py`: independent Python region-major; independent native `settle.h` frontend using frontier, with encoded streaming comparison; [encoding](evidence/m4-settle-specialized.md), [ports](evidence/local-ports.md), [origins](evidence/source-origins.md) | native `settle.h` [verified](evidence/native-settle-frontend.md) for rank-aligned profile |
| SettleGraph specialized | independent Python `settle_chain`, analytic formulas; Attention/SSM isolated roots | layered scalar schedule verified; [stage2 gate](evidence/foundation-stage2.md) |

## Stage gates and required units

| Stage/unit | Required delivery and verification | Status |
| --- | --- | --- |
| S1 | audit implementation/module/training/option matrices; freeze benchmarks, resources and stops; remove superseded tuning priority | verified by [scope audit](evidence/foundation-scope-audit.md) |
| S2.1 | standalone C++ SettleGraph spec, structural encoding, model alias mapping, inputs and complete-boundary projection; no Python dependency; standalone FP32/64 forward/VJP, negative validation and Python/native parity | verified: [native frontend](evidence/native-settle-frontend.md); 230 directed tests and clean C++-only FP64/FP32 gate |
| S2.2 | independent ring and diamond Python/C++ schedules, Python layered SettleGraph; full trace/isolated VJP/initial state/cuts and representative modules | verified; 7167-test [clean stage2 gate](evidence/foundation-stage2.md) |
| S2.3 | explicit Settle → TimedDAG → PDG clock, node/region/edge/source/port, state/history/message/pending/ledger mapping and cut restrictions; source-aware roots | verified for declared profile; [stage2 gate](evidence/foundation-stage2.md), contract in settle-embedding.md |
| S3.1 | actual module step/batch/sequence capabilities and counters, Attention/GQA/window, distinct same-fiber, Linear/Delta/Gated Delta/SSM, FFN/SwiGLU, Agg/Emit | verified including separately named ungated DeltaRule;736 directed tests, S3/S4 and final7741-test gate passed; [capability table](execution-capabilities.md); fallback and new-schedule gates passed |
| S3.2 | migrate packed_sources and batch_next/reset to legal frontier/encoded Settle blocks; compact/region parallel/deferred-release applicability with explicit errors; single-option, interaction and historical failure tests | verified:1098 directed CPU tests, [S3/S4 gate](evidence/foundation-stage34.md) and final gate |
| S3.3 | small deterministic model-style adapter composing layout, norm, explicit position/RoPE/mask/cache; CPU formula, chunk and VJP anchors | verified: tiny explicit RMSNorm/RoPE/GQA/SwiGLU adapter; independent formula/cache/chunk/VJP and final gate passed |
| S3 gate | nontrivial time batches and counted causal fallbacks; independent simple path, node parallel, packed/unpacked, default/option parity; no backward-speed claim from scalar replay | directed option/module gates passed; 7611-test [S3/S4 clean gate](evidence/foundation-stage34.md) passed |
| S4.1 | six-class isolated training roots, shared/unused owners, None vs connected zero, multi-step SGD/momentum/AdamW + decoupled decay, eps=1e-5; detached/chunk/partial buffers | verified extension;274 directed tests plus54 corrected multi-window Settle tests passed; [S3/S4 clean gate](evidence/foundation-stage34.md) passed |
| S4.2 | interrupt/save/new-process restore/continue vs uninterrupted trajectory for promised single-graph, two-clock application and native value scopes | verified:36 fresh subprocesses cover both dtypes, SGD/momentum and AdamW, single/application/native schemas; final payload/identity audit passed |
| S4 gate | transactional malformed-input rejection and publication failure; graph/model/owner/alias/optimizer/group identity | [Python values](evidence/checkpoint-ownership.md), [publication](evidence/checkpoint-io.md), [native format](evidence/cpp-native-checkpoint.md), [application](evidence/token-checkpoint-coordinates.md) verified; retain, do not rebuild formats |
| S5.1 | unified smoke/non-smoke entry and fixed suite below; explicit nograd-forward/grad-forward/backward/optimizer/train-step implementations and bounds | verified unified graph runner;130 directed runner/CLI tests and final gate passed; [entry/modes](foundation-benchmarks.md); LH/PDG portable kit remains inference-only |
| S5.2 | three graph families measured; both large shape presets counted and evaluated under resource preflight, timeout and process reaping; report achieved size and failures | all108 medium runs completed; [medium evidence](evidence/foundation-medium.md); [large assessment](evidence/foundation-large.md) reviewed:7 complete/5 timeout/1 unlaunched across new runs; wide PDG/LH reused, corrected16.608B historical narrow count and frozen8.497B supplement; target-scale success not claimed |
| S5.3 | bounded optional tuning, if justified; all defaults evidence-based, at least three independent repeats for gain claims; export/rebuild/smoke from new directory | closed:zero new candidates, existing negative findings/defaults retained; fresh relocated rebuild/all17 smoke variants passed |
| S6 | freeze clean implementation commit, independent read-only worktree, complete CPU gates + standalone/relocation, source/build/binary/result/exit audit; evidence commit and final matrix | verified: [final gate](evidence/foundation-final.md), clean81a1b26,7741 tests,17 relocated smoke,36 fresh-process payloads and separate no-Python standalone build; all task units terminal |

S6 has finished this acceptance version: all required correctness cells passed,
fixed performance assessment is honestly closed, and no task live job remains.
The final local commit/status is in STATUS. No extra platform/model/tuning work follows.

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

A new bounded performance increment considers at most four bottlenecks and two
main candidates each. The frozen acceptance selected none; separately authorized
Full and Aggregate follow-ups are recorded below. Small correctness/cost probe
first, at least three independent repeats plus spread for speed claims. No benefit/regression/instability closes
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

## User-requested follow-up: Add scale comparison

2026-09-24 completed: configurable Add/grad-forward/outer-timer comparison;
472 directed CPU FP64/FP32 tests,4 fresh native smoke runs and12 D2048/B512/56-core
runs (3 repeats per engine/mode), all passed. [Reviewed results](evidence/add-scale-comparison.md)
retain the exact9,468,020,899 count, overlapping nograd repeat ranges and15.18x
PDG/LH grad-forward median latency. [Contract](add-scale-comparison.md).
No backward-throughput or arbitrary model/backend claim follows.

The subsequent Full/Aggregate follow-ups below implement isolated packed VJPs
while retaining scalar replay, None/connected-zero and ownership. Remaining
extensions include separating startup/steady-state limits in parallel large
sparse runs, costly Linear/Delta scans and one concrete model adapter. Existing
CPU/memory/Ascend bounds,
independent anchors and repeat requirements apply across all graph families.
This recommendation does not reopen completed acceptance or queue another sweep.

## Completed follow-up: PDG grad-forward

2026-09-24: optional native Full batched affine VJP, default replay retained.
[Evidence](evidence/full-batched-autograd.md), [contract](full-batched-autograd.md).
Clean690-test selected FP64/FP32 gate,4 smoke/2 probes/4 wide runs all passed.
D2048/B512 optimized median10.49906ms/sample-token (3 repeats), versus current
same-binary replay110.43917 (1 repeat) and retained replay108.23288 (3 repeats).
No backward-throughput or all-model/platform claim. Implementation reusable by
legal frontier/Settle calls with bounded correctness; no speed claim there.

The subsequent follow-up below supplies Aggregate/state/Read attribution and
small actual complete-training timing. Do not repeat this Full sweep without
a new question. Earlier resource bounds, independent anchors, honest failure records and three-repeat gain checks apply.

## Completed follow-up: Aggregate VJP and remaining grad-forward costs

2026-09-24: Full-batched diagnostics selected Aggregate for the wide Add case;
Attention small state replay remains a separate candidate. Optional native
`aggregate_autograd=replay|batched` is independent of Full; default replay stays.
Source scaling/reduction VJPs are batched, while normalization preserves per-event
Jacobians and near-zero AdamW behavior. [Contract](aggregate-batched-autograd.md),
[reviewed evidence](evidence/aggregate-batched-autograd.md).

Clean 552-test selected CPU FP64/FP32 gate passed after an independent build.
All 25 runs passed: 4 smoke, 2 cost probes, 12 small actual training, 6 wide Add
forwards and 1 separate diagnostic. No tolerance changes. Complete training
owner values/final gradients/losses match across all six small replay/batched
pairs. Source/archive/build/packet and terminal-record audits passed.

D2048/B512/56 CPUs: replay versus batched Aggregate median 11.61919 versus
8.55906 ms/sample-token, with Full batched throughout and 3 processes per policy.
Latency -26.34%, throughput +35.75%; peak RSS 77.43–78.76 versus 73.54–74.09 GiB.
Small D128/B16/T16, median total time across 4 training windows: Add 1.44187→1.04816s,
Attention 5.47745→4.98471s. These small timings do not certify wide-model training.
No new LH or wide no-grad performance result. Portable kit and complete commands
are linked from the evidence; all jobs are terminal, no task is queued.

Next choices, only for a separately requested bounded increment: wide Add now
spends most coordinator time in Full (2.77939s versus Update 0.59177s in a separate
diagnostic), with scalar Full/Aggregate replay zero. Profile within that phase
before choosing another large-Add kernel. For Attention, state/KV VJPs have the
stronger small-probe signal; retain replay and test backward/optimizer as well.
State/Read replay remains. Shared native TimedDAG/Settle paths have bounded
correctness coverage, not a new throughput claim. Resource/repeat limits above
apply across graph families; acceptance stays closed.


## Active extension: cross-family streaming and prefill policies

Authorized 2026-09-24. This finite extension reuses the completed PDG performance
policies in TimedDAG and SettleGraph, for streaming AND legal prefill. It does
not reopen the original six-stage acceptance or remove any semantic roots.
Defaults remain conservative. STATUS owns progress/next commands, this section
owns the required remaining work. Historical foundation-v1 stays frozen.

| Unit | Required result | Status |
| --- | --- | --- |
| E1 | Versioned option/capability and benchmark definitions: separate scheduler, kernel, model-layout and application-head policies; requested/resolved/fallback records; reject unsupported explicit requests | implemented; clean qualification pending |
| E2 | Python independent chain/diamond and Settle layered/chain block schedules; retain scalar schedules; Python isolated Full/Aggregate batched VJPs; native streaming/frontier/specialized/native-Settle uptake and actual sequence/fallback counters | implemented; clean qualification pending |
| E3 | CPU FP64/FP32 full-observable, isolated VJP, aliases/None/zero, optimizer/checkpoint/cut/detach/snapshot and prefill-to-streaming policy-switch tests; targeted combinations plus individual policies | directed1632 + final CLI7 passed; clean qualification pending |
| E4 | Fixed bounded smoke/medium/large assessment below; separate prefill, streaming, transition, forward/backward/optimizer/whole-step timings and actual paths; explicit terminal failures/limits | planned |
| E5 | Frozen clean independent build/gates, relocated export/rebuild/smoke, source/binary/result audit, reviewed evidence and coherent commits; no live task job | planned |

Coverage: PDG generic/specialized are regression baselines. TimedDAG generic
Python/native: streaming, frontier, legal blocks and transition. TimedDAG
specialized Python/native: independent chain/diamond step and block schedules.
Settle generic Python/native: region/block prefill, encoded streaming, chunk
continuation and the native frontend. Settle specialized at least Python:
independent scalar and layered/chain block schedules. Python node workers are
not required; native core stays independent of Python.

Policies include workers/packed/prefill, Full/Aggregate replay|batched,
packed_sources/batch_next, compact_events/parallel_regions/defer_state_release,
applicable exact|single attention packing, scalar|CSR pooling, cloned|owned KV,
event|head layout and input|linear projection layout. Head workers belong to
application DenseLinear, absent in graph-only workloads. A missing implementation
is not semantic N/A. Known causal gates (selected adoption, clear, custom Next,
no exact state sequence) retain observable fallbacks. State/Read replay remains
separately counted; removing all replay is outside this extension.

Fixed measurement budget: at most 12 logical workloads, at most 6 formal
variants each, 3 independent processes for speed claims. Smoke D16/B4/T6;
medium D128/D512, B8/B32, T128/T512, ragged attention T2048 where needed.
Large wide D2048/B512/nominal1:32, Add ~9.468B or Attention ~17B; narrow
D128/B512/nominal1:64 ~8-9B is assessed in resource stages. Count actual
parameters, active work, topology/encoding boundaries and explain differences;
do not repeat known timeouts without a new policy/workload question.

Dynamic aggregate half-effective-CPU/half-memory limits apply to every family;
no fixed256GiB cap. Ascend at most8 cards only for an explicit later backend
extension; this gate requires CPU FP32/FP64. Existing durable-job, frozen-build,
reset, repeats, failure retention and timing denominator rules apply.

Completion requires E1-E5, real sequence batching where legal, independent
reference/specialized schedules, passing mandatory correctness cells and an
honestly terminal fixed performance assessment. Every option need not speed up.
Documenting a mandatory implementation gap does not complete that unit.
