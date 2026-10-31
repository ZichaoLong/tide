# Execution capability audit

## Core-5 weighted binding

The current mathematical source is `tide-core-5` at `0770593`, pinned in
[upstream.json](upstream.json). The [semantic map](semantic-map.md) connects its
definitions, public bindings, independent tests and fixed-source evidence.
The table below describes execution layers; each linked report retains its own
source, dtype, topology, configuration and acceptance scope. It is not one
Cartesian qualification of every option.

Current natural-scale settings and failures are in the [25-configuration study](evidence/core5-natural-study-20261029.md)
and [large checkpoint follow-up](evidence/core5-checkpoint-budget-20261029.md).
The [corrected recipe qualification](evidence/core5-coverage-correction-20261030.md) covers 26 small models through PyTorch and Python/native on CPU FP64 (52 training cases, 73 related checks total). It adds no standalone, accelerator, or performance qualification.
The [configuration/coverage query](core5-study.md), backed by the [per-configuration index](evidence/core5-index.json), separates exact small correctness cases from
performance runs; it never extends a passing small case to another size, dtype, client or source.
The [consumer report's coverage correction](evidence/core5-study-clients-20261029.md) narrows its original
single-layer FC checks. Historical passing counts remain unchanged.

[Parameter-matched workload evidence](evidence/core5-performance-transfer-20261028.md)
adds fixed CPU/NPU gates, storage/copy changes and actual large material. D64
FP32 complete records, gradients and optimizer values can fail strict checks;
the original D2048 residual refusal has a separate local numerical repair. Its
FP32 complete records still fail; explicit CPU FP64 has bounded complete-record
qualification in the [precision report](evidence/core5-stable-preparation-precision-20261028.md).
The [new boundary checks](evidence/core5-nongpu-boundaries-20261029.md) clear the
original SDK fc/affine mixed-a/c witnesses while retaining FP16 GDN, FP32
feedback and default asynchronous NPU peer failures. These historical boundaries
do not imply a CUDA result. The [RTX3090 eager report](evidence/core5-cuda-eager-20261031.md)
adds separately scoped actual x86_64 CUDA evidence and preserves its new strict failures.
The [GPU inference study](evidence/core5-cuda-inference-20261031.md) adds independently checked ragged batches, explicit FP64 scale configurations, complete-return packing and client-specific memory boundaries; inference does not qualify large-scale training.

[Independent-input KV batches](evidence/core5-independent-projection-20261029.md)
qualify 8d97020 on CPU (273 passed, two original FP16 strict xfails), pure
PyTorch and Python/native NPU, and freshly rebuilt standalone SDK core/resident
consumers. Public resident KV has 12 checks; SDK flow/training/placed-node
training passes both dtypes. The added optional virtual capability requires
rebuilding the core and its consumers; this revision has no new CUDA build or
GPU execution evidence. Three paired complete-training measurements show no
clear speed improvement. Matched D64 CPU FP64 complete gradients/optimizer and
exact next-window resume pass; original FP32 and asynchronous failures remain.

| Layer | Implemented interface and evidence boundary |
| --- | --- |
| General graphs and ports | PositiveDelayGraph, nonaligned TimedDAG and general SettleGraph with same-rank regions; [Settle embedding](settle-embedding.md), [typed records](structured-records.md) |
| CPU mathematical baseline | Independent Python schedules and C++ core; five weighted memories, formal sending/scoring/evidence/key configurations and fixed-time convolution; [binding](weighted-message.md), [recurrent functions](weighted-recurrent.md), [sending](weighted-sending.md), [scoring](weighted-scoring.md) |
| Exact joint evaluation | Online greedy prefill, actual ragged projection/affine scans, independent field roots, batched local VJPs and explicit sharing; [packing](weighted-packed.md), [region batches](weighted-region-batch.md), [sharing](weighted-compute-sharing.md) |
| Mixed execution | Explicit payload/Read/control/selection placement; [placement](weighted-placement.md), [device qualification](weighted-device-validation.md), [precision](weighted-precision.md). Python/native and standalone evidence remain separate |
| Resident single device | Independent device readiness/selection/progress, complete fields and raw cuts, retained first-order training, canonical FP32 masters and SGD/AdamW; [FP32/FP16 qualification](evidence/core5-resident-half-training-20261028.md) and [master assertion correction](evidence/core5-resident-weighted-peers-20261028.md) |
| Resident sample replicas | Private graph states, canonical gradient sum and one optimizer boundary; [two-device qualification](evidence/core5-resident-weighted-peers-20261028.md) |
| Resident node placement | One sample split by node, complete regional selection and exact cuts; [forward qualification](evidence/core5-weighted-node-placement-20261028.md). Placed window VJP, canonical optimizer publication, FP32/FP16 training and relocated restore have [independent qualification](evidence/core5-weighted-node-training-20261028.md) |
| CUDA | [x86_64 RTX3090/SM86 eager evidence](evidence/core5-cuda-eager-20261031.md): PyTorch/native FP32 and FP16 each19/19, standalone23/24 and22/24; complete update/resume checks. Ordinary peer copies/explicit Python CPU bridge have scoped evidence; native peer GDN exact resume fails. [Single-device resident](evidence/core5-cuda-resident-20261031.md) has56 component/dtype and53 public checks, zero skips. [FC-small timeline and finite costs](evidence/core5-cuda-finite-20261031.md) separately qualify the actual small configurations; medium and HB-small resident strict differences remain. aarch64 still has build/loader evidence only |
| Performance | [Natural-scale settings/results](evidence/core5-natural-study-20261029.md): 25 configurations, 7 fully checked CPU FP64 results including short-window LH-inspired 1.051B. NPU numerical/resource failures remain; 8B/16B are static plans. Earlier [small/medium comparison](evidence/core5-weighted-performance-20261028.md) and [startup observations](evidence/core5-weighted-observation-20261028.md) retain their own scope; no general large-batch or cross-card throughput claim |

CPU FP64/FP32 remain the reference. NPU reports distinguish Torch2.12/CANN9.1
Python-owned code from Torch2.10/CANN9.0 standalone SDK code. Resident FP16 has
FP32 arithmetic, adjoints and masters with explicit storage-boundary casts;
it does not certify the different eager FP16 path. Explicit unavailable devices,
dtypes, capacities and configurations fail. The retained strict eager GDN,
asynchronous peer and PDG rounding failures, and the separately cleared SDK affine witnesses, are linked from
[STATUS](STATUS.md) and the relevant reports. Source/build evidence never
substitutes for device execution.

## Legacy generic-module qualification

The remaining tables describe the earlier generic module catalog, independently
of the weighted binding above. Qualified source:
`81a1b266af49d918aa6e1587e4ed9e0c4d4e5eb5`;
[7741-test CPU gate](evidence/foundation-final.md). Later legacy extensions retain
their own evidence below; none of these counts certifies core-5 weighted functions.
Current backlog remains in [ROADMAP](ROADMAP.md). CPU FP32/FP64 required.
Device support and finite CUDA/NPU qualification are separate from schedule
capabilities; see [accelerators.md](accelerators.md). Python schedules are serial with batch/sequence
support. Native schedules support node workers. Training denotes first-order
semantic parity, not generally optimized backward; default packed state/Read/Full
replay remains. The optional [native Full affine VJP](full-batched-autograd.md)
removes Full projection replay for its declared profiles. The optional
[Aggregate source VJP](aggregate-batched-autograd.md) batches built-in source
programs with isolated roots and per-event normalization backward. Its
[bounded qualification and performance evidence](evidence/aggregate-batched-autograd.md)
separates graph correctness, wide grad-forward and small complete training.

## Legacy cross-family extension

[cross-family-policies.md](cross-family-policies.md) describes the implemented
v2 interfaces: Python packed tick streaming, independent
chain/diamond/Settle block schedules, Python isolated Full/Aggregate VJPs,
native Settle streaming/frontend transitions, shared native causal state batches
and separated scheduler/kernel/layout benchmark options. Its historical gates
are linked from that contract and the [evidence index](evidence/README.md).
This scope is separate from the current weighted implementation.

## Graph × schedule × option × mode

P = PDG, D = TimedDAG, S = encoded SettleGraph. D/S streaming use ordinary PDG
kernel and preserve their legal topology. Native S construction is [verified](evidence/native-settle-frontend.md) in `settle.h`.
"verified" refers to the existing bounded modules/tests below, not every custom
program. Infer/train entries share values; training uses scalar semantic replay.

| Option / applicable module | P/D/S streaming, infer/train | D/S frontier, infer/train | Independent specialization |
| --- | --- | --- | --- |
| node workers, packed state/Full | verified | verified | native self-loop/ring/chain/diamond verified; Python Settle specializations serial |
| attention_packing exact/single, same-fiber | verified | verified, `test_fiber_single.py` | ring/diamond/chain/self-loop policies covered by frontier-options gate |
| fiber_pooling event/CSR, same-fiber | verified | verified, `test_fiber_efficiency.py` | exact default covered; options covered by frontier-options gate |
| cloned/owned KV, same-fiber | verified | verified, same test | default covered |
| attention layout event/head, same-fiber | verified | verified, same test | default covered |
| projection input/linear layout | same-fiber QKV/output physical strides selected by scale model initialization; [fiber efficiency](evidence/fiber-efficiency.md) | Model projection_layout selects physical QKV/output strides; cross-graph ownership/update tests | default projection covered |
| compact events | native Streaming, trace snapshots retained | consumed fibers moved, trace snapshots retained | same block publication |
| parallel regions | native Streaming, canonical publish order | same-time independent sample owners, canonical publication | same causal waves |
| deferred state release | compact required | compact + trace-free displaced states retire on workers | same block cleanup |
| packed_sources | packed sum transport; counted unsupported Agg fallback | Aggregate and same-fiber sequence transport | independent ring/diamond/chain/self-loop |
| batch_next/reset | packed adopt/selected fiber reset; counted custom fallback | same-time batch owners; times stay causal | same causal Next/reset grouping |
| head workers | native DenseLinear application vocabulary head; separate from graph node pool | same DenseLinear can consume frontier/Settle output; this pool is outside graph scheduling | same application DenseLinear, not an attention-head worker pool |

Worker support does not imply every specialization exposes multiple nodes in
one dispatch. In `cpp/src/specialized.cpp`, self-loop/ring visit nodes before
`evaluate_block`, so their Aggregate/Full dispatch has one node at a time;
sample/region work can still be parallel. No cyclic-specialization node-speedup
claim follows from accepting a worker count. Generic Streaming exposes the
multi-node path measured by P03 and the wide LH-scale comparisons.

S3.2 directed gate: `tests/test_frontier_options.py` and eight related files,
1098 CPU FP64/FP32 tests passed; archived source and terminal audit at
`artifacts/frontier-options-dev-20260923-a/`. Clean7611-test qualification: [S3/S4 evidence](evidence/foundation-stage34.md).
Defaults stay unchanged. Prefill fallback counters distinguish disabled, missing
sequence contract, selected-only adoption, selected clear, and custom Next.
Phase profiling remains explicitly Streaming-only.

Nondefault same-fiber policies on scalar paths report `fiber_policy_scalar_events`;
training oracle calls report `fiber_policy_semantic_replays`. Packed numeric
policies remain distinct from the independent scalar reference.

A local same-fiber policy does not apply to aggregated-event attention kernels;
these remain separate semantic profiles. Explicit unsupported configuration
must fail, never be accepted and ignored. Replay, fallback cause/count and actual
batch lengths must be reported. Existing frontend restrictions are implementation
gaps when the scheduling contract allows the operation, not semantic N/A.

## Modules and batch contracts

| Program | Code (Python / native) | Step / batch / legal sequence; independent gate |
| --- | --- | --- |
| EMA/identity/SSM | memory.py, memory_batch.py / basic_kernels.cpp | exact affine batch/time scan; `test_memory_programs.py`, `test_memory_packing.py` |
| event Attention/GQA/window | attention.py / attention.cpp | ragged KV, grouped batch and causal sequence; `test_attention.py`, `test_attention_packing.py`, `test_attention_schedules.py`, `test_attention_training.py` |
| same-fiber Attention/pooling | fiber_attention.py, fiber_packing.py, fiber_pool.py / corresponding native files | complete fiber visibility; distinct KV/log-bias decay; packed batch/time; fiber formula/schedule/pool/continuation/single/efficiency tests |
| Linear/Gated Delta/DeltaRule (`linear`/`delta`/`delta-rule-v1`) | matrix_memory.py / matrix_kernels.cpp | literal steps and exact scans; ungated profile directed-tested; clean S3/S4 gate passed; `test_matrix_memory.py`, isolated roots |
| FFN/SwiGLU/identity/norm Full | ops.py, lh_full.py / ops.cpp, lh_full.cpp | independent selected rows batched; `test_memory_programs.py`, `test_lh_full_formulas.py` |
| Agg sum/mean/positive mean/active/all softmax | aggregate.py / aggregate_kernel.cpp | complete source domain and contributions, absent != zero; aggregate formula/contract/schedule and source-domain tests |
| HARD/SOFTP/HST Emit, sparse slot payloads | full.py / full_kernel.cpp | explicit HST surrogate, None/zero/unused; emit/isolated-gradient tests |
| Read/Next/Region | readout.py, next.py, region.py / matching native files | history and controls causal; comparison-identity Next permits state prefill; custom Next blocks prefill with counter; contract/schedule/continuation tests |
| position/norm/mask/cache adapter composition | attention kernels provide causal/window mask and cache; LH Full provides norm | [tiny model composition](model-adapter.md) implemented; independent formula/chunk/VJP test; no arbitrary model compatibility |

State prefill requires exact_sequence, observe_all, no selected clear, and
comparison-identity Next. Otherwise state remains causal; Full can still batch.
Counters distinguish state_blocks, state_sequence_calls, scalar_sequence_steps,
state_steps, max_state_sequence, Full blocks and semantic replay. Generic empty
step events never create candidates. Snapshot/alias/parameter-update checks are
part of qualification; physical sharing need not allocate one Tensor per value.

## Training/persistence coverage

| Unit | Existing implementation/evidence | Acceptance extension |
| --- | --- | --- |
| isolated outputs/state/slots/history/pending roots | isolated_* tests, [semantic replay](evidence/isolated-autograd.md) | verified six-class/new-frontend roots, [S3/S4](evidence/foundation-stage34.md) |
| owner/alias, unused vs zero, SGD/momentum/AdamW | parameters/optimizer C++, checkpoint_ownership.py; [native owners](evidence/cpp-optimizer-ownership.md) | verified multi-update six-class trajectories, S3/S4 |
| graph continuation value checkpoint v5 | checkpoint.py and checkpoint_values.py | verified fresh-process trajectories, S3/S4 |
| two-clock application, partial buffer and ledgers | token_checkpoint.py, single_graph training/resume tests; [evidence](evidence/token-checkpoint-coordinates.md) | verified fresh-process trajectories, S3/S4 |
| native named values/optimizer TIDENCK1 | checkpoint codec/preflight/io, standalone check; [evidence](evidence/cpp-native-checkpoint.md) | verified standalone format and fresh-process updates, S3/S4 |

Named values do not include continuation. Graph/application bundles do not
restore a full training controller, framework RNG or dataset cursor. Native
TIDENCK1 is independent of Python files; format interoperation is not promised.

S3.1/S3.3 directed frozen-snapshot gate:736 passed/92.06s. Artifacts and terminal
source/build audit: `artifacts/modules-dev-20260923-a/`; CPU FP64/FP32. This
includes ungated formula/scan, all new schedules/native Settle, projection strides
with three updates, explicit scalar-policy reporting and model adapter roots.
