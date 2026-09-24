# Execution capability audit

Qualified source:81a1b266af49d918aa6e1587e4ed9e0c4d4e5eb5;
[7741-test final CPU gate](evidence/foundation-final.md). This is the current
capability description; ROADMAP owns stage status and extensions. CPU FP32/FP64 required. Python schedules are serial with batch/sequence
support. Native schedules support node workers. Training denotes first-order
semantic parity, not generally optimized backward; default packed state/Read/Full
replay remains. The optional [native Full affine VJP](full-batched-autograd.md)
removes Full projection replay for its declared profiles. The optional
[Aggregate source VJP](aggregate-batched-autograd.md) batches built-in source
programs with isolated roots and per-event normalization backward; STATUS records
the current correctness and performance qualification separately.

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
