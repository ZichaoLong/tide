# Execution capability audit

Snapshot: source in [scope audit](evidence/foundation-scope-audit.md). This is a
capability description, not a second backlog; unfinished acceptance IDs refer to
ROADMAP. CPU FP32/FP64 required. Python schedules are serial with batch/sequence
support. Native schedules support node workers. Training denotes first-order
semantic parity, not optimized backward; packed state/Read/Full replay remains.

## Graph × schedule × option × mode

P = PDG, D = TimedDAG, S = encoded SettleGraph. D/S streaming use ordinary PDG
kernel and preserve their legal topology. Native S construction is [verified](evidence/native-settle-frontend.md) in `settle.h`.
"verified" refers to the existing bounded modules/tests below, not every custom
program. Infer/train entries share values; training uses scalar semantic replay.

| Option / applicable module | P/D/S streaming, infer/train | D/S frontier, infer/train | Independent specialization |
| --- | --- | --- | --- |
| node workers, packed state/Full | verified | verified | chain/self-loop verified |
| attention_packing exact/single, same-fiber | verified | verified, `test_fiber_single.py` | default covered; policy combinations need S3 audit |
| fiber_pooling event/CSR, same-fiber | verified | verified, `test_fiber_efficiency.py` | exact default covered; options not separately certified |
| cloned/owned KV, same-fiber | verified | verified, same test | default covered |
| attention layout event/head, same-fiber | verified | verified, same test | default covered |
| projection input/linear layout | same-fiber QKV/output physical strides selected by scale model initialization; [fiber efficiency](evidence/fiber-efficiency.md) | local kernels accept either stride; explicit cross-graph ownership/update gate remains S3.1 | default projection covered |
| compact events | native Streaming, trace snapshots retained | consumed fibers moved, trace snapshots retained | same block publication |
| parallel regions | native Streaming, canonical publish order | same-time independent sample owners, canonical publication | same causal waves |
| deferred state release | compact required | compact + trace-free displaced states retire on workers | same block cleanup |
| packed_sources | packed sum transport; counted unsupported Agg fallback | Aggregate and same-fiber sequence transport | independent ring/diamond/chain/self-loop |
| batch_next/reset | packed adopt/selected fiber reset; counted custom fallback | same-time batch owners; times stay causal | same causal Next/reset grouping |
| head workers | native scale/portable Full projection pool; separate from executor node pool | no general frontier head-worker API; S3.1 audit of applicable Full path | no general head-worker API |

S3.2 directed gate: `tests/test_frontier_options.py` and eight related files,
1098 CPU FP64/FP32 tests passed; archived source and terminal audit at
`artifacts/frontier-options-dev-20260923-a/`. Stage3 clean qualification remains.
Defaults stay unchanged. Prefill fallback counters distinguish disabled, missing
sequence contract, selected-only adoption, selected clear, and custom Next.
Phase profiling remains explicitly Streaming-only.

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
| Linear/Gated Delta (`linear`/`delta`) | matrix_memory.py / matrix_kernels.cpp | literal steps and exact scans; ungated DeltaRule is still required by S3.1; `test_matrix_memory.py`, isolated roots |
| FFN/SwiGLU/identity/norm Full | ops.py, lh_full.py / ops.cpp, lh_full.cpp | independent selected rows batched; `test_memory_programs.py`, `test_lh_full_formulas.py` |
| Agg sum/mean/positive mean/active/all softmax | aggregate.py / aggregate_kernel.cpp | complete source domain and contributions, absent != zero; aggregate formula/contract/schedule and source-domain tests |
| HARD/SOFTP/HST Emit, sparse slot payloads | full.py / full_kernel.cpp | explicit HST surrogate, None/zero/unused; emit/isolated-gradient tests |
| Read/Next/Region | readout.py, next.py, region.py / matching native files | history and controls causal; comparison-identity Next permits state prefill; custom Next blocks prefill with counter; contract/schedule/continuation tests |
| position/norm/mask/cache adapter composition | attention kernels provide causal/window mask and cache; LH Full provides norm | RoPE and concrete small model composition required S3.3; no arbitrary model compatibility |

State prefill requires exact_sequence, observe_all, no selected clear, and
comparison-identity Next. Otherwise state remains causal; Full can still batch.
Counters distinguish state_blocks, state_sequence_calls, scalar_sequence_steps,
state_steps, max_state_sequence, Full blocks and semantic replay. Generic empty
step events never create candidates. Snapshot/alias/parameter-update checks are
part of qualification; physical sharing need not allocate one Tensor per value.

## Training/persistence coverage

| Unit | Existing implementation/evidence | Acceptance extension |
| --- | --- | --- |
| isolated outputs/state/slots/history/pending roots | isolated_* tests, [semantic replay](evidence/isolated-autograd.md) | newly added schedules/native frontend S4.1 |
| owner/alias, unused vs zero, SGD/momentum/AdamW | parameters/optimizer C++, checkpoint_ownership.py; [native owners](evidence/cpp-optimizer-ownership.md) | new six-class trajectories S4.1 |
| graph continuation value checkpoint v5 | checkpoint.py and checkpoint_values.py | new-process integration S4.2 |
| two-clock application, partial buffer and ledgers | token_checkpoint.py, single_graph training/resume tests; [evidence](evidence/token-checkpoint-coordinates.md) | new-process integration S4.2 |
| native named values/optimizer TIDENCK1 | checkpoint codec/preflight/io, standalone check; [evidence](evidence/cpp-native-checkpoint.md) | reuse; verify process boundary S4.2 |

Named values do not include continuation. Graph/application bundles do not
restore a full training controller, framework RNG or dataset cursor. Native
TIDENCK1 is independent of Python files; format interoperation is not promised.
