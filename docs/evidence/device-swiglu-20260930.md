# Device-selected SwiGLU Full — 2026-09-30

Clean `74cec2f` passes a standalone build, four CPU CTests, all 24 device
component cells and a separate CANN placement profile. The
[manifest](device-swiglu-20260930.json) identifies source/core/component hashes
and original results. This qualifies single-device FP32 HARD inference only.

The device planner packs only selected SwiGLU actions and computes
`content + (silu(comparison @ ffn_gate) * (comparison @ ffn_up)) @ ffn_down`
in bounded batches. Static tables contain actual SwiGLU parameter owners;
non-SwiGLU nodes allocate no dummy matrices. Full consumes the pre-clear
comparison, supports mixed Full programs and feeds the independently qualified
slot-affine/phase emission path. There is no numerical CPU routing prepass.

The new gate passes 16 component cases, 256 complete windows and 4 refusals:
widths 1/7/33/257, chunk limits 1/4, empty/partial selection, inactive poisoned
parameters, large int64 clocks, mixed identity/tanh/LH/SwiGLU Full, slot-affine
and phase emission, content/proposal Read, feedback/DAG, two inputs, both
schedules, clear/selected-only adoption, continuation and lean export.
Tensor comparisons retain rtol1e-5/atol1e-6; discrete observables remain exact.
The matmul path retains FP32 KEEP_DTYPE and does not enable HF32 or FP16.

The trace records 53,667 AIV and 1,195 AI Core tasks, including device planners,
batched gathers, Swish and BatchMatMulV2. No AiCPU task or host fallback
diagnostic was observed. The trace includes setup/export/CPU assertions and
is placement evidence, not throughput. This increment had no development run
failure. FP16 flow, attention/KV, full safe memory planning, public matrix/presets,
peer progression and resident backward/optimizer remain separate obligations.
