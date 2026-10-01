# Resident event and same-fiber attention training qualification

Exact implementation **66a6ca5702d3618ca72ad57ac87f01a7190f9166**, aarch64,
Ascend910_9392, Torch/TorchNPU 2.10 and CANN 9.0.0. Separate clean standalone
and Python-owned builds passed. [The audit](resident-attention-training-20261001.json)
checks source/core identities, binaries, loader closure, raw logs and profiler
CSV hashes. The two runtime owners remain separate.

All **52 component cells**, five build CTests and the installed C++ inference /
retained-training / optimizer-restore client passed. Python passed **196 NPU
cases with no skips**, including 42 new attention cases, actual loss cotangents,
all three graph families, both schedules and independent-process checkpoint
continuation. The CPU interface gate passed 76 tests with 193 optional NPU
skips. Unchanged portable-core CPU qualification was reused, not rerun.

Event attention passed 66 isolated/combined root cases and eight complete
training trajectories against independent CPU FP32/FP64 execution. Same-fiber
attention passed 172 root cases and 20 trajectories; its local VJP additionally
passed 37 configurations and 74 replays. Coverage includes all five pooling
profiles, complete-current-fiber visibility, query/key tiling with global
normalization, QKV/output biases, repeated-tick decay, physical source/scale
gradients, shared parameter owners and publication into live parameter banks.

Actual device cache chains preserve adoption/clear, key/value/log-bias roots,
retained-window bridges and initial-cache gradients. Feedback and DAG cases,
logical source permutations, periodic clocks, widths 1/4/257, mixed event/fiber
cache groups, inactive NaN padding and None/connected-zero distinctions are
covered. Candidates produce their own events, cache associations, gradients
and optimizer updates; CPU reference results never become candidate inputs.
See the [event](../resident-event-vjp.md), [fiber](../resident-fiber-vjp.md) and
[public training](../resident-training.md) contracts for the supported profile.

New attention checks keep strict tensor/control tolerances and exact discrete
comparisons. AdamW trajectory epsilon is explicitly 1e-5; the public default
remains 1e-8. Only the pre-existing LH/SwiGLU regression uses its recorded
conditioned-control policy. Its strict near-zero RMSNorm limitation remains
in [the prior evidence](resident-extra-full-20261001.md).

A separate bounded trace covers mixed event/fiber roots and one full AdamW
trajectory: **13,267 tasks**, including 275 fiber-reverse, 25 event-reverse,
126 cache-bias merge and 16 optimizer tasks. Engines were AI_VECTOR_CORE,
AI_CORE and MIX_AIV; no AiCPU record or logged CPU fallback was observed.
Construction and CPU correctness assertions are included. This profile is
not a full-suite profile, wall-time benchmark or throughput comparison.

The audit preserves five failed fiber development jobs: an Ascend C scalar
operand error, a development launcher include omission, a missing host include,
an ambiguous checker Tensor assignment, and an illegal sparse-domain fixture.
All remain failed in their original records; the clean qualification uses the
normal build script without development overlays.

This qualifies the declared **single-NPU FP32** attention training profile.
The Python result is a C++/CANN client result, not an independent PyTorch
resident scheduler. Complete resident FP16, peer progression/communication /
training, the representative/full-size performance matrix and target-machine
CUDA/version evidence remain separate work. No new speedup is claimed.
