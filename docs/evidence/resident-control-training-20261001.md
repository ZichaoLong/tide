# Resident control and Read training qualification

Exact implementation **4b8ced41f422301a4ced2bf2999c5d1d3cb63e3d**, aarch64
Ascend910_9392, Torch/TorchNPU 2.10 and CANN 9.0.0. Separate clean standalone
and Python-owned builds passed. [The audit](resident-control-training-20261001.json)
checks immutable sources, core identities, binaries, loader closure, logs and
profiler CSV hashes; the Python client does not load the standalone SDK.

All 48 component cells and five CTests passed. Isolated HST/SOFTP control/Read
adjoints passed 36 configurations and 108 long/short/empty replays against CPU
FP32/FP64 autograd (216 comparisons). Public training passed 98 trajectories,
1,568 retained windows and 392 optimizer updates, covering both schedules,
SGD/AdamW, complete candidate-frame softmax gradients, content/old/proposal
Read, parameter sharing, retained continuation and checkpoint suffixes.
Candidates run their own forward/backward/update; no reference trace, numerical
result or gradient is supplied to candidate execution.

The Python client passed 154 NPU cases with no skips, including real device loss
cotangents, all three graph families and new-process disk continuation. The host
interface gate passed 76 tests with 151 optional NPU skips. These are C++/CANN
client results, not qualification of an independent pure-PyTorch resident schedule.

Control trajectories retain strict tensor/control tolerances and exact routes;
AdamW epsilon is explicitly 1e-5. The public default remains 1e-8. Only the
pre-existing LH/SwiGLU regression uses its explicit conditioned control policy
(13 frames, maximum absolute difference 4.470348e-6); its retained near-zero
RMSNorm trajectory limitation remains in [the original evidence](resident-extra-full-20261001.md).
No new tolerance relaxation applies to control training.

A separate complete correctness profile observed 558,031 AI_VECTOR_CORE,
7,604 AI_CORE and 12,390 MIX_AIV records, including 19,264 control, 1,376 Emit mix,
1,568 optimizer, 24,352 graph-reverse and 2,352 window-bridge records. No AiCPU
record or host-fallback diagnostic was observed. Construction and CPU assertions
are included; these counts and summed task times are not throughput.

The audit retains three failed development records: absent edge-scale fixture
indexing, an unsupported Python Full fixture, and a profiler export failure
under the earlier 200 MB aging limit. Their later successful runs do not change
those failures. Attention adjoints, resident FP16, peer progression and the
required performance matrix remain separate work.
