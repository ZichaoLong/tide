# Device identity/tanh Full VJP qualification

Source **3285b13baacfba72266d3dc3377b696a80b23cfb**, clean standalone NPU build;
unchanged core d412541 was verified by source/binary fingerprints. AArch64,
Ascend910_9392, CANN 9.0.0, LibTorch/torch_npu 2.10.0.
[Audited records and hashes](device-full-vjp-20261001.json).

All **35 component cells** and four build CTests passed. The Full VJP check
passed **96 CPU FP32/FP64 autograd cases** and **two actual device-forward tapes**
(streaming/greedy). Each isolated case checks complete, shorter and empty replay.
It covers repeated owners/samples, widths 1/7/257, chunks 1/5, disconnected and
connected-zero gradients, poisoned absent/padding/non-tanh parameters, large
residual content, unsupported kinds, invalid metadata and tensor-budget refusal.
The actual owner rejects unrecorded, closed and unsupported Full tapes.

Separate profiling recorded **8,036 AI_VECTOR_CORE + 288 AI_CORE tasks**, including
998 Full VJP metadata/payload/reduction records, with no AiCPU or host fallback.
This includes setup, assertions and forward integration; summed task time is
neither wall time nor a full-model throughput result.

The [contract](../resident-full-vjp.md) remains an internal first-order component.
It recomputes tanh from saved comparison/parameters, uses device-selected chunks,
unique padding destinations and stable per-node parameter reduction. Per-node
partials still require alias-owner accumulation. This evidence does not certify
complete graph backward, LH/SwiGLU Full, HST/SOFTP, public autograd, optimizer,
FP16 or multi-device training. CPU core/Python sources are unchanged; the prior
8,954-test full CPU gate was not repeated.
