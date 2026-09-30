# Device parameter-owner VJP qualification

Source **1ef23f3ab57024fedd9bda439976aff95ceb7795**, clean standalone NPU build;
unchanged core d412541 verified by source/binary fingerprints. AArch64,
Ascend910_9392, CANN9.0.0, LibTorch/torch_npu2.10.0.
[Audited records and hashes](device-parameter-vjp-20261001.json).

All **38 component cells** and four build CTests passed. Owner reduction passed
**36 actual graph/root cases** against independent CPU FP32 Streaming autograd.
The retained whole-graph gate separately covers FP32/FP64 references. Tests include
cross-node weight sharing, bias/decay/Read sharing, input/Aggregate/output/edge/
retention sharing, widths1/3/257, both schedules, None/connected-zero, poisoned
absent partials, replay, trainable subset, empty registry, bad shapes and budgets.
Distinct TensorImpl owners sharing storage are not incorrectly merged.

Separate profiling recorded **23,580 AI_VECTOR_CORE +334 AI_CORE +144 MIX_AIV**
tasks, including74 parameter-reduction records, with no AiCPU or host fallback.
This is the entire correctness checker with setup/forward/CPU assertions and
is not a throughput result.

The [contract](../resident-parameter-vjp.md) remains the restricted single-NPU
FP32 HARD graph profile. Gradient connection and alias accumulation execute on
device; owners without a differentiable use allocate no parameter-sized gradient.
This qualification does not cover optimizer updates, public or retained-window
training, other module adjoints, FP16 or peer training. Portable core/Python is
unchanged and retains the previous8,954-test full CPU qualification.
