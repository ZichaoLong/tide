# Device whole-window graph VJP qualification

Source **37430e1dfab52238874e3a75a52b052ccd09c233**, clean standalone NPU build;
unchanged core d412541 verified by source/binary fingerprints. AArch64,
Ascend910_9392, CANN 9.0.0, LibTorch/torch_npu 2.10.0.
[Audited records and hashes](device-graph-vjp-20261001.json).

All **37 component cells** and four build CTests passed. The graph check compared
**98 complete actual windows** with independent CPU FP32/FP64 Streaming autograd,
including input, initial-state, incoming-pending, node and physical-scale gradients.
The separate link check covered **64 actual windows**. Coverage includes positive-delay
feedback, self/parallel edges, DAG/edgeless graphs, streaming/greedy, continuation,
large int64 times, widths1/3/257, phase absence, independent output/final/pending
roots, None/connected-zero, empty replay, poisoned padding and malformed/budget refusals.

Separate profiling recorded **88,699 AI_VECTOR_CORE +1,269 AI_CORE +792 MIX_AIV**
tasks, including5,934 graph reverse metadata/payload/scale records, with no AiCPU
or host fallback. The profile includes construction, forward and CPU assertions;
it is placement evidence, not a throughput comparison.

The [contract](../resident-graph-vjp.md) covers single-window FP32 HARD,
sum Aggregate, broadcast, identity/EMA/Add-repeat state and identity/tanh Full.
Actual event/message links and reverse-stage progression execute on device.
Per-node parameter partials still need owner reduction; no optimizer, public
training, retained-window graph, other module VJP, FP16 or peer training claim.
The unchanged portable core/Python retains its previous8,954-test CPU qualification.

Retained development failures:build-graph-vjp-dev01 rejected an Ascend C scalar
template argument; loading it as an ordinary scalar fixed compilation.
graph-vjp-dev03 refused the257-wide forward fixture at its default64MiB budget;
the fixture explicitly uses512MiB. Neither fix changed formulas or tolerances.
