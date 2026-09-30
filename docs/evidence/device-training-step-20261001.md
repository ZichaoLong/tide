# Device optimizer and continued training-step qualification

Source **3b31ee2**, clean standalone NPU build, with unchanged core d412541
verified by source/binary fingerprints. AArch64,Ascend910_9392,CANN9.0.0,
LibTorch/torch_npu2.10.0. [Audited records and hashes](device-training-step-20261001.json).

All **40 component cells** and four build CTests passed. Packed SGD/AdamW
passed **32 trajectories/256 updates** against independent CPU FP32/FP64:
parameter groups,aliases,first-use momentum,dampening,Nesterov,maximize,AMSGrad,
None/zero,decay,small gradients,width/tile tails,slots and int64 counters.
Nonfinite gradients,slots or bias corrections and exhausted counters refuse
without partially modifying parameters,slots or counters. Empty/invalid/budget
and distinct-owner storage-overlap cases also passed.

The complete internal chain passed **18 trajectories/72 continued windows**:
actual online forward,device graph VJP,owner reduction,finite checks,SGD/AdamW,
publication to forward banks and next-window execution. It compares full forward
observables,gradients and updated parameters with independent CPU trajectories.
Coverage includes positive-delay feedback,streaming/greedy,shared Read/Full/state/
scale aliases,normal/zero/absent objectives,and widths3/257. Source-scale diagnostics
retain the values used by the window even after parameter publication.
**Every optimizer boundary in this checker explicitly truncates gradients.**

Separate profiling recorded **79,156 AI_VECTOR_CORE +1,808 AI_CORE +288 MIX_AIV**
tasks,including288 optimizer plan/proposal/gate/commit records,without AiCPU or
host fallback. This is the full correctness checker with construction,forward
and CPU assertions,not a throughput comparison.

The [contract](../resident-optimizer.md) remains internal single-NPU FP32 HARD,
sum/broadcast,identity/EMA/Add state and identity/tanh Full. This is not public
training/checkpoint,retained-window training,additional module VJP,FP16 or peer
training qualification. The unchanged portable core/Python retains its previous
8,954-test CPU qualification.

Retained failures:build-optimizer-dev01 had a task-local static-library link-order
error;build-optimizer-dev01b assumed a direct ascendcl option on an installed
consumer link;optimizer-dev01 exited on its failed dependency before acquiring
an NPU. These wrapper failures remain failed and do not become qualification runs.
