# FP16 resident HARD inference

Implemented source: `8b05c0495fd5bc3cb874f6cb1548b25f3d5ac4b5`.
Seven qualification jobs passed from a clean immutable checkout on aarch64,
Ascend910_9392, Torch/TorchNPU2.10 and CANN9.0.0. The [audit](resident-fp16-inference-20261001.json)
checks source/core/binary/loader/build-reuse/log and profiler CSV identities.

This qualifies single-device FP16 HARD inference with online device queue,
readiness, selection, node-time batching and recursive progression. Candidates
consume common public inputs, parameters and initial state, and restore their
own checkpoints. CPU reference results do not feed device execution. Python is
a client of C++/CANN, not an independent PyTorch device scheduler. FP16 HST/SOFTP,
complete training, peer progression and whole-model throughput remain pending.

## Correctness

- Eight affected standalone cells passed:attention-payload/precision-flow in both
  dtypes,FP32 attention-tile,public resident inference,event and fiber training.
  Component/flow jobs used physical3;regressions physical13;each mapped to logical0.
- Complete flow:78 configurations/312 windows per dtype,including feedback,
  duplicate edges,int64 clocks above2^55,EMA/Add,event and all five fiber profiles,
  normalized Aggregate,tanh/SwiGLU/all9 LH Full profiles,slot-affine/phase emission,
  both schedules,scalar/vector,dense/tiled attention,CPU/NPU boundary inputs and
  candidate-owned continuation with changed batching and disabled diagnostics.
- Attention payload:64 configurations/192 changed-input replays per dtype,
  GQA,widths1/4/33/257,key tiles1/7/128/256,empty/sticky refusals,NaN padding and
  finite40000 outputs whose unnormalized half sums would overflow.
- Python/NPU:215 passed,zero skipped,physical9. Includes19 new FP16 cases,
  all three graph families,both schedules,checkpoint continuation and explicit
  unsupported-scope checks,plus the existing FP32 inference/training regression.
- CPU interfaces:76 passed,212 optional NPU skips. Unchanged portable core source
  and binaries were hash-verified;the8,954 CPU core checks were not rerun.

FP16 complete-flow tolerances are atol2e-3/rtol2e-2. FP32 thresholds are unchanged;
discrete decisions,edge identities and exported dtypes remain exact. Read and
attention normalization/weighted accumulation use FP32. State/KV/inputs/outputs
remain FP16;fiber bias rounds each repeated tick. Journals widen payloads on device,
then export payload/control fields in their public dtype. Control softmax itself
stays FP32. Full/LH allocation minima use the same dtype as actual reservations.

## Build and profile boundaries

Separate standalone/Python builds rebuilt affected host objects and relinked
consumers. Remaining source inputs were byte-matched against terminal successful
builds;CANN archives and matching host objects were reused. These were not fresh
vendor compiler rebuilds. The audit compares every retained content-archive member
and verifies rebuilt members,dependency hashes and loader ownership. Earlier
normal full builds remain recorded in the [component qualification](resident-fp16-forward-components-20261001.md).

The separate FP16 profile covered two attention configurations/eight windows,
physical9,with256MB collection and180s application bounds. It observed
2390 AI_VECTOR_CORE,76 AI_CORE and2 MIX_AIV tasks;no AiCPU task or logged CPU
fallback was observed. Construction,uploads and CPU assertions are included.
This trace establishes placement for its scope,not training or throughput.

Raw records use `artifacts/execution-flows-NAME`:builds
`build-low-precision-inference-clean02` and `build-low-precision-inference-python-clean01`;
gates `low-precision-inference-{components,regression,python,host}-clean01`;
profile `low-precision-inference-profile-clean01`.

Four failures remain unchanged:attention-dev01 build lacked a header;flow-dev03
used FP32 Full/LH budget minima for half reservations;Python-dev04 exported FP32
controls instead of the public payload dtype;standalone clean01's task-local
loader assertion incorrectly required a shared-library dependency for static
components. Each cause was corrected before the successful qualification.
