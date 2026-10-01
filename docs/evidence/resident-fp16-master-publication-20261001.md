# FP32 resident masters and FP16 payload publication

Implementation: `9a84432167a82900c0472b02b2e33ddace6da3eb`.
Verified 2026-10-01 on aarch64 Ascend910_9392, Torch/TorchNPU2.10.0,
CANN9.0.0. [Machine-readable audit](resident-fp16-master-publication-20261001.json).
All six qualification jobs passed at this clean immutable source.

This qualifies the single-device optimizer/publication components and affected
FP32 training/FP16 inference clients. The publication checker supplies common
synthetic owner-gradient packets; independent CPU masters and Streaming perform
their own updates and continuation. No CPU-generated route, result or gradient
is fed into a graph candidate. It does **not** qualify FP16 graph VJP, complete
FP16 training, peer execution or throughput. Public FP16 training remains refused.

## Behavior and checks

Masters, gradients and optimizer slots stay FP32. Payload owners may be FP16.
SGD/momentum/AdamW, canonical aliases, None versus connected zero, poisoned absent
gradients and update counters retain their contracts. Before any live commit,
the device checks finite proposals and their actual FP16 rounding. A65512 master
remains65512 while publishing finite65504;65520 refuses. Rejected updates leave
all live owners, slots and counters bitwise unchanged. Restore validates the same
boundary. Small master updates are not lost to premature half quantization.

Publication uses device kernels for ordinary, event-attention and fiber-attention
banks, including HARD Read aliases. FP32 Aggregate/fiber normalization banks first
round to payload precision and then widen. The checker compares exact published
payloads to the candidate's own verified master and compares CPU/device masters
at the original FP32 tolerance. It also verifies sticky-error no-write behavior.

| Cell | Passing coverage |
| --- | --- |
| FP32 optimizer |32 trajectories/256 accepted updates, CPU FP32/FP64 references |
| FP16 optimizer |32 trajectories/248 accepted updates/two CPU-predicted range refusals |
| FP32 publication |48 trajectories/240 inference windows/192 update calls |
| FP16 publication |48 trajectories/240 inference windows/192 update calls |
| FP32 event training regression |66 root cases/eight trajectories, strict controls |
| FP32 fiber training regression |172 root cases/20 trajectories, strict controls |
| Python-owned resident regression |215 passed,zero skips |

Publication covers streaming/greedy, SGD/AdamW, widths3/33, tanh/EMA, SwiGLU/Add,
LH, event attention/GQA, fiber pooling and normalized Aggregate. CPU and device
independently continue through four updates and a final inference window.
The synthetic update calls include absent and connected-zero gradients; they are
not a claim of192 graph backward computations. The Python gate includes public
inference/training/checkpoint clients and explicit unsupported-FP16-training checks.
The portable core was unchanged; its8,954-check CPU gate was not repeated.

## Build and run identity

`TASK=/mi/data2T/zlong/tide-execution-flows` locally; each name below owns
`TASK/runs/NAME/status.json` and `task.log`. The source snapshot is
`TASK/sources/low-precision-publication-clean01`. Raw evidence is not overwritten.

- `build-low-precision-publication-clean01`: standalone; two publication kernels,
  six host objects and affected checkers rebuilt; three matching optimizer kernels
  and other terminal dependencies reused with byte/source verification.
- `build-low-precision-publication-python-clean01`: six Python-owned host objects
  rebuilt and client relinked; byte-identical CANN archives reused. No standalone
  SDK is loaded into the Python-owned process.
- `low-precision-publication-components-clean01`: physical9→logical0, four cells.
- `low-precision-publication-regression-clean01`: physical3→logical0, two cells.
- `low-precision-publication-python-clean01`: physical13→logical0,215 cases.
- `low-precision-publication-profile-clean01`: physical1→logical0, separate trace.

The builds are controlled dependency reuse, not from-scratch vendor rebuilds.
The audit verifies source/core identity, every retained content-archive member,
rebuilt objects, kernel/binary/loader hashes, raw logs and profiler CSV inputs.
Standalone and Python loader closures are checked separately.

Portable checks use `scripts/verify_device_control.py --device npu:0 --checks
optimizer master-publication` and separately `--checks event-training fiber-training`,
with the matching `--build-dir` and unique `--output-dir`. The Python gate runs
`test_resident_precision`, `test_resident_library`, `test_resident_training`,
`test_resident_event_training`, `test_resident_fiber_training` at `--dtype float32`;
the precision tests explicitly create half workloads. Source/build identity and
exact executed commands remain in the audit/raw records.

## Profiling and retained failures

`profile_device_control.py --dtype float16 --check master-publication
--application-arg=--profile-smoke --storage-limit-mb256` records two trajectories,
ten continued windows and eight updates. Observed task counts:6718 AI_VECTOR_CORE,
288 AI_CORE,5 MIX_AIV. No AiCPU task or logged CPU fallback was observed in this
trace. Device classification is scoped to these recorded tasks; this is not a
throughput comparison or proof of every possible path's placement.

Preserved failures: `low-precision-master-dev01` incorrectly expected every finite
FP32 AdamW proposal to fit half; the corrected test predicts refusals independently.
`build-low-precision-publication-dev01` failed on a const CPU checker registry.
`low-precision-publication-dev02` failed because its continuation used a three-item
stride for a two-item input stream; the checker now derives per-port counts.
Runtime math/validation and comparison tolerances were not relaxed for those fixes.
These original records remain failed. FP16 whole-graph reverse, retained training,
checkpoint integration, HST/SOFTP, peers and the performance matrix remain required.
