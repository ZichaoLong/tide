# Current handoff

Updated: 2026-09-29 04:46 (Asia/Shanghai). Branch graph-execution-foundation.

## Local completion

All authorized locally feasible accelerator work is complete. Implementation is
frozen at b4f26b3659cec0549e18560a1a996d5bee0cae6d; this update is evidence/documentation only.
The final evidence commit follows 0d4c55c; use git log -1 for its identity.
No required project workload remains queued or live. All ten bounded performance
jobs and their coordinator terminated successfully; all 54 cells passed record,
identity, finite-metric and child-cleanup checks. Raw records remain authoritative.
No push or reference-repository mutation is authorized.

## Verified scope and decisions

- Public Python/native/standalone accelerator foundation: prior 8636 CPU tests,
  22 complex cells, 328 positive NPU cases, explicit unsupported-path checks and
  installed consumers. CUDA compile/CPU checks are not CUDA hardware evidence.
- Consumer dispatch/training: 200 CPU/2/4/8-NPU configuration cells plus four
  analytic gates on the immutable implementation. Independent CPU schedules,
  complete observables, exact routes, isolated VJPs, None/zero and optimizer slots.
  See [qualification](evidence/accelerator-dispatch-training-20260928.md).
- Standalone TorchNPU2.9: one SDK build qualified with CANN8.5.0/.1/.2, in addition
  to TorchNPU2.10/CANN9.0. Each runtime passed standalone training/checkpoint,
  installed CMake consumer, 64 two-NPU cells plus analytic gate and hardware trace.
  The broad Python/native-adapter gates remain separate. See
  [SDK evidence](evidence/standalone-sdk29-20260928.md).
- Full-size Add9.468B and Attention17.269B, D2048/B512/V50304, FP32 payload,
  465 nodes and full 12-token windows: seven inference/cold-training choices each,
  2/4/8-device observations, three warmed training processes/model, three matched
  inference pairs/model and one concurrent Add2/Attention4 workflow completed.
  Read precision/placement, controls, ranking and event queue stay independently
  configurable. CPU64/CPU32 names describe Read, not model-payload placement.
- Warmed training choices: Add4 CPU32 median
  42.080 ms/sample-token;
  Attention8 mixed32 (NPU FP32 Read, CPU controls/dispatch) median
  115.678 ms/sample-token.
  These are three-process synthetic throughput results, not convergence or a
  warmed ranking of all configurations. All process variance remains reported.
- Matched inference selected Add cpu64 and Attention cpu64
  for the final concurrent workflow under the preregistered conservative rule.
  Public CPU FP64 Read/CPU control/dispatch defaults remain unchanged.
- Resident payload/state/KV/messages stay on NPUs, while integer histories,
  tensor handles and C++ dispatch remain host-owned. Int64 sorting uses card-local
  AiCPU. Executor guards/scalar synchronization stay in the declared timers.
  Single-process model sharding is not DDP/HCCL or a fully device-resident loop.

The [final performance report](evidence/accelerator-performance-20260928.md) and
[manifest](evidence/accelerator-performance-20260928.json) contain exact scopes,
comparisons, hashes, memory and transfer accounting. Placement/scaling screens and
the one concurrent workflow do not establish causal speedup. The historic
Attention2 grad-forward OOM and cancelled unstarted eight-device waiter remain
failed/cancelled; the successful follow-ups have distinct identities.

## External acceptance and future scope

No locally runnable required item remains. On each NVIDIA, x86_64 or new
host/software combination, rebuild and run [target-machine acceptance](accelerators.md#target-machine-acceptance):
CPU semantics, real-device smoke, full FP32 parity/training/checkpoint and installed
consumer gates. CUDA FP64 requires its own gate. Record exact hardware/software
and keep untested cells implemented rather than verified. Portable source export
and qualification commands are already delivered; do not claim one binary across
architectures or vendor stacks.

FP16/BF16/AMP, compiled/fused paths, arbitrary topology/scale, continuous-stream
training, fully device-resident control and DDP/HCCL remain separate extensions,
not unfinished items in this bounded local acceptance. NPU FP64 and CSR pooling
remain explicitly unsupported. New profiling/tuning requires a new bounded task.

## Retained operations and evidence

Task root: /mi/data2T/zlong/tide-npu-performance. Frozen sources/perf-a4,
builds/client-npu-a4, module libtorch-npu/2.10.0-cann9.0.0. Public /opt modules
are authoritative; shared driver and other workloads were preserved.
All final job identities/terminal states are in runs/closure-a4/performance-evidence.json;
the 54-record validation ledger is runs/record-validations-a4.json.
Raw cells: runs/JOB/CASE/{run.json,summary.json,metrics.jsonl,lifecycle.json};
artifacts/npu-performance-JOB in the repository points to retained local records.
Coordinator: tide-npu-performance-closure-a4.service, terminal passed/exit0;
runs/closure-a4/{status.json,continuation.json,task.log}. Do not relaunch it:
fresh outputs are mandatory and the bounded assessment is already complete.

Trackio 0.35, best-effort local project tide-npu-performance, data root TASK_ROOT/trackio,
storage auto, viewer /home/zlong/venvs/trackio/bin/python. All 54 writers reported
healthy; no per-event durable acknowledgement is claimed. No dashboard is required.
Preserve cited artifacts and historical failures; do not clean reference repos.

Reentry: git status --short --branch; python scripts/status.py. Read ROADMAP for
the completed finite acceptance and external work. Evidence publication is checked
with git diff --check, JSON/record identities, support-matrix validation and local
links. No implementation changed, so the unchanged large correctness gates were
not rerun. STATUS publication uses scripts/durable_records.py.
