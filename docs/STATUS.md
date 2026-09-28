# Current handoff

Updated: 2026-09-28 (Asia/Shanghai). Branch: graph-execution-foundation.

## Authorized performance extension

User requests LibTorch/NPU performance for historical 17B Attention and "8.8B"
Add, both independent concurrent processes and a single model spanning cards.
Up to 8/12 devices allowed, subject to queue availability. Exploit edge locality.
No subagents or pushes. Core single-device library/API remains unchanged.

Exact target: 465 nodes,2208 logical/4418 physical edges,D2048/B512/V50304;
Attention17,269,426,339 parameters, Add9,468,020,899 (8.818*1024^3).
FP32,CPU seed7,original owner/RNG order,2 body ticks/token,clear,all-softmax,
norm-fp64-v1 Read. Twelve growing-context tokens;4 warmup+8 measured.
no_grad and grad-forward (no backward/optimizer/detach). No reduced-scale
result may be labeled full size. See docs/accelerator-scale.md.

Implemented standalone tools/accelerator_scale consumer: memory/locality placement,
host bridge comparison and device-resident state/KV/message executor. The latter
reuses public local programs with its own sealed-window schedule. Same-card
messages stay on device, remote copies use Torch NPU-to-NPU transfer. CPU computes
exact FP64 Read from selected vectors and maintains region histories. CSR pooling
is unrelated; event pooling is used. No arbitrary checkpoint import is promised.

Development source perf-dev04 (baseline1b0cb48 plus archived dirty snapshot) passed
8 cells each on CPU,1 NPU(default queue),2 NPUs(TASK_QUEUE_ENABLE=0): both
transports,models,placements,full observable parity and isolated zero/nonzero VJPs
with None connectivity against scalar CPU slot oracle. Residency is checked.
Records: runs/check-cpu-dev04,check-npu1-dev04,check-npu2-sync-dev04.
Standalone pure CPU client was compiled at dev03; final source needs its clean build.

Known failure: default async NPU queue caused SIGBUS/SIGSEGV on2 NPUs(9,11).
GDB captured libruntime_v100 -> rtStreamWaitEvent -> aclrtStreamWaitEventImpl
-> SDK WaitEventFunc. Diagnostic job exit0 is not a workload pass. Keep
runs/check-npu-dev04 and diag-bus-dev04. Workaround is process-local
TASK_QUEUE_ENABLE=0, now explicit/default in benchmark/verifier wrappers.
Do not infer all async queues or CANN versions fail; this exact tuple was tested.

Earlier retained development failures: dev01 CPU Torch_DIR discovery and missing
SDK allocator logging headers; dev02 invalid cross-profile graph identity assertion;
dev03 forward-only oracle views disconnected from gradient owners. Fixed in
consumer/build/test layer, not public core. Added SDK headers are unmodified from
94f8a8e6b523d7ba553e1b80d5b5248478391526; provenance allocator-header-supplement.json
under the existing SDK installation. Existing SDK library/header files unchanged.

## Next actions and bounded assessment

Commit tested consumer implementation now, freeze clean source perf-a1, build
client-cpu-a1 and client-npu-a1 with one/two compiler workers each. Job names
build-cpu-a1/build-npu-a1 under tide-npu-performance-*.service. Then immutable
CPU and2/8-NPU gates, reduced-tensor real-topology gate, wrapper success/timeout
records and an msprof direct-transfer trace. Only passing gates permit full size.

Initial full-size pilots: Attention/Add on4 NPUs, both no_grad and grad-forward,
resident transport,12 tokens,4 warmup,per-process timeout1800s,RSS cap512GiB.
Stop repeating an unchanged failed/time-limited cell. For completed modes,
compare2/4/8-device memory vs locality with three independent processes, matching
physical allocation/CPU resources within each placement comparison. Independent
concurrent models are separate runs, total occupancy<=8; record interference.
The matrix is bounded to these models,modes,counts,policies and3 repeats; no
hyperparameter search or reduced-precision substitution. Full17B weights exceed
one64GiB device; single-card Attention is a preflight limit, not a timing result.
No performance measurement/throughput/scaling claim exists yet.

Task root /mi/data2T/zlong/tide-npu-performance; launchers/freeze_run.py freezes
sources and uses durable job.py records. Runs/logs/status under runs/; source
snapshots and builds separate. Long jobs use background.slice, queue allocation,
bounded threads and stable sources. Do not edit frozen snapshots. Trackio writer/
viewer /home/zlong/venvs/trackio/bin/python (0.35.0), local root task-root/trackio,
project tide-npu-performance. Raw JSONL/manifests are authoritative.

## Completed foundation baseline

1b0cb48 records accelerator implementation adeb819/10cd630 and verifier correction
bf4ac1c. CPU8636 tests+22 complex cells+installed consumers;4 CANN stacks passed
328 positive NPU gates+4 CSR rejection gates. Standalone2.10/CANN9.0 qualified.
No CUDA hardware evidence. See docs/evidence/accelerators-20260928.md and
accelerators.md. Do not redo that qualification for this isolated consumer.

Use user-selected public /opt CANN modules. Standalone module
libtorch-npu/2.10.0-cann9.0.0 stays separate from Python wheel runtime. This user
selection overrides older account-guide personal defaults. Driver stays unchanged.
Existing core builds under /mi/data2T/zlong/tide-accelerator/builds/native-*-dev03.

Current changes are only consumer/tools/scripts and related docs. No public core
or Python package changes. All development jobs above are terminal.
