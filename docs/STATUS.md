# Current handoff

Updated: 2026-09-29. Branch graph-execution-foundation; no push authorized.

## Active extension

User requested inspection of NPU selection/event scheduling, performance/AiCPU
attribution and configurable FP16. The preceding bounded FP32 acceptance remains
complete at c8c46be (implementation b4f26b3); its 54 records are immutable.

Implement a bounded FP16 extension in the standalone historical-topology
consumer: FP16 payload, independently configured FP32/CPU-FP64 Read, int64 event
keys, FP32 controls, FP32 master parameters/optimizer slots and explicit static
loss scaling. User explicitly expanded scope to public Python/PyTorch and native API too.
Python Session checkpoints support FP32 master optimizer state; existing
FP32/FP64 defaults and standalone C++ owner checkpoint contract remain.
Compare complete observables/discrete routes and VJPs against an independent
CPU oracle at the same dtype. An explicitly selected FP32 quantized-fixture
oracle remains a cross-precision diagnostic. Numerical
tolerances are explicit; route or None-connectivity differences are not hidden.

Next: freeze this implementation commit as sources/fp16-a5. Build isolated
builds/{cpu-fp16-a5,adapter-fp16-a5,core-fp16-a5,client-fp16-a5} with two workers.
Run scripts/verify.py --device cpu --dtype both with cpu-fp16-a5; run complete
scripts/qualify_accelerator.py FP16 Python/native NPU suites; run standalone
FP32/FP16 CPU and two-NPU gates plus the intended 4/8-NPU control strategies.
Then run four fresh FP32/FP16 pairs (8 cells): Add2/Attention4 inference,
Add4/Attention8 training, one warmup and one measured complete update. Each
pair holds one fixed queue allocation. New identities required; no large run
before applicable gates. CUDA device qualification remains external.

Development checks passed: 318 directed CPU regressions; five Python and five
native NPU FP16 complete cases; consumer CPU and two-NPU four scale/four training
configurations each, explicit atol=.002/rtol=.02. Latest client-fp16-dev06 and four
CTests passed; consumer-npu-fp16-dev06 and consumer-cpu-fp16-dev06b validated its
new dtype guards (exit 0). No development jobs remain live. Final comment-only
header correction does not alter behavior and will be included in clean builds.

Retained failures: dev01 invalid ATen half test literal (fixed), NPU dev01
cross-precision gradients, CPU Attention cross-precision routes; same-dtype
CPU/NPU dev03 VJP narrowly failed atol=.001 and passed explicit atol=.002 in
dev05/dev06. consumer-cpu-fp16-dev06 was rejected for mistaken --devices 2 on CPU;
corrected one-device dev06b passed. Discrete routes and None/zero stay exact.

Implementation commit excludes the two profile-analysis evidence drafts, which
are reserved for the later evidence commit. No sub-agents, reference-repository
edits or pushes. Formal job records live under TASK_ROOT/runs/NAME/{status.json,
task.log}; units tide-npu-performance-NAME.service, all in background.slice.

## Existing evidence

The FP32 performance and standalone multi-CANN reports remain under docs/evidence:
accelerator-performance-20260928, accelerator-dispatch-training-20260928 and
standalone-sdk29-20260928. NPU sort/event tensor work exists, but host-owned
histories, scalar extraction, tensor handles and dispatch remain. Tiny msprof
trace contains 7012 operators: 55 AiCPU int64 Sort calls, 3365.9 us of 23020.06 us
summed operator time (14.62%). This is not end-to-end or full-size attribution.
No prior FP16 performance data exists.

Task root /mi/data2T/zlong/tide-npu-performance; existing sources/perf-a4,
builds/client-npu-a4 and runs are retained unchanged. New source/build/run suffix
a5; /opt public modules, shared /usr/local driver. Trackio best-effort local
project tide-npu-performance, TASK_ROOT/trackio, viewer /home/zlong/venvs/trackio/bin/python.
Hardware placement must use the account cooperative queue and current inventory.
CUDA hardware and other host architectures remain external acceptance.

Reentry: git status --short --branch; python scripts/status.py.
