# Current handoff

Updated 2026-09-29 10:16 UTC. Authorized increment: ROADMAP D1-D6, matched-source
full-size CPU/NPU comparison, profiling and bounded device scheduling. Goal active;
do not finish after submissions. No sub-agents. Reference repositories and
ObsidianVault read-only. No push by this task; remote-tracking HEAD was observed
at f8648f7. Runtime/checker source is f8648f7, qualification evidence is being
committed separately. Preserve all frozen sources, live launchers and old records.

## Live work and next actions

TASK_ROOT=/mi/data2T/zlong/tide-device-scheduler. Units:
`tide-device-scheduler-NAME.service`; sources/, builds/, runs/NAME/{status.json,
task.log}; repository links artifacts/device-scheduler-NAME. Read matrix.json
and individual summaries, not just job exit: assessment jobs may finish with
failed cells. Heavy timing cells share timing.lock; small gates/builds use
CPU318/319 outside formal affinities but share host memory/fabric resources.

1. matched-baseline03 is running. Clean sources/baseline02 at951031e,
   builds/client-{cpu,npu}-baseline02, launchers/matched-baseline03.py,
   inputs/matched-plan-v2.json. Current NPU Attention training1 holds9 devices;
   construction196.457s completed, first update active. CPU Attention training1
   reached7200-second native timeout after one complete warmup, no measured
   update, no remaining child. That cell's later repeats are skipped by the
   fixed plan; independent cells continue. Do not interrupt valid work.
2. geometry-bounded-eight03 is queued for8 NPUs after the passed NPU8 gate.
   Source bounded-qualified03 atf8648f7; plan bounded-geometry-plan-v3.json;
   launcher geometry-bounded-eight03.py. Actual465-node/4418-wire topology,
   D8/B1/V17, full12-token window, FP32 Add/Attention infer/train replay,
   iterations2/warmup1,900s/32GiB RSS per cell,64GiB conservative workspace
   threshold. Capacity diagnosis only, not full-size timing. It follows the
   failed two-device geometry02 probes below. Do not edit its live plan/launcher.
3. Inspect eight-device geometry, then freeze a finite full-size bounded
   FP32/FP16 capacity/performance plan. None submitted yet. Conservative
   workspace estimates are not measured HBM or proof of fit/impossibility.
4. Finish baseline03 repetitions and its four automatic profiles; do not
   duplicate them. Analyze exports with scripts/summarize_ascend_profile.py.
5. Audit final raw records/source/binaries/Trackio, write separate performance
   evidence, update ROADMAP/STATUS, commit without pushing. Only complete goal
   after authorized local assessment closes; preserve failures and limitations.

## Immutable bounded qualification: complete

Clean sources/bounded-qualified03 atf8648f766e6ded55eeb3eb886cb94bd109c55a7d.
Fresh build-bounded-qualified03 passed, outputs client-bounded-{cpu,npu}-qualified03;
each4 CTests passed. Two compilers onCPU318/319,09:31:03-09:44:41 UTC, overlapping
CPU Attention baseline on separate affinity/shared memory resources.

qualify-bounded-{cpu,npu1,npu2,npu8}-03 all passed:84 native cases in12 groups.
CPU16 includes tiny/actual topology and both dtypes; NPU1 tiny16; NPU2 tiny and
actual36 (including peer primitive); NPU8 tiny16. Actual topology D8/B1/V17/T3.
FP32 atol1e-6/rtol1e-5; FP16 tiny.004/.02 and actual.02/.02; exact discrete/None.
Three SGD/AdamW updates, masters/slots/counters, isolated VJPs and input-changing
replay are checked. Runtime unchanged fromff25174; f8648f7 fixes the checker.

tracked-smoke-bounded-{cpu,npu}-03 passed8/16 tracked wrapper cases. These are
entry/record checks, not performance evidence. reports/audit-qualified03-final01
passed:12 gates/84 cases,24 successful smoke runs and4 failed geometry records.
All source/build/core/topology/lifecycle identities and Trackio SQLite
steps/metrics checked. Audit validity never relabels a failed case as passed.
New stronger launchers/audit-records02.py audits gate source/build identities;
write a new --out report directory, never overwrite prior audits.

Evidence: docs/evidence/bounded-scheduler-qualification-20260929.{md,json},
contract entries, README/navigation and ROADMAP D3-D5. D1/D2/D6 remain active.

## Capacity finding

geometry-bounded02 finished its finite assessment; all four cells failed before
any observation, with `NPU peer create notify: ACL error207009`. Matching CANN
header maps207009 to NO_NOTIFY_RESOURCE; Add inference project plog1736142
reports8192 allocated Notify IDs on physical13. Host peak RSS1.608-1.970GiB;
wall32.66-43.91s/cell. This is Notify exhaustion, not HBM OOM. Eight-device
follow-up tests distribution of that resource; it does not change the runtime.

Full-size12-token topology-only conservative workspace bounds, GiB:
FP32 Add infer610.450/train12873.757, Attention5580.700/19091.282;
FP16 Add305.262/7380.382, Attention2790.387/11112.782.
--workspace-gib up to32768 is a refusal threshold, not reservation. Physical
HBM/host RSS/wall limits remain independent. Full-size bounded timings not run.

## Formal baseline and profiles

First-process measured ms/sample-token: CPU Add infer7.185662, NPU2 Add17.858098;
CPU Attention infer19.531847, NPU4 Attention53.369812; CPU Add train71.323919,
NPU4 Add46.626633. Add measured updates: CPU438.214157s (forward80.517189,
backward344.503377,optimizer13.087897), NPU286.474036s (218.934891/66.704810/
0.822896). No stable speed claim from one process.

CPU Attention first complete warmup4018.027822s: forward1005.783888,
backward2974.257023,optimizer37.986502,loss10.866511,653.975883ms/sample-token.
No steady-state observation; never mix warmup with measured NPU timings.

Common D2048/B512/V50304/T12/seed7. Add9,468,020,899 parameters (historical8.8B
binary label); Attention17,269,426,339. Inference4 warmup/8 measured tokens.
Training2 complete12-token AdamW windows, first warmup. CPU workers/head56 Add,
160 Attention; forward ATen/BLAS1, backward/optimizer16. NPU workers16/head1;
infer2/4 chips, train4/9. Read/control/ranking/events allCPU FP32.

24 planned timing cells,3 fresh processes per combination; failed combinations
skip later repeats. Then4 profiles: Add cpu32 token4; Attention cpu32 token4;
Attention all32 token4; Attention cpu32 backward update1. Collection4GiB,
export600s/8GiB, minimum12GiB disk free. Task sums are not wall time; host API
levels nest. Heavy jobs serialized, small diagnostics/builds have recorded
overlap. matched-baseline02 retained5 cells with CPU backward/BLAS1, then was
cancelled idle; not the primary comparison.

## Numerical and development records to retain

Qualification02 CPU actual FP16 Add isolated root0/leaf1393 failed max0.033325;
NPU2 actual FP32 Add failed max0.000016. Original checker compared1.4*a with
AD0.7*sum(b*b), using unequal rounded upstream values. Shared quantized
cotangents fixed CPU FP16 at the unchanged envelope. Shared radial probes still
had normalization cancellation. norm-probe21 independently reproduced FP32
error in first-node SiLU/RMSNorm without scheduler/peers: CPU32 vsCPU64 max
2.48616e-5; NPU32 vsCPU64 max1.71172e-5; CPU32/NPU32 max1.56164e-5. Native and
composite normalization match per backend. No runtime change to imitate CPU
rounding and no FP32 tolerance widening. v2 gates use two shared output-
independent exact binary-fraction cotangents plus zero; full training unchanged.
Cached dev22 checks preceded the fresh84-cell gate; not qualification substitutes.

Earlier dev15 eight-card tiny gates passed; actual two-card FP32 training passed;
FP16 Add actual training failed at.004/.02 then passed independently at.02/.02.
Old peer/HCCL/source-push/Event failures and all cancelled/precondition-failed jobs
remain. qualified01 builds were cancelled before compiling; npu8-02/geometry01
never acquired devices after failed prerequisites. Do not repeat these old paths.

## Environment and runtime boundaries

Module libtorch-npu/2.10.0-cann9.0.0; SDK under/opt/software/libtorch-npu/.
Task Python /opt/miniconda/envs/ascend900-train-full-torch-npu-2.10.0-py311/bin/python.
Trackio /home/zlong/venvs/trackio/bin/python v0.35.0, best-effort local TASK_ROOT/
trackio, bounded project tide-device-scheduler, old scale project tide-npu-performance.
No dashboard. Last free disk478GiB task volume/30GiB repository; refresh for writes.
320 CPUs, normal budget160;16 logical Ascend910_9392/A3 chips,64GiB/chip.
Never stop other processes or modify cooperative queue manifests/locks.

Core hash5e342902e64abbc384099c3903317955e2eac2c9bfa599cd1a9ddf9eb9fbf439.
OLD=/mi/data2T/zlong/tide-npu-performance is sealed. Its builds/{cpu,core}-fp16-a5
may be reused only at that core identity; do not run its mutating inspectors.

Bounded contract docs/bounded-scheduler.md: finite empty-state Add/Attention,
exact int64 masks/history, first-order VJP, FP32 masters/slots, native NPUGraph
replay with no per-event host tensor decisions. Host constructs, launches and
owns boundaries. Peer Notify marks source-clone readiness; destination waits
and pulls on consumer stream; retained buffers cover window. Explicit reverse
bridge VJPs avoid cross-model autograd Events. Declared peer capacity16384
channels; runtime Notify resources may exhaust sooner. No imported state,
checkpoint/resume, unbounded queue, HST, higher-order AD or arbitrary program.
AiCPU int64 sort remains exact; never convert keys toFP32 for speed.

Prior public/FP16 task closed at8ce1637; its evidence remains unchanged. CUDA
build/host checks do not prove GPU execution. Re-entry: git status --short
--branch; python scripts/status.py; this handoff.
