# Current handoff

Updated 2026-09-29 11:44 UTC. Authorized increment: ROADMAP D1-D6, matched-source
full-size CPU/NPU comparison, profiling and bounded device scheduling. Goal active;
do not finish after submissions. No sub-agents. Reference repositories and
ObsidianVault read-only. No push by this task; remote-tracking HEAD was observed
at f8648f7. Runtime/checker source is f8648f7, qualification evidence committed separately at 158dc76. Preserve all frozen sources, live launchers and old records.

## Live work and next actions

TASK_ROOT=/mi/data2T/zlong/tide-device-scheduler. Units:
`tide-device-scheduler-NAME.service`; sources/, builds/, runs/NAME/{status.json,
task.log}; repository links artifacts/device-scheduler-NAME. Read matrix.json
and individual summaries, not just job exit: assessment jobs may finish with
failed cells. Heavy timing cells share timing.lock; small gates/builds use
CPU318/319 outside formal affinities but share host memory/fabric resources.

1. matched-baseline03 is running. Clean sources/baseline02 at951031e,
   builds/client-{cpu,npu}-baseline02, launchers/matched-baseline03.py,
   inputs/matched-plan-v2.json. First round:7 passed/CPU Attention training
   timed out; all4 inference cells of repeat2 also passed. NPU Add training2
   is running after the bounded12 follow-up. Ten timing cells remain including
   this live cell, then4 profiles. NPU Attention training1
   measured update788.123357s (forward647.207837/backward136.951850/
   optimizer3.915472),128.275286ms/sample-token; warmup857.817752s.
   CPU Attention training1 hit7200s after one complete warmup, no measured
   update, no remaining child. Its later repeats are skipped by the fixed plan.
   Complete the remaining repeats and4 profiles; do not interrupt valid work.
2. geometry-bounded-eight03 completed10:44:40 UTC. All4 native cells FAILED
   before any observation: bounded peer notification capacity exceeded16384.
   Actual465-node/4418-wire topology,D8/B1/V17/T12,FP32 Add/Attention infer/train
   replay on8 devices. Finite assessment exit0 is not a passing workload.
   reports/audit-geometry-eight03-final01 passed record/source/binary/Trackio audit.
   Keep this distinct from the2-device runtime NO_NOTIFY_RESOURCE failure.
3. bounded-full01 completed11:21:01UTC:6 native attempts,2 passed/4 failed;
   10 cells skipped by prerequisites. Sourcef8648f7,8 devices,fullD2048/B512/
   V50304/T12,eager/replay Add/Attention infer/train finite plan preserved.
   Add eager inference passed: FP3239.380302s andFP1641.778807s per reset-state
   measured window; max per-device allocator peaks11.760/5.885GiB. Only one
   process/dtype, same allocation/locality policy; no exact-node-map proof.
   Attention inference and Add training each OOM in both dtypes before any
   observations. Attention training skipped after inference OOM; all replay
   skipped after corresponding small-tensor12-token Notify-capacity failures.
   Skips are not full-size tests or capacity failures. reports/audit-bounded-
   full01-final01/audit.json passed all6 records/source/binary/Trackio checks.
   Add inference overlapped outside processes on physical9; preserve samples.
   Frozen plan/launcher and original raw files must not be edited.
4. bounded-twelve01 completed11:34:26UTC. It acquired12 devices after24.16s
   queue wait. Both actual-topology D8/B1/V17/T3 Attention forward/isolated-VJP
   prerequisites passed(FP32 unchanged;FP160.02/0.02); no12-device replay or
   complete-training qualification claim. One fullD2048/B512/V50304/T12 FP16
   eager inference attempt then FAILED with NPU OOM before observations.
   Failed logical8=physical9:owned allocation33.95GiB,reserved37.15GiB,free96.98MiB,
   request178MiB. Observer02 samples11:32:43-11:34:19 show outside processes on
   that device(also8). Earlier overlap on7 also recorded. Thus this does NOT
   establish an uncontended12-device capacity limit. Admission was free, but
   unrelated tasks later entered advisory-locked devices; never stop them.
   reports/audit-bounded-twelve01-final01 passed1 native record/1 gate(2 cases).
   Frozen launchers/bounded-twelve01.py and inputs/bounded-twelve-plan-v1.json,
   allocation/queue/lifecycle/Trackio and failures retained. This one-attempt
   capacity follow-up is terminal; do not silently rerun under the same load.
5. Finish baseline03 repetitions and its four automatic profiles; do not
   duplicate them. Analyze exports with scripts/summarize_ascend_profile.py.
6. Capacity evidence is complete at docs/evidence/bounded-scheduler-capacity-
   20260929.{md,json}, with README/contract/navigation updated; commit separately.
   Reports bounded-analysis-final01 and audit-bounded-capacity-final01 cover
   all15 experiments(2 success/13 failures),2 new12-device gate cases,10 skips.
   Resource inputs were copied to hashed immutable snapshots within the analysis
   report; original observers remain live. Support schema and record audits pass.
   Finish baseline/profile analysis and audit, then final evidence/ROADMAP/STATUS
   commit without pushing. Only complete goal
   after authorized local assessment closes; preserve failures and limitations.

The low-frequency observe-resources01 service records npu-smi/queue states and
project PIDs every30s in runs/observe-resources01/observations.jsonl, starting
10:38 UTC. It reads only, never stops/changes workloads, and exits when its three
original jobs terminate (12h wall bound). New observe-resources02 started11:10UTC,
also tracks bounded-twelve01 and exits after all four workloads terminate. Its
separate launcher/records preserve the original observer; use02 for follow-up
allocation reservations and later shared-load evidence. Earlier activity is not covered. At10:38-10:48,other processes used4-6 chips;
no sampled interval showed12 actually unoccupied chips. Distinguish project
use/reservations from unrelated load and instantaneous zero AiCore.
launchers/summarize-bounded01.py reads geometry/full/observer records into a
new --out; reports/bounded-analysis-pilot02 is partial,not final; includes both observers and12-device job.
launchers/analyze-profile01.py adds per-device operator interval unions and
level-separated host API categories using the repo CSV summarizer. Tested on
retained profile-smoke-dev01 export and1000 independent integer-grid coverage
sets. The tiny trace remains a helper check,not full-size profile evidence.
reports/audit-baseline03-pilot02 passed12 terminal records/5 gates;partial only. Observations
can show outside processes entering an advisory-locked device; no exclusivity claim.
launchers/summarize-matched01.py builds process-level/warmup-separated analysis;
reports/matched-analysis-pilot01 is partial (7 terminal timing records), not a
final report. Use a new --out, omit --allow-live only after matrix completion.
reports/geometry-notify-diagnostic01 preserves header identity and matching
project plog error lines for all4 geometry02 failures (8192 Notify IDs).

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
HBM/host RSS/wall limits remain independent. Both8-device and12-device bounded assessments are terminal as above. The12-device failure has observed external device contention.

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
