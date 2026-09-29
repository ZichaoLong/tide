# Current handoff

Updated 2026-09-29. Active authorized increment: ROADMAP D1-D6, matched-source
full-size CPU/NPU comparison, full-size profiling and bounded device scheduling.
Goal remains active. Branch graph-execution-foundation, HEAD ff25174; no push.
No sub-agents. Reference repositories and ObsidianVault remain read-only.

## Live work and immediate next action

TASK_ROOT=/mi/data2T/zlong/tide-device-scheduler. Services use
`tide-device-scheduler-NAME.service`, frozen sources under sources/, independent
builds/, runs/NAME/{status.json,task.log}; repository links are
artifacts/device-scheduler-NAME. Never edit frozen sources or live launchers.

Current uncommitted changes: bounded_check.cpp, verify_bounded_scheduler.py,
docs/bounded-scheduler.md and this file. The checker now gives both independent
schedules two common, output-independent binary-fraction cotangents and a
connected-zero probe. Gate records become v2. FP32 tolerances remain1e-6/1e-5;
FP16 explicit envelopes remain0.004/0.02 tiny and0.02/0.02 reduced real topology.
No runtime or public/core mathematics changed in this follow-up.

Frozen dirty sources/bounded-dev22 passed check-cotangent-{cpu,npu}-dev22,
all4 actual-topology Add/Attention FP32/FP16 cells per backend, including full
observations and isolated VJPs with two directions plus zero on CPU/two NPUs.
The diagnostic binaries use the new checker TU and immutable qualified02 runtime
objects. All service/child exits were successful. All these
cached-object checks remain developmental. Compile/link/input/binary identities
are in runs/build-cotangent-dev22/identity.json. The checker follow-up is ready for commit; freeze its clean source, freshly build
and repeat formal CPU/1/2/8-NPU gates as qualified03. Prepared external drivers
are build-bounded-qualified03.py, qualify-bounded03.py and tracked-smoke-bounded03.py.
Recreate geometry capacity assessment against the accepted new build, then
schedule bounded full-size eager/replay FP32/FP16 assessment. Finish baseline03
and its4 full-size profiles, audit records/Trackio/source/binaries, then create
a separate evidence commit. Do not finish merely after submitting services.

## Numerical finding and retained failures

Fresh build-bounded-qualified02 passed CPU/NPU builds and4 CTests each, clean
sources/bounded-qualified01 at ff2517450493543e0dfed3b3bc57f00bcf3787ab. Outputs
builds/client-bounded-{cpu,npu}-qualified02. Qualification02: CPU tiny both
dtypes and wide FP32 passed; wide FP16 isolated root0/leaf1393 failed (max0.033325,
reference max0.525879). NPU1 tiny both dtypes passed. NPU2 tiny both and wide FP16
passed; wide FP32 isolated root0/leaf1393 failed (max0.000016). Consequently
qualify-bounded-npu8-02 and geometry-bounded01 failed on prerequisites, with no
case/device acquisition. All terminal failures are preserved.

fp16-vjp-diagnostic19 showed that sharing one quantized cotangent fixes CPU Add
FP16 at the unchanged wide envelope; Attention old/shared and both wide CPU
complete-training cases passed. check-cotangent-cpu-dev20 passed4 cells; NPU
passed FP16 Add and both Attention cells, but Add FP32 retained max1.56e-5 error.
The old gate compared1.4*a with AD of0.7*sum(b*b), giving different rounded
upstream values. Sharing the cotangent removed that defect, but its radial
norm-squared direction is still cancellation-sensitive.

norm-probe21 exactly reproduced the FP32 discrepancy in the first-node
SiLU/RMSNorm primitive (embedding leaf1393), independent of scheduling/peers.
Against CPU FP64: CPU FP32 max error2.48616e-5, NPU FP32 max1.71172e-5; CPU/NPU
difference1.56164e-5. Native/composite RMSNorm gave identical results per backend.
No runtime normalization change was made to imitate CPU rounding. Dev22 uses
standard isolated-Jacobian probes with two output-independent common directions;
complete objective/optimizer checks remain independent and unchanged. Preserve
all old radial-probe findings, do not describe them as a runtime semantic fix.

Earlier full training: check-eight-dev15 passed all8 tiny FP32/FP16 Add/Attention
captured forward/training cases, SGD/AdamW,3 updates and3 input trials. Actual
465-node/4418-wire topology at D8/B1/T3 passed two-NPU FP32 forward/training in
check-wide-peer-dev14 and check-wide-training-dev15. check-wide-fp16-training-dev15
failed Add AdamW at atol0.004/rtol0.02 (max0.013916). Independent
check-wide-fp16-add-dev18 passed at0.02/0.02; Attention dev18 passed at0.004/0.02.
All were developmental cached-binary evidence. Earlier dev08-dev13 communication
failures and dev09 checks stay sealed; see their raw records for reproducers.

## Formal baseline and pending performance

matched-baseline03 uses clean sources/baseline02 at951031e, plan
inputs/matched-plan-v2.json and builds/client-{cpu,npu}-baseline02. Its complete
forward/training qualification and tracked smoke passed. It runs24 timing cells
(3 processes per device/model/mode), then4 full-size profiles. Failed cells are
not repeated; independent cells continue. Inspect its matrix.json for progress.
CPU Attention training1 is active. Its first complete warmup update finished:
4018.027822s total,1005.783888s forward,2974.257023s backward,37.986502s optimizer,
loss10.866511. This is warmup, not a measured steady-state result. The second
update is running under the existing7200-second native wall bound. No interruption.

Six first-process timings are complete (ms/sample-token): CPU Add inference
7.185662, NPU2 Add17.858098; CPU Attention19.531847, NPU4 Attention53.369812;
CPU Add complete training71.323919, NPU4 Add46.626633. Add measured updates:
CPU438.214157s (forward80.517189/backward344.503377/optimizer13.087897),
NPU286.474036s (218.934891/66.704810/0.822896). No stable speed claim from1 process.

Common D2048/B512/V50304/T12/seed7; Add9,468,020,899 parameters, Attention
17,269,426,339. Inference4 warmup/8 measured tokens. Training2 complete AdamW
windows, first warmup. CPU workers/head56 Add or160 Attention, forward ATen/BLAS1,
backward/optimizer16. NPU workers16/head1;2/4 chips inference and4/9 training.
All matched baseline Read/control/ranking/events are CPU FP32. Four pending
profiles: Add cpu32 token4; Attention cpu32 token4, all32 token4, cpu32 backward
update1. Collection4GiB, export600s/8GiB, at least12GiB disk free. Do not duplicate
these profiles without coordinating the baseline driver. Task sums are not wall time.

Heavy performance trials share timing.lock. Small gates/probes and compilation
use CPUs318/319 outside timing affinity. qualified02 fresh build ran08:52:51-
09:06:24 UTC with two total compilers on those cores, overlapping CPU Attention
warmup and shared memory resources; record that caveat. Qualified01 build and its
four checks were cancelled while waiting, before any compiler/case/device began.
Never claim all kinds of task were non-overlapping. matched-baseline02 retained
5 old cells with backward/BLAS1 and was cancelled while idle; not primary comparison.

tracked-smoke-bounded-{cpu,npu}-02 passed8 CPU/16 NPU formal wrapper/Trackio
cells with validated terminal records and healthy local Trackio. These timings
are smoke only. Earlier native dev17 smoke passed8 CPU/16 NPU cases plus8 expected
topology-only full-size capacity refusals. The conservative12-token full-size
bounds (GiB) are FP32 Add infer610.45/train12873.76, Attention5580.70/19091.28;
FP16 Add305.26/7380.38, Attention2790.39/11112.78. These are not measured HBM or
proof of fit/impossibility. Explicit threshold now permits up to32768GiB; actual
allocator, RSS and time limits still apply. Full-size bounded trials are not run.
geometry-bounded01 plan remains inputs/bounded-geometry-plan-v1.json: actual
12-token geometry, D8/B1/V17,2 NPUs, FP32 Add/Attention infer/train,900s/32GiB RSS,
64GiB workspace refusal threshold; no cell ran because qualification02 failed.

## Runtime and support boundaries

Native module libtorch-npu/2.10.0-cann9.0.0. SDK under /opt/software/libtorch-npu/.
Task Python /opt/miniconda/envs/ascend900-train-full-torch-npu-2.10.0-py311/bin/python.
Trackio /home/zlong/venvs/trackio/bin/python v0.35.0, best-effort local root
TASK_ROOT/trackio; bounded project tide-device-scheduler, scale project
tide-npu-performance. No dashboard/exposure. Last disk check138GiB task volume,
31GiB repository volume; refresh before large writes. CPU320, normal budget160;
16 logical Ascend910_9392/A3 chips,64GiB/chip. Do not stop other users' processes.

Core C++ hash remains5e342902e64abbc384099c3903317955e2eac2c9bfa599cd1a9ddf9eb9fbf439.
Reuse OLD=/mi/data2T/zlong/tide-npu-performance builds/{cpu,core}-fp16-a5 only with
that identity. OLD is sealed; never rerun its mutating FP16 attempt inspector.

Bounded consumer is committed atff25174, docs/bounded-scheduler.md owns scope:
finite empty-state Add/Attention windows, exact int64 masks/history, explicit
first-order VJP and FP32 optimizer masters/slots. Native NPUGraph replay has no
per-event host tensor decisions. Multi-device Notify marks source-clone readiness;
destination waits and pulls on its consumer stream. Retain buffers throughout
the window. Explicit reverse bridge VJPs avoid unsupported cross-model autograd
Events. Capacity16384 peer channels. No imported state, checkpoint/resume,
unbounded queue, HST, higher-order AD or arbitrary program support. AiCPU int64
sort is accelerator execution; never cast exact integer keys to FP32.

Prior task closed at8ce1637; previous FP16/performance/profile evidence is in
docs/evidence/*20260929*. CUDA compiled/host checks do not prove GPU execution.
Re-entry: git status --short --branch; python scripts/status.py; this handoff.
