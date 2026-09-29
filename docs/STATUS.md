# Current handoff

Updated 2026-09-29. Active authorized increment: matched-source CPU/NPU full-size
comparison, full-size profiling and bounded device-resident scheduling (ROADMAP
D1-D6). Branch graph-execution-foundation, HEAD951031e; no push. No sub-agents.
Reference repositories and ObsidianVault remain read-only. Goal is still active.

## Live work and next action

TASK_ROOT=/mi/data2T/zlong/tide-device-scheduler. All task jobs use
`tide-device-scheduler-NAME.service`, frozen sources under `sources/`, separate
`builds/` and `runs/NAME/{status.json,task.log}`. Repository artifact links are
`artifacts/device-scheduler-NAME`. Never edit frozen sources or live launchers.

`matched-baseline03` runs clean `sources/baseline02` at951031e, using
`inputs/matched-plan-v2.json` and fresh `builds/client-{cpu,npu}-baseline02`.
Both builds passed4 CTests. CPU FP64/FP32 and NPU FP32 forward/VJP, CPU/NPU
training qualification and tracked smoke all passed. Four first-process inference
cells passed; the first CPU/NPU Add complete-training pair passed. CPU Attention training is
currently running. Matrix has24 timing
cells (three processes per device/model/mode), followed by4 full-size profiles.
Read `runs/matched-baseline03/matrix.json` for current cell; no stable speed claim
before all relevant repeats finish. Profile collection is capped4GiB, export8GiB
and600s, requiring12GiB remaining disk. Heavy timing/build work shares timing.lock.
Tiny correctness probes use CPUs318,319 outside timing affinity; do not claim
that every kind of task was non-overlapping.

The latest complete development checks are `check-train-diagnostic-{cpu,npu}-dev15`:
CPU8 cells and NPU10 cells passed, including optimizer guards and one captured
update replayed three times with persistent owners, for Add/Attention FP32/FP16.
Their task-local diagnostic binaries reuse audited older object files; they are
not formal clean manifests. `check-peer-diagnostic-dev14` earlier passed10
NPU cells with an unrolled three-update capture, plus changing float/int64/bool
peer probes. `check-wide-peer-dev14` passed Add/Attention two-NPU replay on the
actual465-node/4418-wire topology at D8/B1/T3, three changed input trials.
`check-wide-training-dev15` passed Add/Attention at D8/B1/T3 on two NPUs,
including SGD/AdamW, three persistent-owner updates and three input trials.
`check-isolated-diagnostic-dev16` passed four FP32/FP16 Add/Attention isolated
root/None/zero VJP cells using the explicit peer tape.


The decisive repair is in peer_transport.cpp: Notify marks source-clone readiness,
then the destination pulls data on its consumer stream. Source-stream push
passed elementary probes but misordered a complex captured consumer. It is no
longer the implemented protocol. All failures and single-TU diagnostic builds
from dev08-dev13 remain sealed. Adding DeviceGuard alone did not repair it and
that diagnostic-only change was removed. Current working tree also adds stronger
replay diagnostics, changing small-mask/large-int peer probes and optimizer
missing-gradient rejection.

`build-bounded-dev14` was cancelled while waiting on timing.lock, before any
compiler ran, because its snapshot was superseded. The attempted
`hold-matched-for-dev14` refused because the next timing cell had already
started; it sent no STOP signal. The baseline coordinator continues normally.
Dev09 CPU8 checks passed; its NPU check retains the old source-push failure.

`build-benchmark-diagnostic-dev17` compiled four changed translation units per
backend sequentially on CPU318. This bounded development compilation overlapped
the formal CPU job on separate affinity; it is not a fresh build. Its
`smoke-benchmark-{cpu,npu}-dev17` passed8 CPU and16 NPU eager/replay benchmark
cells, checking complete metric inventories and custom peer-buffer/channel
counts. CPU also checked8 full-size topology-only workspace refusals before
weight allocation. The conservative bound is not measured HBM. Latest source
allows an explicit refusal threshold up to32768GiB for bounded capacity trials;
host RSS and the requested hardware's real allocators remain separate limits.

Next: commit this tested implementation, freeze that clean commit, fresh-build
both consumers and run verify_bounded_scheduler.py on CPU/one/two-NPU FP32/FP16
and reduced actual topology. Validate benchmark_bounded_scheduler.py with tracked
smoke before bounded performance/capacity assessment. Finish baseline03 repeats
and4 profiles, then write a separate evidence commit. Never label the cached
development binaries as immutable qualification or their smoke timings as
performance evidence. No push.

## Implementation and evidence boundaries

Committed cba3989: CPU node parallelism, phase-scoped CANN profiler and bounded
profile output monitoring. Committed9dbf124: CPU vocabulary-head workers.
Committed951031e: scoped CPU backward/optimizer ATen and OpenBLAS thread pools,
with effective counts in metrics; eight three-update parity cells passed at16
threads. Public/core C++ source hash remains
5e342902e64abbc384099c3903317955e2eac2c9bfa599cd1a9ddf9eb9fbf439.
Reuse OLD=/mi/data2T/zlong/tide-npu-performance builds/{cpu,core}-fp16-a5 only
while that identity matches. Do not rerun OLD's mutating FP16 attempt inspector.

The uncommitted bounded consumer is described in docs/bounded-scheduler.md.
It statically expands a finite empty-state historical topology/window, carries
exact int64 histories and device presence/selection masks, and uses explicit
first-order VJPs with structural None/zero connectivity. Optional native
NPUGraph replay avoids per-event host decisions. Workspace limits are conservative
refusal bounds, not proof that full-size capture fits or is fast. No arbitrary
initial state, checkpoint/resume, unbounded queue, HST or higher-order AD claim.

Development evidence: dev01 CPU complete observables/isolated VJPs passed.
Dev04 CPU and two-NPU eager FP32 forward/training passed, as did single-NPU
Add/Attention forward/training capture. Dev05 CPU8 cells passed with declared
FP16 envelope; single-NPU FP32 replay/training passed. Dev05 two-device forward
replay had numeric mismatch, training failed107032 (cross-model autograd Event).
Its matrix was cancelled; failures remain. Dev07 CPU build/4 CTests passed;
NPU build lacked explicit grad_mode header, now fixed. Its dependent checks did
not execute. Dev06 was cancelled while waiting, before any compiler started.

Cross-device probes: ordinary Torch copy capture fails107027; HCCL capture
fails107025, including persistent-thread/AI_CPU variants. Ordinary external
Events v7/v8 misordered even eager work. CANN IPC Notify v10 passed two warmups
and five changed-input captured replays on two NPUs. This is a primitive result,
not D5 acceptance. The peer adapter uses retained immutable buffers, finite notification inventory
and local VJPs joined by Notify and destination pulls; later dev14/dev15 gates
passed as recorded above. Single-device
int64 sort/pairwise probes also passed above2^55; AiCPU is device execution.

`matched-baseline02` was cancelled while idle on timing.lock after five completed
cells; no workload interrupted. Its CPU training used backward/BLAS1 and is
preserved as an under-parallelized result, not the primary comparison.
All earlier failed/cancelled records retain original states and reproducers.

## Environment and prior baseline

Native NPU module: public libtorch-npu/2.10.0-cann9.0.0. CPU build remains separate.
Trackio writer/viewer:/home/zlong/venvs/trackio/bin/python, version0.35.0;
local best-effort root TASK_ROOT/trackio, no dashboard needed/exposed. Existing
scale wrapper projects to tide-npu-performance; bounded wrapper uses
tide-device-scheduler. Read current storage/resource bounds before large writes.

Prior task is closed at8ce1637: see evidence/accelerator-fp16-performance-20260929.md,
evidence/fp16-qualification-20260929.md and evidence/accelerator-profile-analysis-20260929.md.
Old jobs are terminal; never append to their sealed resource logs. Existing
all32 keeps host-owned dispatch/tensor handles despite NPU payload/control work.
CUDA build/host checks do not establish NVIDIA execution. Re-entry:
`git status --short --branch`; `python scripts/status.py`; read this file.
