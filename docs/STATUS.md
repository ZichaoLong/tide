# Current handoff

Updated: 2026-09-29. Branch graph-execution-foundation; no push authorized.
No sub-agents or reference-repository changes. User authorized full FP16 scope
(public Python/native API and standalone full-size consumer), correctness gates,
profiling and bounded full-size comparisons. Existing FP32 evidence is immutable.

## Current implementation and verified scope

Implementation fc2a76f adds explicit FP16 payload, FP32 master optimizer and
static loss scale, same-dtype CPU oracle and configurable floating tolerances.
cb58110 orders profiler fixture/master copies before its worker stream.
595dccd strengthens consumer checks with FP32 gradient normalization and master
weight trajectories. c8d2b61 adds early rejection of CPU FP16 CSR (local Torch
Half sparse-dense matmul is unavailable); its direct rejection test passed and
boundary-fp16-a5c passed20 focused immutable regressions;4a7dec7 additionally
records the expected rejection in the CLI suite, verified by csr-boundary-fp16-a5d. No mathematical kernel
change after fc2a76f; complete
C++ core source hash remains identical at 595dccd. Public Python checkpoint
supports master state; standalone owner-checkpoint/NamedOptimizer stays FP32/64.

Immutable sources: TASK_ROOT/sources/fp16-a5 (fc2a76f), fp16-profile-a5b
(cb58110), fp16-a5b (595dccd). Builds cpu-fp16-a5, adapter-fp16-a5, core-fp16-a5,
client-fp16-a5b, cuda-fp16-a5b passed with two workers. Latest consumer four CTests
passed. CUDA build and22 CPU/CLI checks passed; no NVIDIA hardware qualification.

Public NPU FP16 complete named suites passed: python-npu-fp16-a5 39 cases,
native-npu-fp16-a5 43 cases and one explicit CSR rejection. Each case covers full
observables/discrete routes, isolated VJPs, chunking, three updates, fresh-process
checkpoint and CPU handoff. Five representative FP32 cases each also passed.
profile-{python,native}-fp16-a5b passed with1644/1638 hardware kernel events;
no named host-fallback event in those tiny traces. Not full-size attribution.

consumer-{cpu,npu2,npu4,npu8}-fp16-a5b all passed:100 positive configuration cells,
both dtypes; CPU/NPU2 CPU64 and all32, NPU4 CPU64 inference and CPU32 training,
NPU8 mixed32 training. FP16 atol=.002/rtol=.02; public FP16 atol=.001/rtol=.02.
Routes/events/None-vs-zero remain exact. Consumers include three optimizer updates
and FP32 master/slot checks; only configured floating comparisons are relaxed.

## Active jobs and next action

cpu-fp16-a5 passed8645 tests in1629.46s; unit inactive, ExecMainStatus0 and no PID.
The full CPU regression and every scoped correctness gate are now complete.
Qualification/profile evidence is ready for its separate documentation commit;
performance records remain independently in progress.

Four bounded performance jobs submitted on595dccd, client-fp16-a5b:
pair-infer-add-fp16-a5b (2 NPUs), pair-infer-attention-fp16-a5b (4),
pair-train-add-fp16-a5b (4), pair-train-attention-fp16-a5b (8).
Every applicable correctness gate passed before submission. Jobs use immutable
launchers/run-dtype-pair-a5b.py OUT launchers/plan-{MODE}-{MODEL}-fp16-a5.json;
exact argv/cwd/source/log/state are in runs/NAME/status.json and launchers/NAME.sh.
A pair holds one cooperative allocation, FP32 then FP16 fresh processes.
D2048/B512/V50304, locality placement, resident messages/state, workers16/ATen1,
TASK_QUEUE_ENABLE=0. Inference12tokens/4warmup; training two12-token updates,
first warmup, second measured. Eight cells total. First completed pair: Add2
inference FP32 17.412729 vs FP16 18.703945 ms/sample-token; max per-chip allocator
peak over all events19.868474 vs9.954680 GiB. Same source/topology/parameter count/
node placement and clean exit checks passed. Selected counts match; other work
counts change slightly across dtypes. One pair cannot establish causal speedup.
Attention4 inference also passed:54.611865 vs45.855257 ms/sample-token,
peak22.286432 vs11.192269 GiB. All pair identity/placement/exit checks passed.
Both training pairs remain running/queued; do not call them complete.

All units tide-npu-performance-NAME.service run in background.slice with Nice10.
Inspect: systemctl --user show UNIT -p ActiveState -p SubState -p ExecMainStatus;
read TASK_ROOT/runs/NAME/{status.json,task.log,queue.json,stages.json}. Stop only
an identified task when needed: systemctl --user stop UNIT. Each pair has a7200s
queue bound, inference3600s/cell or training7200s/cell and256GiB RSS limit.
Trackio best-effort local project tide-npu-performance, TASK_ROOT/trackio, SQLite;
writer/viewer /home/zlong/venvs/trackio/bin/python. Raw JSONL is authoritative.

Next: monitor terminal results; diagnose failures without overwriting attempts.
After all pass, run launchers/collect-fp16-evidence-a5.py qualification and
performance, validate every raw experiment record and compare same placement,
parameter counts, workload, timings/peak memory/loss. Write reviewed small reports,
update ROADMAP and support contract, commit evidence separately; do not push.
Qualification and profile reports, support contract and precision wording are
staged for a separate evidence commit. Two inference pairs are complete; training
results must still be collected, validated, documented and separately committed.
All builds/raw outputs remain outside source under ignored artifacts symlinks.

## Retained limits and failures

Task root /mi/data2T/zlong/tide-npu-performance. /opt public modules and shared
/usr/local driver remain. The preceding b4f26b3 FP32 performance assessment and
four-CANN qualifications remain separate; new FP16 is tested on2.10/CANN9.0 only.
NPU ranking/event tensor work exists; host histories, scalar extraction, tensor
handles and C++ dispatch remain. Tiny old msprof trace:55 AiCPU int64 Sort tasks,
3365.9/23020.06us summed task time (14.62%), not end-to-end/full-size attribution.

Development failures retained: dev01 invalid ATen half test literal (fixed),
cross-precision gradient/Attention-route mismatches; same-dtype dev03 VJP narrowly
failed atol=.001 then explicit .002 passed. consumer-cpu-fp16-dev06 rejected an
erroneous two-device CPU launch; dev06b passed. cuda-build-fp16-a5 was cancelled
early to fix a planned test filename; independent cuda-host-fp16-a5b passed.
No numerical tolerance change hides discrete failure. FP16 does not certify
convergence or route identity to FP32. CUDA devices/new stacks/other host
architectures remain target-machine acceptance.

Re-entry: git status --short --branch; python scripts/status.py.
