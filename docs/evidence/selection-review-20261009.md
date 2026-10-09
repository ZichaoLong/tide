# Decision-sufficient CPU/mixed/resident selection review

Reviewed2026-10-09 under the user-authorized selection criterion:every performance
cell need not execute. The baseline supports practical choices;two finite
follow-ups are running/queued and their results are not included here.
[Scalar evidence and artifact hashes](selection-review-20261009.json).
[Live state](../STATUS.md);[execution contract](../execution-flows.md).

## Selection from existing observations

The original bound FP32 series has **61/120 accepted cells**;8 older unbound
observations stay separate. All4 separate FP16 companions have first-process
results. Timings below are seconds per complete measured step (lower is faster),
after one warmup;each step consumes12288 input tokens across2 connected windows.
D2048/B512/T12/V50304;Add9,468,053,696 parameters,Attention17,521,117,376.
Construction is separate. Each candidate computes independently;no CPU reference
trajectory is fed into another candidate.

| LibTorch PDG scenario | CPU seconds | Mixed seconds | Resident seconds | Engineering guidance |
| --- | ---: | ---: | ---: | --- |
| Add prefill inference | 260.290 | 678.338 (B) | 329.308 | Start with CPU |
| Add prefill complete training | 934.022 | Missing | 2346.573 | CPU over resident;mixed prefill unresolved |
| Add streaming complete training | 1134.130 | 1577.887 (B) | 8683.074 | CPU;prefer prefill when the use case permits |
| Attention prefill inference | 2368.878 | 1611.426 (A) | 403.239 | Resident is the strong current candidate |
| Attention prefill complete training | Pending targeted baseline | 3104.021* (A) | 6080.594 | Mixed ahead of resident in this observation;CPU comparison pending |
| Attention streaming inference | 2820.718 | 1952.465 (C) | Timeout,no complete result | Mixed has an observed advantage over CPU |

Resident Attention prefill inference has about5.87times the CPU input throughput.
Resident Add prefill training has about0.40times the CPU input throughput
(2.51times the elapsed time):it is slower. These are observed input-throughput
ratios,not certified equal-event speedups. Actual event counts and physical groups
differ,strict near-tie route failures remain,and external contention is not
excluded. Single shared-server processes do not establish statistical stability.

The Add prefill training direction also appears in LibTorch TimedDAG/Settle:
CPU978.149/974.595s versus resident2361.619/2363.921s. Settle LibTorch Add inference
is CPU214.649s versus mixed-A541.870s/resident331.282s. TimedDAG LibTorch Attention
inference is CPU1991.864s versus resident405.188s. Thus there is useful cross-family
evidence;large-topology testing is not limited to PDG.

Client identity matters. Python CPU prefill Add inference takes643.225s(TimedDAG)
and834.559s(Settle),versus346.342s/345.415s for Python-owned native resident.
For Python callers,resident can therefore be the faster Add inference choice;
the independent LibTorch CPU Add path remains faster in its recorded cases.
Python Add training instead favors CPU:1627.148/1618.334s versus resident
2468.936/2461.584s. Python resident is a C++/CANN backend owned by Python,not an
independent pure-PyTorch resident scheduler. Python Attention prefill inference
also favors resident in both measured families. Keep client identity explicit.

Prefill is the initial throughput-oriented choice for these profiles;streaming
remains configurable for latency/input constraints. PDG CPU Add inference is
260.290s prefill versus570.014s streaming. Greedy batches can naturally degenerate
to streaming;these observations are not a guarantee for every topology/input or
machine. Avoid selecting one universal CPU/NPU winner across workloads and clients.

## Retained failures and over-bound observations

The old audit-only summary omitted10 CPU failures occurring before any assessment
file existed. Reconciliation now merges terminal receipts and audits,deduplicates
references,and retains historical failures even when a later attempt succeeds.
All accepted metric records are copied unchanged.

| Current unaccepted bound outcome | Cells |
| --- | ---: |
| NPU admission timeout;model not executed | 37 |
| CPU NUMA memory admission refusal;model not executed | 10 |
| Complete observation exceeded original step-time allowance | 5 |
| Execution total-time bound;no complete result | 4 |
| Monitor procfs process-exit race | 2 |
| NPU OOM;external occupancy/vendor overhead attribution unresolved | 1 |

Five completed-over-bound cases were read-only revalidated for source/runtime/
binary/workload identity,full output counts and continuation,finite loss/timing,
placement,allocator/RSS budgets,absence of reported CPU fallback,and terminal
empty-cgroup state. Original case states,failed receipts and budgets are unchanged.
These are engineering observations outside formal acceptance,not five new passes.

| Cell/configuration | Measured seconds | Original step limit |
| --- | ---: | ---: |
| 22 PDG LibTorch Attention mixed-A prefill training | 3104.020558 | 3000 |
| 34 PDG LibTorch Attention mixed-A streaming training | 3087.987731 | 3000 |
| 36 TimedDAG LibTorch CPU Add streaming inference | 699.995797 | 600 |
| 60 Settle LibTorch CPU Add streaming inference | 773.663847 | 600 |
| 113 Settle Python-owned native resident Add streaming training | 8734.147430 | 8700 |

The starred mixed-A training time in the first table is one of these observations.
Warmup times remain separately recorded. The strict near-tie numerical limitation
is independent of wall-clock classifications and remains explicitly nonblocking
by user choice;do not relax comparisons or claim all wide trajectories equivalent.

## Profiling and FP16

Original-size resident Add's five-second trace has166118 AI_VECTOR_CORE tasks,
2190 AI_CORE tasks and14 MIX_AIV tasks;no AiCPU task was observed. Vector tasks
account for87.18% of summed task durations,with gather/scatter,selection and
packing prominent. CPU uses32 physical sample rows (16groups) versus resident4
rows (128groups). Smaller batches,boundary synchronization and data movement
are supported optimization hypotheses,not a measured causal decomposition.
[Add trace](fullsize-resident-add-profile-20261004.md).

The subsequently completed FP32 Attention inference slice contains259354 vector,
13338 AI_CORE and36 MIX_AIV tasks across11 cards,with no observed AiCPU. AI_CORE
accounts for24.78% of summed task duration. Task durations overlap across streams
and cards;these shares are neither end-to-end time shares nor utilization.
The mixed-A Add slice likewise supports many small operations and host calls;
its4500s profiling timeout still has no full-training result.
[Mixed slice](fullsize-mixed-profile-slice-20261004.md).

These traces do not support attributing NPU slowness to AiCPU. Absence in sampled
intervals does not exclude AiCPU elsewhere. Scheduling,packing,communication,
chunking and host boundaries remain relevant. Full-training phase attribution
is incomplete and does not block the current flow choice.

FP16 TimedDAG LibTorch resident prefill observations:inference Add194.098s and
Attention222.541s (8cards);complete training Add2027.354s (8cards) and
Attention6190.644s (11cards). Payloads areFP16;adjoints,loss and masters areFP32.
Cache-off policy and placement/card counts differ from primary FP32;no isolated
dtype speedup claim follows. Four first observations suffice under the latest
policy. [FP16 details](formal-b512-fp16-training-profiles-20261004.md) retains its
dated observation boundary;the JSON linked above includes later Attention training.

## Qualified support and remaining scope

| Path | Current evidence/limit |
| --- | --- |
| aarch64 CPU | Independent FP64/FP32 reference/native/LibTorch paths;9458 checks+12 CTests,654 scoped optional skips |
| NPU eager/mixed and resident,FP32/FP16 | Current e69 integration49 eager+93 resident checks;complete training,continuation,SGD/AdamW and repartition |
| Required family/client/schedule scope | PDG LibTorch;TimedDAG/Settle LibTorch and Python;all10 representative submatrices qualified |
| Older CANN/SDK combinations | Separately scoped older eager/standalone gates;not blanket resident certification |
| NVIDIA/x86_64/other target versions | Source/build/commands available;actual target execution remains pending |
| New selector/expanded upstream semantics | Deferred until updated canon is supplied |

The controller-only monitor repair atd773fe2 treats ENOENT/ESRCH during procfs
reads as process/thread exit;permission/I/O errors and live placement violations
still fail. Clean immutable qualification passed27 focused checks. No model code,
route tolerance,capacity rule or original acceptance budget changed.

Two one-attempt serial follow-ups are submitted as a detached background service:
CPU PDG Attention prefill training with a separately labelled eight-NUMA memory
mask,and Settle resident Attention prefill inference after the monitor fix.
Original CPU affinity/thread counts,memory guards and step bounds remain.
Two lower-value candidates(cells16/59) are deliberately not scheduled. Whole
protective queue limit27920s;no retries,full-matrix restart or budget increase.
On wake-up,review their terminal audits and update scenario guidance. F6/F7 local
closure then needs final outcome/support reconciliation and no live task jobs.
Missing performance cells,external-machine tests and downstream training quality
are not hidden as local passes or automatic new experiment obligations.
