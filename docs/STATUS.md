# Current handoff

Updated 2026-10-01. **PAUSED at the user's request after this evidence commit/push.**
Do not resume implementation or submit new experiments until the user confirms
the progress/goal/priority/contract alignment. All current qualification jobs are
terminal. The historical CPU Attention task remains intentionally paused.
No subagents. Reference repositories and ObsidianVault are read-only.
Repository `/home/zlong/llm/graph-execution-foundation` resolves to
`/var/tmp/zlong-graph-execution-foundation/repository`; branch `graph-execution-foundation`.
[execution-flows.md](execution-flows.md) owns the contract;
[ROADMAP F1–F7](ROADMAP.md) is the only backlog. F1–F7 remain incomplete.

## Contract and scope

Candidates independently consume common inputs/initial state/parameters; no CPU
reference events,routes,numerical results or gradients become candidate inputs.
Online greedy accepts legal topology/input,including positive-delay PDG feedback;
it may naturally degenerate to streaming. Preserve int64,stable order,parallel
edge identity,missing/zero messages and None/zero gradients. Performance matrix:
PDG LibTorch;TimedDAG/Settle LibTorch and PyTorch;CPU/NPU × streaming/prefill ×
inference/complete training. Python resident is a C++/CANN client,not an independent
PyTorch device scheduler. Five placement presets retain finer switches. FP32 main;
FP16 separate. CUDA execution is target-machine-pending.
Current alignment outranks run-ml-experiments: minimal source/input/config/environment
identity,raw failures/results,synchronized complete timing,bounded resources/stops.
No duplicate tracking or Trackio blocker. Training means forward/loss interface/
backward/VJP/optimizer/continuation/throughput;downstream convergence is out of scope.
Use affected checks and byte-verified terminal build reuse. Do not rerun the
unchanged portable core's8,954 CPU checks/23 optional skips. Commit implementation,
qualify clean immutable source,commit evidence separately. Subsequent pushes remain
authorized after execution resumes;this pause requires renewed user confirmation.

## Latest qualification

Internal remote Full inference qualified on
**a8fa371e0088b9e3f1b2ca8cb8b540f6211064db**;implementation committed/pushed.
[Report](evidence/device-peer-full-20261001.md) and [audit](evidence/device-peer-full-20261001.json).
All8 fixed-source jobs PASSED/exit0:two runtime builds,two peer gates,single-device
inference regression,half-cache training smoke,Python regressions,separate profile.
Peer HARD84 configurations/420 windows and HST/SOFTP36/180 per dtype:
combined120 configurations/600 windows each for FP32 and FP16.
Single-device4 inference cells,half-cache1 trajectory/16 windows/4 updates,
Python96 passed/no skips. Capacity and remote-adjoint refusals passed.
No runtime development failure or tolerance relaxation in this increment.

Coordinator NPU owns online readiness/selection/state/attention/emission/queues;
peer owns/executes existing packed Full banks and returns selected results through
device-loop packets. Empty/error windows send a terminal command;all programs
submit before boundary waits. Host has no per-stage dispatch. Fixed packet padding
is transferred;actual Full computation is selected. Total packet/arena budget is
charged before allocation. Public single-device defaults are unchanged.
This is one remote Full phase,not general owner/state sharding or peer training;
remote reverse tapes explicitly fail.

Profile2 fixtures/10 windows,20 host model submissions,2 physical devices:
2435 AI_VECTOR_CORE,76 AI_CORE,2 MIX_AIV;no observed AiCPU/logged fallback.
Peer has Full planner/matmul/tanh;coordinator has closure/readiness/selection/queue
and attention. Device76 notify records/76 waits,461 label switches,744 DMA tasks.
These include setup and boundary work;not throughput or overlap measurements.
TASK/launchers/peer_flow_evidence.py FULL_REV passed. Frozen source and standalone
build peer-flow-clean01;Python-owned build peer-flow-python-clean01. Four content
objects rebuilt per runtime;Python control/peer objects rebuilt,standalone reused
qualified control/peer archives. Core/CANN/public-training reuse is byte-verified.

Immediately preceding completed increments:
- Public FP16 full training on0095048:83 trajectories/1328 windows/332 updates,
  FP32 regressions,138 Python tests;[evidence](evidence/resident-fp16-training-20261001.md).
  FP16 payload/parameters and real forward rounding,FP32 roots/masters/slots.
- Reusable peer packets on5c3662b:ready/pull/consumed protocol,int64/bool/FP32/FP16,
  129 same-notify loop reuses;[evidence](evidence/device-peer-control-20261001.md).
  Prior dev failures retained:Bool fixture and cross-thread CANN107002/507046.
  Same-thread construction/submit-all/wait fixes the context failure.

## Remaining main work and next action after confirmation

1. General multi-device owner/parameter/state/cache placement and packed transport,
   exploiting topology locality without fixture-specific routing. The remote Full
   phase above is only the first graph integration;fixed capacity traffic is not
   yet communication minimization or concurrent shard execution.
2. Cross-card VJPs,shared-owner gradient reduction,FP32 masters/optimizer publication,
   continuation/checkpoint and public C++/Python client qualification.
3. Complete experiment consumers and bounded middle-scale five-preset screening,
   then representative/full-size CPU + selected mixed + resident comparisons:
   both schedules,inference/complete training,FP32 and separate FP16.
4. Additional CANN/environment/target-pending CUDA records and final delivery audit.
Historical CPU Attention remains supplementary,not an implementation blocker.
Before performance,resolve its preserved timing lock and host-memory interference
with a deliberate policy;do not blindly resume/kill it.

Suggested overhead reduction after confirmation:group related multi-device work
into vertical graph/training milestones;reuse fixtures/assertions and one audit
path;only affected checks per increment,broader regression at integration gates.
Do not add another tracking framework or repeat already qualified core gates.
Formal full-size repetitions follow functional integration and capacity calibration.
No new milestone has been started after the user's pause request.

## Environment and bounded execution

TASK=/mi/data2T/zlong/tide-execution-flows;RUN=TASK/runs/NAME;
unit=tide-execution-flows-NAME.service;RUN/status.json and task.log own lifecycle.
Module libtorch-npu/2.10.0-cann9.0.0;Python
/opt/miniconda/envs/ascend900-train-full-torch-npu-2.10.0-py311/bin/python.
Public /opt stack supersedes dated personal guide under user authorization.
TASK_QUEUE_ENABLE=0,TORCH_DEVICE_BACKEND_AUTOLOAD=0;preserve module PYTHONPATH,
prepend snapshot/python. SoC Ascend910_9392. Use leased logical device indices.
Last space check:227GiB data,23GiB root;recheck before large writes.
Core builds:placement-cpu-clean01,placement-npu-clean01(standalone),
placement-npu-python-clean01(Python-owned). Never load standalone SDK into Python.
Long jobs use frozen source,background.slice/Nice10,lease120s/run600s/build900s.
Never mutate active snapshots or terminal evidence. Atomic writes:durable_records.py.

## Preserved historical boundaries

The24 earlier dirty files are SHA256-verified on pushed archive/restricted-flow-20260930
at964bf628c67270200dabe55b1bca026bd403cd37;TASK/restricted-flow-archive.json owns hashes.
Historical historical-cpu-attention-01 remains intentionally SIGSTOP;pause.json
outweighs running status. It retains host memory and TASK/timing.lock.
Historical Add CPU78.793172/NPU4 47.932888ms/token is throughput1.6438x faster;
it does not certify resident execution. No complete CPU Attention training ratio.
