# Current handoff

Updated 2026-10-01; public FP16 training and peer packets qualified; graph peer integration active. **ACTIVE: user confirmed the execution contract and resumed implementation.**
Commit/push authorization remains active; no requested pause. No subagents.
Reference repositories and ObsidianVault are read-only. Repository
`/home/zlong/llm/graph-execution-foundation` resolves to
`/var/tmp/zlong-graph-execution-foundation/repository`; branch `graph-execution-foundation`.
[execution-flows.md](execution-flows.md) owns the contract;
[ROADMAP F1–F7](ROADMAP.md) is the only backlog. F1–F7 remain incomplete.

## Contract and scope

Candidates independently consume common inputs/initial state/parameters. No CPU
reference event, route, numerical result or gradient becomes a candidate input.
Online greedy accepts legal topology/input, including positive-delay PDG feedback;
it may naturally degenerate to streaming. Preserve int64, stable order, parallel
edge identity, missing/zero messages and None/zero gradients. Performance matrix:
PDG LibTorch; TimedDAG/Settle LibTorch and PyTorch; CPU/NPU × streaming/prefill ×
inference/complete training. Python resident is a C++/CANN client, not an independent
PyTorch device scheduler. Five placement presets retain finer switches. FP32 main;
FP16 separate. CUDA execution is target-machine-pending.
Current alignment outranks run-ml-experiments: minimal source/input/config/environment
identity, raw failures/results, synchronized complete timing, bounded resources/stops.
No duplicate tracking or Trackio blocker. Training means forward/loss interface/
backward/VJP/optimizer/continuation/throughput; downstream convergence is out of scope.
Use affected-path checks and byte-verified terminal build reuse. Do not rerun the
unchanged portable core's 8,954 CPU checks/23 optional skips. Commit implementation,
qualify clean immutable source, commit evidence separately.

## Latest clean qualification

Public FP16 training qualified on00950483711a835027b9d570b08042f141dd1838,
implementation committed/pushed. [Report](evidence/resident-fp16-training-20261001.md)
and [audit](evidence/resident-fp16-training-20261001.json). All7 fixed-source
jobs PASSED/exit0:two builds,basic/cache,FP32 regressions,Python and placement profile.
Half83 trajectories/1328 windows/332 updates,independent CPU FP32/FP64 half-forward
adjoints and separate FP32 masters. FP32 control98/1568/392,event66 roots/8
trajectories,fiber172 roots/20 trajectories. Python138 passed,no skips.
Half-cache profile1 trajectory/16 windows/4 updates:35415 AI_VECTOR_CORE,
2076 AI_CORE,298 MIX_AIV,no observed AiCPU/logged fallback;not throughput.
TASK/launchers/precision_training_evidence.py FULL_REV passed. Clean source
low-precision-training-clean01;standalone build same name;Python-owned build
low-precision-training-python-clean01. Four public objects rebuilt per runtime;
byte-verified terminal core/CANN/fixture/support reuse. All qualification jobs terminal.
Development failures retained in audit:dev01 missing checker link object;
Python dev02 library path omission;dev03/dev04 unsupported identity fixture.
No tolerance relaxation or failed production-build reuse.
Prior cache reverse f26f3b0 and extended graph db9e985 are separately qualified;
ROADMAP indexes immutable evidence. No mechanical rerun of earlier suites.

## Active work and next action

Peer packet primitive qualified on5c3662bd1162a9155c933f4953f8ff191b6fdc40,
implementation committed/pushed. [Report](evidence/device-peer-control-20261001.md)
and [audit](evidence/device-peer-control-20261001.json). All4 fixed jobs PASSED:
build,2 peer dtype cells,4 single-device control/failure/numerical cells,profile.
Each dtype11 dynamic windows,int64>2^55,capacity/empty continuation,offset views,
129-iteration same-notify reuse. No graph/peer training/performance claim yet.
Profile2 devices:1946 AI_VECTOR_CORE and2 AI_CPU INT64 OnesLike initializers,
both before first device MODEL_EXECUTE. Host22 model submissions(11×2),4 notify
creates/records/waits each;device636 record/636 wait/614 label switches/1535 DMA.
No logged CPU fallback. This is placement only;task totals include setup/boundaries.
TASK/launchers/peer_control_evidence.py FULL_REV passed. Snapshot/build
peer-control-clean01. All current tasks terminal. Retained dev01 CPU Bool fixture
failure and dev02 CANN107002 no-context/507046 timeout;fixed by same-thread
submit-all then wait,explicit cross-thread refusal,unchanged synchronous run API.
CANN CSVs merge devices in one directory;inspect Device_id rather than dirname.

Uncommitted graph peer increment:RemoteFull service and internal ContentFlow
constructor overload place existing PackedFull/PackedLhFull/PackedSwiGluFull
banks on a second NPU. Coordinator owns actual online readiness/selection/queue;
packed selected actions/content/comparison travel through fixed peer packets.
Returned values/error/chunks feed the same device stage. Device end label sends
terminal packet,including empty/error windows. Both programs are submitted before
waiting;no per-stage host values/branches. Total packet/arena workspace charged
before allocation. Full reverse explicitly refused in this inference increment.
Public single-device owner/config defaults are unchanged.

Shared precision_flow_check now has --peer-full,using its existing independent
CPU fixtures/assertions. New peer-flow/peer-control-flow select HARD and HST/SOFTP;
they explicitly require2 NPUs and are excluded from default single-device gate.
Includes attention/normalized Aggregate,all built-in Fulls,feedback/unaligned
arrivals,high int64 clocks,CPU/device input,continuation and schedule switching.
Peer-only output/stage/trace capacity refusals require an actual device refusal,
poison snapshot and orderly two-program close;remote adjoint is explicitly tested.

Development peer-flow-dev01 all5 tasks PASSED/exit0:affected build,
HARD peer2 dtype cells(78 configs/312 windows each),HST/SOFTP peer2 dtype
cells(36/144 each),single-device4 inference regression cells and one half-cache
training smoke(1 trajectory/16 windows/4 updates). Capacity/adjoint refusals
passed. No tolerance changes;no peer runtime failure in this increment.

Test-only follow-up peer-flow-dev02 completed: affected checker build and both
2-NPU gates PASSED/exit0. Each dtype HARD84 configs/420 windows and control36/180;
combined120 configurations/600 windows per dtype. Explicit zero-length windows
before work and mixed Tanh/SwiGLU/LH Full nodes passed. Production byte-identical
to peer-flow-dev01; no new runtime failure or tolerance change.

Next commit/push this implementation,freeze peer-flow-clean01 on that full
revision,and run TASK/launchers/build_peer_flow_v1.py peer-flow-clean01.
Qualify separate peer-flow and peer-control-flow two-device gates,single-device
precision-flow/precision-control-flow and half-cache training smoke,plus a separate
peer-flow placement profile. Also rebuild affected Python-owned objects and run
resident precision/half-training client regressions without mixing runtimes.
Lease120s/run600s/build900s; immutable source/evidence split.

Remaining main work:graph peer integration,parameter/state sharding and training;
five-preset consumer screening;representative/full-size performance;version/
migration/CUDA records. F1–F7 remain incomplete. Historical CPU Attention stays
paused and does not block implementation. No pause requested;continue after
qualification and each authorized push. The graph peer increment above is uncommitted and unqualified.

## Environment and bounded execution

TASK=/mi/data2T/zlong/tide-execution-flows; RUN=TASK/runs/NAME;
unit=tide-execution-flows-NAME.service; RUN/status.json and task.log own lifecycle.
Gate/profile result.json and per-case logs are under their named subdirectories.
Module libtorch-npu/2.10.0-cann9.0.0; Python
/opt/miniconda/envs/ascend900-train-full-torch-npu-2.10.0-py311/bin/python.
Public /opt stack supersedes dated personal guide under user authorization.
TASK_QUEUE_ENABLE=0,TORCH_DEVICE_BACKEND_AUTOLOAD=0; preserve module PYTHONPATH,
prepend snapshot/python. SoC Ascend910_9392. Use leased logical device indices.
Last space check:227GiB data,23GiB root; recheck before large writes.
Core builds:placement-cpu-clean01,placement-npu-clean01 (standalone),
placement-npu-python-clean01 (Python-owned). Never load standalone SDK into Python.
Freeze with `python TASK/launchers/freeze_run.py --name NAME --snapshot SNAPSHOT
[--commit REV] [--npu --npu-count N --max-wait 120] -- timeout --signal=TERM
--kill-after=10s 900s '{python}' ...`. Long jobs use background.slice/Nice10.
Never mutate active snapshots or terminal evidence. Atomic handoff:durable_records.py.

## Preserved historical boundaries

The24 earlier dirty files are SHA256-verified on pushed archive/restricted-flow-20260930
at964bf628c67270200dabe55b1bca026bd403cd37; TASK/restricted-flow-archive.json owns hashes.
Historical historical-cpu-attention-01 remains intentionally SIGSTOP; pause.json
outweighs running status. Do not blindly resume/stop. It retains host memory and
TASK/timing.lock. Resolve interrupted timing before formal throughput.
Historical Add CPU78.793172/NPU4 47.932888ms/token is throughput1.6438x faster;
it does not certify resident execution. No complete CPU Attention training ratio.
