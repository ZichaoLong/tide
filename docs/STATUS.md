# Current handoff

Updated 2026-10-01; public FP16 training qualified, device peer progression active. **ACTIVE: user confirmed the execution contract and resumed implementation.**
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

Device-side peer completion/communication increment is uncommitted.
peer_api/peer_exchange implement fixed packet ready/pull/consumed Notify pairs
captured by CannProgram. Peer/check CMake and explicit --checks peer entry point
added;default single-device gate excludes peer. Full-build manifest also includes
the existing half-training executable. No portable core or CANN kernels changed.
Development builds01/02 passed. Dev01 runtime failed in CPU Bool fixture setup;
fixed initializer and added offset views. Dev02 failed during submission from a
fresh host thread:vendor plog confirms107002 CONTEXT_NULL;other peer wait timed
out507046 and runtime resources were quarantined. Both original failures retained.
Fix:split CannProgram submit/wait,require its constructing thread and submit both
models before any wait. Synchronous run remains submit+wait. Checker verifies
cross-thread refusal,duplicate-submit/empty-wait guards and11 dynamic windows per
dtype including empty/limited continuation,int64>2^55,bool and FP16/FP32 packets.
Development dev03 now passed:both FP32/FP16 two-device cells,11 dynamic
windows each;all4 single-device control/failure/numerical regression cells.
The former thread-context failure is fixed,not suppressed. No live development
jobs. Next commit/push,freeze peer-control-clean01 to the exact revision,build
 --checks peer control failure numerical --jobs2,then run peer(two cards),
regression(one card) and a separate peer FP32 placement profile(two cards).
Build900s,run600s,lease120s,profile256MiB. Source/build/terminal/CSV audit and
separate evidence commit follow. No graph or multi-device training claim yet.
After transport gate,integrate actual graph task loops and independent gradients;
raw packet success alone does not complete F4/F5.

Remaining main work:device peer progression,completion/communication and training;
five-preset consumer screening;representative/full-size performance;version/
migration/CUDA records. F1–F7 remain incomplete. Historical CPU Attention stays
paused and does not block implementation. No pause requested;continue after
qualification and each authorized push.

## Environment and bounded execution

TASK=/mi/data2T/zlong/tide-execution-flows; RUN=TASK/runs/NAME;
unit=tide-execution-flows-NAME.service; RUN/status.json and task.log own lifecycle.
Gate/profile result.json and per-case logs are under their named subdirectories.
Module libtorch-npu/2.10.0-cann9.0.0; Python
/opt/miniconda/envs/ascend900-train-full-torch-npu-2.10.0-py311/bin/python.
Public /opt stack supersedes dated personal guide under user authorization.
TASK_QUEUE_ENABLE=0,TORCH_DEVICE_BACKEND_AUTOLOAD=0; preserve module PYTHONPATH,
prepend snapshot/python. SoC Ascend910_9392. Use leased logical device indices.
Last space check:232GiB data,25GiB root; recheck before large writes.
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
