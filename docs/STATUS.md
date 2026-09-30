# Current handoff

Updated 2026-09-30T00:03:34.929056+00:00. Active user-authorized delivery: complete all retained gaps in
independent CPU/mixed/NPU flows and three-family active large-topology comparison.
ROADMAP F1-F7 is the single backlog. No sub-agents, no push, reference repos and
ObsidianVault read-only. No task-owned job is currently running.

## Completed increments and current edits

Starting evidence8a4940b is retained.1700396 records the expanded F1-F7 scope.
d366eb7 adds dependency-free hashed active topology packets and execution-flow
contract.11 topology tests passed using explicit Torch2.10 Python. All480 wide
body nodes are input/output reachable; timed-local rejects Settle equivalence.
Historical foundation-v1/v2 capacity presets remain unchanged.

Peer notification/buffer improvement committed with this handoff: two reusable
Notify objects per directed pair, source-consumed acknowledgement before allocator
reuse; no unconditional per-transfer clone retention. Optimizer reset checkpoint
is CPU, FP32 payload aliases master storage, warmup discarded before capture.
Development frozen snapshot peer-dev01 (base1700396 plus hashed dirty patch) built
CPU/NPU successfully; four CTest tests passed in each. Analytic two-chip FP32/FP16
capture/replay passed3 changed-input trials each,64 buffer-reuse round trips,
exact int64>2^55/bool/strided/offset values and None/zero VJPs. CPU8/8 and NPU16/16
full bounded observable/VJP/three-update SGD/AdamW gates passed. These are development
gates; immutable final complete-flow qualification is still required.

Current uncommitted separate work: generalized bounded Schedule/identity-boundary
handling and output summation; Resident permits identity boundary nodes;
flow_fixture.{h,cpp},flow_execution.{h,cpp},flow_config.{h,cpp},flow_check.cpp
under tools/accelerator_scale. Not yet compiled: no CMake targets or benchmark/main
entry added yet. Identity StateKernel preserves old zero state (not content); Full
emits content and identity Read is disconnected zero. Period now belongs to the
finite plan rather than first node state_clock. New body model uses historical
Add/Attention local kernels with rank/locality topology and actual edge projections.

## Exact next actions

1. Finish flow_benchmark.cpp and flow_main.cpp, add CMake target/build manifest.
   Complete-flow CPU/mixed/device frontend must include all request work in timing,
   retain setup/capture/cold costs, and never use CPU reference outputs as inputs.
2. Freeze a new development snapshot, build CPU/NPU, run new small common packet
   checks for streaming/frontier/Settle/Resident/bounded eager and actual replay.
   Diagnose before scale. Preserve exact discrete, None/zero and default FP32 gates.
3. Complete independent Python mapping/semantic tests, full-observable/VJP/update
   gates and capacity reductions before full-size experiments. F1-F7 remain active.

## Task artifacts, commands and environment

TASK=/mi/data2T/zlong/tide-execution-flows with sources/builds/launchers/runs/reports/
inputs/trackio. Large writes stay here (entry free352GiB versus29GiB on root).
Task-private launchers/freeze_run.py creates a stable hashed development snapshot,
launches scripts/job.py as background.slice service and uses the account NPU queue.
It now explicitly exports TASK_QUEUE_ENABLE=0, required by this standalone SDK.
No hardware selected/reserved currently. Never stop unrelated processes.

Module libtorch-npu/2.10.0-cann9.0.0; Python
/opt/miniconda/envs/ascend900-train-full-torch-npu-2.10.0-py311/bin/python.
Native core sources unchanged; reusable core builds are under
/mi/data2T/zlong/tide-npu-performance/builds/cpu-fp16-a5 and core-fp16-a5.
Client build command: python scripts/build_accelerator_scale.py --core-build CORE
--build-dir NEW --jobs2. Use clean frozen source for qualification; development
snapshots carry explicit dirty state and file hashes. Test CLI uses PYTHONPATH=python
TORCH_DEVICE_BACKEND_AUTOLOAD=0 and explicit Python; ambient python lacks full Torch.

Terminal records (artifacts/execution-flows-NAME links):
- build-peer-cpu-dev01 and build-peer-npu-dev01 passed;52 build steps,4 CTests each.
- peer-probe-dev01 passed: operator doctor plus analytic peer gate, physical9,13.
- peer-gates-cpu-dev01 passed8 cases.
- peer-gates-npu-dev01 failed SIGSEGV at first eager training; direct launcher had
  omitted TASK_QUEUE_ENABLE=0. debug-peer-train-dev01 gdb confirms ACL runtime worker
  crash; diagnostic wrapper exit0 is NOT a semantic test pass.
- peer-gates-npu-dev02 passed16 cases on physical1,13 with same binary/source and
  corrected launch environment.208s. All earlier failures retained unchanged.
Each runs/NAME/status.json and task.log is authoritative; gates have gates.json and
per-case logs. NPU runs also retain queue.json with physical/logical mapping.

Old /mi/data2T/zlong/tide-device-scheduler assessment remains sealed and cited by
accelerator-cpu-npu-comparison/bounded-scheduler-{qualification,capacity}-20260929.
Full-size device replay, CPU Attention training ratio, new-family full-size flows
and final portable recommendations are NOT delivered yet. CUDA device execution
remains target-machine work. General unbounded device queue is outside the finite
workload contract. Never treat a failed finite assessment as completing F1-F7.
