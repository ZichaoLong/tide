# Current handoff

Updated 2026-09-30T02:38:10.342592+00:00. **ACTIVE: the user confirmed the consolidated contract and requested execution.**
General online implementation/correctness work has resumed. The historical CPU
Attention unit remains suspended deliberately; do not resume it or launch scale
sweeps ahead of the general algorithm gates. No subagents, no push; reference
repos and ObsidianVault remain read-only.

## Authoritative scope and next action

Read [execution-flows.md](execution-flows.md): consolidated Chinese review contract,
2026-09-30. [ROADMAP F1–F7](ROADMAP.md) is the single backlog. Older finite-static
assessment scope does not supersede this revised general-online delivery.

Current agreement: general online greedy node-time prefill for every legal graph/
input in each family, including positive-delay PDG feedback; exact single-action
fallback may degenerate to streaming. Compress time recursion into legal block
computation when contracts allow it; do not assume constant stage count for arbitrary
feedback. No CPU/numerical route discovery pass and no fixture-specific scheduler.
Static graph/index/owner preparation is allowed. NPU resident means online decisions,
routing/readiness/batching and progression on device, with correct continuation.
Existing static expansion/capture remains a scoped optional backend.

Performance scope: PDG LibTorch; TimedDAG/Settle LibTorch and PyTorch; CPU/NPU,
streaming/prefill, inference/training. PDG PyTorch is correctness-only as needed.
Five selectable presets (CPU, Mixed A/B/C, NPU resident) plus fine-grained switches
are defined once in the contract. Correctness/medium screening cover all five;
full-scale compares CPU + screened mixed + resident, both schedules. FP32 primary,
NPU FP16 separate, CPU FP64 primarily the correctness anchor; GPU execution pending.

Packing must cover emission/routing/buffer placement/aggregation/compute/peer copy,
not only attention. Use bulk primitives and justified isolated backend fusion;
no per-message host-fill loop disguised as a tensor pool. Logical stages, API calls
and actual device kernels are distinct. Profiling is an implementation/verification
feedback loop as well as final evidence; instrumented and timing runs stay separate.

Both conservative and aggressive-safe chunking are supported policy targets.
**Performance validation prioritizes aggressive-safe chunking:** use larger legal
batches within a calibrated explicit memory budget, required persistent allocations
and headroom; split before over-budget submissions rather than deliberately provoke
OOM. Exact thresholds/mechanisms remain to implement and validate. Preserve complete
fibers/normalization, state/KV/messages, loss/VJP/optimizer boundaries. Runtime external
resource changes can still fail explicitly; never drop work or silently fall back.

First implementation increment: audit the general algorithm/continuation contract against
the existing implementation and pinned semantic sources; identify reusable code and
actual gaps. Then small/medium correctness + focused profiles, coherent commits,
complete backend/training paths, staged scale tests and bounded formal comparisons.
Do not blindly restart dev06 gates or full-size static capture as the main plan.
Those regressions remain necessary for reused code, but are not the whole task.
Historical slow CPU Attention is supplementary, not a blocker to implementation.

NEXT: implement a separate general greedy scheduler using current pending-message
metadata and conservative positive-delay closure. Compute earliest possible
unfinished region events from actual pending events using a multi-source shortest-
path bound; batch complete ready region-time prefixes, execute once, publish real
messages, and repeat. No whole-window potential-event expansion or numerical route
prepass. Reuse existing local region/state/Full kernels, retain independent scalar
CPU schedules. Gate feedback, arbitrary delays, region-history order, int64 huge
clock gaps, absence, cuts and isolated VJPs before native/device scaling.

## Paused job and evidence

Unit `tide-execution-flows-historical-cpu-attention-01.service` is still suspended.
At this documentation pass, all four cgroup processes were verified in T state;
systemd reports active/running because SIGSTOP does not terminate the service.
It retains host memory and TASK/timing.lock. No other active tide-execution-flows
unit was listed. The pause sidecar is TASK/runs/historical-cpu-attention-01/pause.json;
its original timestamp is retained. status.json says running because its writer
is suspended. No new NPU task was submitted. Do not call this experiment passed.

Historical source951031e, original baseline02 binary; Attention17,269,426,339 params,
D2048/B512/T12/V50304, FP32, node/head160, ATen1, backward16/optimizer16. Intended
three sequential fresh processes, each1 warmup+1 measured AdamW update,21600s/process.
Repeat1 construction163.880407s; no measured update before pause. Original7200s failures
remain failed. Parent launchers/historical_attention_series.py and task-private
historical_attention_benchmark.py retain native/original-wrapper/source identities.

After explicit continuation, review wall-clock timeout/measurement consequences
before any SIGCONT. A fresh run may be required; paused elapsed time is not throughput
evidence. Preserve logs and do not let this unit's timing lock silently block future
formal timing. Stop/restart handling must be deliberate and recorded, not automatic.

## Source and implementation inventory

Repo /home/zlong/llm/graph-execution-foundation; real path
/var/tmp/zlong-graph-execution-foundation/repository. Branch graph-execution-foundation;
HEAD 6dbece052d19ff3db5df4364cb62768d7541863c. No new implementation commit during alignment.
Significant uncommitted code must be preserved. Alignment edits: STATUS, ROADMAP,
execution-flows.md only; no implementation changes or new device probes/tests.

Prior commits:1700396 scope;d366eb7 hashed reachable topology packets;
6dbece0 reusable peer notifications with acknowledged buffer lifetime.
Uncommitted flow_*.{h,cpp}, tide-complete-flow target, benchmark_execution_flow.py,
verify_execution_flows.py, test_flow_semantics.py; modified bounded schedule/select/
export/update, resident, peer transport and build manifests/CMake. CPU/mixed/finite
captured complete-window consumers exist at development scope. The current topology
reader requires DAG/rank-aligned packets; generic PDG online prefill is not delivered.
Current bounded windows reset graph state; required continuous online scheduling/
state carry and equivalent PyTorch large-flow consumers remain incomplete.

Development results, NOT final immutable qualification:
- Topology tests11 and independent Python CPU FP64/FP32 semantic tests56 passed.
- flow-gates-cpu-dev05:24/24 passed, host families, observables, isolated VJPs and
  three SGD/AdamW updates.
- flow-gates-npu-dev05: first18 FP32 cases passed, mixed/eager/captured paths.
  Whole job FAILED at first FP16 Resident Add training case: scaled gradient2
  max_error0.010742. Do not relabel the overall job.
- dev06 CPU/NPU overlay builds passed; no dev06 gates launched. Changes unscale
  gradient comparisons, compare optimizer slots, and mask inactive proposal BEFORE
  Full. flow_poison.cpp regression compiled but untested. No claim these fixes pass.
- Earlier peer CPU8/8 and NPU16/16 gates passed with TASK_QUEUE_ENABLE=0.
  Prior launcher SIGSEGV and all development build/gate failures remain retained.

Development build_flow_overlay.py reuses byte-identical archived inputs and explicit
peer_transport.cpp/bounded_select.cpp overrides with hashes. Development only:
clean immutable full CMake builds/CTests and relevant public/consumer gates required.
Current lack of new general-online acceptance is not repaired by old finite gates.

## Workload identities and historical numbers

New wide packet:480 body nodes,2208 body edges,2 identity boundaries; D2048/B512/T12/
V50304. Add9,468,053,696 and Attention17,521,117,376 parameters. All body nodes are
input/output reachable; actual selection must be counted. Historical17.269B PDG
numbers are a different topology/timing scope. Timed-local cannot claim Settle parity.

Historical mixed FP32 ms/sample-token: Add inference CPU7.387522/NPU2 17.858098;
Attention inference CPU19.531847/NPU4 56.012343; Add complete training
CPU78.793172/NPU4 47.932888 (NPU throughput1.6438x faster). Attention NPU9
128.275286; no measured CPU training ratio. These do not qualify new complete flows.

## Artifacts, environment and reusable development commands

TASK=/mi/data2T/zlong/tide-execution-flows, with sources/builds/launchers/runs/reports/
inputs/trackio; linked under repository artifacts/execution-flows-NAME.
Latest source flow-dev06; builds flow-cpu-dev06/flow-npu-dev06;
runs/build-flow-{cpu,npu}-dev06/status.json and task.log. Keep all retained failures.
Historical source TASK/sources/historical951, run TASK/runs/historical-cpu-attention-01.
Task-private freeze_run.py creates hashed dirty frozen snapshots and launches detached
background.slice/Nice10 jobs through scripts/job.py; NPU work uses cooperative queue.

Module libtorch-npu/2.10.0-cann9.0.0; explicit Python
/opt/miniconda/envs/ascend900-train-full-torch-npu-2.10.0-py311/bin/python.
TASK_QUEUE_ENABLE=0, TORCH_DEVICE_BACKEND_AUTOLOAD=0. Preserve module PYTHONPATH and
prepend source/python; small CPU gates OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1.
Core builds /mi/data2T/zlong/tide-npu-performance/builds/cpu-fp16-a5 and core-fp16-a5;
TIDE_BUILD_DIR selects CPU binding build for Python tests. Ambient python lacks full Torch.
Trackio writer/viewer /home/zlong/venvs/trackio/bin/python; project tide-execution-flows,
root TASK/trackio. Large writes belong on the data disk; no deletion authorized by pause.

On re-entry, first run `git status --short --branch`, `python scripts/status.py`
and read this contract/backlog. Future dev06 commands are retained for reused-code
regressions only, not the main general-online task:

```text
python TASK/launchers/freeze_run.py --name flow-gates-cpu-dev06 --snapshot flow-dev06 -- '{python}' scripts/verify_execution_flows.py --build-dir '{base}/builds/flow-cpu-dev06' --output-dir '{out}/gates' --device cpu --development
python TASK/launchers/freeze_run.py --name flow-gates-npu-dev06 --snapshot flow-dev06 --npu --npu-count 2 -- '{python}' scripts/verify_execution_flows.py --build-dir '{base}/builds/flow-npu-dev06' --output-dir '{out}/gates' --device npu --devices 2 --development
```

Resolve TASK and verify launcher arguments/current state before use. Newly changed
source needs a new frozen snapshot/build, not mutation of active inputs.
