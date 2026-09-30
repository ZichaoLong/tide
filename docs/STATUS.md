# Current handoff

Updated 2026-09-30T03:09:06.578642+00:00. **ACTIVE: user confirmed execution of the general-online contract.**
Repo /home/zlong/llm/graph-execution-foundation; real path
/var/tmp/zlong-graph-execution-foundation/repository; branch graph-execution-foundation.
No subagents, no push. Reference repos and ObsidianVault remain read-only.

## Contract and next action

[execution-flows.md](execution-flows.md) is the consolidated user-confirmed contract;
[ROADMAP F1–F7](ROADMAP.md) is the single backlog. General online node-time greedy
prefill must accept every legal family topology/input, including positive-delay
PDG feedback. No numerical route prepass, whole-window potential expansion or
fixture-specific shortcut. Allow natural streaming degeneration. NPU residence
includes online decisions and progression, not only device tensors/static capture.

Packing must cover emission/routing/placement/aggregation/compute/peer copies.
Conservative and aggressive-safe byte-budget chunking remain to implement;
performance prioritizes larger legal batches with persistent allocations/headroom.
Do not split complete fibers/attention normalization or change continuation/VJP/
optimizer boundaries. Profiling throughout; formal timing separately.

Required matrix: PDG LibTorch; TimedDAG/Settle LibTorch and PyTorch; CPU/NPU,
streaming/prefill, inference/complete training. Five presets CPU + Mixed A/B/C +
resident, fine switches retained. Small/medium covers all; full size compares CPU,
screened mixed, resident, both schedules. FP32 primary, FP16 separate; CPU FP64
reference. Three fresh processes for recommendations. CUDA hardware remains pending.
Historical slow CPU Attention must not block general implementation.

Committed host greedy implementation:**2038d88**. NEXT: launch clean fixed-source
CPU build/full regression and matching NPU build/semantic gates. Keep source/evidence
commits separate. Continue byte-budget batching and isolate the proven CANN control
primitive into a project-owned adapter; then online device queues/packing, complete
consumers, multi-device/training and staged performance. Do not call the host
scheduler or scalar device-loop probe complete resident execution.

## New greedy implementation and development checks

Source before this increment:0df0de3 (contract). Optional `schedule=greedy` /
`tide::Greedy` added in Python and C++, using actual pending fibers, multi-source
positive-delay region closure, certified region-time prefixes and existing local
block kernels. Public config/runtime/native and native Settle frontend support it.
Defaults unchanged. Physical parallel edges remain distinct. max_events limits
simultaneously live fibers, not total work; it is not a complete memory budget.
Per-atom host bookkeeping remains. Independent scalar references are retained.

Development results:
- build-greedy-cpu-dev01: full CPU core/bindings/clients build passed; frozen
  greedy-dev01, TASK/builds/greedy-core-cpu-dev01. Dirty snapshot hashes retained.
- greedy-gates-cpu-dev02:611 pytest checks passed in102.53s, frozen greedy-dev02;
  directed greedy/public/carried-training/checkpoint/Settle/frontier/cross-family.
  Prior dev01 failed only because the new rejection test omitted native
  OverflowError;97 preceding tests passed. Corrected test was rerun in dev02.
- Additional cyclic packed same-fiber attention/cache/isolated-VJP tests:16 passed
  (CPU FP64/FP32, Python/native, observe-all/selected/clear/old policies).
- greedy-client-dev03: direct C++ Greedy feedback/parallel edges/full observables/
  VJPs/continuation/capacity and extended Settle literal recurrence passed in
  FP64 and FP32 (four runs). Reused byte-identical core archives are hashed in
  reused-core.json; development only, not final qualification.
- dev02 CTest found no registered tests; no CTest pass claim. Current CMake now
  registers four standalone Greedy/Settle CPU tests, pending clean-build gate.
- Python-only development initially46 + public25 + route/overflow4 passed;
  these overlap later611, do not sum them as independent qualification counts.

Task root TASK=/mi/data2T/zlong/tide-execution-flows. Runs/status/logs under
TASK/runs/NAME, linked from artifacts/execution-flows-NAME. Implementation is committed as2038d88; old consumer edits remain separate.
Prepared immutable source greedy-clean01 at2038d88:
- greedy-clean-cpu01: build greedy-cpu-clean01, CTest four cells, full verify.py.
  Two workers, build1800s/full-test7200s bounds; results pending.
- build-greedy-npu-python01 and build-greedy-npu-sdk01: full matching builds
  into greedy-npu-python01/greedy-npu-sdk01, two workers each,1800s bounds.
  NPU Python build passed. Prepared greedy-npu-semantic01: six FP32 public
  Python/native × family cells (all six now PASSED), independent CPU observables/VJPs/optimizer/
  fresh-process checkpoint/CPU handoff; one card,120s queue/900s workload limit.
  Standalone NPU build also passed; CPU build, four CTests and full8849-test FP64/FP32 qualification PASSED.
  Source2038d88; evidence in docs/evidence/online-greedy-20260930.{md,json}.
  Prepared build-device-control-dev01: frozen control-dev01, standalone core
  greedy-npu-sdk01, new build device-control-dev01; CMake/help/loader gate only.
  Build/help/loader PASSED. device-control-gate-dev01 FAILED at first mark:
  CANN507000, own runtime log requires label-list creation before label mark.
  Component now defers task emission until all target lists exist; timeout
  cleanup retains owners/handles if completion cannot be confirmed.
  Prepared build-device-control-dev02/control-dev02, then gate-dev02:
  one NPU,120s queue/90s workload. dev02 build and all eight control cases PASSED.
  Prepared control-dev03/build-device-control-dev03 to add captured numerical
  submodel calls from the device-controlled loop. dev03 build failed because
  NPUGraph public header additionally requires active CANN include directory.
  CMake now discovers it explicitly. Prepared dev04 build/control-dev04, then
  stage gate. dev04 failed on mixing toolkit/SDK ACL declarations; use SDK
  vendored ACL include consistently as existing GraphReplay does. Prepared
  build-device-control-dev05/control-dev05 PASSED. Device-stage gate FAILED107000:
  CANN forbids RIExecuteAsync on a model-bound stream. Unsupported numerical
  submodel bridge removed from working source; frozen dev05 retains reproducer.
  Added portable fixed-capacity packed queue and int64 closure with sample
  workspace chunking. Prepared dev06 build/CPU gates and NPU queue/control gates.
  dev06 build failed on omitted ATen grad-mode header; corrected. Prepared
  dev07 build and CPU queue FP64/FP32 gates PASSED. Prepared
  device-control-gates-dev07: control eight cases, raw packed numerical
  FP32/FP16, tensor queue FP32; one NPU120s queue/300s workload bound. All runtime processes returned0;
  raw control and numerical FP32/FP16 passed. Queue trace reports int64 sort
  on AiCPU and unsupported scatter_reduce CPU fallback. This is NOT native
  NPU closure evidence. Tensor closure now explicitly refuses NPU; CPU oracle
  stays independent. Added optional Ascend C exact-int64 closure/branch kernel
  (one AIV, metadata only), pending build/device/profiling gates. Prepared
  build-device-control-dev08/control-dev08, explicit Ascend910_9392 target;
  600s build bound and two workers; then bounded one-device feature gates. No host-loop substitute for required resident execution. Ascend C
  queue/control stages are the next integration route to investigate.

## Device-control feasibility

All four small probes are terminal and retained, one NPU each,120s queue/90s
workload bounds. Scripts under TASK/launchers/device-loop-probeNN.py print hashes.
- probe01 failed107005: NPUGraph stream is not an explicitly bound model stream.
- probe02 failed107000: ordinary Torch stream cannot be bound in this mode.
- probe03 raw ACL_STREAM_PERSISTENT + explicit model + ACLNN successfully built,
  then execute failed507000. Own runtime log:headSqArrMax=0. The probe mistakenly
  unbound its only stream before execution.
- probe04 retained binding until completion: **passed**, physical13/logical0,
  CANN9.0.0 + TorchNPU2.10. Device int64 count/limit controlled2,5,1 iterations
  of the same compiled loop. No per-iteration host scalar decisions; all cleanup0.

Uncommitted project component:tools/device_online/{cann_api,cann_program,control_check},
CMake, scripts/build_device_control.py, docs/device-control.md. Fixed scalar buffers,
int64/bool/int32, raw ACLNN, persistent model lifecycle and bounded-loop checks.
New build prepared as above; no device test result yet. It does not provide autograd.

This is a capability result, not a graph scheduler, packed queue, full resident
training or performance result. Next keep buffers/workspaces/model/stream/labels
alive through execution and establish project-owned lifecycle/continuation tests.

## Paused job and evidence

Unit `tide-execution-flows-historical-cpu-attention-01.service` is still suspended.
At this documentation pass, all four cgroup processes were verified in T state;
systemd reports active/running because SIGSTOP does not terminate the service.
It retains host memory and TASK/timing.lock. New greedy jobs are recorded above; inspect their live units independently. The pause sidecar is TASK/runs/historical-cpu-attention-01/pause.json;
its original timestamp is retained. status.json says running because its writer
is suspended. New isolated NPU probes are separate from this suspended experiment. Do not call it passed.

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

## Preserved prior complete-flow work

Prior commits:1700396 scope;d366eb7 hashed reachable topology packets;
6dbece0 reusable peer notifications with acknowledged buffer lifetime.
Uncommitted flow_*.{h,cpp}, tide-complete-flow target, benchmark_execution_flow.py,
verify_execution_flows.py, test_flow_semantics.py; modified bounded schedule/select/
export/update, resident, peer transport and build manifests/CMake. CPU/mixed/finite
captured complete-window consumers exist at development scope. The current topology
reader requires DAG/rank-aligned packets; the consumer does not yet use the new generic PDG greedy scheduler.
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

## Environment and commands

Module libtorch-npu/2.10.0-cann9.0.0; Python
/opt/miniconda/envs/ascend900-train-full-torch-npu-2.10.0-py311/bin/python.
TASK_QUEUE_ENABLE=0; TORCH_DEVICE_BACKEND_AUTOLOAD=0. Preserve module PYTHONPATH
and prepend source/python. CPU gates OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1.
TIDE_BUILD_DIR selects matching binding; old cpu-fp16-a5 lacks Greedy.
Trackio:/home/zlong/venvs/trackio/bin/python; project tide-execution-flows.
Large files on data disk; >300GiB free at the first greedy build check.

Task-private freeze_run.py accepts --name/--snapshot/--commit/--npu/--npu-count/
--max-wait, creates immutable hashed snapshots and detached background.slice jobs.
Use --commit for qualification; do not copy unrelated dirty consumer edits into it.
Example after committing (substitute exact COMMIT):

```text
python TASK/launchers/freeze_run.py --name build-greedy-cpu-clean01 --snapshot greedy-clean01 --commit COMMIT -- '{python}' scripts/build.py --backend cpu --build-dir '{base}/builds/greedy-cpu-clean01' --jobs 2
```

On re-entry run git status --short --branch and python scripts/status.py. Inspect
all active/new terminal task records before further actions; never infer success
from a missing unit. No new full-size benchmark has been launched in this increment.
