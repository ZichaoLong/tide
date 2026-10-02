# Current handoff

Updated 2026-10-02. **ACTIVE: user resumed after 7a7293a. Continue implementation,
qualification, commits and pushes. Overall goal incomplete. No subagents.**
Repository `/home/zlong/llm/graph-execution-foundation` resolves to
`/var/tmp/zlong-graph-execution-foundation/repository`; branch
`graph-execution-foundation`. Re-entry: `git status --short --branch`;
`python scripts/status.py`. Latest implementation **4dd8368**; this evidence
checkpoint follows it (use `git log -1`). [execution-flows](execution-flows.md)
owns the contract; [ROADMAP F1–F7](ROADMAP.md) is the sole backlog.
Reference repositories and ObsidianVault are read-only.

## Contract and operating bounds

Candidates independently consume common inputs, parameters and initial state.
CPU events/routes/gradients never supply candidate execution. General online
greedy accepts legal family topology/input, including positive-delay PDG feedback.
Preserve int64, stable order, duplicate edges, missing/zero messages, None/zero
gradients and complete continuation. Performance: PDG LibTorch;
TimedDAG/Settle LibTorch+PyTorch; CPU/NPU × streaming/prefill × inference/complete
training. Five presets plus fine switches. FP32 main, FP16 separate; training
means forward/loss/backward/update/continuation without a convergence requirement.

Implementation commit → clean fixed-source affected qualification → evidence
commit; push each. User contract outranks run-ml-experiments; minimal useful
records only. No unchanged CPU/representative reruns, unbounded queues, blind
retries, OOM searches or relaxed safety/cost gates. Formal heavy timing is serial;
never stop other workloads to free resources.

## Latest completed qualification

All four `training-storage-*-clean01` jobs passed on clean
**4dd8368159ff74e8b45c052cfe85ecbca1e294f2**: CPU17, installed consumer build,
NPU25 (24 independent-CPU comparisons plus one preallocation refusal), and
D512/B8 Attention allocation calibration. No skips. All units inactive/dead,
exit0, empty control groups; leases released.
[Reviewed evidence](evidence/consumer-training-storage-20261002.md).

Only static estimates changed: shared Attention snapshots charged once, private
accumulation charged one extra live bank. Two-input API capacity, device margins,
runtime resident/core/CANN bytes unchanged. Estimates decreased from
[25965880344,18093475736] to [25138936228,17266531620] bytes; actual peaks remain
[8368268800,7520954880]. Loss7.532631874084473, outputs/statistics/continuation and
chunks exactly match prior records. No actual-memory or speed gain is claimed.
No redundant profile was run. Audit:
`python TASK/launchers/training_storage_evidence.py 4dd8368159ff74e8b45c052cfe85ecbca1e294f2`.

## Current implementation and immediate next action

Full snapshot sharing now copies immutable tanh/LH/SwiGLU banks and static
kind/mapping tables once per backward group, for single-device and sharded owners.
Each window retains its own dynamic events/values/counts. Source identity/version/
shape/stride/device/dtype guards and no-publication lifetime apply; backward,
detach and close clear the cache. `retained_full_bytes` records the shared copy.
Standalone retention overloads keep independent snapshots. Consumer estimates
are unchanged and conservative. No full-size fit or speed claim follows.

Development: Python build and16 retention/budget tests passed on frozen
`full-snapshot-dev01`. Standalone dev01 failed only in the new checker at ambiguous
Tensor assignment; preserved. Corrected frozen `full-snapshot-dev02` passed
standalone/installed consumer builds, all6 native cells (144trajectories,
2432windows,560updates; FP32/FP16,CPU FP32/FP64 references), and8 actual sliced
consumer tests without skips. Python dev01 production inputs match dev02 exactly;
only the standalone checker header differs. No development job remains running.

Next: commit this implementation and push; freeze clean `full-snapshot-clean01`.
Qualify affected standalone/Python libraries and installed consumer; run native6,
Python16,consumer32, a same-lease D512 allocator comparison and separate FP16 trace.
Names: `build-full-snapshot-{standalone,python,consumer}-clean01` and
`full-snapshot-{native,python,consumer,memory,profile}-clean01`. Reuse verified
objects with fresh links; build2,queue120s,child480–900s. Logs/status under
`TASK/runs/<name>`. Do not start a new original-width pilot from this small change.

Both wide model fixtures use LH Full, whose parameter storage is small; this
is a general retention improvement, not the main scale-memory solution. Next
investigate the dominant physical projection/attention reverse gradients and
scratch lifetimes. Preserve reverse-window→registry-alias addition order and
explicit device completion; remote Full zeros totals before its first request.
Simply aliasing cross-window buffers or borrowing live banks is unsafe.
Original Attention B512/physicalB1 estimates still refuse; no cost/safety gate
is relaxed and no automatic B512 retry is queued.

## Scale evidence and progress boundary

All ten representative family/client/schedule submatrices complete. OriginalB512
TimedDAG/LibTorch/resident/prefill FP32 inference passed: Attention17.521B325.278s,
Add9.468B278.574s. This is not the full matrix.
Latest original-width AddB4/physicalB1×4 nine-card pilot (cleane82f971):
construction65.793s, complete update27.300s, peak41.097GiB, loss31.586036682128906;
outputs96/cut408/events9265 match priorB4. B512 projection4018.543s exceeds3000s,
so B512 was not started. Preceding ten-card queue timeout remains failed.
[Report](evidence/original-width-add-chunk-selection-20261002.md).

Planning estimate remains about **80% (75–85%)** of local F1–F7 delivery:
functionality85–95%, correctness85–95%, performance/profiling45–55%,
packaging/provenance/migration75–85%. Scope estimates, not pass rates or time
predictions. Accounting alone does not justify a higher total. Main remaining:
originalB512 complete training (especially Attention), eager mixed multi-card,
full-size finite CPU/mixed/resident comparisons with three-process recommendations,
F7 final audit. CUDA/new stack tuples need target-machine gates. Backlog: ROADMAP.

## Environment and protected state

`TASK=/mi/data2T/zlong/tide-execution-flows`; public module
`libtorch-npu/2.10.0-cann9.0.0`; Python
`/opt/miniconda/envs/ascend900-train-full-torch-npu-2.10.0-py311/bin/python`.
User-authorized /opt stack supersedes old private guide.
`TASK_QUEUE_ENABLE=0 TORCH_DEVICE_BACKEND_AUTOLOAD=0`; preserve module PYTHONPATH,
prepend frozen source/python. Lease/remap devices; runtime `env -C {out}`.
Long jobs: frozen source,background.slice,Nice10,two build workers,queue120s.
Formal timing lock `TASK/online-measurement.lock`.
Last free disk:data151GiB/root7.7GiB; check before large writes.

Source `TASK/sources/training-storage-clean01`; consumer
`TASK/builds/training-storage-consumer-clean01`. Resident binaries
`private-accumulation-{standalone,python}-clean01`; core
`placement-{cpu,npu,npu-python}-clean01`; CPUconsumer`source-values-cpu-clean01`.
Consumer object reuse source/header/options-verified with fresh link;
unchanged dependencies hash-verified. No full rebuild claim.

No current job is running/queued before clean qualification. **Preserve deliberately SIGSTOPped
historical-cpu-attention-01**: never resume, stop or clean it. Its old record says
running and it holds old timing.lock. Historical1.6438× meant faster throughput
in the restricted flow, not current online evidence. Restricted archive:
archive/restricted-flow-20260930 at964bf628c67270200dabe55b1bca026bd403cd37.
Old build-reverse-gather-python-dev01 metadata inconsistency remains visible;
not a current failure. Raw records: `TASK/runs/training-storage-{cpu,npu,calibration}-clean01`
and `TASK/runs/build-training-storage-consumer-clean01`; source/build/helper/result
hashes and allocation observations are in the reviewed evidence JSON.
