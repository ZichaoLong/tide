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

## Next implementation

Inspect actual retained/reverse allocations before another large run.
`full_shard_tape.cpp` still clones immutable Full banks for every retained window.
Investigate update-scoped immutable copies analogous to RetainedProjection and
RetainedAttention, preserving default standalone snapshots, byte admission,
guards and lifecycle. Analysis only at this checkpoint; not implemented yet.

Cross-window gradient reuse is separate: preserve reverse-window→registry-alias
addition order and device completion. Remote Full initializes totals before its
first request; simply aliasing these buffers is unsafe. Do not borrow live forward
banks without a proven contract.

Original Attention B512/physicalB1 minimum estimates still refuse: ~68.031GiB
on12cards/~57.097GiB on16cards versus53.875GiB usable. No new original-width pilot
or automatic B512 retry is queued; do not relax bounds.

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

No current job running/queued. **Preserve deliberately SIGSTOPped
historical-cpu-attention-01**: never resume, stop or clean it. Its old record says
running and it holds old timing.lock. Historical1.6438× meant faster throughput
in the restricted flow, not current online evidence. Restricted archive:
archive/restricted-flow-20260930 at964bf628c67270200dabe55b1bca026bd403cd37.
Old build-reverse-gather-python-dev01 metadata inconsistency remains visible;
not a current failure. Raw records: `TASK/runs/training-storage-{cpu,npu,calibration}-clean01`
and `TASK/runs/build-training-storage-consumer-clean01`; source/build/helper/result
hashes and allocation observations are in the reviewed evidence JSON.
