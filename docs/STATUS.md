# Current handoff

Updated 2026-10-03. **PAUSED after this implementation/qualification/evidence
increment, per the user's requested commit checkpoint. Await confirmation before
new work. Overall goal incomplete. No subagents.**
Repository `/home/zlong/llm/graph-execution-foundation` resolves to
`/var/tmp/zlong-graph-execution-foundation/repository`; branch
`graph-execution-foundation`. Re-entry: `git status --short --branch`;
`python scripts/status.py`. Latest implementation **26176de** is pushed and
qualified; this evidence commit records its reviewed clean results.
[execution-flows](execution-flows.md) owns the contract;
[ROADMAP F1–F7](ROADMAP.md) is the sole backlog. Reference repositories and
ObsidianVault are read-only.

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

## Latest completed increment

**26176de888013fda5eccfe039fa504e87c2e7e95** adds optional `--phase-timing` across
Python, native clients and standalone consumers. Default off adds no middle
synchronization. Enabled training synchronizes all resolved devices after sample
work and before final finite checks/update/publication. Sample and optimizer
phases sum to complete elapsed time for measured and warmup steps. Inference
adds no extra boundary. No numerical/update-boundary/core/resident-ABI/kernel change.
[Evidence](evidence/consumer-phase-timing-20261003.md).

Four clean jobs all passed: `build-phase-timing-{cpu,consumer}-clean01` and
`phase-timing-{cpu,npu}-clean01`; CPU13/NPU8,40 actual default/on CLI candidates,
independent CPU full observables/gradients/updates/continuation, sample slicing,
FP32/FP64 CPU,FP32/FP16 single/two-card NPU. No skips. All units terminal exit0,
empty control groups; physical3,9 lease released. Existing core/resident binaries
are byte-verified; consumer objects reused only after source/header/options
verification, freshly linked. No new profiler/throughput/full-size claim.
Audit: `python TASK/launchers/phase_timing_evidence.py 26176de888013fda5eccfe039fa504e87c2e7e95`.
Raw source `TASK/sources/phase-timing-clean01`; consumers
`TASK/builds/phase-timing-{cpu,consumer}-clean01/consumer/tidegraph-online-bench`.

Development builds dev01 and corrected tests dev02 passed. Original
`phase-timing-cpu-dev01` remains failed (exit2, collection-only test dtype parameter
collision). Renamed only that NPU test argument; retained immutable reproducer.
No uncommitted implementation remains. This evidence commit contains docs only.

Since the prior alignment at7a7293a, qualified memory changes also include shared
training lifetime accounting4dd8368, immutable Full snapshotsf360489, ordered
window reduction/projection-adjoint reuse233bf01, Attention adjoint reuse6b9224c,
physical-gradient lifetime accounting789e1a5, and private frozen projection banks
4467493. Their separate immutable evidence stays linked from ROADMAP.

4467493 qualified native160trajectories/2560windows/640updates,Python16,consumer32,
same-lease D512 memory and separate FP16 profile; eight terminal jobs, no skips.
Only aggressive sharded training borrows private frozen forward projection banks;
publication is prohibited with live windows and reverse programs close before step.
Default/conservative/legacy snapshots, Attention/Full snapshots, state/KV/message
bridges, API budgets and consumer estimates retain their scopes.
D512/B8/T4/V257 Attention,physicalB2×4,two windows,FP32 AdamW:
[7804784128,6957622784] → [7508509696,6661348352], -296274432bytes/card.
Loss7.532631874084473/statistics/continuation unchanged. Separate FP16 profile53176
ops, zero observed AiCPU; not a universal performance claim.
[Evidence](evidence/resident-projection-borrow-20261003.md), evidence commitf30e83f.

## Remaining scale and next bounded action after confirmation

Latest original-width Add pilot on clean4467493 is passed:9cards,B4/physicalB2,
9,468,053,696 parameters,two connected windows,FP32 SGD,one cold complete update.
Construction72.962783193s,update20.769816367s,peak43197837312bytes,
loss31.58603858947754,outputs96/events9265/cut408. All memory/context checks pass.
Old projection20.769816367×128×1.15=3057.316969s>3000s; **B512 not started**.
[Evidence](evidence/original-width-add-projection-borrow-20261003.md),commit811196f.
Earlier21.179427721s/3117.611761s and B1 27.300s/4018.543s refusals remain scoped;
ten-card queue failure remains retained. No pilot has become a B512 success.

After confirmation, use the qualified phase timer for ONE original-width cost
diagnostic. Reuse original B4/physicalB2 capacities and parameter/input rules;
9cards,queue120s,pilot timeout900s,formal measurement lock. Prepare a new task-local
launcher from wide_add_projection_borrow.py, using phase-timing-clean01 and its
consumer. Record synchronized sample/optimizer costs and the old total-based
projection. Any revised `(sample*128+optimizer)*1.15` estimate must retain3000s cap,
all capacity/memory checks and explicit scaling assumptions. Do not silently
modify old runs or declare B512 passed. Only after measured admission may a
separately initialized bounded B512 run be considered. No cost diagnostic or
scale task is currently active/queued.

All ten required representative submatrices are complete. OriginalB512 TimedDAG/
LibTorch/resident/prefill FP32 inference passed:Attention17.521B325.278s,
Add9.468B278.574s. OriginalB512 complete training and formal comparisons remain open.
Attention static diagnostic B1/12cards/minimum operator rows still gives coordinator
62.623GiB>53.875GiB usable. Projection borrowing alone is insufficient; no full-size
Attention training run is queued. Eager mixed multi-card is a separate core gap:
current runtime/validation require one payload device; CLI-only changes cannot
implement parameter/state/message placement, transfers, aliases and continuation.

Planning estimate remains **80% (75–85%)** of local F1–F7 delivery:
functionality85–95%,correctness85–95%,performance/profiling45–55%,
packaging/provenance/migration75–85%. Scope estimates, not pass rates or elapsed-time
predictions. Main remaining: originalB512 complete training (especially Attention),
eager mixed multi-card, finite full-size CPU/mixed/resident comparisons with three-
process recommendations, F7 audit. CUDA/new stack tuples need target-machine gates.

## Environment and protected state

`TASK=/mi/data2T/zlong/tide-execution-flows`; public module
`libtorch-npu/2.10.0-cann9.0.0`; Python
`/opt/miniconda/envs/ascend900-train-full-torch-npu-2.10.0-py311/bin/python`.
User-authorized /opt stack supersedes old private guide.
`TASK_QUEUE_ENABLE=0 TORCH_DEVICE_BACKEND_AUTOLOAD=0`; preserve module PYTHONPATH,
prepend frozen source/python. Lease/remap devices; runtime `env -C {out}`.
Long jobs:frozen source,background.slice,Nice10,two build workers,queue120s.
Last free disk:data173GiB/root13GiB; recheck before large writes.
Core `placement-{cpu,npu,npu-python}-clean01`; qualified resident backend/bindings
`projection-borrow-{standalone,python}-clean01`. Preserve cited prior consumers.

No active or queued task job. Units `tide-execution-flows-NAME.service`, logs/status
`TASK/runs/NAME`; inspect `systemctl --user show UNIT -p ActiveState -p Result -p ExecMainStatus`.
**Preserve deliberately SIGSTOPped historical-cpu-attention-01**: never resume,
stop or clean it. Its old record says running and it holds old timing.lock.
Historical1.6438× meant faster throughput in a restricted flow, not current online
evidence. Archive:archive/restricted-flow-20260930 at964bf628c67270200dabe55b1bca026bd403cd37.
Old build-reverse-gather-python-dev01 metadata inconsistency remains visible;
not a current failure. Audit helpers pin source/build/helper/result/profile hashes.
