# Current handoff

Updated 2026-10-03. **ACTIVE under renewed continuous authorization (2026-10-03).** The user
explicitly revoked earlier per-increment pauses: continue through F1–F7 acceptance,
including tested commits/pushes, without asking to resume again. Overall goal
incomplete. No subagents; protected historical CPU task remains stopped.
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

## Remaining scale and active next action

Latest original-width Add pilot on clean4467493 is passed:9cards,B4/physicalB2,
9,468,053,696 parameters,two connected windows,FP32 SGD,one cold complete update.
Construction72.962783193s,update20.769816367s,peak43197837312bytes,
loss31.58603858947754,outputs96/events9265/cut408. All memory/context checks pass.
Old projection20.769816367×128×1.15=3057.316969s>3000s; **B512 not started**.
[Evidence](evidence/original-width-add-projection-borrow-20261003.md),commit811196f.
Earlier21.179427721s/3117.611761s and B1 27.300s/4018.543s refusals remain scoped;
ten-card queue failure remains retained. No pilot has become a B512 success.

Completed `wide-add-phase-diagnostic01` on clean26176de: terminal exit0,
queue completed/lease released. Sample19.105143379s + optimizer1.556835798s =
20.661979177s; unchanged old projection3041.4433348544s refuses, measured phase
projection2814.0674665565s passes3000s with1.15 margin and all memory/context gates.
[Evidence](evidence/original-width-add-phase-diagnostic-20261003.md).
B512 has not yet executed. Prepared next task `wide-add-b512-phase-admitted01`:
`TASK/launchers/wide-add-b512-phase-admitted01.sh` runs frozen26176de source and
`TASK/builds/phase-timing-consumer-clean01`. Exact launcher command:
`python TASK/launchers/wide_add_b512_phase_admitted.py --source TASK/sources/phase-timing-clean01 --build TASK/builds/phase-timing-consumer-clean01 --output TASK/runs/wide-add-b512-phase-admitted01/assessment`.
Service `tide-execution-flows-wide-add-b512-phase-admitted01.service`, queue120s,
child3180s, outer3300s; original B512, physicalB2×256, 9cards, two connected
windows, one complete independently initialized FP32 SGD update. Retain phase
timing for forecast validation, all current capacities/3000s actual-step gate;
no warmup/profile/formal throughput claim. Status/log under matching TASK/runs.
Next: launch after this evidence checkpoint, inspect complete work/loss/cut/memory
and actual costs. Continue Attention memory and eager mixed multi-card work;
this diagnostic and the next scale run are not overall acceptance.

All ten required representative submatrices are complete. OriginalB512 TimedDAG/
LibTorch/resident/prefill FP32 inference passed:Attention17.521B325.278s,
Add9.468B278.574s. OriginalB512 complete training and formal comparisons remain open.
Attention static diagnostic B1/12cards/minimum operator rows still gives coordinator
62.623GiB>53.875GiB usable. Projection borrowing alone is insufficient; no full-size
Attention training run is queued. Eager mixed multi-card is a separate core gap:
current runtime/validation require one payload device; CLI-only changes cannot
implement parameter/state/message placement, transfers, aliases and continuation.

Acceptance follows concrete F1–F7 gates, not a percentage. Remaining: original
B512 Add/Attention complete training, eager mixed multi-card, finite full-size
CPU/mixed/resident comparisons and separate profiling, then F7 audit. CUDA/new
stack tuples require explicit target-machine gates.

## Environment and protected state

`TASK=/mi/data2T/zlong/tide-execution-flows`; public module
`libtorch-npu/2.10.0-cann9.0.0`; Python
`/opt/miniconda/envs/ascend900-train-full-torch-npu-2.10.0-py311/bin/python`.
User-authorized /opt stack supersedes old private guide.
`TASK_QUEUE_ENABLE=0 TORCH_DEVICE_BACKEND_AUTOLOAD=0`; preserve module PYTHONPATH,
prepend frozen source/python. Lease/remap devices; runtime `env -C {out}`.
Long jobs:frozen source,background.slice,Nice10,two build workers,queue120s.
Last free disk:data172GiB/root13GiB; recheck before large writes.
Core `placement-{cpu,npu,npu-python}-clean01`; qualified resident backend/bindings
`projection-borrow-{standalone,python}-clean01`. Preserve cited prior consumers.

Current prepared/live task is recorded above. Units `tide-execution-flows-NAME.service`, logs/status
`TASK/runs/NAME`; inspect `systemctl --user show UNIT -p ActiveState -p Result -p ExecMainStatus`.
**Preserve deliberately SIGSTOPped historical-cpu-attention-01**: never resume,
stop or clean it. Its old record says running and it holds old timing.lock.
Historical1.6438× meant faster throughput in a restricted flow, not current online
evidence. Archive:archive/restricted-flow-20260930 at964bf628c67270200dabe55b1bca026bd403cd37.
Old build-reverse-gather-python-dev01 metadata inconsistency remains visible;
not a current failure. Audit helpers pin source/build/helper/result/profile hashes.
