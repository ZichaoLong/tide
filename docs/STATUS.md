# Current handoff

Updated 2026-10-03. **ACTIVE under renewed continuous authorization (2026-10-03).** The user
explicitly revoked earlier per-increment pauses: continue through F1–F7 acceptance,
including tested commits/pushes, without asking to resume again. Overall goal
incomplete. No subagents; protected historical CPU task remains stopped.
Repository `/home/zlong/llm/graph-execution-foundation` resolves to
`/var/tmp/zlong-graph-execution-foundation/repository`; branch
`graph-execution-foundation`. Re-entry: `git status --short --branch`;
`python scripts/status.py`. Latest implementation **b5e6345** is pushed; clean affected qualification is
running. Prior implementation26176de remains qualified.
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

## Current implementation and next qualification

Private Attention-bank reuse is IMPLEMENTED and DEVELOPMENT-VERIFIED; immutable
qualification pending. Only aggressive private sharded owners borrow event
matrices and complete ordered fiber banks; subset/mixed-head gathers,
default/conservative/legacy retention and dynamic KV/state/message records stay
owned. API budgets and consumer admission are unchanged. Contract:
[resident-retained](resident-retained.md). Internal owner layout changed; all
eight owner users and three retention archive members were rebuilt. Public
ABI/core/CANN kernels unchanged.

Frozen attention-borrow-dev02: standalone/Python/consumer builds passed;
native six FP32/FP16 cells160trajectories/2560windows,Python16,actualconsumer32,
no skips. All six durable jobs terminal exit0 and empty control groups; leased
physical8,12 released. Original attention-borrow-native-dev01 remains FAILED
(exit1): its new ownership check detected tape/forward-bank field-order mismatch;
explicit mapping fixed in dev02, frozen failure retained.

Implementation b5e634588405c47320ff03692bbf6cdb5af1d356 is committed/pushed.
Clean snapshot attention-borrow-clean01 is frozen at it. Standalone/Python clean
builds passed via byte-verified source/header/options-compatible object reuse
and fresh linking; core/CANN untouched. Exact standalone command:
`python TASK/launchers/freeze_run.py --name build-attention-borrow-standalone-clean01 --snapshot attention-borrow-clean01 --commit b5e634588405c47320ff03692bbf6cdb5af1d356 -- timeout 900 {python} {base}/launchers/build_attention_borrow.py attention-borrow-standalone-clean01 --reuse-host attention-borrow-standalone-dev02`.
RUNNING attention-borrow-native-clean01 (same six affected cells, two devices,
queue120s/timeout900s); build-attention-borrow-consumer-clean01 uses
build_capacity_client.py and verified dev02 consumer object reuse. Next Python16
and actual consumer32 on that exact source. No unrelated CPU matrix rerun.
Jobs/units follow tide-execution-flows-NAME.service and TASK/runs/NAME.
Separate same-lease D512 memory calibration via attention_borrow_memory.py and
FP16 profile via profile_retained_journals.py wait until the B512 cost run ends.
Audit via attention_borrow_evidence.py REV, then a separate evidence commit.
Continue scale/mixed/performance work after qualification; no renewed permission.

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
Those phase-timing implementation/evidence commits are complete; the current
Attention working-tree increment is listed above.

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
PASSED and AUDITED `wide-add-b512-phase-admitted01` on clean26176de, terminal
exit0 at2026-10-03T00:59:36Z; unit inactive/empty control group, nine-card lease
released. Original AddD2048/B512/T12/V50304,9.468B parameters, physicalB2×256,
two connected windows, one complete independently initialized FP32 SGD update.
Construction69.389676023s; sample2168.898689171s + optimizer1.651173068s =
2170.549862239s <=3000. Outputs12288,events1183429,loss30.50836181640625,cut408;
max allocator growth43432802304bytes; all memory/context checks passed.
[Evidence](evidence/original-b512-add-training-20261003.md).
Raw source/consumer remain phase-timing-clean01 / phase-timing-consumer-clean01;
run records TASK/runs/wide-add-b512-phase-admitted01/assessment. Audit command:
`python TASK/launchers/wide_add_b512_evidence.py 26176de888013fda5eccfe039fa504e87c2e7e95`.
This is a cold phase-instrumented feasibility result, not formal throughput:
limited concurrent builds and two-device correctness work on disjoint cards;
no allocator comparison/profile overlapped. Existing historical refusals remain.
Attention memory calibration/profile can now run after the remaining clean
correctness gates. This Add case does not complete the overall matrix.

All ten required representative submatrices are complete. OriginalB512 TimedDAG/
LibTorch/resident/prefill FP32 inference passed:Attention17.521B325.278s,
Add9.468B278.574s. OriginalB512 Add complete training now has the passed case above; Attention
training and formal comparisons remain open.
Attention static diagnostic B1/12cards/minimum operator rows still gives coordinator
62.623GiB>53.875GiB usable. Projection borrowing alone is insufficient; no full-size
Attention training run is queued. Eager mixed multi-card is a separate core gap:
current runtime/validation require one payload device; CLI-only changes cannot
implement parameter/state/message placement, transfers, aliases and continuation.

Acceptance follows concrete F1–F7 gates, not a percentage. Remaining: original
B512 Attention complete training, eager mixed multi-card, finite full-size
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
