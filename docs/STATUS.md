# Current handoff

Updated 2026-10-03. **ACTIVE: user resumed after 7a7293a. Continue implementation,
qualification, commits and pushes. Overall goal incomplete. No subagents.**
Repository `/home/zlong/llm/graph-execution-foundation` resolves to
`/var/tmp/zlong-graph-execution-foundation/repository`; branch
`graph-execution-foundation`. Re-entry: `git status --short --branch`;
`python scripts/status.py`. Latest implementation **4467493** pushed and qualified;
its reviewed evidence is ready for the separate evidence commit.
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

## Latest completed qualification

**4467493** private projection-bank borrowing passed eight clean jobs: three builds,
native6 (160trajectories/2560windows/640updates),Python16,consumer32,same-lease D512
calibration and separate FP16 profiling; no skips/failures. All terminal, leases
released. [Evidence](evidence/resident-projection-borrow-20261003.md).
Audit: `python TASK/launchers/projection_borrow_evidence.py 44674936efcc4177d4683a7e8542816aec0928eb`.

Only aggressive sharded training borrows private frozen forward projection banks.
The owner forbids publication with live windows and closes reverse programs before
step. Default retention, conservative/legacy paths keep independent copies.
Attention/Full snapshots, state/KV/message bridges, all retained API budgets and
consumer estimates stay unchanged. No public ABI/class-layout/core/CANN change.
New included counter `borrowed_projection_bytes` is not an allocator measurement.

D512/B8/T4/V257 Attention,physicalB2×4,two windows,FP32 AdamW, same lease:
[7804784128,6957622784] → [7508509696,6661348352], -296274432bytes/card (282.5MiB).
Loss7.532631874084473 and prior statistics/chunks/continuation unchanged. Separate
FP16 profile53176ops, zero observed AiCPU. No throughput/original-size claim.

Preceding qualified changes: 233bf01 window reduction/projection-adjoint reuse
([evidence](evidence/resident-window-reduction-20261003.md));6b9224c Attention
adjoint reuse ([evidence](evidence/resident-attention-adjoints-20261003.md));789e1a5
physical-gradient lifetime accounting,CPU23/NPU25, unchanged actual peaks
([evidence](evidence/consumer-gradient-lifetime-20261003.md), evidence0760300).
Do not repeat unchanged CPU/representative matrices.

## Next bounded action and remaining scale

After evidence commit/push, reassess one original-width Add nine-card B4/physicalB2
update using clean4467493. Prepared `TASK/launchers/wide_add_projection_borrow.py`
checks all clean runtime jobs, uses original B512 shape-only plan, and retains
queue/arrivals896,trace3072,outputs64,KV256,KV-trace8192,60GiB/card,head/context4GiB.
Only enters one cold B512 update if allocator/semantic checks pass and unchanged
measured-seconds×128×1.15<=3000. Queue120s,pilot900s,B5123180s,outer4300s,
`online-measurement.lock`. Not yet submitted; no retry or gate relaxation.
Command: `python TASK/launchers/freeze_run.py --name wide-add-projection-borrow01 --snapshot projection-borrow-clean01 --commit 4467493 --npu --npu-count 9 --max-wait 120 -- env -C {out} timeout 4300 {python} {base}/launchers/wide_add_projection_borrow.py --source {source} --build {base}/builds/projection-borrow-consumer-clean01 --output {out}/assessment`.

Previous789e1a5 original-width Add nine-card B4/physicalB2 pilot passed:
21.179427721s,construction84.305374079s,peak47394238464bytes,loss31.58603858947754,
outputs96/events9265/cut408. Projection3117.611761s>3000s, so B512 was not executed.
[Evidence](evidence/original-width-add-gradient-lifetime-20261003.md),commit749f7c5.
Earlier B1 pilot27.300s/projection4018.543s and ten-card queue failure remain scoped.

All ten required representative submatrices complete. OriginalB512 TimedDAG/
LibTorch/resident/prefill FP32 inference passed:Attention17.521B325.278s,
Add9.468B278.574s. Full-size complete training/formal comparisons remain open.
Attention's current static estimate still refuses: diagnostic B1/12cards minimum
rows gives coordinator62.623GiB>53.875GiB usable; no Attention scale run is queued.
Mixed multi-device is a separate core gap: eager runtime/validation require one
payload device; CLI-only changes cannot implement parameter/state/message placement.

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
Last free disk:data149GiB/root7.7GiB; check before large writes.

Qualified source `TASK/sources/projection-borrow-clean01`; binaries
`TASK/builds/projection-borrow-{standalone,python,consumer}-clean01`.
Core `placement-{cpu,npu,npu-python}-clean01`; CPUconsumer `source-values-cpu-clean01`.
Old/new memory baseline consumer `gradient-lifetime-consumer-clean01` is retained.
Objects/dependencies reused only with source/header/options/hash verification;
consumer freshly linked. No full rebuild claim.

All projection-borrow development6/qualification8 jobs terminal passed. No current
active/queued task job. Units `tide-execution-flows-NAME.service`, logs/status
`TASK/runs/NAME`; inspect `systemctl --user show UNIT -p ActiveState -p Result -p ExecMainStatus`;
stop via `systemctl --user stop UNIT` only if needed.
**Preserve deliberately SIGSTOPped historical-cpu-attention-01**: never resume,
stop or clean it. Its old record says running and it holds old timing.lock.
Historical1.6438× meant faster throughput in a restricted flow, not current online
evidence. Archive:archive/restricted-flow-20260930 at964bf628c67270200dabe55b1bca026bd403cd37.
Old build-reverse-gather-python-dev01 metadata inconsistency remains visible;
not a current failure. Audit helpers pin source/build/helper/result/profile hashes.
