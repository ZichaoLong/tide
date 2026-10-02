# Current handoff

Updated 2026-10-03. **ACTIVE: user resumed after 7a7293a. Continue implementation,
qualification, commits and pushes. Overall goal incomplete. No subagents.**
Repository `/home/zlong/llm/graph-execution-foundation` resolves to
`/var/tmp/zlong-graph-execution-foundation/repository`; branch
`graph-execution-foundation`. Re-entry: `git status --short --branch`;
`python scripts/status.py`. Latest implementation **6b9224c** pushed; latest qualified implementation **233bf01**, its reviewed evidence **d0f9485** is pushed. [execution-flows](execution-flows.md)
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

**6b9224c** Attention parameter-adjoint reuse passed eight clean jobs: three builds,
native6 (160trajectories/2560windows/640updates), Python16, consumer32, same-lease
D512 calibration and independent FP16 profiling. All terminal exit0, no skips/new
failures, leases released. [Evidence](evidence/resident-attention-adjoints-20261003.md).
Audit: `python TASK/launchers/attention_adjoints_evidence.py 6b9224c829183e8aacefdfc5ccd416ecfeb4798d`.

Aggressive sharded reverse reuses event/fiber parameter adjoints only after the
window's canonical completion; state-owner init gates reset. Guards validate
nodes/offsets/device/dtype/shape/aliasing. KV/cache/state/message bridges remain
independent; conservative and legacy paths stay independent. Public ABI/core/CANN
unchanged. New counter`reused_attention_gradient_bytes`; consumer estimates are
still conservative and unchanged in this backend revision.

D512/B8/T4/V257 Attention,physicalB2×4,two windows,FP32 AdamW:
[8073745920,7226584576] → [7804784128,6957622784], -268961792bytes/card.
Loss7.532631874084473 and all prior statistics/chunks agree. Separate FP16 trace
53176ops, zero observed AiCPU. No throughput or original-size training claim.

Preceding **233bf01/d0f9485** window reduction/projection-gradient reuse is also
qualified: native176trajectories/2944windows/688updates,Python16/consumer32,
D512 peaks [8367995392,7520681472] → [8073745920,7226584576]. Combined actual
reductions across the two increments are about537MiB/card on this fixture.
Do not repeat unchanged Full-snapshot/accounting/representative matrices.

## Next WIP: physical parameter-gradient accounting

The next uncommitted consumer change charges physical projection/Attention
parameter gradients once only for aggressive multi-device training. Conservative
and legacy single-device estimates retain the window factor; all state/cache/
message/scratch charges and safety/API margins are unchanged. New included
component `projection_parameter_gradients`; existing Attention component adopts
the correct lifetime. Materialized model inventory tests cover both policies and
one/three devices; C++/Python formulas remain paired.

Changed `tools/online_bench/capacity.{h,py}`, `tests/test_consumer_capacity.py`,
`docs/consumer-capacity.md`. CPU23 and installed consumer build passed on frozen dirty `gradient-lifetime-dev01`
(at6b9224c); NPU25 is active (two devices,queue120s,timeout600), against the
byte-identical Attention backend. Unit`tide-execution-flows-gradient-lifetime-npu-dev01.service`,
logs/status`TASK/runs/gradient-lifetime-npu-dev01`. Next clean qualification and
D512 allocation calibration after the gate passes.
The 6b9224c evidence commit excludes this WIP.

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

Qualified source `TASK/sources/attention-adjoints-clean01`; consumer
`TASK/builds/attention-adjoints-consumer-clean01`. Resident binaries
`attention-adjoints-{standalone,python}-clean01`; core
`placement-{cpu,npu,npu-python}-clean01`; CPUconsumer`source-values-cpu-clean01`.
Consumer object reuse source/header/options-verified with fresh link;
unchanged dependencies hash-verified. No full rebuild claim.

Active: `gradient-lifetime-npu-dev01` (two NPUs,queue120s,timeout600), frozen dirty
`gradient-lifetime-dev01` at6b9224c. CPU23 and installed-client build are terminal
passed. Unit`tide-execution-flows-gradient-lifetime-npu-dev01.service`,
background.slice. Inspect `systemctl --user show UNIT -p ActiveState -p Result
-p ExecMainStatus`; stop via `systemctl --user stop UNIT` only if needed.
Logs/status`TASK/runs/NAME`. Both prior backend qualifications are fully terminal.
**Preserve deliberately SIGSTOPped
historical-cpu-attention-01**: never resume, stop or clean it. Its old record says
running and it holds old timing.lock. Historical1.6438× meant faster throughput
in the restricted flow, not current online evidence. Restricted archive:
archive/restricted-flow-20260930 at964bf628c67270200dabe55b1bca026bd403cd37.
Old build-reverse-gather-python-dev01 metadata inconsistency remains visible;
not a current failure. Current qualification raw records: `TASK/runs/{build-,}attention-adjoints-*-clean01`;
source/build/helper/result hashes and allocation observations are pinned by evidence.
