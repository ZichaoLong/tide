# Current handoff

Updated 2026-10-03. **ACTIVE: user resumed after 7a7293a. Continue implementation,
qualification, commits and pushes. Overall goal incomplete. No subagents.**
Repository `/home/zlong/llm/graph-execution-foundation` resolves to
`/var/tmp/zlong-graph-execution-foundation/repository`; branch
`graph-execution-foundation`. Re-entry: `git status --short --branch`;
`python scripts/status.py`. Latest qualified implementation **233bf01**, pushed; its reviewed evidence is included
in this separate documentation commit. [execution-flows](execution-flows.md)
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

**233bf01** passed eight clean jobs, all terminal exit0: standalone/Python/installed
consumer builds; native8 (176trajectories/2944windows/688updates); Python16,
consumer32; same-lease D512 allocator comparison; independent FP16 profile.
No skips or new failures. [Evidence](evidence/resident-window-reduction-20261003.md).
Audit: `python TASK/launchers/window_reduction_evidence.py 233bf0144db206176f4af4b05b0bb54cb326dbe7`.

Aggressive sharded training reduces each window on device before advancing the
preceding window, preserving reverse-window→registry-alias additions. Projection
parameter adjoints, canonical outputs and numerical packet arenas are reused
behind start/completion barriers. State/cache/message bridge adjoints remain
independent. Conservative and legacy single-device paths are preserved.
New counters: `streamed_parameter_windows`, `reused_projection_gradient_bytes`.
Public ABI/core/device kernels unchanged; affected host objects rebuilt and
clean source/header/options-verified object reuse with fresh links.

D512/B8/T4/V257 Attention,physicalB2×4,two connected windows,FP32 AdamW:
whole allocator [8367995392,7520681472] → [8073745920,7226584576], reductions
[294249472,294096896] bytes. Loss7.532631874084473, other semantic/work/retention
statistics and effective chunks unchanged. Independent FP16 trace53176operators,
zero observed AiCPU. This is actual allocation improvement, not a throughput or
original-size training claim. Consumer admission remains unchanged/conservative.

Previous Full snapshot and training-accounting increments remain qualified on
f360489/4dd8368; evidence1a6d9c0/f03b4d4. Do not rerun their unchanged matrices.

## Current WIP: Attention parameter-adjoint reuse

Uncommitted Attention parameter-adjoint reuse extends the already-tested ordered
window reduction. Only owner-local event/fiber parameter values and connection
flags are borrowed; KV/cache/state/message adjoints remain independent. Parameter
mapping/device/dtype/shape/alias guards precede reuse, and the existing state-owner
init packet gates zeroing after the preceding window's canonical completion.
Conservative/standalone paths keep independent storage. New statistic:
`reused_attention_gradient_bytes`. Modified: state_owner_reverse,sharded_state_vjp,
sharded_full_vjp,sharded/legacy backward and checker/sample-chunk assertions.
Three development builds passed on frozen dirty `attention-adjoints-dev01`
(at233bf01). Python16 passed; native6 and consumer32 gates are active/bounded.
Next: inspect every gate; after passing commit implementation, clean qualification
and same-lease calibration against `window-reduction-consumer-clean01`, separate
FP16 profile, then evidence. Consumer estimates remain unchanged. No full-size
retry or relaxed admission/cost rule is authorized by this intermediate result. Do not include WIP code in the separate 233bf01 evidence commit.

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

Qualified source `TASK/sources/window-reduction-clean01`; consumer
`TASK/builds/window-reduction-consumer-clean01`. Resident binaries
`window-reduction-{standalone,python}-clean01`; core
`placement-{cpu,npu,npu-python}-clean01`; CPUconsumer`source-values-cpu-clean01`.
Consumer object reuse source/header/options-verified with fresh link;
unchanged dependencies hash-verified. No full rebuild claim.

Active development jobs: `attention-adjoints-{native,consumer}-dev01`,
frozen dirty `attention-adjoints-dev01`; three builds passed. Two NPUs each,
queue120s,timeout900/600 respectively. Python16 is terminal passed. Unit prefix `tide-execution-flows-`,
background.slice. Inspect `systemctl --user show UNIT -p ActiveState -p Result
-p ExecMainStatus`; stop via `systemctl --user stop UNIT` only if needed.
Logs/status `TASK/runs/NAME`. Current clean233bf01 qualification is fully terminal;
its raw records use `{build-,}window-reduction-*-clean01`. Preserve receipts.
**Preserve deliberately SIGSTOPped
historical-cpu-attention-01**: never resume, stop or clean it. Its old record says
running and it holds old timing.lock. Historical1.6438× meant faster throughput
in the restricted flow, not current online evidence. Restricted archive:
archive/restricted-flow-20260930 at964bf628c67270200dabe55b1bca026bd403cd37.
Old build-reverse-gather-python-dev01 metadata inconsistency remains visible;
not a current failure. Current qualification raw records: `TASK/runs/{build-,}window-reduction-*-clean01`;
source/build/helper/result hashes and allocation observations are pinned by evidence.
