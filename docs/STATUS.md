# Current handoff

Updated 2026-10-02. **ACTIVE: user resumed after 7a7293a. Continue implementation,
qualification, commits and pushes. Overall goal incomplete. No subagents.**
Repository `/home/zlong/llm/graph-execution-foundation` resolves to
`/var/tmp/zlong-graph-execution-foundation/repository`; branch
`graph-execution-foundation`. Re-entry: `git status --short --branch`;
`python scripts/status.py`. Latest implementation **f360489**, pushed; accounting evidence **f03b4d4**. [execution-flows](execution-flows.md)
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

**f360489** Full snapshot sharing passed eight clean jobs: standalone/Python/
installed-consumer builds; native6 (144trajectories/2432windows/560updates),
Python16 retention/budget checks,consumer32, same-lease D512 allocator comparison
and separate FP16 profile. No skips. All units terminated exit0; leases released.
[Evidence](evidence/resident-full-snapshots-20261002.md). Audit:
`python TASK/launchers/full_snapshot_evidence.py f360489ac965abe9bc326055050ace01b348ebd5`.

Full parameter banks/static kind tables are copied once per backward group;
dynamic event metadata/values/counts remain per-window. Identity/version/layout
checks plus no-publication lifetime; backward/detach/close reset the cache.
`retained_full_bytes` is separate from `retained_window_bytes`. Default standalone
retention keeps independent copies. All private owner-layout users rebuilt;
clean source/header/options-verified object reuse with fresh links, unchanged
core/CANN hashes. Standalone dev01 checker compile failure remains retained;
corrected dev02 and all clean builds passed.

D512 Attention,physicalB2×4,two connected windows,FP32 AdamW: retained storage
-542752bytes; whole allocator peaks [8368268800,7520954880] →
[8367995392,7520681472] (-273408bytes/card). Loss7.532631874084473 and all other
observables/chunks/admission unchanged. Separate FP16 profile53182operators,
zero observed AiCPU. No speed claim. Wide fixtures use small LH Full banks, so
this is not the main full-size memory solution. Consumer estimates remain
unchanged and conservative; no original-width retry is justified by this delta.

Previous **4dd8368/f03b4d4** accounting qualification remains passed: CPU17/NPU25,
D512 allocation calibration. Shared Attention and private accumulation are charged
by lifetime, unchanged runtime and API safety limits. Actual peaks were unchanged
by that estimator-only correction. [Evidence](evidence/consumer-training-storage-20261002.md).

## Next implementation

Focus on dominant physical projection/Attention parameter gradient banks, which
are retained separately for all reverse windows before final canonical reduction.
Relevant files: `sharded_training_backward.cpp`, `sharded_parameter_reduce.cpp`,
`sharded_parameter_sources.cpp`, `projection_shard.cpp`, `emission_vjp.cpp`,
`sharded_state_vjp.cpp`, `state_owner_reverse.cpp`.

Candidate design (not implemented): reduce each completed window's contributions
into a common canonical bank on device before advancing to the preceding window,
then reuse only physical parameter-gradient storage. Keep per-owner addition order
reverse-window→registry-alias; no per-window pre-sum changing parentheses. Bridge
state/cache/message adjoints retain their independent storage. A coordinator-side
recorded reduction and peer start/completion protocol must finish all reads before
any reused bank is zeroed. State reverse already starts after a coordinator init
packet and acknowledges finish. Projection and remote Full currently zero before
their first request and need a start dependency before aliasing. Peer ACK means
copy consumption, not arbitrary subsequent receiver work. Preserve old independent
retention/reverse paths and budget/refusal semantics. No new implementation yet.

Do not spend another increment on tiny wide-fixture Full accounting alone.
Original Attention B512/physicalB1 estimates still refuse; AddB512 still exceeds
cost gate. No safety/cost relaxation or automatic original-width retry is queued.

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

Source `TASK/sources/full-snapshot-clean01`; consumer
`TASK/builds/full-snapshot-consumer-clean01`. Resident binaries
`full-snapshot-{standalone,python}-clean01`; core
`placement-{cpu,npu,npu-python}-clean01`; CPUconsumer`source-values-cpu-clean01`.
Consumer object reuse source/header/options-verified with fresh link;
unchanged dependencies hash-verified. No full rebuild claim.

No current job is running/queued. **Preserve deliberately SIGSTOPped
historical-cpu-attention-01**: never resume, stop or clean it. Its old record says
running and it holds old timing.lock. Historical1.6438× meant faster throughput
in the restricted flow, not current online evidence. Restricted archive:
archive/restricted-flow-20260930 at964bf628c67270200dabe55b1bca026bd403cd37.
Old build-reverse-gather-python-dev01 metadata inconsistency remains visible;
not a current failure. Raw records: `TASK/runs/full-snapshot-{native,python,consumer,memory,profile}-clean01`
and `TASK/runs/build-full-snapshot-{standalone,python,consumer}-clean01`; source/build/helper/result
hashes and allocation observations are in the reviewed evidence JSON.
