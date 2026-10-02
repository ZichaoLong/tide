# Current handoff

Updated 2026-10-02. **ACTIVE: user confirmed resume after checkpoint7a7293a. Continue authorized
implementation, qualification, commits and pushes. Overall goal remains incomplete.
No subagents.**
Repository `/home/zlong/llm/graph-execution-foundation` resolves to
`/var/tmp/zlong-graph-execution-foundation/repository`; branch
`graph-execution-foundation`. Latest implementation e82f971; prior pushed evidence
4e67d57. This checkpoint adds the audited nine-card pilot and pause handoff;
use `git log -1` for its evidence commit. Re-entry:
`git status --short --branch`; `python scripts/status.py`.
[execution-flows](execution-flows.md) owns the contract;
[ROADMAP F1–F7](ROADMAP.md) is the sole backlog. Reference repos and ObsidianVault
remain read-only. The following is a handoff, not another backlog.

## Contract and operating bounds

Every candidate independently consumes common inputs, parameters and initial
state. CPU reference events, routes and gradients never supply candidate work.
General online greedy supports legal family topology/input, including positive-
delay PDG feedback. Preserve int64, stable order, duplicate edges, missing/zero
messages, None/zero gradients and full continuation. Performance: PDG LibTorch;
TimedDAG/Settle LibTorch+PyTorch; CPU/NPU × streaming/prefill × inference/complete
training. Five presets plus fine switches. FP32 main, FP16 separate; training
means forward/loss/backward/optimizer/continuation, with no convergence requirement.

Implementation commit → clean fixed-source affected qualification → separate
evidence commit; push each under the standing authorization. The user has now resumed
execution after the requested checkpoint pause. User contract outranks run-ml-experiments;
minimal useful records only. No unchanged full CPU/representative reruns, unbounded
queue, OOM search, blind retries or relaxed safety/cost limits. Formal heavy timing
is serial; never stop other workloads to free resources.

## Completed checkpoint

Audited `wide-add-chunk-planner-nine01` PASSED on clean
`e82f97199c63dc8505137595adaae9a9c2d12221`. Unit
`tide-execution-flows-wide-add-chunk-planner-nine01` is inactive/dead, successful
exit0 and empty control group. Nine-card lease released. Source/build/helper/
packet/result hashes, static plan recomputation and all allocator/context bounds
passed `TASK/launchers/wide_add_chunk_planner_evidence.py <full SHA>`.
[Reviewed report](evidence/original-width-add-chunk-selection-20261002.md).

Original-width Add9.468B,D2048/T12/V50304, logicalB4/physicalB1 ×4,FP32SGD,
two connected windows,threads8: construction65.793s, complete update27.300s,
maximum allocator41.097GiB. Outputs96/cut408/events9265 match priorB4;
loss31.586036682128906 passes existingFP32 tolerance. Larger operator maxima
Full16/emission4/aggregate8/attention8/keys128/reverse1/head64 came from the
originalB512 shape-only plan. That plan estimates53.012GiB<53.875GiB usable.
The unchanged cost projection is4018.543s>3000s, so **B512 was not launched**.
One cold run with changed placement/physical batch is not a speed recommendation.

The preceding ten-card `wide-add-chunk-planner-pilot01` remains FAILED exit3:
configured queue120s, observed136.61s including polling/probes; consumer never
started. Nine cards used previously validatedB1 queue512/trace2048 capacities.
No automatic B512 stage or further resource retry remains queued.
Raw records: `TASK/runs/wide-add-chunk-planner-nine01/{status.json,task.log,queue.json}`
and `pilot/{result.json,original-plan.json,consumer-run/result.json}`.

Other qualified increments in this round:
- Optimizer finite-check recheck02 passed; recheck01 NPU OOM and collector failure
  remain retained. B4/physicalB2 ten-card25.073s update; B512 projection3690.815s
  still refused. Evidence14bc89c, [report](evidence/original-width-add-training-20261002.md).
- Private numeric accumulation b3a6a24/evidence7e98eef: eight clean jobs;
  50 distinct native boundary cases,32 trajectories/768windows/96updates,
  Python14/actual consumer24, separate FP16 profile53,182ops/zero observed AiCPU.
  Same-lease D512 allocator peaks unchanged; no whole-process speed/peak claim.
- General aggressive chunk selection e82f971/evidence4e67d57: four clean jobs,
  CPU13/NPU17, separate forced-splitting FP16 profile65,727ops/zero observed AiCPU.
  Python/C++ agree; envelope, margin, logical capacities and conservative policy
  unchanged. Only consumer planner/recording changed, resident/core/CANN bytes reused.

## Progress estimate for the requested alignment

Current F1–F7 local delivery is approximately **80% (roughly75–85%)**. This is a
planning estimate, not test pass rate, proof for arbitrary inputs or remaining-time
prediction. Weights: functionality35%, correctness30%, performance25%, delivery10%.
Midpoints90/90/50/80 yield about79%; rounded to80%. External GPU hardware work is
listed separately and is not counted as locally verified.

| Dimension | Estimated completion | Evidence and remaining boundary |
| --- | --- | --- |
| Functional implementation | 85–95% | General online scheduling, public CPU/mixed/resident flows, resident multi-card inference/training, FP32/FP16 and continuation implemented/qualified for declared profiles; eager mixed multi-card placement and full-size memory/cost remain open |
| Correctness qualification | 85–95% | Independent CPU references, full observables, VJPs, updates/restores and affected clean device gates exist; new scale changes and final integrated acceptance still required |
| Performance and profiling | 45–55% | All10 representative family/client/schedule submatrices complete; originalB512 inference passed for both models in LibTorch resident TimedDAG/prefill only; originalB512 complete training and full-size CPU/mixed/resident three-process comparisons remain open |
| Packaging, provenance and migration delivery | 75–85% | Installed clients, clean source/build identities, versioned packets and per-change evidence exist; F7 final support/migration/evidence audit remains open |

The main residual risk is scaling the complete training memory/compute lifecycle,
not basic operator availability. Attention original-width minimum-row estimates
still refuse admission (~90/80GiB on10/12cards); AddB512 fails the current projected
cost gate. Queue availability caused one bounded refusal, not the principal
implementation gap. Snapshot accounting and reverse-window gradient lifetimes
are promising analysis directions, not implemented solutions or a guarantee.

## Current next action after confirmed resume

Read ROADMAP F4–F7 and the reviewed pilot first. Investigate the dominant retained
training allocations before any new large run:
`rg -n 'snapshot|retained|gradient' tools/online_bench/capacity.py tools/online_bench/capacity.h tools/device_online/retained_attention.h`.
Immutable attention parameters are already shared at38858d0; estimator reductions
need exact component/lifetime proof. Dynamic KV/bias/journals stay per-window.
Any reverse-window workspace reuse must preserve reverse-window→registry-alias
addition order and device completion dependencies; never alias remote Full zeroing
before a prior use finishes. No such new change exists in this checkpoint.
Other remaining acceptance is listed only under ROADMAP F1–F7.

## Environment and protected state

`TASK=/mi/data2T/zlong/tide-execution-flows`;
module `libtorch-npu/2.10.0-cann9.0.0`;
Python `/opt/miniconda/envs/ascend900-train-full-torch-npu-2.10.0-py311/bin/python`.
User-authorized public /opt stack supersedes the old private account guide.
Preserve module PYTHONPATH; prepend frozen source/python.
`TASK_QUEUE_ENABLE=0 TORCH_DEVICE_BACKEND_AUTOLOAD=0`; lease/remap devices.
Long jobs use frozen source,background.slice,Nice10,two build workers;
runtime `env -C {out}` avoids writes into snapshots. Queue120s, formal timing
lock `TASK/online-measurement.lock`. Last free disk:data156GiB/root13GiB.

Latest source `TASK/sources/chunk-planner-clean01`; consumer
`TASK/builds/chunk-planner-consumer-clean01`. Resident libraries are
`private-accumulation-{standalone,python}-clean01`; core
`placement-{cpu,npu,npu-python}-clean01`; CPUconsumer `source-values-cpu-clean01`.
Private accumulation reused source/compile-command-matched objects with fresh
links; unchanged core/CANN were hash-verified. No all-dependencies-rebuilt claim.

No current experiment remains running or queued. **Preserve deliberately SIGSTOPped
historical-cpu-attention-01**, whose old durable record says running and which
holds old timing.lock: never resume, kill or clean it without a new instruction.
Historical1.6438× refers to faster throughput in the restricted historical flow,
not current general-online evidence. Restricted archive remains
archive/restricted-flow-20260930 at964bf628c67270200dabe55b1bca026bd403cd37.
The old build-reverse-gather-python-dev01 metadata inconsistency remains visible
in status.py; do not relabel it or confuse it with a new failure.

OriginalB512 inference evidence: Attention17.521B325.278s/14.562GiB per card;
Add9.468B278.574s/7.812GiB per card. These are cold capacity observations, not the
full performance matrix. Representative entry:
[evidence](evidence/representative-settle-python-20261002.md).
CUDA true-device execution and new host/CANN tuples require target-machine gates;
older eager four-stack acceptance does not certify the new resident backend there.
Re-entry was clean at7a7293a. Current implementation: consumer capacity now
charges one shared immutable Attention snapshot and one extra live private FP32
accumulation bank; the original two-input accumulation API limit and safety
margins remain unchanged. Development frozen source TASK/sources/training-storage-dev01
passed CPU16, installed consumer build and NPU17 (16 independent CPU comparisons
plus one preallocation refusal), without skips. No active current jobs.

Next: commit this planner/docs/test increment; qualify that exact clean commit
as TASK/sources/training-storage-clean01. Planned bounded jobs: CPU17, installed
consumer build, NPU25 including automatic sample admission and actual sliced
updates, and one D512 allocator calibration using the matching qualified backend.
Reuse unchanged core/CANN binaries with verified hashes; no redundant profiler or
whole CPU/representative matrix. Task helper training_storage_calibration.py keeps
prior measurements and verifies the smaller estimate against actual allocation.
Large original Attention admission still refuses (about68GiB on12cards/57GiB
on16cards with B1 capacities); this correction does not complete scale acceptance.
No new original-width pilot is queued or planned without another concrete gain.
