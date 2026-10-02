# Current handoff

Updated 2026-10-02. **ACTIVE: user confirmed resumption.** Continue implementation,
qualification, commits and pushes under [execution-flows.md](execution-flows.md).
No subagents. Overall goal incomplete. Repository
`/home/zlong/llm/graph-execution-foundation` resolves to
`/var/tmp/zlong-graph-execution-foundation/repository`, branch
`graph-execution-foundation`. Latest pushed implementation: **1757b90** (shared reverse gather); latest evidence **9829e8a**.
Reference repositories and ObsidianVault are read-only. Re-entry:
`git status --short --branch`; `python scripts/status.py`.
[ROADMAP F1–F7](ROADMAP.md) is the sole backlog. This file owns current jobs/next actions.

## Contract and resource rules

Every candidate independently consumes common inputs, parameters and initial state;
reference events/routes/results/gradients never supply execution. General online
greedy supports legal family topology/input, including positive-delay PDG feedback.
Preserve int64, stable order, duplicate edges, missing/zero messages, None/zero
gradients and complete continuation. Performance: PDG LibTorch; TimedDAG/Settle
LibTorch+PyTorch; CPU/NPU × streaming/prefill × inference/complete training.
Five presets plus fine switches; FP32 main, FP16 separate. No convergence requirement.

Implementation commit → fixed clean affected qualification → separate evidence
commit; push each. Current user contract outranks run-ml-experiments; reuse minimal
records. No repeated unchanged8,954 CPU checks or completed representative timings.
Formal heavy timings are serial. No unbounded queue, automatic OOM search or blind
retry. CANN/runtime jobs use `env -C {out}` to keep vendor files out of frozen source.

`TASK=/mi/data2T/zlong/tide-execution-flows`. Module `libtorch-npu/2.10.0-cann9.0.0`;
Python `/opt/miniconda/envs/ascend900-train-full-torch-npu-2.10.0-py311/bin/python`.
User-authorized public /opt stack supersedes the older account guide. Preserve
module PYTHONPATH, prepend frozen source/python. `TASK_QUEUE_ENABLE=0`,
`TORCH_DEVICE_BACKEND_AUTOLOAD=0`. Sixteen logical64GiB Ascend910_9392; lease/remap
only. `launchers/freeze_run.py`: immutable snapshot, background.slice, Nice10,
two build workers. Last disk check: data174GiB/root13GiB free.
Formal timing lock: `TASK/online-measurement.lock`.

**Preserve historical-cpu-attention-01:** deliberately SIGSTOP, holds old timing.lock.
Its durable record says running; never resume, kill or clean it. Historical1.6438×
was faster throughput, not evidence for current general-online flows. Restricted
history remains on pushed archive/restricted-flow-20260930 at964bf628c67270200dabe55b1bca026bd403cd37.

## Original-width Add failures: preserved

`wide-add-training-b2-01` failed calibration on clean c3ed0f2; its old writer lost
per-card measurements. No OOM/timeout, B512 did not run. Raw records and refusal
are preserved in [report](evidence/original-width-add-training-20261002.md).
Reporting3462dae now preserves complete measurements even on post-run calibration
failure; independently qualified CPU19/NPU9, evidencefddf079. No estimate was relaxed.

## Attention snapshots: fully qualified

Implementation **38858d0** is qualified by nine PASSED clean jobs. Immutable
QKV/output matrices, parameter biases, decay and pooling weights share one
snapshot per backward group; dynamic KV/log-bias/lengths/journals stay separate.
Forward-bank identity/version/layout guards plus no publication with live tapes
protect lifetime; backward/detach/close reset caches. Public ABI/core/CANN unchanged.
Consumer physical admission remains unchanged.

Source `TASK/sources/attention-snapshot-clean01`; three clean builds and six runs
`attention-snapshot-{native,python,consumer,restore,memory,profile}-clean01` passed.
Python16/consumer24 checks, native128trajectories/2048windows/512updates and
explicit2→3 restore32trajectories/512windows/128updates, FP32/FP16 and independent
CPUFP32/FP64. Same-lease D512/B8/T4/V257 Attention FP32AdamW two-window update:
allocator/card reduced275,262,464B; exact loss/non-memory counters unchanged.
Separate FP16 two-update profile:53,274operators; no observed AiCPU.
`launchers/attention_snapshot_evidence.py` audited all source/object/kernel/loader,
JUnit, trajectory, allocator and CSV identities. [Report](evidence/resident-attention-snapshots-20261002.md).
This is not original-size training or throughput proof. No attention jobs remain live.

## Original-width diagnostic finished; shared reverse gather development

Reporting3462dae / evidencefddf079 are committed/pushed, CPU19/NPU9 qualified.
`wide-add-memory-diagnostic01` completed diagnostic collection; its actual
consumer **FAILED** post-run calibration, and original B512 never ran.
Clean3462dae source `TASK/sources/failure-records-clean01`, ten-card lease released.
Raw `runs/wide-add-memory-diagnostic01/diagnostic/{result.json,consumer-run/result.json}`.
OriginalD2048/B4/physicalB2 AddFP32SGD,2windows/1update: construction243.789s,
step39.390s,96outputs/cut408,pending768/maxevents2335. Coordinator actual51.4198GiB
versus estimated50.0756GiB; other cards29.5GiB<36.4GiB. 60GiB cap/53.875GiB usable
unchanged. One GDB sample caught CPUvalidate_model finite reduction during
construction; timings are diagnostic, not a throughput recommendation.
Do not raise estimates/caps blindly or repeat the full run without a code cause.

Implementation1757b90 shares read-only padded gather inputs within
one reverse program phase across owners (Full values/gradients, state journals
and scales, stage cotangents/flags). Per-owner indices/results/flags remain
independent; each device-loop iteration refreshes the shared source. Helper
allocates only after existing owner preflight, and storage stays owned by program.
No core/public ABI/device kernel change or estimator reduction.
Qualified clean source `TASK/sources/reverse-gather-clean01` at1757b90; all nine
jobs PASSED: three builds, native128trajectories/2048windows/512updates, Python16,
consumer24, explicit2→3 restore32trajectories/512windows/128updates, same-lease
memory comparison and independent FP16 two-update profile. Coordinator peak
8396636672→8368268800B (-28367872B), other card7520954880B unchanged; loss and all
work/retained-storage counters exact.53,190profile operators, zero observed AiCPU.
[Report](evidence/resident-reverse-gathers-20261002.md). Public ABI/core/kernels and
consumer estimates/caps unchanged. Development module-name startup failure remains
preserved; corrected/clean jobs use established LibTorch-NPU2.10/CANN9.0 stack.

Original-width Add `wide-add-reverse-gather-recheck01` PASSED on1757b90; ten cards
released. Same B4/physicalB2 geometry/caps/chunks/FP32SGD/two windows as failed
3462dae diagnostic. Coordinator55211536896→49339172864B (-5872364032B), below
unchanged53768286884B estimate. All cards calibrated; loss/work/retained counters
exact. Construction254.961s, update27.757s,96outputs/cut408. Cold descriptive timing,
not causal speed recommendation. B512 cost projection4085.875s>3000; B512 not run.
Raw TASK/runs/wide-add-reverse-gather-recheck01/{status.json,task.log,recheck/result.json};
reviewed evidence prepared in original-width-add-reverse-gather-20261002.json and report.

## Completed evidence; do not repeat

All ten required representative family/client/schedule matrices completed on
80dae6e; last evidence270ee2e, last matrix-remaining02 passed/released.
[Settle/Python report](evidence/representative-settle-python-20261002.md) links the
matrix set. Warm resident2.698–8.848× versus default PythonCPU is not versus tuned
LibTorchCPU or original-wide performance.

Both original-wide FP32 LibTorch resident TimedDAG/prefill inference runs passed:
480body/2208edges,D2048/B512/T12/V50304,two windows,128physicalB4 groups.
- Attention17.521B: source48e44b0, wide-inference-staged01; construction817.996s,
  step325.278s, peak14.562GiB/card. [Report](evidence/original-wide-inference-20261002.md).
- Add9.468B: sourcebe380db, wide-add-inference02; construction182.038s,
  step278.574s, peak7.812GiB/card. [Report](evidence/original-wide-add-inference-20261002.md).
Both produced12,288outputs/cut408. Cold capacity evidence, not formal throughput
or all-family/client/schedule certification. Original packet files are unchanged.
Region(budget) defaultsobserve_all=true: unselected nodes retainKV. Never shrink
logicalB512, dropKV or change precision silently to claim original size passed.

Original-width AddB2 training (11cards,physicalB1,source475d4af) passed in staged02:
construction249.211s, step20.2794s, peak42.216GiB/card,48outputs,cut408,
pending384/maxevents1177. B512 cost gate5191.5s>3000 stopped parent.
Ten-card profile02 passed:701,139operators/zero observed AiCPU; trace includes
construction/cleanup, not throughput. Profile01 queue timeout and staged01
reverse-capability refusal remain FAILED. [Report](evidence/original-width-add-training-20261002.md).

Recent qualified increments (retain cited artifacts/failures):
- compact context pool48e44b0: CPU17/NPU64, allocator−5.39% at its fixture.
  [Report](evidence/resident-context-pool-20261002.md).
- retained journals0fbc1b2: CPU17/NPU44, representative allocator−22.10%.
  [Report](evidence/resident-retained-journals-20261002.md).
- exact initializerbe380db: CPU25/NPU30;6.25–6.51× initializer-only CPU gain.
  [Report](evidence/exact-initializer-20261002.md).
- window peaks475d4af:34checks; actual device counts at boundaries.
  [Report](evidence/resident-window-peaks-20261002.md).
- automatic sample admission3c3b4e7: CPU9/NPU18; fixed logical semantics.
  [Report](evidence/resident-auto-samples-20261002.md).
- shared packetsa72868f: eight jobs passed; representative D512/B8 Attention
  old/new each card−268,437,504bytes; no speed claim. [Report](evidence/resident-shared-packets-20261002.md).
- Add-specific admissionc3ed0f2: CPU10/NPU9; remove nonexistent Attention matrix
  gradients only. [Report](evidence/consumer-add-capacity-20261002.md).

Latest fully qualified resident libraries: reverse-gather-{standalone,python}-clean01;
NPUconsumer reverse-gather-consumer-clean01. Core placement-{cpu,npu,npu-python}-clean01;
CPUconsumer source-values-cpu-clean01. Previous builds remain for cited evidence.
Standalone LibTorch and Python-native runtimes are separate.

## Next implementation and remaining delivery

Qualify shared reverse gather, then use measured allocation savings to decide
one bounded original-width Add revisit. Full originalB512 training/formal comparisons remain.
AttentionFP32 still statically refuses at minimal rows: inspect real lifetime of
physical/canonical gradients and optimizer/accumulation storage, not arbitrary
safety reductions. Read-only inspection found accumulation keeps independent
old/replacement banks so public backward views remain stable; aliasing flags in
its kernel would race. No accumulation change implemented or admitted yet.
Eager mixed multi-device parameter/payload placement remains open. Finish required
full-size family/client/schedule comparisons and three-process recommendations,
then F7 migration/evidence/support audit. CUDA and other environment tuples require
target-machine execution. Historical CPU Attention is supplementary, not blocking.

Uncommitted: optimizer_vector.h uses vector abs/ordered compare and packed
mask checks instead of per-element scalar finite reads. Local comparison scratch
is rounded to64 lanes, zero-filled; only actual data is read and valid bits checked.
optimizer_finite_check.h adds504 actual transaction probes for nine widths,
mask/tile tails, signed NaN/Inf, finite extremes and disconnected poison.
No arithmetic/tiling/budget/ABI change. Source optimizer-finite-dev02 is frozen.
All three dev02 builds PASSED. optimizer-finite-component-dev03 PASSED:
local32 trajectories/payload, peer4 trajectories/payload,504 boundary probes,
CPUFP32/FP64, finite transaction and four1M-element allocator calibrations.
Actual24 consumer checks PASSED optimizer-finite-consumer-dev01 on dev02 source,
2card/queue120s/workload600s; TASK/runs/optimizer-finite-consumer-dev01.
Retain dev01 CLI refusal (peer-optimizer misspelling), dev02 actual device failure
(unrounded Compare primitive skipped tails), and corrected dev03 results separately.
Task helper launchers/build_optimizer_finite.py rebuilds one CANN kernel/checkers,
byte-verifies unchanged host archive/core/other kernels, fresh links.
Next commit/push tested implementation; fixed-source component/consumer/2→3 restore,
independent profile and task-local optimizer_finite_bench.py:three old/new independent
processes on one lease,16,777,219elements, momentum/AdamW,2warmup/5measured updates,
CPU scalar oracle outside timing. Isolated optimizer evidence, not graph throughput.

Original-width Add recheck terminal passed; reviewed evidence committed9829e8a.
Preserve the stopped historical job.
