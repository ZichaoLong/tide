# Current handoff

Updated2026-10-02. **ACTIVE: user authorized continued implementation, qualification,
commits and pushes. No subagents. Overall goal incomplete.** Repository
`/home/zlong/llm/graph-execution-foundation` resolves to
`/var/tmp/zlong-graph-execution-foundation/repository`; branch
`graph-execution-foundation`. Latest pushed implementation **b3a6a24**;
latest evidence committed/pushed7e98eef. Re-entry:
`git status --short --branch`; `python scripts/status.py`.
[execution-flows](execution-flows.md) owns the contract; [ROADMAP F1–F7](ROADMAP.md)
is the sole backlog. References and ObsidianVault stay read-only.

## Contract and operating bounds

Each candidate independently consumes common inputs/parameters/initial state;
CPU reference events/routes/gradients never become candidate execution inputs.
General online greedy supports legal family topology/input, including positive-
delay PDG feedback. Preserve int64, stable order, duplicate edges, missing/zero
messages, None/zero gradients and complete continuation. Performance: PDG LibTorch;
TimedDAG/Settle LibTorch+PyTorch; CPU/NPU×streaming/prefill×inference/complete training.
Five presets plus fine switches. FP32 main, FP16 separate; no convergence requirement.

Implementation commit → fixed clean affected qualification → separate evidence
commit; push each. Current user contract outranks run-ml-experiments; minimal
records only. No repeated unchanged8,954 CPU checks or completed representative
timings. Formal heavy timings serial; no unbounded queue, OOM search or blind retry.

`TASK=/mi/data2T/zlong/tide-execution-flows`. Module `libtorch-npu/2.10.0-cann9.0.0`;
Python `/opt/miniconda/envs/ascend900-train-full-torch-npu-2.10.0-py311/bin/python`.
User-authorized /opt stack supersedes older account guide. Preserve module
PYTHONPATH and prepend frozen source/python. `TASK_QUEUE_ENABLE=0`,
`TORCH_DEVICE_BACKEND_AUTOLOAD=0`.16 logical64GiB Ascend910_9392; lease/remap only.
`launchers/freeze_run.py`: immutable snapshot,background.slice,Nice10,2 build workers.
Runtime commands use `env -C {out}` to avoid CANN writes in frozen source.
Queue120s; formal timing lock `TASK/online-measurement.lock`.
Last disk check:data171GiB/root13GiB free.

**Preserve historical-cpu-attention-01:** deliberately SIGSTOP; holds old timing.lock.
Its durable record says running. Never resume, kill or clean it. Historical1.6438×
was faster throughput, not current general-online evidence. Restricted archive:
archive/restricted-flow-20260930 at964bf628c67270200dabe55b1bca026bd403cd37.

## Current result and next implementation

Private numeric accumulation **b3a6a24** passed all8 clean jobs; no live task jobs
except the deliberately stopped historical CPU process. Evidence:
[report](evidence/resident-private-accumulation-20261002.md), audited by
TASK/launchers/private_accumulation_evidence.py.50 distinct native boundary cases,
32 trajectories/768windows/96updates,Python14,actual consumer24. Separate FP16
profile53,182ops,zero observed AiCPU. First accumulation copies exports; later
updates reuse private values while old/new flags remain separate. Public ABI/core,
CANN,consumer estimator,max_bytes admission and safety margins unchanged.
Same-lease D512/B8 Attention peak8,368,268,800/7,520,954,880B unchanged; exact loss
and work/retention counters. This is not a whole-process memory or speed gain.

Source TASK/sources/private-accumulation-clean01. Latest qualified builds
private-accumulation-{standalone,python,consumer}-clean01. Two matching dev host
objects reused after source/compile-command verification; fresh clean links.
Other host/core/CANN and client objects reused by hash; no from-scratch claim.
Core placement-{cpu,npu,npu-python}-clean01; CPUconsumer source-values-cpu-clean01.

Now developing **aggressive physical chunk selection**. Current planner halves all fields together, needlessly shrinking
low-memory work. Investigate deterministic one-field greedy reduction of actual
per-card peak excess, with safe plateau fallback. Keep the same envelope,
headroom,logical capacities and general topology/input behavior. Validate Python/
C++ planner parity and directed consumer/profile cases before another bounded
original-width pilot. No arbitrary estimator or3000s cost-limit relaxation.

## Retained full-width results and failures

All ten required representative family/client/schedule submatrices are complete
on80dae6e; [entry report](evidence/representative-settle-python-20261002.md).
Do not rerun them unchanged. Warm resident2.698–8.848× versus default PythonCPU
is not versus tuned LibTorchCPU or original-wide throughput.

Original FP32 LibTorch resident TimedDAG/prefill inference passed, two windows:
480body/2208edges,D2048/B512/T12/V50304,128 physicalB4 groups,12,288outputs/cut408.
- Attention17.521B:325.278s,14.562GiB/card peak,source48e44b0;
  [report](evidence/original-wide-inference-20261002.md).
- Add9.468B:278.574s,7.812GiB/card peak,sourcebe380db;
  [report](evidence/original-wide-add-inference-20261002.md).
Cold capacity evidence, not full-matrix formal CPU comparison. Original packets
remain unchanged. Region budget observe_all=true retains KV for unselected nodes.

[Original-width training history](evidence/original-width-add-training-20261002.md):
- AddB2/physicalB1 on11cards passed source475d4af:20.279s update,42.216GiB peak.
  B512 projection5191.5s>3000 stopped parent. Ten-card profile701,139ops,zero AiCPU.
- B4/physicalB2 on10cards,c3ed0f2 failed post-run calibration and lost peak details.
  Reporting3462dae fixed failure recording, qualified CPU19/NPU9.
- Same3462dae diagnostic preserved failure:coordinator51.4198GiB>50.0756GiB estimate.
  Other cards29.5GiB. No OOM/timeout. construction243.789s/update39.390s.
- Shared reverse gather1757b90 passed identicalB4 geometry:coordinator45.950GiB,
  loss/counters exact,update27.757s. B512 projection4085.875s>3000.
- Vector optimizerbb40cff recheck01 failed NPU OOM then collectorKeyError; records
  remain FAILED. Allocation requested3.46GiB;26.11allocated/28.16reserved/1.79free
  of61.27GiB; no contemporaneous inventory identifies the other occupancy.
- One bounded recheck02 PASSED with corrected collector and before/after snapshots.
  Same caps/model/B4/physicalB2/two windows; explicit ATen threads8 instead of1.
  Peaks/loss/all counters exactly match1757b90; construction66.567s/update25.073s.
  B512 projection3690.815s>3000; B512 never ran. No further resource retries.
  [audited record](evidence/original-width-add-optimizer-finite-20261002.json),
  evidence commit14bc89c. Separate leases/thread changes prevent causal speed claims.

Recent qualified generic improvements: immutable attention snapshots38858d0,
shared reverse gathers1757b90,vector optimizer finitebb40cff. The latter has three
independent same-card isolated phase timings:SGD6.778×/AdamW7.007× throughput,
not whole-graph speed; [report](evidence/resident-optimizer-finite-20261002.md).
All cited raw sources/builds/runs and failures remain retained.

## Remaining delivery

OriginalB512 complete training and Attention original-width admission are open;
minimum-row old envelopes estimate about90/80GiB on10/12cards. Do not equate a
failed capacity estimate with completion. Consider reverse-window gradient
lifetime, exact retained snapshot accounting and general memory-aware placement.
Maintain reverse-window→registry-alias floating addition order; per-window sums
then addition would change grouping. Current private flags must never be aliased.
Eager mixed multi-device parameter/payload placement remains unimplemented.
Required full-size CPU/screened mixed/resident comparisons and three independent
process recommendations remain, followed by F7 migration/support/evidence audit.
CUDA and other environment tuples require target-machine validation.

Aggressive one-field selection passed dirty-source development:
chunk-planner-cpu-dev01 CPU13 (including Python/C++ constrained-shape parity),
build-chunk-planner-consumer-dev01 installed consumer build/loader,
chunk-planner-npu-dev01 NPU17 actual split/automatic-sample cases with CPU oracle,
FP32/FP16 and native/LibTorch. Every observed allocation passed the same envelope.
No core/resident backend/CANN changes. New row_selection records the rule.

Commit implementation, then clean chunk-planner-clean01 snapshot: CPU13,installed
consumer build with source/header/options-verified object reuse,NPU17,and separate
forced-splitting FP16 profile. Profile task helper profile_chunk_planner.py,
2cards,device-memory-bytes1839217549,queue120s/workload480s. After qualification,
one bounded ten-card original-width AddB4/physicalB2 pilot using the plan selected
for unchanged B512. No automatic full-batch stage; retain3000s gate.
Static new plan with default head4GiB:Full16/emission4/aggregate8/attention8/
keys128/reverse1/head64,coordinator53.142GiB<53.875GiB usable. This is a plan only,
not verified original-width performance. No estimated component/margin changed.

Uncommitted: planner,tests,consumer docs and handoff. No live NPU jobs.
