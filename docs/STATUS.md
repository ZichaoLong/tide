# Current handoff

Updated 2026-10-03. **ACTIVE under continuous user authorization.** Continue to
F1–F7 acceptance; the user revoked per-increment pauses. Commits/pushes, validation
and context compression are not stopping conditions. Overall goal incomplete.
No subagents. Reference repositories and ObsidianVault are read-only.

Repository `/home/zlong/llm/graph-execution-foundation` resolves to
`/var/tmp/zlong-graph-execution-foundation/repository`; branch
`graph-execution-foundation`. Latest implementation **29effaed064d59b9da930ec3acec810be58b2e1a** is pushed and qualified;
fixed-source CPU45/NPU22 and standalone build passed below. Prior **c38b72e** is qualified.
The prior **219719d** accounting implementation is qualified below. Prior evidence
ce3206d qualifies b5e6345 private Attention banks; f27a4dc records actual Add B512 training.
Re-entry: `git status --short --branch`; `python scripts/status.py`.
[execution-flows](execution-flows.md) owns the contract;
[ROADMAP F1–F7](ROADMAP.md) is the sole backlog.

## Contract and operation

Each candidate independently consumes common inputs, parameters and initial
state; CPU events/routes/gradients never drive it. General online greedy accepts
legal topology/input, including positive-delay PDG feedback, and may naturally
degenerate to streaming. Preserve int64/stable order/duplicate edge identities,
missing versus zero messages, None versus zero gradients and full continuation.
Performance matrix: PDG LibTorch; TimedDAG/Settle LibTorch+PyTorch;
CPU/NPU × streaming/prefill × inference/complete training. Five presets plus fine
switches; FP32 main and FP16 separate. Training includes forward/loss/backward/
update/continuation; model convergence is a later experiment.

Implementation commit → clean fixed-source affected qualification → separate
evidence commit; push each and continue. User contract outranks experiment skill;
minimal useful records only. No unrelated passed-test reruns, unlimited queues,
blind retries, OOM search or relaxed safety gates. Formal heavy timing is serial;
profiling separate. Never stop another workload to free resources. Keep3000s,
1.15 and current capacities comparable unless evidence justifies a change.

## Qualified private-bank backend

**b5e634588405c47320ff03692bbf6cdb5af1d356** qualifies private event/full-fiber
Attention-bank borrowing during aggressive sharded training. Complete ordered
banks only; default/conservative/legacy retention, subset/mixed-head gathers and
dynamic KV/state/messages remain owned. Publication barriers and source/version/
mode/node-map guards remain; explicit tape-to-bank field mapping. Consumer
estimates and retained API budgets unchanged. Internal owner layout required all
eight owner users and three retention archive members; core/public ABI/CANN unchanged.
[Evidence](evidence/resident-attention-borrow-20261003.md).

All eight clean jobs passed/exit0,empty control groups,released leases:
`build-attention-borrow-{standalone,python,consumer}-clean01` and
`attention-borrow-{native,python,consumer,memory,profile}-clean01`.
Native six FP32/FP16 cells160trajectories/2560windows/640updates;Python16;
actualconsumer32;no skips. Source/header/options-compatible objects and unchanged
core/kernel dependencies byte-verified; freshly linked. D512/B8/physicalB2,
two windows,FP32 AdamW same-lease peaks [7508509696,6661348352] →
[7233247232,6386085888]bytes, -275262464bytes/card. Loss7.532631874084473 and
old statistics unchanged. Separate FP16 profile53176ops,zero observed AiCPU.
Allocator/profile ran after B512 ended; neither is formal throughput evidence.
Raw source `TASK/sources/attention-borrow-clean01`, builds
`TASK/builds/attention-borrow-{standalone,python,consumer}-clean01`.
Audit: `python TASK/launchers/attention_borrow_evidence.py b5e634588405c47320ff03692bbf6cdb5af1d356`.

Dev02 three builds and native/Python/consumer gates passed. Original
attention-borrow-native-dev01 stays FAILED/exit1: new whole-bank ownership
assertion detected field-order mismatch before trajectories; immutable failure
and corrected dev02 source retained. This evidence commit is documentation only;
Current uncommitted consumer accounting work is listed below.

## Original scale result and remaining gaps

**Original Add B512 complete training passed and was audited on clean26176de.**
Job `wide-add-b512-phase-admitted01`,nine devices,D2048/B512/T12/V50304,
9,468,053,696 parameters,physicalB2×256,two connected windows,one independent
FP32 SGD update. Construction69.389676023s; sample2168.898689171s +
optimizer1.651173068s =2170.549862239s <=3000. Outputs12288,events1183429,
loss30.50836181640625,cut408; maximum allocator growth43432802304bytes;
all allocator/context gates pass. Terminal exit0 at2026-10-03T00:59:36Z,
unit inactive/empty control group and nine-card lease released.
[Evidence](evidence/original-b512-add-training-20261003.md),commitf27a4dc.
Raw source/consumer:phase-timing-clean01 / phase-timing-consumer-clean01;
records `TASK/runs/wide-add-b512-phase-admitted01/assessment`.
Audit: `python TASK/launchers/wide_add_b512_evidence.py 26176de888013fda5eccfe039fa504e87c2e7e95`.

Admission preserved measured B4 sample/optimizer phases and1.15 margin:
2814.0674665565s<=3000; old whole-update projection3041.4433348544s>3000
remains a retained refusal ([diagnostic](evidence/original-width-add-phase-diagnostic-20261003.md)).
No safety/capacity gate was removed. This is one cold phase-instrumented
feasibility execution, with limited concurrent development/checks on other cards;
no allocator comparison or profile overlapped. It is not formal throughput,
a full-size CPU gradient oracle or a complete performance matrix.

All ten required representative submatrices are qualified. OriginalB512
TimedDAG/LibTorch/resident/prefill FP32 inference passed for Attention17.521B
(325.278s) and Add9.468B(278.574s). Those immutable reports and earlier storage/
gradient/optimizer improvements stay linked from ROADMAP; no retesting merely
because of this handoff. Attention original B512 training remains unexecuted.
Eager mixed multi-card remains a library gap: current model/validation assume
one payload device; real parameter/state/message placement, autograd copies,
alias ownership and continuation must be implemented in Python and native paths.

## Latest qualified capacity changes

**219719dfb15cf9e2c30f7e7facac5feb1a5c19cb** qualifies consumer private-bank
liveness accounting:aggressive multi-device declared consumers charge frozen
projection/Attention banks once in forward parameters; conservative/legacy
copies,dynamic records,API limits and safety margins unchanged. Qualified b5e6345
runtime bytes are unchanged. [Evidence](evidence/consumer-private-bank-capacity-20261003.md).
Four clean jobs passed/exit0,no skips,empty cgroups,released leases:
private-bank-capacity-{cpu,npu,calibration}-clean01 and
build-private-bank-capacity-consumer-clean01. CPU26,NPU25 (24 actual independent
CPU-referenced candidates plus one pre-allocation refusal). D512/B8/physicalB2,
two windows,FP32 AdamW:estimates24575275428/16702870820 →24006886288/16134481680;
actual peaks7233247232/6386085888,loss/statistics/continuation unchanged.
No allocation or throughput gain follows from accounting. Initial audit helper
had a stale dependency path; identity check rejected it; corrected helper passed,
original retained. Audit:
`python TASK/launchers/private_bank_capacity_evidence.py 219719dfb15cf9e2c30f7e7facac5feb1a5c19cb`.

**c38b72ebf1e66804e09dcf72f083ee4e6c1e5850** qualifies bounded static
memory-aware joint Full/state owner placement and failure-only completed-program
queue/KV-journal diagnostics. Automatic aggressive maps only,after all operator
cuts fail; at most2N moves/4096 trials,no empty owner set,strict integer memory
improvement. Nonempty explicit Python maps/conservative policies stay fixed;
empty tuples/lists mean automatic. Canonical owners,API capacities and margins
unchanged. [Evidence](evidence/consumer-memory-balance-20261003.md).

Seven clean jobs passed/exit0,no skips,empty cgroups,released leases: CPU29,
NPU36 (32 actual independent CPU-referenced candidates,three capacity refusals,
one repeated CPU interface check),three builds,D512 calibration,separate FP16
profile. D512/B8/physicalB2×4,two windows,one FP32 AdamW:one owner move/six
trials; peaks6347777024/5545201152 below estimates21159582340/13365092892.
Loss7.532632350921631 matches prior within existing FP32 tolerance;events3163,
stages104,outputs/cut/continuation match. Profile88089ops,zero observed AiCPU;
not formal throughput. Raw source memory-balance-clean01; builds
memory-balance-standalone-clean02 / memory-balance-{python,consumer}-clean01.
Audit: `python TASK/launchers/memory_balance_evidence.py c38b72ebf1e66804e09dcf72f083ee4e6c1e5850`.

Retain CPU dev01,NPU dev03 and diagnostic dev04/dev05 failures. Empty-map bug
fixed; larger B17 exposed the tiny test's2048-row KV journal. Complete tiny tests
use static all-body5712 rows,independent of events/placement; two tests still
refuse under the old cap. Original wide capacities unchanged. Standalone-clean01
build helper failed recursive object-reuse provenance; clean02 uses original
source/header/options-compatible dev05 objects. Initial audit helper expected an
absent profiler marker; actual hashed serialized success marker corrected,original
helper retained. No failed result rewritten.

Static original Attention B512 probes with current caps:12cards/physicalB2 fits
53.598GiB<53.875usable (24 moves/946 trials). With11cards,B2 remains refused;
B1 fits53.649GiB (29 moves/1740 trials). These are shape plans,NOT actual training.
Next original-width pilot must keep its owner map/operator geometry comparable
to B512; automatic placement depends on logical-batch continuation storage.
Explicit joint-map replay is implemented in the offline planner and both
consumer CLIs; development checks passed below. Qualify its exact implementation
commit before relying on pilot-to-B512 cost extrapolation.

## Next implementation and validation

1. Commit/push the reviewed owner-map evidence after its successful audit.
   No implementation edits remain. Continue to the Attention pilot below.
2. Run a bounded original-width Attention pilot at a currently available card
   count with the same explicit map/chunks as its B512 plan. Keep current caps,
   3000s and1.15; measure sample/optimizer phases,allocator and continuation.
   Actually execute B512 after valid admission; optimize from evidence if refused.
3. Implement eager mixed multi-card in the common library and both consumers,
   preserving independent CPU references, aliases, gradients and continuation;
   validate public Python/native and standalone paths with focused tests/profile.
4. Finish full-size CPU/screened-mixed/resident streaming/prefill comparisons,
   three fresh processes per recommendation and separate profiles, then F7
   integration/evidence/portable-command audit. CUDA/new stack cells require
   explicit target-machine validation; never claim local execution without hardware.

**29effaed064d59b9da930ec3acec810be58b2e1a** fixed joint-map CLI is qualified:
[CPU45/NPU22 and build evidence](evidence/consumer-owner-map-20261003.md).
All three clean jobs passed/exit0,no skips,empty cgroups,released leases;
NPU finished02:49:25Z. Twenty complete independent CPU-referenced candidates,
twelve explicit/eight automatic,plus two invalid-map refusals. Source
owner-map-clean01;build owner-map-consumer-clean01;unchanged qualified
memory-balance-standalone-clean02 backend. Audit passed:
`python TASK/launchers/owner_map_evidence.py 29effaed064d59b9da930ec3acec810be58b2e1a`.
Retain owner-map-cpu-dev01 failure:metadata test used unregistered npu device;
corrected resolved-index fixture and actual device CLI gates passed.

Next task `wide-attention-owner-pilot01` uses clean29effae/owner-map-clean01 and
owner-map-consumer-clean01,11devices/physicalB1,B4,two windows,one FP32 SGD
update. Helper TASK/launchers/wide_attention_owner_pilot.py --source SOURCE
--build BUILD --out TASK/runs/wide-attention-owner-pilot01/assessment
--devices 11 --sample-rows 1. B512-fixed map/operator maxima also fit B4;
static estimated peaks49.896/53.649GiB within53.875GiB usable. This is not a
runtime result. Wait120s,child900s,8threads,online-measurement.lock; preserve
3000s/1.15/current caps. If measured phase forecast passes,run a separate
bounded B512 via wide_attention_b512_admitted.py; otherwise inspect the actual
cost and optimize. Eleven cards currently available after qualification; the
lease helper rechecks inventory. No current mainline job live yet.
The historical task remains deliberately stopped and protected.
New long jobs: frozen source,background.slice,Nice10,two build workers,explicit
child timeout,lease wait120s; unit `tide-execution-flows-NAME.service`,persistent
logs/status under `TASK/runs/NAME`. Confirm terminal records,workload exit,result
hashes and empty cgroups; a collected unit's default success is insufficient.

## Environment and protected state

`TASK=/mi/data2T/zlong/tide-execution-flows`; public module
`libtorch-npu/2.10.0-cann9.0.0`; Python
`/opt/miniconda/envs/ascend900-train-full-torch-npu-2.10.0-py311/bin/python`.
User-authorized /opt stack supersedes the old private guide.
`TASK_QUEUE_ENABLE=0 TORCH_DEVICE_BACKEND_AUTOLOAD=0`; preserve module PYTHONPATH,
prepend frozen source/python; lease/remap devices; runtime `env -C {out}`.
Last free disk:data157GiB/root11GiB; recheck before large writes.
Core dependencies:placement-{cpu,npu,npu-python}-clean01. Preserve cited consumers,
failed reproducers and immutable qualification artifacts.

**Protect deliberately SIGSTOPped historical-cpu-attention-01**: never resume,
stop or clean it. Its historical running record and held old timing.lock do not
block current work, which uses online-measurement.lock. Historical1.6438× meant
faster throughput in a restricted flow, not current online evidence.
Archive:archive/restricted-flow-20260930 at964bf628c67270200dabe55b1bca026bd403cd37.
Historical build-reverse-gather-python-dev01 metadata inconsistency remains
visible; status.py exits1 for that record, not a current failure. Do not rewrite it.
