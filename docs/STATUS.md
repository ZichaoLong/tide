# Current handoff

Updated 2026-10-03. **ACTIVE under continuous user authorization.** Continue to
F1–F7 acceptance; the user revoked per-increment pauses. Commits/pushes, validation
and context compression are not stopping conditions. Overall goal incomplete.
No subagents. Reference repositories and ObsidianVault are read-only.

Repository `/home/zlong/llm/graph-execution-foundation` resolves to
`/var/tmp/zlong-graph-execution-foundation/repository`; branch
`graph-execution-foundation`. Latest implementation **219719d** is pushed and qualified below. Prior evidence
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

## Latest qualified accounting and current prototype

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

Uncommitted next-work **capacity_balance.py** is an unintegrated Python prototype:
bounded static owner moves from complete-envelope costs; at most2×nodes moves,
4096 envelope trials,strict memory-score improvement and physical-edge locality
for ties. Shape-equivalent donor nodes share a trial; integer budgets only, no
model values/events/routes. A draft static original-Attention12-card/B2 probe
moves24 owners in946 trials,63.923→53.598GiB maximum estimate; this is NOT a
runtime/admission/support result. C++ counterpart,planner integration,tests and
actual allocator/trajectory validation still required. No full-size Attention job queued.
This evidence commit contains docs only; prototype remains uncommitted.

## Next implementation and validation

1. Finish generic bounded memory-aware placement using the prototype: independently
   implement C++ equivalent and integrate only after the existing aggressive
   initial placement/operator cuts cannot fit. Preserve explicit Python owner
   maps; static integer cost improvements and finite search,all safety margins.
   Start from capacity.py/.h,capacity_record.h and tests/test_consumer_capacity.py;
   keep Python/C++ plan parity and actual independent CPU comparisons. Do not
   claim prototype execution or weaken admission to obtain a pass.
2. Calibrate the new placement on a bounded actual consumer,then original-width
   Attention under comparable capacities/3000s/1.15; actually execute B512 after
   valid memory/cost admission. No full-size training claim from static fitting.
3. Implement eager mixed multi-card in the common library and both consumers,
   preserving independent CPU references, aliases, gradients and continuation;
   validate public Python/native and standalone paths with focused tests/profile.
4. Finish full-size CPU/screened-mixed/resident streaming/prefill comparisons,
   three fresh processes per recommendation and separate profiles, then F7
   integration/evidence/portable-command audit. CUDA/new stack cells require
   explicit target-machine validation; never claim local execution without hardware.

All current compute jobs are terminal. Next implementation is the uncommitted
placement prototype above; the protected historical task stays separately stopped.
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
Last free disk:data172GiB/root13GiB; recheck before large writes.
Core dependencies:placement-{cpu,npu,npu-python}-clean01. Preserve cited consumers,
failed reproducers and immutable qualification artifacts.

**Protect deliberately SIGSTOPped historical-cpu-attention-01**: never resume,
stop or clean it. Its historical running record and held old timing.lock do not
block current work, which uses online-measurement.lock. Historical1.6438× meant
faster throughput in a restricted flow, not current online evidence.
Archive:archive/restricted-flow-20260930 at964bf628c67270200dabe55b1bca026bd403cd37.
Historical build-reverse-gather-python-dev01 metadata inconsistency remains
visible; status.py exits1 for that record, not a current failure. Do not rewrite it.
