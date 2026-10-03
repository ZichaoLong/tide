# Current handoff

Updated 2026-10-03. **PAUSED after this round under the latest user authorization
to pause after committing and receive a progress summary.** Overall F1–F7 is
incomplete. All current development and qualification jobs are terminal; no new
B512 pilot or mixed-multi-card work was launched in this round. Resume on user
instruction. No subagents; reference repositories and ObsidianVault are read-only.

Repository `/home/zlong/llm/graph-execution-foundation` resolves to
`/var/tmp/zlong-graph-execution-foundation/repository`; branch
`graph-execution-foundation`. Latest implementation
**af137df13198b4e0bbd5713249369c691204e8e2** is pushed and qualified below.
This documentation increment contains its separate reviewed evidence.
Re-entry: `git status --short --branch`; `python scripts/status.py`.
[execution-flows](execution-flows.md) owns the contract;
[ROADMAP F1–F7](ROADMAP.md) is the sole backlog.

## Contract and operation

Each candidate independently consumes common inputs, parameters and initial
state; CPU events/routes/gradients never drive it. General online greedy accepts
legal topology/input, including positive-delay PDG feedback, and may naturally
degenerate to streaming. Preserve int64/stable order/duplicate edge identities,
missing versus zero messages, None versus zero gradients and complete continuation.
Performance matrix: PDG LibTorch; TimedDAG/Settle LibTorch+PyTorch;
CPU/mixed/resident × streaming/prefill × inference/complete training. Five presets
plus fine switches; FP32 main and FP16 separate. Training includes forward/loss/
backward/update/continuation; model convergence belongs to later experiments.

Implementation commit → clean fixed-source affected qualification → separate
evidence commit. Commit/push is authorized. User contract outranks experiment
skill; minimal useful records only. No unrelated passed-test reruns, unlimited
queues, blind retries, OOM search or relaxed safety gates. Formal heavy timing
is serial; profiling separate. Never stop another workload to free resources.
Keep3000s,1.15 and current capacities comparable unless evidence justifies a change.

## This round: journal accounting qualified

**af137df** updates Python/C++ consumer admission only: aggressive multi-device
Attention training charges two live FP32 KV journal banks and one retained bank
per connected window. Each bank is `rows*(5*8+(2W+1)*4)+4096` bytes. Source review
confirms one compiled reusable proposal per state owner and deduplicated retained
journal clones. Conservative/single-device/inference bounds, capacities, other
storage allowances and margins remain. Runtime/core/CANN bytes are unchanged.
[Evidence](evidence/consumer-journal-capacity-20261003.md).

Four clean jobs passed/exit0 with empty cgroups and released device leases:

- `journal-capacity-cpu-clean01`:38 passed,no skips; static inventory/lifetime,
  C++/Python parity, FP32/FP16 and one/two/four windows, legacy policies.
- `build-journal-capacity-consumer-clean01`:installed standalone client; compatible
  objects reused after source/header/options checks, fresh link and loader audit.
- `journal-capacity-npu-clean01`:17 passed,no skips; sixteen native/standalone,
  Add/Attention, FP32/FP16, operator/owner-pressure candidates independently compare
  two complete updates/two connected windows against CPU; one allocation refusal.
- `journal-capacity-calibration-clean01`:D512/B8/T4/V257,128 body nodes, two devices,
  physicalB2×4,two connected windows,one FP32 AdamW update. Replay the prior actual
  owner map/chunks/capacities/budget. Estimates21159582340/13365092892 →
  17375287940/9580798492bytes; actual peaks6347777024/5545201152 unchanged.
  Loss7.532632350921631,outputs64,cut80,events3163,stages104,all statistics and
  continuation-pool records match. This is allocation calibration,not throughput
  or a runtime-memory saving. No new profile for unchanged execution bytes.

Development `journal-capacity-{cpu,npu}-dev01` and
`build-journal-capacity-consumer-dev01` also passed before implementation commit.
Clean source `TASK/sources/journal-capacity-clean01`; client
`TASK/builds/journal-capacity-consumer-clean01`; unchanged qualified backend
`memory-balance-standalone-clean02` / `memory-balance-python-clean01` at c38b72e.
Core dependencies `placement-{cpu,npu,npu-python}-clean01` remain byte-verified.
Raw logs/status/results: `TASK/runs/NAME`; unit prefix `tide-execution-flows-`.
Audit passed:
`python TASK/launchers/journal_capacity_evidence.py af137df13198b4e0bbd5713249369c691204e8e2`.
Initial calibration-helper static preflight used invalid `owners` keyword and
failed before device work. Original helper/receipt retained under
`TASK/launchers/journal_capacity_calibration_preflight_*`; corrected full/state
map helper is copied and hashed in the successful run. No failed run rewritten.

## Established scale results and preceding work

**Original Add B512 complete training passed on clean26176de.** Nine devices,
D2048/B512/T12/V50304,9,468,053,696 parameters,physicalB2×256,two connected windows,
one FP32 SGD update. Construction69.389676023s; sample2168.898689171s +optimizer
1.651173068s =2170.549862239s <=3000. Loss30.50836181640625,outputs12288,
events1183429,cut408; maximum allocator growth43432802304bytes; all memory/context
checks pass. This is cold feasibility evidence,not formal throughput or a full-
size CPU gradient oracle. [Evidence](evidence/original-b512-add-training-20261003.md).
Job `wide-add-b512-phase-admitted01`, source/client `phase-timing-clean01` /
`phase-timing-consumer-clean01`; evidence f27a4dc. Old whole-step projection
3041.4433348544s>3000 remains a refusal; phase admission2814.0674665565s retained
1.15 and all guards. No gate was removed.

All ten required representative submatrices are qualified under ROADMAP F6.
Original B512 TimedDAG/LibTorch/resident/prefill FP32 inference passed for
Attention17.521B and Add9.468B. Those qualifications were not repeated here.

**Original-width Attention B4 completed; B512 was refused by measured cost.**
Fixed joint-map CLI/offline replay is qualified on clean29effae,CPU45/NPU22/build
([evidence](evidence/consumer-owner-map-20261003.md),140d698). Job
`wide-attention-owner-pilot01` uses that clean source/client,11devices,
D2048/T12/V50304,17,521,117,376 parameters,B4/physicalB1×4,two windows,one FP32 SGD
update,original capacities and the same owner map/chunks as B512. Construction
282.182195511s; sample47.781389243s +optimizer2.636987134s=50.418376377s.
Loss20.07830810546875,outputs96,events9256,stages224,cut408; observed allocator
peaks42.839–43.996GB covered by estimates. Phase forecast7036.453031774s>3000
at1.15; B512 NOT executed. Terminal/lease/hash audit passed; evidence3ef8ec8
([report](evidence/original-width-attention-owner-diagnostic-20261003.md)).
Do not modify audited `TASK/launchers/wide_attention_owner_pilot.py`; new configs
need a new helper. `wide_attention_b512_admitted.py` remains an unexecuted helper.

Prior private-bank/runtime/storage and placement qualifications remain indexed
in ROADMAP: b5e6345,219719d,c38b72e and29effae. Preserve their cited sources/builds,
profiles and failure reproducers. Accounting changes alone do not complete F6.

## Next work after resumption

1. Prove the original rank-aligned packet's finite KV upper bound before changing
   its experiment capacity: node rank is `ranks[node_regions[v]]`; every delay
   equals target rank minus source rank. Empty initial state,no warmup,two windows
   means24 input positions and at most5 incoming atoms per position:120 rows/node.
   A128-row cap can be safe for that finite horizon if verified; this is not a
   bound for arbitrary topology,initial KV or unlimited continuation. Do not
   truncate KV or generalize the finite argument into the scheduler.
2. Replan with qualified af137df. Old29effae/11-card/KV128 static probes admitted
   physicalB2 but refused B4; the new B4 plan has not been evaluated. If admitted,
   use logicalB8/physicalB4×2 pilot with the same B512 map/chunks and complete SGD;
   two sample groups exercise accumulation. Keep3000s/1.15 and record all phases,
   allocator/context checks. A single-group B4 pilot is insufficient for the same
   extrapolation. Full B512 follows only valid admission. Keep old B1 cost refusal.
   Construction282s informs a separate process timeout; it does not relax the
   3000s complete-update threshold. No new wide runtime was started this round.
3. Implement eager mixed multi-card in common Python/native/standalone paths:
   real parameter/state/message ownership, autograd copies, aliases, region
   control and owner-aware checkpoint restoration. Existing library/validation
   assume one payload device. Packed cross-card training must preserve None/zero
   under isolated roots; naive cat/stack can create connected-zero gradients.
   Historical accelerator-scale transfer code is reference only,not a substitute
   for the general public scheduler. No mixed-multi-card code has been changed.
4. Finish original-scale CPU/screened-mixed/resident inference/complete-training,
   both schedules and required graph/language cells,three fresh processes per
   recommendation and separate profiling. Then F7 integration,evidence/support
   audit and portable build/validation commands. CUDA and other hardware/stacks
   remain target-machine-pending without real evidence.

## Environment and protected state

`TASK=/mi/data2T/zlong/tide-execution-flows`; public module
`libtorch-npu/2.10.0-cann9.0.0`; Python
`/opt/miniconda/envs/ascend900-train-full-torch-npu-2.10.0-py311/bin/python`.
User-authorized /opt stack supersedes the old private guide.
`TASK_QUEUE_ENABLE=0 TORCH_DEVICE_BACKEND_AUTOLOAD=0`; preserve module PYTHONPATH,
prepend frozen source/python; lease/remap devices; runtime `env -C {out}`.
Last free disk:data157GiB/root11GiB; recheck before major writes. Long jobs use
`TASK/launchers/freeze_run.py`,frozen source,background.slice,Nice10,two build
workers,explicit timeout and120s device lease wait. Never call a live job passed.

**Protect deliberately SIGSTOPped historical-cpu-attention-01**: never resume,
stop or clean it. Its historical running record and held old `timing.lock` do
not block current work,which uses `online-measurement.lock`. The latest read-only
inspection leaves its service and cgroup intact. Historical1.6438× was a restricted
flow result,not general-online evidence. Archive:
`archive/restricted-flow-20260930` at964bf628c67270200dabe55b1bca026bd403cd37.
Historical `build-reverse-gather-python-dev01` metadata inconsistency remains
visible; `status.py` exits1 for that record,not a current failure. Do not rewrite it.
