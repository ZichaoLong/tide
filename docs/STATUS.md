# Current handoff

Updated 2026-10-03. **ACTIVE**: the user cancelled the old goal and explicitly
authorized its replacement; that goal is active. Commit/push, evidence and context
compression do not stop execution. A later explicit user pause overrides these
next actions; repository text never authorizes resuming after a pause.

Repo `/home/zlong/llm/graph-execution-foundation` resolves to
`/var/tmp/zlong-graph-execution-foundation/repository`; branch
`graph-execution-foundation`. Re-entry: `git status --short --branch` and
`python scripts/status.py`. [execution-flows](execution-flows.md) owns the contract;
[ROADMAP F1–F7](ROADMAP.md) owns acceptance/backlog. Overall acceptance is open.
No subagents; reference repositories and ObsidianVault are read-only.

## Contract and priorities

Deliver the 20-tide common calling/equivalence foundation: independent CPU,
mixed multi-device and NPU resident streaming/online-greedy prefill, complete
training/continuation/correctness and formal performance comparisons. Candidates
independently consume common inputs/parameters/initial state; no CPU event/route/
gradient prepass. Preserve exact message identity, stable integer order,
parallel edges, missing/zero and None/zero connectivity. General PDG includes
positive-delay feedback; legal greedy may naturally become streaming.

Required performance: PDG LibTorch; TimedDAG/Settle LibTorch and PyTorch;
CPU/screened mixed/resident × both schedules × inference/complete training.
FP32 main, FP16 separate; five presets and fine controls. Prefer calibrated
aggressive-safe splitting. Three fresh processes per recommendation, independent
profiles. Model convergence is later work. No speedup threshold was specified.

Single performance issue: normally two evidence-backed improvement rounds and
about90min active diagnosis; launched tasks keep their declared timeout. Preserve
3000s/1.15/current capacities and refusals. Measurements may justify new explicit
budgets or models; never remove protection to pass. Advance independent work
when cards/performance are blocked. No unlimited queues or blind retry.

Implementation commit → clean fixed-source affected qualification → separate
reviewed evidence commit/push. Minimal useful records; user contract outranks
experiment skill. Do not rerun unrelated passed gates. Formal heavy timing is
serial and separate from profiles. Update this file atomically with
`scripts.durable_records.replace_text` and read back.

## Completed mainline evidence

- Original Add B512 complete FP32 training: clean26176de,9.468B parameters,
  nine cards,physicalB2×256,two connected windows,one SGD update2170.549862239s.
  [Evidence](evidence/original-b512-add-training-20261003.md).
- **Original Attention B512 complete FP32 training now passed**: clean29effae,
  17.521B parameters,11 cards,physicalB1×512,two connected windows,one SGD update
  5695.490452595s. Construction164.822193006s;sample5692.808817143s;
  optimizer2.681635452s;outputs12288/cut408/loss21.380962371826172. All allocator
  and saved-context checks pass. New9000s budget passes; old3000s refusal and1.15
  remain unchanged. [Evidence](evidence/original-b512-attention-training-20261003.md),
  committed/pushed338f104. Both full-size results are cold feasibility, not formal
  throughput or full-size CPU gradient oracles.
- All ten representative performance submatrices and original Add/Attention
  resident/prefill FP32 inference were already qualified. Do not repeat them
  without an affected change or unresolved concern.
- Eager owner/control/checkpoint implementation55c3960 qualified CPU310/NPU87,
  standalone CPU24/NPU36 configurations and a separate two-device profile.
  [Evidence](evidence/eager-payload-owners-20261003.md).
- Actual mixed consumers e5d91d7 qualified CPU73/NPU40, no skips; both memories,
  all clients/families/schedules,mixedA/B/C,two AdamW updates/two connected windows,
  independent CPU complete records/gradients/updates and separate actual profile.
  [Evidence](evidence/eager-consumer-owners-20261003.md).
- F7 CUDA-linked aarch64 core/installed client, CPU187 and no-Git relocated7
  qualified on2c04005. Four terminal jobs and all1522 source hashes audited.
  [Evidence](evidence/eager-cuda-host-20261003.md), committed/pushedaac8c46.
  GPU execution remains unverified; [target commands](eager-target-validation.md).

## Packed-transfer qualification (a785d43)

Exact source `TASK/sources/packed-transfer-clean01`,
`a785d43cfd9662c7d7262c0fdd8c59ec44729f6f`. Completed-message copies only,
8MiB caps, retained missing/zero/frozen/alias VJPs; eager metadata stays host work.
Five clean builds passed: `build-packed-transfer-{cpu,npu-python,npu-standalone}-clean01`
and `build-packed-consumer-{cpu,npu}-clean01`. Source/hash-verified reused cores,
changed objects rebuilt and outputs freshly linked.
CPU352,standalone CPU12 configs per FP64/FP32,actual-consumer64/deselected8 passed.
NPU Python/native `packed-transfer-npu-clean01` passed119/no skips/exit0.

**Standalone runtime finding:** `packed-transfer-cpp-npu-clean01` terminated
failed/exit124 at its900s timeout; lease released,cgroup empty. It hung in CANN compiler cleanup. Saved
`hang-backtrace.log` shows `te::fusion::HandleManager::Finalize` → `Py_FinalizeEx`
→ threading lock; autograd workers idle. Installed header documents
ACL_OP_INIT_MODE0=eager,1=lazy,2=disabled. Retained older vendor source explains
A2/A3 default1. Same binary with explicit `ACL_OP_INIT_MODE=0` in new bounded
`packed-transfer-cpp-npu-init01` **passed36 configurations/72 updates/144 connected
windows/exit0 in31s**, completed lease. This changes compiler initialization,
not operators/semantics. Preserve the original failure; do not bypass finalize.
Use explicit eager init for subsequent standalone NPU qualification/profiles.

`packed-consumer-npu-clean01` passed40/deselected9/no skips/exit0, completed lease.
Exacta785d43, two-card lease120s,
900s bound, ACL_OP_INIT_MODE=0, tests/test_online_eager_owners.py --dtype float32
-k 'per_owner_constants or actual_two_device or unified_cli_owner'; expected40.
NPU Python core `packed-transfer-npu-python-clean01`, installed client
`packed-consumer-npu-clean01/consumer/tidegraph-online-bench`.
`packed-consumer-profile-clean01` passed/exit0 with completed lease; used
`profile_eager_consumer.py --source SOURCE --build TASK/builds/packed-consumer-npu-clean01
--out OUT/profile`, separate profile, two cards/120s queue/600s command, env init0.
Terminal audit passed for all12 accepted jobs/source/archive/loaders/profile,
retaining original timeout. [Reviewed packed evidence](evidence/eager-packed-transfer-20261003.md)
records the result and compiler initialization requirement. No phase is a pause.

## Eager admission and CPU recalibration

Committed/pushed implementation2c04005255b3d0b674cef302894bb9a5d0f0378e:
finite DAG static traffic bounds, complete-run memory envelope, preallocation
admission, bounded sample halving and post-run rejection records. Generic online
feedback support is unchanged. [Contract](eager-consumer-capacity.md).
Frozen `TASK/sources/eager-capacity-clean01`. Fresh installed CPU/NPU builds
`eager-capacity-{cpu,npu}-clean01` passed using byte-verified packed-transfer cores.
Clean CPU68/deselected47/no skips passed. Four fresh D256/B8/T4/V4096 CPU
calibrations passed (Add/Attention × Python/LibTorch,two SGD updates/two windows).
`calibrate_eager_capacity.py`, records `runs/eager-capacity-calibration-cpu-clean01`.
These calibrate RSS, not formal throughput. Preserve dev01 JSON-array and dev02
packet-hash failures; dev03 fixed both without weakening identity validation.

`wide-eager-cpu-pilot01` is terminal **failed/exit1**, empty cgroup:
original480-node/D2048/T12/V50304 parameters,B4/physicalB2,two connected windows,
one SGD update,16 ATen threads/one worker;512GiB host/80GiB parameter/4GiB head caps,
900s per child/1850s outer. Add59.403475322s,RSS growth104698310656bytes passed.
Attention256.699170879s,outputs96/cut408/finite loss20.078292847, but observed
228749291520bytes > estimate227592210112bytes. Both construction RSS observations
also exceed their individual phase estimates. [Retained failure](evidence/original-width-eager-cpu-calibration-20261003.md),
committed2c38a24/compact report0a2bd28; full raw results retained. Do not rerun B512 CPU
until calibrated. No OOM or budget relaxation occurred.

**Uncommitted next implementation:** CPU-only RSS retained-allocation allowance,
6.25% of learned bytes at each phase (observed construction residual up to4.3%);
accelerator allocated-byte envelope and all budgets/margins remain unchanged.
Python/C++ plans record observation_counter and resolve it from actual backend,
including CPU auto resolution; offline preset follows same rule. Added fixed
observed-phase anchors and expanded no-hardware static backend parity.
Build `build-eager-rss-cpu-dev01` passed/exit0; gate `eager-rss-cpu-dev01`
passed70/deselected47/no skips/exit0, frozen dirty snapshot
`eager-rss-dev01`, via
`build_eager_consumer_core.py --backend cpu --name eager-rss-cpu-dev01
--core TASK/builds/packed-transfer-cpu-clean01`; gate test_eager_capacity.py,
test_consumer_traffic_bounds.py,test_online_consumer_memory.py,test_online_eager_owners.py
with the existing CPU selection. Commit after affected tests; then freeze/build/
qualify and rerun original-width fresh CPU calibration with a **new helper/output**.
Original audited helper/output must remain unchanged. Formal timing is separate.

Packed runtime/profile jobs passed with retained lazy-init timeout; audit passed.
`eager-capacity-npu-clean01` failed6/passed6/deselected20 on exact2c04005:
Python/native three training cases initialized CPU autograd before explicit NPU
plugin registration; engine device-ready-queue assertion. Standalone and
inference cases passed. Preserve raw log/terminal failure. Updated test now
preflights its explicitly selected backend before independent CPU backward.
Development build`build-eager-rss-npu-dev02` passed/exit0 from frozen
`eager-rss-dev02`; launching `eager-rss-npu-dev02` for its12 affected device cases,
22 deselected expected,900s bound/two-card120s queue,explicit ACL_OP_INIT_MODE=0.
CPU70 already passed prior preflight-only test correction. Then finish
actual-consumer40, finite calibration7 (D256 Python/C++ both memories; D2048/six-node
C++ both and Python Attention), separate actual profile. `calibrate_eager_capacity.py`
uses480s/child,7 cases/3600s outer,16GiB/card/4GiB head,two cards, first failure stop.
Use existing packed-transfer NPU Python core and new matching installed NPU client,
ACL_OP_INIT_MODE=0. Qualify the final corrected implementation after commit, rather
than rerunning the superseded2c04005 test harness. Prepared `eager_capacity_evidence.py`
needs final source/job names and retained failure update; original-width fresh CPU
recalibration replaces redundant small CPU calibration. Do not claim original
NPU full-scale calibration from the seven smaller cases.

## Remaining acceptance and environment

Static original B512 plans retained at `TASK/plans/eager-wide-capacity-2c04005.json`,
no Torch/model execution. Old CPU512GiB plan admitted Add/Attention physicalB32;
11×60GiB mixed admitted AddB32/AttentionB8 for one update. Three measured+one warmup
Attention steps refuse atB1 under11-card locality (54.06GiB >53.875GiB usable).
This is a planning refusal, not completion. Revisit measured capacity/placement/
finite horizon with explicit reasons; no route prepass or relaxed protection.

Still required: actual full-size mixed/CPU, original-scale formal CPU/screened
mixed/resident comparisons across required matrix,3 fresh processes/recommendation,
independent profiles, final integrated evidence/support audit and current jobs
terminal. Eager FP16 training and real CUDA/x86_64 hardware stay separately open;
no available local item may be declared complete from refusal/compilation alone.

`TASK=/mi/data2T/zlong/tide-execution-flows`; public authorized
`libtorch-npu/2.10.0-cann9.0.0`, Python
`/opt/miniconda/envs/ascend900-train-full-torch-npu-2.10.0-py311/bin/python`.
This authorized public stack supersedes old private guide paths. Preserve module
PYTHONPATH; prepend frozen source/python. TASK_QUEUE_ENABLE=0,
TORCH_DEVICE_BACKEND_AUTOLOAD=0. NPU children use `env -C OUT` and absolute source
paths to avoid fusion_result.json in read-only snapshots. background.slice,Nice10,
two build workers,explicit timeouts,120s device waits. Last disk:data155GiB/root11GiB;
recheck before large writes. Raw job records `TASK/runs/NAME/{status.json,task.log}`,
units `tide-execution-flows-NAME`; queue/profile records where applicable.

**Never resume, stop or clean deliberately SIGSTOPped historical-cpu-attention-01.**
Its old timing.lock does not block `online-measurement.lock` used by current work.
No signal/cleanup is authorized for that historical task. Historical restricted
flow archive964bf628c67270200dabe55b1bca026bd403cd37 remains scoped.
`status.py` may exit1 for retained `build-reverse-gather-python-dev01` inconsistent
metadata; do not rewrite it. Unexecuted `finite_ranked_horizon.py` and
`wide_attention_horizon_pilot.py` are unqualified drafts; do not use their48-row
bound or alter audited original pilot helpers without review.
