# Current handoff

Updated 2026-10-03. **ACTIVE**: following a read-only handoff audit, the user
explicitly authorized this thread to resume under the existing execution contract
and finish the overall goal. Commit/push remains authorized; no per-commit pause.
A later explicit user pause overrides these next actions. No subagents.

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

## Newly closed qualifications

Packed-transfer `a785d43cfd9662c7d7262c0fdd8c59ec44729f6f` is qualified:
CPU352+64,NPU119+40,standalone CPU24/NPU36 configurations and separate actual
Attention profile. Twelve accepted jobs audited; evidence committed/pushed e654164.
[Report](evidence/eager-packed-transfer-20261003.md). Original standalone
`packed-transfer-cpp-npu-clean01` remains failed/exit124 at900s. The same binary
passed with explicit `ACL_OP_INIT_MODE=0`; CANN compiler lazy Python initialization
caused a cleanup wait. Use init0 for standalone NPU qualification/profiles;
do not disable operators or bypass finalization. All its jobs are terminal.

Eager admission implementation2c04005 plus CPU RSS/test preflight correction
**c68609603c310f7121cb6f887afcadd019973209** now qualified on clean
`TASK/sources/eager-rss-clean01`. Fresh installed CPU/NPU clients use matching
hash-verified packed-transfer cores. Eight jobs audited:
`build-eager-rss-{cpu,npu}-clean01`, `eager-rss-{cpu,npu}-clean01`,
`eager-rss-consumer-npu-clean01`, `wide-eager-cpu-rss-clean01`,
`eager-rss-calibration-npu-clean01`, `eager-rss-profile-clean01`.
All passed/exit0, inactive/empty cgroups and completed leases.
[Evidence](evidence/eager-consumer-capacity-20261003.md) committed/pushed8e6d960.

- CPU70/47 deselected; NPU capacity12/22 deselected; actual consumer40/9 deselected;
  all0 skips. Independent full records/gradients/updates and forced splitting.
- Original480-node/D2048/T12/V50304 CPU B4/physicalB2,one complete SGD update/two
  windows: Add50.543220376s,Attention254.676481222s. RSS growth97.503/213.050GiB
  <= estimates116.059/216.041GiB; construction also passes its own estimate.
  CPU RSS now charges6.25% learned storage; accelerator envelopes,all budgets and
 10%/25%+128MiB margins unchanged. These are calibration, not formal timings.
- Seven two-card calibrations pass: D256 Python/C++ Add/Attention and D2048/six-node
  C++ Add/Attention plus Python Attention. Two updates/two windows,16GiB/card.
- Separate trace:14,104 operators,128 AiCPU (80 BOOL/INT64 ScatterElements,
  48 INT64 Sort),both devices; no observed host tensor-compute fallback.

Retain failed originals: `wide-eager-cpu-pilot01` on2c04005 underestimated
Attention228749291520 >227592210112bytes after completing update; its
[report](evidence/original-width-eager-cpu-calibration-20261003.md) is unchanged.
`eager-capacity-npu-clean01` failed6/passed6 on2c04005 because CPU autograd first
use preceded explicit NPU plugin registration. Test preflight fixes initialization
order; independent CPU reference and candidate inputs are unchanged. Retain
other dev failures and all raw results; no relabelling or capacity relaxation.
Re-audit: `python TASK/launchers/eager_capacity_evidence.py c68609603c310f7121cb6f887afcadd019973209`.

## Completed CPU chunk pilot and next scale work

`wide-eager-cpu-chunk01` is now terminal passed/exit0 (09:27:52 UTC), not running.
Clean c686096, `TASK/sources/eager-rss-clean01`, installed `eager-rss-cpu-clean01`:
original D2048/T12/V50304, B64/physicalB32, two connected windows, one FP32 SGD
update, TimedDAG/LibTorch/prefill. Add166.401850828s and Attention2559.648031075s;
RSS growth110.683/232.169GiB <= estimates184.393/402.231GiB. Both full original
parameter counts, outputs, continuation and memory checks pass. Raw assessment:
`TASK/runs/wide-eager-cpu-chunk01/assessment/result.json`. The build binary hash
still matches its manifest. Concurrent bounded CPU development builds/tests
make this diagnostic feasibility/calibration, not formal throughput.

Mixed chunk pilot `wide-eager-mixed-chunk01` is terminal passed/exit0.
Add B64/physicalB32:188.765083486s; AttentionB16/physicalB8:77.788109866s.
Both complete updates pass all capacity checks. Same physical B512 phase
forecasts with1.15: Add1727.9504790471s,Attention2820.9879602833s; both<=3000s.
Raw `TASK/runs/wide-eager-mixed-chunk01/assessment/result.json`. These are
calibration observations, not formal performance; CPU B512 launch overlaps.

Running `wide-eager-mixed-b512-01`, same clean c686096/source/client,
11-card120s lease. Two original B512 fresh processes sequentially, physical
Add32/Attention8, two windows/one FP32 SGD update. Same mixed-A/2 ATen threads/
four workers,60GiB/card,80GiB parameter/4GiB head caps. Keep3000s actual step,
3240s child/6530s outer,1.15 forecast factor. First failure stops. No result yet.

```bash
python TASK/launchers/freeze_run.py --name wide-eager-mixed-b512-01 --snapshot eager-rss-clean01 --commit c68609603c310f7121cb6f887afcadd019973209 --npu --npu-count 11 --max-wait 120 -- timeout --signal=TERM --kill-after=10s 6530s env -C '{out}' ACL_OP_INIT_MODE=0 '{python}' '{base}/launchers/wide_eager_mixed_b512.py' --source '{source}' --out '{out}/assessment'
```
Unit `tide-execution-flows-wide-eager-mixed-b512-01`; status/log/queue under
`TASK/runs/wide-eager-mixed-b512-01`. May overlap CPU feasibility, never formal
throughput. Preserve terminal source/build/packet/admission hashes for audit.

## Parallel CPU Add B512 feasibility

Running `wide-eager-cpu-add-b512-01`, clean c686096/eager-rss-clean01,
using `wide_eager_cpu_add_b512.py` and the qualified eager-rss CPU client.
The completed B64/physicalB32 pilot forecasts1502.145802685s with coefficient1.15.
Run original B512/physicalB32×16, two connected windows, one FP32 SGD update,
TimedDAG/LibTorch/prefill,16 ATen threads/one worker. Preserve3000s step guard,
3240s child/3300s outer,512GiB CPU cap; check aggregate half-resource budget
including192GiB reservation for historical RSS and concurrent mixed host work.
This overlaps mixed calibration deliberately; neither is formal throughput.
No result yet. First failure stops; no blind retry.

```bash
python TASK/launchers/freeze_run.py --name wide-eager-cpu-add-b512-01 --snapshot eager-rss-clean01 --commit c68609603c310f7121cb6f887afcadd019973209 -- timeout --signal=TERM --kill-after=10s 3300s env -C '{out}' '{python}' '{base}/launchers/wide_eager_cpu_add_b512.py' --source '{source}' --out '{out}/assessment'
```
Unit `tide-execution-flows-wide-eager-cpu-add-b512-01`, records at
`TASK/runs/wide-eager-cpu-add-b512-01/{status.json,task.log,assessment/result.json}`.
Do not overlap formal timing with either diagnostic.

## Qualified eager FP16 consumers

Implementation **7b1fae504ec779143655b9221c82d8c14a69b410** committed/pushed.
Clean source `TASK/sources/eager-half-clean01` retains all1535 source hashes.
Eight accepted jobs are terminal passed/exit0, inactive/empty cgroups, completed
leases: `build-eager-half-{cpu,npu}-clean01`, `eager-half-cpu-clean01`,
`eager-half-npu-clean01`, `eager-half-cpu-regression-clean01`,
`eager-half-calibration-{cpu,npu}-clean01`, `eager-half-profile-clean01`.
CPU61/22 deselected,NPU49/no skips,CPU FP32/FP64 regression55 passed.
Four CPU/seven two-NPU fresh-process memory calibrations passed at D256 and
D2048/six-node, two updates/two windows, scale1,16GiB/device. Actual separate
Attention mixed-C profile has15180ops,128 AiCPU:80 BOOL/INT64 ScatterElements
and48 INT64 Sort; no observed host tensor-compute fallback. Not throughput.
[Reviewed evidence](evidence/eager-fp16-consumers-20261003.md) and its JSON audit
committed as db7a284; push completed. The scale jobs below have subsequently started.

Python public FP32MasterOptimizer and independent C++ consumer masters preserve
FP16 payload autograd gradients,FP32 loss/masters/slots,explicit static scale,
aliases/None/zero and update boundaries. Master memory charged; FP32 envelopes
unchanged. No C++ public checkpoint schema extension. Retained dev01 duplicate
parametrization failure is unchanged. Re-audit:
`python TASK/launchers/eager_half_evidence.py 7b1fae504ec779143655b9221c82d8c14a69b410`.
Next: finish/audit the scale jobs above; follow with bounded CPU Attention
policy measurement and continued formal comparisons.
Continue toward F6/F7 without a per-commit pause; do not rerun unaffected gates.

## Bounded CPU Attention policy diagnosis

Running `wide-eager-cpu-policy01`, clean c686096/eager-rss-clean01,
qualified CPU client. Exactly two original-width Attention B8/physicalB4×2
complete updates: (ATen16,workers1) and (ATen1,workers16), packed in both.
D2048/T12/V50304/480body/17.521B, two connected windows, one FP32 SGD update.
Each child900s, outer1850s; first failure stops, no further worker sweep. Same
loss/discrete counts checked after independent executions. Resource discovery
charges its peak estimate plus387GiB for live CPU Add,historical RSS,mixed host
work and integration. Shared diagnostic interference is explicit; no formal
speed recommendation. B512 projections remain forecasts,3000s/1.15 unchanged.

```bash
python TASK/launchers/freeze_run.py --name wide-eager-cpu-policy01 --snapshot eager-rss-clean01 --commit c68609603c310f7121cb6f887afcadd019973209 -- timeout --signal=TERM --kill-after=10s 1850s env -C '{out}' '{python}' '{base}/launchers/wide_eager_cpu_policy.py' --source '{source}' --out '{out}/assessment'
```
Unit `tide-execution-flows-wide-eager-cpu-policy01`; records at
`TASK/runs/wide-eager-cpu-policy01/{status.json,task.log,assessment/result.json}`.
No result yet. Do not confuse these policies with the earlier D128 screen.

## Integration CPU gate

Running `integration-cpu-clean01`, clean7b1fae5/eager-half-clean01.
Run public `scripts/verify.py --device cpu --dtype both` against hash-matching
`packed-transfer-cpu-clean01`, with installed `eager-half-cpu-clean01` selected
for standalone checks. One ATen/BLAS thread,1800s outer. This single full CPU
integration pass checks interactions of the qualified owner/packing/admission/
FP16 increments; it is not another development sweep. Optional device gates
remain separate. Running; early failures identified: eight CLI cells require four executables
absent from the reused component-only CPU build, and one old sample-admission
test still expects eager automatic chunking to be rejected. Keep this run and
let remaining tests collect; prepare a complete CPU build and correct only the
obsolete test expectation. It may overlap the diagnostic scale jobs;
formal timings wait until all such jobs terminate.

```bash
python TASK/launchers/freeze_run.py --name integration-cpu-clean01 --snapshot eager-half-clean01 --commit 7b1fae504ec779143655b9221c82d8c14a69b410 -- timeout --signal=TERM --kill-after=10s 1800s env TIDE_ONLINE_BINARY='{base}/builds/eager-half-cpu-clean01/consumer/tidegraph-online-bench' '{python}' '{source}/scripts/verify.py' --device cpu --dtype both --build-dir '{base}/builds/packed-transfer-cpu-clean01' --output-dir '{out}/verification'
```
Unit `tide-execution-flows-integration-cpu-clean01`; records
`TASK/runs/integration-cpu-clean01/{status.json,task.log,verification/result.json}`.
Retain any failure; fix the affected problem before repeating an appropriate gate.

## Remaining acceptance and environment

Static original B512 plans retained at `TASK/plans/eager-wide-capacity-2c04005.json`,
no Torch/model execution. Old CPU512GiB plan admitted Add/Attention physicalB32;
11×60GiB mixed admitted AddB32/AttentionB8 for one update. Three measured+one warmup
Attention steps refuse atB1 under11-card locality (54.06GiB >53.875GiB usable).
This is a planning refusal, not completion. New static c686096 plans are
`TASK/plans/eager-wide-capacity-c686096.json`: one-update CPU B32,mixed11-card
AddB32/AttentionB8 remain admitted. With one continued warmup+one measured step,
CPU AttentionB16/mixed AttentionB4 fit. No execution result is inferred. Revisit
measured capacity/placement/finite horizon with reasons; no route prepass or
relaxed protection.

Still required: actual full-size mixed/CPU, original-scale formal CPU/screened
mixed/resident comparisons across required matrix,3 fresh processes/recommendation,
independent profiles, final integrated evidence/support audit and current jobs
terminal. Real CUDA/x86_64 hardware stays target-pending;
no available local item may be declared complete from refusal/compilation alone.

`TASK=/mi/data2T/zlong/tide-execution-flows`; public authorized
`libtorch-npu/2.10.0-cann9.0.0`, Python
`/opt/miniconda/envs/ascend900-train-full-torch-npu-2.10.0-py311/bin/python`.
This authorized public stack supersedes old private guide paths. Preserve module
PYTHONPATH; prepend frozen source/python. TASK_QUEUE_ENABLE=0,
TORCH_DEVICE_BACKEND_AUTOLOAD=0. NPU children use `env -C OUT` and absolute source
paths to avoid fusion_result.json in read-only snapshots. background.slice,Nice10,
two build workers,explicit timeouts,120s device waits. Last disk:data170GiB/root11GiB;
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
