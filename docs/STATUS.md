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
[New evidence](evidence/eager-consumer-capacity-20261003.md) ready to commit.

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

## Next finite scale increment

Commit/push the reviewed capacity evidence, then launch **one** heavy pilot at a
time, using already-qualified clean c686096 and `wide_eager_chunk_pilot.py`.
This new helper is prepared, not executed yet; immutable original CPU calibration
helper/output stay unchanged. It validates full original parameter counts,
output/cut/finite loss, all observed memory bounds, two physical chunks/two
connected windows/one SGD update, and records sample-work/optimizer timing.

CPU first: `wide-eager-cpu-chunk01`, build `eager-rss-cpu-clean01`,device cpu,
Add/Attention B64/physicalB32,16 ATen threads/one worker,512GiB host/80GiB learned/
4GiB head. Each child3000s,outer6050s,first failure stops. Command shape:
`freeze_run.py --name wide-eager-cpu-chunk01 --snapshot eager-rss-clean01 --commit
c68609603c310f7121cb6f887afcadd019973209 -- timeout --signal=TERM --kill-after=10s
6050s env -C {out} {python} {base}/launchers/wide_eager_chunk_pilot.py --source
{source} --build {base}/builds/eager-rss-cpu-clean01 --device cpu --out {out}/assessment`.
Helper takes nonblocking `online-measurement.lock`; no indefinite lock wait.

Then mixed: same helper/device npu:0/build `eager-rss-npu-clean01`,11-card120s
lease,60GiB/card,2 ATen threads/four workers,mixed-A (representative screen choice),
Add B64/physicalB32,Attention B16/physicalB8, same3000s child/6050s outer/init0.
No NPU job is submitted yet. If cards unavailable, retain queue refusal and move
to independent implementation (eager FP16 master consumer remains open),not an
unbounded wait. Use measured cost and unchanged1.15 to set explicit B512 budget;
keep old3000s refusal and do not infer B512 completion from a forecast.

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
