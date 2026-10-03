# Current handoff

Updated 2026-10-04 (Asia/Shanghai). **ACTIVE**: user explicitly authorized continuing under the
execution contract until the overall goal is complete. Commit/push authorized;
no per-commit pause, no subagents. A later explicit user pause takes precedence.
Repo `/home/zlong/llm/graph-execution-foundation` resolves to
`/var/tmp/zlong-graph-execution-foundation/repository`, branch
`graph-execution-foundation`. Re-entry: `git status --short --branch` and
`python scripts/status.py`. Reference repos and ObsidianVault stay read-only.

## Contract and priorities

[execution-flows](execution-flows.md) is the user-approved contract;
[ROADMAP F1–F7](ROADMAP.md) is the backlog. Deliver the 20-tide public execution/
equivalence foundation: independent CPU,mixed and device-resident online flows,
streaming/general greedy prefill, complete training and continuation. No CPU
trace/route/gradient prepass feeds candidates. Preserve int64 ordering,edge
identity,missing/zero and None/zero semantics. General PDG includes positive-delay
feedback. Packed execution,explicit bounded capacity and calibrated safe splitting.

Required performance: PDG LibTorch; TimedDAG/Settle LibTorch and PyTorch;
CPU/screened mixed/resident,both schedules,inference/complete training. FP32 main,
FP16 separate; five presets/fine controls. Three fresh processes per formal
recommendation and independent profiles. Model convergence is later work.
Preserve3000s/1.15 and historical capacities/refusals. Evidence may justify a
separately declared longer budget; never remove protection to pass. Normally
at most two measured improvement rounds/about90min active diagnosis per issue.
No unlimited device waits. Formal heavy timing serial; current overlapping
scale jobs are explicitly feasibility/diagnosis,not formal throughput.

Implementation commit → affected clean immutable qualification → separate
reviewed evidence commit/push. User contract outranks experiment skill; minimal
existing records,no new tracking infrastructure. Do not rerun unrelated passed
gates. Update this handoff atomically with scripts.durable_records.replace_text.

## Qualified baseline and newest evidence

- All ten representative family/client/schedule submatrices complete; F6 full-size
  matrix remains open. Do not mechanically repeat representative scans.
- Original resident/prefill TimedDAG/LibTorch FP32 inference passed for both models.
  B512 complete training: Add9.468B/nine cards/physicalB2×256,2170.549862239s
  ([evidence](evidence/original-b512-add-training-20261003.md));
  Attention17.521B/11 cards/physicalB1×512,5695.490452595s
  ([evidence](evidence/original-b512-attention-training-20261003.md)). Cold feasibility,
  two connected windows/one SGD update,not formal throughput. Attention's separately
  declared9000s budget preserves the old3000s refusal.
- Eager multi-device owners/complete consumers,packed transfer and admission
  qualified on their fixed sources. [Packed evidence](evidence/eager-packed-transfer-20261003.md),
  [admission/RSS evidence](evidence/eager-consumer-capacity-20261003.md).
  c686096 corrects CPU6.25% learned-storage RSS allowance and explicit NPU test
  preflight; safety margins/accelerator estimates unchanged. All historical
  underestimates/compiler-cleanup/test failures remain failed with raw records.
- **Eager FP16 implementation7b1fae5 qualified**: CPU61+55,NPU49,eleven fresh-process
  memory calibrations and separate actual trace. All1535 source hashes intact,
  eight accepted jobs terminal. Evidence committed/pushed **db7a284**
  ([report](evidence/eager-fp16-consumers-20261003.md)). FP16 payload autograd;
  FP32 loss/masters/slots,explicit static scale,no hidden retry/skip. Standalone
  master state process-local; resident FP32-adjoint policy remains distinct.
  Trace15180ops,128 AiCPU (80 BOOL/INT64 ScatterElements,48 INT64 Sort),no observed
  host tensor-compute fallback. Calibration/trace are not full-size timing.
- **CPU/mixed original-width chunk calibration c686096 passed**; evidence
  committed/pushed **2c9d34c** ([report](evidence/original-width-eager-chunks-20261003.md)).
  CPU B64/physicalB32: Add166.401850828s,Attention2559.648031075s. Mixed-A11 cards:
  AddB64/physicalB32=188.765083486s;AttentionB16/physicalB8=77.788109866s.
  All complete updates and memory checks pass. B512 phase forecasts×1.15:
  CPU Add1502.145802685s,CPU Attention23391.025824722s;
  mixed Add1727.950479047s,mixed Attention2820.987960283s. Forecasts are not results.
- CUDA-linked aarch64 build/installed client,CPU187 and relocated7 qualified on
  2c04005 ([evidence](evidence/eager-cuda-host-20261003.md)). Real CUDA/x86_64
  execution remains target-pending; [portable commands](eager-target-validation.md).

## Exact discrete tensor comparator correction

Re-entry audit reproduced a validation defect: equivalent() applied payload
float tolerances to int64/bool tensors, accepting2**53 versus2**53+1 and explicit
atol100 integer/mask differences. Fix only integer/bool comparisons to atol=rtol=0;
floating tolerance,discrete route rules and candidate computations are unchanged.
New regression includes int64 limits,nested records,dtype/shape/None checks.
Development48 checks passed; initial test collection failed on reserved fixture
name dtype,corrected to storage_dtype before tests ran. Historical route failure
is independent and remains failed. Next commit implementation,then affected clean
CPU qualification using source-matching integration-cpu-clean02 core;no full gate
repeat. All prior evidence keeps its exact source scope.

## Active jobs and next actions

`TASK=/mi/data2T/zlong/tide-execution-flows`. Every unit is
`tide-execution-flows-NAME` in background.slice,Nice10. Status/log are
`TASK/runs/NAME/{status.json,task.log}`. No live job is passed.
`eager-rss-clean01` is clean **c68609603c310f7121cb6f887afcadd019973209**;
`eager-half-clean01` is clean **7b1fae504ec779143655b9221c82d8c14a69b410**.
Frozen source under TASK/sources, installed clients/core builds under TASK/builds.

| Job NAME | Source / outcome sought | Declared bound and records |
| --- | --- | --- |
| wide-eager-cpu-add-b512-01 | **passed/exit0**; Add B512 CPU complete update1458.897225208s | construction50.482667954s; all memory checks pass; [audit passed](evidence/original-b512-eager-cpu-add-20261003.md) |
| wide-eager-mixed-b512-01 | **passed/exit0**,both actual B512 updates: Add1287.28431384s,Attention2655.242050484s | all memory checks pass,under3000s/update; [audit passed](evidence/original-b512-eager-mixed-20261004.md) |
| wide-eager-cpu-policy01 | **failed/exit1** at first policy; second never entered | Completed finite update599.905596491s,then CPU peak estimate refused |
| integration-cpu-clean01 | **failed/exit1**;9337 passed,120 failed,654 skipped,1783.33s | verification/result.json/tests.log; preserved incomplete build and old-assertion failures |
| build-integration-cpu-clean02 | **passed/exit0**; fresh full CPU core,module and CLI/check clients | TASK/builds/integration-cpu-clean02/build-manifest.json; exact C++ hash unchanged |

The mixed lease acquired physical1,2,3,4,5,6,7,8,9,11,12 (logical0..10).
CPU Add uses512GiB cap; mixed uses60GiB/card,80GiB parameter and4GiB head caps.
CPU policy probe checks its envelope plus387GiB reserved for other task memory
against the dynamic half-memory rule; total thread budget remains below half
available CPUs. Intentionally overlapping diagnostic timings carry interference;
none is a formal recommendation. No profiling/reference is inside these timers.

Launch helpers (immutable while active): `wide_eager_cpu_add_b512.py`,
`wide_eager_mixed_b512.py`, `wide_eager_cpu_policy.py` in TASK/launchers.
Each run archives helper bytes and result hashes. Original c686096 clients:
`TASK/builds/eager-rss-{cpu,npu}-clean01`. Inspect/stop only a known current task
with `systemctl --user show|stop tide-execution-flows-NAME`; never touch historical
protected work below. Follow declared timeouts,retain first failure,no blind retry.

**Integration repair:** retained integration-cpu-clean01 failed with9337 passed,
120 failed,654 skipped. The narrow build omitted CLI/check binaries; two assertions
were obsolete (eager automatic chunks and resolved owner metadata). Test-only
corrections530ae85/78e9df6 preserve valid rejection checks. Fresh complete CPU core
build-integration-cpu-clean02 passed all130 build actions; core C++ hash
ca3597e96eb7a09dc68542b7b1c0904ec39f93375fe358524bf9b6273508868c.
It does not alter the older qualified narrow build. Failed selection recheck
integration-cpu-recheck-clean02 **passed120 in272.99s,exit0**, clean78e9df6,
integration-tests-clean01. Unit inactive,empty cgroup. Original failure retained.

Full integration-cpu-clean02 **passed9458,654 optional-device skips,2022.25s,exit0**,same clean78e9df6 snapshot,
complete CPU core plus qualified c6ef224 consumer. Core/consumer bytes match
these sources; differing test/document commits are explicit. Declared2400s outer:
prior incomplete gate1783s plus restored real clients273s justifies this bound.
Terminal result and [reviewed audit](evidence/integrated-cpu-20261004.md) passed. One ATen/BLAS thread; inspect
TASK/runs/integration-cpu-clean02/verification/{result.json,tests.log} and
systemctl --user show tide-execution-flows-integration-cpu-clean02.

```bash
python /mi/data2T/zlong/tide-execution-flows/launchers/freeze_run.py --name integration-cpu-clean02 --snapshot integration-tests-clean01 --commit 78e9df6 -- timeout --signal=TERM --kill-after=10s 2400s env TIDE_ONLINE_BINARY='{base}/builds/eager-rss-training-cpu-clean01/consumer/tidegraph-online-bench' '{python}' '{source}/scripts/verify.py' --device cpu --dtype both --build-dir '{base}/builds/integration-cpu-clean02' --output-dir '{out}/verification'
```

Passed/exit0 integration-ctest-cpu-clean02 on the complete CPU build,clean78e9df6
snapshot: all12 registered FP32/FP64 CTests,serial,240s bound,explicit CPU.
This is the missing complete standalone CTest gate; no NPU qualification is
implied. Unit tide-execution-flows-integration-ctest-cpu-clean02;result in task.log
and build Testing/Temporary/LastTest.log:12/12 passed in12.61s.

Source/build/test receipts audited and evidence committed in b5de3b4.
Scale diagnostics may overlap this correctness gate; formal heavy timing waits
until these finish. No unrelated passed gate needs repetition.

## CPU training allocation correction and scale continuation

Implementation c6ef224 applies the same6.25% CPU allocation allowance to
learned+masters+gradient/optimizer storage during training,construction still
learned+masters only. Accelerator estimates,budgets,margins and mathematics
unchanged. Both development and clean builds/gates passed108 checks,no skips.
Raw build/gate names `build-eager-rss-training-cpu-{dev,clean}01` and
`eager-rss-training-cpu-{dev,clean}01`; clean snapshot eager-rss-training-clean01.

Retain original wide-eager-cpu-policy01 failure: B8/physicalB4 update completed
599.905596491s,then RSS248721317888bytes exceeded245300487792 estimate. Not OOM
or numerical failure. `wide-eager-cpu-policy-recheck01` is now **failed/exit1**:
first ATen16/workers1 case **passed425.637614531s**,observed246219792384bytes
within corrected254061046480 estimate. The second ATen1/workers16 case hit its
900s child timeout and was terminated; no result,not a speedup and no further
worker sweep. Original failure remains failed. First-case calibration and108
clean checks qualify the stricter allowance; first-case evidence audit passed; [report](evidence/cpu-training-rss-20261004.md).

**Running wide-eager-cpu-attention-b512-extended01**, clean c6ef224 and clean
eager-rss-training-cpu-clean01 client,ATen16/workers1,physicalB32×16. Reuse the
actual original-width B64/B32 chunk policy with measured phase forecast
23391.025824722s including1.15. Original3000s refusal remains intact. Declare a
separate27000s actual-update guard,27500s child and27600s outer; first failure
stops,no automatic retry. One original17.521B/B512 complete FP32 SGD update,two
connected windows,no warmup/profiler. Cold feasibility may overlap small
integration/forensic diagnostics,not formal throughput. Admitted429.436GiB CPU
peak below512GiB cap; aggregate forecast+protected historical/mixed/probe/gate
memory must fit half available memory at launch. No result yet.

```bash
python /mi/data2T/zlong/tide-execution-flows/launchers/freeze_run.py --name wide-eager-cpu-attention-b512-extended01 --snapshot eager-rss-training-clean01 --commit c6ef224 -- timeout --signal=TERM --kill-after=10s 27600s env -C '{out}' '{python}' '{base}/launchers/wide_eager_cpu_attention_b512_extended.py' --source '{source}' --out '{out}/assessment'
```

Inspect `TASK/runs/wide-eager-cpu-attention-b512-extended01/assessment/result.json`
and unit tide-execution-flows-wide-eager-cpu-attention-b512-extended01. Its long
budget is justified by measurement; it does not replace the old refusal or
certify formal performance. Keep its helper immutable once submitted.

## Route discrepancy located; strict full-size parity remains failed

`route-witness02` terminal passed/exit0 means forensic collection completed,
not equivalence. Tiny matched pair passed4windows; original matched physicalB2
CPU/resident comparisons first differ at sample17,window1,time280,region7.
CPU chooses246 (score4.005112648010254) over245 (4.005112171173096),one FP32 ULP.
Resident scores both4.005106449127197 and correctly breaks that rounded tie in
favor of245. FP64 norms of the exported FP32 proposals still rank246 above245
on both sides; resident values round to the same FP32 score. Maximum proposal
absolute difference before divergence7.703900337219238e-6. This window has
CPU2325/resident2327 events. It demonstrates a floating near-tie path split;
not proof that no other full-batch differences exist. Do not weaken exact
route checks or change stable tie policy/model/fixture to hide it.

Raw TASK/runs/route-witness02/assessment/original/observation/{witness.json,windows.jsonl}.
Forensic public pair uses exact historical26176de;three cards and CPU matchedB2,
not the later CPU physicalB32 experiment. No CPU prepass feeds resident.
All3-card leases released. `route-witness01` remains failed:tiny passed,original
setup refused an incorrectly copied512MiB default workspace ceiling. Separate
probe02 restored historical512GiB whole-program ceiling,retaining60GiB/card
admission and all queue/trace limits. Task-local probes/builds/helpers01 and02
must remain for provenance. [Reviewed diagnosis](evidence/original-add-route-witness-20261004.md) audited;strict parity stays failed.

## FP16 scale continuation and updated CUDA host gate

Both original-width FP16 pilots are terminal passed/exit0; audited sources,
binaries, result hashes and eight-card lease release. [Reviewed pilot report](evidence/original-width-eager-fp16-20261004.md).
Add B64/physicalB32:195.599333427s, forecast B512×1.15=1778.033584300s.
Attention B8/physicalB4:50.268560833s, forecast3418.386618841s still refuses3000s.
Additional Attention B16/physicalB8:71.174303463s, forecast2488.157225114s.
Original rows16 static capacity refusal63.887GiB/card remains; rows8 admits51.312GiB.

**Submitted wide-eager-half-b512-01:** clean7b1fae5/eager-half-clean01,
qualified eager-half-npu-clean01. Eight NPUs,Add physicalB32 andAttentionB8,
original B512 packets,FP16 payload/FP32 masters and loss,static scale128,
Mixed-A/LibTorch/TimedDAG/prefill,one complete SGD update/two connected windows,
ATen2/workers4. New helper wide_eager_half_b512.py; never edit it while live.
Both forecasts satisfy retained3000s/1.15; child3240s each/outer6530s,queue120s.
Cold feasibility may overlap CPU Attention; not formal timing. First failure
stops; no blind retry. Aggregate admission reserves430GiB CPU feasibility,
128GiB protected historical worker,64GiB NPU host and8GiB small work. No result yet.

```bash
python /mi/data2T/zlong/tide-execution-flows/launchers/freeze_run.py --name wide-eager-half-b512-01 --snapshot eager-half-clean01 --commit 7b1fae5 --npu --npu-count 8 --max-wait 120 -- timeout --signal=TERM --kill-after=10s 6530s env -C '{out}' '{python}' '{base}/launchers/wide_eager_half_b512.py' --source '{source}' --out '{out}/assessment'
```

**CUDA-linked update qualified**,c6ef224/eager-rss-training-clean01.
Fresh external eager-half-cuda-clean01 consumer reuses exact source-matching
CUDA core eager-cuda-clean01. Build/loader passed; eager-half-cuda-host-clean01
passed108 affected CPU checks in80.50s,no skips/deselections. Both jobs terminal,
[reviewed audit](evidence/eager-fp16-cuda-host-20261004.md). GPU/x86_64 target-pending.
Module torch-cuda/2.10.0-cu128; Python
/mi/data2T/zlong/gpu-toolchains/envs/torch2.10.0-cu128-py311/bin/python.

## Remaining acceptance and environment

CPU/mixed Add B512 share exactly1183427 candidate events,208896 selected events,
12288 outputs/cut408; losses30.5003700256/30.5003738403. The earlier resident B512
record has1183429 events/loss30.5083618164 at physicalB2,versus currentB32.
Do not claim full-size discrete/numerical parity from the feasibility passes;
this difference needs explanation before a matched formal comparison.

Still open: actual original B512 CPU Attention and FP16 Add/Attention results;
full original-scale CPU/screened mixed/resident performance matrix across required
families/clients/schedules/inference/training,appropriate continuous warmup and
measurement,three-process recommendations,separate profiles,FP16 comparisons,
final integrated support/portability audit and terminal jobs. Compilation and
refusal alone cannot close locally available work. No speedup threshold required.

Current static c686096 planning: one-update CPU B32 and mixed11 AddB32/AttentionB8
fit. One continued warmup plus one measured step admits CPU AttentionB16 and
mixed AttentionB4. These are static estimates,not observations. Three measured
plus one warmup Attention steps refuse at B1 under11-card locality; preserve
that refusal. Plans live at TASK/plans/eager-wide-capacity-{2c04005,c686096}.json.

Authorized public module `libtorch-npu/2.10.0-cann9.0.0`; Python
`/opt/miniconda/envs/ascend900-train-full-torch-npu-2.10.0-py311/bin/python`.
User authorization supersedes old private guide paths; public /usr/local driver
untouched. Preserve module PYTHONPATH,prepend frozen source/python;
TASK_QUEUE_ENABLE=0,TORCH_DEVICE_BACKEND_AUTOLOAD=0,PYTHONDONTWRITEBYTECODE=1.
Standalone NPU uses ACL_OP_INIT_MODE=0. NPU runtime cwd OUT avoids writing vendor
fusion_result.json into frozen sources. CPU build two workers,tests one ATen/
BLAS thread. Last disk:data170GiB/root11GiB; recheck before large writes.

**Never resume,stop,signal or clean historical-cpu-attention-01.** Worker2686919
is deliberately SIGSTOPped,about123.47GiB RSS; status saying running does not
mean computation. Its old timing.lock does not block current online-measurement.lock.
status.py exits1 for retained malformed build-reverse-gather-python-dev01 metadata;
this is not a current job failure. Do not rewrite that receipt.
Unexecuted finite_ranked_horizon.py and wide_attention_horizon_pilot.py remain
unqualified drafts; no48-row bound or changed original capacity is accepted.
