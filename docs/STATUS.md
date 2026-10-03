# Current handoff

Updated 2026-10-03. **ACTIVE**: user explicitly authorized continuing under the
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

## Active jobs and next actions

`TASK=/mi/data2T/zlong/tide-execution-flows`. Every unit is
`tide-execution-flows-NAME` in background.slice,Nice10. Status/log are
`TASK/runs/NAME/{status.json,task.log}`. No live job is passed.
`eager-rss-clean01` is clean **c68609603c310f7121cb6f887afcadd019973209**;
`eager-half-clean01` is clean **7b1fae504ec779143655b9221c82d8c14a69b410**.
Frozen source under TASK/sources, installed clients/core builds under TASK/builds.

| Job NAME | Source / outcome sought | Declared bound and records |
| --- | --- | --- |
| wide-eager-cpu-add-b512-01 | **passed/exit0**; Add B512 CPU complete update1458.897225208s | construction50.482667954s; all memory checks pass; audit pending |
| wide-eager-mixed-b512-01 | Running Attention; Add child passed1287.28431384s,construction43.420013183s | 3000s/update,3240s/child,6530s outer; do not mark entire job passed |
| wide-eager-cpu-policy01 | **failed/exit1** at first policy; second never entered | Completed finite update599.905596491s,then CPU peak estimate refused |
| integration-cpu-clean01 | 7b1fae5; complete CPU FP64/FP32 integration tests against reused packed-transfer CPU core | 1800s; verification/result.json and verification/tests.log; known failures below |
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

**Integration finding:** the reused narrow packed-transfer CPU build lacks
`tidegraph-smoke`, `tidegraph-kernel-check`, `tidegraph-full-check` and
`tidegraph-aggregate-check`, causing eight CLI failures. One old test still
requires eager automatic sample chunking to fail,contradicting the qualified
admission feature. Removed that obsolete assertion only; nonboolean rejection
still tested. Affected check passed1,10 deselected on the current CPU core.
Production code unchanged. Let the original full gate finish to collect all
issues; preserve it as failed. The new complete build addresses the missing
executables without changing any prior build.

Test-only correction committed/pushed as530ae8553a184c3f20135f2e7899e371cf01bfab.
The later CPU allowance fix below now also needs qualification before the next
complete integration gate; do not launch the pending retry on the older test-only source. Once the complete build passes,use the public gate:

```bash
python TASK/launchers/freeze_run.py --name integration-cpu-clean02 --snapshot integration-tests-clean01 --commit NEXT_QUALIFIED_IMPLEMENTATION_COMMIT -- timeout --signal=TERM --kill-after=10s 1800s env TIDE_ONLINE_BINARY='{base}/builds/eager-half-cpu-clean01/consumer/tidegraph-online-bench' '{python}' '{source}/scripts/verify.py' --device cpu --dtype both --build-dir '{base}/builds/integration-cpu-clean02' --output-dir '{out}/verification'
```

All production C++/consumer bytes match7b1fae5; test-only source differences must
be explicitly recorded. Check full original failure details before this retry.
After source/build/test receipts are audited,commit evidence separately.
In parallel,finish scale jobs and use measured CPU policy results for the next
bounded CPU Attention step. Formal comparisons wait until diagnostics/build/gates
finish; no extra performance sweep has been authorized beyond the contract.

## Current correction: CPU training allocation allowance

Retain `wide-eager-cpu-policy01` failure: original-width AttentionB8/physicalB4
update completed with finite loss20.681140899658203,output192,cut408; measured
RSS growth248721317888bytes exceeded245300487792 estimate (about1.39%). No OOM
or numerical failure; post-run calibration correctly failed and the second
worker policy did not start. The old recorded gate remains failed.

Working change in eager_capacity.{py,h}: apply the same6.25% CPU allocation
allowance to learned+masters+gradient/optimizer storage in training phases,
with construction still learned+masters only. Accelerator estimates,budgets,
safety margins,mathematical execution and physical update semantics unchanged.
Regression adds the measured B8/physicalB4 peak alongside earlier B4 anchors.
Docs describe the measured limitation; fresh qualification remains pending.

Passed/exit0 development build `build-eager-rss-training-cpu-dev01`, frozen dirty
`eager-rss-training-dev01` from530ae85 plus this correction,900s,two workers,
using `build_eager_consumer_core.py --backend cpu --name eager-rss-training-cpu-dev01
--core TASK/builds/packed-transfer-cpu-clean01`. Exact core bytes unchanged.
Development `eager-rss-training-cpu-dev01` passed/exit0:108 checks in75.32s,
no skips, on the exact frozen modified source and fresh client. Includes static
Python/C++ CPU/CUDA/NPU geometry parity,FP32/FP64 forced splitting,FP16 masters
and the corrected nonboolean test; full integration-cpu-clean02 core.
Implementation ready to commit; then build and rerun this affected gate on
clean eager-rss-training-clean01 plus the failed B8/physicalB4 calibration.
The correction estimates254061046480bytes versus retained248721317888bytes;
this arithmetic alone is not a fresh calibration pass. Commit implementation after affected tests; qualify clean
source and rerun the failed original-width B8 calibration with the same budgets.
Do not mechanically retry the unchanged old source or alter an active helper.

## Remaining acceptance and environment

CPU/mixed Add B512 share exactly1183427 candidate events,208896 selected events,
12288 outputs/cut408; losses30.5003700256/30.5003738403. The earlier resident B512
record has1183429 events/loss30.5083618164 at physicalB2,versus currentB32.
Do not claim full-size discrete/numerical parity from the feasibility passes;
this difference needs explanation before a matched formal comparison.

Still open: actual original B512 CPU Attention and pending mixed Attention results;
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
