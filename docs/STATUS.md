# Current handoff

Updated 2026-10-02. **ACTIVE; user authorized continuing implementation, commits
and pushes. No pause instruction. No subagents.** Reference repositories and
ObsidianVault are read-only. Repository resolves to
/var/tmp/zlong-graph-execution-foundation/repository, branch graph-execution-foundation.
Implementationca26b47 is committed/pushed; all six immutable qualification jobs passed.
Training-record and PDG matrix evidence are being committed separately.
[execution-flows.md](execution-flows.md) is the contract;
[ROADMAP F1–F7](ROADMAP.md) is the only backlog. Overall task remains incomplete.

## Contract and working policy

Each candidate independently consumes shared inputs/parameters/initial state;
never reference events/routes/results/gradients. General online greedy accepts
legal topology/input, including positive-delay PDG feedback; natural streaming
fallback is valid. Preserve int64, stable order, parallel-edge identity,
missing/zero messages, None/zero gradients and complete continuation.
Performance: PDG LibTorch; TimedDAG/Settle LibTorch+PyTorch;
CPU/NPU × streaming/prefill × inference/complete training, five presets and fine
switches. Python resident is a native C++/CANN client. FP32 main, FP16 separate.
Training means loss/VJP/optimizer/continuation/throughput, not downstream convergence.
The contract outranks run-ml-experiments; existing minimal records suffice.
Implementation commit → immutable affected qualification → separate evidence
commit; push each. Do not rerun unchanged8,954 CPU checks or edit live-job inputs.

## Completed evidence; do not repeat

- Complete consumer admission, physical splitting and early refusal:
  clean0b1a5aa, CPU8/NPU19, eight jobs;
  [consumer capacity](evidence/consumer-capacity-20261002.md).
- Native workers/packed-source/batch-next controls: clean2222d9d,
  CPU11/NPU10, no skips,16 complete-training comparisons;
  [host execution](evidence/consumer-host-execution-20261002.md).
- TimedDAG/LibTorch/prefill five-preset screen and finite host-policy tuning:
  CPU16packed/mixed4packed selected, ATen/BLAS1, three fresh confirmations.
  Resident/CPU throughput: Add inference0.833×,training1.549×;
  Attention inference0.855×,training1.126×. CPU wins short-process totals.
  No observed AiCPU in four initial and one selected-mixed profiles;
  host launches/synchronization dominate mixed. Not full-size.
  [Host policy](evidence/representative-host-policy-20261002.md), committed8695804.
- Append-only fiber KV proposal staging: clean80dae6e, seven jobs,
  CPU4/public NPU33/four component cells. Same D128 Attention allocator peak
  2,560,387,072→2,426,168,320bytes, reduction128.001MiB;
  exact loss/output/cut/work counts, no throughput claim.
  [Fiber append](evidence/resident-fiber-append-20261002.md), committed9be8e52.
  Retained journals/live KV/reverse storage remain scale limitations.

## Active finite representative matrix

TASK=/mi/data2T/zlong/tide-execution-flows.
Source TASK/sources/fiber-append-clean01, clean80dae6e14d41614d0cdb1056bb39b57ca10d07ed.
Both PDG/LibTorch modes are complete: each20pilot +36confirmation processes.
Each confirmation uses1continued warmup+3measured steps, two windows/64tokens
per step, FP32, own heavy timings serial. CPU16packed, mixed4packed, ATen/BLAS1.
Per-step medians CPU / screened mixed / resident, seconds:

| Mode/work | CPU | Mixed | Resident | Resident/CPU throughput |
| --- | ---: | ---: | ---: | ---: |
| prefill Add inference | .132957 | B1.226391 | .159859 | .832× |
| prefill Add training | .548210 | B2.824072 | .362354 | 1.513× |
| prefill Attention inference | .250326 | A2.405648 | .273881 | .914× |
| prefill Attention training | 1.180975 | A4.173503 | 1.018009 | 1.160× |
| streaming Add inference | .139970 | A1.119666 | .419918 | .333× |
| streaming Add training | .629859 | B3.214827 | .718168 | .877× |
| streaming Attention inference | .262756 | C2.713564 | .540818 | .486× |
| streaming Attention training | 1.204491 | A5.121948 | 1.408153 | .855× |

Exact events/output/cut and loss checks pass. Resident stages per step26prefill,
80streaming. CPU wins all short-process totals. Mixed single-step selection is
only a pilot; not proof of optimality. PDG matrix report/audit are prepared in docs/evidence, not yet committed.
Task-local family_matrix_evidence.py audit passed for the two PDG submatrices.
TimedDAG/LibTorch/streaming screen and confirm01 also PASSED.

Parent unit tide-execution-flows-matrix-remaining01.service:
TASK/runs/matrix-remaining01/{status.json,task.log,queue.json,sequence.json}.
Started01:58:22UTC, total9000s, child900s, queue120s. Parent leases physical3.
The measurement boundary hold is finished. Exact recipe PID3771316 (start identity
1849248082) was SIGCONT at02:34UTC after all development/qualification gates passed.
boundary-hold.json records the transition. No sample was interrupted/discarded.
Current child at resume: matrix-timed-dag-libtorch-streaming-confirm02.
Use sequence.json for later progress. Parent total9000s timeout remains in force.

Recipe launchers/remaining_family_matrix.py will collect the finished child and
continue TDG/LibTorch/streaming confirm02/03; Settle/LibTorch both modes;
TimedDAG/Python both modes; Settle/Python both modes. Every new submatrix has
20pilot+3×12confirmations. Existing TDG/LibTorch/prefill evidence is complete.
Do not relaunch existing run names, edit helpers, or overlap heavy measurement
with our builds/gates. Low-impact implementation and evidence work may continue.

## Current implementation and development checks

Training optional diagnostic exports are separated from mandatory VJP journals.
Owner-only ContentFlow constructors request backward retention independently;
message/contribution/history/source-scale export copies are omitted with trace
false. All real VJP journals remain, public limits/checkpoint ABI/CANN kernels
unchanged. Both consumers stop forcing diagnostics for ordinary training.
Observer requests still enable complete diagnostics and their capacity allowance.
The admission envelope remains conservative and continues charging VJP journals.
This is not compact live KV, sample slicing or full-size capacity closure.

Changed: content_flow.{h,cpp},content_flow_internal.h,content_export.cpp,
training_owner.cpp,sharded_training_owner.cpp,resident_inputs.py,
resident_run.cpp,resident.py,resident_capacity.cpp; affected C++ checkers,
Python tests/helpers and three documents. New test_resident_training_records.py:
6 independent CPU autograd trajectories,8 diagnostic-mode consumer pairs,
2 observer-only consumer cases. C++ checkers switch recording at restore.

PASSED build-training-records-{standalone,python,consumer}-dev01,
frozen training-records-dev01. All affected development gates PASSED, no skips:
- training-records-components-dev01: two cards, four cells (FP16 base/cache;
  sharded FP32/FP16),900s runtime/120s queue.
- training-records-public-dev01: two cards,49affected cases,900s/120s.
- training-records-observer-dev03: final two observer cases; corrected CPU oracle
  to enable reference diagnostics. Initial dev02 failed because the reference
  trace was disabled; preserve its record. No numerical tolerance changed.

Build helper launchers/build_training_records.py verifies unchanged core/CANN
inputs and source/header identity, compiles three archive/two owner objects plus
changed checkers, then links and checks loaders. --reuse-host permits exact
qualified-source object reuse. Consumer helper build_capacity_client.py reuses
unaffected objects and rebuilds resident_run/resident_capacity.
PASSED clean training-records-clean01 atca26b47: all three builds, components4,
public51 (no skips) and allocator. Peak2,426,168,320→2,409,123,840bytes,
16.26MiB (0.70%) reduction; exact semantic work/loss/output/cut checks.
Audit launchers/training_records_evidence.py passed. Report/audit:
docs/evidence/resident-training-records-20261002.{md,json}.
Next implementation: physical sample slices for eager CPU/mixed consumers,
keeping global sample IDs, whole-batch loss, accumulated gradients/None flags,
one optimizer update and per-slice continuation. Resident sample slicing and
multi-device mixed placement remain separate follow-up. Implement in the working
tree while fixed-source matrix runs; perform heavy builds/gates only at a verified
measurement boundary. Do not alter frozen source or helpers.

## Remaining scale work

Representative inputs screening-representative-{add,attention}01:
128nodes/544edges,D128/B8/T4/V257, clear=true,8,995,632/17,384,240parameters.
These are not original wide. Full-size packets already exist:
TASK/inputs/fullsize-{add,attention}01/workload.json,480nodes/2208edges,
D2048/B512/T12/V50304,9,468,053,696 /17,521,117,376parameters.
Never reduce logical B512 and claim the wide target is complete.
CPU/mixed still use one payload device;17B FP32 cannot fit one64GiB chip.
Mixed multi-device placement, compact KV/journals and safe physical slicing
remain real work. Dense batch×local_nodes×KV capacity and coordinator-wide
journals dominate. More parameter shards alone do not solve these limits.
Sample slicing must share parameter generation, accumulate gradients/None flags,
keep global loss reduction and use one optimizer update; not implemented yet.
F7 final qualification/migration and CUDA/other CANN target tests remain pending.

## Environment and preserved history

Public module libtorch-npu/2.10.0-cann9.0.0; Python
/opt/miniconda/envs/ascend900-train-full-torch-npu-2.10.0-py311/bin/python.
User-authorized /opt stack supersedes dated personal guide. Default shell Python
cannot run Torch tests. TASK_QUEUE_ENABLE=0,TORCH_DEVICE_BACKEND_AUTOLOAD=0;
retain module PYTHONPATH,prepend snapshot/python. Ascend910_9392,16logical64GiB
chips; only leased/remapped devices. Standalone and Python-owned builds separate.
freeze_run.py uses background.slice/Nice10, builds2workers, finite queue/runtime,
runtime cwd env -C RUN. Data212GiB/root14GiB free at02:22UTC.
Use durable_records.replace_text for atomic handoff updates.
Current measurement lock TASK/online-measurement.lock.

historical-cpu-attention-01 deliberately remains SIGSTOP, holds old timing.lock;
pause.json overrides its running status. Never resume/kill/clean it. Historical
Add1.6438× means faster throughput but does not qualify current online flows.
Earlier restricted work is preserved on pushed archive/restricted-flow-20260930
at964bf628c67270200dabe55b1bca026bd403cd37; inventory TASK/restricted-flow-archive.json.
