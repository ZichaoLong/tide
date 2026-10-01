# Current handoff

Updated 2026-10-02. **ACTIVE: user resumed and accepted overhead reduction.**
Continue the overall goal; commit/push authorized. No current pause instruction.
No subagents. Reference repositories and ObsidianVault are read-only.
Repository `/home/zlong/llm/graph-execution-foundation` resolves to
`/var/tmp/zlong-graph-execution-foundation/repository`; branch `graph-execution-foundation`.
[execution-flows.md](execution-flows.md) is the contract; [ROADMAP F1–F7](ROADMAP.md)
is the only backlog and remains incomplete.

## Contract and priorities

Candidates independently consume common inputs/parameters/initial state, never
CPU reference events/routes/results/gradients. General online greedy permits legal
feedback/input and natural streaming degeneration. Preserve int64,stable ordering,
parallel-edge identity,missing/zero messages,None/zero gradients. Matrix: PDG
LibTorch;TimedDAG/Settle LibTorch+PyTorch;CPU/NPU × streaming/prefill × inference/
complete training. Python resident is a C++/CANN client,not independent PyTorch
scheduling. Five presets retain fine switches. FP32 main,FP16 separate;CUDA
execution remains target-machine-pending. Training includes loss interface,
backward/VJP,optimizer,continuation and throughput;downstream convergence excluded.
Current alignment outranks run-ml-experiments;use minimal existing records.

Priorities: complete compact state/cache reverse and public multi-card training/
consumers,reuse verified canonical atomic updates,then bounded medium/full-size five-
preset screening and complete CPU/mixed/resident performance. Do not continually
polish forward fragments. Profiling is an implementation feedback loop. Use affected
checks/byte-verified terminal builds,not repeated unchanged8,954 CPU checks/23 skips.
Implementation commit → immutable qualification → evidence commit;push each.

## Latest verified work

Compact state/cache reverse **7cb7909** plus bounded device program chaining
**49541be3c465548ac0179c29ab0239e8b37d70f6** committed/pushed.
[Report](evidence/device-state-reverse-20261002.md),
[audit](evidence/device-state-reverse-20261002.json). Ten immutable jobs PASSED:
two isolated builds, CMake closure, full two-card FP32/FP16 VJP/training,
three-card memory/locality, one-owner subsets, sequence/control/lifecycle/numerical,
old public half-cache training, five Python oracle/fresh-process regressions,
separate two-card training profile. Each dtype50 VJP trajectories/200 windows;
40 training trajectories/640 windows/160 updates. No tolerance changes.
Owner-local actual tapes, stable packing, global HST/SOFTP normalization, local
Read/state/event/fiber VJP, retained cache bridges, None/zero, physical identities,
canonical aliases, atomic SGD/AdamW, exact publication and continuation verified.
Public multi-device training is NOT yet qualified by these internal checks.

A single persistent CANN stream overflowed its static task buffer in three-card
FP16 retained reverse (507002). HUGE flag did not help; that attempt was removed.
One coordinator program per retained window now uses reusable device completion
notifications; all programs/peers submit before the backward boundary wait.
Dedicated4096-task/8-program sequence test and original failure reproduce/pass.
Source-specific builds/failures are retained in the audit, including old header/
checker build issues, explicit forward budget refusals and missing Python artifact.
No prior failure was relabelled. Final clean build recompiles the close-failure
poison guard; normal and chain/lifecycle device gates passed on that exact source.

Profile:38535 AI_VECTOR_CORE/819 AI_CORE/601 MIX_AIV,no observed AiCPU;212 model
submissions,2192 matching device notify record/wait,4878 switches,17085 DMA.
Both cards execute state reverse. Includes construction/exports/assertions and
refused updates; not throughput. Exact trace/CSV/archive/loader hashes audited.
Forward62935e9 and canonical Full-only a365e2f retain their independent evidence.

## Current work and next action

Previous compact reverse evidence2aa248a is pushed. The public multi-device
training implementation is ready for its implementation commit; formal fixed-source
qualification remains next. No new live jobs; historical CPU task stays paused.

Implemented: public C++ session/placement/owner-local state and KV roots/gradients,
canonical static parameter owners, separate backward/atomic step, portable schema1
checkpoint with changed cards/maps/schedule; Python client/manifest and explicit
model_device=cpu parameter construction; installed external C++ consumer and tests.
No coordinator forward state/KV replica and no CPU-reference event/gradient input.
Public struct ABI changed: clients/bindings and recursive header dependents rebuilt.

Development PASSED (not immutable-source qualification):
- standalone build-public-shards-dev09; Python build-public-shards-python-dev09;
  sources/public-shards-dev09, byte-verified affected host objects and old kernels.
- public-shards-cpp-dev06: FP32/FP16 each32 trajectories/512 windows/128 updates,
  independent CPU FP32/FP64, both schedules/optimizers and state/cache roots.
- public-shards-repartition-dev06: FP16 two-to-three owners;
  public-shards-maps-dev07: FP32 separate Full/state maps,three-to-legacy-single.
- public-shards-python-dev07:41 cases,three families/both schedules,FP16 loss,
  independent CPU oracle,None/zero/refusal,legacy regressions and fresh-process
  single-to-two/two-to-three disk restore. Four affected cases passed again on dev09.
- build-public-shards-consumer-dev08 + public-shards-consumer-dev08: current CMake
  package install and external public-header-only two-card training/restore.
- CPU interface25 passed. No tolerance changes.
- public-shards-profile-dev10 PASSED on source snapshot public-shards-dev10,
  build dev09; separate half-cache training trace. Prior profiledev07 found16
  Bool ScatterUpdate AiCPU tasks in consumer-root assembly; integer flags/cast
  removed those tasks. Confirm exact final counts in profile CSV; not throughput.

All development failures retained. Dev01/02 launcher/include mismatches,dev03
missing ParameterVjp dependency,standalonedev04 test-only const,dev05 link missing
new TUs,Pythondev04 import unresolved symbol,consumerdev07 vendor-static install,
profiledev09 source guard refusal from concurrent vendor fusion_result.json in
snapshot. No source code changed in that profile; its failed status remains.
Future device commands run via env -C RUN using absolute source script/test paths,
keeping vendor files outside frozen source. Builders still use source cwd.
Current external builders: launchers/build_public_shards_v3.py and
build_public_consumer.py. Partial compiled-object reuse is explicitly recorded;
clean builds may reuse byte-identical successful dev09 host objects, not failures.

Next: commit/push implementation, freeze clean public-shards-clean01; build
public-shards-clean01/public-shards-python-clean01 with --reuse-host matching dev09;
qualify public C++ both dtypes,3/1-device placements,41 Python cases,CPU25,
installed consumer and separate profile. Commit reviewed evidence separately and
push. Then continue public scale consumers and F6 representative/full-size five-
preset screening. No new resident throughput/full-size claim has been established.

Historical CPU Attention stays deliberately paused; resolve its retained memory/
timing lock before formal timing. No full-size CPU/mixed/resident result yet.

## Environment and bounded execution

TASK=/mi/data2T/zlong/tide-execution-flows; RUN=TASK/runs/NAME;
unit=tide-execution-flows-NAME.service; RUN/status.json/task.log own lifecycle.
Module libtorch-npu/2.10.0-cann9.0.0; Python
/opt/miniconda/envs/ascend900-train-full-torch-npu-2.10.0-py311/bin/python.
Public /opt stack supersedes dated personal guide by user authorization.
TASK_QUEUE_ENABLE=0,TORCH_DEVICE_BACKEND_AUTOLOAD=0; preserve module PYTHONPATH,
prepend snapshot/python. SoC Ascend910_9392. Use leased logical devices only.
Last space: data225GiB/root14GiB; large writes go under TASK, recheck capacity.
Core builds: placement-cpu-clean01,placement-npu-clean01(standalone),
placement-npu-python-clean01(Python-owned). Never mix SDK and Python runtimes.
Use existing launchers/freeze_run.py, frozen sources, background.slice/Nice10,
queue120s/run600s/build900s. Atomic handoff writes via scripts/durable_records.py.

## Preserved historical boundaries

Earlier24 dirty files are SHA256-verified on pushed archive/restricted-flow-20260930
at964bf628c67270200dabe55b1bca026bd403cd37; TASK/restricted-flow-archive.json.
Historical historical-cpu-attention-01 remains intentionally SIGSTOP; pause.json
outweighs running status and retains host memory/TASK/timing.lock.
Historical Add CPU78.793172/NPU4 47.932888ms/token is throughput1.6438× faster;
it does not certify the new resident path. No full CPU Attention training ratio.
