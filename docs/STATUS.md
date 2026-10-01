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

No new live jobs. Frozen qualification source TASK/sources/program-chain-clean01;
standalone/Python builds TASK/builds/program-chain{,-python}-clean01. Audit command:
`python TASK/launchers/state_reverse_evidence.py 49541be3c465548ac0179c29ab0239e8b37d70f6`.
Commit/push only the reviewed evidence/ROADMAP/STATUS now; preserve current code edits.

Public multi-device training is being implemented, **unbuilt/unverified**:
- public placement, owner-local state/cache root/output and parameter-gradient structs;
- shared parameter freeze/session-ID helpers and single-session delegation;
- borrowed compact state views, static canonical layout, update-only composition;
- ShardedTrainingOwner forward/backward/step and portable schema1 checkpoint draft.
Files include public resident_training.h, training_{owner,parameters,backward,
checkpoint,internal}, sharded_training_{owner,internal,backward,checkpoint},
sharded_parameter_{layout,sources,reduce}, state_owner_tape/sharded_state/content_flow.
These are this task's edits; do not discard them or claim public support yet.
Next: review/fix draft compilation (notably mixed auto declarations), register new
CMake units, add Python bindings/client placement and owner roots, and meaningful
public multi-card oracle/update/repartitioned checkpoint/continuation checks.
Python GraphRuntime currently materializes all model parameters on runtime.device;
large sharded consumers need explicit CPU initial-parameter storage (no CPU events)
or a suitable new construction boundary. Finish that before claiming model-scale use.
Use affected builds/gates; do not rerun unchanged CPU core suites. Then commit
implementation, qualify fixed source, publish evidence, and continue consumers/F6.

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
