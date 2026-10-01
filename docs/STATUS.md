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

Compact state/Read/KV forward **62935e98772fdacff742be915d588109aee63300** committed/pushed.
[Report](evidence/device-state-owners-20261001.md),[audit](evidence/device-state-owners-20261001.json).
Nine fixed jobs PASSED: two isolated builds,CMake closure,six two-card cells,
three-card memory/locality,single-owner degeneration,two-card trace,public half-cache
training smoke and five Python scalar-oracle/fresh-process regressions. Each dtype:
84 HARD/420 windows +36 HST/SOFTP/180 windows,3 real capacity refusals,28 transaction
windows/16 refused commits with byte-exact unchanged state/KV. No tolerance changes.
All services terminate on empty/refused windows; restored continuation can repartition.

State/Read/Upd/KV live only on compact owners; no full coordinator state/KV replica.
Global region selection,queue/readiness,history,Aggregate and emission stay on the
coordinator NPU. Three-phase device Read→selection→common-commit protocol.
Explicit StateKernelProfile is a view,never a fake compiled Graph. Old monolithic
reverse/state/publication interfaces refuse this new placement. Compact-state
reverse and public multi-device training are NOT yet implemented.

Profile:5283 AI_VECTOR_CORE/110 AI_CORE/3 MIX_AIV,no observed AiCPU. Both cards run
Read,state,event/fiber KV commits.45 model submissions for15 windows;305 matching
notify record/wait,4932 switches,2064 DMA. Includes construction/exports/assertions;
not a throughput comparison. Reuse hashes/archives/loaders/logs/CSV audited.
Failures kept: dev02 test omitted Streaming Options;dev03 fixture stale compiled
port layout;cmake-clean01 audit expected static closure on consumer instead of
shared library. Fixed checks passed; none was a numerical candidate mismatch.

Canonical Full-only multi-card training a365e2f remains qualified independently:
each dtype40 trajectories/640 windows/160 updates,canonical device alias sum,
all-device atomic SGD/AdamW and bank publication. That path still owns state/KV on
the coordinator; its training evidence does not automatically cover62935e9 shards.
Earlier Full reverse5591319,metadata fusionfe68065 and public FP16 training0095048
retain their scoped evidence. No new qualification job is live.

## Current work and next action

Commit/push reviewed evidence/STATUS/ROADMAP; implementation already pushed.
Then continue F1–F7, no pause instruction. Next main implementation: retained
compact state/cache owner tapes, reverse progression and cache boundary adjoints;
reuse canonical reduction/atomic optimizer/publication and expose public multi-card
C++/Python consumers. Do not spend another increment polishing forward fragments.

Reverse design inspected only, not implemented: graph_vjp.cpp has a FullStageVjp
seam but state/control/cache reverse are still monolithic. Event/fiber reverse read
Graph primarily for node count,source counts/input sizes and parameter offsets.
Introduce an explicit compact reverse view instead of compiling an invalid local
Graph. Actual coordinator device journals can be packed by owner on device at
reverse preparation, preserving global event/atom row maps; existing owner KV
journals/caches must remain local. Never send full KV to the coordinator or use CPU
reference tapes. State cotangents/None flags and per-stage message gradients need
batched pack/return; local cache adjoints must bridge retained windows on owners.
Control/read derivatives require their own split,while global selection/softmax
semantics remain unchanged. StateOwner currently exposes neither retained cache
tapes nor bank publication; add these explicit seams with reverse integration.
ShardedParameterSources/Banks still assume non-Full gradients/banks on coordinator;
extend ownership maps without reimplementing canonical alias/optimizer rules.

Fixed source TASK/sources/state-owners-clean01. Builds state-owners-clean01 and
state-owners-python-clean01; audit TASK/launchers/state_owners_evidence.py 62935e98772fdacff742be915d588109aee63300.
Affected builder build_state_owners.py compiles15 host objects/3 new kernels;
clean standalone reused passed dev05 source/header/object hashes,Python rebuilt15.
CMake gate is configure/link dependency generation,not full fresh compile.
No unrelated8,954-check CPU rerun. Next complete training/consumer correctness,
then medium/full-size CPU/mixed/resident performance. Historical CPU Attention
stays deliberately paused; resolve retained memory/timing lock before formal timing.

## Environment and bounded execution

TASK=/mi/data2T/zlong/tide-execution-flows; RUN=TASK/runs/NAME;
unit=tide-execution-flows-NAME.service; RUN/status.json/task.log own lifecycle.
Module libtorch-npu/2.10.0-cann9.0.0; Python
/opt/miniconda/envs/ascend900-train-full-torch-npu-2.10.0-py311/bin/python.
Public /opt stack supersedes dated personal guide by user authorization.
TASK_QUEUE_ENABLE=0,TORCH_DEVICE_BACKEND_AUTOLOAD=0; preserve module PYTHONPATH,
prepend snapshot/python. SoC Ascend910_9392. Use leased logical devices only.
Last space: data227GiB/root14GiB; large writes go under TASK, recheck capacity.
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
