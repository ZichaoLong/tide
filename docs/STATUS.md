# Current handoff

Updated 2026-10-01. **ACTIVE: user resumed and accepted overhead reduction.**
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

Priorities: canonical device owners/alias reduction and atomic multi-card updates,
full state/KV placement and public consumers,then bounded medium/full-size five-
preset screening and complete CPU/mixed/resident performance. Do not continually
polish forward fragments. Profiling is an implementation feedback loop. Use affected
checks/byte-verified terminal builds,not repeated unchanged8,954 CPU checks/23 skips.
Implementation commit → immutable qualification → evidence commit;push each.

## Latest verified work

Canonical device owners **a365e2f1635f3ef8f9b66176662b058ef1b39645** committed/pushed.
[Report](evidence/device-canonical-owners-20261001.md) and
[audit](evidence/device-canonical-owners-20261001.json). All9 fixed jobs PASSED:
two isolated runtime builds,standard CMake preflight,eight two-card component
cells,three-card memory/locality,fixed three-card profile,one-owner degeneration,
public half-cache smoke and five Python scalar-oracle/fresh-process checks.
Each dtype: canonical retained VJP50 trajectories/200 windows; SGD/AdamW4/32
updates; actual sharded training40 trajectories/640 windows/160 updates.
CPU FP32/FP64 are independent. Exact master-to-bank publication,None/zero,
shared HARD Read,strided QKV,FP16 widened normalization and all-device atomic
nonfinite/half-overflow refusal passed. Masters/partials never feed through CPU.
State/KV/readiness remain coordinator-owned; public multi-card API is still absent.

Profile:28187 AI_VECTOR_CORE/802 AI_CORE/409 MIX_AIV,zero observed AiCPU.
Every card runs canonical gather/reduce,optimizer,publication.198 submissions,
1282 matching notify records/waits,4419 switches,10575 DMA. Two trajectories:
8 accepted updates +2 refused replays;counts include construction/assertions,
not throughput. Full raw records in TASK; audit verifies hashes and terminals.

Development failures retained: dev01 test initializer brace;dev03 empty-partition
64KiB workspace below77312-byte operator requirement (test arena raised to1MiB);
graph-training-dev03 scalar CPU byte-view assertion (reshape fixed). No candidate
numerical failure or tolerance change. Source snapshot canonical-owners-clean01;
builds canonical-owners-clean01/canonical-owners-python-clean01. Audit command:
TASK/launchers/canonical_owners_evidence.py a365e2f1635f3ef8f9b66176662b058ef1b39645.
No qualification job is live. Implementation→evidence lifecycle is preserved.

Earlier Full reverse5591319,metadata fusionfe68065,Full shards2617a11 and public
single-device FP16 training0095048 remain scoped qualified evidence. The former
120 Bool ScatterUpdate AiCPU finding/failures are retained in their own reports.

## Current work and next action

Commit/push this reviewed evidence/STATUS/ROADMAP,then continue overall F1–F7.
No user pause instruction. No implementation changes remain uncommitted.
Next priority: partition state/attention parameters and persistent KV while
preserving device queue/readiness and boundary VJP links; expose a coherent public
C++/Python multi-card owner and consumer. Reuse the now-verified canonical reducer,
optimizer consensus and publication instead of reimplementing alias handling.
Do not call Full-only placement whole-model sharding. CPU independent reference
and old eager/host presets remain available. Bounded medium/full-size complete
CPU/mixed/resident comparisons follow public-path/placement correctness.

Implementation entry points: content_flow_internal.h/content_stages.cpp and
packed_event_attention/packed_fiber_attention own coordinator state; remote_full/
sharded_full provide reusable device command/peer completion patterns.
sharded_parameter_banks.cpp currently maps state/Read/attention to coordinator;
sharded_parameter_sources.cpp maps non-Full gradients there. Those assumptions
must evolve explicitly with state/KV ownership; avoid assembling full banks on
coordinator or copying candidate parameters back to CPU between updates.
Current canonical budget is for fixed retained-window device-pair packets; full
scale needs calibrated peak estimates/safe splitting,not OOM search.

Before formal performance,resolve historical CPU Attention's retained memory and
timing lock deliberately;do not blindly resume/kill. Additional CANN environments,
CUDA target-pending records and final delivery audit remain F1–F7 work.

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
