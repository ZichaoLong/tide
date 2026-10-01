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

Compact state/Read/KV owner increment ready for implementation commit/push.
Explicit kernel view (not a fake Graph), whole-fiber packing, compact owners,
three-phase device request/selection/commit and ContentFlow ModelPlacement.
No complete coordinator state/KV replica. Old Full-only training remains available;
compact-state reverse/public multi-device training are explicitly gated pending
owner tapes/adjoints/publication. Contract: resident-peers.md.

Development terminal checks: builds dev01/dev03/dev04/dev05 PASSED. dev01 two-card
FP32/FP16 smoke each3 configurations/15 windows PASSED. state-flow-dev04 four cells
PASSED: each dtype84 HARD/420 windows +36 HST/SOFTP/180 windows, plus3 real capacity
refusals and incomplete-adjoint refusal. state-transaction-dev05 both dtypes PASSED:
whole-fiber/int64/parallel-edge/padding tests, each28 windows/16 refused commits,
byte-exact unchanged state/KV and independent CPU accepted steps. No numerical
mismatch or changed tolerance. No new development job is live.
Failures retained: build-state-owners-dev02 test omitted Streaming Options;
state-gates-dev03 test reused compiled port layout after changing ports (fixed by
fresh Graph). Packing checks passed before that fixture failure.

Next: commit/push implementation, use that immutable commit in freeze_run.py for
state-owners-clean01. Build command: timeout900 python TASK/launchers/
build_state_owners.py state-owners-clean01 --kernel-build state-owners-dev01
--reuse-host state-owners-dev05. Python runtime build uses --runtime python with
its isolated parent; never reuse standalone host objects. Then six affected cells
(peer-state-transaction/flow/control-flow),1/3-owner smoke via state_owner_policy_smoke.py,
state_cmake_preflight.py, separate two-card profile and small existing public
training/Python regressions. Queue120/run600/build900. Evidence separate commit/push.
Raw development records remain TASK/runs and source snapshots TASK/sources.

Continue next with compact retained owner state/cache tapes and boundary adjoints,
canonical publication and public multi-device training/consumer integration, then
F6 throughput/full-size work. No forward/profile smoke throughput claim. Canonical
source a365e2f and pushed evidence bc778d3 stay valid. Historical CPU Attention stays
paused. Do not repeat unrelated CPU tests or run an unbounded queue.

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
