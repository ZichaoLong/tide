# Current handoff

Updated 2026-10-01. **ACTIVE: user resumed execution and accepted overhead reduction.**
Continue coherent multi-device graph/training increments; commit and push authorized.
No new pause instruction. No subagents. Reference repositories and ObsidianVault
are read-only. Repository `/home/zlong/llm/graph-execution-foundation` resolves to
`/var/tmp/zlong-graph-execution-foundation/repository`; branch `graph-execution-foundation`.
[execution-flows.md](execution-flows.md) owns the contract;
[ROADMAP F1–F7](ROADMAP.md) is the only backlog and remains incomplete.

## Contract

Candidates independently consume common inputs, initial state and parameters;
never CPU reference events, routes, results or gradients. Online greedy accepts
legal topology/input, including positive-delay PDG feedback; natural streaming
degeneration is valid. Preserve int64, stable ordering, parallel-edge identity,
missing/zero messages and None/zero gradients. Matrix: PDG LibTorch;
TimedDAG/Settle LibTorch and PyTorch; CPU/NPU × streaming/prefill × inference/full
training. Python resident is a C++/CANN client, not a separate PyTorch scheduler.
Five presets retain fine switches; FP32 main, FP16 separate; CUDA target-pending.
Training covers forward/loss interface/backward/VJP/optimizer/continuation/throughput;
downstream convergence is outside this task. Current alignment outranks
run-ml-experiments: minimal existing records, no additional tracking framework.

Use affected checks and source/object/hash-verified terminal build reuse. Do not
rerun the unchanged core's 8,954 CPU checks/23 optional skips. Commit implementation,
qualify immutable source, commit evidence separately and push. Group related work
into vertical increments; reuse fixtures/assertions/audits. Early profiles diagnose
placement/cost; formal full-size repetitions follow integration/capacity calibration.

## Latest completed qualification

Compact Full parameter shards on **2617a11e58b5ffe04cc1d752645cff0d6c15f427**;
[report](evidence/device-full-shards-20261001.md), [audit](evidence/device-full-shards-20261001.json).
All ten fixed-source jobs PASSED/exit 0: two runtime builds, two two-device gates,
single-device inference regression, three-device placement smoke, one-shard smoke,
half-cache training regression smoke, Python regression and three-device profile.
FP32/FP16 each: two-device HARD 84 configurations/420 windows, HST/SOFTP 36/180.
Three-device memory/locality × two dtypes: four cells, each 3 configurations/15 windows.
One-shard 3/15; single-device training 1 trajectory/16 windows/4 updates;
Python 96 passed/no skips. Development also passed without tolerance relaxation.

Actual selected actions are packed stably into compact owner banks. All nonempty
peer requests precede local Full and result waits. Memory/locality placement is
static and generic; continuation changes schedule and placement. Per-shard memory,
chunk and actual/capacity work are recorded. The coordinator retains state/KV,
Read, readiness, selection and queues. This is Full-only inference sharding;
remote/sharded reverse remains explicitly refused, public defaults single-device.

Profile: three fixtures/15 windows, 45 host model submissions, three devices;
4067 AI_VECTOR_CORE,86 AI_CORE,3 MIX_AIV; no observed AiCPU. Full work on all cards,
packing/selection on coordinator,165 device notify records/waits,713 switches,
1419 DMA tasks. Counts include setup/boundaries; no throughput/overlap claim.

TASK/launchers/full_shards_evidence.py FULL_REV passed. Frozen standalone source/
build full-shards-clean01 and Python build full-shards-python-clean01. Eight host
objects rebuilt per runtime; new kernel reused with source/archive checks from
successful full-shards-dev01. Other source/object/archive reuse is byte-verified.

Earlier qualified foundations: remote Full a8fa371, reusable peer packets 5c3662b,
public FP16 full training 0095048 (83 trajectories/1328 windows/332 updates,FP32
masters/slots and actual half rounding). Their reports retain earlier failures.

## Current next work

1. Cross-card Full reverse integrated with actual graph reverse stages, owner
   parameter gradients and shared-parameter reduction; FP32 master optimizer,
   atomic finite/error decision and alias publication. Do not keep polishing only
   forward components. Shared aliases cannot be assumed local to one shard.
2. General persistent state/KV placement and packed transport, continuation and
   public C++/Python multi-device consumers, then complete experiment consumers.
3. Bounded middle-scale five-preset screening followed by representative/full-size
   CPU + selected mixed + resident comparisons, both schedules/inference/training,
   FP32 and separate FP16. Additional CANN/environment and target-pending CUDA
   records, final delivery audit.

Before performance, deliberately resolve historical CPU Attention's retained
host memory and timing lock; do not blindly resume/kill it. No qualification jobs
remain live. Evidence commit c57f6b0 pushed. Current implementation: explicit sharded tapes
and per-device retention, Full-stage graph-VJP seam, stable connected-row packing,
owner-local Full parameter partials, coordinator state/message/cache reverse and
retained-window bridges. Public single-device ABI/defaults unchanged. No device
alias reduction/optimizer/public multi-device training claim yet.

Development full-reverse-dev03 standalone/Python builds PASSED; two dtype smoke
runs passed, then full FP32/FP16 gates each50 trajectories/200 retained windows
plus empty/malformed/budget/type/owner refusals and exact replay. Three-device
memory/locality × two dtypes smoke passed (four cells,each2 trajectories/8 windows).
Four single-device inference cells and half-cache training1/16/4 also passed.
Python runtime regression is deferred to immutable qualification, avoiding a
redundant dev pass. Only follow-up since dev03: initialize unused accumulator
chunk_rows to zero and clarify internal capability comments; no numerical change.

Retained development failures: both dev01 dtype smoke tasks stopped in the CPU
assertion's unsupported at::tensor(bool), fixed with at::full; dev02 build missed
an explicit NoGradGuard header; dev03 three-card wrapper used the old forward
success marker although its first child passed, fixed in the separate
full_reverse_policy_smoke.py and verified by full-reverse-three-dev03b.
No observed production numerical/runtime failure or tolerance relaxation.

Next: commit/push this coherent retained cross-card VJP increment; qualify clean
source full-reverse-clean01 with both runtime builds,full FP32/FP16 gates,3-device
policy smoke,1-shard degeneration,affected inference/training/Python regressions
and a separate3-device profile. Use TASK/launchers/build_full_reverse.py NAME
[--runtime python] --kernel-build full-reverse-dev01; preserve prior failures.
Existing qualification_audit.py supports trajectory counts. Then continue the
owner alias reduction/optimizer/publication main work without requesting a pause.

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
