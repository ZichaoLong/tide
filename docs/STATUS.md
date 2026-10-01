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

Retained cross-card Full VJP on **5591319c2821f40f3f5fa93317e2d745de7e4559**,
implementation committed/pushed; [report](evidence/device-full-reverse-20261001.md),
[audit](evidence/device-full-reverse-20261001.json). All ten immutable jobs
PASSED/exit0. FP32/FP16 each50 trajectories/200 windows; three-device
memory/locality × two dtypes four cells,each2/8; one-owner half2/8;
four single-device inference cells; half-cache training1/16/4; Python96/no skips.
Independent CPU FP32/FP64 autograd,after-close retention,window bridges,exact
candidate replay,empty/malformed/duplicate/budget/dtype/single-device misuse refusals.
No numerical threshold changed. Parameter partials remain on Full owners;
canonical alias reduction and optimizer publication are not yet multi-device.

Profile3 devices,two fixtures/eight windows,60 host model submissions,336 notify
records/waits,1713 switches,3258 DMA tasks. Engines11562 AI_VECTOR_CORE,
366 AI_CORE,196 MIX_AIV,120 AiCPU. All AiCPU rows are coordinator Bool ScatterUpdate
connection merges,approximately9.048ms summed task time. This is device AiCPU,
not logged host fallback,not wall time or throughput. Immediate improvement:
replace Bool scatter/error/chunk merge with one metadata kernel per shard stage.

Standalone/Python fixed builds full-reverse-clean01/full-reverse-python-clean01;
source full-reverse-clean01. Six host objects rebuilt per runtime; source/hash
checked kernel reuse from passed full-reverse-dev01. Audit script:
TASK/launchers/full_reverse_evidence.py FULL_REV. All qualification jobs terminal.
Development failures preserved: unsupported CPU Bool test constructor in dev01,
missing explicit NoGradGuard include in dev02,wrong three-card wrapper marker
in dev03 despite successful first child. Fixed in dev03/dev03b; no observed
production numerical failure. Earlier Full shards2617a11,remote Fulla8fa371,
peer packets5c3662b,public single-device FP16 training0095048 remain qualified.

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
host memory and timing lock; do not blindly resume/kill it. No current qualification
jobs remain live. Evidence/STATUS/ROADMAP changes are the only uncommitted work.

Next action: record/commit/push the audited5591319 evidence. Apply the observed
Bool scatter metadata fusion with affected checks/profile, then continue the main
owner alias reduction,atomic optimizer and bank publication integration. Do not
pause or ask for new permission after a commit. Use existing build/lease launchers
and fixed-source evidence; no repeat of unrelated old gates.

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
