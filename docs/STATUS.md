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

Full reverse metadata fusion **fe68065fbf41eebbba003051a84d3567e59028d1** committed/
pushed;[report](evidence/device-full-reverse-merge-20261001.md),
[audit](evidence/device-full-reverse-merge-20261001.json). All4 fixed jobs PASSED:
two runtime builds,two-dtype gate,3-device profile. FP32/FP16 each50 trajectories/
200 retained windows plus empty/malformed/budget/dtype/owner refusals and replay.
Profile:120 Bool ScatterUpdate AiCPU tasks replaced by40 metadata merges;
11418 AI_VECTOR_CORE,366 AI_CORE,196 MIX_AIV,zero observed AiCPU. Host submissions60,
notify records/waits336,DMA3258 unchanged;switches1713→1673. Counts include setup/
assertions,not throughput. No tolerance changes or development failure in fusion.
Only2 host objects rebuilt;source/hash-checked kernel reuse from passed
full-reverse-merge-dev01. Snapshot/build full-reverse-merge-clean01;Python build
full-reverse-merge-python-clean01. Audit command:
TASK/launchers/full_reverse_merge_evidence.py FULL_REV. No qualification jobs live.

Preceding retained cross-card VJP5591319:FP32/FP16 each50 trajectories/200 windows,
3-card memory/locality × two dtypes,1-owner degeneration,four inference cells,
FP16 single-card cache training1/16/4,Python96/no skips. Full parameters/partials
remain on owners;state/message/control/cache reverse and window bridges remain
coordinator-owned. After-close retention and exact replay verified against CPU
FP32/FP64. CPU alias sums in the assertion adapter never feed candidates. This
is not device canonical owner reduction or multi-device optimizer training.
[Report](evidence/device-full-reverse-20261001.md) preserves120-AiCPU trace and
prior development failures:CPU Bool assertion constructor,missing test include,
wrong wrapper marker despite successful child. Numerical tolerances unchanged.

Earlier Full shards2617a11,remote Fulla8fa371,peer packets5c3662b and public
single-device FP16 training0095048 remain qualified. Prior raw failures are retained.

## Current work and next action

Fusion evidence/STATUS/ROADMAP are ready to commit and push. Continue afterwards.
Untracked draft for next canonical owner increment:
- sharded_parameter_sources.h/.cpp maps actual per-window Full/coordinator
  partials to global registry owners in reverse-window/alias order and assigns
  canonical owners with deterministic LPT. Static tensor views only,no host
  numerical gradient reads. Metadata budget is separate from partition tensors.
- owner_gradient_packet.h/.cpp and ascendc/tide_owner_gradient_pack.cpp,
  ascendc/tide_owner_gradient_reduce.cpp draft one packed gather/reduce per group.
  The source pointer descriptors refer only to retained tensors on the same NPU;
  cross-card copying must use PeerExchange. None payloads are not evaluated.

These drafts are NOT wired into CMake or built/tested. Next implement bounded
per-source/per-target packet assembly and ordered owner reduction on the assigned
cards,then global finite/error decision before any optimizer commit and publication
to all aliased banks. Reuse actual graph/retained fixtures and independent CPU
oracles;no CPU comparison sum may become a candidate training input. Check budget,
None/zero,aliases across Full and state/Read,FP32 masters/slots and FP16 publication.
Do not call the draft a functioning reduction or complete training.

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
