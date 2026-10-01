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

Implementation ready to commit/push: canonical device owner reduction,globally
validated groups/shared-storage refusal,all-device finite consensus,FP32 master
SGD/AdamW and packed publication to used aliases. Explicit sharded bank views
cover HARD Read,strided QKV and FP16 values widened into FP32 normalization banks.
No CPU numerical gradients/masters feed candidates. State/KV remain coordinator-
owned; public multi-card training/whole-model placement are not delivered yet.

Development PASSED:
- build canonical-owners-dev02 and canonical-owners-dev02 gate: FP32/FP16 retained
  canonical VJP each50 trajectories/200 windows plus original optimizer gates.
- build canonical-owners-dev05 and canonical-training-dev05 gate: global optimizer
  each4 trajectories/32 updates; actual graph training each40 trajectories/
  640 windows/160 updates,independent CPU FP32/FP64,exact alias publication,
  None/zero,empty partitions,nonfinite/half-overflow atomic refusals,continuation.

Retained development failures: dev01 test initializer missing brace;dev03 empty-
partition test's64KiB arena versus required77312bytes (raised only test arena to
1MiB);graph-training-dev03 CPU byte assertion on0-d Float bank (reshape fixed).
No candidate numerical failure or tolerance change. All development jobs terminal.

Next after implementation commit: freeze canonical-owners-clean01 at exact HEAD;
standalone build reuses hash-matched passed dev05 host objects and dev02 kernels;
Python-owned build rebuilds affected8 host objects against its own runtime.
Qualify affected8 standalone cells,3-card memory/locality smoke and profile,
1-owner training smoke and single-card public half-cache regression. Configure
standard CMake targets; audit hashes/terminal records,then evidence commit/push.
Use existing launchers/build_canonical_owners.py,freeze_run.py and
sharded_training_policy_smoke.py. Builds900s,gates120s queue/600s run. Continue
F1–F7 afterwards; no user pause instruction. No unchanged full CPU/96-Python rerun.

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
