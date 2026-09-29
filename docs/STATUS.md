# Current handoff

Updated 2026-09-29. Active authorized increment: matched-source CPU/NPU full-size
comparison, full-size profiling and bounded device-resident scheduling. See
ROADMAP D1-D6. Branch graph-execution-foundation, starting8ce1637; no push.
No sub-agents; reference repositories and ObsidianVault remain read-only.

## Current work and next action

Initial implementation committed atcba3989: CPU node parallelism and optional
phase-scoped CANN profiler, plus bounded profile storage monitoring. Independent
dev01 CPU gates passed16 forward/VJP and4 training cells; both builds passed4
CTests;8 lifecycle tests passed. Named CPU profiler rejection passed (the first
manual harness omitted --run-id and was corrected; no workload ran). Scoped NPU
token smoke produced1744 operator records and exported successfully.

CPU vocabulary projection now uses the existing public DenseLinear pool via
explicit --head-workers (default1; NPU requires1). Dev02 CPU/NPU builds passed
all4 CTests; CPU forward/VJP gate passed16 cells, including head-workers3,
and4 Add/Attention x SGD/AdamW three-update training cells passed. These are
dirty development results. Commit this increment and freeze clean baselines;
next build client-{cpu,npu}-baseline01 and run immutable CPU/NPU gates before
formal D1 timing. No full-size experiment has started yet; no production D3-D6
backend exists yet. All dev02 jobs are terminal and passed.

Separate native NPUGraph probes passed sort and pairwise variants: three replays
with changed inputs, five selections/history updates per replay, int64 base2^55,
independent CPU tuple oracle. First probe build failed for a missing official SDK
include root; corrected v2 build retained separately. These are primitive probes,
not proof of complete resident scheduling. Sources:inputs/graph-probe{,-v2};
binary:builds/graph-probe-dev02/graph-probe. No environment/SDK files were edited.

All earlier dev01/dev02/probe jobs are terminal. Exact state/logs are
TASK_ROOT/runs/NAME/{status.json,task.log}, units use tide-device-scheduler-NAME,
artifacts links use device-scheduler-NAME. Frozen development sources are
TASK_ROOT/sources/dev02, with adjacent snapshot identity; builds under
TASK_ROOT/builds/client-{cpu,npu}-dev02. Never edit those frozen source files.

TASK_ROOT=/mi/data2T/zlong/tide-device-scheduler; preflight.json records current
CPU/cgroup/NUMA/memory bounds. Sources/builds/runs/launchers/inputs/trackio are
separate from the closed prior task. Initial free space:74GiB on task volume,
31GiB on repository filesystem. Bound traces; preserve cited old artifacts.
Use public libtorch-npu/2.10.0-cann9.0.0 for native NPU; CPU build remains separate.
Trackio writer/viewer:/home/zlong/venvs/trackio/bin/python, existing0.35.0;
local best-effort projection, no dashboard required or exposed.

## Prior verified baseline

Previous FP16/performance work is closed at8ce1637. See
[eight-cell report](evidence/accelerator-fp16-performance-20260929.md),
[FP16 qualification](evidence/fp16-qualification-20260929.md) and
[profile analysis](evidence/accelerator-profile-analysis-20260929.md).
All prior jobs are terminal; resource failures retain their original states.
Do not append to finalized prior resource logs or rerun their inspector.
Payload/state/KV/messages reside on NPUs, but histories, scalar/index returns,
tensor handles and C++ dispatch remain host-owned. Current all32 is not full
device scheduling. CUDA build/host checks do not establish NVIDIA execution.

Re-entry:git status --short --branch; python scripts/status.py; read this file.
