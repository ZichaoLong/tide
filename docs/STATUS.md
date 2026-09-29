# Current handoff

Updated 2026-09-29. Active authorized increment: matched-source CPU/NPU full-size
comparison, full-size profiling and bounded device-resident scheduling. See
ROADMAP D1-D6. Branch graph-execution-foundation, starting8ce1637; no push.
No sub-agents; reference repositories and ObsidianVault remain read-only.

## Current work and next action

Initial audit/skills/site checks complete. Frozen dirty development snapshot dev01
contains CPU node parallelism and phase-scoped profiling. Live build units:
`tide-device-scheduler-build-cpu-dev01.service` and
`tide-device-scheduler-build-npu-dev01.service`, each two build workers,
background.slice, exact source TASK_ROOT/sources/dev01. Status/logs:
TASK_ROOT/runs/build-{cpu,npu}-dev01/{status.json,task.log}; artifacts links
are named device-scheduler-build-{cpu,npu}-dev01. No pass result yet.
CPU Resident currently creates one worker per device, hence only one CPU node
worker despite --workers. Correct this with independent scalar value/route/VJP
and training gates before a fair pure-CPU baseline. Add phase-scoped CANN
profiling outside normal benchmark timing. Freeze and build each increment.

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
