# Current handoff

Updated 2026-09-29. The authorized local FP16/profiling/performance extension is
complete. No project benchmark or fallback remains running or queued.
Branch graph-execution-foundation; no push authorized. No sub-agents; reference
repositories and ObsidianVault remain read-only. Evidence is committed separately
from the immutable implementations; no production source changed in this increment.

## Completed scope

Explicit FP16 public Python/native payload and standalone consumer support use
FP32 masters/slots/loss and static scaling. Read/control placement and precision
remain independent. Same formulas do not imply cross-dtype route identity or
training convergence. See precision.md and accelerator-scale.md for API boundaries.

Implementation fc2a76f; profiler copy ordering cb58110; matched-dtype consumer
VJP/master oracle595dccd; early CPU Half CSR rejection c8d2b61/4a7dec7. The full
C++ source hashes match the reused fc2a76f core build. Nine-chip qualification was
recorded in7c2d49e. All performance below uses clean595dccd0b23bd809bed28874591ae4f5f268512f.

Passed qualification:8645 CPU tests;39 Python and43 native FP16 NPU cases plus
explicit CSR rejection;10 FP32 public representatives;124 consumer cells on
CPU/2/4/6/7/8/9 NPUs;4 consumer CTests;20 CSR boundary follow-ups plus one named
CLI rejection. FP16 public atol.001/rtol.02; consumer explicit .002/.02; discrete
and None/zero contracts stay exact. CUDA-linked build and22 host checks passed,
without NVIDIA hardware qualification. Python/native FP16 tiny profiles contain
1644/1638 hardware kernels. New FP16 covers Torch/TorchNPU2.10/CANN9.0 only;
prior CANN8.5.0/.1/.2 FP32 qualification is unchanged.

## Completed performance

All eight accepted Add/Attention inference/training cells passed. Full-size
D2048/B512/V50304,465 nodes/2208 logical/4418 physical edges, seed7, locality,
resident payload, workers16/ATen1,TASK_QUEUE_ENABLE=0. Add9,468,020,899 and
Attention17,269,426,339 parameters. Inference uses12 tokens (4 warmup/8 measured);
training uses two complete12-token AdamW updates (1 warmup/1 measured).

- infer-add, 2 chips: FP32/FP16 17.412729/18.703945 ms/sample-token; peak19.868474/9.954680 GiB/chip.
- infer-attention, 4 chips: FP32/FP16 54.611865/45.855257 ms/sample-token; peak22.286432/11.192269 GiB/chip.
- train-add, 4 chips: FP32/FP16 39.482509/42.135417 ms/sample-token; peak41.058367/47.849997 GiB/chip.
- train-attention, 9 chips: FP32/FP16 105.064709/110.660020 ms/sample-token; peak35.854781/40.230472 GiB/chip.

The first three pairs use FP32 then FP16. Final Attention uses FP16 then FP32
on physical1,2,3,4,5,11,12,13,14, retaining the same logical placement. External
processes were allowed and their changes recorded; these are shared-load
exploratory observations, not stable averages, causal speedups or convergence.
Every accepted raw record and Trackio SQLite step/metric matches its JSONL;
all native exits are zero and no child process remains. See
[eight-cell report](evidence/accelerator-fp16-performance-20260929.md) and adjacent
JSON for exact configuration/source/binary/input/record hashes and work counters.

TASK_ROOT=/mi/data2T/zlong/tide-npu-performance. Final unit
`tide-npu-performance-pair-train-attention9-fp16-a5h.service` is terminal/passed;
run TASK_ROOT/runs/pair-train-attention9-fp16-a5h, linked under ignored artifacts/.
Exact command/cwd/terminal exit are in status.json; resolved plan and commands
are in plan.json/stages.json. Frozen source TASK_ROOT/sources/fp16-a5b; matching
build TASK_ROOT/builds/client-fp16-a5b. Public module libtorch-npu/2.10.0-cann9.0.0
and shared /usr/local driver were used. The public C++ core has no Python or
Trackio dependency; the external wrapper owns experiment recording.

Trackio best-effort local project tide-npu-performance, TASK_ROOT/trackio,
SQLite tide-npu-performance.db, version0.35.0; writer/viewer
/home/zlong/venvs/trackio/bin/python. No dashboard was exposed. New-extension
ledgers retain14 terminal raw cells (9 completed/5 failed or cancelled) and
10 Trackio checks (9 completed/1 failed warmup). Only eight accepted cells enter
the primary pair table; the extra successful prior FP32 is separately retained.
Do not append to finalized resource logs: their hashes are now evidence.

## Retained resource failures

The original8-chip waiter and unstarted11-chip waiter were cancelled. Two7-chip
runs were cancelled after external allocations, with no complete update. The
6-chip FP32 OOM occurred with external use; its automatically started FP16 was
cancelled. Their successful small-tensor gates remain valid.

The first shared9-chip attempt a5g passed FP32 but FP16 failed in its measured
update on logical0/physical0:82MiB requested,39.39GiB allocated,41.19GiB reserved,
47.53MiB free of61.27GiB usable capacity, with external processes present.
It has one FP16 warmup observation and no measured FP16 timing. Its failure and
successful FP32 sibling remain in the report; neither is silently overwritten.
The successful retry replaces physical0 with physical1 and checks FP16 first;
shape/batch/window and implementation are unchanged. No unrelated process was stopped.
The7-chip fallback a5d timed out at03:44:19Z after7200s, exit3, no cell started;
its state remains failed. The final launcher verified it was already inactive.

## Boundaries and next work

No authorized local test remains. On a future target machine, follow
accelerators.md and precision.md for CUDA real-device or other architecture/
stack/dtype acceptance. CUDA compilation does not qualify GPU execution.

NPU payload/state/KV/messages and device copies are implemented. Read/controls/
ranking/event keys are configurable, but host histories, scalar/index returns,
tensor handles and dispatch remain. This is not a fully device-resident control
loop or DDP/HCCL. Explicit byte counters do not measure all physical fabric traffic.
Tiny old AiCPU int64 Sort tasks are14.62% of summed task time, not end-to-end or
full-size attribution. New tiny FP16 traces use another path and cannot establish
that dtype removes AiCPU. Tools and limits are in the profiling report.
Standalone public NamedOptimizer/owner checkpoints remain FP32/FP64; the consumer
owns FP32 masters. BF16/AMP and arbitrary-scale/convergence claims remain outside scope.

Re-entry: git status --short --branch; python scripts/status.py; read this file.
