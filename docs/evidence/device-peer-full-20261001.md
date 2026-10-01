# Device-side remote Full inference

Implementation: `a8fa371e0088b9e3f1b2ca8cb8b540f6211064db`.
All eight immutable-source tasks passed with exit0. The [audit](device-peer-full-20261001.json)
checks source, both runtime builds, rebuilt/reused archive members, loaders,
terminal runs, raw logs and profile CSV hashes.

## Behavior and scope

The internal ContentFlow peer constructor keeps online readiness, selection,
state, attention, emission and queues on the coordinator NPU. A second NPU owns
and executes the existing packed identity/tanh/LH/SwiGLU Full banks. Actual
selected actions, content and comparison values travel through device-loop
packets; returned values feed subsequent device stages. A terminal command
exits the peer loop on empty and failed windows as well as successful work.
Host work is construction, input/output boundaries and submit-all/wait, with no
per-stage numerical dispatch. See [the protocol](../resident-peers.md).

This is a remote Full phase, not general owner/state sharding or peer training.
Remote Full reverse requests explicitly fail. Public single-device defaults are
unchanged. Both endpoint packet buffers are budgeted; fixed-capacity padding is
transferred but unselected node computations are not evaluated. Reduced
communication and concurrent shard execution remain future work.

## Independent checks

| Fixed-source check | Completed scope |
| --- | --- |
| Peer HARD, each of FP32/FP16 | 84 configurations, 420 windows |
| Peer HST/SOFTP, each of FP32/FP16 | 36 configurations, 180 windows |
| Single-device inference regression | Four precision/control cells |
| Single-device half-cache training regression smoke | One trajectory, 16 windows, four updates |
| Python-owned single-device precision/training client | 96 passed, no skips |

Peer candidates independently consume common inputs, initial state and parameters.
CPU Streaming produces only expected observables, never candidate events or routes.
Checks cover both schedules, attention and normalized Aggregate, all built-in Full
profiles, mixed Full nodes, positive-delay feedback, unaligned arrivals, int64
clocks above2^55, CPU/device input payloads and zero-length initial windows.
Candidate-only continuation is restored with a changed schedule and physical
attention tiling. State/history/messages/pending/output/trace comparisons retain
the existing tolerances. HARD checks also require three actual device capacity
refusals, poisoned snapshot export, orderly peer close and explicit adjoint refusal.

Single-device half training and Python regressions protect the shared ContentFlow
layout and CANN lifecycle changes. They are not multi-device training evidence.
No implementation/runtime development failure occurred in this increment; the
prior transport increment's separate failures remain in its original evidence.

## Build and placement

Local aarch64 Ascend910_9392, CANN9.0, Torch/TorchNPU2.10. Standalone SDK and
Python-owned runtimes are separate. Four content objects were rebuilt for each;
Python also rebuilt the CANN program and peer objects. The standalone build
reuses byte-matched qualified control/peer archives. Unchanged core, vendor
kernels and public training objects come from successful terminal builds, with
source, object and archive checks. The unchanged8,954-check CPU suite was not rerun.

The separate FP32 profile uses two attention fixtures and ten windows, including
empty windows and continuation. It records2,435 AI_VECTOR_CORE,76 AI_CORE and
2 MIX_AIV tasks; no AiCPU or logged CPU fallback was observed. Device CSV rows
cover both leased physical devices. Full planning,15 BatchMatMulV2 and15 Tanh
operations appear on the peer; closure/readiness/selection/queue and attention
operations appear on the coordinator. The host submits20 models (two per window).
Device records include76 notify records/76 waits,461 label switches and744 DMA
tasks. Counts include setup/boundaries and are not steady-state or throughput data.
This establishes observed placement, not speedup or absence of host boundary work.

## Reproduction

`TASK=/mi/data2T/zlong/tide-execution-flows`; frozen source is
`TASK/sources/peer-flow-clean01`. Build directories are `peer-flow-clean01` and
`peer-flow-python-clean01`. All eight job names, commands and terminal hashes are
in the audit. Bounds:queue120s, build900s, run600s, profile storage512MiB.

```bash
python "$TASK/launchers/peer_flow_evidence.py" a8fa371e0088b9e3f1b2ca8cb8b540f6211064db
```

All qualification jobs are terminal. Full graph sharding, cross-card reverse and
optimizer publication, public multi-device consumers and complete performance
comparisons remain under ROADMAP F1–F7.
