# Retained graph reverse through compact Full owners

Implementation: `5591319c2821f40f3f5fa93317e2d745de7e4559`.
All ten immutable-source jobs passed with exit 0. The [audit](device-full-reverse-20261001.json)
checks both runtime builds, source/object/archive reuse, loaders, terminal jobs,
raw results and profiler CSV hashes. No numerical threshold changed.

## Behavior and scope

The explicit sharded tape retains actual coordinator journals and the compact
parameter banks used by forward, on their original devices. Retained copies
remain valid after forward close and subsequent overwrites. An ordinary
single-device reverse request cannot consume the bank-free coordinator view.

The existing graph reverse loop decides stages and connections on device. A
Full-stage executor stably packs connected rows, translates global node IDs,
sends all nonempty peer requests, runs local Full reverse, and receives stage
content/comparison adjoints. Peers reuse the qualified FP32/FP16 local VJPs;
parameter partials accumulate on their Full owner. Parameter matrices do not
round-trip each stage. State/message/control/Aggregate/attention reverse and
window bridges stay on the coordinator. Empty or failed windows still terminate
all peer services. See [the protocol](../resident-peers.md).

This qualifies retained graph VJPs and physical Full-node partials. Device alias
reduction, a single canonical owner update, atomic multi-device optimizer
publication, public multi-device training and throughput remain separate work.
The checker downloads completed partials and sums aliases only for assertions;
those CPU values never feed the candidate or an optimizer.

## Completed checks

| Fixed-source gate | Passed scope |
| --- | --- |
| Two devices, FP32 | 50 trajectories, 200 retained windows, replay and boundary refusals |
| Two devices, FP16 | Same scope, actual half forward with FP32 adjoints |
| Three devices, memory/locality × FP32/FP16 | Four cells, each two trajectories/eight windows |
| One-owner degeneration | Two FP16 trajectories/eight windows |
| Single-device inference regression | Four precision/control cells |
| Single-device cache training smoke | One trajectory, 16 windows, four updates |
| Python-owned clients | 96 passed, no skips |

Independent CPU FP32/FP64 Streaming autograd covers feedback/unaligned clocks,
both schedules, None/connected-zero/combined roots, all built-in Full profiles,
normalized Aggregate, HARD/HST/SOFTP, event/fiber caches and widths 1/4/257.
Comparisons include forward observables, parameter contributions with aliases,
initial state/cache gradients, every external boundary gradient and connections.
Replay must produce exactly the previous candidate gradients. Admission rejects
insufficient tape/reverse budgets, duplicate ownership, half cotangents and misuse
of the single-device API. Empty windows and malformed stage journals terminate
all peer programs without fabricated parameter gradients.

## Profiling finding

A separate three-device FP32 profile covers two fixtures/eight retained windows
and two reverse replays per fixture. It records 60 host model submissions:
24 forward submissions and 36 reverse submissions. Full VJP planning runs on all
three cards; graph reverse stages, connected-row packing and window bridges run
on the coordinator. Device records contain 336 notify records/336 waits,
1,713 label switches and 3,258 DMA tasks.

Observed tasks: 11,562 AI_VECTOR_CORE, 366 AI_CORE, 196 MIX_AIV and **120 AiCPU**.
All AiCPU rows are coordinator `aclnnInplaceIndexCopy_ScatterUpdateAiCPU_ScatterUpdate`
with BOOL/INT64/BOOL inputs, used to scatter reverse connection flags. Their
summed profiled task duration is approximately 9.048 ms, including a 0.930 ms
first task. This sum is neither wall time nor a throughput denominator.
No logged host CPU fallback was observed; AiCPU here is on-device execution.
The trace motivates fusing flag/error/count merge into a metadata kernel.
It does not establish a speedup, overlap, full-size throughput or an AiCPU-free path.

## Retained development failures and reproduction

Both initial dtype smokes reached comparison and failed in the test-only CPU
`at::tensor(bool)` constructor; `at::full` fixed it. The second build lacked an
explicit NoGradGuard include in a new checker header. One three-device wrapper
used the old forward success marker although its first child returned zero.
The corrected wrapper then passed all four cells. These raw failures remain in
the audit; none was relabelled. No production numerical failure was observed.

Local aarch64 Ascend910_9392, CANN 9.0, Torch/TorchNPU 2.10. Standalone and
Python-owned runtime builds are separate. Six affected host objects were rebuilt
for each; the new packing kernel is reused only from a successful build with
matching source/archive hashes. Other objects and kernels are byte-verified
terminal dependencies. The unchanged portable CPU core suite was not rerun.

`TASK=/mi/data2T/zlong/tide-execution-flows`; frozen source/standalone build
`full-reverse-clean01`, Python build `full-reverse-python-clean01`. Exact jobs and
hashes are in the audit. Queue 120s, run 600s, build 900s, profile storage 512 MiB.

```bash
python "$TASK/launchers/full_reverse_evidence.py" 5591319c2821f40f3f5fa93317e2d745de7e4559
```

All qualification tasks are terminal. ROADMAP F1–F7 remain incomplete.
