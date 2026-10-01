# Compact Full shards and device packing

Implementation: `2617a11e58b5ffe04cc1d752645cff0d6c15f427`.
All ten immutable-source jobs passed with exit 0; the [audit](device-full-shards-20261001.json)
checks both runtime builds, terminal runs, source/object/archive reuse, loaders,
raw logs and profiler CSV hashes. No tolerance was relaxed.

## Behavior and limits

An internal ContentFlow placement selects logical devices and each node's Full
owner. Each owner allocates compact identity/tanh/LH/SwiGLU banks. Device packing
stably collects the actually selected actions and maps global IDs to local bank
IDs; results scatter back using distinct scratch rows for padding. Selected work
counts must equal actual active events. The coordinator sends all nonempty peer
requests, performs its own Full subset, then receives the peer results. Empty
subsets skip service work; every peer receives a terminal command at window end.

Static `memory` placement balances estimated parameter bytes. `locality` applies
stable moves/swaps that strictly reduce physical-edge cuts without increasing
that maximum estimated load; parallel edges count separately. A hand-calculated
four-node fixture checks a cut reduction from six to two with two nodes per card.
Invalid owners, duplicate devices and empty shards are rejected. These generic
policies do not consume inputs or a precomputed numerical trace.

This increment shards Full parameters only. Persistent state, attention/KV,
Read, online selection and event queues remain on the coordinator. Fixed-capacity
packets still transfer padding; unselected Full computations are not evaluated.
Remote/sharded reverse is explicitly refused. Public defaults remain single-device.
This is not full state/KV sharding, distributed training, or a throughput result.
See [the protocol](../resident-peers.md).

## Independent verification

| Fixed-source check | Passed scope |
| --- | --- |
| Two-device HARD, per dtype FP32/FP16 | 84 configurations, 420 windows |
| Two-device HST/SOFTP, per dtype | 36 configurations, 180 windows |
| Three-device memory/locality × FP32/FP16 | Four cells, each three configurations/15 windows |
| One-shard degeneration smoke | Three configurations, 15 windows |
| Single-device inference regression | Four precision/control cells |
| Single-device FP16 cache training smoke | One trajectory, 16 windows, four updates |
| Python-owned single-device clients | 96 passed, no skips |

Candidates independently consume common initial state, parameters and inputs.
Existing CPU Streaming assertions cover complete observables, feedback, parallel
edges, unaligned arrivals, high int64 clocks, empty windows, capacity failures,
attention/normalized Aggregate and mixed Full profiles. Candidate continuation
also switches schedule and placement policy. Per-shard node counts, parameter
bytes, chunk rows, selected/capacity rows and retained/workspace bytes are reported.
Single-device training and Python checks are regression evidence, not peer training.
All development runs passed; no runtime failure or changed numerical threshold.

## Actual execution placement

The separate three-device FP32 profile covers three fixtures and 15 windows.
It records 4,067 AI_VECTOR_CORE, 86 AI_CORE and three MIX_AIV tasks, with no
observed AiCPU rows. All three physical devices execute Full planning and matrix
work; the coordinator additionally executes packing, readiness, selection,
attention and queues. Peers have no frame-selection operator.

There are 45 host model submissions, three per window. Device records include
165 notify records/165 waits, 713 label switches and 1,419 DMA tasks. These counts
include construction/boundaries and CPU assertions. They verify placement and
bounded submission structure, not steady-state cost, concurrency or speedup.

## Build and reproduction

Local aarch64 Ascend910_9392, CANN 9.0, Torch/TorchNPU 2.10. Standalone SDK and
Python-owned runtimes remain separate. Eight affected host objects were rebuilt
for each runtime. The new packing kernel comes from a passed development build
with matching source/archive hashes. Other kernels, core and public training
objects are reused only after byte verification against terminal successful builds.
The unchanged 8,954-check portable CPU suite was not rerun.

`TASK=/mi/data2T/zlong/tide-execution-flows`; snapshot/build `full-shards-clean01`,
Python build `full-shards-python-clean01`. Job identities and hashes are in the
audit. Bounds: queue 120s, run 600s, build 900s, profile storage 512 MiB.

```bash
python "$TASK/launchers/full_shards_evidence.py" 2617a11e58b5ffe04cc1d752645cff0d6c15f427
```

All qualification jobs are terminal. Cross-card reverse/optimizer, general
state/KV placement, public multi-device consumers and complete performance
comparisons remain under ROADMAP F1–F7.
