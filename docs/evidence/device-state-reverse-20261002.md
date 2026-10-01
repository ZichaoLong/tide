# Compact state/cache reverse and internal multi-device training

Qualified source: **49541be3c465548ac0179c29ab0239e8b37d70f6**.
State reverse implementation: **7cb79090757f0a0a46d106e51438d0a533b27fb7**.
The [audit](device-state-reverse-20261002.json) verifies ten terminal qualification
jobs, isolated standalone/Python runtimes, underlying state-reverse builds,
compiled-object/archive reuse, loaders, raw assertions and profiler CSV hashes.
No numerical tolerance was relaxed.

## Behavior and scope

Owner-local retained tapes now support state, Read, event/fiber KV adjoints,
including cache roots and bridges across retained windows. Device packing consumes
only the candidate's actual journals. Global region HST/SOFTP normalization stays
on the coordinator; Read derivatives and parameter/cache partials stay on owners.
Physical source identities, stable row order and explicit connection flags survive
packing and gradient merging. Fiber source derivatives add to Aggregate derivatives.
Canonical alias reduction, all-device atomic SGD/AdamW and publication update the
actual compact Full/state/cache banks before continued forward windows.

A bounded `CannSequence` joins one coordinator program per retained window with
reusable local device notifications. All coordinator/peer programs are submitted
before the backward-boundary wait. Device completion enforces reverse-window
order; state/cache bridges retain the complete requested gradient graph. This
avoids concatenating all windows into one persistent stream's static task buffer.
Each window still makes event/stage decisions on device. The CPU reference never
supplies routes, events, results or gradients to the candidate.

This qualifies the internal training composition, not the pending public
multi-device session/checkpoint/client or model-scale throughput. Public single-
device client regressions remain separately scoped. [Protocol](../resident-peers.md).

## Completed checks

| Fixed-source gate | Passed scope |
| --- | --- |
| Two devices, each FP32/FP16, retained VJP | 50 trajectories / 200 windows per dtype, after-close retention, replay and boundary checks |
| Two devices, each FP32/FP16, complete internal training | 40 trajectories / 640 windows / 160 updates per dtype |
| Three devices, memory/locality × FP32/FP16 | Each cell: VJP 2 trajectories / 8 windows; training 2 / 32 / 8 updates |
| One-owner degeneration, FP32/FP16 | Same VJP/training subsets |
| Program sequence/control/lifecycle/numerical | 4096 static adds over eight programs; four device-selected empty/partial/full/replay cases; int64, capacity rejection, single-program chain; prior failure/numerical gates |
| Old public half-cache training | 1 trajectory / 16 windows / 4 updates, CPU FP32/FP64 and resume |
| Python-owned public clients | 5 passed, no skips; scalar oracle and fresh-process checkpoint suffix |
| Standard CMake | Configure/link dependency generation, including new sequence target; not a fresh full rebuild |

Full gates cover both schedules, HARD/HST/SOFTP, mixed Full profiles, normalized
Aggregate, retained event/fiber cache roots, None/connected-zero/disconnected
poison, aliases, optimizer slots/counters, exact rounded publication and continued
windows. State and Full placement deliberately differ. Reverse replays must not
accumulate prior contributions. Tests include large int64, empty/error completion,
real packet capacity refusal and zero numerical gathers after refusal. Nonfinite
updates must leave every master/slot/forward bank byte-exactly unchanged.

## Separate profiling

The two-device FP32 training trace covers two trajectories, 32 forward windows
and eight updates, including assertion exports and deliberately refused updates.
Observed operators: **38,535 AI_VECTOR_CORE, 819 AI_CORE, 601 MIX_AIV; no AiCPU
operator observed**. Both devices execute state VJP; actual journal packing,
local Read adjoints, cache reverse and canonical optimizer/publication are present.

The trace contains 212 host model submissions, 2,192 matched device notify
record/wait pairs, 4,878 label switches and 17,085 DMA tasks. A runtime program may
contain many device loop iterations, and a high-level API may emit many tasks.
These counts include setup/exports/checks and are not a throughput denominator or
a CPU speedup claim. Full-size CPU/mixed/resident measurement remains F6.

## Preserved failures and limits

The original three-device FP16 memory placement failed with CANN **507002** during
construction: the persistent stream's static task buffer was full. The local
vendor HUGE flag did not fix it; both the small control stress and original VJP
reproducer retained that failure. Device program chaining passed both reproducers
and the complete directed matrix. It does not remove finite per-program or tensor
capacities, and capacity errors never authorize silently dropping work/gradients.

Earlier build failures involved stale generated kernel headers, an ambiguous Tensor
initializer and a missing checker include. Earlier large-profile tests hit explicit
forward workspace admission; compact tests now declare 1 GiB total forward budget,
2 GiB for wide tensors. Reverse budgets and numerical tolerances were unchanged.
A Python builder omitted `_tide_resident.so`; the failed client stopped before
execution. Two chain-builder failures involved old link templates/missing checker
objects. Completed compilation was reused only with identical frozen inputs;
the failed jobs remain failed. All receipts and reasons remain in the audit.

Local aarch64 Ascend910_9392, CANN9, Torch/TorchNPU2.10. The unchanged CPU core and
its 8,954-check suite were not rerun. The final builds recompile the changed chain
unit and relink byte-verified dependencies in each runtime, including the Python
extension; underlying 25-unit state reverse/five-kernel builds are audited at their
own immutable source. No new qualification job remains live.

Local reproduction: `TASK=/mi/data2T/zlong/tide-execution-flows`, frozen source
`TASK/sources/program-chain-clean01`, builds `program-chain-clean01` and
`program-chain-python-clean01`; audit command:

```bash
python "$TASK/launchers/state_reverse_evidence.py" 49541be3c465548ac0179c29ab0239e8b37d70f6
```

Exact commands, hashes, assertions and physical-to-logical mappings are in retained
receipts. Queue120s, run600s, build900s; profile storage512MiB. F1–F7 remain incomplete.
