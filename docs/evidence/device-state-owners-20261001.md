# Compact state, Read and KV owners

Source: `62935e98772fdacff742be915d588109aee63300`.
[Machine-readable audit](device-state-owners-20261001.json).
[Contract](../resident-peers.md#compact-state-read-and-kv-owners).
All nine immutable qualification tasks finished PASSED/exit0. Implementation and
qualification evidence are separate commits. No throughput claim is made here.

## Behavior and boundary

Internal `ContentFlow` can place Full and state/Read/KV on independent static node
owners. State values/clocks, Read/Upd banks and persistent event/fiber caches have
compact owner storage; no complete coordinator state/KV replica is constructed.
The coordinator keeps online queue/readiness, global region selection, Aggregate,
history and emission. Every candidate independently consumes the common input,
parameters and initial state; no CPU reference event/result is a candidate input.

An explicit kernel view preserves physical ports/parallel edges and source slots
while mapping local node IDs monotonically. It is not a Graph and is not compiled
as one. Device packing moves complete fibers and their payloads. Per-message host
placement and CPU-driven inner stage progression are absent.

Remote owners propose Read, receive global selection and propose state/cache
adoption, then receive the common commit decision after all downstream preflights.
Errors preserve all live state/KV; empty/refused windows still terminate every
service. Snapshots can restore under another layout. Padding has independent zero
source/discard rows; capacity errors do not silently split an attention group.

The old monolithic reverse/state/publication interfaces explicitly refuse this
placement. Compact retained state/cache adjoints, canonical publication into the
new banks, public multi-device training/checkpoint clients and complete consumers
remain pending. The separately qualified Full-only training path is preserved.

## Qualification

| Fixed job | Observed scope |
| --- | --- |
| build-state-owners-clean01 | Standalone runtime; 15 affected host objects authenticated against passed development source/header/object hashes and relinked |
| build-state-owners-python-clean01 | Python-owned runtime; 15 affected host objects rebuilt independently |
| state-cmake-clean02 | Standard CMake configure and executable/shared-library dependency generation; separate from compilation |
| state-owners-clean01 | Six cells: two dtypes × HARD forward, HST/SOFTP forward, state/KV transaction |
| state-policy-clean01 | Three devices, memory/locality Full plans, different state owner map, both dtypes; four cells, each3 configurations/15 windows |
| state-single-clean01 | One-owner degeneration, both dtypes; each3 configurations/15 windows |
| state-profile-clean01 | Separate two-device FP32 profile, three configurations/15 windows |
| state-public-clean01 | Existing public single-device half-cache training:1 trajectory/16 windows/4 updates |
| state-python-clean01 | Existing independent scalar oracle and fresh-process half-training checkpoint:5 passed,0 skips |

For **each dtype**, the two-device gate passes84 HARD configurations/420 windows
and36 HST/SOFTP configurations/180 windows. Both streaming and online greedy
prefill, feedback/nonaligned delays, parallel edges, initial caches, large int64
coordinates, present zero, vector/scalar state/Read, normalized Aggregate,
event/fiber attention, LH/SwiGLU and continued/restored execution are covered.
Forward comparison uses an independent CPU streaming schedule with the matching
payload dtype and the existing tolerances; discrete traces/history/pending are
also compared. These are finite fixtures, not a claim of exhaustive topology tests.

Each dtype also passes28 direct transaction windows, including16 refused commits,
8 accepted updates and4 empty windows. State values, clocks, cache slot tensors and
presence remain byte-identical after rejection; accepted updates agree with the
independent CPU reference. Separate packing checks cover exact large int64 time,
physical parallel-edge order, NaN padding avoidance and whole-fiber capacity
refusal. Each HARD gate additionally checks three real output/stage/journal
capacity failures and refuses the incomplete reverse interface.

Both runtime builds reuse authenticated terminal core/CANN dependencies. The three
new kernels are authenticated against passed `state-owners-dev01` source and
library hashes. This is affected recompilation/relinking, not a fresh rebuild of
all unchanged core and vendor kernels. Standalone SDK and Python-owned runtimes
are kept separate. Stack: aarch64, Torch/TorchNPU2.10.0, CANN9.0.0,
Ascend C target `Ascend910_9392`, task queue disabled.

## Device trace

Observed5283 AI_VECTOR_CORE,110 AI_CORE and3 MIX_AIV tasks; **no AiCPU observed**.
Both devices execute `tide_state_read`, `tide_content_state`, `tide_event_cache`
and `tide_fiber_commit`. The coordinator additionally performs compact pack,
selection mapping and global event-identity merge. Record/wait notifications match
at305 each. There are45 host model submissions for15 windows,4932 device switches
and2064 asynchronous DMA tasks.

Construction, input validation, checkpoint exports and CPU comparisons are inside
this trace. Those switch/synchronization counts do not describe a steady-state
inference loop and do not establish a CPU/NPU speed ratio. Capacity-sized packets
and owner-local KV staging copies remain measured optimization targets; no cache
truncation or precision change is hidden behind them.

## Preserved failures and audit

- `build-state-owners-dev02`: new test omitted the required CPU Streaming Options
  argument; fixed without changing candidate mathematics.
- `state-gates-dev03`: new fixture changed ports on an already compiled graph,
  retaining an invalid slot layout; rebuilt the fixture from a fresh Graph.
  Packing checks passed before the fixture error.
- `state-cmake-clean01`: preflight incorrectly required static kernels on the
  consumer link line; they correctly reside in `libtide-resident`'s link closure.
  CMake configure succeeded; corrected closure audit passed as clean02.

Original failures, frozen sources and logs remain. No numerical tolerance changed.
Audit command: `TASK/launchers/state_owners_evidence.py 62935e98772fdacff742be915d588109aee63300`.
It checks clean source, runtime/core identities, archive members, reused objects,
new kernel bytes, binaries/loaders, all terminal receipts, case/log hashes, CMake
closure and raw profile CSV hashes. Raw records remain under
`TASK=/mi/data2T/zlong/tide-execution-flows`; only reviewed reports are committed.
