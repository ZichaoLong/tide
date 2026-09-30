# Packed device stages and runtime lifecycle — 2026-09-30

Clean source `eff5945c741886bf15d6095da38f44ab565ab8e0`. The [manifest](device-packing-20260930.json)
records all eight terminal jobs, source/binary/report hashes and operator placement.
Standalone aarch64 LibTorch-NPU2.10/CANN9.0.0, Ascend910_9392; no Python library
or stub dependency in the component executables.

The CPU core build passed six CTests (lifecycle, Greedy and Settle × FP32/FP64).
The independent standalone NPU core and component builds passed; four component
CPU CTests and loader checks passed. Eight fresh NPU processes passed nested
runtime ownership, explicit idempotent finalization and rejected reopening.
This checks the repair for a SIGSEGV caused by a static NPU finalizer running
after thread-local stream destruction. Original development failures remain failed.

All11 component cells passed with zero process exit codes: eight control cases,
four numerical cases each FP32/FP16, eager queue checks,60 closure cases,46 queue
transactions each FP32/FP16,37 broadcast-routing cases each FP32/FP16 and54 ready
packing/progression cases each FP32/FP16. The new stages preserve exact integer
coordinates, stable ties, physical parallel edges, present zeros, live capacity,
complete fibers/region-time candidate sets and pending data across windows.
Capacity, malformed coordinates, real arrival overflow and loop exhaustion are
explicitly refused. These are inference components without an autograd contract.

Separate msprof probes passed:

| Scope | Custom tasks | Placement |
| --- | --- | --- |
| Queue transaction |51 proposals | All218 recorded device tasks AI_VECTOR_CORE |
| Broadcast delivery |37 route +37 queue tasks | All394 tasks AI_VECTOR_CORE |
| Ready packing and device drain |65 closure +65 pack +11 queue tasks |538 AIV tasks;2 AiCPU OnesLike setup tasks |

No host fallback warning appeared. Integer eager sorting in the older independent
queue check still uses device-local AiCPU. Profiles include setup, input writes
and CPU assertions, and are not steady-state throughput measurements. Kernel
metadata processing remains scalar AIV code and needs performance optimization.
The profile driver also rejects msprof's child-failure warning: msprof itself can
exit zero even when the application fails after printing an acceptance marker.

The device loop here consumes ready batches **without producing new graph
messages**. Broadcast routing is independently checked from supplied selected
Full outputs. These results do not establish a complete numerical graph loop,
node selection, full state semantics, attention, sparse slot delivery, peer
progression, training/VJPs/optimizer, large-model memory budgeting or performance.
Timeout/quarantined-resource failure injection remains pending. New selector and
multi-queue transaction development is outside this immutable source's scope.
