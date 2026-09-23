# Comparable native operator timing

Source: `2619ed3fc79fe299e1a573b5088c552748f87196`.
Frozen source and records: `operator-profile-20260923-105551` under the project
qualification and artifacts directories. Driver: artifacts/qualify_operator_profile.py.
Terminal service `tide-operator-profile-20260923-105551` is inactive/dead,
MainPID0, Result=success, ExecMainStatus0. status.json and pipeline.json passed;
terminal-audit.json verifies the clean source and all12 native binary hashes.

The directed frozen CPU FP64/FP32 gate passed67 tests/70.94s. Earlier directed
development passed178 tests/85.20s, including exact/single full-state checks,
serial/parallel values, counters, and nested exclusive/thread-local timing.
Fresh original LH small and wide copies built from a10fdb1. Small complete logits
match the older anchor exactly; counting/profiling on/off and1/4 worker variants
pass. Only owned copies receive instrumentation. No original LH repository changed.
This is directed timing qualification, not a new complete CPU regression claim.

## Fixed workload

17,269,426,339 parameters, D2048/B512/V50304, FP32/no_grad, four attention heads,
12 tokens/warmup4, fixed IDs/seed7. Compare indices4–11; divide elapsed time by512.
Common CPU affinity160–319, 1024GiB address-space cap, 1200s per native process;
PDG160 node/head workers in separate phases and ATen/OpenMP/BLAS1; LH's original
parallel schedule requests160 OpenMP/ATen threads. Source/binary/input hashes,
individual observations, loads, limits and degraded best-effort Trackio state are
retained in each run. No weights were imported between engines.

| Engine | Detail timers | ms/sample-token | Peak RSS GiB |
| --- | --- | ---: | ---: |
| LH | off | 25.10840 | 148.65564 |
| PDG exact | off | 30.36451 | 109.44319 |
| PDG exact | on | 28.64634 | 109.44284 |
| LH | on | 23.88574 | 148.43994 |

Unprofiled PDG is20.93% slower in this one paired window. Profiling on/off changes
are −5.66% for PDG and−4.87% for LH; these are observations under a changing shared
host, not negative profiler overhead or evidence that enabling timers is faster.
Within each engine, all12 token model/work/operator inventories and output sums
are exactly unchanged by the switch. Complete small values/states provide the
numerical anchor; large sums alone are not a correctness proof.

## What the new timers show

Seconds below are mean **summed exclusive calling-thread elapsed duration per
batch token**, including descheduling. They are not CPU time or critical-path
wall duration. They cannot be added to coordinator wall intervals or subtracted
between engines to explain the latency gap causally. Boundaries and known
mismatches are specified in [operator-profiling](../operator-profiling.md).

| Timed work | LH worker seconds | PDG worker seconds |
| --- | ---: | ---: |
| QKV | 262.051 | 219.675 |
| KV construction | 7.378 | 49.591 |
| Explicit KV gather | 38.702 | 34.632 |
| Attention, including implicit matmul packing | 135.833 | 381.448 |
| Pooling | 10.234 | 6.861 |
| Output projection | 43.418 | 62.498 |
| Emit Linear | 339.897 | 685.889 |

PDG coordinator event preparation.303320 + publication.420099 + cleanup.404022
=1.127441 seconds/batch-token. Aggregate, local state assembly and Read/Next also
have separate worker records; see analysis.json for the complete inventory.
Pool calls are74857.125 for PDG versus921.875 for LH, but this measured pooling
scope is small. Call count alone did not establish pooling as the main bottleneck.
QKV timing also fails to support a blanket claim that PDG projections are slower.

The profile redirects attention toward memory movement/layout and scheduling
alongside event/state management. Original LH uses `at::parallel_for(0,232,1,...)`
per cortex; PDG schedules across the unified graph with its own node pool. Equal
requested160 threads do not imply equal active dense concurrency. The different
Emit grouping and concurrency prevent worker sums from being interpreted as an
intrinsic2× slowdown of its matrix operation.

The original PDG temporary KV `[sample,row,head,D]`, after permutation, cannot
merge sample/head as a view in the general bucket shape. A small FP64/FP32 actual
Torch probe confirms that reshape copies, while stacking `[sample,head,row,D]`
allows a view, with exactly equal values. Artifact: attention-layout-probe.json.
This proves a layout/materialization difference, not an end-to-end speedup.
Independent opt-in ablations now being developed cover head layout, immutable
KV reuse, CSR pooling, deferred container retirement and projection storage;
a lower worker-count case will separately test concurrency pressure.

Retain the initial failed namespace-error development archive at
artifacts/operator-profile-dev-20260923-104828 and its corrected passed retest
at artifacts/operator-profile-retest-20260923-105234. Never relabel the failure.
