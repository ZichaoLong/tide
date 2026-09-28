# Standalone NPU node-sharding qualification

Implementation **bb0ecc6bd98c94c868ad7707bec43248b9ddbd2e**, frozen as perf-a2.
[Machine-readable identity and gates](accelerator-scale-20260928.json).
The installed public core is unchanged from the prior accelerator qualification.
These are consumer correctness/transport results, not full-size throughput.

## Qualification

Both clean standalone CPU/NPU builds passed, with no Python or stub libraries
in the loader closure. The VJP verifier's CPU suite accepts identity cases and
rejects injected wrong Jacobians, None-vs-zero changes and nonfinite gradients.

| Gate | Cells | Numerical policy | Result |
| --- | ---: | --- | --- |
| CPU,2 NPU,8 NPU; tiny Add/Attention × resident/host × memory/locality |24|strict|passed|
| Exact465-node topology; D8/B1/V17; Add/Attention × seeds0/7 ×2/8 NPU |8|basis-conditioned|passed|

Each cell checks both no-grad and autograd execution against the independent CPU
scalar-slot schedule: complete traces, state/cache, routes/history, pending,
logits, isolated VJPs, None/zero connectivity, owner aliases and device residency.
The wide tensor dimensions are deliberately small correctness fixtures.
No full-size equivalence, optimizer step or checkpoint import is certified.

Runtime: A3 Ascend910_9392,aarch64,TorchNPU2.10.0,CANN9.0.0,driver25.3.rc1,
standalone SDK,process-local TASK_QUEUE_ENABLE=0. This dispatch policy avoids an
observed default-queue crash in cross-device rtStreamWaitEvent. It does not stage
messages through CPU. Original failed jobs remain failed.

Explicit SDK finalization after local tensors/worker destruction resolved the
intermittent8-device exit failure in the subsequent dev05,dev08 and immutable-a2
checks. A later duplicate-finalize warning from the installed core is retained;
all reported passes require normal exit. Other SDK/driver tuples are unverified.

## Numerical boundary

The default strict quadratic VJP test still fails wide Add seed0:
max_abs7.7188e-6,tolerance ratio3.06184,first Full root to embedding/input.
Seed7 shows the same sensitivity (max_abs1.56164e-5,ratio2.69894).
Neither failed seed was replaced. The explicit FP32 RMS expansion did not help.

A single-node SiLU/RMS expression, with identical FP32 inputs and no graph or
transfers, reproduces the seed0 mismatch exactly. CPU FP32 itself differs from
CPU FP64 by9.20483e-6 (NPU FP32 by1.37738e-5). The normalized squared norm's
backward subtracts large terms. Its complete local Jacobian agrees within the
original componentwise tolerance: maximum ratio0.03123.

The explicit `basis-conditioned` policy verifies every root-coordinate VJP with
rtol1e-5/atol1e-6, then reconstructs each quadratic VJP in CPU FP64 and checks
error against `atol + rtol * sum(abs(J_ij * cotangent_j))`. It is limited to
roots of at most64 coordinates and retains strict zero VJPs, exact connectivity,
all forward comparisons and every discrete check. The wide fixture's complete
first-root basis passes (maximum ratio0.117 across the recorded seeds/runs).
All other root VJPs pass directly. Logs retain each strict quadratic failure.
This is a benchmark-specific numerical policy, not a passing strict quadratic
claim or a change to public core tolerances, model operators or FP32 precision.

## Message paths and locality

Resident state, KV caches and local operations stay on assigned NPUs. Same-device
messages remain local; remote inputs use Torch device-to-device copies. CPU owns
scheduling metadata and exact historical FP64 Read from the necessary vectors.
`host` is an explicit earlier comparison path. Neither path uses CSR pooling.

In the tiny2-device fixture, locality reduces physical cuts30/60 to24/60.
Last-token remote message bytes fall1408→1024 for Add and1408→1088 for Attention;
CPU Read transfer volume is unchanged. These observations establish functioning
placement and counters, not a speedup or a full-size locality result.

The earlier same-stack tiny msprof trace records1059 MEMCPY_ASYNC D2D events
(240940 bytes). It includes internal copies, so it does not measure only peer
traffic or link bandwidth. Adapter counters separately classify cross-device
message payloads. No HCCL collective is involved in this single-process executor.

The unchanged recording wrapper previously passed a local Trackio success case
and an intentional timeout case retaining73 partial observations with no live
child processes. Raw manifests/JSONL remain authoritative; Trackio is a local
projection. Full-size17B/8.8B pilots, placement timing and independent concurrent
models are separate experiments whose results are still pending.
