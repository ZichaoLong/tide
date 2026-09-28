# Tensor dispatch and complete-window training qualification

Implementation `b4f26b3659cec0549e18560a1a996d5bee0cae6d`, clean frozen
`perf-a4`. The public single-device core and Python package are unchanged.
This report qualifies the standalone historical-topology consumer; it does not
extend the public runtime to multiple devices. [Structured evidence](accelerator-dispatch-training-20260928.json)
records job names, counts, hashes and reviewed operator placement.

## Completed immutable gates

| Target | Configuration cells | Additional analytic dispatch gate |
| --- | ---: | --- |
| CPU, one device | 64 | passed |
| CANN9.0 / Torch2.10, two NPUs | 64 | passed |
| Same stack, four NPUs | 36 | passed |
| Same stack, eight NPUs | 36 | passed |

All 200 cells passed. These are finite configurations, not 200 independent
models. Each parity cell includes grad/no-grad comparisons and isolated VJPs;
training cells compare three complete updates. All targets also passed Add and
Attention checks on the real465-node/4418-physical-edge topology with D8/B1,
including complete training. These wide-topology gates explicitly use the
previously documented `basis-conditioned` isolated-VJP policy. Tiny gates use
strict comparisons. Reduced tensor sizes are correctness evidence, not evidence
of full-size capacity or performance.

CPU/two-NPU full scopes cover historical CPU FP64/FP32 Read with host/resident
transport, the four independent CPU/model ranking and event-queue combinations,
and extra complete-training cases for CPU FP64, CPU FP32 and model FP32 Read
with CPU controls. Four/eight-NPU scopes cover the four dispatch combinations,
both memory/locality placements for forward parity, and both optimizers/models
for training. Training tests assert values, exact discrete choices, state/history/
pending, isolated VJPs, all gradients including absent versus connected-zero,
updated weights and optimizer slots against an independent CPU scalar-slot
schedule. The analytic gate additionally checks one-ULP scores, exact int64 keys
above2^53, ties, event ready/stop boundaries, parallel-edge identities and payload
VJPs. Scores/controls are compared against the declared matching precision;
this does not promise arbitrary FP32 and FP64 executions choose identical routes.

Both clean CPU and NPU consumer builds passed their three CTests and standalone
loader checks. Development snapshots/CLI rejection results are retained in raw
artifacts; they are not substituted for the immutable results above. The unchanged
core retains its previous [8636-test CPU and four-stack NPU qualification](accelerators-20260928.md).

## Actual device execution

`profile-dispatch-a4` profiled a two-NPU Attention consumer at D32/B4/T3 with FP32
model Read/controls/ranking/events. This is placement evidence, not timing evidence.
`msprof` collected7012 operator records. Selected observations:

| Operator / input | Engine | Count |
| --- | --- | ---: |
| Sort / int64 | card-local AI_CPU | 55 |
| Sort / FP32 | MIX_AIV | 5 |
| LpNormV2 / FP32 | MIX_AIV | 41 |
| SoftmaxV2 / FP32 | AI_VECTOR_CORE | 349 |
| ReduceMin / int64 | MIX_AIV | 12 |
| NonZero / bool | MIX_AIV | 18 |
| GatherV3 / int64 payload | AI_VECTOR_CORE | 113 |

Int64 ordering stays exact; it is not approximated with floating keys. AiCPU is
on the accelerator and is distinct from host CPU and AiCore. Host code still owns
history maps, differentiable payload handles and C++ dispatch; tensor scheduling
returns indices to that host code. The new flags therefore do not make the whole
control loop device-resident. Explicit metadata transfer counters are separate
from payload counters and do not count every implicit scalar synchronization.

## Training and performance boundary

`--training-steps` measures complete synthetic cross-entropy forward/backward/
optimizer updates. Each update resets graph state, retaining full history within
the declared token window. Parameters and optimizer slots persist across updates.
HARD signaling, the historical topology and FP32 payload are unchanged. This is
independent-sequence throughput, not continuous-stream training or convergence.
See the [consumer contract](../accelerator-scale.md) for the objective, timing
scope, optimizers and flags. `TASK_QUEUE_ENABLE=0` is explicit and qualified.

These gates permit the subsequent full-size experiments; they do not by themselves
choose a faster dispatch implementation. Full-size results and capacity failures
are reported separately. CUDA hardware execution and new host architectures remain
external acceptance work.
