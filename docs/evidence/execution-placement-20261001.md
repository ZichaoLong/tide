# Public execution placement — 2026-10-01

Clean `d412541` passes all three native builds, **8,952 CPU FP64/FP32 tests**,
10 standalone CPU CTests, 26 public Python/native NPU cases, and standalone NPU
placement/Read-precision/accelerator checks. The [manifest](execution-placement-20261001.json)
records source/build/raw-result identities and the separate profiling result.

Read, control calculation, ranking and event progression have separate requested
and resolved settings. CPU and mixed A/B/C use independent Python/C++ adapters;
CPU FP64 Read/control/ranking with NPU payloads is explicitly supported. Parameter
leaves, alias names, checkpoint identity and isolated softmax/VJP roots are retained.
Both streaming and online greedy schedules pass carried training, multi-update
SGD, checkpoint and schedule-switch cases across PDG, TimedDAG and Settle. PDG's
Python cases are correctness coverage, not an added required performance cell.

The standalone NPU placement gate completes 121 schedules and 363 updates,
including exact int64 counts/stable ranking and explicit refusals. The separate
Read-precision check covers six schedules, 18 updates and two analytic payloads.
The default accelerator check also passes forward/gradients/optimizer/checkpoint
and non-default-stream coverage. Default unset placement retains the original
independent CPU Read/selector. No CPU numerical route prepass feeds a candidate.

The separate three-schedule mixed trace contains 4,079 AIV, 862 MIX_AIV, 86 AI Core
and **66 AiCPU tasks**. These are 30 INT64 Sort tasks (1,947.50 microseconds summed)
and 36 BOOL/INT64 ScatterElements tasks (3,130.44 microseconds summed). Their
38.32% share is a fraction of summed device task time, **not total wall time**.
No host CPU fallback diagnostic occurred. Integer sort keys were not converted
to lossy FP32. This host-dispatched mixed path differs from the separately
qualified Ascend C resident selector, whose cited traces have no AiCPU tasks.

Original development failures remain archived: illegal zero region budget in
C++ test setup; Settle body/encoding parameter-name comparison; lookup of Full
on an unselected node. A later production correction reads ranking budget/count
priority from the current request layout rather than caching constructor policy;
this final clean source includes its compatible-layout regression.

This evidence qualifies placement and finite correctness/training cases. It does
not measure throughput, qualify resident backward/multi-device execution, or
complete F1–F7. The later optional resident inference interface has its own source
and qualification; it is not certified by this report.
