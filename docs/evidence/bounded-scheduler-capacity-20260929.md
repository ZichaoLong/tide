# Bounded scheduler capacity and exploratory timing

The finite assessment at clean source `f8648f766e6ded55eeb3eb886cb94bd109c55a7d`
is complete. On eight Ascend910_9392/A3 chips, full-size Add eager inference
completed in FP32 and FP16. Attention inference and Add training ran out of
device memory in both dtypes. Full-size captured replay was not launched:
the full 12-token window already exhausted notification resources with small
tensors. A subsequent 12-device FP16 Attention attempt encountered OOM while
another process occupied its failing device; uncontended 12-device fit remains
unknown.

This is a capacity assessment with two exploratory timing observations, not
full-size numerical equivalence, convergence or stable-speed qualification.
The [adjacent JSON](bounded-scheduler-capacity-20260929.json) retains commands,
configuration, identities, failures, skipped cells and resource evidence.
The [84-cell qualification](bounded-scheduler-qualification-20260929.md) remains
the independent correctness anchor. Runtime implementation is `ff251745`;
the public C++ core is unchanged.

## Eight-device full-size assessment

The workload is D2048/B512/V50304, 465 nodes, 4418 physical wires, and 12 tokens
from empty graph state. Add has 9,468,020,899 parameters; Attention has
17,269,426,339. Payload dtype varies; Read, controls, optimizer masters and
slots use FP32. Topology, seed7, locality policy, window and timing scope match
within each dtype pair. Each pair held one physical allocation. Full node maps
were not recorded, so identical node placement is not established.

| Full-size configuration | FP32 | FP16 |
| --- | --- | --- |
| Add eager inference | completed | completed |
| Attention eager inference | OOM, no observations | OOM, no observations |
| Add eager complete training | OOM, no observations | OOM, no observations |
| Attention eager training | skipped after inference OOM | skipped after inference OOM |
| Add/Attention captured inference and training | skipped after notification prerequisite failure | same prerequisite skips |

There were six native attempts: two completed and four failed. Ten configured
cells were skipped. The assessment driver's zero exit means it finished this
finite inventory; it does not turn failed or skipped cells into passing tests.

Each admitted cell requested two complete windows, first warmup. Training owners
would persist across reset-state windows. Synchronized window timing includes
input upload, forward and, for training, VJP/optimizer. Construction, preparation,
CPU input construction, finite checks and metrics are outside the timed window.
CPU affinity has 16 cores; ATen/inter-op/worker count is one. Heavy project timing
is serialized. Limits were 1800s inference, 2400s training, dynamic host RSS up
to 256GiB, and a 32768GiB conservative workspace refusal threshold. That threshold
does not reserve HBM or guarantee a fit.

| Eight-device Add eager inference | FP32 | FP16 |
| --- | ---: | ---: |
| Measured reset-window time, seconds | 39.380302 | 41.778807 |
| Milliseconds per sample-token | 6.409554 | 6.799936 |
| Largest per-device allocator peak, GiB | 11.759881 | 5.884657 |
| Construction, seconds | 91.504732 | 105.357927 |
| Preparation, seconds | 42.710230 | 43.889525 |

One process and one measured window per dtype cannot establish stable speed.
The FP16 observation took 1.061 times the FP32 time and roughly halved the largest
allocator peak. The pair also had sampled external device contention. These
whole-window times must not be divided by the older growing-context token-warmup
measurements to claim a scheduler speedup. In particular, this successful Add
path uses host operator submission with device tensor decisions; it is not a
successful full-size captured-replay result.

The four eight-device OOMs occurred before any benchmark observation. Their
failing allocator reported 52.95–55.59GiB already allocated. Reserved versus
active memory and outside load also matter; these failures are not exact
mathematical minimum-memory estimates. The finite tensor expansion can evaluate
inactive rows and retain padded state/cache entries, so its memory behavior
differs from the existing sparse eager scheduler.

## Notification limits precede captured full-size work

Actual topology, D8/B1/V17, FP32 and the full 12-token window were tested with
Add/Attention inference and complete training replay. All four two-device
attempts failed before observations with `NPU peer create notify: ACL error
207009`. The matched CANN header names this `ACL_ERROR_RT_NO_NOTIFY_RESOURCE`;
the retained project log reported 8192 allocated notification IDs. This was not
an HBM OOM.

All four eight-device follow-ups also failed before observations, at the
adapter's declared global limit of 16384 peer notification channels. Distributing
work over more chips did not remove that software limit. Consequently all eight
full-size replay cells were skipped, not tested and failed at full size. The
three-token correctness qualification does not certify a 12-token inventory.
Notification reuse or another completion protocol would need its own semantic,
ordering, gradient and replay validation before a larger replay claim.

## Twelve-device follow-up and shared load

The separate, immutable `bounded-twelve01` plan requested 12 devices after the
eight-device assessment ended. It held the timing lock and acquired devices
after 24.16s of queue wait. Two actual-topology D8/B1/V17/T3 Attention eager
checks passed: complete observations, independent isolated VJPs and None/zero,
at FP32 atol1e-6/rtol1e-5 and FP16 atol0.02/rtol0.02. This adds two finite
forward/VJP cases; it does not qualify 12-device replay or complete training.

The single full-size FP16 eager inference attempt then failed before observations.
Logical device8 mapped to physical9. Its allocator reported 33.95GiB allocated,
37.15GiB reserved, 96.98MiB free and an unsuccessful 178MiB request. Independent
samples at 11:32:43, 11:33:15, 11:33:47 and 11:34:19 UTC show an external process
on that same device. Other overlaps were also retained. Devices passed admission
checks, but outside workloads subsequently entered advisory-locked devices.

This establishes a failure under observed shared load. It does not establish
whether an uncontended 12-device run fits. Similarly, acquiring 12 devices at one
instant is not evidence that 12 devices had been continuously idle. Sampling
started at 10:38 UTC; gaps and zero instantaneous AiCore readings cannot establish
continuous idleness or exclusivity. No unrelated process was stopped.

## Audit and evidence boundary

`reports/audit-bounded-capacity-final01/audit.json` passed source, build, binary,
topology, terminal lifecycle, portable-record and Trackio SQLite comparisons.
It covers 15 experiment records: two successes and 13 retained failures, plus
the two successful 12-device forward/VJP prerequisite cases. All child groups
were terminal. Tracking used local Trackio0.35.0 in best-effort mode; recorded
steps and metrics matched raw JSONL. No dashboard was required.

Raw jobs resolve through `artifacts/device-scheduler-JOB`. The reviewed analysis
is `reports/bounded-analysis-final01/analysis.json`; its resource inputs were
copied to hashed, immutable snapshots while the separate observers continued
monitoring the CPU/NPU baseline task. The original live observer files were not
edited. The CPU/NPU repeated baseline and four full-size profiles are separate
from this terminal capacity assessment.
