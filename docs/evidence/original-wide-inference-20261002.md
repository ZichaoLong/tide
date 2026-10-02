# Original 17.5B Attention resident inference execution

Fixed qualified source `48e44b03ec592523d4f5e78980c5a70b86156215` passed
both stages of the bounded eight-NPU run.
[Audited identities, observations and receipts](original-wide-inference-20261002.json).
[Library/consumer qualification](resident-context-pool-20261002.md) supplies the
independent small/representative correctness evidence.

The original packet retains all 480 body nodes/2,208 edges and
D2048/B512/T12/V50304: 17,521,117,376 parameters. The standalone LibTorch FP32
resident TimedDAG/prefill candidate independently executes two connected windows,
using locality placement and 128 physical B4 groups. Logical B512 and all message,
KV and continuation semantics are retained. The eight NPUs perform online
scheduling and computation; no CPU reference event trace is supplied.

| Observation | D512/B32 prerequisite | Original D2048/B512 |
| --- | ---: | ---: |
| Parameters | 1,133,889,728 | 17,521,117,376 |
| Construction seconds | 44.877 | 817.996 |
| Complete inference-step seconds | 11.432 | 325.278 |
| Maximum NPU allocator peak GiB | 1.640 | 14.562 |

The original step produced all 12,288 outputs, processed 1,184,430 events and
reached cut 408. Loss was finite at 21.380956649780273. Every saved context pool
stayed below its per-device 8 GiB budget; NPU allocator peaks ranged
11.777–14.562 GiB, below the conservative maximum estimate of 53.798 GiB.
Peak CPU RSS was 208.770 GiB. Head workspace was 512 MiB, queue/arrivals 2,048,
KV 256 and per-device total admission 60 GiB. No CPU-fallback warning appeared.

This is one cold process and one step per shape, without warmup, independent
repetitions or a full-size CPU oracle. Development builds and separate-device
correctness/allocator work overlapped; the observed times support capacity and
execution closure, not a stable throughput recommendation or a speedup claim.
The full-size checks cover exact input/model identity, complete output counts,
continuation endpoint, finite loss and memory bounds. No fresh full-size profile
was collected. The original Add workload, full-size training, other schedules,
family/client settings and the finite CPU/mixed/resident comparison remain open.
