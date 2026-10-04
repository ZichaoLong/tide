# Original-size CPU/NPU isolation screen: retained lifecycle failure

Controller `a3e7270ec66ed043ca6ef8171e854b0d00929b1d`, workload
`e69b3bde3d53b0d6a019e89e82d1a78c3a91a7b8`. The terminal audit is in the
[JSON report](measurement-isolation-20261004.json).

`qualify-flow-isolation01` failed with exit 1. CPU solo passed its resource
monitor and case audit. The NPU consumer returned 0 and wrote a passed result,
but its monitor detected remaining group processes immediately after that exit.
Cleanup succeeded; the service cgroup is empty and the eleven-device lease is
released. The overlap stage never started. **No parallel timing is qualified.**

| Observed consumer | Warmup seconds | Measured seconds | Candidate/resident events | Acceptance |
| --- | ---: | ---: | ---: | --- |
| CPU Add inference | 213.536893 | 225.642711 | 1188500 | Solo case passed |
| Resident11 Add inference | 328.190307 | 328.715674 | 1188494 | Consumer completed; monitor failed |

Both used unchanged D2048/B512/T12/V50304, one continued warmup and measured
step, two windows each. These belong to a new NUMA-bound screen and are not
repetitions of the old unbound series. Their work counts differ. No strict
equivalence or recommended speed ratio follows.

The monitor checks for any remaining descendant before reaping exited adopted
grandchildren. The retained record does not identify those processes or their
live/zombie states, so the exact trigger in this run is unknown. The control
fix must reap exited descendants, permit only a finite natural teardown, and
continue to fail/clean up live leaks. Directed real-process tests must cover
both cases before a new immutable qualification. This changes process lifecycle
handling, not graph scheduling or any numerical contract. Existing process,
step, queue and whole-job limits remain in force; this failure is not retried
with a larger budget.
