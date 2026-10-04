# Original-size isolation screen: execution passes, performance screen rejected

Controller `103f5b6c8e9b2e6e46a7f2733185433de2cb5f3f`, unchanged workload
`e69b3bde3d53b0d6a019e89e82d1a78c3a91a7b8`. The terminal
[report](measurement-isolation-result-20261004.json) retains every raw-record hash.

`qualify-flow-isolation02` completed all four original B512 Add inference
processes and the resource/lifecycle audit, exit0 at
2026-10-04T02:42:16.833543Z. The cgroup is empty and all eleven NPU leases are
released. The monitor reaped two successfully exited descendants in each NPU
process, with **zero natural-teardown delay**. All sampled thread masks and
private anonymous NUMA pages stayed within their lanes; no RSS limit was exceeded.
The [earlier failed screen](measurement-isolation-20261004.md) remains failed.

| Flow / condition | Warmup seconds | Measured seconds | Construction seconds |
| --- | ---: | ---: | ---: |
| CPU solo | 245.199256 | 194.385205 | 45.534829 |
| CPU overlapping NPU | 275.404938 | 233.628268 | 47.399889 |
| Resident11 solo | 324.879933 | 325.315243 | 85.419058 |
| Resident11 overlapping CPU | 327.513912 | 329.014670 | 71.671169 |

The measured overlap/solo ratios are **1.201883 CPU** and **1.011372 NPU**.
The predefined threshold is 1.05 for each lane, so the performance result is
**screen-rejected**, despite successful execution. Subsequent formal cases stay
solo. No wider workload class or concurrent recommendation is qualified.

CPU masks80/NPU-host78 plus two controller slots respected the aggregate160-core
allowance. Private memory used nodes0–3/4–7. Declared lane RSS caps were
122.559/280 GiB plus136 GiB shared reserve. Sampled overlapping peak was
143.365 GiB. The process lifetimes overlapped557.835 seconds, but that does not
prove uniform overlap of the measured phases. Shared file pages, physical NPU
host locality and unrelated server activity remain uncontrolled.

Within each backend, configurations, thread environments, physical NPU mapping,
losses, outputs and all work counters matched exactly between solo/overlap.
CPU had1188500 candidate events;resident had1188494. The comparison does not
establish strict cross-backend equivalence. Each process retained D2048/B512/T12/
V50304, one continued warmup and one measured step, with two windows per step.

This single matched screen is not a three-process recommendation or a proof of
why the CPU timings changed. CPU warmup also increased, and prior independent
single-process CPU observations varied. No further worker sweep, automatic
retrial or budget increase follows. The process-control correction is qualified;
full-size matrix work continues under the declared solo policy.
