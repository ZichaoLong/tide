# Eager admission and measured CPU RSS calibration

Clean source `c68609603c310f7121cb6f887afcadd019973209` qualifies the
[eager admission contract](../eager-consumer-capacity.md). The
[machine audit](eager-consumer-capacity-20261003.json) verifies eight terminal
jobs, all1530 source hashes, fresh installed clients, byte-identical matching
core archives, loader closure, test counts, calibration results and trace.

| Gate | Result |
| --- | --- |
| CPU FP64/FP32 | 70 passed,47 deselected,0 skipped; static Python/C++ backend parity, traffic bounds, capacity refusals and complete consumer comparisons |
| Two-device capacity | 12 passed,22 deselected,0 skipped; forced physical splitting preserves independent CPU full records/gradients/updates |
| Actual multi-owner consumers | 40 passed,9 deselected,0 skipped; three families,both schedules,mixed A/B/C,Python/native/standalone,two updates and connected windows |
| CPU original-width calibration | Add and Attention,480 body nodes,D2048/T12/V50304,B4/physicalB2,two connected windows,one complete FP32 SGD update |
| Two-device calibration | Seven fresh processes: D256 Add/Attention in Python/C++; D2048/six-node Add/Attention C++ and Attention Python;two SGD updates/two windows |
| Separate profile | Actual two-device mixed-C Attention consumer,two complete AdamW updates;normal cleanup,0 observed host-compute fallback diagnostics |

The CPU correction charges6.25% of learned storage for retained host allocations.
The prior original-width observations had up to4.3% unmodelled construction
storage and an Attention total peak0.51% above its estimate. Accelerator
allocated-byte estimates,512GiB CPU/16GiB calibration-card budgets,4GiB head cap,
and10%/25%+128MiB margins are unchanged. This is a measured envelope, not a proof
against external memory pressure or arbitrary later workloads.

The new original-width CPU processes pass both construction and complete-run
checks. Values below are incremental process-lifetime peak RSS in GiB.

| Model | Construction observed / estimate | Complete observed / estimate | Complete update seconds |
| --- | ---: | ---: | ---: |
| Add | 39.709 / 41.079 | 97.503 / 116.059 | 50.543220376 |
| Attention | 71.656 / 72.954 | 213.050 / 216.041 | 254.676481222 |

These use all9,468,053,696/17,521,117,376 parameters,16 ATen threads/one node
worker,packed sources/batched Next,outputs96 and final cut408. Losses are
31.585994720459/20.078292846680. The calibration overlapped small independent
NPU qualification work; timings are incidental, not formal performance.
NPU D2048 calibration reaches3.728GiB maximum allocator growth; it has only
six body nodes and does not certify original480-node or B512 device capacity.

`wide-eager-cpu-pilot01` remains failed/exit1 at its original source2c04005;
its [underestimate](original-width-eager-cpu-calibration-20261003.md) is not
rewritten. `eager-capacity-npu-clean01` also remains failed/exit1 (6 failed,
6 passed,20 deselected). Its Python/native training tests first initialized
CPU autograd, then registered the NPU plugin, triggering Torch2.10's device
queue assertion. The corrected test preflights its explicit backend before
the independent CPU reference. No candidate consumes CPU events or gradients.

Standalone NPU processes explicitly use `ACL_OP_INIT_MODE=0`, following the
[qualified compiler lifecycle](eager-packed-transfer-20261003.md). The separate
trace records14,104 operators,including128 AiCPU tasks:80 Boolean/INT64
ScatterElements and48 INT64 Sort. Both devices appear; integer semantics remain
exact. These instrumented observations are not throughput measurements.

Raw jobs are `TASK/runs/NAME/{status.json,task.log}`, names in the audit;
source is `TASK/sources/eager-rss-clean01`, installed builds
`TASK/builds/eager-rss-{cpu,npu}-clean01`. Re-audit with
`python TASK/launchers/eager_capacity_evidence.py c68609603c310f7121cb6f887afcadd019973209`.
All accepted services are inactive with empty cgroups and completed leases.
B512 CPU/mixed runs,formal comparisons,eager FP16 training and final integration
remain open. CUDA hardware/x86_64 remain target-machine work.
