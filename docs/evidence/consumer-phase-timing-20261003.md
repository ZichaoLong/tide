# Synchronized consumer phase timing

Implementation **26176de888013fda5eccfe039fa504e87c2e7e95** passed
[four clean qualification jobs](consumer-phase-timing-20261003.json).
Audit: `TASK/launchers/phase_timing_evidence.py <full implementation SHA>`.
Both builds and both test services terminated with exit0 and empty control groups;
the two-card lease was released. This qualifies diagnostic instrumentation,
not larger-batch execution or a throughput improvement.

`--phase-timing` is available in the shared launcher and standalone C++ consumer;
the Python consumer accepts `phase_timing=True`. It defaults off, emitting empty
phase arrays and adding no synchronization. Enabled training synchronizes every
resolved device once after all physical sample chunks and connected windows,
before final finite checks, the single logical optimizer update and publication.

- Sample work includes input preparation/upload, forward/loss, backward,
  accumulation, continuation and existing per-chunk checks/eager zeroing.
- Optimizer time includes final finite checks, update/publication and the existing
  final synchronization. Inference records all elapsed time as sample work and
  zero optimizer time, without a new middle synchronization.

Measured and warmup arrays each align with the existing complete elapsed times.
Their two components sum to that elapsed time; the extra training synchronization
is included. No numerical operation, update boundary, scheduler or device kernel
was changed. Core and resident public ABI are unchanged. This split allows a
future measured cost model to distinguish sample-scaled and once-per-update work;
it does not establish that either scales exactly or approve a large run by itself.

| Qualification | Result |
| --- | --- |
| CPU FP32/FP64 | 13 passed: 12 paired entrypoint cases and boolean validation |
| NPU FP32/FP16 | 8 passed: mixed A/B/C and single/two-device resident paths |
| Actual CLI candidates | 40: 20 default/enabled pairs, no skips |

Each executed candidate independently constructs the common model/input and runs
two physical sample chunks, two connected windows per step, one warmup and two
measured steps. Training uses complete AdamW updates. The tests cover Python,
native Python clients and standalone LibTorch, all three families and both
schedules without expanding every combination. Independent CPU streaming checks
all exposed states, histories, pending messages/routes, outputs, None/zero
gradients and every parameter generation. Default/enabled candidates retain the
same losses, work counters, final cut and sample grouping. FP16 uses the existing
cross-dtype tolerance; no CPU events or gradients feed any candidate.

The local target is aarch64, Torch/TorchNPU2.10.0, CANN9.0.0, Ascend910_9392.
Consumer objects are source/header/options verified from the passed development
builds, with fresh clean-source links and installed-package loader checks.
Core and resident/CANN component hashes match the already qualified dependencies;
there is no claim of a fresh backend compilation, new profile or CUDA validation.
All test timings include diagnostic exports and are excluded from throughput
comparisons. Original B512 complete training remains open; earlier cost refusals,
the 3000s cap and the1.15 safety factor remain unchanged.

The first development CPU job failed during collection because the new NPU test
argument `dtype` collided with the repository's global dtype parametrization.
Only that argument was renamed to `payload_dtype`; frozen dev02 passed CPU13/NPU8
before the implementation commit. The original failed job and reproducer remain
retained and are hashed in the evidence, rather than relabeled as passing.
