# Reusable Attention parameter adjoints

Implementation **6b9224c829183e8aacefdfc5ccd416ecfeb4798d** passed
[eight clean qualification jobs](resident-attention-adjoints-20261003.json).
Audit: `TASK/launchers/attention_adjoints_evidence.py <full implementation SHA>`.
All qualification jobs are terminal exit0; device leases are released.

Aggressive sharded training now reuses owner-local event/fiber Attention parameter
adjoints and their connection masks after each window's completed canonical
reduction. The existing state-owner init packet gates reset. Guards validate
node mapping, parameter offsets, device, dtype, shape and storage aliasing.
KV key/value/log-bias gradients, state/message cotangents and other continuation
bridges remain window-owned. This preserves reverse-window then registry-alias
addition order, None/connected-zero handling, and optimizer boundaries.
Conservative and legacy single-device paths keep independent parameter adjoints.

Validation passed without skips:

- Six standalone FP32/FP16 cells: **160 trajectories, 2,560 windows, 640 updates**,
  compared with independent CPU FP32/FP64. Event/fiber caches, mixed profiles,
  streaming/prefill, aliases, disconnected/zero roots, full state/gradient/master/
  optimizer observations, retained boundaries and checkpoint repartition pass.
- **16** Python retention/budget checks and **32** actual-consumer comparisons
  across three families, standalone LibTorch/Python clients, Add/Attention,
  FP32/FP16, physical sample slicing and complete continued updates.
- Affected host units compiled during development; clean builds use verified
  identical objects and fresh links. Public ABI/core/CANN kernels are unchanged.

Same-lease two-card D512/B8/T4/V257 Attention, physical B2 ×4, two connected windows,
one FP32 AdamW update:

| Logical device | Projection-reuse baseline peak (bytes) | Attention-reuse peak (bytes) | Reduction (bytes) |
| --- | ---: | ---: | ---: |
| 0 | 8,073,745,920 | 7,804,784,128 | 268,961,792 |
| 1 | 7,226,584,576 | 6,957,622,784 | 268,961,792 |

This is **256.5 MiB less per card**. Loss remained exactly **7.532631874084473**;
all existing statistics, admitted/effective chunks and complete continuation
observations agree. `reused_attention_gradient_bytes` reports 537,923,020 bytes
of avoided duplicate parameter-gradient storage per backward group. Allocator
peaks and storage counters have separate meanings. Admission estimates remain
unchanged/conservative in this revision.

A separate FP16 complete-training trace recorded **53,176 operators**
(50,628 AI_VECTOR_CORE, 723 MIX_AIV, 1,825 AI_CORE), **zero observed AiCPU**.
The result is a memory/correctness qualification, not a throughput recommendation
or original-size training claim. Original B512 complete training and full-size
CPU/mixed/resident comparisons remain open. No new development/runtime failure
occurred; raw source/build/job/results and profiler CSVs remain in task artifacts.
