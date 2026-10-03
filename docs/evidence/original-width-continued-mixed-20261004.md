# Original-width continued mixed calibration

Clean source `e69b3bde3d53b0d6a019e89e82d1a78c3a91a7b8`, 2026-10-04. Four LibTorch pilots passed; the parent job remains **failed** because the next Python Add pilot exceeded its declared900s child bound. The last three Python pilots were not started. [Audited records](original-width-continued-mixed-20261004.json).

All pilots use original D2048/T12/V50304 model dimensions and parameters, eleven NPUs, FP32 and complete SGD training. Add uses logicalB64/physicalB32×2; Attention uses logicalB8/physicalB4×2. Each process performs one continued warmup and one measured update, with two connected windows per update. The original B512 capacity plans are retained. These are reduced-batch cost/peak calibrations, not actual B512 timing.

| Client/schedule/model | Preset | Warmup seconds | Measured seconds | B512 warmup forecast | B512 measured forecast |
| --- | --- | ---: | ---: | ---: | ---: |
| libtorch-prefill-add | mixed-a | 144.437850 | 199.759663 | 1320.739057 | 1834.738455 |
| libtorch-prefill-attention | mixed-a | 40.469307 | 40.552868 | 2873.406051 | 2954.875761 |
| libtorch-streaming-add | mixed-c | 175.802385 | 198.437740 | 1609.586966 | 1822.292959 |
| libtorch-streaming-attention | mixed-b | 36.921939 | 40.807258 | 2632.015958 | 2974.940510 |

Forecasts scale only the sample phase by512/logicalB, retain one optimizer phase, and multiply by the unchanged1.15 factor. All eight phase forecasts fit the original3000s/update guard. This is a cost admission estimate, not proof that a full run will finish within that guard. Actual B512 warmup and measured steps must pass independently.

All four completed cases passed their allocator estimates, retained two physical sample contexts and reached continuation cut816. Measured output counts were1536 for Add and192 for Attention. Exact event/work counters, losses, physical shapes and complete commands are retained in the audit. The fixed mixed candidates come from the completed representative screens.

Python/prefill/Add used the same B64/physicalB32 and900s child bound as this batch. It produced no complete result before timeout. This establishes a bounded-run failure, not a measured complete-step duration or a diagnosis of AiCPU, memory or scheduling overhead. No later Python result is inferred from the LibTorch passes. A subsequent cost probe should preserve original width/physical rows, declare its reduced logical batch and new bound, and retain this failure.

`wide-mixed-continued-pilots01` is terminal/exit1, with an empty service cgroup and the eleven-card lease released. The audit checks frozen source/build/helper/packet/result hashes and all four completed cases separately from the failed parent. Original CPU Attention feasibility overlapped, and extra phase synchronization was enabled; none of these values enter a formal throughput or equal-work speedup table.

Re-audit: `python TASK/launchers/audit_mixed_continued.py`. Raw records: `TASK/runs/wide-mixed-continued-pilots01/assessment`. The accepted original-scale near-tie strict-equivalence failure remains [separately documented](original-add-route-witness-20261004.md).
