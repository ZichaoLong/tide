# Original-B512 bound Add prefill results

Five independent FP32 processes passed their consumer,monitor and terminal evidence checks on workload `e69b3bde3d53b0d6a019e89e82d1a78c3a91a7b8`,using measurement controller `103f5b6c8e9b2e6e46a7f2733185433de2cb5f3f`. The six-cell parent is retained as **failed**:its final cell98 could not acquire eleven NPUs and never started. [Reviewed audit and dependency records](formal-b512-bound-add-prefill-20261004.json).

The five results belong to `numa-bound-solo-v1`. They are separate from prior unbound serial timings and from the rejected CPU/NPU overlap screen. CPU uses80cores onNUMA0–3,ATen/OpenBLAS16;resident uses78host cores onNUMA4–7 and eleven NPUs. Thread masks/private anonymous placement were sampled,with explicit RSS and time bounds. External load and shared file mappings remain uncontrolled.

| Cell | Family/client/flow | Warmup s | Measured s | Input tokens/s | Actual events |
| --- | --- | ---: | ---: | ---: | ---: |
| 12 | PDG / LibTorch / CPU | 246.612680 | 260.290475 | 47.208796 | 1188500 |
| 14 | PDG / LibTorch / resident11 | 328.608589 | 329.307822 | 37.314631 | 1188494 |
| 48 | Settle / LibTorch / CPU | 254.448579 | 214.649084 | 57.246925 | 1188500 |
| 50 | Settle / LibTorch / resident11 | 329.739408 | 331.282152 | 37.092249 | 1188494 |
| 96 | Settle / pure Python / CPU | 623.105501 | 834.559428 | 14.723936 | 1188500 |

Original D2048/B512/T12/V50304,480body nodes,2208body edges and9,468,053,696 Add parameters are unchanged. Each process independently performs one continued warmup and one measured step,two windows per step,12288 measured input/output tokens and final cut816. No CPU reference routes/events/gradients,profiling or phase instrumentation feed these timings. Construction is separate in the JSON. Physical sample rows are32 for CPU and4 for resident.

In these single observations,resident took 1.265times the PDG CPU time and 1.543times the Settle LibTorch CPU time:longer elapsed time. These are descriptive ratios,not recommendations or strict equal-work speedups. CPU records1188500 candidate events versus resident1188494 events. Retain the [strict near-tie numerical limitation](original-add-route-witness-20261004.md);one located witness does not establish the cause of every discrepancy.

Cell96 has no newly measured Python-owned native NPU counterpart:cell98 is unstarted. The pure Python client and Python-owned native resident client must remain labelled separately.

The final queue requested11 devices with max-wait120s. Repeated queue observations reported9 free devices;the final `timed_out` record was written after137.089s,including polling granularity. Its child exit was3;the parent exited1. No cell98 consumer directory exists. Passed group positions0–4 were audited individually;the failed parent was not relabelled passed. The failed parent and its dependency successors have empty cgroups;the two earlier NPU leases are completed.

The first audit attempt rejected the queue state because its accepted terminal-state list omitted `timed_out`. That attempt and the original helper are retained. Separate `audit_bound_matrix_v2.py` verifies that a timed-out queue has no started consumer and preserves the failed parent,while keeping every completed-consumer check. It accepted five cases and one unstarted group.

The four previously submitted successor services ended with dependency failure before creating a queue,assessment or consumer:FP16 Add inference,FP16 Attention inference,the ten-cell streaming group and the FP32 Attention profile. These are dependency failures,not model or profiler execution failures. Their receipts,plans,hashes and logs remain. A new independent CPU-only streaming group covers cells24,36,60,84,108 using unchanged budgets;it is not evidence of a completed test here.

Total audited first-process coverage is13/120 across two separate measurement series. No cell has three fresh processes within one series. The task-local aggregate is `TASK/audits/formal-series-review02`;its hash is retained in the JSON. Remaining FP32 cells,recommendation repetitions,FP16 companions and profiling remain open.

Re-audit the terminal group with:

```bash
python TASK/launchers/audit_bound_matrix_v2.py --name formal-bound-next01 --output NEW_AUDIT_JSON
```
