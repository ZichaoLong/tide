# Ordered window reduction and reusable projection adjoints

Implementation **233bf0144db206176f4af4b05b0bb54cb326dbe7** passed
[eight clean qualification jobs](resident-window-reduction-20261003.json).
Audit: `TASK/launchers/window_reduction_evidence.py <full implementation SHA>`.
All qualification jobs terminated exit0 and released their device leases.

Aggressive sharded training now reduces each reverse window's parameter aliases
on device before the preceding window executes. It then reuses physical projection
weight/bias/connection gradients, canonical numerical outputs and packet arenas.
Each owner's addition order remains reverse-window then registry-alias, without
pre-summing a window's aliases. Coordinator start packets gate peer zeroing;
final device consensus confirms all reduction reads before storage can be reused.
State/cache/message bridge gradients remain independent. Conservative and legacy
single-device backward retain their existing paths. Public ABI, CPU/core and
CANN kernels are unchanged; admission estimates remain conservative and unchanged.

Validation passed without skips:

- Eight standalone FP32/FP16 checks: **176 trajectories, 2,944 windows,
  688 updates**, compared with independent CPU FP32/FP64 references. New
  compact-projection gates cover aliases, streaming/prefill, event/fiber caches
  and SGD/AdamW. Existing gates cover None/zero roots, full state/master/optimizer
  observations, accumulation, checkpoint repartition and legacy continuation.
- **16** Python retention/budget checks and **32** actual-consumer comparisons:
  standalone LibTorch and Python clients, three families, Add/Attention,
  FP32/FP16 and sliced continued complete training.
- Source/header/options-verified object reuse, fresh standalone/Python links and
  installed public consumer; byte-verified unchanged core and device archives.

A same-lease two-card D512/B8/T4/V257 Attention calibration used physical B2 ×4,
two connected windows and one complete FP32 AdamW update. Loss remained exactly
**7.532631874084473**; all existing semantic/work/retention statistics and effective
chunks agree. The new counters report eight streamed windows across the four
sample groups and 588,350,000 bytes of duplicate projection-gradient storage avoided
per backward group. These planning counters are separate from allocator peaks.

| Logical device | Previous allocator peak (bytes) | New peak (bytes) | Reduction (bytes) |
| --- | ---: | ---: | ---: |
| 0 | 8,367,995,392 | 8,073,745,920 | 294,249,472 |
| 1 | 7,520,681,472 | 7,226,584,576 | 294,096,896 |

This is approximately **281/280 MiB less per card**, measured over the complete
consumer. It is not a throughput recommendation or an original-size training
result. Attention parameter adjoints still remain per-window in this revision;
full-size capacity/cost gates and the broader performance matrix remain open.

A separate FP16 complete-training trace recorded **53,176 operators**:
50,628 AI_VECTOR_CORE, 723 MIX_AIV and 1,825 AI_CORE, with **zero observed AiCPU**.
Instrumented timings are excluded from throughput conclusions. Development and
clean gates passed without a new failure; raw snapshots, manifests, CPU comparisons,
allocator observations and profiler CSVs are retained in task artifacts.
