# Shared immutable Full snapshots

Implementation **f360489ac965abe9bc326055050ace01b348ebd5** passed
[eight clean qualification jobs](resident-full-snapshots-20261002.json).
Audit: `TASK/launchers/full_snapshot_evidence.py <full implementation SHA>`.
All jobs terminated successfully; device leases released.

Guarded single-device and sharded training owners now retain one independent
copy of the Full parameter banks and static kind/mapping tables per backward
group. This covers identity/tanh/LH/SwiGLU, FP32/FP16 and repeated TensorImpl
aliases. Actual event metadata, values and counts remain per-window. Identity,
version, pointer, shape, stride, dtype and device checks guard reuse; the owner
forbids parameter publication while retained windows exist. Backward, detach and
close clear the cache, and subsequent updates capture the new banks. Default
standalone retention overloads still copy each tape independently.

`retained_full_bytes` reports the shared copy. Pre-advance capacity includes it
once and reserves each window's dynamic bound. No gradient summation, scheduling,
logical batch, numerical tolerance or optimizer boundary changes. Complete-consumer
estimates remain unchanged and conservative.

Validation passed without skips:

- Six standalone FP32/FP16 cells: **144 trajectories, 2,432 windows, 560 updates**,
  with independent CPU FP32/FP64 references. Dense/compact journals, aliases,
  None/zero roots, parameter/master/optimizer state, accumulation, continued
  execution and sharded-to-single-owner checkpoint restore are covered. Direct
  snapshot checks reject changed banks and confirm independent copy lifetimes.
- **16** Python retention tests check complete CPU autograd/state/KV/update/restore,
  exact retained budgets, refusal before progress, detach and empty windows.
- **32** actual-consumer comparisons cover native Python/standalone LibTorch,
  three families, Add/Attention, FP32/FP16 and sliced continued training.

A same-lease two-card D512/B8/T4/V257 Attention comparison used physical B2 ×4,
two connected windows and one complete FP32 AdamW update. Retained bytes fell by
**542,752**. Loss **7.532631874084473**, event/output/cut statistics, effective
chunks and the unchanged admission plans agree with the previous backend.

| Logical device | Previous allocator peak (bytes) | Shared Full peak (bytes) | Reduction (bytes) |
| --- | ---: | ---: | ---: |
| 0 | 8,368,268,800 | 8,367,995,392 | 273,408 |
| 1 | 7,520,954,880 | 7,520,681,472 | 273,408 |

This is a small real allocation reduction, **not** a throughput comparison or
full-size training result. Both wide fixtures use LH Full with small parameter
banks, so this change does not address their dominant memory costs.

A separate two-card FP16 complete-training trace recorded **53,182 operators**
(50,634 AI_VECTOR_CORE, 723 MIX_AIV, 1,825 AI_CORE), **zero observed AiCPU**.
It is an instrumented coverage check, not formal throughput evidence.

All users of the private training-owner layouts and affected retention objects
were rebuilt during development. Clean builds reuse source/header/options-verified
objects and perform fresh links; core/CANN dependencies are byte-verified and
unchanged. Public ABI is unchanged. The first development standalone build failed
on an ambiguous Tensor assignment in the new checker; its failed record is retained.
The corrected build and all clean jobs passed. Raw snapshots, manifests, tests,
calibration results and profiler CSVs remain in task artifacts.
